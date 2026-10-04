"""Stages 1–3 of the Toolformer training pipeline (paper §3).

1. **Sample** (:func:`sample_candidates`): propose candidate API calls at
   positions in a raw text where a tool might help. The paper does this with
   few-shot in-context prompting; here each tool ships a rule-based proposer
   that mirrors what such a prompt would surface, so the pipeline runs
   offline. The set of candidates is deliberately *over-generated* — most of
   them are useless, which is exactly what filtering is for.
2. **Execute**: every candidate is run against the :class:`ToolRegistry`
   (paper §3.2).
3. **Filter** (:func:`filter_candidates`): keep a candidate only if inserting
   its call *and its response* reduces the model's loss on the tokens that
   follow the call position by at least ``tau`` — and only if the response
   itself (not just the call marker) is what helps. The paper compares against
   two baselines: no call, and a call with an empty result ("blind" call).
   Candidates whose tool errored are always dropped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from toolformer.model import LanguageModel
from toolformer.tools import ToolRegistry

# Inline encoding of a call and of a call+response (paper §3.1: no new tokens).
CALL_FMT = '[{name}("{api_input}")]'


@dataclass
class CandidateCall:
    """One proposed API call inside a text."""

    position: int  # char offset where the call is inserted
    tool_name: str
    tool_input: str
    response: str = ""  # filled in after execution
    loss_without: float = field(default=0.0, repr=False)
    loss_with: float = field(default=0.0, repr=False)
    loss_delta: float = field(default=0.0, repr=False)  # without − with

    @property
    def call_text(self) -> str:
        escaped = self.tool_input.replace('"', '\\"')
        return CALL_FMT.format(name=self.tool_name, api_input=escaped)

    @property
    def annotated_text(self) -> str:
        if self.response:
            return f"{self.call_text} -> {self.response}"
        return self.call_text


# ---------------------------------------------------------------------------
# Stage 1 — sampling (rule-based proposers standing in for few-shot prompts)
# ---------------------------------------------------------------------------

_ARITH_RE = re.compile(r"(?P<expr>\d[\d\s.,]*[+\-*/]\s*[\d\s.,()^*]+)")
# Matches e.g. "capital of Ohio" (entity = 1 capitalized word); the call is
# inserted right *after* the question phrase so the continuation holds the
# answer — mirroring the paper's "The population of Toronto is [QA(...)]
# 2,794,356" layout.
_QA_RE = re.compile(r"(?P<q>(?:capital|population) of [A-Z][a-z]+)")
_DATE_Q_RE = re.compile(
    r"(?P<n>\d+)\s+days?\s+(?:after|before)\s+(?P<date>\d{4}-\d{2}-\d{2})|"
    r"what day (?:of the week )?(?:was|is) (?P<date2>\d{4}-\d{2}-\d{2})",
    re.IGNORECASE,
)


def _propose_calculator(text: str) -> list[CandidateCall]:
    out = []
    for m in _ARITH_RE.finditer(text):
        expr = m.group("expr").strip().rstrip(".,")
        if len(re.findall(r"\d", expr)) >= 2 and any(op in expr for op in "+-*/"):
            out.append(CandidateCall(position=m.end(), tool_name="Calculator", tool_input=expr))
    return out


def _propose_qa(text: str) -> list[CandidateCall]:
    out = []
    for m in _QA_RE.finditer(text):
        q = m.group("q").strip()
        out.append(CandidateCall(position=m.end(), tool_name="QA", tool_input=q))
    return out


def _propose_calendar(text: str) -> list[CandidateCall]:
    out = []
    for m in _DATE_Q_RE.finditer(text):
        out.append(CandidateCall(position=m.start(), tool_name="Calendar", tool_input=m.group(0).strip()))
    return out


_PROPOSERS = {
    "Calculator": _propose_calculator,
    "QA": _propose_qa,
    "Calendar": _propose_calendar,
}


def sample_candidates(text: str, registry: ToolRegistry) -> list[CandidateCall]:
    """Propose candidate API calls for ``text`` (paper §3.1, sampling)."""
    candidates: list[CandidateCall] = []
    for name in registry.names():
        proposer = _PROPOSERS.get(name)
        if proposer:
            candidates.extend(proposer(text))
    return sorted(candidates, key=lambda c: c.position)


# ---------------------------------------------------------------------------
# Stage 3 — filtering (paper §3.3)
# ---------------------------------------------------------------------------


def filter_candidates(
    text: str,
    candidates: list[CandidateCall],
    registry: ToolRegistry,
    lm: LanguageModel,
    tau: float = 0.1,
) -> list[CandidateCall]:
    """Execute every candidate and keep only the useful ones (paper §3.3).

    A candidate survives iff::

        loss(continuation | prefix) - loss(continuation | prefix + call + response) >= tau

    *and* the response beats a "blind" baseline (the call marker with an empty
    response), proving the *tool's answer* — not the call syntax — is what
    helps. Each candidate is judged independently on the original text.
    """
    kept: list[CandidateCall] = []
    for cand in candidates:
        try:
            response = registry.call(cand.tool_name, cand.tool_input)
        except Exception:  # noqa: BLE001 — failed calls are filtered out
            continue
        if response.startswith("ERROR") or response in ("I don't know.", "No results found."):
            continue
        cand.response = response

        prefix, continuation = text[: cand.position], text[cand.position :]
        if not continuation.strip():
            continue  # nothing after the call to be helped by it
        loss_without = lm.continuation_loss(prefix, continuation)
        loss_blind = lm.continuation_loss(prefix + cand.call_text + " -> ", continuation)
        loss_with = lm.continuation_loss(prefix + cand.annotated_text, continuation)

        cand.loss_without, cand.loss_with = loss_without, loss_with
        cand.loss_delta = loss_without - loss_with
        if cand.loss_delta >= tau and loss_with < loss_blind:
            kept.append(cand)
    return kept


def annotate_text(text: str, calls: list[CandidateCall]) -> str:
    """Splice kept calls (with responses) back into ``text``.

    Calls are applied right-to-left so earlier offsets stay valid.
    """
    out = text
    for cand in sorted(calls, key=lambda c: c.position, reverse=True):
        head, tail = out[: cand.position], out[cand.position :]
        sep = "" if not tail or tail[0] in " \t\n.,;:!?)]" else " "
        out = head + " " + cand.annotated_text + sep + tail
    return out
