"""Language-model backends behind the Toolformer pipeline.

The paper's filtering step needs, for any prefix/continuation pair, the model's
loss on the continuation tokens (to compare "with tool result" vs. "without").
:meth:`LanguageModel.continuation_loss` is that hook.

Two backends are provided:

* :class:`MockLM` — deterministic, offline, dependency-free. It implements a
  simple word-coverage model: words already present in the prefix are treated
  as likely, everything else as unlikely. This faithfully reproduces the
  *shape* of the paper's filtering criterion (a tool call survives only when
  its response makes the following tokens more predictable) without needing a
  real LM. Its :meth:`generate` uses the same proposal heuristics as the
  annotator so the offline demo exercises the full loop.
* :class:`OpenAILM` — optional backend for any OpenAI-compatible chat API
  (set ``TOOLFORMER_BASE_URL`` / ``TOOLFORMER_API_KEY``). Uses the real model
  for generation and estimates loss from token log-probs when available.
"""

from __future__ import annotations

import json
import math
import os
import re
import urllib.request
from typing import Protocol


class LanguageModel(Protocol):
    """Minimal interface the pipeline needs from a language model."""

    def generate(self, prompt: str, max_new_tokens: int = 64) -> str:
        """Generate a continuation of ``prompt``."""
        ...

    def continuation_loss(self, prefix: str, continuation: str) -> float:
        """Mean negative log-likelihood (nats) of ``continuation`` given ``prefix``.

        Lower = the model finds the continuation more predictable.
        """
        ...


_WORD_RE = re.compile(r"[A-Za-z0-9_.\-]+")
_CALL_MARKER_RE = re.compile(r"\[(\w+)\(\"(?:[^\"\\]|\\.)*\"\)\]")


def _words(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


class MockLM:
    """Deterministic offline stand-in for a real LM.

    Loss model: each continuation word gets probability ``p_hit`` if it is
    "covered" by the prefix (the word — or a super/substring of it — already
    appears there), else ``p_miss``. Covered words stand in for tokens the
    model can copy or predict confidently; uncovered words stand in for tokens
    it would have to guess. A tool response that contains the answer therefore
    lowers the loss of a continuation that states the answer — exactly the
    signal the paper filters on.
    """

    p_hit: float = 0.9
    p_miss: float = 0.05

    def continuation_loss(self, prefix: str, continuation: str) -> float:
        prefix_words = set(_words(prefix))
        total, n = 0.0, 0
        for word in _words(continuation):
            covered = any(word in pw or pw in word for pw in prefix_words)
            p = self.p_hit if covered else self.p_miss
            total += -math.log(p)
            n += 1
        return total / n if n else 0.0

    # -- generation: rule-based proposals so the offline demo exercises the
    #    full sample/execute/filter/inference loop -------------------------

    def generate(self, prompt: str, max_new_tokens: int = 64) -> str:
        tail = prompt[-400:]
        if "->" in tail or _CALL_MARKER_RE.search(tail):
            # A call (and its result) is already present: do not propose again.
            return self._complete(tail)
        proposal = self._propose(tail)
        if proposal:
            return proposal
        return self._complete(tail)

    def _propose(self, tail: str) -> str | None:
        # Arithmetic questions: "What is 847 * 923?" / "calculate 3+4"
        math_q = re.search(
            r"(?:what is|calculate|compute)\s+(?P<expr>[\d][\d\s.,+\-*/()^]*\d)",
            tail,
            re.IGNORECASE,
        )
        if math_q:
            expr = math_q.group("expr").strip().rstrip("?.!")
            return f'[Calculator("{expr}")]'
        # Calendar questions ("what day ...") — checked before QA so "What day
        # of the week was ...?" does not get misrouted to the Q&A tool.
        cal_q = re.search(
            r"(?P<q>(?:what day|which day)[^?.]{4,80}[?.]?)",
            tail,
            re.IGNORECASE,
        )
        if cal_q:
            return f'[Calendar("{cal_q.group("q").strip()}")]'
        # QA questions: "...capital of Ohio?" style
        qa_q = re.search(
            r"(?P<q>(?:what|which|who|where)\s+[^?]{4,80}\?)",
            tail,
            re.IGNORECASE,
        )
        if qa_q:
            return f'[QA("{qa_q.group("q").strip()}")]'
        return None

    def _complete(self, tail: str) -> str:
        # After a tool result was spliced in, state it plainly.
        result = re.search(r"->\s*(?P<r>[^\[\]\n]+?)\s*$", tail)
        if result:
            return f"The answer is {result.group('r').strip()}."
        return "I don't have enough information to answer."


class OpenAILM:
    """Backend for any OpenAI-compatible chat-completions endpoint.

    Reads ``TOOLFORMER_BASE_URL`` (default ``https://api.openai.com/v1``) and
    ``TOOLFORMER_API_KEY`` from the environment. Loss is estimated from token
    log-probs when the API returns them (``logprobs=True``); otherwise it falls
    back to a constant so the pipeline still runs.
    """

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model
        self.base_url = os.environ.get("TOOLFORMER_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.api_key = os.environ.get("TOOLFORMER_API_KEY", "")

    def _post(self, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.load(resp)

    def generate(self, prompt: str, max_new_tokens: int = 64) -> str:
        data = self._post(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_new_tokens,
                "temperature": 0.0,
            }
        )
        return data["choices"][0]["message"]["content"].strip()

    def continuation_loss(self, prefix: str, continuation: str) -> float:
        data = self._post(
            {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": prefix},
                    {"role": "assistant", "content": continuation},
                ],
                "max_tokens": 1,
                "temperature": 0.0,
                "logprobs": True,
            }
        )
        try:
            logprobs = data["choices"][0]["logprobs"]["content"]
            return -sum(t["logprob"] for t in logprobs) / max(len(logprobs), 1)
        except (KeyError, IndexError, TypeError):
            return 1.0  # backend gave no logprobs; neutral value
