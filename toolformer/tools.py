"""Tool implementations — the five tool families from the Toolformer paper.

Paper §3 uses a calculator, a Q&A system, two search engines, a translation
system and a calendar. Each :class:`Tool` below exposes the same contract the
paper's API calls use:

* ``name`` — the identifier the model writes inside ``[Name("input")]``.
* ``description`` — one-line summary (shown to the model in few-shot prompts).
* ``example`` — a usage demonstration, mirroring the handful of examples the
  paper provides per API in its sampling prompt.
* ``run(input)`` — executes the call and returns the response as text.

API call format (paper §3.1, adapted): a call is written inline as
``[ToolName("input")]``; after execution the response is spliced in as
``[ToolName("input")] -> response``. The paper encodes calls with ordinary
tokens (``[``, ``]``, ``->``) so no vocabulary change is needed.
"""

from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass
from datetime import date, timedelta


@dataclass
class Tool:
    """A single callable API available to the model."""

    name: str
    description: str
    example: str

    def run(self, api_input: str) -> str:
        """Execute the tool on ``api_input`` and return its response as text."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval(expr: str) -> float:
    """Evaluate a numeric expression safely (no names, no calls)."""
    node = ast.parse(expr, mode="eval")

    def _eval(n):
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _SAFE_OPS:
            return _SAFE_OPS[type(n.op)](_eval(n.left), _eval(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _SAFE_OPS:
            return _SAFE_OPS[type(n.op)](_eval(n.operand))
        raise ValueError(f"unsupported expression: {expr!r}")

    return _eval(node)


class CalculatorTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            name="Calculator",
            description="Evaluates arithmetic expressions.",
            example='[Calculator("400 / 1400 * 100")]',
        )

    def run(self, api_input: str) -> str:
        try:
            value = _safe_eval(api_input.strip())
        except Exception as exc:  # noqa: BLE001 — surfaced as a tool error
            return f"ERROR: {exc}"
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(round(value, 10) if isinstance(value, float) else value)


# ---------------------------------------------------------------------------
# Q&A (mock knowledge base)
# ---------------------------------------------------------------------------

_KB = {
    "capital of ohio": "Columbus",
    "capital of france": "Paris",
    "capital of japan": "Tokyo",
    "largest planet": "Jupiter",
    "chemical symbol for gold": "Au",
    "first man on the moon": "Neil Armstrong",
    "author of 1984": "George Orwell",
    "population of toronto": "2,794,356",
}


class QATool(Tool):
    def __init__(self) -> None:
        super().__init__(
            name="QA",
            description="Answers factual questions from a knowledge base.",
            example='[QA("What is the capital of Ohio?")]',
        )

    def run(self, api_input: str) -> str:
        query = api_input.strip().lower().rstrip("?")
        for key, answer in _KB.items():
            if key in query or query in key:
                return answer
        return "I don't know."


# ---------------------------------------------------------------------------
# Search (mock document index)
# ---------------------------------------------------------------------------

_CORPUS = [
    ("GPT-4", "GPT-4 is a large multimodal model released by OpenAI in 2023."),
    ("Toolformer", "Toolformer (Schick et al., 2023) teaches language models to use tools via self-supervised learning."),
    ("Python", "Python is a high-level programming language created by Guido van Rossum."),
    ("Eiffel Tower", "The Eiffel Tower is a wrought-iron lattice tower in Paris, completed in 1889."),
]


class SearchTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            name="Search",
            description="Returns a short snippet from a document index.",
            example='[Search("Toolformer paper")]',
        )

    def run(self, api_input: str) -> str:
        query = api_input.strip().lower()
        best, best_score = None, 0
        for title, text in _CORPUS:
            score = sum(1 for w in query.split() if w in text.lower() or w in title.lower())
            if score > best_score:
                best, best_score = text, score
        return best if best else "No results found."


# ---------------------------------------------------------------------------
# Translation (mock phrasebook)
# ---------------------------------------------------------------------------

_PHRASEBOOK = {
    ("en", "es"): {
        "hello": "hola",
        "good morning": "buenos días",
        "thank you": "gracias",
        "good night": "buenas noches",
    },
    ("es", "en"): {
        "hola": "hello",
        "buenos días": "good morning",
        "gracias": "thank you",
        "buenas noches": "good night",
    },
}

_TRANSLATE_RE = re.compile(r"^\s*(?P<src>[a-z]{2})\s*->\s*(?P<tgt>[a-z]{2})\s*:\s*(?P<text>.+)$")


class TranslatorTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            name="Translator",
            description="Translates short phrases. Input format: 'en->es: hello'.",
            example='[Translator("en->es: hello")]',
        )

    def run(self, api_input: str) -> str:
        match = _TRANSLATE_RE.match(api_input)
        if not match:
            return "ERROR: expected format 'src->tgt: text', e.g. 'en->es: hello'"
        src, tgt, text = match.group("src"), match.group("tgt"), match.group("text").strip().lower()
        table = _PHRASEBOOK.get((src, tgt))
        if table is None:
            return f"ERROR: unsupported language pair {src}->{tgt}"
        return table.get(text, f"[untranslated: {text}]")


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------

_DATE_RE = re.compile(r"(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})")
_SHIFT_RE = re.compile(
    r"(?P<n>\d+)\s+days?\s+(?P<dir>after|before)\s+(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})",
    re.IGNORECASE,
)


class CalendarTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            name="Calendar",
            description="Answers date questions: weekday of a date, or date arithmetic.",
            example='[Calendar("What day of the week was 2015-01-01?")]',
        )

    def run(self, api_input: str) -> str:
        text = api_input.strip()
        shift = _SHIFT_RE.search(text)
        if shift:
            base = date(int(shift.group("y")), int(shift.group("m")), int(shift.group("d")))
            delta = timedelta(days=int(shift.group("n")))
            result = base + delta if shift.group("dir").lower() == "after" else base - delta
            return f"{result.isoformat()} ({result.strftime('%A')})"
        found = _DATE_RE.search(text)
        if found and ("day" in text.lower() or "weekday" in text.lower()):
            day = date(int(found.group("y")), int(found.group("m")), int(found.group("d")))
            return day.strftime("%A")
        if found:
            day = date(int(found.group("y")), int(found.group("m")), int(found.group("d")))
            return f"{day.isoformat()} is a {day.strftime('%A')}"
        return "ERROR: expected a date like 2015-01-01"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """Named collection of tools; dispatches ``[Name("input")]`` calls."""

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name!r}")
        return self._tools[name]

    def call(self, name: str, api_input: str) -> str:
        """Execute ``[name("api_input")]`` and return the response text."""
        return self.get(name).run(api_input)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def few_shot_block(self) -> str:
        """The handful of demonstrations the paper prompts with (one per tool)."""
        lines = ["You can call these tools:"]
        for tool in (self._tools[n] for n in self.names()):
            lines.append(f"- {tool.name}: {tool.description} e.g. {tool.example}")
        return "\n".join(lines)


def build_default_registry() -> ToolRegistry:
    """The five tool families used in the Toolformer paper."""
    return ToolRegistry([CalculatorTool(), QATool(), SearchTool(), TranslatorTool(), CalendarTool()])
