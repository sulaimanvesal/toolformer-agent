"""Inference-time tool use (paper §4, "decoding with API calls").

The trained Toolformer decides *during generation* when to call a tool: the
decoder is interrupted whenever the model emits an API call, the tool is
executed, and its response is spliced into the sequence before generation
continues. :class:`ToolformerDecoder` implements that loop on top of any
:class:`LanguageModel`.

The model is allowed at most ``max_api_calls`` per completion (the paper caps
API calls per sequence as well). Unknown tools and tool errors are spliced in
as ``[ERROR: ...]`` markers rather than crashing the loop.
"""

from __future__ import annotations

import re

from toolformer.model import LanguageModel
from toolformer.tools import ToolRegistry

_CALL_RE = re.compile(r"\[(?P<name>\w+)\(\"(?P<input>(?:[^\"\\]|\\.)*)\"\)\]")
_ERROR_FMT = '[ERROR: {message}]'


class ToolformerDecoder:
    """Generate text with interrupt-and-execute tool calls (paper §4)."""

    def __init__(
        self,
        lm: LanguageModel,
        registry: ToolRegistry,
        max_api_calls: int = 5,
        max_rounds: int = 8,
    ) -> None:
        self.lm = lm
        self.registry = registry
        self.max_api_calls = max_api_calls
        self.max_rounds = max_rounds

    def complete(self, prompt: str) -> str:
        """Generate a completion of ``prompt``, executing tool calls inline."""
        sequence = prompt
        n_calls = 0
        for _ in range(self.max_rounds):
            chunk = self.lm.generate(sequence)
            sequence += " " + chunk.strip()
            match = _CALL_RE.search(chunk)
            if not match or n_calls >= self.max_api_calls:
                break
            name = match.group("name")
            api_input = match.group("input").replace('\\"', '"')
            try:
                response = self.registry.call(name, api_input)
            except KeyError:
                response = _ERROR_FMT.format(message=f"unknown tool {name!r}")
            except Exception as exc:  # noqa: BLE001 — tool failures become text
                response = _ERROR_FMT.format(message=str(exc))
            sequence += f" -> {response}"
            n_calls += 1
        return sequence

    def trace(self, prompt: str) -> list[dict]:
        """Like :meth:`complete`, but also return a step-by-step call log."""
        sequence = prompt
        steps: list[dict] = []
        n_calls = 0
        for _ in range(self.max_rounds):
            chunk = self.lm.generate(sequence)
            sequence += " " + chunk.strip()
            match = _CALL_RE.search(chunk)
            if not match or n_calls >= self.max_api_calls:
                steps.append({"type": "text", "text": chunk.strip()})
                break
            name, api_input = match.group("name"), match.group("input").replace('\\"', '"')
            try:
                response = self.registry.call(name, api_input)
            except KeyError:
                response = _ERROR_FMT.format(message=f"unknown tool {name!r}")
            except Exception as exc:  # noqa: BLE001
                response = _ERROR_FMT.format(message=str(exc))
            sequence += f" -> {response}"
            steps.append(
                {"type": "call", "tool": name, "input": api_input, "response": response}
            )
            n_calls += 1
        return steps
