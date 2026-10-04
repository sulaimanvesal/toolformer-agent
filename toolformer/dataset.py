"""Stage 4 — build the self-supervised training set (paper §3.4).

The paper fine-tunes the LM on the filtered, annotated sequences with ordinary
causal language-model training. Real fine-tuning needs a GPU and a base model,
so this module does the reproducible part: it turns raw texts into the
annotated training examples the paper trains on, and writes them to JSONL.
Each example is plain text with ``[Tool("input")] -> response`` spliced in —
no vocabulary change required, exactly as in the paper.
"""

from __future__ import annotations

import json

from toolformer.annotator import annotate_text, filter_candidates, sample_candidates
from toolformer.model import LanguageModel
from toolformer.tools import ToolRegistry


def build_training_data(
    texts: list[str],
    registry: ToolRegistry,
    lm: LanguageModel,
    tau: float = 0.1,
) -> list[dict]:
    """Run sample → execute → filter over ``texts``; return annotated examples."""
    examples = []
    for text in texts:
        candidates = sample_candidates(text, registry)
        kept = filter_candidates(text, candidates, registry, lm, tau=tau)
        examples.append(
            {
                "source": text,
                "annotated": annotate_text(text, kept),
                "n_candidates": len(candidates),
                "n_kept": len(kept),
                "calls": [
                    {
                        "position": c.position,
                        "tool": c.tool_name,
                        "input": c.tool_input,
                        "response": c.response,
                        "loss_delta": round(c.loss_delta, 4),
                    }
                    for c in kept
                ],
            }
        )
    return examples


def write_jsonl(examples: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


if __name__ == "__main__":  # python -m toolformer.dataset
    from toolformer.model import MockLM
    from toolformer.tools import build_default_registry

    samples = [
        "The ratio was 400/1400 = 0.2857142857 exactly.",
        "The capital of Ohio is Columbus, a growing Midwestern city.",
        "I had coffee this morning and then went for a walk.",
    ]
    registry = build_default_registry()
    examples = build_training_data(samples, registry, MockLM())
    out = "toolformer_training_data.jsonl"
    write_jsonl(examples, out)
    kept_total = sum(e["n_kept"] for e in examples)
    print(f"wrote {len(examples)} examples, {kept_total} kept API calls -> {out}")
