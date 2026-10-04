"""Offline demo of the Toolformer pipeline.

Shows, with no API key and no network:

1. The training-data loop: sample candidate API calls -> execute them ->
   filter down to the calls that measurably reduce the model's loss
   (paper §3), including one candidate that gets *rejected*.
2. Inference with tool use: the decoder interrupts generation at API calls,
   executes the tool, and splices the response back in (paper §4).

Run:  python examples/run_demo.py
"""

import sys

sys.path.insert(0, ".")

from toolformer import (
    MockLM,
    ToolformerDecoder,
    annotate_text,
    build_default_registry,
    build_training_data,
    filter_candidates,
    sample_candidates,
)


def demo_training_loop() -> None:
    print("=" * 70)
    print("1) SAMPLE -> EXECUTE -> FILTER  (paper section 3)")
    print("=" * 70)
    registry = build_default_registry()
    lm = MockLM()

    texts = [
        "The ratio was 400/1400 = 0.2857142857 exactly.",
        "The capital of Ohio is Columbus, a growing Midwestern city.",
        "The meeting is at 3pm. Everyone should arrive ten minutes early.",
    ]
    for text in texts:
        print(f"\nTEXT: {text}")
        candidates = sample_candidates(text, registry)
        # Add one deliberately useless candidate to show filtering at work.
        from toolformer.annotator import CandidateCall

        candidates.append(CandidateCall(position=0, tool_name="Calculator", tool_input="9*9"))
        kept = filter_candidates(text, candidates, registry, lm, tau=0.1)
        kept_ids = {id(c) for c in kept}
        for cand in candidates:
            verdict = "KEEP" if id(cand) in kept_ids else "drop"
            print(
                f"  [{verdict:>4}] {cand.call_text:45s}"
                + (f" -> {cand.response}   (loss -{cand.loss_delta:.2f} nats)" if cand.response else "")
            )
        print(f"  annotated: {annotate_text(text, kept)}")


def demo_inference() -> None:
    print("\n" + "=" * 70)
    print("2) INFERENCE WITH TOOL USE  (paper section 4)")
    print("=" * 70)
    registry = build_default_registry()
    decoder = ToolformerDecoder(MockLM(), registry)

    for prompt in [
        "What is 847 * 923?",
        "What is the capital of Ohio?",
        "What day of the week was 2015-01-01?",
    ]:
        print(f"\nPROMPT: {prompt}")
        print(f"OUTPUT: {decoder.complete(prompt)}")


def demo_dataset_export() -> None:
    print("\n" + "=" * 70)
    print("3) TRAINING-DATA EXPORT  (paper section 3.4)")
    print("=" * 70)
    registry = build_default_registry()
    examples = build_training_data(
        ["The ratio was 400/1400 = 0.2857142857 exactly."],
        registry,
        MockLM(),
    )
    for ex in examples:
        print(f"kept {ex['n_kept']}/{ex['n_candidates']} candidates")
        print(f"annotated: {ex['annotated']}")
    print("\n(Fine-tuning itself is standard causal-LM training on these")
    print(" annotated sequences — no vocabulary change needed, per the paper.)")


if __name__ == "__main__":
    demo_training_loop()
    demo_inference()
    demo_dataset_export()
