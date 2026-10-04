"""Toolformer (Schick et al., 2023) — self-supervised tool use for language models.

This package implements the core pipeline from
"Toolformer: Language Models Can Teach Themselves to Use Tools"
(arXiv:2302.04761):

1. **Sample** — propose candidate API calls in plain text (few-shot prompted).
2. **Execute** — run every candidate against the tool implementations.
3. **Filter** — keep only calls whose response measurably reduces the model's
   loss on the following tokens (the paper's perplexity-based criterion).
4. **(Fine-tune)** — export the surviving annotations as a JSONL training set;
   the paper then fine-tunes the LM on it with ordinary causal-LM training
   (no vocabulary change required).
5. **Inference** — decode text, interrupt generation when the model emits an
   API call, execute the tool, and splice the result back in.

Runs fully offline with a deterministic mock LM; a real OpenAI-compatible
backend can be plugged in via environment variables (see ``model.py``).
"""

from toolformer.annotator import (
    CandidateCall,
    annotate_text,
    filter_candidates,
    sample_candidates,
)
from toolformer.dataset import build_training_data
from toolformer.inference import ToolformerDecoder
from toolformer.model import LanguageModel, MockLM, OpenAILM
from toolformer.tools import Tool, ToolRegistry, build_default_registry

__all__ = [
    "CandidateCall",
    "annotate_text",
    "LanguageModel",
    "MockLM",
    "OpenAILM",
    "Tool",
    "ToolRegistry",
    "ToolformerDecoder",
    "build_default_registry",
    "build_training_data",
    "filter_candidates",
    "sample_candidates",
]

__version__ = "0.1.0"
