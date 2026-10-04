"""Tests for the Toolformer pipeline. Run with: pytest -q"""

import pytest

from toolformer.annotator import (
    CandidateCall,
    annotate_text,
    filter_candidates,
    sample_candidates,
)
from toolformer.dataset import build_training_data
from toolformer.inference import ToolformerDecoder
from toolformer.model import MockLM
from toolformer.tools import build_default_registry


@pytest.fixture()
def registry():
    return build_default_registry()


@pytest.fixture()
def lm():
    return MockLM()


# ---------------------------------------------------------------- tools -----


def test_calculator(registry):
    assert registry.call("Calculator", "400 / 1400 * 100") == "28.5714285714"
    assert registry.call("Calculator", "847 * 923") == "781781"
    assert registry.call("Calculator", "2 + 2").startswith("4")
    assert registry.call("Calculator", "1/0").startswith("ERROR")


def test_calculator_rejects_code_injection(registry):
    assert registry.call("Calculator", "__import__('os').system('x')").startswith("ERROR")


def test_qa(registry):
    assert registry.call("QA", "What is the capital of Ohio?") == "Columbus"
    assert registry.call("QA", "capital of France") == "Paris"
    assert registry.call("QA", "who won the 2099 world cup?") == "I don't know."


def test_search(registry):
    assert "Toolformer" in registry.call("Search", "Toolformer paper")
    assert registry.call("Search", "zzzzqqq") == "No results found."


def test_translator(registry):
    assert registry.call("Translator", "en->es: hello") == "hola"
    assert registry.call("Translator", "es->en: gracias") == "thank you"
    assert registry.call("Translator", "hello").startswith("ERROR")


def test_calendar(registry):
    assert registry.call("Calendar", "What day of the week was 2015-01-01?") == "Thursday"
    assert "2015-01-11" in registry.call("Calendar", "10 days after 2015-01-01")
    assert registry.call("Calendar", "hello").startswith("ERROR")


def test_unknown_tool_raises(registry):
    with pytest.raises(KeyError):
        registry.call("Nope", "x")


# ------------------------------------------------- sample / filter ---------


def test_sample_finds_calculator_candidate(registry):
    cands = sample_candidates("The result of 400/1400*100 was shown.", registry)
    assert any(c.tool_name == "Calculator" for c in cands)


def test_sample_finds_qa_candidate(registry):
    cands = sample_candidates("The capital of Ohio is Columbus.", registry)
    assert any(c.tool_name == "QA" for c in cands)


def test_filter_keeps_useful_call(registry, lm):
    text = "The ratio was 400/1400 = 0.2857142857 exactly."
    cands = sample_candidates(text, registry)
    assert cands, "expected at least one candidate"
    kept = filter_candidates(text, cands, registry, lm, tau=0.1)
    assert len(kept) == 1
    assert kept[0].tool_name == "Calculator"
    assert kept[0].loss_delta >= 0.1


def test_filter_drops_useless_call(registry, lm):
    # The tool result ("2") appears nowhere in the continuation, so it cannot
    # help predict it: the candidate must be filtered out.
    text = "The meeting is at 3pm. Everyone should arrive early."
    cand = CandidateCall(position=0, tool_name="Calculator", tool_input="1+1")
    kept = filter_candidates(text, [cand], registry, lm, tau=0.1)
    assert kept == []


def test_filter_drops_failed_tool(registry, lm):
    text = "Some text here about nothing in particular at all."
    cand = CandidateCall(position=5, tool_name="Calculator", tool_input="1/0")
    kept = filter_candidates(text, [cand], registry, lm, tau=0.0)
    assert kept == []


def test_annotate_splices_calls(registry, lm):
    text = "i.e. 400/1400*100 = 28.57 percent."
    cands = sample_candidates(text, registry)
    kept = filter_candidates(text, cands, registry, lm, tau=0.1)
    annotated = annotate_text(text, kept)
    assert '[Calculator("400/1400*100")]' in annotated
    assert "->" in annotated


# ------------------------------------------------- inference ---------------


def test_decoder_executes_calculator(registry, lm):
    decoder = ToolformerDecoder(lm, registry)
    out = decoder.complete("What is 847 * 923?")
    assert "781781" in out
    assert '[Calculator("847 * 923")]' in out


def test_decoder_executes_qa(registry, lm):
    decoder = ToolformerDecoder(lm, registry)
    out = decoder.complete("What is the capital of Ohio?")
    assert "Columbus" in out


def test_decoder_trace_logs_calls(registry, lm):
    decoder = ToolformerDecoder(lm, registry)
    steps = decoder.trace("What is 6 * 7?")
    calls = [s for s in steps if s["type"] == "call"]
    assert calls and calls[0]["response"] == "42"


def test_decoder_unknown_tool_becomes_error_text(registry):
    class WeirdLM(MockLM):
        def generate(self, prompt, max_new_tokens=64):
            return '[Nope("x")]'

    decoder = ToolformerDecoder(WeirdLM(), registry, max_rounds=2)
    out = decoder.complete("hi")
    assert "unknown tool" in out


def test_decoder_respects_max_api_calls(registry):
    class ChattyLM(MockLM):
        def generate(self, prompt, max_new_tokens=64):
            return '[Calculator("1+1")]'

    decoder = ToolformerDecoder(ChattyLM(), registry, max_api_calls=2, max_rounds=10)
    out = decoder.complete("go")
    assert out.count("-> 2") == 2


# ------------------------------------------------- dataset -----------------


def test_build_training_data(registry, lm):
    texts = [
        "The ratio was 400/1400 = 0.2857142857 exactly.",
        "I had coffee this morning and then went for a walk.",
    ]
    examples = build_training_data(texts, registry, lm, tau=0.1)
    assert len(examples) == 2
    assert examples[0]["n_kept"] == 1
    assert examples[1]["n_kept"] == 0
    assert "annotated" in examples[0]
    assert examples[0]["calls"][0]["tool"] == "Calculator"
