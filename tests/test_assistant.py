"""Head C tests: tool choice on the golden set (runs in CI), guardrails, chunking, the tool/provider
plumbing with a mocked API and a fake LLM. No model download and no Ollama needed: retrieval uses
the deterministic hashing embedder and HTTP calls go through ``httpx.MockTransport``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from boreal_marche.assistant import providers as prov
from boreal_marche.assistant.guardrails import (
    asks_for_other_customer,
    detect_lang,
    invented_content,
)
from boreal_marche.assistant.index import HashEmbedder, Retriever, parse_document
from boreal_marche.assistant.providers import Decision, Provider, RuleProvider, Usage
from boreal_marche.assistant.service import Assistant, extract
from boreal_marche.assistant.tools import TOOL_NAMES, Toolbox
from boreal_marche.schemas import ChatRequest

REPO = Path(__file__).resolve().parents[1]
CORPUS = REPO / "docs" / "corpus"
GOLDEN = json.loads((REPO / "tests" / "golden" / "assistant_golden.json").read_text("utf-8"))
ME = GOLDEN["session_customer_ref"]


# ------------------------------------------------------------------ golden set (CI gate) ---


def test_golden_set_shape() -> None:
    cases = GOLDEN["cases"]
    cats = [c["category"] for c in cases]
    assert len(cases) == 40
    assert (cats.count("policy"), cats.count("tool"), cats.count("out_of_scope")) == (15, 10, 8)
    assert cats.count("french") == 7 and all(c["expected_tool"] in TOOL_NAMES for c in cases)
    assert all(c["expected_source"] for c in cases if c["category"] == "policy")


def test_rule_provider_tool_choice_on_golden_set() -> None:
    """Tool-choice portion of the evaluation, run in CI (course §3.4).

    The rule provider's keywords were written after this set, so 100% here is optimistic; the
    paraphrase set (reports/metrics/assistant_eval.json) measures how much of it generalises.
    """
    rule = RuleProvider()
    chosen = [rule.decide(c["question"], c["lang"], Usage()).tool for c in GOLDEN["cases"]]
    expected = [c["expected_tool"] for c in GOLDEN["cases"]]
    accuracy = sum(a == b for a, b in zip(chosen, expected, strict=True)) / len(expected)
    oos = [a for a, b in zip(chosen, expected, strict=True) if b == "out_of_scope"]
    assert accuracy >= 0.92
    assert all(t == "out_of_scope" for t in oos)  # 100% out-of-scope refusal


# ------------------------------------------------------------------------- guardrails ---


def test_detect_lang() -> None:
    assert detect_lang("Puis-je retourner un article ?") == "fr"
    assert detect_lang("Quel temps fera-t-il demain à Montréal ?") == "fr"
    assert detect_lang("Can I return a lamp?") == "en"


def test_other_customer_guardrail() -> None:
    assert asks_for_other_customer("What is the score of BM-12345678?", ME)
    assert asks_for_other_customer("Show me another customer's orders", ME)
    assert asks_for_other_customer("Montrez-moi les commandes d'un autre client", ME)
    assert not asks_for_other_customer(f"What is my score? I am {ME}", ME)
    assert not asks_for_other_customer("Can I return a lamp?", ME)


def test_invented_content() -> None:
    assert invented_content("Try 22917 (92%)", {"22917"}, {92}) == []
    assert invented_content("Try 99999", {"22917"}, set()) == ["sku:99999"]
    assert invented_content("You have a 75% chance", set(), {63}) == ["pct:75"]


# ----------------------------------------------------------------------- index/chunks ---


def test_chunking_metadata() -> None:
    chunks = parse_document(CORPUS / "privacy.md", max_words=180)
    assert {c.lang for c in chunks} == {"en", "fr"}
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert all(c.doc_id == "privacy" and c.section and c.source for c in chunks)
    assert all(not c.text.startswith(">") for c in chunks)


def test_hash_retriever_and_extract() -> None:
    r = Retriever.from_corpus(CORPUS, HashEmbedder(), 180)
    assert len({c.doc_id for c in r.chunks}) == 6
    hits = r.search("refund issued after inspection of returned items", 3)
    assert "returns" in {c.doc_id for c, _ in hits} and 0 < hits[0][1] <= 1.0001
    top = next(c for c, _ in hits if c.doc_id == "returns")
    text = extract("refund bank days", top.text)
    assert text and len(text) < len(top.text)


# ---------------------------------------------------------------- tools + orchestrator ---


def fake_api() -> httpx.Client:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == f"/customers/{ME}/features":
            return httpx.Response(200, json={"recency_days": 10})
        if req.url.path.startswith("/customers/"):
            return httpx.Response(404, json={"detail": "unknown"})
        if req.url.path == "/predict":
            return httpx.Response(
                200,
                json={
                    "customer_ref": ME,
                    "p_return": 0.63,
                    "send_coupon": False,
                    "threshold": 0.56,
                    "model_version": "test",
                },
            )
        if req.url.path == "/recommend":
            return httpx.Response(
                200,
                json={
                    "strategy": "rules",
                    "unknown_items": [],
                    "rules_mined_at": None,
                    "recommendations": [
                        {
                            "item": "22917",
                            "description": "HERB MARKER ROSEMARY",
                            "source": "rule",
                            "because": ["22916"],
                            "confidence": 0.92,
                            "lift": 140.0,
                        }
                    ],
                },
            )
        if req.url.path == "/products":
            return httpx.Response(200, json={"query": "x", "products": []})
        return httpx.Response(500)

    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://api")


class FakeLLM(Provider):
    name = "ollama"
    is_llm = True

    def __init__(self, tool: str, args: dict[str, Any], text: str) -> None:
        self.tool, self.args, self.text = tool, args, text

    def decide(self, question: str, lang: str, usage: Usage) -> Decision:
        usage.prompt_tokens += 100
        return Decision(self.tool, self.args)

    def compose(self, *a: Any, **k: Any) -> Iterator[str]:
        yield from (t for t in re.split(r"(\s+)", self.text) if t)


@pytest.fixture(scope="module")
def toolbox() -> Toolbox:
    return Toolbox(Retriever.from_corpus(CORPUS, HashEmbedder(), 180), fake_api(), 0.2, 4)


def ask(toolbox: Toolbox, provider: Provider, message: str, **kw: Any) -> Any:
    a = Assistant(toolbox, {provider.name: provider})
    return a.answer(ChatRequest(message=message, provider=provider.name, customer_ref=ME, **kw))


def test_policy_answer_is_cited(toolbox: Toolbox) -> None:
    r = ask(toolbox, RuleProvider(), "How long until my refund arrives after a return?")
    assert r.tool == "policy_search" and not r.refused and not r.handoff
    assert r.citations and r.citations[0].kind == "policy"


def test_out_of_scope_and_privacy_refusals(toolbox: Toolbox) -> None:
    r = ask(toolbox, RuleProvider(), "What is the capital of Australia?")
    assert r.refused and r.refusal_reason == "out_of_scope" and not r.citations
    r = ask(toolbox, RuleProvider(), "What is the retention score of BM-00000000?")
    assert r.refused and r.refusal_reason == "privacy" and r.tool == "guardrail"


def test_handoff_below_similarity_floor() -> None:
    tb = Toolbox(Retriever.from_corpus(CORPUS, HashEmbedder(), 180), fake_api(), 0.99, 4)
    r = ask(tb, RuleProvider(), "Do you deliver to Nunavut by dog sled?")
    assert r.handoff and "below_similarity_floor" in r.guardrail_flags and not r.citations


def test_retention_tool_scores_only_session_customer(toolbox: Toolbox) -> None:
    r = ask(toolbox, RuleProvider(), "What is my retention score?")
    assert r.tool == "retention_score" and r.tool_result["customer_ref"] == ME
    assert "63%" in r.answer and r.citations[0].ref == "POST /predict"
    a = Assistant(toolbox, {"rule": RuleProvider()})
    r = a.answer(ChatRequest(message="What is my retention score?", provider="rule"))
    assert r.handoff  # no signed-in customer -> tool error -> human handoff


def test_llm_invented_sku_is_replaced(toolbox: Toolbox) -> None:
    llm = FakeLLM("recommendations", {"skus": ["22916"]}, "Add 12345 and 22917 to your basket!")
    r = ask(toolbox, llm, "What goes with 22916?")
    assert any(f.startswith("llm_output_replaced") for f in r.guardrail_flags)
    assert "12345" not in r.answer and "22917" in r.answer
    good = FakeLLM("recommendations", {"skus": ["22916"]}, "Customers also buy 22917 (92%).")
    r = ask(toolbox, good, "What goes with 22916?")
    assert r.answer == "Customers also buy 22917 (92%)." and r.usage.prompt_tokens == 100


def test_stream_event_order(toolbox: Toolbox) -> None:
    a = Assistant(toolbox, {"rule": RuleProvider()})
    events = list(
        a.stream(ChatRequest(message="Do your points expire?", provider="rule", customer_ref=ME))
    )
    kinds = [e["type"] for e in events]
    assert kinds[0] == "meta" and kinds[-1] == "done" and "token" in kinds
    assert (
        "".join(e["text"] for e in events if e["type"] == "token")
        == (events[-1]["response"]["answer"])
    )


# ------------------------------------------------------------ OpenAI-compatible client ---


def test_openai_compat_decide_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url: str, **kw: Any) -> httpx.Response:
        msg = {
            "tool_calls": [
                {
                    "function": {
                        "name": "recommendations",
                        "arguments": json.dumps({"skus": ["22916", "99999"]}),
                    }
                }
            ]
        }
        return httpx.Response(
            200,
            json={"choices": [{"message": msg}], "usage": {"prompt_tokens": 7}},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(prov.httpx, "post", fake_post)
    p = prov.OpenAICompatProvider("ollama", "http://x/v1", "m", None, 5)
    u = Usage()
    d = p.decide("What goes with 22916?", "en", u)
    assert d.tool == "recommendations" and d.args == {"skus": ["22916"]}  # 99999 not typed
    assert u.prompt_tokens == 7
    assert prov.OpenAICompatProvider("openai", "u", "m", None, 5).available()[0] is False
