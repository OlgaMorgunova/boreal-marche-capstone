"""Three interchangeable providers behind one interface (``decide`` a tool, ``compose`` an answer).

* ``rule`` — keyword routing + extractive templates. Deterministic, free, instant. It exists to show
  where a language model adds value (paraphrase, French answers from English sources, fluent
  composition) and where it does not (tool routing on a narrow domain is mostly lexical).
* ``ollama`` — a local model (default ``llama3.1:8b``) through Ollama's OpenAI-compatible endpoint.
  Personal data never leaves our infrastructure: the answer to the data-residency question.
* ``openai`` — hosted model through the same client code. Disabled unless ``OPENAI_API_KEY`` is set;
  no real personal data may be sent to it (course §8.5), so the retention tool's output is the only
  customer-derived value it would ever see.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from boreal_marche.assistant.guardrails import SKU_RE, fold
from boreal_marche.assistant.tools import TOOL_NAMES, TOOL_SPECS


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, raw: dict[str, Any] | None) -> None:
        if raw:
            self.prompt_tokens += int(raw.get("prompt_tokens") or 0)
            self.completion_tokens += int(raw.get("completion_tokens") or 0)


@dataclass
class Decision:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)


GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"


class ProviderError(RuntimeError):
    pass


class Provider:
    name = "base"
    is_llm = False

    def available(self) -> tuple[bool, str]:
        return True, "ok"

    def decide(self, question: str, lang: str, usage: Usage) -> Decision:
        raise NotImplementedError

    def compose(
        self, question: str, lang: str, tool: str, result_json: str, template: str, usage: Usage
    ) -> Iterator[str]:
        raise NotImplementedError


# --------------------------------------------------------------------------------------- rule ---

_RETENTION = [
    r"\bmy (retention )?score\b", r"\bretention\b", r"\blikely to (order|buy|shop)\b",
    r"\b(will|do|am) i\b.*\b(offer|coupon|discount|selected|qualify|order again)\b",
    r"\border again\b", r"\bmon score\b", r"\bprobabilite\b",
    r"\b(vais-je|suis-je|est-ce que je)\b.*\b(offre|rabais|coupon|commander)\b",
    r"\bcommander (a nouveau|de nouveau)\b",
]  # fmt: skip
_RECOMMEND = [
    r"\brecommend", r"\bsuggest", r"\bgoes? (well )?with\b", r"\bgo with\b", r"\bwhat else\b",
    r"\balso like\b", r"\badd to\b", r"\bpopular products\b", r"\bpair with\b",
    r"\brecommand", r"\bsugger", r"\bva(?:s)? bien avec\b", r"\bquoi d'autre\b",
]  # fmt: skip
_PRODUCT = [
    r"\bhow much (does|is|are) (the|a|an)\b", r"\bprice of\b", r"\bdo you (sell|have|carry)\b",
    r"\bwhat is (product|item|sku)\b", r"\bproduct \d", r"\bcost\b", r"\bin stock\b",
    r"\bcombien (coute|coutent)\b", r"\bprix (du|de la|des)\b", r"\bvendez-vous\b",
    r"\bavez-vous\b", r"\bc'est quoi (le )?produit\b",
]  # fmt: skip
# Word-prefix patterns (\b...): "fee" must not fire on "coffee", nor "van" on "vanilla".
_POLICY = [
    r"\bdeliver", r"\bshipping\b", r"\bship\b", r"\bfees?\b", r"\bwindows?\b", r"\bvans?\b",
    r"\bnot home\b", r"\btracking\b", r"\bholiday", r"\breturn", r"\brefund", r"\bexchange",
    r"\bdamaged\b", r"\bdefect", r"\bspoil", r"\bwarm\b", r"\bquality\b", r"\bfinal sale\b",
    r"\ballerg", r"\bnuts?\b", r"\bpeanut", r"\bgluten", r"\bcontains?\b", r"\brecall",
    r"\blabels?\b", r"\bfood safety\b", r"\bloyalty\b", r"\bpoints?\b", r"\brewards?\b",
    r"\btiers?\b", r"\bbirthday\b", r"\bprivacy\b", r"\bpersonal information\b", r"\bmy data\b",
    r"\b(delete|close) my account\b", r"\bcancel", r"\bpayment", r"\bpay\b", r"\bgift cards?\b",
    r"\btax", r"\bsubstitut", r"\bwholesale\b", r"\bbusiness account", r"\bcustomer service\b",
    r"\bopening hours\b", r"\border", r"\baccount\b", r"\bkeep my\b",
    r"\blivraison", r"\blivrer\b", r"\bfrais\b", r"\bretour", r"\brembours", r"\bechange",
    r"\bendommag", r"\ballergene", r"\barachide", r"\bnoix\b", r"\bcontenir\b", r"\brappel",
    r"\betiquette", r"\bfidelite\b", r"\bpaliers?\b", r"\brecompenses\b", r"\bconfidentialite\b",
    r"\brenseignements\b", r"\bdonnees\b", r"\bmes droits\b", r"\bannuler\b", r"\bcommande",
    r"\bpaiement", r"\bcarte-cadeau", r"\btaxes?\b", r"\badresse", r"\bcompte\b",
    r"\bservice a la clientele\b",
]  # fmt: skip
_STOP = set(  # noqa: SIM905 - a word list reads better as one string
    [
        "how",
        "much",
        "does",
        "do",
        "is",
        "are",
        "the",
        "a",
        "an",
        "of",
        "for",
        "what",
        "which",
        "you",
        "sell",
        "have",
        "carry",
        "product",
        "item",
        "sku",
        "price",
        "cost",
        "costs",
        "in",
        "stock",
        "please",
        "can",
        "i",
        "me",
        "my",
        "your",
        "we",
        "our",
        "to",
        "it",
        "this",
        "that",
        "there",
        "any",
        "some",
        "combien",
        "coute",
        "coutent",
        "le",
        "la",
        "les",
        "des",
        "du",
        "de",
        "un",
        "une",
        "vendez-vous",
        "avez-vous",
        "prix",
        "produit",
        "quel",
        "quelle",
        "est",
    ]
)


def _matches(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text) for p in patterns)


def product_query(question: str) -> str:
    """Content words of a product question ('How much is the regency cakestand?' -> 'regency
    cakestand'); a SKU wins when present."""
    skus = SKU_RE.findall(question)
    if skus:
        return skus[0].upper()
    words = [w for w in re.findall(r"[a-z0-9'-]+", fold(question)) if w not in _STOP]
    return " ".join(words)


class RuleProvider(Provider):
    name = "rule"

    def decide(self, question: str, lang: str, usage: Usage) -> Decision:
        t = fold(question)
        skus = [s.upper() for s in SKU_RE.findall(question)]
        if _matches(_RETENTION, t):
            return Decision("retention_score")
        if _matches(_RECOMMEND, t):
            return Decision("recommendations", {"skus": skus})
        if _matches(_POLICY, t):
            return Decision("policy_search", {"query": question})
        if skus or _matches(_PRODUCT, t):
            return Decision("product_lookup", {"query": product_query(question)})
        return Decision("out_of_scope", {"reason": "no domain keyword"})

    def compose(
        self, question: str, lang: str, tool: str, result_json: str, template: str, usage: Usage
    ) -> Iterator[str]:
        for word in re.split(r"(\s+)", template):
            if word:
                yield word


# ------------------------------------------------------------------------- OpenAI-compatible ---

DECIDE_SYSTEM = (
    "You are the customer-service assistant of Boréal Marché, a Québec online grocery and "
    "specialty-goods retailer. For every user message call exactly ONE tool and never answer "
    "directly. policy_search: questions about delivery, fees, returns, refunds, allergens, "
    "recalls, loyalty points, privacy, ordering, payment or accounts. retention_score: the user"
    " asks about their OWN likelihood to order again, their score, or whether they will get the"
    " quarterly offer. recommendations: what to add to a basket or what goes with products "
    "(pass the product codes). product_lookup: whether a product is sold, its price, or what a "
    "product code is. out_of_scope: anything not about shopping at Boréal Marché. Users may "
    "write in French."
)
COMPOSE_SYSTEM = (
    "You are Boréal Marché's customer-service assistant. Answer the customer's question using ONLY "
    "the tool output provided. Do not add facts, numbers, percentages, prices or product codes "
    "that are not in the tool output. If the tool output does not answer the question, say so and "
    "offer customer service. Be concise (at most 5 sentences or a short list). Answer in {lang}."
)
LANG_NAME = {"en": "English", "fr": "French (français canadien)"}


class OpenAICompatProvider(Provider):
    """Chat-completions client with tool calling, shared by Ollama (``/v1``) and OpenAI."""

    is_llm = True

    def __init__(
        self,
        name: str,
        base_url: str,
        model: str,
        api_key: str | None,
        timeout: float,
        health_url: str | None = None,
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.health_url = health_url
        self._avail: tuple[float, tuple[bool, str]] | None = None

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def available(self) -> tuple[bool, str]:
        now = time.monotonic()
        if self._avail and now - self._avail[0] < 30:
            return self._avail[1]
        status: tuple[bool, str]
        if self.name == "openai" and not self.api_key:
            status = (False, "OPENAI_API_KEY not set")
        elif self.name == "gemini" and not self.api_key:
            status = (False, "GEMINI_API_KEY not set")
        elif self.health_url:
            try:
                tags = httpx.get(self.health_url, timeout=2).json()
                names = {m.get("name") for m in tags.get("models", [])}
                status = (
                    (True, "ok")
                    if self.model in names
                    else (False, f"model {self.model} not pulled")
                )
            except (httpx.HTTPError, ValueError) as exc:
                status = (False, f"unreachable ({type(exc).__name__})")
        else:
            status = (True, "ok")
        self._avail = (now, status)
        return status

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            r = httpx.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
            )
            r.raise_for_status()
            return r.json()  # type: ignore[no-any-return]
        except (httpx.HTTPError, ValueError) as exc:
            raise ProviderError(f"{self.name}: {type(exc).__name__}: {exc}") from exc

    def decide(self, question: str, lang: str, usage: Usage) -> Decision:
        body = self._post(
            {
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": DECIDE_SYSTEM},
                    {"role": "user", "content": question},
                ],
                "tools": TOOL_SPECS,
            }
        )
        usage.add(body.get("usage"))
        calls = body["choices"][0]["message"].get("tool_calls") or []
        if not calls:
            return Decision("out_of_scope", {"reason": "no tool call"}, ["no_tool_call"])
        fn = calls[0]["function"]
        name = fn.get("name", "")
        try:
            args = fn.get("arguments") or {}
            args = json.loads(args) if isinstance(args, str) else dict(args)
        except (json.JSONDecodeError, TypeError, ValueError):
            args = {}
        if name not in TOOL_NAMES:
            return Decision("out_of_scope", {"reason": f"unknown tool {name}"}, ["unknown_tool"])
        if name in ("policy_search", "product_lookup"):
            args = {"query": str(args.get("query") or question)}
        elif name == "recommendations":
            raw = args.get("skus") or []
            skus = [str(s).upper() for s in (raw if isinstance(raw, list) else [raw])]
            # Only codes the customer actually typed: the model must not invent product codes.
            typed = {s.upper() for s in SKU_RE.findall(question)}
            args = {"skus": [s for s in skus if s in typed] or sorted(typed)}
        return Decision(name, args)

    def compose(
        self, question: str, lang: str, tool: str, result_json: str, template: str, usage: Usage
    ) -> Iterator[str]:
        payload = {
            "model": self.model,
            "temperature": 0.2,
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [
                {"role": "system", "content": COMPOSE_SYSTEM.format(lang=LANG_NAME[lang])},
                {
                    "role": "user",
                    "content": f"Customer question: {question}\n\nTool `{tool}` output (JSON):\n"
                    f"{result_json}",
                },
            ],
        }
        try:
            with httpx.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    usage.add(chunk.get("usage"))
                    for choice in chunk.get("choices") or []:
                        token = (choice.get("delta") or {}).get("content")
                        if token:
                            yield token
        except (httpx.HTTPError, ValueError) as exc:
            raise ProviderError(f"{self.name}: {type(exc).__name__}: {exc}") from exc


def build_providers(
    ollama_model: str,
    openai_model: str,
    timeout: float,
    gemini_model: str = "gemini-flash-lite-latest",
) -> dict[str, Provider]:
    ollama = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    return {
        "rule": RuleProvider(),
        "ollama": OpenAICompatProvider(
            "ollama",
            f"{ollama}/v1",
            os.environ.get("OLLAMA_MODEL", ollama_model),
            None,
            timeout,
            health_url=f"{ollama}/api/tags",
        ),
        "openai": OpenAICompatProvider(
            "openai",
            os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            openai_model,
            os.environ.get("OPENAI_API_KEY") or None,
            timeout,
        ),
        "gemini": OpenAICompatProvider(
            "gemini",
            os.environ.get("GEMINI_BASE_URL", GEMINI_BASE_URL),
            os.environ.get("GEMINI_MODEL", gemini_model),
            os.environ.get("GEMINI_API_KEY") or None,
            timeout,
        ),
    }
