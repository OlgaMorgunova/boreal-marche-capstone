"""The assistant's four typed tools. Each returns a validated Pydantic model.

The retention and recommendation tools call this platform's own API (``/customers``, ``/predict``,
``/recommend``) over HTTP — the language model never produces a probability or a product identifier
from its own weights; it can only relay what these tools return. ``retention_score`` takes *no*
customer argument: it always scores the signed-in customer of the chat session, so a model cannot
be talked into scoring somebody else.
"""

from __future__ import annotations

from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from boreal_marche.assistant.index import Retriever

ToolName = Literal[
    "policy_search", "retention_score", "recommendations", "product_lookup", "out_of_scope"
]
TOOL_NAMES: list[str] = list(ToolName.__args__)  # type: ignore[attr-defined]


class PolicyHit(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    section: str
    lang: str
    score: float
    text: str


class PolicySearchResult(BaseModel):
    query: str
    hits: list[PolicyHit]
    best_score: float
    similarity_floor: float
    above_floor: bool


class RetentionScore(BaseModel):
    customer_ref: str
    p_return: float = Field(ge=0, le=1)
    send_coupon: bool
    threshold: float
    model_version: str


class RecommendationItem(BaseModel):
    sku: str
    description: str
    source: Literal["rule", "popularity"]
    because: list[str] | None
    confidence: float | None
    lift: float | None


class Recommendations(BaseModel):
    basket: list[str]
    strategy: str
    items: list[RecommendationItem]
    unknown_items: list[str]


class Product(BaseModel):
    sku: str
    description: str
    category: str
    median_price: float
    n_orders: int


class ProductLookupResult(BaseModel):
    query: str
    products: list[Product]


class ToolError(RuntimeError):
    """A tool could not complete (API unavailable, unknown customer): the assistant hands off."""


# OpenAI-style function specifications (also understood by Ollama's OpenAI-compatible endpoint).
TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "policy_search",
            "description": "Search Boréal Marché's policies (delivery, shipping fees, returns, "
            "refunds, allergens, food safety, recalls, loyalty points, privacy, ordering, payment, "
            "account). Use for any question about how the store works.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "The question"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retention_score",
            "description": "Score the signed-in customer: probability of ordering again within 90 "
            "days and whether they are proposed for the quarterly discount offer. Use when the "
            "customer asks about their own likelihood to order, their score or the offer.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommendations",
            "description": "Recommend products to add to a basket, from association rules. Pass "
            "the product codes (SKUs) mentioned by the customer; an empty list returns popular "
            "products.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skus": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["skus"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "product_lookup",
            "description": "Look up products in the catalogue by product code (SKU) or by name: "
            "description, category and typical price. Use for 'do you sell', 'how much is', "
            "'what is product X'.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "SKU or product name"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "out_of_scope",
            "description": "Call this when the question is NOT about Boréal Marché shopping, "
            "orders, products or policies (weather, general knowledge, homework, finance, "
            "creative writing, other businesses).",
            "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}},
        },
    },
]


class Toolbox:
    def __init__(
        self,
        retriever: Retriever,
        api: httpx.Client,
        similarity_floor: float,
        top_k: int,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.retriever = retriever
        self.api = api
        self.similarity_floor = similarity_floor
        self.top_k = top_k
        self.headers = {"X-Internal-Caller": "assistant"} | (headers or {})

    def _call(self, method: str, url: str, **kwargs: Any) -> Any:
        try:
            r = self.api.request(method, url, headers=self.headers, **kwargs)
        except httpx.HTTPError as exc:
            raise ToolError(f"{url}: {type(exc).__name__}") from exc
        if r.status_code != 200:
            raise ToolError(f"{url}: HTTP {r.status_code}")
        return r.json()

    def policy_search(self, query: str) -> PolicySearchResult:
        hits = [
            PolicyHit(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                title=c.title,
                section=c.section,
                lang=c.lang,
                score=round(s, 4),
                text=c.text,
            )
            for c, s in self.retriever.search(query, self.top_k)
        ]
        best = hits[0].score if hits else 0.0
        return PolicySearchResult(
            query=query,
            hits=hits,
            best_score=best,
            similarity_floor=self.similarity_floor,
            above_floor=best >= self.similarity_floor,
        )

    def retention_score(self, customer_ref: str | None) -> RetentionScore:
        if not customer_ref:
            raise ToolError("no signed-in customer")
        features = self._call("GET", f"/customers/{customer_ref}/features")
        pred = self._call(
            "POST", "/predict", json={"customer_ref": customer_ref, "features": features}
        )
        return RetentionScore.model_validate(pred)

    def recommendations(self, skus: list[str], k: int = 5) -> Recommendations:
        body = self._call("POST", "/recommend", json={"basket": skus, "k": k})
        return Recommendations(
            basket=skus,
            strategy=body["strategy"],
            unknown_items=body["unknown_items"],
            items=[
                RecommendationItem(
                    sku=r["item"],
                    description=r["description"],
                    source=r["source"],
                    because=r["because"],
                    confidence=r["confidence"],
                    lift=r["lift"],
                )
                for r in body["recommendations"]
            ],
        )

    def product_lookup(self, query: str, limit: int = 5) -> ProductLookupResult:
        body = self._call("GET", "/products", params={"q": query, "limit": limit})
        return ProductLookupResult(
            query=query, products=[Product.model_validate(p) for p in body["products"]]
        )
