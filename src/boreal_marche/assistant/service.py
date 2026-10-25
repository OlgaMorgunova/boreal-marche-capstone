"""Assistant orchestration: guardrails -> provider chooses a tool -> tool runs -> answer streams.

``Assistant.stream`` yields events, which ``/chat/stream`` sends as Server-Sent Events:

* ``meta``    — provider, chosen tool, language, citations and the tool output (before any text);
* ``token``   — answer text as it is produced (LLM tokens, or template words for the rule provider);
* ``replace`` — the guardrail rejected the streamed LLM text; the UI shows this text instead;
* ``done``    — the complete validated ``ChatResponse``.

The deterministic template for each tool is always rendered first. It is the rule provider's answer
and the fallback whenever an LLM answer fails validation.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel

from boreal_marche.assistant.guardrails import (
    MESSAGES,
    PCT_RE,
    asks_for_other_customer,
    detect_lang,
    invented_content,
)
from boreal_marche.assistant.providers import Decision, Provider, ProviderError, Usage
from boreal_marche.assistant.tools import (
    PolicySearchResult,
    ProductLookupResult,
    Recommendations,
    RetentionScore,
    Toolbox,
    ToolError,
)
from boreal_marche.schemas import ChatRequest, ChatResponse, ChatUsage, Citation, ProviderStatus

Event = dict[str, Any]


class ProviderUnavailable(RuntimeError):
    pass


# ------------------------------------------------------------------------------- templates ---


def _units(text: str) -> list[str]:
    """Sentences of a chunk; markdown table rows become 'cell · cell · cell'."""
    out: list[str] = []
    for block in text.split("\n\n"):
        lines = block.strip().splitlines()
        if lines and lines[0].startswith("|"):
            rows = [ln for ln in lines if not re.match(r"^\|[\s|:-]+\|$", ln)]
            out += [" · ".join(c.strip() for c in r.strip("|").split("|")) for r in rows[1:]]
        else:
            out += re.split(r"(?<=[.!?])\s+", " ".join(ln.strip() for ln in lines))
    return [u for u in out if u]


def extract(question: str, text: str, n: int = 3) -> str:
    """Extractive answer: the ``n`` units of a chunk that share most words with the question."""
    q = set(re.findall(r"[a-z]{3,}", question.lower()))
    units = _units(text)
    scored = sorted(
        range(len(units)), key=lambda i: -len(q & set(re.findall(r"[a-z]{3,}", units[i].lower())))
    )
    best = sorted(scored[:n]) if q & set(re.findall(r"[a-z]{3,}", text.lower())) else [0, 1, 2]
    return " ".join(units[i] for i in best if i < len(units))


def render(tool: str, result: BaseModel, question: str, lang: str) -> str:
    fr = lang == "fr"
    if isinstance(result, PolicySearchResult):
        top = next(h for h in result.hits if h.score >= result.similarity_floor)
        lead = (
            f"Selon notre politique « {top.title} » ({top.section})"
            + (" — texte original en anglais" if top.lang != "fr" else "")
            + " : "
            if fr
            else f"According to our {top.title} ({top.section}): "
        )
        return lead + extract(question, top.text)
    if isinstance(result, RetentionScore):
        pct = round(100 * result.p_return)
        if fr:
            listed = "fait partie" if result.send_coupon else "ne fait pas partie"
            return (
                f"Notre modèle estime à {pct} % la probabilité que vous passiez une nouvelle "
                f"commande d'ici 90 jours. Selon cette estimation, votre compte {listed} de la "
                "liste proposée pour l'offre trimestrielle; un membre de notre équipe marketing "
                "examine chaque liste avant l'envoi. Vous pouvez demander les principaux facteurs "
                "utilisés ou une révision : privacy@borealmarche.example."
            )
        listed = "is on" if result.send_coupon else "is not on"
        return (
            f"Our model estimates a {pct}% probability that you will order again within 90 days. "
            f"Based on this estimate your account {listed} the proposed list for the quarterly "
            "offer; a member of our marketing team reviews every list before it is sent. You can "
            "ask for the main factors used or for a review: privacy@borealmarche.example."
        )
    if isinstance(result, Recommendations):
        if not result.items:
            return "Aucune suggestion pour le moment." if fr else "No suggestions right now."
        lines = []
        for it in result.items:
            if it.because:
                why = (
                    f"parce que vous avez ajouté {', '.join(it.because)}; confiance "
                    f"{round(100 * (it.confidence or 0))} %"
                    if fr
                    else f"because you added {', '.join(it.because)}; confidence "
                    f"{round(100 * (it.confidence or 0))}%"
                )
            else:
                why = "parmi nos plus populaires" if fr else "one of our most popular"
            lines.append(f"- {it.description.title()} ({it.sku}) — {why}")
        unknown = (
            (
                f"\n(Codes inconnus : {', '.join(result.unknown_items)})"
                if fr
                else f"\n(Unknown codes: {', '.join(result.unknown_items)})"
            )
            if result.unknown_items
            else ""
        )
        head = "Suggestions pour votre panier :" if fr else "Suggestions for your basket:"
        return head + "\n" + "\n".join(lines) + unknown
    if isinstance(result, ProductLookupResult):
        if not result.products:
            return (
                f"Je n'ai trouvé aucun produit pour « {result.query} »."
                if fr
                else f"I couldn't find a product matching '{result.query}'."
            )
        head = "Produits trouvés :" if fr else "Products found:"
        rows = [
            f"- {p.description.title()} ({p.sku}) — {p.category.replace('_', ' ')}, "
            + (f"prix habituel {p.median_price:.2f} $" if fr else f"usually ${p.median_price:.2f}")
            for p in result.products
        ]
        return head + "\n" + "\n".join(rows)
    raise TypeError(type(result))


def citations_for(tool: str, result: BaseModel) -> list[Citation]:
    if isinstance(result, PolicySearchResult):
        return [
            Citation(kind="policy", ref=h.chunk_id, title=h.title, section=h.section, score=h.score)
            for h in result.hits
            if h.score >= result.similarity_floor
        ]
    if isinstance(result, RetentionScore):
        return [Citation(kind="tool", ref="POST /predict", title=f"model {result.model_version}")]
    if isinstance(result, Recommendations):
        return [Citation(kind="tool", ref="POST /recommend", title=f"strategy {result.strategy}")]
    return [Citation(kind="tool", ref="GET /products", title="product catalogue")]


def allowed_values(result: BaseModel) -> tuple[set[str], set[int]]:
    """SKUs and percentages an LLM answer may mention: only those in the tool output."""
    raw = result.model_dump_json()
    skus = {s.upper() for s in re.findall(r"\b\d{5}[A-Za-z]{0,2}\b", raw)}
    pcts = {round(float(m.replace(",", "."))) for m in PCT_RE.findall(raw)}
    if isinstance(result, RetentionScore):
        pcts |= {round(100 * result.p_return), round(100 * result.threshold)}
    if isinstance(result, Recommendations):
        pcts |= {round(100 * (i.confidence or 0)) for i in result.items}
    return skus, pcts


def _llm_view(result: BaseModel) -> str:
    """What the composing LLM sees: the tool output, trimmed (policy chunks above the floor)."""
    if isinstance(result, PolicySearchResult):
        hits = [
            {"source": h.chunk_id, "section": h.section, "text": h.text}
            for h in result.hits
            if h.score >= result.similarity_floor
        ]
        return json.dumps({"policy_excerpts": hits}, ensure_ascii=False)
    if isinstance(result, RetentionScore):
        d = result.model_dump()
        d["p_return_percent"] = round(100 * result.p_return)
        d["note"] = "send_coupon is a proposal; a staff member approves every offer list"
        return json.dumps(d, ensure_ascii=False)
    return result.model_dump_json()


# ---------------------------------------------------------------------------- orchestrator ---


class Assistant:
    def __init__(
        self,
        toolbox: Toolbox,
        providers: dict[str, Provider],
        openai_price: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        self.toolbox = toolbox
        self.providers = providers
        self.openai_price = openai_price

    def status(self) -> list[ProviderStatus]:
        out = []
        for name, p in self.providers.items():
            ok, detail = p.available()
            out.append(
                ProviderStatus(
                    name=name,  # type: ignore[arg-type]
                    available=ok,
                    detail=detail,
                    model=getattr(p, "model", None),
                )
            )
        return out

    def _run_tool(self, d: Decision, customer_ref: str | None) -> BaseModel:
        tb = self.toolbox
        if d.tool == "policy_search":
            return tb.policy_search(str(d.args.get("query", "")))
        if d.tool == "retention_score":
            return tb.retention_score(customer_ref)
        if d.tool == "recommendations":
            return tb.recommendations(list(d.args.get("skus", [])))
        if d.tool == "product_lookup":
            return tb.product_lookup(str(d.args.get("query", "")))
        raise ValueError(d.tool)

    def _cost(self, provider: str, usage: Usage) -> float:
        if provider != "openai":
            return 0.0
        pin, pout = self.openai_price
        return (usage.prompt_tokens * pin + usage.completion_tokens * pout) / 1e6

    def stream(self, req: ChatRequest) -> Iterator[Event]:
        t0 = time.perf_counter()
        lang = req.lang or detect_lang(req.message)
        provider = self.providers[req.provider]
        ok, detail = provider.available()
        if not ok:
            raise ProviderUnavailable(f"{req.provider}: {detail}")
        usage, flags = Usage(), []
        tool, result = "none", None
        refusal: str | None = None
        handoff = False
        citations: list[Citation] = []
        answer = ""

        if asks_for_other_customer(req.message, req.customer_ref):
            refusal, tool = "privacy", "guardrail"
            answer = MESSAGES["privacy"][lang]
        else:
            try:
                decision = provider.decide(req.message, lang, usage)
                flags += decision.flags
                tool = decision.tool
                if tool == "out_of_scope":
                    refusal, answer = "out_of_scope", MESSAGES["out_of_scope"][lang]
                else:
                    result = self._run_tool(decision, req.customer_ref)
                    if isinstance(result, PolicySearchResult) and not result.above_floor:
                        handoff, answer = True, MESSAGES["handoff"][lang]
                        flags.append("below_similarity_floor")
            except ToolError as exc:
                handoff, answer = True, MESSAGES["tool_error"][lang]
                flags.append(f"tool_error: {exc}")
            except ProviderError as exc:
                handoff, answer = True, MESSAGES["tool_error"][lang]
                flags.append(f"provider_error: {exc}")

        grounded = result is not None and not handoff
        if grounded:
            assert result is not None
            citations = citations_for(tool, result)
            answer = render(tool, result, req.message, lang)
        yield {
            "type": "meta",
            "provider": req.provider,
            "tool": tool,
            "lang": lang,
            "citations": [c.model_dump() for c in citations],
            "tool_result": result.model_dump() if grounded and result is not None else None,
        }
        if grounded and provider.is_llm:
            assert result is not None
            text = ""
            try:
                for token in provider.compose(
                    req.message, lang, tool, _llm_view(result), answer, usage
                ):
                    text += token
                    yield {"type": "token", "text": token}
                skus, pcts = allowed_values(result)
                problems = invented_content(text, skus, pcts)
                if problems or not text.strip():
                    flags.append("llm_output_replaced: " + (", ".join(problems) or "empty"))
                    yield {"type": "replace", "text": answer}
                else:
                    answer = text
            except ProviderError as exc:
                flags.append(f"provider_error: {exc}")
                yield {"type": "replace", "text": answer}
        else:
            for word in re.split(r"(\s+)", answer):
                if word:
                    yield {"type": "token", "text": word}

        response = ChatResponse(
            answer=answer,
            lang=lang,  # type: ignore[arg-type]
            provider=req.provider,
            tool=tool,
            refused=refusal is not None,
            refusal_reason=refusal,  # type: ignore[arg-type]
            handoff=handoff,
            citations=citations,
            tool_result=result.model_dump() if grounded and result is not None else None,
            guardrail_flags=flags,
            latency_ms=round(1000 * (time.perf_counter() - t0), 1),
            usage=ChatUsage(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                est_cost_usd=round(self._cost(req.provider, usage), 6),
            ),
        )
        yield {"type": "done", "response": response.model_dump()}

    def answer(self, req: ChatRequest) -> ChatResponse:
        last: Event = {}
        for ev in self.stream(req):
            last = ev
        return ChatResponse.model_validate(last["response"])
