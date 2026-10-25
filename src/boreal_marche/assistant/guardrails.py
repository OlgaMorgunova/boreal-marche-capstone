"""Provider-independent guardrails, applied around every provider.

1. **Other customers' data** — refused *before* any provider runs (a request naming a customer
   reference other than the signed-in one, or asking about "another customer").
2. **Out of scope** — the provider must choose the ``out_of_scope`` tool; any provider reply without
   a tool call is treated the same way (the assistant never answers from the model's own knowledge).
3. **No grounding** — a policy question whose best retrieval score is below the similarity floor is
   handed to a human instead of answered.
4. **Citations** — every grounded policy answer carries the chunks it was built from.
5. **No invented numbers or product codes** — an LLM answer that mentions a SKU or a percentage not
   present in the tool output is replaced by the deterministic template answer.
"""

from __future__ import annotations

import re
import unicodedata

SKU_RE = re.compile(r"\b\d{5}[A-Za-z]{0,2}\b")
REF_RE = re.compile(r"\bBM-[0-9A-F]{8}\b", re.IGNORECASE)
PCT_RE = re.compile(r"(\d{1,3}(?:[.,]\d+)?)\s?%")

_FR_WORDS = {
    "le", "la", "les", "des", "est", "que", "quel", "quels", "quelle", "vous", "vos", "votre",
    "mes", "mon", "ma", "comment", "combien", "avec", "pour", "une", "du", "au", "aux", "je",
    "puis", "est-ce", "qu'est-ce", "pourquoi", "sont", "fera-t-il", "recommandez", "un",
    "puis-je", "dois-je", "vais-je", "a-t-il", "y", "il", "ce", "ces", "nous", "quand", "quoi",
    "avez", "pouvez", "merci", "bonjour", "dans", "sur", "et", "ou", "ne", "pas",
}  # fmt: skip
_OTHER_CUSTOMER = [
    "another customer", "other customer", "someone else", "my neighbour", "my neighbor",
    "other people's", "all customers", "list of customers", "customer list",
    "autre client", "autres clients", "un client", "d'un client", "mon voisin", "liste des clients",
]  # fmt: skip

MESSAGES = {
    "out_of_scope": {
        "en": "I can only help with Boréal Marché: orders, products, delivery, returns, allergens, "
        "the Boréal Rewards programme and privacy. For anything else, please use another source.",
        "fr": "Je peux seulement vous aider pour Boréal Marché : commandes, produits, livraison, "
        "retours, allergènes, programme Boréal Récompenses et confidentialité. Pour toute autre "
        "question, veuillez consulter une autre source.",
    },
    "privacy": {
        "en": "I can't share or look up information about another customer. I can only discuss the "
        "account you are signed in with.",
        "fr": "Je ne peux pas consulter ni communiquer les renseignements d'un autre client. Je "
        "peux seulement parler du compte avec lequel vous êtes connecté.",
    },
    "handoff": {
        "en": "I couldn't find this in our published policies, so I'm passing your question to a "
        "member of our team. You can also reach customer service at support@borealmarche.example "
        "or 1-800-555-0142 (Mon–Fri 8:00–20:00, Sat 9:00–17:00 ET).",
        "fr": "Je n'ai pas trouvé la réponse dans nos politiques publiées; je transmets donc votre "
        "question à un membre de notre équipe. Vous pouvez aussi joindre le service à la clientèle "
        "à support@borealmarche.example ou au 1-800-555-0142 (lun.–ven. 8 h–20 h, sam. 9 h–17 h).",
    },
    "tool_error": {
        "en": "I can't reach that information right now, so I'm passing your question to a member "
        "of our team (support@borealmarche.example, 1-800-555-0142).",
        "fr": "Je ne peux pas obtenir cette information pour le moment; je transmets votre "
        "question à un membre de notre équipe (support@borealmarche.example, 1-800-555-0142).",
    },
}


def fold(text: str) -> str:
    """Lower-case and strip accents, so 'fidélité' matches 'fidelite'."""
    nfkd = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def detect_lang(text: str) -> str:
    words = re.findall(r"[\w'-]+", text.lower())
    score = sum(w in _FR_WORDS for w in words) + 2 * bool(re.search(r"[éèêàâçùûôî]", text))
    return "fr" if score >= 2 else "en"


def asks_for_other_customer(text: str, session_ref: str | None) -> bool:
    refs = {r.upper() for r in REF_RE.findall(text)}
    if refs and refs != {(session_ref or "").upper()}:
        return True
    t = fold(text)
    return any(fold(p) in t for p in _OTHER_CUSTOMER)


def invented_content(text: str, allowed_skus: set[str], allowed_pcts: set[int]) -> list[str]:
    """Product codes or percentages in an LLM answer that the tool output does not support."""
    problems = [f"sku:{s}" for s in SKU_RE.findall(text) if s.upper() not in allowed_skus]
    for m in PCT_RE.findall(text):
        v = round(float(m.replace(",", ".")))
        if not any(abs(v - a) <= 1 for a in allowed_pcts):
            problems.append(f"pct:{m}")
    return problems
