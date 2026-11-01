"""UI strings, English and French. ``t(key, lang)`` falls back to English."""

from __future__ import annotations

STRINGS: dict[str, dict[str, str]] = {
    "title": {
        "en": "Boréal Marché — Customer Intelligence",
        "fr": "Boréal Marché — Intelligence client",
    },
    "tab_predict": {"en": "Prediction", "fr": "Prédiction"},
    "tab_basket": {"en": "Basket", "fr": "Panier"},
    "tab_assistant": {"en": "Assistant", "fr": "Assistant"},
    "tab_monitor": {"en": "Monitoring", "fr": "Surveillance"},
    "tab_about": {"en": "About", "fr": "À propos"},
    "language": {"en": "Language", "fr": "Langue"},
    "customer": {"en": "Signed-in customer", "fr": "Client connecté"},
    "api_down": {"en": "The API is not reachable", "fr": "L'API est injoignable"},
    "p_return": {
        "en": "Probability of ordering again within 90 days",
        "fr": "Probabilité de commander à nouveau d'ici 90 jours",
    },
    "proposed": {
        "en": "Proposed for the quarterly offer",
        "fr": "Proposé pour l'offre trimestrielle",
    },
    "not_proposed": {
        "en": "Not proposed for the quarterly offer",
        "fr": "Non proposé pour l'offre trimestrielle",
    },
    "human_review": {
        "en": "A proposal only: a marketing staff member reviews and approves every offer list (Law 25, s. 12.1).",
        "fr": "Une proposition seulement : un membre de l'équipe marketing examine et approuve chaque liste (Loi 25, art. 12.1).",
    },
    "action_holdout": {
        "en": "Hold-out this quarter: proposed, but in the 10% random sample that receives no "
        "coupon, so the coupon's real effect can be measured.",
        "fr": "Groupe témoin ce trimestre : proposé, mais dans l'échantillon aléatoire de 10 % qui "
        "ne reçoit pas de coupon, pour mesurer l'effet réel du coupon.",
    },
    "action_review": {
        "en": "High-value review list: not proposed automatically; a staff member decides "
        "(fairness mitigation for top spenders).",
        "fr": "Liste de révision à forte valeur : non proposé automatiquement; un membre du "
        "personnel décide (mesure d'équité pour les gros acheteurs).",
    },
    "campaign": {
        "en": "Campaign list for approval (demo customers)",
        "fr": "Liste de campagne à approuver (clients de démonstration)",
    },
    "campaign_help": {
        "en": "coupon = proposed · holdout = proposed, withheld to measure the save rate · "
        "review = top spenders the single threshold misses, for a human decision · none.",
        "fr": "coupon = proposé · holdout = proposé, retenu pour mesurer le taux de sauvegarde · "
        "review = gros acheteurs que le seuil unique manque, décision humaine · none.",
    },
    "what_if": {"en": "What if…", "fr": "Et si…"},
    "what_if_help": {
        "en": "Move a slider: the probability is recomputed by the API on every change.",
        "fr": "Déplacez un curseur : l'API recalcule la probabilité à chaque changement.",
    },
    "reset": {
        "en": "Reset to the customer's real values",
        "fr": "Revenir aux valeurs réelles du client",
    },
    "threshold": {"en": "decision threshold", "fr": "seuil de décision"},
    "factors": {
        "en": "Principal factors of this estimate",
        "fr": "Principaux facteurs de cette estimation",
    },
    "factors_help": {
        "en": "▲ raises, ▼ lowers the estimated probability. They describe how the model weighed "
        "your order history, not why you order; the time of year is part of the baseline.",
        "fr": "▲ augmente, ▼ diminue la probabilité estimée. Ils décrivent comment le modèle a "
        "pondéré votre historique de commandes, pas pourquoi vous commandez; la période de "
        "l'année fait partie de la base.",
    },
    "search": {
        "en": "Search the catalogue (name or SKU)",
        "fr": "Chercher dans le catalogue (nom ou code)",
    },
    "add": {"en": "Add", "fr": "Ajouter"},
    "remove": {"en": "Remove", "fr": "Retirer"},
    "your_basket": {"en": "Your basket", "fr": "Votre panier"},
    "empty_basket": {"en": "Your basket is empty.", "fr": "Votre panier est vide."},
    "suggestions": {"en": "You might also like", "fr": "Vous aimerez peut-être aussi"},
    "because": {"en": "Because you added", "fr": "Parce que vous avez ajouté"},
    "popular": {"en": "Popular with our customers", "fr": "Populaire auprès de nos clients"},
    "strategy": {"en": "Strategy that fired", "fr": "Stratégie utilisée"},
    "provider": {"en": "Provider", "fr": "Fournisseur"},
    "ask": {
        "en": "Ask about delivery, returns, allergens, loyalty, privacy, products…",
        "fr": "Posez une question : livraison, retours, allergènes, fidélité, confidentialité, produits…",
    },
    "sources": {"en": "Sources", "fr": "Sources"},
    "refresh": {"en": "Refresh", "fr": "Actualiser"},
    "overall": {"en": "Overall status", "fr": "État global"},
    "alerts": {"en": "Alert history", "fr": "Historique des alertes"},
    "features_psi": {
        "en": "Input drift by feature (PSI, higher = more drift)",
        "fr": "Dérive des entrées par variable (PSI)",
    },
    "no_traffic": {
        "en": "Not enough traffic yet. Run the traffic simulator (`make simulate`).",
        "fr": "Pas encore assez de trafic. Lancez le simulateur (`make simulate`).",
    },
    "privacy_notice": {"en": "Privacy notice", "fr": "Avis de confidentialité"},
    "model_card": {"en": "Model card", "fr": "Fiche du modèle"},
    "data_card": {"en": "Data provenance", "fr": "Provenance des données"},
    "architecture": {"en": "Architecture", "fr": "Architecture"},
    "not_found": {"en": "No product found.", "fr": "Aucun produit trouvé."},
    "handoff": {"en": "Passed to a human", "fr": "Transmis à un humain"},
    "replaced": {
        "en": "The model's wording was replaced by the verified template (guardrail).",
        "fr": "La formulation du modèle a été remplacée par le gabarit vérifié (garde-fou).",
    },
}


def t(key: str, lang: str) -> str:
    entry = STRINGS[key]
    return entry.get(lang) or entry["en"]
