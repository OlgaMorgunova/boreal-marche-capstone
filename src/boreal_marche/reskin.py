"""Documented re-skin of UCI Online Retail II as the fictional Québec retailer Boréal Marché.

The underlying data are a public UK giftware wholesaler's transactions. For the pedagogical scenario
we relabel *geography* and add a *keyword category*; we do not alter quantities, prices, dates or
identifiers. Monetary values are labelled CAD but are numerically the source GBP amounts (no FX
conversion). The mapping below is reproduced in DATA_CARD.md.
"""

from __future__ import annotations

import pandas as pd

# Source country -> Boréal Marché delivery region. The home market (UK, ~90% of lines) becomes
# Québec; continental Europe becomes Ontario; islands/coastal markets become the Maritimes;
# everything else is international export.
REGION_MAP: dict[str, str] = {
    "United Kingdom": "QC",
    **{
        c: "ON"
        for c in [
            "Germany",
            "France",
            "Netherlands",
            "Belgium",
            "Switzerland",
            "Austria",
            "Denmark",
            "Sweden",
            "Finland",
            "Norway",
            "Italy",
            "Spain",
            "Portugal",
            "Poland",
            "Greece",
            "Czech Republic",
            "Lithuania",
            "European Community",
        ]
    },
    **{c: "MARITIMES" for c in ["EIRE", "Channel Islands", "Iceland", "Malta", "Cyprus"]},
}
DEFAULT_REGION = "INTL"
REGIONS = ["QC", "ON", "MARITIMES", "INTL"]

# First matching keyword wins; order matters (e.g. "CHRISTMAS CANDLE" -> seasonal).
CATEGORY_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("seasonal", ("CHRISTMAS", "XMAS", "EASTER", "HALLOWEEN", "ADVENT", "SANTA", "SNOWFLAKE")),
    ("lighting", ("CANDLE", "T-LIGHT", "TLIGHT", "LANTERN", "LIGHT", "LAMP")),
    ("bags_wrap", ("BAG", "WRAP", "TISSUE", "RIBBON", "GIFT BOX")),
    (
        "kitchen",
        (
            "MUG",
            "CUP",
            "PLATE",
            "BOWL",
            "TEA",
            "CAKE",
            "JAR",
            "BAKING",
            "LUNCH",
            "SPOON",
            "NAPKIN",
            "TRAY",
            "BOTTLE",
            "KITCHEN",
            "JUG",
            "GLASS",
            "TIN",
        ),
    ),
    (
        "stationery_party",
        (
            "CARD",
            "PAPER",
            "NOTEBOOK",
            "PEN ",
            "PENCIL",
            "STICKER",
            "PARTY",
            "BUNTING",
            "BALLOON",
            "CHALK",
            "BIRTHDAY",
        ),
    ),
    ("kids", ("TOY", "DOLL", "GAME", "CHILDREN", "KIDS", "SPACEBOY", "JIGSAW", "PLAYHOUSE")),
    (
        "home_decor",
        (
            "HEART",
            "SIGN",
            "FRAME",
            "CLOCK",
            "CUSHION",
            "DOORMAT",
            "HOOK",
            "MIRROR",
            "VASE",
            "DECORATION",
            "ORNAMENT",
            "HANGING",
            "WALL",
        ),
    ),
    ("garden", ("GARDEN", "PLANT", "FLOWER", "POT", "BIRD", "WATERING")),
    ("accessories", ("BRACELET", "NECKLACE", "EARRING", "PURSE", "HAIR", "UMBRELLA", "KEY")),
]
CATEGORIES = [name for name, _ in CATEGORY_KEYWORDS] + ["other"]


def region_of(country: str | None) -> str:
    if country is None or pd.isna(country):
        return DEFAULT_REGION
    return REGION_MAP.get(str(country), DEFAULT_REGION)


def category_of(description: str | None) -> str:
    if description is None or pd.isna(description):
        return "other"
    d = str(description).upper()
    for name, words in CATEGORY_KEYWORDS:
        if any(w in d for w in words):
            return name
    return "other"


def add_reskin_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    countries = pd.Series(out["country"].unique())
    out["region"] = out["country"].map(dict(zip(countries, countries.map(region_of), strict=True)))
    out["region"] = out["region"].fillna(DEFAULT_REGION).astype("string")
    if "description" in out:
        descs = pd.Series(out["description"].dropna().unique())
        cat = dict(zip(descs, descs.map(category_of), strict=True))
        out["category"] = out["description"].map(cat).fillna("other").astype("string")
    return out
