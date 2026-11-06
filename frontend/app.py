"""Boréal Marché front end (Streamlit): prediction, basket, assistant, monitoring, about.

Talks only to the API (``API_URL``); every call carries ``X-Traffic-Cohort: ui`` so the monitoring
log can tell interface traffic from simulated traffic. Run: ``streamlit run frontend/app.py``.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import altair as alt
import httpx
import pandas as pd
import streamlit as st
from i18n import t

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
PUBLIC_API_URL = os.environ.get("PUBLIC_API_URL", API_URL).rstrip("/")
ROOT = Path(os.environ.get("BOREAL_ROOT") or Path(__file__).resolve().parents[1])
HEADERS = {"X-Traffic-Cohort": "ui"}
# Status palette: always shown with a symbol and a word, never colour alone.
STATUS = {
    "green": ("#0ca30c", "●", {"en": "OK", "fr": "OK"}),
    "amber": ("#fab219", "▲", {"en": "Warning", "fr": "Avertissement"}),
    "red": ("#d03b3b", "■", {"en": "Alert", "fr": "Alerte"}),
    "insufficient_data": ("#8a8a85", "○", {"en": "Not enough data", "fr": "Données insuffisantes"}),
}
BLUE = "#2a78d6"
WHAT_IF = [  # (feature, label en, label fr, min, max, step)
    ("recency_days", "Days since last order", "Jours depuis la dernière commande", 0.0, 180.0, 1.0),
    ("orders_90d", "Orders in last 90 days", "Commandes (90 derniers jours)", 0, 30, 1),
    ("orders_180d", "Orders in last 180 days", "Commandes (180 derniers jours)", 1, 60, 1),
    (
        "revenue_180d",
        "Spend in last 180 days ($)",
        "Dépenses (180 derniers jours, $)",
        1.0,
        20000.0,
        10.0,
    ),
    ("active_months_180d", "Months with an order (of the last 6)",
     "Mois avec commande (6 derniers)", 1, 7, 1),
]  # fmt: skip

st.set_page_config(page_title="Boréal Marché", page_icon="🍁", layout="wide")


# ------------------------------------------------------------------------------- API ---


@st.cache_resource
def client() -> httpx.Client:
    return httpx.Client(base_url=API_URL, headers=HEADERS, timeout=60)


def api(method: str, path: str, **kw: Any) -> Any:
    r = client().request(method, path, **kw)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=300)
def customers() -> list[dict[str, Any]]:
    return api("GET", "/customers", params={"limit": 300})["customers"]


@st.cache_data(ttl=300)
def customer_features(ref: str) -> dict[str, Any]:
    return api("GET", f"/customers/{ref}/features")


@st.cache_data(ttl=60)
def providers() -> list[dict[str, Any]]:
    return api("GET", "/chat/providers")


def status_badge(status: str, lang: str) -> str:
    color, icon, words = STATUS.get(status, STATUS["insufficient_data"])
    return f"<span style='color:{color};font-weight:600'>{icon}</span> {words[lang]}"


# ---------------------------------------------------------------------------- layout ---

with st.sidebar:
    lang = st.radio(
        "Language / Langue",
        ["en", "fr"],
        format_func=lambda x: {"en": "English", "fr": "Français"}[x],
        horizontal=True,
    )
    try:
        version = api("GET", "/version")
        ready = api("GET", "/ready")
    except httpx.HTTPError as exc:
        st.error(f"{t('api_down', lang)}: {API_URL} ({type(exc).__name__})")
        st.stop()
    refs = customers()
    labels = {
        c["ref"]: f"{c['ref']} · {c['region']} · {c['orders_180d']} orders/180 d" for c in refs
    }
    me = st.selectbox(t("customer", lang), list(labels), format_func=labels.__getitem__)
    st.caption(
        f"API {version['api_version']} · commit `{version['git_commit'][:7]}` · "
        f"model `{version['model_version']}` · {version['rule_count']} rules"
    )
    st.markdown(f"[API docs]({PUBLIC_API_URL}/docs)")
    if not ready["ready"]:
        st.warning("API not ready: " + "; ".join(ready["errors"]))

st.title(t("title", lang))
tabs = st.tabs(
    [t(k, lang) for k in ("tab_predict", "tab_basket", "tab_assistant", "tab_monitor", "tab_about")]
)


# ----------------------------------------------------------------------- 1. prediction ---

with tabs[0]:
    real = customer_features(me)
    if st.session_state.get("whatif_for") != me:  # new customer: sliders start at real values
        for f, *_ in WHAT_IF:
            st.session_state[f"wi_{f}"] = real[f]
        st.session_state["whatif_for"] = me
    left, right = st.columns([2, 3], gap="large")
    with left:
        st.subheader(t("what_if", lang))
        st.caption(t("what_if_help", lang))
        feats = dict(real)
        for f, en, fr, lo, hi, step in WHAT_IF:
            cur = st.session_state[f"wi_{f}"]
            hi_ = max(hi, cur) if isinstance(hi, float) else max(hi, int(cur))
            feats[f] = st.slider(fr if lang == "fr" else en, lo, hi_, step=step, key=f"wi_{f}")
        # Keep the vector internally consistent (the API validates ranges).
        feats["orders_180d"] = max(feats["orders_180d"], feats["orders_90d"], 1)
        feats["active_months_180d"] = min(feats["active_months_180d"], feats["orders_180d"])
        if st.button(t("reset", lang)):
            st.session_state.pop("whatif_for")
            st.rerun()
    with right:
        base = api("POST", "/predict", json={"customer_ref": me, "features": real})
        pred = api(
            "POST", "/predict", params={"explain": 3}, json={"customer_ref": me, "features": feats}
        )
        delta = pred["p_return"] - base["p_return"]
        st.metric(
            t("p_return", lang),
            f"{100 * pred['p_return']:.0f}%",
            f"{100 * delta:+.0f} pts" if abs(delta) > 0.004 else None,
        )
        st.progress(pred["p_return"])
        if pred["send_coupon"]:
            st.warning(
                f"{t('proposed', lang)} (p ≤ {pred['threshold']:.2f}, {t('threshold', lang)})"
            )
        else:
            st.success(
                f"{t('not_proposed', lang)} (p > {pred['threshold']:.2f}, {t('threshold', lang)})"
            )
        if pred.get("action") in ("holdout", "review"):
            st.info(t(f"action_{pred['action']}", lang))
        st.caption(t("human_review", lang) + f" · model `{pred['model_version']}`")
        if pred.get("top_factors"):
            st.markdown(f"**{t('factors', lang)}**")
            for fct in pred["top_factors"]:
                arrow = "▲" if fct["effect"] > 0 else "▼"
                val = "—" if fct["value"] is None else fct["value"]
                st.write(f"{arrow} {fct['label']} = `{val}`")
            st.caption(t("factors_help", lang))
        try:
            camp = api("GET", "/campaign")
        except httpx.HTTPStatusError:  # an API older than v1.0, or no campaign policy loaded
            camp = None
        if camp:
            with st.expander(t("campaign", lang)):
                cols = st.columns(4)
                for col, (k, v) in zip(cols, camp["counts"].items(), strict=True):
                    col.metric(k, v)
                st.caption(
                    t("campaign_help", lang) + f" · campaign `{camp['campaign_id']}`, "
                    f"hold-out {100 * camp['holdout_share']:.0f}%, review thresholds "
                    + ", ".join(f"{q} ≤ {v:.2f}" for q, v in camp["review_thresholds"].items())
                )
                if camp["review_list"]:
                    st.dataframe(
                        pd.DataFrame(camp["review_list"])[
                            ["ref", "region", "spend_quartile", "revenue_180d", "p_return"]
                        ],
                        width="stretch",
                        hide_index=True,
                    )
        with st.expander("Features sent to /predict"):
            used = {k: v for k, v in real.items() if v is not None or feats.get(k) is not None}
            st.dataframe(
                pd.DataFrame({"real": used, "what-if": {k: feats[k] for k in used}}).astype(str),
                width="stretch",
            )


# --------------------------------------------------------------------------- 2. basket ---


def _basket() -> list[str]:
    if "basket" not in st.session_state:  # persisted in the URL: survives a page refresh
        raw = st.query_params.get("basket", "")
        st.session_state["basket"] = [s for s in raw.split(",") if s]
    return st.session_state["basket"]  # type: ignore[no-any-return]


def _save_basket(items: list[str]) -> None:
    st.session_state["basket"] = items
    st.query_params["basket"] = ",".join(items)


@st.cache_data(ttl=600)
def product(sku: str) -> dict[str, Any] | None:
    hits = api("GET", "/products", params={"q": sku, "limit": 1})["products"]
    return hits[0] if hits and hits[0]["sku"] == sku else None


def describe(sku: str) -> str:
    p = product(sku)
    return p["description"].title() if p else sku


def category_label(cat: str) -> str:
    return cat.replace("_", " ").capitalize()


@st.cache_data(ttl=600)
def rules_summary() -> dict[str, Any]:
    return api("GET", "/recommend/rules")


@st.cache_data(ttl=600)
def rule_items(category: str) -> list[dict[str, Any]]:
    q = {"in_rules": True, "limit": 500, "category": category}
    return api("GET", "/products", params=q)["products"]


def _set_basket(items: list[str]) -> None:
    _save_basket(items)
    st.rerun()


with tabs[1]:
    basket = _basket()
    summary = rules_summary()
    st.caption(t("basket_intro", lang))
    c1, c2 = st.columns([1, 2], gap="large")

    with c1:
        st.subheader(f"🛍️ {t('build_basket', lang)}")
        cats = ["", *summary["categories"]]
        cat = st.selectbox(
            t("category", lang),
            cats,
            format_func=lambda c: category_label(c) if c else t("all_categories", lang),
        )
        names = {p["sku"]: p["description"].title() for p in rule_items(cat)}
        options = [*basket, *(s for s in names if s not in basket)]
        picked = st.multiselect(
            t("pick_items", lang),
            options,
            default=basket,
            format_func=lambda s: f"{names.get(s) or describe(s)} · {s}",
            help=t("pick_help", lang),
        )
        if picked != basket:
            _set_basket(picked)
        with st.expander(t("search_all", lang)):
            q = st.text_input(t("search", lang), placeholder="cake cases, 22423, lunch bag…")
            if q:
                found = api("GET", "/products", params={"q": q, "limit": 8})["products"]
                if not found:
                    st.caption(t("not_found", lang))
                for p in found:
                    a, b = st.columns([4, 1])
                    a.write(
                        f"**{p['description'].title()}** · `{p['sku']}` · ${p['median_price']:.2f}"
                    )
                    if b.button(t("add", lang), key=f"add_{p['sku']}", disabled=p["sku"] in basket):
                        _set_basket([*basket, p["sku"]])
        st.markdown(f"**{t('your_basket', lang)}**")
        if not basket:
            st.info(t("empty_basket", lang))
        for sku in basket:
            p = product(sku)
            cat_txt = f" *({category_label(p['category'])})*" if p else ""
            st.markdown(f"- {describe(sku)} · `{sku}`{cat_txt}")

    with c2:
        st.subheader(f"💡 {t('suggestions', lang)}")
        rec = api("POST", "/recommend", json={"basket": basket, "k": 10})
        by_rule = [r for r in rec["recommendations"] if r["source"] == "rule"]
        popular = [r for r in rec["recommendations"] if r["source"] == "popularity"]
        if not basket:
            st.warning(t("pick_first", lang))
        elif by_rule:
            st.success(t("found_rules", lang).format(n=len(by_rule)))
            table = pd.DataFrame(
                {
                    t("col_item", lang): [
                        f"{r['description'].title()} · {r['item']}" for r in by_rule
                    ],
                    t("col_trigger", lang): [
                        ", ".join(describe(s) for s in r["because"] or []) for r in by_rule
                    ],
                    t("col_conf", lang): [f"{r['confidence']:.1%}" for r in by_rule],
                    t("col_lift", lang): [round(r["lift"], 2) for r in by_rule],
                    t("col_support", lang): [
                        f"{r['support']:.2%}" if r.get("support") is not None else "—"
                        for r in by_rule
                    ],
                },
                index=pd.RangeIndex(1, len(by_rule) + 1, name=t("col_rank", lang)),
            )
            st.dataframe(table, width="stretch")
            top = by_rule[0]
            st.markdown(
                "> 🌟 "
                + t("insight", lang).format(
                    a=", ".join(f"`{describe(s)}`" for s in top["because"] or []),
                    b=top["description"].title(),
                    lift=top["lift"],
                )
            )
        else:
            st.info(t("no_rule", lang))
        if popular:
            st.caption(f"{t('popular', lang)}: " + ", ".join(
                f"{r['description'].title()} (`{r['item']}`)" for r in popular[:5]
            ))  # fmt: skip
        candidates = [r["item"] for r in (by_rule or popular[:5])]
        if candidates:
            a, b = st.columns([4, 1], vertical_alignment="bottom")
            choice = a.selectbox(
                t("add_suggestion", lang),
                candidates,
                format_func=lambda s: f"{describe(s)} · {s}",
            )
            if b.button(t("add", lang), key="add_suggestion"):
                _set_basket([*basket, choice])
        st.caption(
            f"{t('strategy', lang)}: `{rec['strategy']}` · rules mined {rec['rules_mined_at'] or '—'}"
        )

    st.divider()
    st.subheader(f"⚙️ {t('rules_metrics', lang)}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(t("m_rules", lang), f"{summary['rule_count']:,}")
    m2.metric(t("m_baskets", lang), f"{summary['n_baskets'] or 0:,}")
    m3.metric(t("m_support", lang), f"{summary['min_support'] or 0:.1%}")
    m4.metric(t("m_conf", lang), f"{summary['min_confidence'] or 0:.0%}")
    st.caption(
        f"{summary['algorithm'] or '—'} · min lift {summary['min_lift']:.1f} · "
        f"max itemset size {summary['max_len'] or '—'} · "
        f"{summary['n_antecedent_items']:,} products can trigger a rule · "
        f"mined {summary['mined_at'] or '—'}"
    )


# ------------------------------------------------------------------------ 3. assistant ---


def stream_chat(body: dict[str, Any], meta: dict[str, Any]) -> Iterator[str]:
    """Yield answer tokens from the SSE stream; ``meta`` collects the meta/replace/done events."""
    with client().stream("POST", "/chat/stream", json=body) as r:
        if r.status_code != 200:
            r.read()
            yield f"⚠️ {r.json().get('detail', r.status_code)}"
            return
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            ev = json.loads(line[6:])
            if ev["type"] == "token":
                yield ev["text"]
            else:
                meta[ev["type"]] = ev


def render_sources(resp: dict[str, Any], lang: str) -> None:
    if resp.get("handoff"):
        st.caption(f"↪ {t('handoff', lang)}")
    if any(f.startswith("llm_output_replaced") for f in resp.get("guardrail_flags", [])):
        st.caption(f"🛡 {t('replaced', lang)}")
    if resp.get("citations"):
        with st.expander(f"{t('sources', lang)} ({len(resp['citations'])})"):
            for c in resp["citations"]:
                score = f" · similarity {c['score']:.2f}" if c.get("score") is not None else ""
                st.markdown(
                    f"- `{c['ref']}` — {c['title']}{' · ' + c['section'] if c.get('section') else ''}{score}"
                )
    st.caption(f"{resp['provider']} · tool `{resp['tool']}` · {resp['latency_ms']:.0f} ms")


with tabs[2]:
    provs = providers()
    names = [p["name"] for p in provs]
    avail = {p["name"]: p for p in provs}
    choice = st.radio(
        t("provider", lang),
        names,
        horizontal=True,
        format_func=lambda n: f"{n}{'' if avail[n]['available'] else ' (unavailable)'}",
    )
    if not avail[choice]["available"]:
        st.info(f"{choice}: {avail[choice]['detail']}")
    history: list[dict[str, Any]] = st.session_state.setdefault("chat", [])
    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["text"])
            if msg.get("resp"):
                render_sources(msg["resp"], lang)
    if prompt := st.chat_input(t("ask", lang), disabled=not avail[choice]["available"]):
        history.append({"role": "user", "text": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            meta: dict[str, Any] = {}
            text = st.write_stream(
                stream_chat({"message": prompt, "provider": choice, "customer_ref": me}, meta)
            )
            resp = meta.get("done", {}).get("response")
            if "replace" in meta:  # guardrail rejected the streamed wording
                text = meta["replace"]["text"]
                st.markdown(text)
            if resp:
                render_sources(resp, lang)
        history.append({"role": "assistant", "text": str(text), "resp": resp})


# ----------------------------------------------------------------------- 4. monitoring ---


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def _ts(x: float | None) -> str:
    return datetime.fromtimestamp(x, tz=UTC).strftime("%Y-%m-%d %H:%M UTC") if x else "—"


@st.fragment(run_every="20s")
def monitoring(lang: str) -> None:
    rep = api("GET", "/monitoring/report")
    st.markdown(
        f"### {t('overall', lang)}: {status_badge(rep['status'], lang)}", unsafe_allow_html=True
    )
    w = rep["window"]
    st.caption(
        f"Window: last {w['predictions']} predictions ({_ts(w['from'])} → {_ts(w['to'])}), "
        f"last {w['requests']} requests · reference: {rep['reference']['source']} · refreshes every 20 s"
    )
    cols = st.columns(len(rep["checks"]))
    for col, c in zip(cols, rep["checks"], strict=True):
        with col, st.container(border=True):
            st.markdown(f"**{c['name'].replace('_', ' ')}**")
            st.markdown(status_badge(c["status"], lang), unsafe_allow_html=True)
            st.caption(f"{_fmt(c['value'])} · {c['unit']}")
    if w["predictions"] < w["min_predictions"]:
        st.info(t("no_traffic", lang))

    ops = rep["operations"]
    st.markdown("#### Operations")
    if ops.get("by_endpoint"):
        st.dataframe(
            pd.DataFrame(ops["by_endpoint"]).T.rename_axis("endpoint").round(3),
            width="stretch",
        )
    ts = pd.DataFrame(api("GET", "/monitoring/timeseries"))
    if len(ts):
        ts["time"] = pd.to_datetime(ts["ts"], unit="s")
        charts = [
            ("requests", "Requests per minute"),
            ("p95_ms", "p95 latency (ms)"),
            ("mean_p_return", "Mean predicted p_return"),
        ]
        if "shifted_share" in ts:
            charts.append(
                ("shifted_share", "Share of simulated shifted cohort (label from the simulator)")
            )
        grid = st.columns(2)
        for i, (col, title) in enumerate(charts):
            if col not in ts:
                continue
            chart = (
                alt.Chart(ts.dropna(subset=[col]), title=title)
                .mark_line(strokeWidth=2, color=BLUE, point=alt.OverlayMarkDef(size=30, color=BLUE))
                .encode(
                    x=alt.X("time:T", title=None),
                    y=alt.Y(f"{col}:Q", title=None),
                    tooltip=[
                        alt.Tooltip("time:T", format="%H:%M"),
                        alt.Tooltip(f"{col}:Q", format=".3f"),
                    ],
                )
                .properties(height=200)
            )
            grid[i % 2].altair_chart(chart, width="stretch")

    if rep["features"]:
        st.markdown(f"#### {t('features_psi', lang)}")
        feats = pd.DataFrame(rep["features"])
        top = feats.head(15)
        bars = (
            alt.Chart(top)
            .mark_bar(color=BLUE, cornerRadiusEnd=4)
            .encode(
                x=alt.X("psi:Q", title="PSI"),
                y=alt.Y("feature:N", sort="-x", title=None),
                tooltip=[
                    "feature",
                    alt.Tooltip("psi:Q", format=".3f"),
                    alt.Tooltip("ks:Q", format=".3f"),
                    "status",
                ],
            )
        )
        rules = (
            alt.Chart(pd.DataFrame({"x": [0.10, 0.25], "label": ["amber 0.10", "red 0.25"]}))
            .mark_rule(strokeDash=[4, 3], color="#8a8a85")
            .encode(x="x:Q", tooltip=["label"])
        )
        st.altair_chart((bars + rules).properties(height=28 * len(top) + 40), width="stretch")
        with st.expander("All features (table)"):
            st.dataframe(
                feats[
                    ["feature", "psi", "ks", "ks_pvalue", "missing_ref", "missing_live", "status"]
                ].round(4),
                width="stretch",
            )
        st.caption(
            f"Excluded from drift: {', '.join(rep['excluded_features']) or 'none'} (see params.yaml)"
        )

    p = rep["prediction"]
    st.markdown("#### Prediction drift and data quality")
    a, b, c = st.columns(3)
    a.metric(
        "Mean p_return (live vs reference)",
        _fmt(p.get("mean_live")),
        None if p.get("mean_live") is None else f"{p['mean_live'] - p['mean_ref']:+.3f}",
    )
    b.metric(
        "Proposed-coupon rate",
        _fmt(p.get("send_coupon_rate_live")),
        None
        if p.get("send_coupon_rate_live") is None
        else f"{p['send_coupon_rate_live'] - p['send_coupon_rate_ref']:+.3f}",
    )
    rf = rep["recommend"]
    c.metric(
        "Rule fire-rate (reference)",
        _fmt(rf["rule_fire_rate"]),
        None
        if rf["rule_fire_rate"] is None
        else f"{rf['rule_fire_rate'] - rf['rule_fire_rate_ref']:+.3f}",
    )
    dq = rep["data_quality"]
    st.caption(
        f"Worst missing-rate change: {dq['worst_missing_feature'] or '—'} ({_fmt(dq['max_missing_delta'])}) · "
        f"/predict 422 rate {_fmt(dq['predict_validation_error_rate'])} · unknown-SKU rate {_fmt(dq['unknown_item_rate'])}"
    )

    gt = rep.get("ground_truth", {})
    st.markdown("#### Delayed ground truth")
    st.write(
        f"The retention label is observable only {gt.get('label_window_days', 90)} days after a prediction. "
        f"First logged prediction: **{_ts(gt.get('first_prediction_ts'))}** → first true accuracy signal "
        f"expected **{_ts(gt.get('first_true_signal_ts'))}** (90 days + weekly backfill). "
        f"Labelled predictions so far: **{gt.get('n_labelled', 0)}**"
        + (
            f" — realised PR-AUC {gt['pr_auc']:.3f}, lapser recall {gt['lapser_recall']:.3f} (historical replay)"
            if "pr_auc" in gt
            else ""
        )
    )

    st.markdown(f"#### {t('alerts', lang)}")
    al = api("GET", "/monitoring/alerts", params={"limit": 30})
    if al:
        df = pd.DataFrame(al)
        df["time"] = df["ts"].map(_ts)
        df["level"] = df["level"].map(
            lambda s: f"{STATUS.get(s, STATUS['insufficient_data'])[1]} {s}"
        )
        st.dataframe(
            df[["time", "check_name", "previous", "level", "value", "notified", "message"]],
            width="stretch",
            hide_index=True,
        )
    else:
        st.caption("—")


with tabs[3]:
    monitoring(lang)


# ---------------------------------------------------------------------------- 5. about ---


def _read(name: str) -> str:
    p = ROOT / name
    return p.read_text(encoding="utf-8") if p.exists() else f"({name} not found)"


def privacy_notice(lang: str) -> str:
    gov = _read("GOVERNANCE_LOI25.md")
    tag = "**Français.**" if lang == "fr" else "**English.**"
    m = re.search(re.escape(tag) + r"(.*?)(?:\n\n\*\*|\n## |\Z)", gov, flags=re.S)
    return m.group(1).strip() if m else gov


with tabs[4]:
    st.subheader(t("privacy_notice", lang))
    st.info(privacy_notice(lang))
    arch = ROOT / "reports" / "figures" / "architecture.png"
    st.subheader(t("architecture", lang))
    if arch.exists():
        st.image(
            str(arch), caption="Offline pipeline → artifacts → online API → interface & monitoring"
        )
    with st.expander(t("model_card", lang)):
        st.markdown(_read("MODEL_CARD.md"))
    with st.expander(t("data_card", lang)):
        st.markdown(_read("DATA_CARD.md"))
    st.caption(
        "Boréal Marché is a fictional retailer; the data is the public UCI Online Retail II dataset "
        "(CC BY 4.0) re-skinned for a pedagogical scenario (420-977-VA capstone)."
    )
