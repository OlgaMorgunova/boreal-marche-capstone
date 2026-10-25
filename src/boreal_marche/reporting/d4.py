"""D4 template values and figure, computed from the saved evidence (never typed by hand).

Sources: ``reports/metrics/d4_evidence.json`` (scripts/collect_d4_evidence.py), the two monitoring
snapshots and alert history (scripts/simulate_traffic.py), ``assistant_eval.json``,
``backfill_demo.json``, ``latency_probe.json``, the Locust CSVs and the reference profile.
Missing live evidence renders as an explicit "PENDING" — the note never claims what was not done.
"""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from boreal_marche.config import Paths, load_params
from boreal_marche.reporting.style import BLUE, INK_2, NEUTRAL, ORANGE
from boreal_marche.utils import read_json

PENDING = "**PENDING**"


def _md(header: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _opt(path: Path) -> Any:
    return read_json(path) if path.exists() else None


def _pct(x: float | None, d: int = 1) -> str:
    return "—" if x is None else f"{100 * x:.{d}f}%"


def _locust(paths: Paths, phase: str) -> dict[str, dict[str, str]]:
    f = paths.reports / "loadtest" / f"{phase}_stats.csv"
    if not f.exists():
        return {}
    with f.open(encoding="utf-8") as fh:
        return {r["Name"]: r for r in csv.DictReader(fh)}


def monitoring_figure(paths: Paths, normal: dict, shifted: dict) -> None:
    feats = [f["feature"] for f in shifted["features"][:12]]
    psi_n = {f["feature"]: f["psi"] for f in normal["features"]}
    psi_s = {f["feature"]: f["psi"] for f in shifted["features"]}
    y = np.arange(len(feats))[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.barh(y + 0.2, [psi_n.get(f, 0) for f in feats], 0.38, color=BLUE, label="normal traffic")
    ax.barh(y - 0.2, [psi_s[f] for f in feats], 0.38, color=ORANGE, label="shifted cohort")
    for x, lab in ((0.10, "amber 0.10"), (0.25, "red 0.25")):
        ax.axvline(x, color=NEUTRAL, ls="--", lw=1)
        ax.text(x * 1.04, len(feats) - (0.3 if x < 0.2 else 0.9), lab, color=INK_2, fontsize=7)
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_yticks(y, feats)
    ax.set_xlabel("PSI vs reference (symmetric log scale)")
    ax.set_title("Input drift per feature — 12 most drifted under the shifted cohort")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(paths.figures / "d4_monitoring_psi.png", dpi=160)
    plt.close(fig)


def d4_context(paths: Paths) -> dict[str, str]:
    m = paths.metrics
    ev = _opt(m / "d4_evidence.json") or {}
    normal, shifted = _opt(m / "monitoring_normal.json"), _opt(m / "monitoring_shifted.json")
    alerts = _opt(m / "monitoring_alerts.json") or []
    ae = _opt(m / "assistant_eval.json")
    bf = _opt(m / "backfill_demo.json")
    probe = _opt(m / "latency_probe.json")
    prof = read_json(paths.reference_profile)
    idx = read_json(paths.index_dir / "index_meta.json")
    p = load_params()
    c: dict[str, str] = {}

    live = ev.get("live") or {}
    c["live_ui_url"] = live.get("ui_url") or PENDING
    c["live_api_docs"] = f"{live['api_url']}/docs" if live.get("api_url") else PENDING
    c["live_version"] = (
        f"`{live['version']['git_commit'][:12]}` (checked {live['checked_at']})"
        if live.get("version")
        else PENDING
    )
    c["video_url"] = ev.get("video_url") or PENDING

    imgs = ev.get("images", {})
    before, api, ui = (
        imgs.get(k, {}) for k in ("boreal-api:v0.3", "boreal-api:d4", "boreal-ui:d4")
    )
    c["img_table"] = _md(
        ["Image", "Unpacked size", "Compressed (pushed to ECR)", "Contents"],
        [
            ["API v0.3 (D3, single venv, `pip install .`)", f"{before.get('unpacked_mb', '—')} MB",
             f"{before.get('compressed_bytes', 0) / 1e6:.0f} MB", "Heads A+B only"],
            ["API v0.9 (this Dockerfile)", f"{api.get('unpacked_mb', '—')} MB",
             f"{api.get('compressed_bytes', 0) / 1e6:.0f} MB",
             "Heads A+B+C, monitoring, 241 MB embedding model baked in"],
            ["Interface v0.9", f"{ui.get('unpacked_mb', '—')} MB",
             f"{ui.get('compressed_bytes', 0) / 1e6:.0f} MB", "Streamlit + Altair, no model code"],
        ],
    )  # fmt: skip
    c["img_reduction"] = (
        _pct(1 - api["unpacked_mb"] / before["unpacked_mb"], 0) if api and before else "—"
    )
    t = ev.get("tests", {})
    c["tests_passed"], c["coverage"] = str(t.get("passed", "—")), f"{t.get('coverage_pct', '—')}%"
    corpus = ev.get("corpus", {})
    c["corpus_words"] = f"{corpus.get('words', 0):,}"
    c["corpus_docs"] = str(corpus.get("documents", "—"))
    c["n_chunks"], c["embed_model"] = str(idx["n_chunks"]), idx["embedding_model"]
    c["sim_floor"] = f"{p.assistant.similarity_floor:.2f}"

    # Assistant evaluation table.
    if ae:
        rows = []
        for set_name in ("golden", "paraphrase"):
            for prov, r in ae["results"].get(set_name, {}).items():
                if "skipped" in r:
                    rows.append([set_name, prov, "not run: " + r["skipped"], "", "", "", "", ""])
                    continue
                rows.append([
                    f"{set_name} ({r['n']})", prov, _pct(r["tool_choice_accuracy"]),
                    _pct(r["groundedness_source_at_k"]), _pct(r["oos_refusal_rate"]),
                    _pct(r["false_refusal_rate"]), f"{r['latency_ms_p50']:.0f} ms",
                    f"${r['cost_usd_per_query']:.4f}",
                ])  # fmt: skip
        c["assistant_table"] = _md(
            ["Question set", "Provider", "Tool choice", "Source cited", "OOS refused",
             "False refusals", "Latency p50", "Cost/query"],
            rows,
        )  # fmt: skip
        g = ae["results"]["golden"]
        c["rule_golden"] = _pct(g["rule"]["tool_choice_accuracy"], 0)
        c["rule_para"] = _pct(ae["results"]["paraphrase"]["rule"]["tool_choice_accuracy"], 0)
        c["ollama_golden"] = _pct(g.get("ollama", {}).get("tool_choice_accuracy"), 1)
        c["ollama_para"] = _pct(
            ae["results"]["paraphrase"].get("ollama", {}).get("tool_choice_accuracy"), 0
        )
    # Monitoring before/after.
    if normal and shifted:
        monitoring_figure(paths, normal, shifted)
        sn = {x["name"]: x for x in normal["checks"]}
        rows = []
        for x in shifted["checks"]:
            a = sn[x["name"]]
            fmt = lambda v: "—" if v is None else f"{v:.3f}"  # noqa: E731
            rows.append([x["name"].replace("_", " "), f"{a['status']} ({fmt(a['value'])})",
                         f"{x['status']} ({fmt(x['value'])})", x["unit"]])  # fmt: skip
        c["monitor_table"] = _md(
            ["Check", "After normal traffic", "After shifted cohort", "Measure"], rows
        )
        c["top_drift"] = ", ".join(
            f"`{f['feature']}` (PSI {f['psi']:.2f})" for f in shifted["features"][:4]
        )
        c["n_red_features"] = _pct(
            next(x for x in shifted["checks"] if x["name"] == "input_drift")["value"], 0
        )  # noqa: E501
        c["unknown_rate"] = _pct(shifted["data_quality"]["unknown_item_rate"], 0)
        c["fire_shifted"] = _pct(shifted["recommend"]["rule_fire_rate"], 0)
        c["fire_ref"] = _pct(shifted["recommend"]["rule_fire_rate_ref"], 0)
        c["p95_predict_server"] = (
            f"{shifted['operations']['by_endpoint']['/predict']['p95']:.0f} ms"
        )
    c["n_alerts"] = str(len(alerts))
    c["alerts_table"] = _md(
        ["Time (UTC)", "Check", "Transition", "Value", "Notified via"],
        [
            [datetime.fromtimestamp(a["ts"], tz=UTC).strftime("%H:%M:%S"), a["check_name"],
             f"{a['previous']} → **{a['level']}**",
             "—" if a["value"] is None else f"{a['value']:.3f}", a["notified"]]
            for a in sorted(alerts, key=lambda a: a["ts"])
        ],
    )  # fmt: skip
    for ph in ("normal", "shifted"):
        st = _locust(paths, ph)
        agg, pr = st.get("Aggregated", {}), st.get("/predict", {})
        c[f"lt_{ph}_reqs"] = agg.get("Request Count", "—")
        c[f"lt_{ph}_fails"] = agg.get("Failure Count", "—")
        c[f"lt_{ph}_rps"] = f"{float(agg.get('Requests/s', 0)):.1f}"
        c[f"lt_{ph}_p95"] = pr.get("95%", "—")
    if probe:
        c["probe_predict_p95"] = f"{probe['client_side_ms']['/predict']['p95']:.1f} ms"
        c["probe_recommend_p95"] = f"{probe['client_side_ms']['/recommend']['p95']:.1f} ms"
    ks = prof["known_shift_vs_pooled_training"]
    c["ref_rows"] = f"{prof['n_rows']:,}"
    c["known_shift_share"] = _pct(ks["share_of_features_psi_red"], 0)
    c["known_shift_top"] = ", ".join(f"`{f['feature']}` {f['psi']:.2f}" for f in ks["features"][:3])
    if bf:
        c["bf_n"] = f"{bf['n_labelled']:,}"
        c["bf_prauc"] = f"{bf['pr_auc']:.3f}"
        c["bf_recall"] = f"{bf['lapser_recall']:.3f}"
    return c
