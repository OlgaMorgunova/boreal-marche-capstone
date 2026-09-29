"""Report figures built from pipeline outputs (``make reports``). Every number comes from
``reports/metrics/*.json`` or the data layers — nothing is typed in by hand."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from boreal_marche.config import Paths, load_params
from boreal_marche.reporting import style
from boreal_marche.utils import read_json


def _box(ax, x, y, w, h, text, fc=style.SURFACE, ec=style.BLUE, size=7.5, bold=False):  # type: ignore[no-untyped-def]
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.4
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=size,
        color=style.INK,
        weight="bold" if bold else "normal",
        wrap=True,
    )


def _arrow(ax, a, b, label=None, color=style.INK_2):  # type: ignore[no-untyped-def]
    ax.add_patch(
        FancyArrowPatch(
            a, b, arrowstyle="-|>", mutation_scale=10, lw=1.1, color=color, shrinkA=2, shrinkB=2
        )
    )
    if label:
        ax.text(
            (a[0] + b[0]) / 2 + 0.05,
            (a[1] + b[1]) / 2 + 0.08,
            label,
            fontsize=6.2,
            color=style.INK_2,
            ha="center",
        )


def architecture(paths: Paths) -> None:
    fig, ax = plt.subplots(figsize=(13, 7.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    ax.grid(False)
    ax.add_patch(
        FancyBboxPatch(
            (0.1, 3.75),
            12.8,
            3.3,
            boxstyle="round,pad=0.02",
            fc="#f3f7fd",
            ec=style.BLUE,
            lw=1,
            ls="--",
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (0.1, 0.15),
            12.8,
            3.35,
            boxstyle="round,pad=0.02",
            fc="#fdf5f1",
            ec=style.ORANGE,
            lw=1,
            ls="--",
        )
    )
    ax.text(
        0.25,
        6.85,
        "OFFLINE — scheduled batch (minutes), Makefile / weekly retrain workflow",
        fontsize=9,
        weight="bold",
        color=style.BLUE,
    )
    ax.text(
        6.5,
        3.28,
        "ONLINE — every request (milliseconds), FastAPI container",
        fontsize=9,
        weight="bold",
        color=style.ORANGE,
        ha="center",
    )
    # offline row 1: data pipeline
    _box(ax, 0.3, 5.6, 1.7, 0.95, "UCI Online Retail II\nxlsx + SHA-256\n(DVC-tracked)")
    _box(ax, 2.4, 5.6, 1.6, 0.95, "BRONZE\nParquet, explicit\nschema + manifest")
    _box(ax, 4.4, 5.6, 1.6, 0.95, "SILVER\n11 cleaning rules\n+ cleaning report")
    _box(ax, 6.4, 5.6, 1.7, 0.95, "GOLD (Head A)\npoint-in-time\nsnapshots, 37 feats")
    _box(ax, 6.4, 4.1, 1.7, 0.95, "Baskets (Head B)\nsparse CSR\ninvoice x item")
    _box(ax, 8.5, 5.6, 1.9, 0.95, "Spot-check (10+3)\nOptuna TPE+pruning\n[SageMaker HPO]")
    _box(ax, 10.8, 5.6, 2.0, 0.95, "Calibrate + profit\nthreshold -> MLflow\nregistry @champion")
    _box(ax, 8.5, 4.1, 1.9, 0.95, "FP-Growth mining\nsupport/conf sweep\nlift filter")
    _box(
        ax,
        2.4,
        4.1,
        1.6,
        0.95,
        "Policy corpus (Wk 9)\nchunk + multilingual\nembeddings",
        ec=style.NEUTRAL,
    )
    for a, b in [
        ((2.0, 6.07), (2.4, 6.07)),
        ((4.0, 6.07), (4.4, 6.07)),
        ((6.0, 6.07), (6.4, 6.07)),
        ((8.1, 6.07), (8.5, 6.07)),
        ((10.4, 6.07), (10.8, 6.07)),
        ((8.1, 4.57), (8.5, 4.57)),
    ]:
        _arrow(ax, a, b)
    _arrow(ax, (5.2, 5.6), (6.4, 4.8))
    # artifacts crossing the line, each directly below its producer
    _box(
        ax,
        10.5,
        2.35,
        2.3,
        0.75,
        "model.joblib + model_meta.json\n(pipeline+calibrator+threshold)",
        fc="#ffffff",
        ec=style.AQUA,
        size=7,
    )
    _box(
        ax,
        8.4,
        2.35,
        1.9,
        0.75,
        "rules.json\n(header: date,\nparams, data hash)",
        fc="#ffffff",
        ec=style.AQUA,
        size=7,
    )
    _box(
        ax,
        2.35,
        2.35,
        1.7,
        0.75,
        "vector index\n(artifacts/index)",
        fc="#ffffff",
        ec=style.NEUTRAL,
        size=7,
    )
    ax.text(
        6.5,
        3.6,
        "ARTIFACTS CROSS THE LINE — NOT THE ALGORITHM",
        fontsize=7.5,
        ha="center",
        color=style.INK_2,
        weight="bold",
    )
    _arrow(ax, (11.8, 5.6), (11.65, 3.1), color=style.AQUA)
    _arrow(ax, (9.45, 4.1), (9.35, 3.1), color=style.AQUA)
    _arrow(ax, (3.2, 4.1), (3.2, 3.1), color=style.NEUTRAL)
    # online row
    _box(
        ax,
        0.3,
        0.55,
        1.9,
        1.1,
        "Streamlit UI (Wk 13)\nprediction · basket\nassistant · monitor · about",
        ec=style.NEUTRAL,
        size=7,
    )
    _box(
        ax,
        2.7,
        0.45,
        4.6,
        1.35,
        "FastAPI (Pydantic v2, extra=forbid)\n/health /ready /version\n"
        "/predict /predict/batch /recommend   [/chat /metrics: D4]\nlifespan: registry@champion "
        "-> local fallback",
        ec=style.ORANGE,
        size=7.2,
    )
    _box(
        ax,
        7.8,
        0.45,
        2.1,
        1.35,
        "Head C assistant (Wk 9-10)\nrule | OpenAI | Ollama\ntools call /predict,\n/recommend",
        ec=style.NEUTRAL,
        size=7,
    )
    _box(
        ax,
        10.4,
        0.45,
        2.4,
        1.35,
        "Prediction log (DB) ->\nmonitoring: PSI/KS drift,\nlatency, fire-rate, alerts\n(D4)",
        ec=style.NEUTRAL,
        size=7,
    )
    _arrow(ax, (2.2, 1.1), (2.7, 1.1))
    _arrow(ax, (11.2, 2.35), (7.1, 1.8), "load once at startup")
    _arrow(ax, (8.9, 2.35), (6.2, 1.8))
    _arrow(ax, (3.2, 2.35), (3.4, 1.8), color=style.NEUTRAL)
    _arrow(ax, (7.8, 1.1), (7.3, 1.1), "tool calls")
    _arrow(ax, (7.3, 0.7), (10.4, 0.7), "structured JSON log")
    ax.text(
        12.85,
        0.2,
        "grey = designed, not yet built (later deliverables)",
        fontsize=6.5,
        ha="right",
        color=style.INK_2,
    )
    fig.tight_layout()
    fig.savefig(paths.figures / "architecture.png", dpi=170)
    plt.close(fig)


def windows(paths: Paths) -> None:
    w = load_params().windows
    cuts = [(s, c) for s in ("train", "validation", "test") for c in getattr(w.cutoffs, s)]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    colors = {"train": style.BLUE, "validation": style.ORANGE, "test": style.AQUA}
    for i, (split, c) in enumerate(cuts):
        t = pd.Timestamp(c)
        y = len(cuts) - i
        obs0 = t - pd.Timedelta(days=w.observation_days)
        ax.barh(y, (t - obs0).days, left=obs0, height=0.5, color=colors[split], alpha=0.35)
        ax.barh(y, w.label_days, left=t, height=0.5, color=colors[split])
        ax.plot([t, t], [y - 0.35, y + 0.35], color=style.INK, lw=1.2)
        ax.text(
            pd.Timestamp("2009-12-05"),
            y,
            f"{split} · T={c}",
            va="center",
            fontsize=7.5,
            color=style.INK,
        )
    ax.axvline(pd.Timestamp("2009-12-01"), color=style.NEUTRAL, ls=":", lw=1)
    ax.axvline(pd.Timestamp("2011-12-09"), color=style.NEUTRAL, ls=":", lw=1)
    ax.set_yticks([])
    ax.set_xlim(pd.Timestamp("2009-11-01"), pd.Timestamp("2012-01-15"))
    ax.set_title(
        f"Observation window ({w.observation_days} d, light) and label window "
        f"({w.label_days} d, solid) per cutoff T"
    )
    ax.text(
        pd.Timestamp("2011-12-12"), 0.6, "data end\n2011-12-09", fontsize=6.5, color=style.INK_2
    )
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(paths.figures / "windows.png", dpi=160)
    plt.close(fig)


def eda(paths: Paths) -> None:
    lines = pd.read_parquet(
        paths.silver_lines, columns=["invoice", "invoice_date", "has_customer", "line_revenue"]
    )
    inv = lines.groupby("invoice").agg(
        d=("invoice_date", "min"), cust=("has_customer", "first"), n=("invoice", "size")
    )
    monthly = inv.groupby([inv["d"].dt.to_period("M"), "cust"]).size().unstack(fill_value=0)
    monthly = monthly[monthly.index < pd.Period("2011-12", "M")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    x = monthly.index.to_timestamp()
    axes[0].plot(x, monthly[True], color=style.BLUE, label="registered customer")
    axes[0].plot(x, monthly[False], color=style.ORANGE, label="guest (no customer id)")
    axes[0].set_title("Clean invoices per month (Dec 2011 partial, omitted)")
    axes[0].set_ylabel("invoices")
    axes[0].legend()
    sizes = inv["n"].clip(upper=100)
    axes[1].hist(sizes, bins=50, color=style.BLUE, edgecolor=style.SURFACE, linewidth=0.5)
    axes[1].axvline(inv["n"].median(), color=style.INK_2, ls="--", lw=1)
    axes[1].text(
        inv["n"].median() + 2,
        axes[1].get_ylim()[1] * 0.9,
        f"median {inv['n'].median():.0f} lines",
        fontsize=8,
        color=style.INK_2,
    )
    axes[1].set_title("Lines per invoice (capped at 100 for display)")
    axes[1].set_xlabel("distinct lines in basket")
    fig.tight_layout()
    fig.savefig(paths.figures / "eda_overview.png", dpi=160)
    plt.close(fig)


def cleaning(paths: Paths) -> None:
    rep = read_json(paths.metrics / "cleaning_report.json")
    rules = [r for r in rep["rules"] if r["rows_removed"] > 0]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    names = [r["rule"] for r in rules][::-1]
    vals = [r["rows_removed"] for r in rules][::-1]
    ax.barh(names, vals, color=style.BLUE, height=0.6)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:,}", va="center", fontsize=7.5, color=style.INK)
    ax.set_xscale("log")
    ax.set_xlabel("rows removed (log scale)")
    ax.set_title("Rows affected per cleaning rule (R10 applies to the Head A view only)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(paths.figures / "cleaning_rules.png", dpi=160)
    plt.close(fig)


def rules_sweep(paths: Paths) -> None:
    t = pd.DataFrame(read_json(paths.metrics / "rules_sweep.json")["table"])
    rp = load_params().recommend
    sub = t[(t["min_lift"] == rp.min_lift) & (t["rank_by"] == "confidence")]
    piv = sub.pivot(index="min_confidence", columns="min_support", values="hit_rate_hybrid")
    cnt = sub.pivot(index="min_confidence", columns="min_support", values="n_rules")
    fig, ax = plt.subplots(figsize=(8, 3.4))
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq", style.SEQ)
    im = ax.imshow(piv.to_numpy(), cmap=cmap, aspect="auto", origin="lower")
    ax.set_xticks(range(piv.shape[1]), [f"{c:g}" for c in piv.columns])
    ax.set_yticks(range(piv.shape[0]), [f"{c:g}" for c in piv.index])
    ax.set_xlabel("min_support")
    ax.set_ylabel("min_confidence")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            ax.text(
                j,
                i,
                f"{v:.3f}\n{int(float(cnt.iat[i, j])):,} rules",
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if v > np.nanmean(piv.to_numpy()) else style.INK,
            )
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="hit-rate@5 (validation slice)")
    ax.set_title(
        f"Head B sweep: hit-rate@5 and rule count (lift >= {rp.min_lift:g}, rank by confidence)"
    )
    fig.tight_layout()
    fig.savefig(paths.figures / "rules_sweep.png", dpi=160)
    plt.close(fig)


def segments(paths: Paths) -> None:
    seg = pd.DataFrame(read_json(paths.metrics / "test_evaluation.json")["segments"])
    seg = seg[seg["n"] >= 50].copy()
    seg["label"] = seg["segment"].str.replace("_", " ") + ": " + seg["value"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    y = np.arange(len(seg))
    axes[0].barh(y, seg["pr_auc"], color=style.BLUE, height=0.6)
    axes[0].set_yticks(y, seg["label"])
    axes[0].set_title("PR-AUC by segment (test, n >= 50)")
    axes[1].barh(y, seg["lapse_recall"], color=style.ORANGE, height=0.6)
    axes[1].set_title("Recall of lapsers at the operating threshold")
    for ax, col in ((axes[0], "pr_auc"), (axes[1], "lapse_recall")):
        for i, v in enumerate(seg[col]):
            ax.text(v, i, f" {v:.2f}", va="center", fontsize=7, color=style.INK)
        ax.set_xlim(0, 1.1)
        ax.grid(axis="y", visible=False)
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(paths.figures / "error_segments.png", dpi=160)
    plt.close(fig)


def run() -> None:
    paths = Paths()
    paths.ensure()
    for fn in (architecture, windows, eda, cleaning, rules_sweep, segments):
        fn(paths)
        print("figure:", fn.__name__)


if __name__ == "__main__":
    run()
