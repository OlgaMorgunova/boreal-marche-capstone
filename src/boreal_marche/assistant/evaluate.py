"""Head C evaluation on the golden set (and the post-hoc paraphrase set), per provider.

    python -m boreal_marche.assistant.evaluate --providers rule,ollama

Runs every question through the real ``/chat`` endpoint in-process (tools call ``/predict``,
``/recommend`` and ``/products`` exactly as in production) and scores:

* **tool-choice accuracy** — first tool chosen == expected tool (``out_of_scope`` included);
* **groundedness** — policy questions answered (not handed off) whose citations include the
  expected source document (``source@k``), and whether it is the top citation (``source@1``);
* **out-of-scope refusal rate** and **false refusal rate** (in-scope questions refused);
* **latency** p50/p95 and **cost per query** (token-priced for hosted providers; $0 for local);
* **guardrail replacements** — LLM answers rejected for an invented SKU or percentage.

Also writes a similarity-floor sweep (handoff rate on answerable policy questions per floor).
Output: ``reports/metrics/assistant_eval.json``.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np
from fastapi.testclient import TestClient

from boreal_marche.config import Paths, load_params
from boreal_marche.utils import read_json, utc_now, write_json

SETS = {
    "golden": "tests/golden/assistant_golden.json",
    "paraphrase": "tests/golden/assistant_paraphrase.json",
}


def score(cases: list[dict[str, Any]], results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for c, r in zip(cases, results, strict=True):
        cited = [x["ref"].split("#")[0] for x in r["citations"] if x["kind"] == "policy"]
        rows.append(
            {
                "id": c["id"],
                "category": c["category"],
                "expected_tool": c["expected_tool"],
                "tool": r["tool"],
                "tool_ok": r["tool"] == c["expected_tool"],
                "refused": r["refused"],
                "handoff": r["handoff"],
                "source_at_k": c["expected_source"] in cited if c["expected_source"] else None,
                "source_at_1": (cited[:1] == [c["expected_source"]])
                if c["expected_source"]
                else None,
                "latency_ms": r["latency_ms"],
                "cost_usd": r["usage"]["est_cost_usd"],
                "replaced": any(f.startswith("llm_output_replaced") for f in r["guardrail_flags"]),
                "flags": r["guardrail_flags"],
                "answer": r["answer"][:300],
            }
        )
    policy = [x for x in rows if x["expected_tool"] == "policy_search"]
    oos = [x for x in rows if x["expected_tool"] == "out_of_scope"]
    in_scope = [x for x in rows if x["expected_tool"] != "out_of_scope"]
    lat = np.array([x["latency_ms"] for x in rows])
    by_cat = {}
    for cat in sorted({x["category"] for x in rows}):
        sub = [x for x in rows if x["category"] == cat]
        by_cat[cat] = float(np.mean([x["tool_ok"] for x in sub]))
    return {
        "n": len(rows),
        "tool_choice_accuracy": float(np.mean([x["tool_ok"] for x in rows])),
        "tool_choice_by_category": by_cat,
        "groundedness_source_at_k": float(np.mean([bool(x["source_at_k"]) for x in policy]))
        if policy
        else None,
        "groundedness_source_at_1": float(np.mean([bool(x["source_at_1"]) for x in policy]))
        if policy
        else None,
        "policy_handoff_rate": float(np.mean([x["handoff"] for x in policy])) if policy else None,
        "oos_refusal_rate": float(np.mean([x["refused"] for x in oos])) if oos else None,
        "false_refusal_rate": float(np.mean([x["refused"] for x in in_scope])),
        "latency_ms_p50": float(np.percentile(lat, 50)),
        "latency_ms_p95": float(np.percentile(lat, 95)),
        "cost_usd_per_query": float(np.mean([x["cost_usd"] for x in rows])),
        "llm_answers_replaced": int(sum(x["replaced"] for x in rows)),
        "cases": rows,
    }


def floor_sweep(client: TestClient, cases: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Best retrieval score of each answerable policy question vs candidate floors."""
    from boreal_marche import api

    tb = api.STATE["assistant"].toolbox
    best = [tb.policy_search(c["question"]).best_score for c in cases if c["expected_source"]]
    oos = [
        tb.policy_search(c["question"]).best_score
        for c in cases
        if c["expected_tool"] == "out_of_scope"
    ]
    return [
        {
            "floor": f,
            "policy_handoff_rate": float(np.mean([b < f for b in best])),
            "oos_below_floor_rate": float(np.mean([b < f for b in oos])),
        }
        for f in np.round(np.arange(0.20, 0.61, 0.05), 2)
    ]


def run(providers: list[str], sets: list[str]) -> dict[str, Any]:
    os.environ.setdefault("BOREAL_MODEL_SOURCE", "local")
    os.environ.setdefault("BOREAL_MONITOR_LOOP", "0")
    os.environ.setdefault("DATABASE_URL", "sqlite:///./eval_chat.db")
    from boreal_marche import api

    params, paths = load_params(), Paths()
    out: dict[str, Any] = {
        "generated_at": utc_now(),
        "similarity_floor": params.assistant.similarity_floor,
        "embedding_model": params.assistant.embedding_model,
        "models": {
            "ollama": params.assistant.ollama_model,
            "openai": params.assistant.openai_model,
            "gemini": params.assistant.gemini_model,
        },
        "results": {},
    }
    with TestClient(api.app) as client:
        api.STATE["assistant"].toolbox.api = client  # tools call this in-process API
        status = {p.name: p for p in api.STATE["assistant"].status()}
        for set_name in sets:
            data = read_json(paths.root / SETS[set_name])
            cases = data["cases"]
            out["results"][set_name] = {}
            for prov in providers:
                if not status[prov].available:
                    out["results"][set_name][prov] = {"skipped": status[prov].detail}
                    print(f"{set_name}/{prov}: skipped ({status[prov].detail})")
                    continue
                results = []
                for c in cases:
                    r = client.post(
                        "/chat",
                        json={
                            "message": c["question"],
                            "provider": prov,
                            "customer_ref": data["session_customer_ref"],
                        },
                        headers={"X-Traffic-Cohort": "eval"},
                    )
                    r.raise_for_status()
                    results.append(r.json())
                s = score(cases, results)
                out["results"][set_name][prov] = s
                print(
                    f"{set_name}/{prov}: tool={s['tool_choice_accuracy']:.3f} "
                    f"source@k={s['groundedness_source_at_k']} oos_refusal={s['oos_refusal_rate']} "
                    f"false_refusal={s['false_refusal_rate']:.3f} p50={s['latency_ms_p50']:.0f}ms "
                    f"replaced={s['llm_answers_replaced']}"
                )
        golden = read_json(paths.root / SETS["golden"])["cases"]
        out["floor_sweep"] = floor_sweep(client, golden)
    write_json(paths.metrics / "assistant_eval.json", out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--providers", default="rule,ollama,openai")
    ap.add_argument("--sets", default="golden,paraphrase")
    a = ap.parse_args()
    run(a.providers.split(","), a.sets.split(","))


if __name__ == "__main__":
    main()
