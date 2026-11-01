"""D5 cost evidence: tokens per assistant query, measured with the local Ollama provider.

    python scripts/assistant_token_probe.py          # needs Ollama running with llama3.1:8b

Runs the 40 golden questions through the in-process API with ``provider=ollama`` and records the
prompt/completion tokens and latency of each turn. The hosted provider sends the same prompts, so
these counts (llama3 tokenizer, within ~10% of OpenAI's) give the hosted cost per query at the list
prices in ``params.yaml -> assistant.openai_price_per_mtok``. Writes
``reports/metrics/assistant_tokens.json``.
"""

from __future__ import annotations

import os
import statistics
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boreal_marche.config import load_params  # noqa: E402
from boreal_marche.utils import read_json, utc_now, write_json  # noqa: E402


def main() -> None:
    os.environ.update(BOREAL_MODEL_SOURCE="local", BOREAL_MONITOR_LOOP="0",
                      DATABASE_URL=f"sqlite:///{Path(tempfile.mkdtemp()) / 'tok.db'}")  # fmt: skip
    from fastapi.testclient import TestClient

    from boreal_marche import api

    cases = read_json(ROOT / "tests" / "golden" / "assistant_golden.json")["cases"]
    demo = read_json(ROOT / "artifacts" / "demo_customers.json")["customers"][0]["ref"]
    rows = []
    with TestClient(api.app) as c:
        api.STATE["assistant"].toolbox.api = c
        for case in cases:
            body = {"message": case["question"], "provider": "ollama", "customer_ref": demo}
            r = c.post("/chat", json=body)
            r.raise_for_status()
            j = r.json()
            rows.append({"id": case["id"], "tool": j["tool"], "latency_ms": j["latency_ms"],
                         "prompt_tokens": j["usage"]["prompt_tokens"],
                         "completion_tokens": j["usage"]["completion_tokens"]})  # fmt: skip
            print(f"{case['id']:>8s} {j['tool']:16s} {rows[-1]['prompt_tokens']:5d} in "
                  f"{rows[-1]['completion_tokens']:4d} out {j['latency_ms']:7.0f} ms")  # fmt: skip
    price = load_params().assistant.openai_price_per_mtok
    pin = statistics.fmean(r["prompt_tokens"] for r in rows)
    pout = statistics.fmean(r["completion_tokens"] for r in rows)
    doc = {
        "generated_at": utc_now(),
        "model": load_params().assistant.ollama_model,
        "n": len(rows),
        "mean_prompt_tokens": round(pin, 1),
        "mean_completion_tokens": round(pout, 1),
        "latency_ms_p50": statistics.median(r["latency_ms"] for r in rows),
        "openai_list_price_per_mtok": {"input": price.input, "output": price.output},
        "openai_est_usd_per_1000_queries": round(
            (pin * price.input + pout * price.output) / 1e3, 4
        ),
        "cases": rows,
    }
    write_json(ROOT / "reports" / "metrics" / "assistant_tokens.json", doc)
    print({k: v for k, v in doc.items() if k != "cases"})


if __name__ == "__main__":
    main()
