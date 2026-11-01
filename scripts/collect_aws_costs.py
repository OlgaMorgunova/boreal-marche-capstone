"""D5 evidence: actual AWS spend by service, from Cost Explorer (the "total cloud spend" figure).

    aws sso login --profile admin-profile
    python scripts/collect_aws_costs.py --profile admin-profile --start 2026-09-01

Writes ``reports/metrics/aws_costs.json`` (monthly unblended cost per service, all regions). Each
Cost Explorer API call is billed $0.01; this script makes one. Until it is run the D5 report shows
the total spend as PENDING rather than an estimate.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "metrics" / "aws_costs.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="admin-profile")
    ap.add_argument("--start", default="2026-09-01", help="first day (inclusive), YYYY-MM-DD")
    ap.add_argument("--end", default=date.today().isoformat(), help="last day (exclusive)")
    a = ap.parse_args()
    raw = subprocess.run(
        ["aws", "ce", "get-cost-and-usage", "--profile", a.profile, "--region", "us-east-1",
         "--time-period", f"Start={a.start},End={a.end}", "--granularity", "MONTHLY",
         "--metrics", "UnblendedCost", "--group-by", "Type=DIMENSION,Key=SERVICE",
         "--output", "json"],
        check=True, capture_output=True, text=True,
    ).stdout  # fmt: skip
    by_service: dict[str, float] = {}
    months = []
    for period in json.loads(raw)["ResultsByTime"]:
        month = {}
        for g in period["Groups"]:
            amount = float(g["Metrics"]["UnblendedCost"]["Amount"])
            if amount >= 0.005:
                month[g["Keys"][0]] = round(amount, 2)
                by_service[g["Keys"][0]] = by_service.get(g["Keys"][0], 0.0) + amount
        months.append({"start": period["TimePeriod"]["Start"], "services": month})
    doc = {
        "collected_at": date.today().isoformat(),
        "period": {"start": a.start, "end_exclusive": a.end},
        "currency": "USD",
        "total": round(sum(by_service.values()), 2),
        "by_service": {k: round(v, 2) for k, v in sorted(by_service.items(), key=lambda x: -x[1])},
        "months": months,
    }
    OUT.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(doc, indent=1))


if __name__ == "__main__":
    main()
