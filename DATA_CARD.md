# Data Card — Boréal Marché (re-skinned UCI Online Retail II)

> **Provenance, stated plainly:** the underlying data are the public **UCI Online Retail II**
> dataset — real transactions of a UK-based online giftware wholesaler, 2009-12-01 to 2011-12-09 —
> **re-skinned as the fictional Québec retailer Boréal Marché for a pedagogical scenario.** They are
> not Boréal Marché data, not proprietary, and not Québec customers.

| Field | Value |
|---|---|
| Source | UCI Machine Learning Repository, dataset 502 — https://archive.ics.uci.edu/dataset/502 |
| Download URL | https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip |
| Citation | Chen, D. (2012). *Online Retail II* [Dataset]. UCI MLR. https://doi.org/10.24432/C5CG6D |
| Licence | CC BY 4.0 (attribution required; redistribution permitted). The repository still does not commit the raw file (course rule §3.1). |
| File | `online_retail_II.xlsx`, two sheets (`Year 2009-2010`: 525,461 rows; `Year 2010-2011`: 541,910 rows) |
| SHA-256 | `bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980` (verified by `make data`) |
| Rows | 1,067,371 lines · 53,628 invoices · 5,305 stock codes · 5,942 customer ids |
| Retrieval | `data/manifest.json` (`retrieved_at`, acquisition method) |
| Versioning | DVC (`data/raw/online_retail_II.xlsx.dvc`); never in Git |

## Schema (bronze, explicit Parquet schema)

`row_id int64` · `invoice string` (prefix `C` = cancellation, `A` = bad-debt adjustment) ·
`stock_code string` · `description string` · `quantity int32` · `invoice_date timestamp[ns]` ·
`price float64` (unit price, GBP in the source) · `customer_id int64, nullable` · `country string` ·
`source_sheet string`.

## Re-skin mapping (`src/boreal_marche/reskin.py`)

| Source | Boréal Marché | Note |
|---|---|---|
| United Kingdom | `QC` (Québec) | home market, ~92% of lines |
| Continental Europe (DE, FR, NL, BE, CH, AT, DK, SE, FI, NO, IT, ES, PT, PL, GR, CZ, LT, "European Community") | `ON` (Ontario) | |
| EIRE, Channel Islands, Iceland, Malta, Cyprus | `MARITIMES` | island/coastal markets |
| Everything else incl. "Unspecified" | `INTL` | export |
| Product description | keyword category (`seasonal`, `lighting`, `bags_wrap`, `kitchen`, `stationery_party`, `kids`, `home_decor`, `garden`, `accessories`, `other`) | heuristic, first matching keyword |
| Amounts | labelled CAD | **numerically the GBP amounts; no FX conversion** |

Quantities, prices, dates and identifiers are unchanged. The products remain giftware (the scenario
says "grocery and specialty goods"); this mismatch is accepted and stated.

## Known quality issues and treatment

See `reports/cleaning_report.md` for exact counts. Overlap of the two sheets (2010-12-01…09,
22,523 lines) · exact duplicate lines · cancellations (separated, never netted) · bad-debt
adjustments · non-positive quantity/price · service and fee codes (POST, DOT, M, C2, BANK CHARGES,
AMAZONFEE, TEST*, gift vouchers…) · spelling variants per SKU (canonicalised) · extreme bulk lines
(flagged, revenue capped) · ~23% of lines without customer id (excluded from Head A, kept for Head B).

## Intended and out-of-scope uses

**Intended:** teaching and portfolio demonstration of retention modelling, basket analysis and MLOps.
**Out of scope:** any claim about real Québec consumers, real regional behaviour (regions are a
mapping), grocery demand, or individuals. **Do not** attempt to re-identify customers (course §8.5).

## Personal information

Customer ids are opaque integers; no names, e-mails or addresses exist. Country (re-skinned as region) is the only
quasi-identifier. Hosted LLM providers never receive rows from this dataset.

## What the platform uses and stores (D5 data minimisation)

The gold table still computes all 37 point-in-time features (`reports/feature_spec.md`), so the
D2/D3 analyses stay reproducible. Since v0.9.5 the model uses 19 of them and the API stores only
those plus `region` (delivery, monitoring, fairness audit); the 18 removed features — time of day,
cancellations, category mix, basket shape, outlier share, lifetime value, and region as a model
input — failed the necessity test (`reports/metrics/minimisation.json`, D5 §6).

## Known biases relevant to the fairness audit

The data come from one UK giftware wholesaler whose customers include many small businesses ordering
in bulk, so spend is very unevenly distributed (top spend quartile ≥ $1,155 over 180 days). The
region field is a re-skin of countries (92% "QC"), so region subgroups other than QC are small
(e.g. 14 MARITIMES customers on the test snapshot) and their metrics are noisy. Lapse rates differ
strongly by spend and tenure, which is what drives the recall gaps reported in D5 §7.

## Retention

Raw data: 2 years in the scenario, DVC-tracked and never in Git; derived tables are rebuilt by
`make all` and can be deleted at any time. Prediction logs: 13 months
(`GOVERNANCE_LOI25.md` §6).
