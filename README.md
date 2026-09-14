# Boréal Marché — AI Development Project Capstone 1

**Student:** Olga Morgunova  
**Course:** 420-977-VA — AI Development Project  
**Program:** LEA.EU — Data Science and Artificial Intelligence  
**Project:** Boréal Marché  
**Dataset:** UCI Online Retail II  
**Workload:** Individual project

## Project goal

Build one reproducible AI system for a Quebec online grocery/specialty retailer with three model heads:

1. **Retention prediction** — predict whether a customer will place another order within 90 days.
2. **Basket recommendations** — mine association rules and serve next-item recommendations.
3. **Bilingual grounded assistant** — answer policy/product questions in French and English using RAG and typed tools.

The three heads share one data pipeline and are exposed through one API and one interface.

## Status

- [ ] D1 — Project Proposal and Charter
- [ ] D2 — Model Design and Architecture
- [ ] D3 — Model Implementation and Validation
- [ ] D4 — Deployment Demonstration and Dashboard
- [ ] D5 — Optimization and Ethics
- [ ] D6 — Communication and Presentation
- [ ] D7 — Complete System and Defence

## Quickstart

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -e ".[dev]"
```

The data pipeline will be added after the source/provenance and checksum are finalized.

## Repository map

- `src/` — importable production/business logic
- `notebooks/` — exploration only
- `tests/` — unit, contract, data-quality and assistant tests
- `data/` — DVC-tracked data; raw data is never committed to Git
- `artifacts/` — model, rules and retrieval artifacts
- `docker/` — Docker/Compose files
- `frontend/` — Streamlit interface
- `docs/corpus/` — assistant policy/product documents
- `reports/` — cleaning and evaluation reports
- `.github/workflows/` — CI/CD/retraining workflows

## Data provenance

The underlying dataset is a public dataset re-skinned for the educational Boréal Marché scenario. It must not be represented as proprietary client data.

## Reproducibility

All required environment variables will be documented in `.env.example`. Secrets must never be committed.

## License / source

To be completed after verifying the current UCI dataset source and licence information.
