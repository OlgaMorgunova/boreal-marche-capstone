<div class="title" markdown="1">

# D7 Defence Brief — Complete System, Readiness and Prepared Answers

Olga Morgunova · 420-977-VA AI Development Project (Capstone 1) · Learning Integration Assessment ·
release `v1.0` · instructor: Steve Ataky, Ph.D. · evidence collected 2026-11-01 20:45 UTC

</div>

This brief is the panel's map of the system and my preparation for the defence. The fifteen-minute
talk follows the D6 deck, demonstration first. Every number below is rendered from
`reports/metrics/*.json` by `python -m boreal_marche.reporting.build D7`. The readiness table
comes from `scripts/collect_d7_evidence.py`, run against the live service.

## 1. The system the panel can use

| Component | Where | State at collection |
|---|---|---|
| Interface (EN/FR) | https://boreal-marche-capstone-4ay3m2fn6qdo4jtgy72vpr.streamlit.app | sleeps when unused; the first visit wakes it in ~30 s |
| API and its documentation | https://padp32qm6p.us-east-1.awsapprunner.com/docs | API `0.9.6`, commit `bc0d660` (head of `master`: `bc0d660`) |
| Monitoring dashboard | interface → *Monitoring* tab; `GET /monitoring/report`, `/metrics` | in-process checks every 60 s; hourly external check (`monitor.yml`) |
| Source, release | `github.com/OlgaMorgunova/boreal-marche-capstone`, tag `v1.0` | tags `v0.1`, `v0.2`, `v0.3`, `v0.9`, `v0.9.5`, `v0.9.6` |
| Tests | `make cov` | 89 passed, 82% line coverage |
| Recorded defence presentation | https://youtu.be/STCDsmDr3s8 | "Boréal Marché Defense" (YouTube) |
| Fallback video | https://youtu.be/C_Qv7lu3cpY (walkthrough) · https://youtu.be/2bcNpRwMQxE (D6 talk) | in case the network fails |

**Demonstration path (≈ 4 minutes, all live).**
1. *Prediction tab*: first customer → probability, decision, principal factors. Drag
   "spend in last 180 days" to show the decision move.
2. Open *Campaign list for approval*: coupon / holdout / review / none for 300 customers, and the
   high-value review list.
3. *Basket tab*: add `22916`, `22918` → herb-marker rules (lift ≈ 140); an unknown item → the
   popularity fallback, labelled as such.
4. *Assistant tab*: "Combien de temps pour un remboursement ?" → cited French answer; "show me
   customer BM-…" for someone else → refused; a poem request → out of scope.
5. *Monitoring tab*, then `/docs` → `GET /version`: the commit equals the head of `master`.

## 2. Readiness check (course Appendix A)

| Area | Check | Status | Evidence |
|---|---|---|---|
| live | /health | ✔ pass | 200 |
| live | /ready | ✔ pass | [] |
| live | /version = head of master | ✔ pass | live bc0d6603ab70 · master bc0d6603ab70 · API 0.9.6 |
| head A | golden prediction reproduced | ✔ pass | p_return=0.4060586238648678 |
| head A | campaign action on /predict | ✘ **fail** | None |
| head A | /campaign list (review + hold-out) | ✘ **fail** | 404 |
| head B | /recommend fires a rule | ✔ pass | rules |
| head C | /chat answers in French with a citation | ✔ pass | tool=policy_search lang=fr citations=4 |
| head C | /chat refuses out-of-scope | ✔ pass | out_of_scope |
| monitoring | dashboard has traffic | ✔ pass | 8 predictions in window |
| live | interface reachable (303 = waking up) | ✔ pass | 303 |
| repository | public | ✘ **fail** | HTTP 404 (private or absent) |
| repository | ci.yml green on master | … pending | not readable while private (set GITHUB_TOKEN) |
| repository | cd.yml green on master | … pending | not readable while private (set GITHUB_TOKEN) |
| repository | annotated tag v0.1 | ✔ pass | tag |
| repository | annotated tag v0.2 | ✔ pass | tag |
| repository | annotated tag v0.3 | ✔ pass | tag |
| repository | annotated tag v0.9 | ✔ pass | tag |
| repository | annotated tag v1.0 | … pending | missing |
| repository | §7.2 structure | ✔ pass | 22 paths |
| repository | raw data not committed | ✔ pass | only the .dvc pointer |
| repository | README video above the fold | ✔ pass | link in the first 20 lines |
| cloud | SageMaker spot quota | … pending | CASE_OPENED (case 179073448000753, 4 instances) |
| manual | Application URL opens in an incognito window on a phone | ☐ manual |  |
| manual | Compose stack tested from a fresh clone on a different machine | ☐ manual |  |
| manual | Application warm and open in a browser tab before the defence | ☐ manual |  |
| manual | Fallback video at hand in case the network fails | ☐ manual |  |
| manual | Prepared: primary metric, baseline and threshold with their justification | ☐ manual |  |
| manual | Prepared: one answer to 'what would you do differently' and one negative result | ☐ manual |  |

*Table 1 — 16 automated checks pass, 3 fail, 4 are pending
and 6 need a person. Re-run the morning of the defence; a failure is fixed before
the defence, not explained during it.*

## 3. What v1.0 adds: the fairness mitigation is now served

D5 found that one value-blind threshold proposes only 46% of lapsing top-quartile
spenders against 89% overall. It adopted, without building, a human review list and
a randomised hold-out. Both now run in the API (`retention/campaign.py`). Every `/predict` returns an
`action`. `GET /campaign` returns the list the reviewer approves. The automatic proposal is
unchanged, so the golden predictions still hold.

* **review**: spend Q3 or Q4, not proposed, but under the M2 group threshold (Q3 ≤ 0.73,
  Q4 ≤ 0.88). Spend routes the case to a person; it never changes the automatic
  decision.
* **holdout**: 10% of proposed customers, drawn by
  `sha256(campaign_id:customer_ref)`. The draw is stateless, auditable and blind to every feature,
  and a new `campaign_id` each quarter draws a new sample. The customer notice now says so.

| Snapshot (served model) | Validation | Test |
|---|---:|---:|
| Proposed / review list / hold-out | 1,814 / 368 / 186 | 1,212 / 450 / 131 |
| Q4 lapser recall: automatic → if the whole list is approved | 0.46 → **0.89** | 0.16 → **0.82** |
| Q3 lapser recall: automatic → if approved | 0.70 → 0.89 | 0.34 → 0.66 |
| Lapsers in the review list | 26% | 28% |
| Profit if the whole list is approved (assumed save rate 20%) | −$8,221 | −$16,921 |
| Hold-out balance: mean `p_return` held out vs treated | 0.36 vs 0.35 | 0.45 vs 0.45 |
| Assumed profit given up by the hold-out (campaign net) | $2,663 ($7,695) | $452 (−$1,039) |
| Smallest detectable save rate after one quarter (5%, 80% power) | 0.16 | 0.21 |

*Table 2 — `reports/metrics/campaign_policy.json`. These counts reproduce D5's offline audit
exactly (M0 and M2 coupons), which checks the served code against the analysis. Campaign profit
here uses the served calibrator; D3's $7,421 used out-of-fold calibration.*

**Read honestly:** the list closes the gap only if a person approves it, and approving it all
*loses* money under the assumptions. The list is for judgment, not rubber-stamping. The hold-out is
the price of finally measuring the save rate: it needs one to two quarters to confirm the assumed
20%.

## 4. Prepared answers to the question bank (Appendix C)

**Data and modelling**

**4. Windows; why 90 days.** Observation 180 days before a cut-off *T*, label = any order
in [T, T+90 d). Cut-offs: train 2010-06-01, 2010-09-01, 2010-12-01, 2011-03-01, validation 2011-06-01, test
2011-09-01. Ninety days matches the client's quarterly campaign, and the median gap between
orders is about two months, so 90 days separates "slow" from "gone".

**5. Why PR-AUC.** Classes are near balance overall but the action targets one side. Accuracy hides
which errors we make, and ROC-AUC rewards ranking the many easy negatives. PR-AUC measures
precision and recall across thresholds. Test 0.847 [0.832, 0.863] against
0.7123 for recency deciles, 0.607 for the prior. I also report the lapse
direction (0.6438).

**6. Threshold 0.61.** It maximises campaign profit on validation under stated costs:
$4.00 per coupon, a 15% discount, 20% save rate,
30% margin and a 365-day value horizon. At a 10% save rate the optimum is
"contact nobody" (sensitivity table, D6 §5.3). The number is a business decision with its
assumptions printed.

**7. What calibration buys.** We do not only rank. Expected profit multiplies a *probability* by a
value, the threshold is in probability units, and monitoring compares mean predicted `p_return`
with the realised return rate. Test ECE 0.041.

**8. Most important feature.** `active_months_180d` (months with an order of the last six), with
50% of attribution. It surprised me that regularity beats recency; recency is
partly redundant with it. Top five: `active_months_180d`, `mean_gap_days`, `revenue_180d`, `std_gap_days`, `tenure_days`.

**9. If the model loses to the business rule on recall.** Measured: "no order in 60 days" as a
coupon policy reaches lapser recall 0.56 / 0.61
(val / test) with 1,142 / 1,288 coupons and
$2,118 / −$12,182 profit. The model reaches
0.89 / 0.68 and loses less. If it lost on recall I
would compare at *equal coupons*: recall alone rises by contacting more people. If it still lost,
I would ship the rule and report the model as a negative result. The promotion gate already
requires +0.05 PR-AUC over the best baseline.

**10. Test protection.** One quarter after validation, never used for any choice. Scored
3 times, all disclosed: look 1 before the duplicate-key fix (0.8504),
look 2 after the re-run (identical), look 3 for the minimised 19-feature model (0.847).
`make evaluate` refuses a new look without `--force`, which records it.

**Recommendation**

**11. Association rules, not collaborative filtering.** 22.6% of lines have no customer,
baskets are the unit, the catalogue is 4,466 items at 0.54% density, and a rule is
explainable ("bought with"). I would switch to implicit-feedback CF or a two-tower model once most
traffic is signed in with a history and cross-session personalisation is the goal.

**12. The lift trap.** `22294 HEART FILIGREE DOVE SMALL` → `85123A WHITE HANGING HEART T-LIGHT HOLDER`: confidence 44.7%, lift
2.998. The consequent is in 14.9% of all baskets, so the rule mostly
restates popularity. `min_lift` 3 discards it; the median served lift is
7.9.

**13. Why not mine per request.** Apriori over 32,126 baskets takes 2.1 s
(FP-Growth in mlxtend 241.4 s) and needs the whole history. Mining is offline; serving is an
index lookup at 2.2 ms p95.

**14. 500,000 rules.** Measured with 480,768 rules (64× ours): `/recommend` p95
46 ms ms, 219 MB memory, because the engine indexes rules by antecedent
item and checks only matching ones. Beyond that I would prune by lift and support or move rules to
a key-value store.

**15. No rule matches.** The strategy becomes `popularity_fallback`, and the response marks each item `source: popularity` and the interface
labels it "Popular with our customers", never "bought with". Coverage: 91.7% of holdout baskets
fire a rule; hit-rate@5 0.145 against 0.017 for best-sellers
(8.6×).

**Generative AI**

**16. RAG, not fine-tuning.** Six policies (4,995 words, 66 chunks) that
change. RAG cites the exact chunk and updates by re-indexing. Fine-tuning would bake in stale
policies, cannot cite, and needs data and GPUs we do not have.

**17. No invented SKUs or probabilities.** Product codes and percentages come only from tool output
(`/recommend`, `/predict`). An LLM answer containing a SKU or a percentage absent from that output
is replaced by the deterministic template answer (`guardrails.py`, rule 5), and a test covers it.

**18. Is the assistant good?** On 40 golden and 20 paraphrased questions, tool choice is
97.5% / 95% with local `llama3.1:8b`, and out-of-scope refusal is 100%.
The keyword-rule provider scores 100% / **35%**, which is why an LLM earns
its place. The live service runs the rule provider (no GPU on App Runner): stated, not hidden.

**19. Retrieval returns nothing relevant.** Below a cosine similarity of 0.30 (chosen by
a floor sweep on the golden set), a policy question is handed to a human instead of answered. Tested
with out-of-corpus questions in the golden set.

**20. Why a local model.** Law 25 and the course rule: no personal data to a hosted provider. The
local model is the data-residency answer. It also costs $0.23 per 1,000 queries against
an estimated $0.19 hosted (unverified price), at 1.6 s per call.

**Engineering and MLOps**

**21. joblib or pickle.** joblib is pickle with efficient NumPy arrays. Both execute code on load,
so I load only artifacts this pipeline built, pinned to the library versions in `model_meta.json`.
XGBoost's JSON or ONNX would remove the risk for untrusted sources.

**22. A "model version".** The pipeline (preprocessing, estimator, calibrator), the threshold and
the feature list in one artifact, plus `model_meta.json`: git commit, data SHA-256, training rows,
library versions, MLflow run, registry version. Served now: `0.9.5+xgboost`.

**23. /health vs /ready.** Health says the process is alive (restart if not). Ready says every
artifact is loaded and warm (503 until then). A load balancer sends traffic only to *ready*
instances, so a deploy never serves half-loaded containers.

**24. OIDC, not keys.** One-hour credentials, a trust policy limited to this repository's immutable
IDs and the `production` environment, and nothing to rotate or leak. A stored key works from
anywhere until someone notices.

**25. Retraining opens a pull request.** A retrain is a code-and-data change. CI, the promotion
gates (baseline, calibration, champion, fairness, minimisation) and a human review apply before
deploy. Unattended promotion is how a silent regression ships.

**26. Degradation overnight.** Input PSI/KS, prediction drift, proposed-coupon rate, errors and
p95 are checked every 60 s, alerts fire on level changes, and `monitor.yml` opens a GitHub issue
hourly if the status is red or the service is down.

**27. A label 90 days late.** Meanwhile I monitor inputs, the prediction distribution, coupon rate
and data quality. A weekly backfill joins predictions to later orders. The replay showed PR-AUC
0.768 on 2,849 backfilled predictions.

**28. When managed training is worth it.** When tuning outgrows a laptop: hours not minutes, large
data or GPUs. Here XGBoost tuning took 1.6 min on 12,277 rows. The
SageMaker job is implemented and dry-run tested. Quota request: CASE_OPENED, case 179073448000753. Result:
not run: the quota request is still open, so every model was tuned locally.

**Governance**

**29. Automated decisions.** The coupon proposal (Law 25 s. 12.1). By design it is not exclusively
automated: a manager approves every list. We still disclose, give the principal factors on request
(`?explain=3`), allow observations to a person, and now also disclose the random hold-out.

**30. The recall gap.** The options are M1 expected value (profitable, abandons most lapsers),
M2 group thresholds (closes the gap, +20% coupons, treats by spend) and M3
re-weighting (halves it, fragile). Chosen: keep one threshold for the automatic decision and send
M2's extra cases to a human. Built in v1.0 (§3); the hold-out decides the rest.

**31. A hosted LLM outside Canada.** Not with customer data until a privacy impact assessment and
contractual protection exist (s. 17). Pseudonymise and minimise first; the local provider is the
default. Our own serving in `us-east-1` is the same issue, acceptable only because the data is
public and pseudonymised; real data means ECS in `ca-central-1`.

**32. "Why didn't I get an offer?"** Within 30 days (s. 32, to verify): the decision and the main
factors from `?explain=3`. If the customer was in the random hold-out, we say so; it is not about
them. They can submit observations to a staff member who can review the decision.

## 5. Negative results to volunteer, and the open limits

* Tuning added < 0.005 PR-AUC; the limit is the features, not the algorithm.
* The first profit model made "contact nobody" optimal until the value horizon was fixed.
* The model does not beat "contact nobody" on the test quarter (−$1,039). It beats
  the blanket coupon by $54,137.
* The fairness gap is routed to people, not solved; approving the whole review list loses money
  under the assumptions (§3).
* In mlxtend, Apriori is 116× faster than FP-Growth at this size.
* Open: save rate unmeasured (the hold-out now exists); serving outside Canada; the live log is
  per-instance SQLite; the hosted LLM provider is unevaluated; the SageMaker run waits on the quota;
  the LégisQuébec article numbers are to be checked by hand.
