<div class="title" markdown="1">

# Final Technical Report
**Boréal Marché Customer Intelligence Platform** · Capstone Project 1 · D6 (consolidates D2–D5)

Olga Morgunova (individual project) · 420-977-VA AI Development Project · Instructor: Steve Ataky, Ph.D. · Vanier College, LEA.EU · Repository tag `v0.9.6` · KLO 6, 7
</div>

## 1. Overview

Boréal Marché, a (fictional) Québec online grocer, asked three questions: *which active customers
are about to stop ordering, and is the quarterly coupon worth sending them?* (Head A, retention);
*what else belongs in this basket?* (Head B, recommendations); and *can the forty repetitive
customer-service questions be answered, in French or English, without inventing anything?* (Head C,
assistant). This report describes the one system that answers all three as it runs today, how each
design decision was made and tested, and what is still weak. It replaces D2–D5 as the reference
document: where a later deliverable changed an earlier result, only the current result is reported
here and the change is listed in §11.

**The system is live.** Interface: {{ d6_ui }} · API documentation: {{ d6_docs }} · deployed
commit `{{ d6_commit }}`, API {{ d6_api_version }}, model `{{ d6_model_version }}` (checked
{{ d6_checked }}, `/ready` = {{ d6_ready }}). Demonstration video: {{ video_url }}. Recorded D6
presentation (15 minutes, demonstration first): {{ d6_talk }}.

| Question | Result (held-out, temporal) | Baseline | § |
|---|---|---|---|
| Head A — who will order again within {{ label_days }} days | PR-AUC **{{ test_pr_auc }}** {{ test_pr_ci }}, ECE {{ test_ece }}, {{ n_model }} features | {{ test_best_base }} recency deciles · dummy {{ test_dummy }} | 5 |
| Head A — is the coupon worth it | test quarter: **{{ test_gain }}** better than coupon-to-everyone ({{ test_profit }} vs {{ test_everyone }}) | today's blanket coupon | 5.5 |
| Head B — basket recommendations | hit-rate@5 **{{ hold_hybrid }}**, {{ hold_cov }} of baskets served by a rule | {{ hold_pop }} best-sellers ({{ hold_ratio }}×) | 6 |
| Head C — assistant | tool choice {{ ollama_golden }} golden / {{ ollama_para }} paraphrases (local LLM), out-of-scope refused 100% | rule baseline {{ rule_golden }} / **{{ rule_para }}** | 7 |
| Platform | `/predict` warm p50 {{ predict_after }} ms (live {{ live_predict }} ms incl. network); image {{ img_api_mb }}; {{ d6_tests }} tests, coverage {{ d6_cov }} | D3 image {{ img_v03_mb }} | 8–9 |
| Cost | **{{ aws_total }}** total AWS spend ({{ aws_period }}) | course target < $25 | 9.3 |
| Fairness | lapser recall {{ top_recall }} in the top spend quartile vs {{ overall_recall }} overall — **flagged, not fixed** | — | 10.1 |

*Table 1 — Results. Every number in this report is generated from `reports/metrics/*.json` by
`python -m boreal_marche.reporting.build D6`; none is typed by hand.*

## 2. Problem, data and scope

**Business problem.** Today the client sends a {{ discount_rate }} coupon to every customer every
quarter. On the test quarter that policy loses **{{ test_everyone }}** under the stated cost
assumptions, mostly by discounting loyal customers who would have ordered anyway. The cross-sell
strip is hand-picked, and two employees answer the same questions by e-mail. Success was defined in
D1 as: beat the business rule on PR-AUC and express the gain in dollars; recommendations at least
2.5× better than best-sellers; an assistant that never invents a product or a probability and
refuses what it cannot ground; and a governed, monitored, reproducible deployment under $25.

**Data.** UCI Online Retail II (Chen, 2012, DOI 10.24432/C5CG6D, CC BY 4.0): {{ raw_rows }}
invoice lines, {{ date_min }} to {{ date_max }}, real UK giftware transactions **re-skinned** as
Boréal Marché (regions and categories mapped, currency shown as $). It is public, pseudonymised data,
not client data, and the conclusions about grocery behaviour are bounded by that (§12). The raw file
is DVC-tracked and verified by SHA-256 (`{{ sha_short }}…`); it is never committed.

## 3. Architecture

![d6_architecture](figures/d6_architecture.png)
*Figure 1 — The system as built and deployed. Only artifacts cross from the offline pipeline into
the online service; the algorithms never run on a request.*

**Walkthrough.** `make data` downloads and checksums the workbook into **bronze** Parquet; `make
etl` applies eleven ordered cleaning rules into **silver**; `make features` builds **gold**, one row
per (customer, cut-off) with {{ n_all }} point-in-time features (the model uses {{ n_model }}, §5.4)
and the {{ label_days }}-day label. Head A is selected, tuned, calibrated and thresholded offline and
registered in MLflow as `champion`; Head B mines rules weekly; Head C embeds a {{ corpus_docs }}-document
policy corpus into {{ n_chunks }} chunks. Every artifact is baked into one API image, so a deployed
version is one image digest. A pull request runs CI; a merge runs CD, which authenticates to AWS
with a short-lived OIDC token, pushes the image tagged with the commit SHA to ECR, deploys it to App
Runner and smoke-tests the live service. The interface (Streamlit Community Cloud) holds no data and
calls the API over HTTPS. Every prediction is logged, and the monitoring endpoints compute drift,
data quality and latency from that log.

**Why this shape.** Everything expensive — cleaning, mining ({{ b_fp_s }} s), tuning — is offline;
everything on the request path is a lookup in a loaded artifact, which is why `/predict` takes
milliseconds (§9.1). Preprocessing lives inside the model artifact so training and serving cannot
drift apart. Rules and the vector index are files with provenance headers, so the API can say which
rule fired and which document it cited.

## 4. Data pipeline and leakage control

{{ cleaning_table }}

*Table 2 — Cleaning rules in execution order (`reports/cleaning_report.md`).*

Bronze → silver keeps {{ silver_lines }} of {{ raw_rows }} lines ({{ removed_pct }} removed). The
largest single correction is **R1, the {{ overlap_rows }}-line overlap between the two yearly
sheets**: the course's starter script concatenated them and would have double-counted every order
of 1–9 December 2010. The key design decision is **R3: cancellations are separated, not netted** —
netting deletes a purchase made before a cut-off because of a cancellation after it, i.e. future
information in past features. A data-quality suite guards silver and gold in CI; it caught a bug of
mine in R2 (the duplicate key included the description, so two spellings of one product survived)
which is now covered by a unit test.

**Windows.** For a cut-off T, features see only orders before T (observation window
{{ obs_days }} days, plus lifetime history); the label is "at least one order in [T, T + {{ label_days }} d)".
Ninety days matches the client's quarterly campaign and the ~two-month median gap between orders.

{{ snapshot_table }}

*Table 3 — Snapshots. Train {{ train_cutoffs }}; validation {{ val_cutoff }} (calibration and
threshold); test {{ test_cutoff }}. Label windows never overlap. The base rate moves with the
season, which is why calibration uses its own snapshot.*

**Leakage was tested on purpose** (validation snapshot, untuned model): computing features through
T + 90 days lifts PR-AUC from {{ leak_clean }} to **{{ leak_leaky }}** — a perfect score, because
recency below 90 days *is* the label. Row-level versus customer-grouped folds differ by only
{{ split_row }} vs {{ split_group }}; grouping is kept as the safe default. Unit tests assert that
adding orders after T changes no feature, and `check_gold` rejects any feature row that saw the
future. Appendix B of the course warns that 0.95 means leakage; the 1.000 above is that warning,
reproduced and fenced off.

## 5. Head A — retention

### 5.1 Evaluation design and model selection

The primary metric is **PR-AUC** of the "returns" class: accuracy is meaningless under a base rate
that moves with the season (Table 3), and ROC-AUC is insensitive to the top of the ranking where money is
spent. Brier and ECE measure the probability, because it is multiplied by dollars. Three baselines
are scored identically: a dummy prior, the client's rule "ordered in the last 60 days", and a
recency-decile heuristic. Ten model families were compared on **identical** preprocessing and
{{ n_folds }} customer-grouped folds (selection was done with all {{ n_all }} features, before the
§5.4 audit):

{{ spot_table }}

*Table 4 — Spot-check ({{ spot_rows }} training rows, cross-validation).*

Every learned model beats every baseline by at least {{ spot_margin }} PR-AUC, and the top is flat: {{ spot_within_sd }}
families lie within one standard deviation ({{ spot_top_sd }}) of the best, logistic regression
included. Model choice matters less than features and decision design. Extra trees and XGBoost were
tuned with Optuna (TPE, median pruning, {{ trials_per_model }} trials each): the gain was
**{{ tune_xgboost_impr }}** for XGBoost and {{ tune_extra_trees_impr }} for extra trees — less than
one fold standard deviation. I report this as a finding, not a failure: the problem is limited by
the information in the features, not by hyper-parameters. XGBoost won on validation
({{ val_pr_auc }} vs {{ challenger_val_pr_auc }}).

### 5.2 Calibration

{{ calibration_table }}

*Table 5 — 5-fold out-of-fold calibration on the validation snapshot.*

Sigmoid (Platt) scaling was chosen: lowest Brier, ECE {{ ece_raw }} → **{{ ece_cal }}**, monotone and
smooth at the threshold (isotonic produces ties with ~2.7k rows). On the test quarter ECE is
{{ test_ece }}, so calibration survives the seasonal shift.

### 5.3 From a probability to a decision — the hard decision

A probability is not a decision. The coupon goes to customers whose calibrated P(return) is **at or
below** a threshold. Per contacted customer the expected profit is `save_rate × (value − discount) −
coupon` for a lapser and `−(coupon + discount)` for a customer who would have returned anyway, with
{{ coupon_cost }} handling, a {{ discount_rate }} discount, a stated save rate of {{ save_rate }} and a
{{ horizon }}-day value horizon at {{ gross_margin }} margin (`params.yaml → business`).

![profit](figures/profit_curve_validation.png)
*Figure 2 — Validation campaign profit by threshold for three save rates; threshold 1.0 is today's
coupon-to-everyone.*

The profit-maximising threshold on validation is **{{ threshold }}**: it contacts
{{ val_contacted }} of {{ val_n }} customers, reaches {{ val_lapsers_hit }} of {{ val_lapsers }}
lapsers and earns {{ val_profit }}, against {{ val_everyone }} for coupon-to-everyone. The model's
first value is **not sending** {{ val_not_sent_pct }} of coupons.

{{ sensitivity_table }}

*Table 6 — Sensitivity. At a 10% save rate no campaign pays; the dollar figures rest on an
assumption that has never been measured.*

**Negative result.** I first valued a saved customer at one 90-day window of margin. Every
threshold then lost money and the optimum was "contact nobody" — the 15% discount is as large as one
quarter's margin of a small customer. The assumption, not the model, was wrong: saving a lapser
restores a relationship. It was replaced by a stated 365-day horizon, and both versions are kept in
`train_final.json`. The deeper lesson, which drives §10.1, is that the threshold — not the model —
decides who gets the offer, and one threshold treats a $50 customer and a $5,000 customer alike.

### 5.4 Data minimisation and the served model

Law 25 requires that only necessary information be collected. The rule was written before the test:
a feature group whose removal costs less than 0.005 PR-AUC is not necessary. Each group was removed
in turn on the same folds:

{{ minimisation_table }}

*Table 7 — Minimisation ablation (loss = full minus reduced PR-AUC).*

No group was necessary; removing all {{ n_removed }} together cost {{ mini_full_cv }} →
{{ mini_comb_cv }} in CV. The served model `{{ model_version }}` reads **{{ n_model }} features**
({{ model_features }}); the other fields are no longer stored by the API. Because the promotion gate
required beating the champion, I added a minimisation clause (strictly fewer features, loss ≤ 0.005)
and every gate passed:

{{ gate_table }}

*Table 8 — Promotion gates for the served model (`docs/MLFLOW_PROMOTION_POLICY.md`).*

### 5.5 Held-out test results

Test snapshot {{ test_cutoff }}: {{ test_n }} customers, base rate {{ test_pos }}. The test set has
been scored **{{ test_looks }} times**, each after every choice was fixed: look 1 before the R2
duplicate fix ({{ look1_pr_auc }}), look 2 after the re-run (identical), look 3 for the minimised
model. No decision in this report used test data.

{{ test_table }}

*Table 9 — Test metrics with 95% bootstrap intervals.*

The model beats the best heuristic by **{{ test_gain_vs_base }} PR-AUC** with non-overlapping
intervals. PR-AUC of the lapse class, the class the campaign acts on, is {{ test_pr_lapse }}.

![d6_campaign](figures/d6_campaign.png)
*Figure 3 — One quarterly campaign by policy on the two held-out quarters.*

On the test quarter the validation threshold contacts {{ test_contacted }} customers and roughly
breaks even ({{ test_profit }}), against {{ test_everyone }} for coupon-to-everyone:
**{{ test_gain }} better than the status quo per quarter** (validation quarter: {{ val_gain }}).
Annualised over four campaigns that is **{{ annual_low }}–{{ annual_high }} a year**, under the
stated assumptions. It does **not** beat "contact nobody" on the test quarter: with a {{ test_pos }} base rate
few customers are worth a coupon. The per-customer expected-value policy stays positive
({{ test_ev_profit }}) because it adapts to each probability — but it reaches far fewer lapsers
(§10.1).

### 5.6 Where the model fails, and why it decides what it decides

{{ segment_table }}

*Table 10 — Test performance by segment (n ≥ 50).*

Ranking collapses for **one-order customers** (PR-AUC {{ seg_one_pr }}, n = {{ seg_one_n }}) and the
lowest spend quartile ({{ seg_q1_pr }}): with one purchase there is no cadence to learn from. The
decision fails differently: lapser recall is {{ seg_q1_recall }} in the lowest spend quartile but
**{{ seg_q4_recall }} in the highest** (§10.1).

![d5_importance](figures/d5_importance.png)
*Figure 4 — Global importance on validation: mean |TreeSHAP| and permutation drop.*

The first factor is `{{ imp_top1 }}` (months with an order out of the last six) for {{ imp_top1_share }}
of customers. Permutation drops are small everywhere (≤ {{ max_perm }}) because cadence and
frequency features are correlated: importance is shared by a cluster of habits. Each prediction can
return its exact factors (`POST /predict?explain=3`): XGBoost contributions in log-odds, scaled by
the Platt slope so they add up to the calibrated probability (error {{ additivity }}). Directions
agree with domain expectations for {{ dir_ok }} audited features, and the top-3 set is stable under
±5% input noise for {{ stab_top3 }} of customers. The factors are shown to the reviewer always and to
a customer only on request, with the sentence "this describes how the model weighed your order
history, not why you order".

## 6. Head B — basket recommendations

{{ b_baskets }} invoices × {{ b_items }} items at density {{ b_density }}: a sparse matrix takes
{{ b_sparse_mb }} MB instead of {{ b_dense_mb }} MB. Rules were preferred over collaborative
filtering because the question is basket-conditional, {{ guest_pct }} of lines are guests with no
history, and every suggestion must say *why*. Parameters were swept on a slice inside the mining
period; the chosen point (support {{ b_support }}, i.e. ≥ {{ b_support_invoices }} invoices,
confidence {{ b_conf }}, lift ≥ {{ b_lift }}) gives **{{ b_rules }} rules**. An honest surprise: in
mlxtend, FP-Growth is {{ b_speedup }}× *slower* than Apriori at this size (pure-Python recursion vs
vectorised NumPy); FP-Growth stays the production miner because it is offline and identical
itemsets are asserted in code.

**The lift trap, from this data.** {{ trap_ante }} → {{ trap_cons }} has confidence {{ trap_conf }}
but lift {{ trap_lift }}: the consequent is the best-seller, in {{ trap_cons_support }} of all
baskets, so the rule only says "people who buy anything buy the best-seller". The lift floor
removes it.

| Strategy (temporal holdout, {{ hold_cases }} baskets) | Hit-rate@5 | Basket coverage |
|---|---:|---:|
| Rules + popularity fill (served) | **{{ hold_hybrid }}** | {{ hold_cov }} |
| Global popularity baseline | {{ hold_pop }} | 100% |

*Table 11 — {{ hold_ratio }}× the baseline (course "excellent": ≥ 2.5× with coverage > 40%).*

At serving time rules are indexed by antecedent item; the response names the rule that fired, and
an unmatched basket returns `popularity_fallback` with the unknown items listed, so the interface
never presents a best-seller as a personal suggestion. Example: [{{ example_basket }}] →
{{ example_recs }}.

## 7. Head C — grounded bilingual assistant

Retrieval-augmented generation was chosen over fine-tuning because policies change and every answer
must cite a source. {{ corpus_docs }} documents ({{ corpus_words }} words, EN + FR) are chunked into
{{ n_chunks }} chunks and embedded with a multilingual model, so a French question retrieves an
English chunk. The language model is the **interface, not the source of truth**: four typed tools —
`policy_search`, `retention_score` and `recommendations` (which call this API's own endpoints) and
`product_lookup` — return validated objects. Guardrails run around every provider: another
customer's data is refused before any model runs; a policy answer below similarity {{ sim_floor }}
is handed to a human; an LLM answer mentioning a SKU or percentage absent from the tool output is
replaced by the verified template.

{{ assistant_table }}

*Table 12 — Assistant evaluation on a 40-question golden set written before the assistant and 20
paraphrases written after the rule provider.*

The rule baseline scores {{ rule_golden }} on the golden set and {{ rule_para }} on paraphrases: its
keywords were written by someone who had read the golden questions. The local model (Ollama,
llama3.1:8b) holds up at {{ ollama_golden }} / {{ ollama_para }}. That gap is the measured case for a
language model. The hosted provider is implemented but **not evaluated** (no API key; course §8.5
also forbids sending personal data to it). The live service runs the rule provider, because an 8B
model needs a GPU instance ({{ g4dn_month }}/month); the local model runs in the Compose stack.

## 8. Deployment and MLOps

### 8.1 Containers, CI and CD

The API image is multi-stage, pinned by digest, runs as non-root UID 10001, has a `HEALTHCHECK` on
`/ready`, and installs a serving-only lock (`xgboost-cpu`, no training stack): **{{ img_api_mb }}**, down from
{{ img_v03_mb }} at D3, while also carrying the assistant and its embedding model.

CI (`.github/workflows/ci.yml`) runs on every pull request: ruff, mypy `--strict`, {{ d6_tests }}
tests with a 70% coverage gate (currently {{ d6_cov }}, including the API contract and a
golden-prediction regression test), the data-quality suite, the assistant tool-choice evaluation,
both image builds with their sizes, a container smoke test, and gitleaks over the full history.
CD (`cd.yml`) runs on merge: GitHub OIDC → role `BorealGithubDeploy` (one-hour credentials, trust
limited to this repository's immutable IDs and the `production` environment) → image tagged with the
commit SHA → ECR → App Runner → `scripts/smoke_test.py`, which fails the run unless `/version`
equals the merged commit and the golden prediction is reproduced. **No AWS key exists in GitHub.**
The OIDC set-up failed twice, and each failure was debugged from CloudTrail evidence (the subject
claim carries immutable IDs; a `PassRole` condition App Runner does not send).

### 8.2 Hosting decisions and their cost

The API runs on AWS App Runner (1 vCPU / 2 GB, {{ container_d5_mem }} at idle). The interface could not follow:
App Runner's proxy rejected Streamlit's WebSocket upgrade (a `403` from `envoy` that never reached
the container), so it runs on Streamlit Community Cloud. **Region deviation:** App Runner is not
offered in `ca-central-1`, so serving runs in `us-east-1`. That is acceptable for public,
pseudonymised data; for real customers it would be a transfer outside Québec (Law 25 s. 17), and the
same image would run on ECS Fargate in `ca-central-1` (~$30/month more, mostly the load balancer).

### 8.3 Monitoring and retraining

A Locust traffic simulation (adapted from the course lab) replays the held-out customers, then a
deliberately shifted cohort (recent Maritimes sign-ups, short histories, unseen SKUs):

{{ monitor_table }}

*Table 13 — Traffic lights after each phase (`docs/MONITORING.md` states the thresholds).*

Under the shifted cohort {{ n_red_features }} of features cross PSI 0.25, {{ unknown_rate }} of basket
items are unknown and the rule fire rate falls from {{ fire_ref }} to {{ fire_shifted }}; each level
change raised an alert ({{ n_alerts }} in total). **A correction found by monitoring:** with the
pooled training data as the reference, healthy traffic was permanently red, because later snapshots
have longer histories by construction; a monitor that is always red is ignored, so the reference is
now the calibration snapshot. **Delayed ground truth:** the label exists 90 days after a prediction;
a weekly backfill joins predictions to later orders, and until then drift and data quality are the
leading indicators. The weekly retraining workflow re-runs the pipeline and the promotion gates
(including fairness and minimisation) and **opens a pull request — it never deploys**; a human
merges.

### 8.4 Managed training (SageMaker)

{{ sm_section }}

**When managed training would pay off.** One local trial takes ~2 s (5 folds × 12k rows); a
SageMaker job spends ~3 minutes starting its container. Managed training wins when trials take
tens of minutes (≈ 100× the data, or deep models), need more memory than a laptop, must run
unattended, or the data may not leave the cloud account. None applied here.

## 9. Optimisation

### 9.1 Latency

{{ latency_table }}

*Table 14 — Latency by endpoint, ms, before (v0.9) and after the D5 changes, same machine.*

Each change followed a measurement: building the one-row model input column-wise instead of through
pandas ({{ predict_before }} → {{ predict_after }} ms p50 for `/predict`), folding catalogue
accents once at start-up, caching the monitoring report for {{ cache_s }} s ({{ report_before }} →
{{ report_after }} ms), and a warm-up before `/ready` turns 200. **Cold start was not improved**
({{ container_d5 }} for the container): most of it is Python importing scientific libraries. On App
Runner the network round trip (`/health` alone {{ live_health }} ms) dominates every handler; the real latency lever for users is
region, not code.

### 9.2 Scalability

One worker saturates at ~{{ tp_predict }} predictions/s locally (~{{ tp_predict_ar }}/s assumed on an
App Runner vCPU); batch scoring reaches {{ tp_batch_rows }} rows/s. The first hard limit is the
per-instance SQLite prediction log (one writer, lost on redeploy); Postgres is already in the
Compose stack. Rules scale linearly ({{ rules_64_n }} rules → {{ rules_64_p95 }} p95), retrieval is
brute force and free at {{ ret_current }} chunks ({{ ret_1m }} at a million), and the monitoring
report grows with its window ({{ rep_50k }} at 50,000 predictions) until it is pre-aggregated.

### 9.3 Cost

{{ aws_table }}

*Table 15 — Total AWS spend {{ aws_total }} ({{ aws_period }}, Cost Explorer). A $10 budget alarm
has been active since Week 1.*

The service costs {{ ar_idle_month }}/month while provisioned and idle and is paused between grading
windows. At realistic volumes the fixed memory charge dominates: about {{ cost_1k_pred_marginal }}
of compute per 1,000 predictions at full load. Assistant queries cost {{ ollama_1k }} per 1,000 on a
dedicated GPU (local model, the only one allowed to see personal data) or an unverified
{{ openai_1k }} with the hosted provider.

## 10. Responsible AI and Law 25

### 10.1 Fairness: the gap I found and what I recommend

Lapser recall — the share of customers who did lapse who were proposed for the offer — is
{{ overall_recall }} overall but **{{ top_recall }} in the top spend quartile** (validation; flagged
groups: {{ flagged_val }}). {{ missed_top }} top-spend lapsers were not proposed, worth
{{ missed_top_value }} of margin. The model is not wrong about them (its PR-AUC there is among the
highest); the **single threshold** is: high spenders rarely lapse, so their probabilities are high
and one value-blind cut-off reaches few of those who do. Removing {{ n_removed }} features neither caused nor
fixed the gap (v0.9 gap {{ v09_gap_val }} points).

{{ mitigation_table }}

*Table 16 — Four policies, chosen on validation and scored once on test.*

![d5_fairness](figures/d5_fairness.png)
*Figure 5 — Lapser recall by spend quartile under each policy.*

No mitigation dominates: expected value (M1) is most profitable but abandons most lapsers; group
thresholds (M2) close the gap but send {{ m2_extra_pct }} more coupons to high spenders who would have returned
and set different treatment by spend; re-weighting (M3) halves the gap but the gain shrinks on test.
**Recommendation:** keep the single threshold for the automatic proposal; add the customers M2
would add as a separate **high-value review list** for the marketing reviewer, who already approves
every list; and run a **randomised 10% hold-out** for two quarters to measure the save rate per
quartile — the unknown on which every dollar figure and every mitigation depends. Neither the
review list nor the hold-out is built yet (§13).

### 10.2 Law 25 governance

`GOVERNANCE_LOI25.md` holds the register of automated decisions (D-A coupon targeting, D-B
recommendations, D-C assistant answers and hand-off), the customer notice in French and English
(shown in the interface), the data-minimisation audit (§5.4), and the retention schedule (raw
transactions 2 years, prediction log 13 months then aggregated, last 5 model versions; deletion
requests honoured within 30 days). **Design consequence of s. 12.1:** a decision based exclusively
on automated processing must be disclosed with its principal factors and be reviewable by a staff
member. The API therefore returns a *proposal* and a person approves every campaign list, so the
decision is not exclusively automated, and the factors are available for each customer. The CAI's
published guidance was read on 2026-11-01; `MODEL_CARD.md` and `DATA_CARD.md` record intended and
out-of-scope uses, subgroup results and known failure modes.

## 11. What changed since the earlier submissions

| Earlier document | What it said | What is true now, and where |
|---|---|---|
| D2 architecture | design figure with Head C, monitoring and interface "not yet built" | all built and deployed: Fig. 1 replaces it (§3) |
| D3 §7–8 | threshold 0.56 from an earlier run; test policy figures for that run | the served artifact's threshold is reported; with the 19-feature model it is {{ threshold }}, and every policy figure here uses it (§5.3, §5.5) |
| D3 §8 | 37-feature model, test PR-AUC {{ test_pr_old }} | 19-feature model {{ test_pr_new }} {{ test_ci_new }}, third test look disclosed (§5.4–5.5) |
| D3 §11 | SageMaker as a "control point" with planned numbers | status stated plainly; no managed-run number is claimed (§8.4) |
| D3 §14 | Head C not built; image 2.2 GB | Head C built and evaluated (§7); image {{ img_api_mb }} (§8.1) |
| D4 §1 | live URLs, CD run and video PENDING | live and checked {{ d6_checked }} (§1) |
| D4 §7 | `/predict` p95 ~50 ms through Docker | traced to Docker Desktop's port proxy on Windows; {{ predict_after }} ms warm p50 locally, network-bound live (§9.1) |
| D4 §7 | drift reference = pooled training data | calibration snapshot; always-red monitor corrected (§8.3) |
| D2/D3 | Law 25 articles "to be verified" | CAI guidance read and cited; design checked against it (§10.2) |
| D3 §9, D4 §5 | spend-quartile recall gap noted | audited, four mitigations scored, recommendation made (§10.1) |

## 12. Limitations

* **Propensity, not uplift.** The model predicts who returns, not who returns *because of* the
  coupon. Every dollar figure rests on a save rate of {{ save_rate }} that has never been measured;
  at 10% no campaign pays.
* **One test quarter**, at a higher base rate than training; seasonality is seen once.
* **Public, re-skinned data.** Regions are a synthetic mapping, so regional fairness results
  describe the mapping; the catalogue is giftware bought largely by businesses.
* **Weak for new customers.** One-order customers are close to coin flips (§5.6).
* **Fairness gap open** in the top spend quartile; the recommended review list and hold-out are not
  built.
* **Operations.** The live prediction log is per-instance SQLite and lost on redeploy; the live
  assistant runs the rule provider ({{ rule_para }} on paraphrases); the hosted LLM provider is unevaluated;
  serving runs outside Canada; cold start {{ container_d5 }}; the SageMaker job has not run (quota).
* **Legal check.** Article numbers in the governance file follow the CAI's guidance; the consolidated
  text on LégisQuébec refused automated reading and is checked by hand.

## 13. What I would do next

1. **Measure the coupon's effect**: a randomised 10% hold-out for two quarters, then an uplift model
   if the effect differs by customer. This is the single largest source of uncertainty.
2. **Build the high-value review list** in the interface and the backfilled subgroup audit, then
   decide between M0 and M1 on measured save rates.
3. **Move serving to `ca-central-1`** (ECS Fargate) and the prediction log to Postgres (RDS) before
   any real customer data.
4. **Run the SageMaker HPO job** as soon as the quota is granted, and report its cost and score
   against local Optuna.
5. **Onboarding rule for one-order customers** instead of a model that cannot rank them.

## 14. Reproducibility

```
git clone https://github.com/OlgaMorgunova/boreal-marche-capstone.git && cd boreal-marche-capstone
docker compose -f docker/compose.yml up --build -d    # API :8000/docs, UI :8501, Postgres, Ollama
python3 scripts/smoke_test.py --api http://127.0.0.1:8000 --ui http://127.0.0.1:8501
make setup-pinned && make all && make test            # every artifact from the raw data (~15 min)
python -m boreal_marche.reporting.build D6 D6-ES D6-DECK
```

Windows, step by step from an empty machine: `docs/RUN_FROM_ZERO_WINDOWS.md`. Every parameter
(windows, cut-offs, costs, grids, seed {{ seed }}) is in `params.yaml`; every run is tracked in
MLflow with its git commit and data checksum. Repository tags: {{ d6_tags }}.

## 15. Sources and AI-tool use

Chen (2012), UCI Online Retail II, CC BY 4.0. Commission d'accès à l'information du Québec, "Vos
droits" and "Principaux changements de la Loi 25" (read 2026-11-01); Act respecting the protection of
personal information in the private sector, CQLR c. P-39.1. Huyen, *Designing Machine Learning
Systems*; Lundberg & Lee (2017) for SHAP; Hardt et al. (2016) for equal opportunity. Documentation of
scikit-learn, XGBoost, Optuna, MLflow, mlxtend, FastAPI, AWS App Runner, ECR and GitHub OIDC. Course
Locust lab (instructor material).

AI-assisted coding and drafting tools were used for code, tests, the policy corpus and the text of
every deliverable; I reviewed each module and can explain every line. **Where AI-generated output
had to be corrected**, each found by a test or a measurement: the 90-day value horizon (§5.3); the
duplicate key including descriptions (§4); `mean_gap_days` computed between invoices rather than
order days (`overdue_ratio` reached 253,881); the pooled drift reference (§8.3); a test that hid the
SKU guardrail by stripping spaces from streamed tokens; a Docker clean-up step that deleted
`numpy/_core/tests` and crashed the container; `season_q` presented as a reason in explanations; a
parallel loader presented as a cold-start fix until measurement showed no gain; and a promotion check
that compared against the registry alias instead of the served artifact.
