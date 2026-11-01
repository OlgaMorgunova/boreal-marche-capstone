<div class="title" markdown="1">

# D5 — Optimization and Ethics Report
**Boréal Marché Customer Intelligence Platform** · Capstone Project 1

Olga Morgunova (individual project) · 420-977-VA AI Development Project · Instructor: Steve Ataky, Ph.D. · Vanier College, LEA.EU · Repository tag `v0.9.5` · KLO 5
</div>

## 1. Summary

Every number below is generated from an evidence file in `reports/metrics/` by
`python -m boreal_marche.reporting.build D5`; the scripts that produced them are named in each
section. What changed in the system for this deliverable:

| Area | Change | Evidence |
|---|---|---|
| Performance | column-wise model input, folded catalogue, cached monitoring report, warm-up before `/ready`, startup timings in `/ready` | §2, `latency_profile.json` |
| Interpretability | exact per-prediction factors, `POST /predict?explain=N`, shown in the interface | §5, `interpretability.json` |
| Data minimisation | **the model now uses {{ n_model }} of {{ n_all }} features**; the other {{ n_removed }} are no longer stored | §6, `minimisation.json` |
| Fairness | subgroup audit, four policies compared, a recommendation with its cost | §7, `fairness.json` |
| Governance | register, notice, retention and deletion schedule finalised; promotion gate gains a minimisation clause | §8, `GOVERNANCE_LOI25.md` |

Model `{{ model_version }}` replaces `0.3.0+xgboost`: test PR-AUC {{ test_pr_new }} {{ test_ci_new }}
against {{ test_pr_old }}, scored once after every choice was fixed (test look {{ test_looks }}, §6).

## 2. Performance optimisation

**Method.** `scripts/latency_profile.py` starts the real server (`uvicorn`, one worker, as in the
container), measures spawn → `/ready` = 200 three times (cold start), the first call to each
endpoint after start-up, then {{ n_warm }} sequential warm calls per endpoint over one keep-alive
connection; the API's own middleware timer (`request_log`) gives the server-side share. "Before" is
the unchanged v0.9 commit run from a clean `git worktree`; "after" is this tag. Machine:
{{ machine }}. I profiled the slow handlers with `cProfile` and per-stage timers before changing
anything.

{{ latency_table }}

*Table 1 — Latency by endpoint, ms (client side unless "server"). `/chat` server time includes the
assistant's internal HTTP call to `/predict`, so only client time is compared.*

![d5_latency](figures/d5_latency.png)
*Figure 1 — Warm p50 by endpoint before and after.*

**What I measured, what I changed, what it did.**

1. **`/predict` ({{ predict_before }} → {{ predict_after }} ms p50).** Of ~7 ms server time, the
   XGBoost call was 0.3 ms; building a one-row pandas frame from a dict and casting it took 2.2 ms
   and the sklearn preprocessing 1.8 ms. The frame is now built column by column with NumPy (0.3 ms);
   a test asserts the frame and the predictions are identical to the old path. Measured with the old
   model first ({{ predict_code }} ms), so the gain is the code, not the smaller model.
2. **`/products` ({{ products_before }} → {{ products_after }} ms).** Each search re-normalised the
   accents of all 4,724 descriptions. They are folded once at start-up.
3. **`/monitoring/report` ({{ report_before }} → {{ report_after }} ms).** The report was recomputed
   on every request while the dashboard polls it. A report younger than {{ cache_s }} s is served
   as is (`?fresh=true` recomputes; the alert loop is unchanged), and the KS test uses the
   asymptotic p-value — the exact method was the largest single cost and only the statistic is
   thresholded.
4. **Cold path.** One warm-up call per head now runs before `/ready` turns 200, so XGBoost, ONNX
   Runtime and pandas initialise before the first customer: first `/predict` {{ first_predict_before }}
   → {{ first_predict_after }} ms.

**Cold start versus warm, and loading models at start-up.** Cold start (spawn → ready) is
{{ cold_before }} locally and {{ container_v09 }} → {{ container_d5 }} for the container
(`docker run` → ready, image already pulled) — **unchanged, and I report that rather than hide it**.
Inside the container the model loads in {{ ct_model_ms }} and the embedding model in
{{ ct_assistant_ms }}; the rest is Python importing scikit-learn, SciPy, pandas and FastAPI (~2 s)
and container start. I loaded the two slow artifacts in parallel: it saves ~0.45 s when the model
comes from the MLflow registry (I/O-bound) and nothing measurable from the file, so the gain is
real only in the registry deployment. Loading **per request** instead of at start-up would cost,
measured in a warm process: {{ load_model_warm }} per prediction to unpickle the model ({{ lazy_factor }}
the warm request), {{ load_rules }} per recommendation for the rules, and {{ load_embed_warm }} per
assistant question to create the ONNX session ({{ lazy_chat_factor }} the warm request). In a fresh
process the first model load is {{ load_model_cold }} because it imports XGBoost. Start-up loading
is therefore decisive for the assistant and cheap insurance for Head A. The lever left for cold
start is not code: App Runner keeps one provisioned instance warm, so customers meet a cold start
only on deploy or scale-out, never after idling.

**Live service.** On App Runner (v0.9, called from Québec) `/health` takes {{ live_health }} ms and
`/predict` {{ live_predict }} ms p50: the ~30 ms floor is the network round trip, an order of
magnitude above any handler. The D4 finding (POSTs ~50 ms through Docker Desktop's Windows port
proxy) is a local artefact: inside the Docker network POSTs take milliseconds, and on App Runner a
POST costs ~10 ms more than a GET, which is the handler plus the request body, not a 45 ms stall.
The real latency lever for users is region (§8: `ca-central-1`), not code.

## 3. Scalability: where it breaks

`scripts/scalability_probe.py` grows one dimension at a time and measures it.

{{ tp_table }}

*Table 2 — One uvicorn worker on this PC, N concurrent clients, 8 s per cell. Saturation shows as
flat requests/s with rising p95.*

![d5_scalability](figures/d5_scalability.png)
*Figure 2 — Measured cost as rules, documents and the monitoring window grow (log-log; dashed: a
50 ms budget).*

| Dimension | Breaks at (estimate from the measurements) | What I would change first |
|---|---|---|
| **Traffic** | One worker saturates at ~{{ tp_predict }} single predictions/s here (p95 {{ p95_predict_16 }} ms at 16 clients); assuming an App Runner vCPU is half this core, ~{{ tp_predict_ar }}/s per instance. Batch scoring reaches {{ tp_batch_rows }} predictions/s. The SQLite log on the instance is the first hard limit: one writer, lost on redeploy, no second instance can share it. | Postgres for the log (already in the Compose stack), then App Runner autoscaling on concurrency; `/predict/batch` for any bulk caller |
| **Customers** | Scoring is not the limit: 100,000 customers in {{ score_100k }} ({{ score_rate }} rows/s). The limit is the offline feature build, which reads every transaction into pandas for each snapshot, and the monitoring window. | Incremental features in DuckDB/Polars per day, not a full rebuild per quarter |
| **Rules** | /recommend grows linearly with the rules indexed per item: {{ rules_mined }} rules → 0.25 ms p95; {{ rules_64_n }} (×64) → {{ rules_64_p95 }} and {{ rules_64_mem }} per worker. Lowering `min_support` one grid step multiplies rules ×2–2.6 (D3), so ~5 steps cross 50 ms. | Cap rules per antecedent item (top-k by lift) at mining time; past ~10⁵ rules, precompute top-5 per item set |
| **Documents** | Brute-force cosine search over {{ ret_current }} chunks is free; 100,000 chunks {{ ret_100k }}, 1,000,000 chunks {{ ret_1m }} and {{ ret_1m_mem }} of RAM per worker. | An approximate index (FAISS HNSW or pgvector) beyond ~10⁵ chunks |
| **Monitoring** | Report cost is linear in the window: {{ rep_500 }} at 500 predictions, {{ rep_10k }} at 10,000, {{ rep_50k }} at 50,000. Each logged prediction + request costs {{ log_bytes }} bytes: at a steady 1 prediction/s, {{ log_gb_year_1rps }} a year, beyond what the 13-month retention on an instance disk can hold. | Pre-aggregated hourly histograms (PSI needs only bin counts), sampling above 10 predictions/s |
| **Assistant (LLM)** | The local model answers one question per ~{{ ollama_ms }} on one GPU, so ~0.6 questions/s; ten simultaneous users already queue. | Batching server (vLLM) or the hosted provider for non-personal questions only |

## 4. Cost

**Total cloud spend:** {{ aws_total }} ({{ aws_period }}, all services, from Cost Explorer).

{{ aws_table }}

**Cost per 1,000 predictions.** App Runner bills {{ price_vcpu }}/vCPU-hour and {{ price_gb }}/GB-hour
while serving, and memory only ({{ ar_idle_h }}/hour for 2 GB) while idle; prices read on
2026-11-01. Always serving, the 1 vCPU / 2 GB instance costs {{ ar_active_month }}/month; idle-but-
provisioned {{ ar_idle_month }}. At full load (~{{ tp_predict_ar }} predictions/s per instance, the
{{ cpu_derate }}× assumption above) the marginal cost is {{ cost_1k_pred_marginal }} per 1,000
predictions; at realistic volumes the fixed idle memory dominates:

{{ cost_pred_table }}

*Table 3 — Steady traffic assumed; scaling out is triggered when an instance would be more than 60%
busy. The quarterly campaign itself (scoring all customers) costs less than a cent.*

**Cost per 1,000 assistant queries, by provider.** Tokens measured with the local model on the 40
golden questions: {{ tok_in }} prompt and {{ tok_out }} completion tokens per query on average
(`assistant_tokens.json`).

{{ assistant_cost_table }}

*Table 4 — Rule: CPU time from Table 2. Ollama: measured ~{{ ollama_ms }} per answer on a
g4dn.xlarge at {{ price_g4dn }}/hour (third-party price listing, the AWS page is dynamic), fully
used. OpenAI: list price {{ price_openai }} per million input/output tokens from `params.yaml`;
openai.com refused the automated price check, so this figure is **unverified**.*

The hosted model is the cheapest language model per query and the local one is the only one that
may receive personal information (course §8.5, Law 25 s. 17): the cost of data residency is a GPU
that costs {{ g4dn_month }}/month whether or not anyone asks a question. **Levers**, largest first:
pause the service between demonstration windows (runbook §4: $0 while paused); route only the
questions the rule provider cannot classify to a language model (it is free and correct on the
golden set); keep personal questions on the local model and send only policy questions to the
hosted one; ECR lifecycle rule keeping the last 5 images ($0.10/GB-month); SageMaker spot training
for HPO (quota pending, §11); and reserved or spot GPU capacity only if the assistant has real
traffic.

## 5. Interpretability

**Global.** Two measures on the validation snapshot (`retention/explain.py`): the mean absolute
TreeSHAP contribution, and permutation importance (drop in PR-AUC, 5 repeats).

![d5_importance](figures/d5_importance.png)
*Figure 3 — The five most influential features: {{ imp_top }}.*

`{{ imp_top1 }}` (how many of the last six months had an order) is the first factor for
{{ imp_top1_share }} of customers. The two measures disagree in an instructive way: permutation
drops are small everywhere (at most {{ max_perm }}) because the cadence and frequency features are
correlated — permute one and the others carry its signal. The model's importance is real, but it
is spread over a cluster of related habits, not owned by one feature.

**Per prediction.** XGBoost computes exact additive contributions in log-odds; the Platt
calibrator is affine in log-odds, so scaling by its slope gives contributions that sum to the
**calibrated** probability the API returns (maximum error {{ additivity }}). Indicator and one-hot
columns are folded back to the input feature. `POST /predict?explain=3` returns the three largest,
with the customer's value, a plain definition and a direction; the interface shows them under the
probability, and `POST /predict/batch?explain=3` produces them for every customer on a campaign
list, for the human reviewer.

{{ examples_table }}

*Table 5 — Three validation customers (baseline probability {{ base_p }}).*

**Checks I ran on the explanations.** Directions agree with domain expectations for
{{ dir_ok }} features (correlation of value with contribution: {{ dir_table }}). Under a ±5% change
of every numeric input, the top-3 set is identical for {{ stab_top3 }} of customers (mean Jaccard
{{ stab_jacc }}) and the first factor for {{ stab_top1 }}. **One correction:** `season_q` appeared in
{{ season_top3 }} of top-3 lists while its permutation importance is {{ season_perm }} — within one
quarter it is a constant, i.e. the season's baseline, not a reason about the person. Served
explanations now fold it into the baseline.

**Would a customer-facing explanation be honest?** Partly, and I would only show it on request.
*Honest:* the factors are faithful to the model (exact, not a surrogate), stable for most customers,
directionally sensible, and they are the "principal factors" s. 12.1 entitles a person to ask for.
*Not honest if presented naively:* (1) they explain the model's arithmetic, not the customer's
motives — "you received an offer because you ordered in one month of six" states a correlation the
customer may read as a judgement; (2) correlated features split credit, so for a near-threshold
customer (Table 5, second row) a tiny change reorders the factors and the "reason" is fragile;
(3) the threshold, not the factor, makes the decision, and the threshold depends on business
assumptions (save rate {{ save_rate }}) the customer cannot see. **Decision:** the factors are shown
to the reviewer always, to the customer on request in plain language with the sentence "this
describes how the model weighed your order history, not why you order", and never in marketing
copy.

## 6. Data-minimisation audit and the model change

The rule was written in GOVERNANCE §5 before the ablation: a feature group whose removal costs less
than 0.005 PR-AUC is not necessary. `retention/minimise.py` refits the champion's pipeline and
hyper-parameters without each group, on the same 5 customer-grouped folds as the D2 spot-check, and
on validation.

{{ minimisation_table }}

*Table 6 — Losses are full minus reduced PR-AUC; negative means the model improved without the
group.*

No group is necessary under the rule; removing **all {{ n_removed }} together** costs
{{ mini_full_cv }} → {{ mini_comb_cv }} in CV and {{ mini_full_val }} → {{ mini_comb_val }} on
validation. I removed them all. The model now reads {{ model_features }}. `region` is still
collected — it is needed to deliver and it is a fairness-audit dimension — but the model no longer
sees it; the other 17 fields are optional in the API, ignored, and **no longer written to the
prediction log** (a test asserts the stored keys).

**Promotion.** The written gate required a candidate to *beat* the champion by 0.005 PR-AUC, which
a minimised model cannot do. I added a minimisation clause — strictly fewer features and a loss of
at most 0.005 — to both the training and the weekly CI gate (`docs/MLFLOW_PROMOTION_POLICY.md`).
Against the served v0.9 artifact every gate passes (validation {{ gate_champ_pr }} →
{{ gate_cand_pr }}):

{{ gate_table }}

The local MLflow `champion` alias pointed at registry version {{ registry_prev }}, an older model
than the committed artifact the service actually ran; against it the candidate fell 0.0005 short of
the clause. Per the policy ("otherwise a human"), I moved the alias to version {{ registry_new }} and
recorded the reason in the registry tags and `model_meta.json`. **Consequences, reported:** the
test set was scored a third time, after the decision ({{ test_pr_old }} → {{ test_pr_new }}
{{ test_ci_new }}, ECE {{ ece_new }}); the profit-optimal threshold moved to {{ threshold_new }}; and
the test-quarter campaign result fell to {{ test_profit_new }} — still {{ test_vs_everyone_new }}
better than the current coupon-to-everyone policy. **A correction found on the way:** D3's test
policy figures used threshold {{ threshold_d3 }} from an earlier run (test profit {{ test_profit_d3 }}),
while the committed v0.9 artifact used 0.53; the model card now states the served value.

## 7. Bias and fairness audit

**Audit** (`retention/fairness.py`, validation snapshot, served model, threshold {{ threshold }}).
Lapser recall is the share of customers who did lapse who were proposed for the offer — the offer
is the benefit. Spend quartiles use edges fixed on the training snapshots ({{ spend_edges }} of
180-day spend), so any group policy can be served. A group is flagged when it is more than 5 points
below overall recall and has at least 30 lapsers.

{{ audit_table }}

*Table 7 — Subgroup audit of the served policy on the validation snapshot.*

**Flagged:** {{ flagged_val }} on validation; {{ flagged_test }} on test. Overall recall is
{{ overall_recall }} but only {{ top_recall }} in the top spend quartile: {{ missed_top }} top-spend
lapsers were not proposed, worth {{ missed_top_value }} of margin over the value horizon. The
worst gap is {{ worst_gap_val }} points on validation and {{ worst_gap_test }} on test (v0.9 model:
{{ v09_gap_val }} and {{ v09_gap_test }}; minimisation did not cause the gap and did not fix it).
**Why it happens:** high spenders and long-tenure customers rarely lapse, so the model gives them
high probabilities of returning; one value-blind threshold then reaches few of those who do lapse.
The model is not mis-calibrated for them (PR-AUC in the top quartile is among the highest) — the
decision rule is the cause. This is not the usual direction of harm (the advantaged group gets
*fewer* discounts), but equal opportunity among lapsers is the stated goal of the campaign, and the
missed value is the business's own loss.

**Mitigations** — chosen on validation, then each scored once on test (no choice used the test
set):

{{ mitigation_table }}

![d5_fairness](figures/d5_fairness.png)
*Figure 4 — Lapser recall per spend quartile under each policy, validation and test.*

| Mitigation | What it fixes | What it costs |
|---|---|---|
| **M1 expected value** — contact when expected campaign profit > 0 | Value-aware: highest profit on both snapshots; spend-quartile recall becomes even on test | Overall recall collapses (most lapsers are not worth a coupon at a {{ save_rate }} save rate); low spenders lose the offer; depends entirely on the unmeasured save rate |
| **M2 group thresholds** — one threshold per quartile ({{ m2_thresholds }}) | Equal opportunity: worst gap 7 points on validation | ~20% more coupons to high spenders who would have returned anyway; test profit far below M0; uses spend explicitly to set different treatment |
| **M3 re-weighting** — up-weight lapsers of under-served groups in training | Halves the top-quartile gap without touching the decision rule | Gap reduction shrinks on test, profit lower, a retrain is needed and the gain is fragile |

**Recommendation.** No mitigation dominates, because the real unknown is the coupon's effect per
group, not the model. (1) Keep the single threshold for the automatic proposal; (2) **add the extra
customers M2 would select (top-two quartiles between {{ threshold }} and their group threshold) as a
separate "high-value review" list for the marketing reviewer**, who already approves every list
(§8) — the human decides, with the factors in front of them, at no cost unless they choose to send;
(3) run a randomised hold-out (10% of proposed customers receive no coupon) for two quarters to
measure the save rate per spend quartile, then move to M1 if it is confirmed. The promotion gate
keeps "no worse than the champion"; it becomes the hard 5-point limit once the hold-out data exist.

## 8. Law 25 governance

The full file is `GOVERNANCE_LOI25.md`; its D5 state:

* **Automated-decision register** (§2): retention-offer targeting (D-A), recommendations (D-B),
  assistant answers and hand-off (D-C), each with what is decided, the personal information used,
  the principal factors (now served per prediction), whether a human reviews and how rights are
  exercised. D-A now lists the 19 features actually used.
* **Notice in both languages** (§4), rendered in the interface's *About* section, with the factors
  shown under every probability.
* **Design consequence** (§3): because a decision based exclusively on automated processing must be
  disclosed with its principal factors and reviewable by a staff member (s. 12.1), the API returns a
  *proposal* and a marketing manager approves every list; the weekly retraining opens a pull request
  and never deploys. D5 strengthens the review rather than replacing it (§7 recommendation).
* **Data-minimisation audit** (§5): finalised by the ablation of §6 — 18 features removed and no
  longer stored.
* **Retention and deletion** (§6): raw transactions 2 years; prediction log 13 months then
  aggregated; rows written by v0.9 still hold the removed features and age out under that 13-month
  purge (on App Runner the log is lost at each deploy anyway); model
  artifacts the last 5 registry versions; deletion requests honoured in the next ETL and model
  retrain, within the 30-day response limit.
* **Primary sources.** The CAI's published guidance was read on 2026-11-01 ("Vos droits" and
  "Principaux changements de la Loi 25"): notice no later than when the decision is communicated;
  on request the information used and "les raisons ainsi que les principaux facteurs et
  paramètres"; correction; observations to "un membre du personnel en mesure de réviser la
  décision". The design matches each point. The consolidated text of P-39.1 on LégisQuébec refused
  automated reading; the article references in the file are flagged for a manual check against it.
* **Data residency.** The demonstration API runs in `us-east-1` (App Runner is not offered in
  Canada) on public, pseudonymised data; real customers would require the ECS Fargate deployment in
  `ca-central-1` and a s. 17 assessment.

## 9. Model card, data card and monitoring policy

`MODEL_CARD.md` and `DATA_CARD.md` are complete for v0.9.5: intended and out-of-scope uses, the
19-feature model, test metrics after look 3, the served threshold, subgroup results with the flagged
gaps, interpretability limits and known failure modes. `docs/MONITORING.md` states the thresholds
(input drift PSI 0.10/0.25 and KS, prediction drift, data quality, error rate, p95 latency
150/500 ms, rule fire-rate), what each triggers, the alerting loop, and the retraining rule: retrain
when input drift is red 7 days, prediction drift red 3 days, realised PR-AUC falls 0.05 below
validation, or a quarter starts — then every promotion gate, including fairness and the new
minimisation clause, and a human merge. **Delayed ground truth:** the label exists 90 days after a
prediction; a weekly backfill joins logged predictions to later orders, so the first true accuracy
signal arrives 90 days + up to 7 days after the first prediction; until then the leading indicators
are drift and data quality. Next step, not yet built: run the subgroup audit of §7 on the
backfilled labels, which the hold-out experiment needs.

## 10. Limitations

Latency was measured on one desktop; the App Runner capacity uses a stated 0.5 derating. Cold
start was not improved. {{ aws_limit }}The OpenAI
price is unverified and the provider was never evaluated. The fairness audit uses a synthetic region
mapping and two quarterly snapshots; recall estimates in small groups are noisy. The coupon save
rate, on which every profit and M1 depend, has never been measured.

## 11. SageMaker status

The spot-instance quota request for `ml.m5.large` in `ca-central-1` (Service Quotas case
179073448000753, opened 2026-09-29) is still open on 2026-11-01. The HPO search therefore ran with the
local Optuna backend; the SageMaker launcher (`python scripts/run.py hpo --provider aws`) is
implemented, uses the same 19 features, and is tested in dry-run mode. When the quota is granted,
the run follows `docs/runbooks/SAGEMAKER_HPO.md` and D3 §11 is updated.

## 12. Sources and AI-tool use

CAI, "Vos droits" and "Principaux changements de la Loi 25" (cai.gouv.qc.ca, read 2026-11-01);
LégisQuébec P-39.1 (manual verification pending); AWS App Runner and ECR pricing pages (read
2026-11-01); XGBoost documentation (`pred_contribs`); Lundberg & Lee (2017) for SHAP; Hardt et al.
(2016) for equal opportunity. AI-assisted coding and drafting tools were used for code, tests and
this text; I reviewed and can explain each part. **Where AI-generated output had to be corrected:**
the first `season_q` handling in explanations (§5); a parallel start-up loader presented as a cold
start fix until measurement showed no gain on the production path (§2); and a promotion check
that compared against the registry alias instead of the served artifact (§6).
