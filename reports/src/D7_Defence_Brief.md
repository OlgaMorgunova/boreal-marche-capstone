<div class="title" markdown="1">

# D7 Defence Brief — Complete System, Readiness and Prepared Answers

Olga Morgunova · 420-977-VA AI Development Project (Capstone 1) · Learning Integration Assessment ·
release `v1.0` · instructor: Steve Ataky, Ph.D. · evidence collected {{ d7_checked }}

</div>

This brief is the panel's map of the system and my preparation for the defence. The fifteen-minute
talk follows the D6 deck, demonstration first. Every number below is rendered from
`reports/metrics/*.json` by `python -m boreal_marche.reporting.build D7`. The readiness table
comes from `scripts/collect_d7_evidence.py`, run against the live service.

## 1. The system the panel can use

| Component | Where | State at collection |
|---|---|---|
| Interface (EN/FR) | {{ live_ui_url }} | sleeps when unused; the first visit wakes it in ~30 s |
| API and its documentation | {{ live_api_docs }} | API `{{ d7_api_version }}`, commit `{{ d7_commit }}` (head of `master`: `{{ d7_master }}`) |
| Monitoring dashboard | interface → *Monitoring* tab; `GET /monitoring/report`, `/metrics` | in-process checks every 60 s; hourly external check (`monitor.yml`) |
| Source, release | `github.com/OlgaMorgunova/boreal-marche-capstone`, tag `v1.0` | tags {{ d7_tags }} |
| Tests | `make cov` | {{ d7_tests }} passed, {{ d7_cov }} line coverage |
| Recorded defence presentation | {{ d7_defence_video }} | "{{ d7_defence_video_title }}" (YouTube) |
| Fallback video | https://youtu.be/C_Qv7lu3cpY (walkthrough) · {{ d6_talk }} (D6 talk) | in case the network fails |

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

{{ d7_checklist }}

*Table 1 — {{ d7_n_pass }} automated checks pass, {{ d7_n_fail }} fail, {{ d7_n_pending }} are pending
and {{ d7_n_manual }} need a person. Re-run the morning of the defence; a failure is fixed before
the defence, not explained during it.*

## 3. What v1.0 adds: the fairness mitigation is now served

D5 found that one value-blind threshold proposes only {{ top_recall }} of lapsing top-quartile
spenders against {{ overall_recall }} overall. It adopted, without building, a human review list and
a randomised hold-out. Both now run in the API (`retention/campaign.py`). Every `/predict` returns an
`action`. `GET /campaign` returns the list the reviewer approves. The automatic proposal is
unchanged, so the golden predictions still hold.

* **review**: spend Q3 or Q4, not proposed, but under the M2 group threshold (Q3 ≤ {{ camp_q3_th }},
  Q4 ≤ {{ camp_q4_th }}). Spend routes the case to a person; it never changes the automatic
  decision.
* **holdout**: {{ camp_holdout_share }} of proposed customers, drawn by
  `sha256(campaign_id:customer_ref)`. The draw is stateless, auditable and blind to every feature,
  and a new `campaign_id` each quarter draws a new sample. The customer notice now says so.

| Snapshot (served model) | Validation | Test |
|---|---:|---:|
| Proposed / review list / hold-out | {{ camp_proposed_val }} / {{ camp_review_val }} / {{ camp_hold_val }} | {{ camp_proposed_test }} / {{ camp_review_test }} / {{ camp_hold_test }} |
| Q4 lapser recall: automatic → if the whole list is approved | {{ camp_q4_auto_val }} → **{{ camp_q4_review_val }}** | {{ camp_q4_auto_test }} → **{{ camp_q4_review_test }}** |
| Q3 lapser recall: automatic → if approved | {{ camp_q3_auto_val }} → {{ camp_q3_review_val }} | {{ camp_q3_auto_test }} → {{ camp_q3_review_test }} |
| Lapsers in the review list | {{ camp_review_lapsers_val }} | {{ camp_review_lapsers_test }} |
| Profit if the whole list is approved (assumed save rate {{ save_rate }}) | {{ camp_review_profit_val }} | {{ camp_review_profit_test }} |
| Hold-out balance: mean `p_return` held out vs treated | {{ camp_p_hold_val }} vs {{ camp_p_treat_val }} | {{ camp_p_hold_test }} vs {{ camp_p_treat_test }} |
| Assumed profit given up by the hold-out (campaign net) | {{ camp_forgone_val }} ({{ camp_proposed_profit_val }}) | {{ camp_forgone_test }} ({{ camp_proposed_profit_test }}) |
| Smallest detectable save rate after one quarter (5%, 80% power) | {{ camp_mde_val }} | {{ camp_mde_test }} |

*Table 2 — `reports/metrics/campaign_policy.json`. These counts reproduce D5's offline audit
exactly (M0 and M2 coupons), which checks the served code against the analysis. Campaign profit
here uses the served calibrator; D3's {{ val_profit }} used out-of-fold calibration.*

**Read honestly:** the list closes the gap only if a person approves it, and approving it all
*loses* money under the assumptions. The list is for judgment, not rubber-stamping. The hold-out is
the price of finally measuring the save rate: it needs one to two quarters to confirm the assumed
{{ save_rate }}.

## 4. Prepared answers to the question bank (Appendix C)

**Data and modelling**

**4. Windows; why 90 days.** Observation {{ obs_days }} days before a cut-off *T*, label = any order
in [T, T+{{ label_days }} d). Cut-offs: train {{ train_cutoffs }}, validation {{ val_cutoff }}, test
{{ test_cutoff }}. Ninety days matches the client's quarterly campaign, and the median gap between
orders is about two months, so 90 days separates "slow" from "gone".

**5. Why PR-AUC.** Classes are near balance overall but the action targets one side. Accuracy hides
which errors we make, and ROC-AUC rewards ranking the many easy negatives. PR-AUC measures
precision and recall across thresholds. Test {{ test_pr_new }} {{ test_ci_new }} against
{{ test_best_base }} for recency deciles, {{ test_dummy }} for the prior. I also report the lapse
direction ({{ test_pr_lapse }}).

**6. Threshold {{ threshold }}.** It maximises campaign profit on validation under stated costs:
{{ coupon_cost }} per coupon, a {{ discount_rate }} discount, {{ save_rate }} save rate,
{{ gross_margin }} margin and a {{ horizon }}-day value horizon. At a 10% save rate the optimum is
"contact nobody" (sensitivity table, D6 §5.3). The number is a business decision with its
assumptions printed.

**7. What calibration buys.** We do not only rank. Expected profit multiplies a *probability* by a
value, the threshold is in probability units, and monitoring compares mean predicted `p_return`
with the realised return rate. Test ECE {{ test_ece }}.

**8. Most important feature.** `{{ imp_top1 }}` (months with an order of the last six), with
{{ imp_top1_share }} of attribution. It surprised me that regularity beats recency; recency is
partly redundant with it. Top five: {{ imp_top }}.

**9. If the model loses to the business rule on recall.** Measured: "no order in 60 days" as a
coupon policy reaches lapser recall {{ camp_rule_recall_val }} / {{ camp_rule_recall_test }}
(val / test) with {{ camp_rule_coupons_val }} / {{ camp_rule_coupons_test }} coupons and
{{ camp_rule_profit_val }} / {{ camp_rule_profit_test }} profit. The model reaches
{{ camp_model_recall_val }} / {{ camp_model_recall_test }} and loses less. If it lost on recall I
would compare at *equal coupons*: recall alone rises by contacting more people. If it still lost,
I would ship the rule and report the model as a negative result. The promotion gate already
requires +0.05 PR-AUC over the best baseline.

**10. Test protection.** One quarter after validation, never used for any choice. Scored
{{ test_looks }} times, all disclosed: look 1 before the duplicate-key fix ({{ look1_pr_auc }}),
look 2 after the re-run (identical), look 3 for the minimised 19-feature model ({{ test_pr_new }}).
`make evaluate` refuses a new look without `--force`, which records it.

**Recommendation**

**11. Association rules, not collaborative filtering.** {{ guest_pct }} of lines have no customer,
baskets are the unit, the catalogue is {{ b_items }} items at {{ b_density }} density, and a rule is
explainable ("bought with"). I would switch to implicit-feedback CF or a two-tower model once most
traffic is signed in with a history and cross-session personalisation is the goal.

**12. The lift trap.** `{{ trap_ante }}` → `{{ trap_cons }}`: confidence {{ trap_conf }}, lift
{{ trap_lift }}. The consequent is in {{ trap_cons_support }} of all baskets, so the rule mostly
restates popularity. `min_lift` {{ b_lift }} discards it; the median served lift is
{{ lift_median }}.

**13. Why not mine per request.** Apriori over {{ b_baskets }} baskets takes {{ b_ap_s }} s
(FP-Growth in mlxtend {{ b_fp_s }} s) and needs the whole history. Mining is offline; serving is an
index lookup at {{ probe_recommend_p95 }} p95.

**14. 500,000 rules.** Measured with {{ rules_64_n }} rules (64× ours): `/recommend` p95
{{ rules_64_p95 }} ms, {{ rules_64_mem }} memory, because the engine indexes rules by antecedent
item and checks only matching ones. Beyond that I would prune by lift and support or move rules to
a key-value store.

**15. No rule matches.** The strategy becomes `popularity_fallback`, and the response marks each item `source: popularity` and the interface
labels it "Popular with our customers", never "bought with". Coverage: {{ hold_cov }} of holdout baskets
fire a rule; hit-rate@5 {{ hold_hybrid }} against {{ hold_pop }} for best-sellers
({{ hold_ratio }}×).

**Generative AI**

**16. RAG, not fine-tuning.** Six policies ({{ corpus_words }} words, {{ n_chunks }} chunks) that
change. RAG cites the exact chunk and updates by re-indexing. Fine-tuning would bake in stale
policies, cannot cite, and needs data and GPUs we do not have.

**17. No invented SKUs or probabilities.** Product codes and percentages come only from tool output
(`/recommend`, `/predict`). An LLM answer containing a SKU or a percentage absent from that output
is replaced by the deterministic template answer (`guardrails.py`, rule 5), and a test covers it.

**18. Is the assistant good?** On 40 golden and 20 paraphrased questions, tool choice is
{{ ollama_golden }} / {{ ollama_para }} with local `llama3.1:8b`, and out-of-scope refusal is 100%.
The keyword-rule provider scores {{ rule_golden }} / **{{ rule_para }}**, which is why an LLM earns
its place. The live service runs the rule provider (no GPU on App Runner): stated, not hidden.

**19. Retrieval returns nothing relevant.** Below a cosine similarity of {{ sim_floor }} (chosen by
a floor sweep on the golden set), a policy question is handed to a human instead of answered. Tested
with out-of-corpus questions in the golden set.

**20. Why a local model.** Law 25 and the course rule: no personal data to a hosted provider. The
local model is the data-residency answer. It also costs {{ ollama_1k }} per 1,000 queries against
an estimated {{ openai_1k }} hosted (unverified price), at {{ ollama_ms }} per call.

**Engineering and MLOps**

**21. joblib or pickle.** joblib is pickle with efficient NumPy arrays. Both execute code on load,
so I load only artifacts this pipeline built, pinned to the library versions in `model_meta.json`.
XGBoost's JSON or ONNX would remove the risk for untrusted sources.

**22. A "model version".** The pipeline (preprocessing, estimator, calibrator), the threshold and
the feature list in one artifact, plus `model_meta.json`: git commit, data SHA-256, training rows,
library versions, MLflow run, registry version. Served now: `{{ model_version }}`.

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
{{ bf_prauc }} on {{ bf_n }} backfilled predictions.

**28. When managed training is worth it.** When tuning outgrows a laptop: hours not minutes, large
data or GPUs. Here XGBoost tuning took {{ tune_wall_xgboost }} min on {{ model_n_rows }} rows. The
SageMaker job is implemented and dry-run tested. Quota request: {{ d7_quota }}. Result:
{{ d7_sm_result }}.

**Governance**

**29. Automated decisions.** The coupon proposal (Law 25 s. 12.1). By design it is not exclusively
automated: a manager approves every list. We still disclose, give the principal factors on request
(`?explain=3`), allow observations to a person, and now also disclose the random hold-out.

**30. The recall gap.** The options are M1 expected value (profitable, abandons most lapsers),
M2 group thresholds (closes the gap, +{{ m2_extra_pct }} coupons, treats by spend) and M3
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
* The model does not beat "contact nobody" on the test quarter ({{ test_profit_new }}). It beats
  the blanket coupon by {{ test_vs_everyone_new }}.
* The fairness gap is routed to people, not solved; approving the whole review list loses money
  under the assumptions (§3).
* In mlxtend, Apriori is {{ b_speedup }}× faster than FP-Growth at this size.
* Open: save rate unmeasured (the hold-out now exists); serving outside Canada; the live log is
  per-instance SQLite; the hosted LLM provider is unevaluated; the SageMaker run waits on the quota;
  the LégisQuébec article numbers are to be checked by hand.
