# Governance and Québec Law 25 — Boréal Marché platform

**Status:** D5 version (v0.9.5). The subgroup fairness audit (§7), the minimisation decisions (§5)
and the retention schedule (§6) are complete; evidence in
[`reports/977_D5_Morgunova_OptimizationEthics.md`](reports/977_D5_Morgunova_OptimizationEthics.md).
**Author:** Olga Morgunova · **Scenario note:** the data are the public UCI Online Retail II dataset,
re-skinned; no real Boréal Marché customer is involved. The obligations below are designed *as if*
the platform processed real personal information of Québec residents.

> **Primary sources consulted.**
> * Commission d'accès à l'information (CAI), read 2026-11-01:
>   "Vos droits" — https://cai.gouv.qc.ca/protection-renseignements-personnels/citoyens-protection-renseignements-personnels/acces-et-protection-renseignements-personnels
>   (section "Être informé d'une décision automatisée et en demander la révision") and
>   "Principaux changements de la Loi 25" — https://www.cai.gouv.qc.ca/protection-renseignements-personnels/sujets-et-domaines-dinteret/principaux-changements-loi-25
>   (sections "Traitement automatisé", "Profilage", "Évaluation des facteurs relatifs à la vie privée").
>   They confirm for s. 12.1: notice "au plus tard au moment où [l'entreprise] l'informe de cette
>   décision"; on request, the personal information used and "les raisons ainsi que les principaux
>   facteurs et paramètres ayant mené à la décision"; rectification; observations to "un membre du
>   personnel en mesure de réviser la décision". Profiling functions "ne pourront être activées par
>   défaut". An EFVP is required for any information-system project involving personal information
>   and before communication outside Québec, proportionate to sensitivity, purpose and quantity.
> * *Act respecting the protection of personal information in the private sector*, CQLR c. P-39.1
>   (LégisQuébec, https://www.legisquebec.gouv.qc.ca/en/document/cs/P-39.1): the site refused
>   automated reading on 2026-11-01. **Article numbers and the 30-day delay (s. 32) still need one
>   manual check against the consolidated text — Olga, date: 01.11.2026.**

## 1. Relevant obligations

| Article | Obligation (summary) | Where the design responds |
|---|---|---|
| s. 12.1 | A decision based **exclusively** on automated processing of personal information must be disclosed to the person no later than when they are informed of the decision; on request they are told the personal information used, the **reasons and principal factors** and their right to have the information corrected; they may submit observations to a staff member who can **review the decision**. | Human approval step (§3); per-customer factors (D5 interpretability); notice (§4); review channel |
| s. 8.1 | Technology with functions that allow a person to be identified, located or **profiled** must be disclosed, and such functions deactivated by default where the law requires it. | Retention scoring is profiling → disclosed in the notice; opt-out honoured (§4) |
| s. 3.3 | Privacy impact assessment (PIA / *EFVP*) for any project to acquire, develop or overhaul an information system involving personal information. | This file + DATA_CARD serve as the PIA working draft |
| s. 17 | Communicating personal information **outside Québec** requires a prior PIA and adequate protection. | Hosted LLM provider receives no personal data; local Ollama provider is the data-residency answer (Head C); training in `ca-central-1`; the D4 demo service runs in `us-east-1` (App Runner is not offered in Canada) and processes only public, pseudonymised data — real customer data would require ECS in `ca-central-1` |
| s. 3.1 | A person in charge of the protection of personal information (default: the CEO), published. | Scenario role: Boréal Marché privacy officer |
| s. 23 | Destroy or anonymise personal information once the purpose is achieved. | Retention policy (§6) |

## 2. Automated-decision register

| # | Decision point | What is decided | Personal information used | Principal factors | Exclusively automated? | How the customer exercises rights |
|---|---|---|---|---|---|---|
| D-A | **Retention offer targeting** (Head A) | Whether the customer is put on the quarterly 15% coupon list | Since v0.9.5 only order dates and amounts (19 features, §5); region is stored for delivery and the fairness audit but not used by the model; no name/e-mail/ID | Months with an order, gap between orders and its regularity, recent spend, tenure — exact top 3 per prediction from `POST /predict?explain=3` (D5 §5) | **No — by design.** The API returns a *proposed* action (`send_coupon`); a marketing manager reviews and approves the list before sending | Notice in account page + e-mail footer; request factors via privacy@; observations reviewed by a staff member within 30 days |
| D-B | **Basket recommendations** (Head B) | Which 5 products are shown in "you might also like" | Current basket contents only (no customer id) | The association rule that fired ("because you added X"), shown in the UI | Yes, but not a decision *about the person* (same basket → same output for everyone); low impact | Explanation shown inline; strategy disclosed (rules vs. popularity) |
| D-C | **Assistant answers & hand-off** (Head C, Week 9–10) | Answer, refusal, or hand-off to a human | The question text; for account questions, the customer's own data through the retention/recommendation tools | Retrieved policy passages (cited); tool outputs | Refusal/hand-off paths route to a human; never discloses another customer's data | "Talk to a person" always available; citations on every grounded answer |

## 3. Design consequence (stated)

Because a decision based **exclusively** on automated processing triggers the s. 12.1 disclosure and
review obligations, **we inserted a human approval step between the model and the campaign**: the
model scores and proposes (`send_coupon: true/false`, calibrated probability, threshold, model
version); a marketing manager approves the campaign list, can override individual customers, and the
override is logged. This is why `/predict` returns a *proposal* and not a sent coupon, and why the
weekly retraining workflow opens a pull request instead of deploying. We still publish the notice
(§4) because the system profiles customers (s. 8.1) and because the human review must be meaningful,
not a rubber stamp — the reviewer gets the principal factors for each proposed customer
(`/predict/batch?explain=3`) and, since v1.0, a separate "high-value review" list of top-spend
customers the single threshold misses (`GET /campaign`, the `action` field of `/predict`; §7).

## 4. Customer-facing notice (rendered in the app "About" section — D4)

**English.** Boréal Marché uses an automated system to estimate how likely you are to order again in
the next 90 days, based on the dates and amounts of your past orders. We use this estimate only to decide who receives our quarterly discount offer; a member of
our marketing team reviews and approves every offer list. Each quarter a small random group of
customers (one in ten of those selected) does not receive the offer, so that we can measure whether
it works; the draw does not depend on any information about you. You can ask us which information and main
factors were used, have inaccurate information corrected, submit your observations to a staff member
who can review the decision, or opt out of this profiling at any time: privacy@borealmarche.example.

**Français.** Boréal Marché utilise un système automatisé pour estimer la probabilité que vous passiez
une nouvelle commande au cours des 90 prochains jours, à partir des dates et des montants de vos
commandes passées. Nous utilisons cette estimation
uniquement pour déterminer qui reçoit notre offre de rabais trimestrielle; un membre de notre équipe
marketing examine et approuve chaque liste d'offres. Chaque trimestre, un petit groupe de clients tiré
au hasard (un sur dix parmi ceux qui sont sélectionnés) ne reçoit pas l'offre, afin de mesurer si elle
est efficace; ce tirage ne dépend d'aucun renseignement vous concernant. Vous pouvez nous demander quels renseignements et
quels principaux facteurs ont été utilisés, faire corriger des renseignements inexacts, présenter vos
observations à un membre du personnel en mesure de réviser la décision, ou refuser ce profilage en
tout temps : privacy@borealmarche.example.

## 5. Data-minimisation audit (final — D5)

Rule, written before the ablation: a feature group whose removal costs less than **0.005 PR-AUC**
(5 customer-grouped CV folds of the training snapshots, champion hyper-parameters) is not necessary
and is removed. Result (`reports/metrics/minimisation.json`, `retention/minimise.py`):

| Feature group | CV PR-AUC loss if removed | Decision |
|---|---:|---|
| Time of day (`weekend_share_180d`, `mean_order_hour_180d`) | −0.0001 | **Removed** — intrusive and unnecessary |
| `region` | −0.0001 | **Removed from the model**; still collected for delivery, monitoring and the fairness audit |
| Cancellations (`cancel_lines_180d`, `cancel_rate_180d`) | +0.0001 | **Removed** |
| Category mix (5 shares + entropy) | −0.0002 | **Removed** — what kind of products a person buys is not needed |
| Basket shape (4 features) | +0.0000 | **Removed** |
| Outlier-line share | +0.0000 | **Removed** |
| Lifetime value (`revenue_lifetime`, `orders_lifetime`) | +0.0008 | **Removed** |
| Recency, frequency, monetary (180 days), cadence, tenure, quarter (19 features) | not ablated: the core signal | Kept |

All 18 removed together: CV PR-AUC 0.7809 → 0.7802, validation 0.8094 → 0.8052. The model
`0.9.5+xgboost` uses 19 features; the API accepts the removed fields for compatibility, ignores them
and **does not store them** (`features.COLLECTED`, asserted by a test). Customer identifier, name,
e-mail and address were never inputs.

## 6. Retention and deletion policy (final — D5)

| Data | Kept | Then | Mechanism |
|---|---|---|---|
| Raw transactions | 2 years (scenario analysis horizon) | deleted; models retrained without them | DVC-tracked, never in Git |
| Prediction / request / chat log | 13 months (one year-over-year comparison) | aggregated (daily counts, histograms), rows deleted | `LogStore.purge`, monthly |
| Removed features in v0.9 log rows | until those rows age out (≤ 13 months; App Runner's log is lost at each deploy) | — | no new row contains them |
| Model artifacts | last 5 registry versions | older versions deleted | MLflow registry |
| Deletion request | rows removed from the next ETL run; models trained on them retired at the next weekly retrain | confirmation to the person | privacy@ inbox, within the legal response delay (30 days, s. 32 — to verify) |

## 7. Fairness audit (executed — D5)

Lapser recall, coupon precision and selection rate by **region**, **tenure band** and **spend
quartile** (edges fixed on the training snapshots), validation snapshot, served model, threshold
0.61 (`retention/fairness.py`, `reports/metrics/fairness.json`). **Flagged (> 5 points below
overall recall, ≥ 30 lapsers):** tenure > 12 months, spend Q3 and **spend Q4 (top): 46% vs 89%
overall**; on test also region ON. Cause: one value-blind threshold applied to customers whose
probability of returning is high. Mitigations compared — expected-value policy (M1), per-quartile
thresholds (M2), re-weighted training (M3) — none dominates (D5 §7). **Adopted:** keep the single
threshold for the automatic proposal, give the human reviewer a separate high-value review list
(the customers M2 would add), and run a 10% randomised hold-out for two quarters to measure the
coupon's effect per spend quartile before changing the rule. The promotion gate keeps "no worse
than the champion" until then.

**Built and served (v1.0, `retention/campaign.py`).** Every `/predict` response carries an
`action`: `coupon` (proposed), `holdout` (proposed, withheld this campaign), `review` (not proposed;
spend Q3/Q4 with `p_return` between 0.61 and the M2 group threshold — Q3 ≤ 0.73, Q4 ≤ 0.88 — listed
for the reviewer) or `none`. `GET /campaign` returns the list the reviewer approves. The automatic
proposal is unchanged; spend only routes a case to a person. The hold-out is drawn by
`sha256(campaign_id:customer_ref)`, so it is stateless, auditable and blind to every feature; a new
`campaign.campaign_id` in `params.yaml` draws a new sample each quarter. The thresholds are frozen in
`artifacts/campaign_policy.json` for one model version; the API refuses a policy written for another.
Scored on the snapshots (`reports/metrics/campaign_policy.json`): if the reviewer approves the whole
list, top-quartile lapser recall goes from 0.46 to 0.89 (validation) and from 0.16 to 0.82 (test),
for 368 / 450 extra cases to review, of which 26–28% are lapsers. The hold-out arms are balanced
(mean `p_return` 0.36 vs 0.35 on validation); with one quarter of this dataset's size it detects a
save rate of about 0.16–0.21, so confirming the assumed 0.20 needs **one to two quarters** — the
design's "two quarters" stands. The disclosure (§4) now mentions the random group.

**What it costs, under the stated assumptions (20% save rate, `params.yaml`).** Approving the whole
review list *loses* $8.2k (validation) and $16.9k (test): most of those high spenders would have
returned anyway, so the list is for judgment, not for blanket approval — the same trade-off that kept
M2 from being adopted. The hold-out gives up $2.7k (validation, 35% of that campaign's net $7.7k)
and $0.5k (test) of assumed profit. That is the price of measuring a save rate that every other
dollar figure depends on, and it is the reason the hold-out lasts two quarters, not indefinitely.
