# Governance and Québec Law 25 — Boréal Marché platform

**Status:** design version for D2/D3 (v0.3). The subgroup fairness audit, the final minimisation
decisions and the retention schedule are completed in D5.
**Author:** Olga Morgunova · **Scenario note:** the data are the public UCI Online Retail II dataset,
re-skinned; no real Boréal Marché customer is involved. The obligations below are designed *as if*
the platform processed real personal information of Québec residents.

> **Primary-source check — PENDING (Olga).** The article summaries below are the author's reading of
> the *Act respecting the protection of personal information in the private sector* (CQLR c. P-39.1,
> as amended by Law 25, 2021, c. 25). Before D5 they must be verified against the current text on
> LégisQuébec and the Commission d'accès à l'information (CAI) guidance, as the assignment requires.
> Record the URLs and consultation date here: `____`.

## 1. Relevant obligations (to verify)

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
| D-A | **Retention offer targeting** (Head A) | Whether the customer is put on the quarterly 15% coupon list | Purchase history (dates, amounts, items, cancellations), delivery region; no name/e-mail/ID enters the model | Recency, order frequency and cadence, spend, basket breadth (top factors per prediction published in D5) | **No — by design.** The API returns a *proposed* action (`send_coupon`); a marketing manager reviews and approves the list before sending | Notice in account page + e-mail footer; request factors via privacy@; observations reviewed by a staff member within 30 days |
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
not a rubber stamp — the reviewer sees the principal factors for each proposed customer (D5).

## 4. Customer-facing notice (rendered in the app "About" section — D4)

**English.** Boréal Marché uses an automated system to estimate how likely you are to order again in
the next 90 days, based on your past orders (dates, amounts, products, cancellations) and delivery
region. We use this estimate only to decide who receives our quarterly discount offer; a member of
our marketing team reviews and approves every offer list. You can ask us which information and main
factors were used, have inaccurate information corrected, submit your observations to a staff member
who can review the decision, or opt out of this profiling at any time: privacy@borealmarche.example.

**Français.** Boréal Marché utilise un système automatisé pour estimer la probabilité que vous passiez
une nouvelle commande au cours des 90 prochains jours, à partir de vos commandes passées (dates,
montants, produits, annulations) et de votre région de livraison. Nous utilisons cette estimation
uniquement pour déterminer qui reçoit notre offre de rabais trimestrielle; un membre de notre équipe
marketing examine et approuve chaque liste d'offres. Vous pouvez nous demander quels renseignements et
quels principaux facteurs ont été utilisés, faire corriger des renseignements inexacts, présenter vos
observations à un membre du personnel en mesure de réviser la décision, ou refuser ce profilage en
tout temps : privacy@borealmarche.example.

## 5. Data-minimisation audit (preliminary — decisions finalised in D5)

| Feature group | Needed? | Note |
|---|---|---|
| Recency, frequency, monetary, cadence | Yes | Core signal (spot-check: recency alone PR-AUC 0.58 vs. full model 0.78 CV) |
| Basket shape, category mix | Yes, to be confirmed by ablation in D5 | Derived from products bought; no sensitive categories in this catalogue |
| `weekend_share_180d`, `mean_order_hour_180d` | **Candidate for removal** | Time-of-day behaviour is intrusive relative to its expected value; ablation in D5 |
| `region` | Needed for delivery; **used as a fairness subgroup** | Keep in the audit; if it does not improve PR-AUC materially, remove from the model and keep only for monitoring |
| Customer identifier | Not a model input | Used only to join; `customer_ref` in the API is opaque and not used by the model |
| Name, e-mail, address | Never used | Not present in the feature table |

## 6. Retention and deletion policy (draft)

Raw transactions: kept for the scenario's 2-year analysis horizon, DVC-tracked, never in Git.
Prediction logs (D4): 13 months (one year-over-year comparison), then aggregated and deleted.
Model artifacts: last 5 registry versions. On a verified deletion request the customer's rows are
removed from the next ETL run; models trained on them are retired at the next scheduled retrain.

## 7. Fairness audit plan (executed in D5)

Recall, precision and selection rate by **region**, **tenure band** and **spend quartile**; flag any
group more than 5 points below overall recall; propose ≥ 3 mitigations with trade-offs. A first,
unaudited look at the test-set segments is in D3 §9 (error analysis) and already shows large recall
differences by spend quartile — the D5 audit must address it.
