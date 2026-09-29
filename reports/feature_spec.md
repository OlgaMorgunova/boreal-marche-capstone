# Feature specification (Head A)

37 features. Source: `boreal_marche.features.FEATURE_SPEC`. All features are computed from rows strictly before the cutoff `T`.

| # | Feature | Family | Type | Definition | Rationale | Missingness policy |
|---:|---|---|---|---|---|---|
| 1 | `recency_days` | recency | float | Days from last order to cutoff | Strongest classical churn signal (RFM). | never missing (0 is a true zero) |
| 2 | `tenure_days` | recency | float | Days from first order in history to cutoff | Long-standing customers churn less. | never missing; left-censored at 2009-12-01 (see tenure_censored) |
| 3 | `tenure_censored` | recency | int | 1 if first observed order is within 30 days of the data start (true tenure unknown) | Makes left-censoring explicit instead of pretending tenure is exact. | never missing (0 is a true zero) |
| 4 | `orders_30d` | frequency | int | Distinct invoices in [T-30d, T) | Very recent engagement. | never missing (0 is a true zero) |
| 5 | `orders_90d` | frequency | int | Distinct invoices in [T-90d, T) | Same horizon as the label window. | never missing (0 is a true zero) |
| 6 | `orders_180d` | frequency | int | Distinct invoices in the observation window | Core frequency signal. | never missing (0 is a true zero) |
| 7 | `orders_lifetime` | frequency | int | Distinct invoices before T | Habit strength; left-censored. | never missing (0 is a true zero) |
| 8 | `active_months_180d` | frequency | int | Distinct calendar months with an order in the observation window | Regularity vs. one burst of orders. | never missing (0 is a true zero) |
| 9 | `revenue_30d` | monetary | float | Capped revenue in [T-30d, T) | Recent spend. | never missing (0 is a true zero) |
| 10 | `revenue_90d` | monetary | float | Capped revenue in [T-90d, T) | Recent spend. | never missing (0 is a true zero) |
| 11 | `revenue_180d` | monetary | float | Capped revenue in the observation window | Customer value; also the spend-quartile for the fairness audit. | never missing (0 is a true zero) |
| 12 | `revenue_lifetime` | monetary | float | Capped revenue before T | Lifetime value proxy; left-censored. | never missing (0 is a true zero) |
| 13 | `aov_180d` | monetary | float | Mean order value in the observation window | Wholesale vs. small buyers behave differently. | never missing (0 is a true zero) |
| 14 | `max_order_value_180d` | monetary | float | Largest order value in window | Captures one-off bulk orders. | never missing (0 is a true zero) |
| 15 | `mean_unit_price_180d` | monetary | float | Mean unit price of lines in window | Price tier of the assortment bought. | never missing (0 is a true zero) |
| 16 | `mean_gap_days` | cadence | float | Mean days between consecutive distinct order days (lifetime) | Expected re-order cycle. | NaN when fewer than 2 distinct order days (gap undefined); imputed with median + indicator |
| 17 | `std_gap_days` | cadence | float | Std of days between distinct order days (lifetime) | Regular vs. erratic buyers. | NaN when fewer than 3 order days; median + indicator |
| 18 | `overdue_ratio` | cadence | float | recency_days / mean_gap_days | >1 means the customer is late versus their own cycle. | NaN when fewer than 2 distinct order days (gap undefined); imputed with median + indicator |
| 19 | `orders_trend` | cadence | float | orders in last 90d minus orders in the previous 90d | Accelerating vs. decelerating engagement. | never missing (0 is a true zero) |
| 20 | `revenue_trend` | cadence | float | (revenue last 90d - revenue previous 90d) / revenue_180d | Scale-free spend trend in [-1, 1]. | never missing (0 is a true zero) |
| 21 | `lines_per_order_180d` | basket | float | Mean invoice lines per order | Basket breadth. | never missing (0 is a true zero) |
| 22 | `units_per_line_180d` | basket | float | Mean units per line | Wholesale (large multiples) vs. retail buying. | never missing (0 is a true zero) |
| 23 | `distinct_skus_180d` | basket | int | Distinct stock codes bought in window | Assortment breadth. | never missing (0 is a true zero) |
| 24 | `sku_repeat_rate_180d` | basket | float | 1 - distinct SKUs / lines in window | Replenishment buyers repeat SKUs. | never missing (0 is a true zero) |
| 25 | `share_seasonal` | category | float | Share of window revenue in category 'seasonal' (keyword-derived) | Category mix; seasonal buyers have predictable gaps. | never missing (0 is a true zero) |
| 26 | `share_kitchen` | category | float | Share of window revenue in category 'kitchen' (keyword-derived) | Category mix; seasonal buyers have predictable gaps. | never missing (0 is a true zero) |
| 27 | `share_home_decor` | category | float | Share of window revenue in category 'home_decor' (keyword-derived) | Category mix; seasonal buyers have predictable gaps. | never missing (0 is a true zero) |
| 28 | `share_bags_wrap` | category | float | Share of window revenue in category 'bags_wrap' (keyword-derived) | Category mix; seasonal buyers have predictable gaps. | never missing (0 is a true zero) |
| 29 | `share_lighting` | category | float | Share of window revenue in category 'lighting' (keyword-derived) | Category mix; seasonal buyers have predictable gaps. | never missing (0 is a true zero) |
| 30 | `category_entropy_180d` | category | float | Shannon entropy of category revenue shares | Diversified buyers are stickier. | never missing (0 is a true zero) |
| 31 | `cancel_lines_180d` | behaviour | int | Cancellation lines in window (point-in-time, from the separated cancellation table) | Dissatisfaction signal. | never missing (0 is a true zero) |
| 32 | `cancel_rate_180d` | behaviour | float | cancel lines / (purchase + cancel lines) | Normalised dissatisfaction. | never missing (0 is a true zero) |
| 33 | `weekend_share_180d` | behaviour | float | Share of orders placed on Sat/Sun | Business vs. consumer buying pattern. | never missing (0 is a true zero) |
| 34 | `mean_order_hour_180d` | behaviour | float | Mean hour of day of orders | Business-hours buyers are likely B2B. | never missing (0 is a true zero) |
| 35 | `outlier_line_share_180d` | behaviour | float | Share of lines flagged as outliers by cleaning rule R9 | Keeps information about bulk orders that were capped. | never missing (0 is a true zero) |
| 36 | `region` | geography | category | Re-skinned delivery region of the last order (QC/ON/MARITIMES/INTL) | Delivery geography; also a fairness-audit subgroup. | never missing; unknown countries map to INTL |
| 37 | `season_q` | context | category | Calendar quarter of the cutoff (1-4) | Label base-rate is seasonal (Q4 peak); known at prediction time. | never missing (derived from the cutoff date) |
