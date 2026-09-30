# Cleaning report

Generated 2026-09-29T23:04:27+00:00 by `make etl` (`boreal_marche.data.cleaning`).

| Rule | Rows before | Rows after | Removed | Flagged | Rationale |
|---|---:|---:|---:|---:|---|
| R0 normalise codes | 1,067,371 | 1,067,371 | 0 | 0 | Stock codes/invoices contain trailing spaces and mixed case ('47503J ', 'm'); without normalisation the same SKU is counted twice. |
| R1 drop sheet overlap | 1,067,371 | 1,044,848 | 22,523 | 0 | Both yearly sheets contain 2010-12-01..2010-12-09. Naive concatenation double-counts those orders, inflating frequency and monetary features. |
| R2 drop exact duplicate lines | 1,044,848 | 1,033,034 | 11,814 | 0 | Identical lines (same invoice, SKU, quantity, timestamp, price, customer) are system double-scans; they inflate quantities and basket co-occurrence. Description is not part of the key (spelling variants of one SKU). |
| R3 separate cancellations | 1,033,034 | 1,013,930 | 19,104 | 0 | Invoices prefixed 'C' are reversals, not purchases. Kept in a separate table used only as a point-in-time feature; never netted (netting leaks future events). |
| R4 drop bad-debt adjustments | 1,013,930 | 1,013,924 | 6 | 0 | Invoices prefixed 'A' are accounting adjustments ('Adjust bad debt'). |
| R5 drop non-positive quantity | 1,013,924 | 1,010,531 | 3,393 | 0 | Non-cancellation lines with quantity <= 0 are stock write-offs/damages ('lost', 'damaged', '?') with no customer. |
| R6 drop non-positive price | 1,010,531 | 1,007,910 | 2,621 | 0 | Price <= 0 lines are free samples or manual corrections; they are not sales and would distort monetary features. |
| R7 drop service / fee SKUs | 1,007,910 | 1,003,338 | 4,572 | 0 | Postage, carriage, manual, bank/Amazon fees, test and gift-voucher codes are not products: they would dominate baskets ('recommend postage') and skew category mix. |
| R8 canonicalise descriptions | 1,003,338 | 1,003,338 | 0 | 58,116 | The same SKU appears under several spellings; one canonical label per stock code keeps the rule engine and catalogue consistent. |
| R9 flag + cap outliers | 1,003,338 | 1,003,338 | 0 | 1,642 | Lines above the 99.9% quantile of quantity (520) or line revenue (734.40) — e.g. the 80,995-unit error — are flagged and revenue is capped for features instead of dropped (they are real orders but not representative). Caps are estimated on pre-validation data only. |
| R10 null customer id (Head A view) | 1,003,338 | 776,577 | 226,761 | 226,761 | Guest checkouts cannot be attributed to a customer, so they are excluded from the retention table (Head A) but kept for basket mining (Head B) where the invoice is the unit. |

## Before / after

| Measure | Bronze | Silver |
|---|---:|---:|
| lines | 1,067,371 | 1,003,338 |
| invoices | 53,628 | 39,516 |
| skus | 5,305 | 4,724 |
| customers | 5,942 | 5,852 |
| head_a_lines |  | 776,577 |
| head_a_invoices |  | 36,594 |
| head_b_invoices |  | 39,516 |
| revenue | 19,287,250.57 | 19,643,309.77 |
| date_min | 2009-12-01 07:45:00 | 2009-12-01 07:45:00 |
| date_max | 2011-12-09 12:50:00 | 2011-12-09 12:50:00 |
| mean_lines_per_invoice |  | 25.39 |

Distinct descriptions: 5,335 -> 4,684.
