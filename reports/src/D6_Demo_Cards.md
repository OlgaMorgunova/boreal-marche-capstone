<section class="slide title-slide" markdown="1">

<div style="height:1.1in"></div>

# Boréal Marché — the system in 75 seconds

<p class="fr">Le système en 75 secondes · démonstration du service en ligne, commit {{ d6_commit }}</p>

<div style="height:0.3in"></div>

1. **Retention** — who is about to stop ordering, and the coupon the model *proposes*
2. **Basket** — what else belongs in this basket, and the rule that says why
3. **Assistant** — a French question answered from our policy, with its source
4. **Monitoring** — drift, data quality and latency computed from the live prediction log

<div class="foot" style="color:#cde2fb"><span>Olga Morgunova · Capstone Project 1 · Vanier College</span><span>{{ d6_ui }}</span></div>
</section>

<section class="slide title-slide" markdown="1">

<div style="height:0.9in"></div>

# Live, tested, governed

<p class="fr">En service, testé, gouverné</p>

| | |
|---|---|
| Retention, held-out quarter | PR-AUC **{{ test_pr_auc }}** vs {{ test_best_base }} best heuristic |
| Campaign vs coupon-to-everyone | **{{ annual_low }}–{{ annual_high }} a year** (assumed {{ save_rate }} save rate) |
| Recommendations | **{{ hold_ratio }}×** better than best-sellers |
| Total cloud spend | **{{ aws_total }}** |

Interface: {{ d6_ui }}<br>
API: {{ d6_docs }}<br>
Code: github.com/OlgaMorgunova/boreal-marche-capstone

</section>
