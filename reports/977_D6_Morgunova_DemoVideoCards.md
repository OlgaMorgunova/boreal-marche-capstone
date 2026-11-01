<section class="slide title-slide" markdown="1">

<div style="height:1.1in"></div>

# Boréal Marché — the system in 75 seconds

<p class="fr">Le système en 75 secondes · démonstration du service en ligne, commit 422bbab</p>

<div style="height:0.3in"></div>

1. **Retention** — who is about to stop ordering, and the coupon the model *proposes*
2. **Basket** — what else belongs in this basket, and the rule that says why
3. **Assistant** — a French question answered from our policy, with its source
4. **Monitoring** — drift, data quality and latency computed from the live prediction log

<div class="foot" style="color:#cde2fb"><span>Olga Morgunova · Capstone Project 1 · Vanier College</span><span>https://boreal-marche-capstone-4ay3m2fn6qdo4jtgy72vpr.streamlit.app</span></div>
</section>

<section class="slide title-slide" markdown="1">

<div style="height:0.9in"></div>

# Live, tested, governed

<p class="fr">En service, testé, gouverné</p>

| | |
|---|---|
| Retention, held-out quarter | PR-AUC **0.8472** vs 0.7123 best heuristic |
| Campaign vs coupon-to-everyone | **$120k–$217k a year** (assumed 20% save rate) |
| Recommendations | **8.6×** better than best-sellers |
| Total cloud spend | **$3.05** |

Interface: https://boreal-marche-capstone-4ay3m2fn6qdo4jtgy72vpr.streamlit.app<br>
API: https://padp32qm6p.us-east-1.awsapprunner.com/docs<br>
Code: github.com/OlgaMorgunova/boreal-marche-capstone

</section>
