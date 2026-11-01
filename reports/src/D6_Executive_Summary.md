<p class="lang">English · français à la page 2</p>

<div class="title" markdown="1">

# Executive summary — Customer Intelligence Platform
**Boréal Marché** · for the management committee · Olga Morgunova · Capstone Project 1 (D6)
</div>

## The problem

Every quarter we send a {{ discount_rate }} coupon to every active customer. On the most recent
quarter we tested, that policy **lost {{ k_test_everyone }}**: most coupons went to loyal customers
who would have ordered anyway, and the discount they used was pure loss. The cross-sell strip on the
website is chosen by hand, and two employees answer the same forty questions by e-mail.

## What was built

Before each campaign, the system estimates for every customer the chance they order again within
three months, and **proposes** a coupon only for those at risk; a manager approves the list. It also
suggests products that go with a basket — {{ hold_ratio }} times more often right than showing our
best-sellers — and answers common questions in French and English, always quoting our policy and
passing the customer to staff when unsure. It is running today: {{ d6_ui }} (15-minute
presentation: {{ d6_talk }}).

## Expected annual impact

| One quarterly campaign | Last quarter tested | Quarter before |
|---|---:|---:|
| Coupon to everyone (today) | {{ test_everyone }} | {{ val_everyone }} |
| Targeted list (proposed) | {{ test_profit }} | {{ val_profit }} |
| **Difference** | **{{ test_gain }}** | **{{ val_gain }}** |

Replacing the blanket coupon with the targeted list is worth **{{ annual_low }} to {{ annual_high }}
a year** over four campaigns. Most of the saving comes from **not sending** about
{{ val_not_sent_pct }}–{{ test_not_sent_pct }} of today's coupons. **The figure rests on one
assumption we have never measured:** that a coupon wins back one in five customers who would
otherwise leave. If it is one in ten, no coupon campaign makes money — but stopping the blanket
coupon still avoids its loss. Running cost: {{ aws_total }} of cloud spend to date; about
{{ ar_idle_month }} a month while the service is on.

## Risk and governance

* **Law 25.** The system proposes and a person decides, so no customer is targeted by a machine
  alone. On request, a customer can be told the main factors behind a proposal. The bilingual notice
  is in place, and only {{ n_model }} of the {{ n_all }} pieces of purchase information are used — the
  rest are no longer stored. No personal information is sent to an outside AI provider.
* **Fairness — an open issue.** The targeted list reaches {{ overall_recall }} of customers who are
  about to leave, but only {{ top_recall }} of those among our **highest spenders**: about
  {{ missed_top_value }} of margin missed in one quarter. A manager-review list for these customers
  is the proposed fix.
* **Data location.** The demonstration runs in a US data centre on public, anonymised data. With
  real customer data it must move to Canada (about $30 a month more).

<div class="decision" markdown="1">

**Recommendation — approve a two-quarter pilot.** (1) Stop the blanket coupon and send the targeted
list after manager review. (2) Hold back 10% of the proposed customers as a no-coupon control group,
to measure what the coupon really does. (3) Add the high-spender review list. (4) Move hosting to
Canada before any real customer data is used. After two quarters, decide with measured results:
keep, refine, or stop coupons.
</div>

<div class="pagebreak"></div>

<p class="lang">Français · English on page 1</p>

<div class="title" markdown="1">

# Sommaire exécutif — Plateforme d'intelligence client
**Boréal Marché** · pour le comité de direction · Olga Morgunova · Projet de synthèse 1 (D6)
</div>

## Le problème

Chaque trimestre, nous envoyons un coupon de {{ fr_discount }} à tous les clients actifs. Au
dernier trimestre testé, cette politique a **perdu {{ fr_k_test_everyone }}** : la plupart des
coupons sont allés à des clients fidèles qui auraient commandé de toute façon, et le rabais utilisé
est une perte nette. Les suggestions de produits du site sont choisies à la main, et deux employés
répondent par courriel aux mêmes quarante questions.

## Ce qui a été construit

Avant chaque campagne, le système estime pour chaque client la probabilité qu'il commande de nouveau
d'ici trois mois et **propose** un coupon seulement aux clients à risque ; un gestionnaire approuve
la liste. Il suggère aussi des produits qui complètent un panier — {{ fr_hold_ratio }} fois plus souvent
justes que nos meilleurs vendeurs — et répond aux questions courantes en français et en anglais, en
citant toujours notre politique et en transférant le client à un employé en cas de doute. Il est en
service : {{ d6_ui }} (présentation de 15 minutes : {{ d6_talk }}).

## Impact annuel attendu

| Une campagne trimestrielle | Dernier trimestre testé | Trimestre précédent |
|---|---:|---:|
| Coupon à tous (aujourd'hui) | {{ fr_test_everyone }} | {{ fr_val_everyone }} |
| Liste ciblée (proposée) | {{ fr_test_profit }} | {{ fr_val_profit }} |
| **Écart** | **{{ fr_test_gain }}** | **{{ fr_val_gain }}** |

Remplacer le coupon à tous par la liste ciblée vaut **de {{ fr_annual_low }} à {{ fr_annual_high }}
par année** sur quatre campagnes. L'essentiel de l'économie vient du fait de **ne pas envoyer**
environ {{ fr_val_not_sent_pct }} à {{ fr_test_not_sent_pct }} des coupons actuels. **Ce chiffre
repose sur une hypothèse jamais mesurée :** qu'un coupon ramène un client sur cinq qui serait parti.
Si c'est un sur dix, aucune campagne de coupons n'est rentable — mais cesser le coupon à tous évite
quand même sa perte. Coût d'exploitation : {{ fr_aws_total }} en services infonuagiques à ce jour ; environ
{{ fr_ar_idle_month }} par mois lorsque le service est actif.

## Risques et gouvernance

* **Loi 25.** Le système propose et une personne décide : aucun client n'est ciblé par une machine
  seule. Sur demande, un client peut connaître les principaux facteurs d'une proposition. L'avis
  bilingue est en place, et seuls {{ n_model }} des {{ n_all }} renseignements d'achat sont utilisés —
  les autres ne sont plus conservés. Aucun renseignement personnel n'est transmis à un fournisseur
  d'IA externe.
* **Équité — un enjeu ouvert.** La liste ciblée rejoint {{ fr_overall_recall }} des clients sur le
  point de partir, mais seulement {{ fr_top_recall }} parmi nos **plus gros acheteurs** : environ
  {{ fr_missed_top_value }} de marge manquée en un trimestre. Une liste de révision par un
  gestionnaire pour ces clients est la correction proposée.
* **Lieu des données.** La démonstration tourne dans un centre de données américain, sur des
  données publiques anonymisées. Avec de vraies données clients, elle doit déménager au Canada
  (environ 30 $ de plus par mois).

<div class="decision" markdown="1">

**Recommandation — approuver un projet pilote de deux trimestres.** (1) Cesser le coupon à tous et
envoyer la liste ciblée après révision par un gestionnaire. (2) Garder 10 % des clients proposés
comme groupe témoin sans coupon, pour mesurer l'effet réel du coupon. (3) Ajouter la liste de
révision des plus gros acheteurs. (4) Déménager l'hébergement au Canada avant d'utiliser de vraies
données clients. Après deux trimestres, décider avec des résultats mesurés : maintenir, ajuster ou
cesser les coupons.
</div>
