<p class="lang">English · français à la page 2</p>

<div class="title" markdown="1">

# Executive summary — Customer Intelligence Platform
**Boréal Marché** · for the management committee · Olga Morgunova · Capstone Project 1 (D6)
</div>

## The problem

Every quarter we send a 15% coupon to every active customer. On the most recent
quarter we tested, that policy **lost −$55k**: most coupons went to loyal customers
who would have ordered anyway, and the discount they used was pure loss. The cross-sell strip on the
website is chosen by hand, and two employees answer the same forty questions by e-mail.

## What was built

Before each campaign, the system estimates for every customer the chance they order again within
three months, and **proposes** a coupon only for those at risk; a manager approves the list. It also
suggests products that go with a basket — 8.6 times more often right than showing our
best-sellers — and answers common questions in French and English, always quoting our policy and
passing the customer to staff when unsure. It is running today: https://boreal-marche-capstone-4ay3m2fn6qdo4jtgy72vpr.streamlit.app (15-minute
presentation: https://youtu.be/2bcNpRwMQxE).

## Expected annual impact

| One quarterly campaign | Last quarter tested | Quarter before |
|---|---:|---:|
| Coupon to everyone (today) | −$55,176 | −$22,565 |
| Targeted list (proposed) | −$1,039 | $7,421 |
| **Difference** | **$54,137** | **$29,986** |

Replacing the blanket coupon with the targeted list is worth **$120k to $217k
a year** over four campaigns. Most of the saving comes from **not sending** about
32%–56% of today's coupons. **The figure rests on one
assumption we have never measured:** that a coupon wins back one in five customers who would
otherwise leave. If it is one in ten, no coupon campaign makes money — but stopping the blanket
coupon still avoids its loss. Running cost: $3.05 of cloud spend to date; about
$10.22 a month while the service is on.

## Risk and governance

* **Law 25.** The system proposes and a person decides, so no customer is targeted by a machine
  alone. On request, a customer can be told the main factors behind a proposal. The bilingual notice
  is in place, and only 19 of the 37 pieces of purchase information are used — the
  rest are no longer stored. No personal information is sent to an outside AI provider.
* **Fairness — an open issue.** The targeted list reaches 89% of customers who are
  about to leave, but only 46% of those among our **highest spenders**: about
  $74,672 of margin missed in one quarter. A manager-review list for these customers
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

Chaque trimestre, nous envoyons un coupon de 15 % à tous les clients actifs. Au
dernier trimestre testé, cette politique a **perdu −55 k$** : la plupart des
coupons sont allés à des clients fidèles qui auraient commandé de toute façon, et le rabais utilisé
est une perte nette. Les suggestions de produits du site sont choisies à la main, et deux employés
répondent par courriel aux mêmes quarante questions.

## Ce qui a été construit

Avant chaque campagne, le système estime pour chaque client la probabilité qu'il commande de nouveau
d'ici trois mois et **propose** un coupon seulement aux clients à risque ; un gestionnaire approuve
la liste. Il suggère aussi des produits qui complètent un panier — 8,6 fois plus souvent
justes que nos meilleurs vendeurs — et répond aux questions courantes en français et en anglais, en
citant toujours notre politique et en transférant le client à un employé en cas de doute. Il est en
service : https://boreal-marche-capstone-4ay3m2fn6qdo4jtgy72vpr.streamlit.app (présentation de 15 minutes : https://youtu.be/2bcNpRwMQxE).

## Impact annuel attendu

| Une campagne trimestrielle | Dernier trimestre testé | Trimestre précédent |
|---|---:|---:|
| Coupon à tous (aujourd'hui) | −55 176 $ | −22 565 $ |
| Liste ciblée (proposée) | −1 039 $ | 7 421 $ |
| **Écart** | **54 137 $** | **29 986 $** |

Remplacer le coupon à tous par la liste ciblée vaut **de 120 k$ à 217 k$
par année** sur quatre campagnes. L'essentiel de l'économie vient du fait de **ne pas envoyer**
environ 32 % à 56 % des coupons actuels. **Ce chiffre
repose sur une hypothèse jamais mesurée :** qu'un coupon ramène un client sur cinq qui serait parti.
Si c'est un sur dix, aucune campagne de coupons n'est rentable — mais cesser le coupon à tous évite
quand même sa perte. Coût d'exploitation : 3,05 $ en services infonuagiques à ce jour ; environ
10,22 $ par mois lorsque le service est actif.

## Risques et gouvernance

* **Loi 25.** Le système propose et une personne décide : aucun client n'est ciblé par une machine
  seule. Sur demande, un client peut connaître les principaux facteurs d'une proposition. L'avis
  bilingue est en place, et seuls 19 des 37 renseignements d'achat sont utilisés —
  les autres ne sont plus conservés. Aucun renseignement personnel n'est transmis à un fournisseur
  d'IA externe.
* **Équité — un enjeu ouvert.** La liste ciblée rejoint 89 % des clients sur le
  point de partir, mais seulement 46 % parmi nos **plus gros acheteurs** : environ
  74 672 $ de marge manquée en un trimestre. Une liste de révision par un
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
