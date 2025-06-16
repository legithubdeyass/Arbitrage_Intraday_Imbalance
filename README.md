# Arbitrage_Intraday_Imbalance

Ce dépôt GitHub présente ma réponse au Case Study proposé par ENGIE pour un poste de **_Quant / Trading assistant en VIE en Roumanie_**, consistant à concevoir une stratégie algorithmique permettant d’exploiter les **opportunités d’arbitrage** entre le **marché Intraday** et le **mécanisme d’Imbalance**, sur la période Juillet à Décembre 2024.

## Objectif de ce projet

Ce projet vise à simuler une **stratégie algorithmique d’optimisation des déséquilibres** sur le marché de l’électricité, dans un contexte réaliste.

À chaque quart d’heure, un trader peut volontairement créer un **déséquilibre** (positif ou négatif) entre la production et la consommation de son portefeuille. Ce déséquilibre sera ensuite réglé par le gestionnaire de réseau, au **prix d’Imbalance**. En choisissant judicieusement ce déséquilibre, le trader peut espérer **profiter d’un arbitrage** favorable entre ce prix d’Imbalance et les prix de marché (notamment le prix Intraday).

Trois cas de figure principaux peuvent se présenter :

1. **Créer un déséquilibre positif (surproduction volontaire)** :  
   Cela revient à injecter plus que nécessaire sur le réseau. Si le prix d’Imbalance est **supérieur** au prix auquel on aurait vendu en Intraday, alors c’est une décision rentable.
   ➝ Il fallait **vendre via l’Imbalance**.

2. **Créer un déséquilibre négatif (sous-production volontaire)** :  
   Cela revient à injecter moins que prévu. Si le prix d’Imbalance est **inférieur** au prix auquel on aurait acheté en Intraday, alors c’est également gagnant.  
   ➝ Il fallait **acheter via l’Imbalance**.

3. **Ne rien faire (rester équilibré)** :  
   Dans les cas où l’arbitrage est défavorable (spread faible ou contre-productif), la meilleure décision est de **ne pas créer de déséquilibre**, et simplement solder sa position sur le marché Intraday.

L’objectif de ce projet est donc de **modéliser, à chaque pas de temps, la meilleure action à prendre** (vendre, acheter ou rester neutre), en fonction des données disponibles :  
prévisions de charge (`load_fcst`), prévisions de production (`solar_fcst`, `wind_fcst`), production réelle (`load_real`, `solar_real`, `wind_real`, `nuclear_real`, `fossil_gas_real`), historiques de prix (`ID_QH_VWAP`, `imb_price_pos`, `imb_price_neg`), niveaux de réserves (`afrr_up`, `afrr_down`, `mfrr_up`, `mfrr_down`), et déséquilibres précédents (`imb_volume`, `imbalance_status`).

C’est un problème **décisionnel** avant d’être un problème de régression pure : il ne s’agit pas seulement de prédire un prix, mais de **choisir une action qui maximise le gain** à la lumière des règles réelles du marché de l’énergie.

## Étapes de ce projet

### 1. Construction de la pipeline

Ce projet a été conçu pour reproduire de manière réaliste le raisonnement d’un desk Intraday, avec une forte exigence de cohérence métier à chaque étape.

#### 1.a : Préparation des données et création de variables explicatives

Une **ingénierie de features** poussée a été mise en place pour exploiter au maximum les informations disponibles :

- **Erreurs de prévision** : écart entre prévisions et valeurs réelles (`load_err`, `solar_err`, `wind_err`),
- **Indicateurs de réserve** : agrégation des capacités de réserve disponibles (`afrr_cover_ratio`, `mfrr_cover_ratio`),
- **État de déséquilibre du réseau** : construction d’un indicateur synthétique (`imbalance_status`) indiquant si les déséquilibres étaient bien couverts par les réserves,
- **Spreads d’arbitrage** : calculs des opportunités entre les prix de marché (`spread_long`, `spread_short`),
- **Historique du comportement marché** : création d’une variable de spread historique pondérée (`historical_spread`),
- **Encodage temporel** : ajout de variables comme l’heure, le jour de la semaine, le mois, etc.,
- **Lagging** : ajout de décalages temporels sur les variables clés pour modéliser les dynamiques (`*_lagged_4/5/6`).

Toutes ces variables ont été soigneusement sélectionnées pour refléter les signaux qu’un trader aurait en temps réel.

#### 1.b : Normalisation

Les variables ont été standardisées via un `StandardScaler`, une étape essentielle pour stabiliser l’apprentissage du modèle de deep learning utilisé ensuite.

#### 1.c : Séparation temporelle des données

Les données ont été divisées en **jeu d’entraînement** (jusqu’à fin 2024) et **jeu de test** (à partir de janvier 2025), afin de garantir une séparation chronologique stricte. 
En entraînement, une validation croisée par `TimeSeriesSplit` (5 folds) a été utilisée, pour respecter l’ordre temporel des données (on ne regarde jamais le futur).

#### 1.d : Gestion des valeurs manquantes

Certaines colonnes critiques comportaient des données manquantes qu’il a fallu traiter avec soin :

- `load_fcst` (prévision de charge) : imputée via **SARIMA** bidirectionnel pour capter la forte saisonnalité journalière,
- `solar_fcst`, `wind_fcst`, `solar_real`, `wind_real` (production) : interpolées par la méthode temporelle `interpolate(method='time')`, suffisante du fait de leur continuité horaire,
- Les valeurs résiduelles manquantes sur les autres colonnes ont été comblées par un **forward fill** (`ffill`) pour éviter toute fuite de données futures.

Ces choix ont été faits pour garantir que chaque ligne utilisée en modélisation représente fidèlement l'information réellement disponible à l’instant t.

### 2. Modélisation et apprentissage

#### 2.a : Formulation du problème

Dans une logique d’aide à la décision, le but est de prédire le **volume à engager** en fonction du signal d’arbitrage identifié sur les prix d’équilibrage.  
La variable cible `target_volume` est définie comme :

- `+10` MW si le spread long (`spread_long`) est positif et justifie une **vente** au prix d'équilibrage,
- `-10` MW si le spread short (`spread_short`) est positif et justifie un **achat** au prix d'équilibrage,
- `0` sinon.

Le modèle doit donc apprendre **à estimer le volume optimal à engager**, entre -10 et +10 MW, sur la base des signaux disponibles.

#### 2.b : Architecture du modèle

Le modèle principal repose sur un **réseau de neurones fully connected** en PyTorch :

- Entrée : les variables explicatives normalisées (voir section 1.a),
- 2 couches cachées avec `ReLU` (128 puis 32 neurones),
- Une couche de sortie avec `Tanh`, multipliée par 10 pour sortir des valeurs entre -10 et 10,
- Optimiseur : `Adam`,
- Fonction de coût : `MSELoss` (Mean Squared Error), car le problème reste formulé comme une **régression continue**.

Une tentative a été faite avec une fonction de coût personnalisée **asymétrique**, pénalisant davantage les **contre-sens** (prédiction d’une position contraire au signal optimal), mais au détriment de la précision globale.  
Le modèle original a donc été conservé.

#### 2.c : Entraînement

L’apprentissage est réalisé avec `TimeSeriesSplit` (5 folds) pour chaque segment temporel, avec :

- 20 époques d'entraînement,
- Batch size : 64,
- Apprentissage sur le passé uniquement, validation sur le futur (logique métier respectée).

### 3. Évaluation des performances

L’évaluation du modèle a été centrée sur **la qualité des décisions prises** et leur **impact financier simulé**.

#### 3.a : PnL réalisé vs PnL optimal

Pour chaque prédiction, le **PnL réalisé** est calculé selon les règles suivantes :

- Si le modèle **vend** (`prediction > 0`), le gain est `spread_long × prediction`,
- Si le modèle **achète** (`prediction < 0`), le gain est `-spread_short × prediction`,
- Sinon, le gain est nul.

Le **PnL optimal** est défini comme le gain maximal possible si on avait parfaitement suivi le spread identifié (`target_volume` appliqué sur le bon spread).

Le ratio `sum(PnL réalisé) / sum(PnL optimal)` donne la **performance de captation** du modèle, ici d’environ **62.55 %**, un résultat encourageant.

#### 3.b : Analyse qualitative des décisions

Chaque ligne de prédiction est ensuite classée dans une des 5 catégories suivantes :

| Type de décision             | Description                                                                 |
|-----------------------------|-----------------------------------------------------------------------------|
| Bonne prédiction            | Bonne prise de position, dans le bon sens du spread                         |
| Opportunité manquée         | Le modèle n’a rien fait alors qu’il y avait un spread exploitable           |
| Mauvaise prise de position  | Le modèle a pris position sans spread exploitable                           |
| Contre-sens                 | Le modèle a pris position **dans le sens opposé** au spread                 |
| Neutre                      | Aucune action, et aucun spread exploitable                                  |

Cette typologie permet une **lecture intuitive** de ce que le modèle réussit ou non, en se plaçant dans la peau d’un trader algorithmique.

Un exemple de résultat :

Bonne prédiction               :  6163  (70.09%)
Opportunité manquée            :   378  (4.30%)
Mauvaise prise de position     :   294  (3.34%)
Contre-sens                    :  1941  (22.07%)
Neutre                         :    17  (0.19%)
