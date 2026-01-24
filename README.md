# IA-AirBnB

## Objectif
Construire un pipeline complet de data science pour predire le prix d'un logement Airbnb
a partir des donnees de Lyon et Paris. Le projet suit le document de reference
`projet_AirBnB.pdf` et couvre le pretraitement, l'analyse descriptive, la gestion des
valeurs aberrantes, la separation train/test et la modelisation (regressions lineaires).

## Donnees
Les fichiers sources proviennent de InsideAirbnb :
- `listingsLyon.csv`
- `listingsParis.csv`
- `projet_AirBnB.pdf` (document de reference)

Les gros fichiers CSV et le PDF sont versionnes via Git LFS.

## Contenu principal
Le script `ia_airbnb_pipeline.py` realise l'ensemble du pipeline :
1. Chargement des donnees et verification des dimensions/colonnes
2. Pretraitement (selection des variables, doublons, valeurs manquantes, conversions)
3. Analyse descriptive (graphiques comparatifs Lyon vs Paris)
4. Valeurs aberrantes + normalisation MinMax de `price`
5. Train/test split (test_size = 0.4)
6. Regressions lineaires (simple et multiple) avec R2 et visualisations

Les figures sont generees dans `outputs/figures/`.

## Execution
Installer les dependances principales via :

```bash
pip install -r requirements.txt
```

Puis executer :

```bash
python ia_airbnb_pipeline.py
```

## Resultats
Le script affiche les dimensions, le nombre de doublons, les scores R2 et enregistre
les graphiques dans `outputs/figures/` (dossier ignore par Git).