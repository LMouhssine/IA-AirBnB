import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score


# Chemins des donnees d'entree (fichiers locaux)
DATA_LYON_PATH = "listingsLyon.csv"
DATA_PARIS_PATH = "listingsParis.csv"

# Dossiers de sortie pour stocker les figures generees
OUTPUT_DIR = "outputs"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")


# Liste des colonnes conserves pour la prediction du prix.
# On privilegie les variables directement liees au prix (localisation, capacite,
# type de logement, disponibilite, avis) et on exclut les champs textuels longs,
# identifiants, URLs ou metadonnees peu utiles.
RELEVANT_COLUMNS = [
    # Cible
    "price",
    # Localisation
    "latitude",
    "longitude",
    "neighbourhood_cleansed",
    # Type de logement et chambre
    "property_type",
    "room_type",
    # Capacite
    "accommodates",
    "bathrooms_text",
    "bedrooms",
    "beds",
    # Disponibilite
    "has_availability",
    "availability_30",
    "availability_60",
    "availability_90",
    "availability_365",
    "minimum_nights",
    "maximum_nights",
    # Avis
    "number_of_reviews",
    "reviews_per_month",
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value",
    # Comportement de reservation
    "instant_bookable",
    "host_is_superhost",
]


# Colonnes a forcer en numerique. Certaines colonnes peuvent arriver en texte
# (ex: "price" avec symbole monetaire ou "bathrooms_text" en texte).
NUMERIC_COERCE_COLUMNS = [
    "price",
    "bathrooms_text",
    "bedrooms",
    "beds",
    "accommodates",
    "minimum_nights",
    "maximum_nights",
    "availability_30",
    "availability_60",
    "availability_90",
    "availability_365",
    "number_of_reviews",
    "reviews_per_month",
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value",
]


# Colonnes categorielles qui seront encodees via one-hot pour la regression multiple.
CATEGORICAL_COLUMNS = [
    "neighbourhood_cleansed",
    "property_type",
    "room_type",
]


def ensure_dirs():
    # Creer les dossiers de sortie si besoin (idempotent).
    os.makedirs(FIG_DIR, exist_ok=True)


def load_data():
    # Chargement brut des CSV pour Lyon et Paris.
    lyon = pd.read_csv(DATA_LYON_PATH)
    paris = pd.read_csv(DATA_PARIS_PATH)

    print("=== Chargement des donnees ===")
    print(f"Dimensions Lyon: {lyon.shape}")
    print(f"Dimensions Paris: {paris.shape}")
    print(f"Memes colonnes: {list(lyon.columns) == list(paris.columns)}")
    print("====================\n")

    # Retour des deux DataFrames pour la suite du pipeline.
    return lyon, paris


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    # On garde uniquement les colonnes pertinentes presentes dans le jeu de donnees.
    # Cela rend le pipeline robuste si une colonne manque.
    cols = [col for col in RELEVANT_COLUMNS if col in df.columns]
    return df[cols].copy()


def drop_duplicates(df: pd.DataFrame, label: str) -> pd.DataFrame:
    # Comptage explicite des doublons pour tracer la qualite des donnees.
    dup_count = df.duplicated().sum()
    print(f"{label}: doublons avant suppression = {dup_count}")
    # Suppression des lignes dupliquees pour eviter de biaiser les statistiques.
    return df.drop_duplicates()


def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    # Conversion du prix: on retire tout ce qui n'est pas chiffre ou point,
    # puis on force en float. Les erreurs deviennent NaN (et seront imputees).
    if "price" in df.columns:
        df["price"] = pd.to_numeric(
            df["price"].replace(r"[^0-9.]", "", regex=True), errors="coerce"
        )

    # "bathrooms_text" contient du texte ("1 bath", "2.5 baths").
    # On extrait la valeur numerique pour l'utiliser dans les modeles.
    if "bathrooms_text" in df.columns:
        df["bathrooms_text"] = pd.to_numeric(
            df["bathrooms_text"].astype(str).str.extract(r"([0-9]*\.?[0-9]+)")[0],
            errors="coerce",
        )

    # Force la conversion en numerique sur toutes les colonnes choisies.
    for col in NUMERIC_COERCE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Colonnes binaires (t/f -> 1/0) pour faciliter la regression lineaire.
    for col in ["has_availability", "instant_bookable", "host_is_superhost"]:
        if col in df.columns:
            df[col] = df[col].map({"t": 1, "f": 0})

    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    # Imputation numerique par la moyenne pour conserver un maximum de lignes.
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Imputation categorielles avec un libelle unique.
    # Cela evite la perte de lignes lors de l'encodage one-hot.
    categorical_cols = df.select_dtypes(include=["object"]).columns
    if len(categorical_cols) > 0:
        df[categorical_cols] = df[categorical_cols].fillna("Inconnu")

    # Hypothese documentee: on arrondit au dessus (ceil) apres imputation,
    # pour approcher le nombre de salles de bain entier.
    if "bathrooms_text" in df.columns:
        df["bathrooms_text"] = np.ceil(df["bathrooms_text"]).astype(int)

    return df


def recode_rare_categories(df: pd.DataFrame, column: str, min_freq: float = 0.01) -> pd.DataFrame:
    # Regroupe les categories rares sous "Autre" pour limiter la dimension
    # lors de l'encodage one-hot, tout en gardant l'information globale.
    if column in df.columns:
        freq = df[column].value_counts(normalize=True)
        rare = freq[freq < min_freq].index
        df[column] = df[column].where(~df[column].isin(rare), other="Autre")
    return df


def preprocess(df: pd.DataFrame, label: str) -> pd.DataFrame:
    # Pipeline de pretraitement: selection -> doublons -> conversion -> imputation.
    df = select_columns(df)
    df = drop_duplicates(df, label)
    df = convert_types(df)
    df = fill_missing_values(df)

    # Etape de recodage (exigee par le document de reference).
    # On l'applique sur quelques colonnes categorielles pour stabiliser le modele.
    df = recode_rare_categories(df, "property_type", min_freq=0.01)
    df = recode_rare_categories(df, "neighbourhood_cleansed", min_freq=0.01)

    return df


def save_figure(fig, filename: str):
    # Centralise l'enregistrement des figures dans le dossier de sortie.
    path = os.path.join(FIG_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Figure enregistree: {path}")


def descriptive_analysis(lyon: pd.DataFrame, paris: pd.DataFrame):
    # Style seaborn pour des graphes lisibles et coherents.
    sns.set_theme(style="whitegrid")
    # On concatene Lyon et Paris pour faciliter les comparaisons.
    combined = pd.concat(
        [lyon.assign(city="Lyon"), paris.assign(city="Paris")], ignore_index=True
    )

    # 1) Relation entre note globale et prix (par ville)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    sns.scatterplot(
        data=lyon, x="review_scores_rating", y="price", alpha=0.4, ax=axes[0]
    )
    axes[0].set_title("Lyon: note vs prix")
    sns.scatterplot(
        data=paris, x="review_scores_rating", y="price", alpha=0.4, ax=axes[1]
    )
    axes[1].set_title("Paris: note vs prix")
    fig.suptitle("Note vs prix")
    save_figure(fig, "rating_vs_price.png")
    plt.close(fig)

    # 2) Prix moyen par type de propriete (on limite au top 10 pour la lisibilite)
    top_props = combined["property_type"].value_counts().head(10).index
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(
        data=combined[combined["property_type"].isin(top_props)],
        x="property_type",
        y="price",
        hue="city",
        ax=ax,
    )
    ax.set_title("Prix moyen par type de propriete (top 10)")
    ax.set_xlabel("Type de propriete")
    ax.set_ylabel("Prix moyen")
    plt.xticks(rotation=30, ha="right")
    save_figure(fig, "mean_price_by_property_type.png")
    plt.close(fig)

    # 3) Relation disponibilite vs prix (boxplots)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(
        data=combined,
        x="has_availability",
        y="price",
        hue="city",
        ax=ax,
    )
    ax.set_title("Disponibilite vs prix")
    ax.set_xlabel("Disponibilite (0/1)")
    ax.set_ylabel("Prix")
    save_figure(fig, "availability_vs_price.png")
    plt.close(fig)

    # 4) Repartition des logements disponibles / non disponibles
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=combined, x="has_availability", hue="city", ax=ax)
    ax.set_title("Distribution de la disponibilite")
    ax.set_xlabel("Disponibilite (0/1)")
    ax.set_ylabel("Effectif")
    save_figure(fig, "availability_distribution.png")
    plt.close(fig)

    # 5) Distribution des prix par type de chambre
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(
        data=combined,
        x="room_type",
        y="price",
        hue="city",
        ax=ax,
    )
    ax.set_title("Distribution des prix par type de chambre")
    ax.set_xlabel("Type de chambre")
    ax.set_ylabel("Prix")
    plt.xticks(rotation=20, ha="right")
    save_figure(fig, "price_by_room_type.png")
    plt.close(fig)


def outlier_and_scaling(lyon: pd.DataFrame, paris: pd.DataFrame):
    # Visualisation simple des valeurs aberrantes via boxplots.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    sns.boxplot(y=lyon["price"], ax=axes[0])
    axes[0].set_title("Valeurs aberrantes - Lyon")
    sns.boxplot(y=paris["price"], ax=axes[1])
    axes[1].set_title("Valeurs aberrantes - Paris")
    save_figure(fig, "price_outliers_boxplot.png")
    plt.close(fig)

    # Normalisation MinMax du prix, separement pour chaque ville.
    # Cela respecte la consigne et evite l'influence d'une ville sur l'autre.
    scaler_lyon = MinMaxScaler()
    scaler_paris = MinMaxScaler()
    lyon["price_scaled"] = scaler_lyon.fit_transform(lyon[["price"]])
    paris["price_scaled"] = scaler_paris.fit_transform(paris[["price"]])

    return lyon, paris


def train_test_split_data(df: pd.DataFrame, features: list, target: str = "price"):
    # Separation explicite entre variables explicatives (X) et cible (y).
    X = df[features]
    y = df[target]
    # Test size 0.4 comme demande, random_state fixe pour reproductibilite.
    return train_test_split(X, y, test_size=0.4, random_state=42)


def linear_regression_simple(df: pd.DataFrame, label: str):
    # Regression lineaire simple avec une seule variable (accommodates).
    # Cela permet d'evaluer l'impact de la capacite seule sur le prix.
    X_train, X_test, y_train, y_test = train_test_split_data(df, ["accommodates"])

    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)

    print(f"{label} - Regression lineaire simple R2: {r2:.4f}")

    # Tracer la droite de regression sur le nuage de points de test.
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(X_test["accommodates"], y_test, alpha=0.4, label="Reel")
    x_line = np.linspace(df["accommodates"].min(), df["accommodates"].max(), 100)
    x_line_df = pd.DataFrame({"accommodates": x_line})
    y_line = model.predict(x_line_df)
    ax.plot(x_line, y_line, color="red", label="Droite de regression")
    ax.set_title(f"{label} - Regression lineaire simple")
    ax.set_xlabel("Accommodates")
    ax.set_ylabel("Prix")
    ax.legend()
    save_figure(fig, f"simple_regression_{label.lower()}.png")
    plt.close(fig)


def linear_regression_multiple(df: pd.DataFrame, label: str):
    # Regression lineaire multiple avec un ensemble de variables pertinentes.
    # La presence de colonnes categorielles necessite un encodage one-hot.
    feature_cols = [
        "accommodates",
        "bathrooms_text",
        "bedrooms",
        "beds",
        "latitude",
        "longitude",
        "minimum_nights",
        "maximum_nights",
        "availability_365",
        "number_of_reviews",
        "review_scores_rating",
        "instant_bookable",
        "host_is_superhost",
        "room_type",
        "property_type",
        "neighbourhood_cleansed",
    ]

    # Garder uniquement les colonnes presentes pour eviter les erreurs.
    feature_cols = [col for col in feature_cols if col in df.columns]
    model_df = df[feature_cols + ["price"]].copy()

    # Encodage one-hot des colonnes categorielles.
    # drop_first pour limiter la colinearite parfaite.
    cat_cols = [col for col in CATEGORICAL_COLUMNS if col in model_df.columns]
    model_df = pd.get_dummies(model_df, columns=cat_cols, drop_first=True)

    X = model_df.drop(columns=["price"])
    y = model_df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)

    print(f"{label} - Regression lineaire multiple R2: {r2:.4f}")

    # Visualisation: projection (reel vs predit) pour evaluer la qualite du modele.
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_test, preds, alpha=0.4)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    ax.set_title(f"{label} - Regression multiple (Reel vs predit)")
    ax.set_xlabel("Prix reel")
    ax.set_ylabel("Prix predit")
    save_figure(fig, f"multiple_regression_{label.lower()}.png")
    plt.close(fig)


def main():
    # Point d'entree du script: enchaine toutes les etapes du pipeline.
    ensure_dirs()

    # 1) Charger les donnees
    lyon_raw, paris_raw = load_data()

    # 2) Pretraitement des donnees
    lyon = preprocess(lyon_raw, "Lyon")
    paris = preprocess(paris_raw, "Paris")

    # 3) Analyse descriptive
    descriptive_analysis(lyon, paris)

    # 4) Valeurs aberrantes et mise a l'echelle
    lyon, paris = outlier_and_scaling(lyon, paris)

    # 5) Separation train/test + 6) Modelisation
    linear_regression_simple(lyon, "Lyon")
    linear_regression_multiple(lyon, "Lyon")
    linear_regression_simple(paris, "Paris")
    linear_regression_multiple(paris, "Paris")

    print("\nPipeline termine.")


if __name__ == "__main__":
    main()
