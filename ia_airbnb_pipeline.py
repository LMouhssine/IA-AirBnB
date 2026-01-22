import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score


DATA_LYON_PATH = "listingsLyon.csv"
DATA_PARIS_PATH = "listingsParis.csv"
OUTPUT_DIR = "outputs"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")


RELEVANT_COLUMNS = [
    # Target
    "price",
    # Location
    "latitude",
    "longitude",
    "neighbourhood_cleansed",
    # Property and room
    "property_type",
    "room_type",
    # Capacity
    "accommodates",
    "bathrooms_text",
    "bedrooms",
    "beds",
    # Availability
    "has_availability",
    "availability_30",
    "availability_60",
    "availability_90",
    "availability_365",
    "minimum_nights",
    "maximum_nights",
    # Reviews
    "number_of_reviews",
    "reviews_per_month",
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value",
    # Booking behavior
    "instant_bookable",
    "host_is_superhost",
]


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


CATEGORICAL_COLUMNS = [
    "neighbourhood_cleansed",
    "property_type",
    "room_type",
]


def ensure_dirs():
    os.makedirs(FIG_DIR, exist_ok=True)


def load_data():
    lyon = pd.read_csv(DATA_LYON_PATH)
    paris = pd.read_csv(DATA_PARIS_PATH)

    print("=== Data loading ===")
    print(f"Lyon shape: {lyon.shape}")
    print(f"Paris shape: {paris.shape}")
    print(f"Same columns: {list(lyon.columns) == list(paris.columns)}")
    print("====================\n")

    return lyon, paris


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in RELEVANT_COLUMNS if col in df.columns]
    return df[cols].copy()


def drop_duplicates(df: pd.DataFrame, label: str) -> pd.DataFrame:
    dup_count = df.duplicated().sum()
    print(f"{label}: duplicates before drop = {dup_count}")
    return df.drop_duplicates()


def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    # Convert price to float by removing currency symbols and commas
    if "price" in df.columns:
        df["price"] = pd.to_numeric(
            df["price"].replace(r"[^0-9.]", "", regex=True), errors="coerce"
        )

    # Extract numeric part from bathrooms_text
    if "bathrooms_text" in df.columns:
        df["bathrooms_text"] = pd.to_numeric(
            df["bathrooms_text"].astype(str).str.extract(r"([0-9]*\.?[0-9]+)")[0],
            errors="coerce",
        )

    # Coerce numeric columns that may be stored as strings
    for col in NUMERIC_COERCE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Binary columns (t/f -> 1/0)
    for col in ["has_availability", "instant_bookable", "host_is_superhost"]:
        if col in df.columns:
            df[col] = df[col].map({"t": 1, "f": 0})

    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    # Fill numeric missing values with column mean
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Fill categorical missing values with a placeholder
    categorical_cols = df.select_dtypes(include=["object"]).columns
    if len(categorical_cols) > 0:
        df[categorical_cols] = df[categorical_cols].fillna("Unknown")

    # Convert bathrooms_text to integer (assumption: round after mean imputation)
    if "bathrooms_text" in df.columns:
        df["bathrooms_text"] = np.ceil(df["bathrooms_text"]).astype(int)

    return df


def recode_rare_categories(df: pd.DataFrame, column: str, min_freq: float = 0.01) -> pd.DataFrame:
    # Assumption: group rare categories to reduce sparsity before one-hot encoding.
    if column in df.columns:
        freq = df[column].value_counts(normalize=True)
        rare = freq[freq < min_freq].index
        df[column] = df[column].where(~df[column].isin(rare), other="Other")
    return df


def preprocess(df: pd.DataFrame, label: str) -> pd.DataFrame:
    df = select_columns(df)
    df = drop_duplicates(df, label)
    df = convert_types(df)
    df = fill_missing_values(df)

    # Recoding step (explicitly required by the reference document)
    df = recode_rare_categories(df, "property_type", min_freq=0.01)
    df = recode_rare_categories(df, "neighbourhood_cleansed", min_freq=0.01)

    return df


def save_figure(fig, filename: str):
    path = os.path.join(FIG_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved figure: {path}")


def descriptive_analysis(lyon: pd.DataFrame, paris: pd.DataFrame):
    sns.set_theme(style="whitegrid")
    combined = pd.concat(
        [lyon.assign(city="Lyon"), paris.assign(city="Paris")], ignore_index=True
    )

    # 1) Relationship between review_scores_rating and price
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    sns.scatterplot(
        data=lyon, x="review_scores_rating", y="price", alpha=0.4, ax=axes[0]
    )
    axes[0].set_title("Lyon: rating vs price")
    sns.scatterplot(
        data=paris, x="review_scores_rating", y="price", alpha=0.4, ax=axes[1]
    )
    axes[1].set_title("Paris: rating vs price")
    fig.suptitle("Review score vs price")
    save_figure(fig, "rating_vs_price.png")
    plt.close(fig)

    # 2) Mean price by property type (top 10 for readability)
    top_props = combined["property_type"].value_counts().head(10).index
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(
        data=combined[combined["property_type"].isin(top_props)],
        x="property_type",
        y="price",
        hue="city",
        ax=ax,
    )
    ax.set_title("Mean price by property type (top 10)")
    ax.set_xlabel("Property type")
    ax.set_ylabel("Mean price")
    plt.xticks(rotation=30, ha="right")
    save_figure(fig, "mean_price_by_property_type.png")
    plt.close(fig)

    # 3) Relationship between availability and price
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(
        data=combined,
        x="has_availability",
        y="price",
        hue="city",
        ax=ax,
    )
    ax.set_title("Availability vs price")
    ax.set_xlabel("Has availability (0/1)")
    ax.set_ylabel("Price")
    save_figure(fig, "availability_vs_price.png")
    plt.close(fig)

    # 4) Distribution of available / not available listings
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=combined, x="has_availability", hue="city", ax=ax)
    ax.set_title("Availability distribution")
    ax.set_xlabel("Has availability (0/1)")
    ax.set_ylabel("Count")
    save_figure(fig, "availability_distribution.png")
    plt.close(fig)

    # 5) Price distribution by room type
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(
        data=combined,
        x="room_type",
        y="price",
        hue="city",
        ax=ax,
    )
    ax.set_title("Price distribution by room type")
    ax.set_xlabel("Room type")
    ax.set_ylabel("Price")
    plt.xticks(rotation=20, ha="right")
    save_figure(fig, "price_by_room_type.png")
    plt.close(fig)


def outlier_and_scaling(lyon: pd.DataFrame, paris: pd.DataFrame):
    # Boxplots for price outliers
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    sns.boxplot(y=lyon["price"], ax=axes[0])
    axes[0].set_title("Lyon price outliers")
    sns.boxplot(y=paris["price"], ax=axes[1])
    axes[1].set_title("Paris price outliers")
    save_figure(fig, "price_outliers_boxplot.png")
    plt.close(fig)

    # MinMax scaling for price (separately per city)
    scaler_lyon = MinMaxScaler()
    scaler_paris = MinMaxScaler()
    lyon["price_scaled"] = scaler_lyon.fit_transform(lyon[["price"]])
    paris["price_scaled"] = scaler_paris.fit_transform(paris[["price"]])

    return lyon, paris


def train_test_split_data(df: pd.DataFrame, features: list, target: str = "price"):
    X = df[features]
    y = df[target]
    return train_test_split(X, y, test_size=0.4, random_state=42)


def linear_regression_simple(df: pd.DataFrame, label: str):
    # Simple linear regression with one explanatory variable
    X_train, X_test, y_train, y_test = train_test_split_data(df, ["accommodates"])

    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)

    print(f"{label} - Simple Linear Regression R2: {r2:.4f}")

    # Plot regression line
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(X_test["accommodates"], y_test, alpha=0.4, label="Actual")
    x_line = np.linspace(df["accommodates"].min(), df["accommodates"].max(), 100)
    x_line_df = pd.DataFrame({"accommodates": x_line})
    y_line = model.predict(x_line_df)
    ax.plot(x_line, y_line, color="red", label="Regression line")
    ax.set_title(f"{label} - Simple Linear Regression")
    ax.set_xlabel("Accommodates")
    ax.set_ylabel("Price")
    ax.legend()
    save_figure(fig, f"simple_regression_{label.lower()}.png")
    plt.close(fig)


def linear_regression_multiple(df: pd.DataFrame, label: str):
    # Multiple linear regression with numeric + categorical features
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

    # Keep only columns that exist
    feature_cols = [col for col in feature_cols if col in df.columns]
    model_df = df[feature_cols + ["price"]].copy()

    # One-hot encode categorical columns
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

    print(f"{label} - Multiple Linear Regression R2: {r2:.4f}")

    # Visualization: projection (actual vs predicted)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_test, preds, alpha=0.4)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    ax.set_title(f"{label} - Multiple Regression (Actual vs Predicted)")
    ax.set_xlabel("Actual price")
    ax.set_ylabel("Predicted price")
    save_figure(fig, f"multiple_regression_{label.lower()}.png")
    plt.close(fig)


def main():
    ensure_dirs()

    # 1) Load data
    lyon_raw, paris_raw = load_data()

    # 2) Preprocess data
    lyon = preprocess(lyon_raw, "Lyon")
    paris = preprocess(paris_raw, "Paris")

    # 3) Descriptive analysis
    descriptive_analysis(lyon, paris)

    # 4) Outliers and scaling
    lyon, paris = outlier_and_scaling(lyon, paris)

    # 5) Train/test split + 6) Modeling
    linear_regression_simple(lyon, "Lyon")
    linear_regression_multiple(lyon, "Lyon")
    linear_regression_simple(paris, "Paris")
    linear_regression_multiple(paris, "Paris")

    print("\nPipeline completed.")


if __name__ == "__main__":
    main()
