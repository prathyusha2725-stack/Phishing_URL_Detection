
from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = (
    BASE_DIR
    / "dataset"
    / "PhiUSIIL_Phishing_URL_Dataset.csv"
)
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "phishing_model.joblib"


def main():
    print("Loading dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    # Normalize column names to handle extra spaces.
    df.columns = df.columns.str.strip()

    required_columns = {"URL", "label"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            "Expected columns 'URL' and 'label'. "
            f"Found: {df.columns.tolist()}"
        )

    # Remove missing URLs and invalid labels.
    df = df[["URL", "label"]].dropna()
    df["URL"] = df["URL"].astype(str).str.strip()
    df = df[df["URL"] != ""]

    # Dataset convention: 1 = legitimate, 0 = phishing.
    df["label"] = pd.to_numeric(
        df["label"], errors="coerce"
    )
    df = df[df["label"].isin([0, 1])]
    df["label"] = df["label"].astype(int)

    if df["label"].nunique() != 2:
        raise ValueError(
            "Dataset must contain both labels 0 and 1."
        )

    # X contains raw URLs; y contains their labels.
    X = df["URL"]
    y = df["label"]

    print(f"Usable records: {len(df)}")
    print("Label counts:")
    print(y.value_counts().sort_index())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=(3, 5),
                max_features=100000,
                sublinear_tf=True,
                dtype=float,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])

    print("Training model...")
    model.fit(X_train, y_train)

    print("Evaluating model...")
    predictions = model.predict(X_test)

    print(
        "Test accuracy:",
        round(accuracy_score(y_test, predictions), 4),
    )
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Phishing", "Legitimate"],
            zero_division=0,
        )
    )
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
