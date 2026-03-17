from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "Churn_Modelling.csv"
ARTIFACT_DIR = ROOT_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "churn_model.joblib"

TARGET_COLUMN = "Exited"
EXCLUDED_COLUMNS = ["RowNumber", "CustomerId", "Surname"]
NUMERIC_COLUMNS = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]
CATEGORICAL_COLUMNS = ["Geography", "Gender"]
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

MODEL_NAME = "Random Forest Churn Classifier"
DEFAULT_THRESHOLD = 0.50


def load_dataset(path: Path | str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def get_default_input(data: pd.DataFrame) -> dict[str, Any]:
    return {
        "CreditScore": int(data["CreditScore"].median()),
        "Geography": str(data["Geography"].mode().iloc[0]),
        "Gender": str(data["Gender"].mode().iloc[0]),
        "Age": int(data["Age"].median()),
        "Tenure": int(data["Tenure"].median()),
        "Balance": float(data["Balance"].median()),
        "NumOfProducts": int(data["NumOfProducts"].mode().iloc[0]),
        "HasCrCard": int(data["HasCrCard"].mode().iloc[0]),
        "IsActiveMember": int(data["IsActiveMember"].mode().iloc[0]),
        "EstimatedSalary": float(data["EstimatedSalary"].median()),
    }


def prepare_training_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    features = data.drop(columns=[TARGET_COLUMN, *EXCLUDED_COLUMNS], errors="ignore")
    target = data[TARGET_COLUMN].astype(int)
    return features[FEATURE_COLUMNS].copy(), target


def build_model_pipeline() -> Pipeline:
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numerical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                NUMERIC_COLUMNS,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                drop="first",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                CATEGORICAL_COLUMNS,
            ),
        ],
        verbose_feature_names_out=False,
    )

    classifier = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=4,
        min_samples_split=8,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessing),
            ("model", classifier),
        ]
    )


def _format_feature_name(name: str) -> str:
    if name.startswith("Geography_"):
        return f"Geography: {name.split('_', 1)[1]}"
    if name.startswith("Gender_"):
        return f"Gender: {name.split('_', 1)[1]}"
    return name


def _top_feature_importance(pipeline: Pipeline, top_n: int = 10) -> list[dict[str, float | str]]:
    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    feature_names = preprocess.get_feature_names_out()
    importances = model.feature_importances_
    pairs = sorted(
        zip(feature_names, importances),
        key=lambda item: item[1],
        reverse=True,
    )[:top_n]
    return [
        {"feature": _format_feature_name(name), "importance": float(score)}
        for name, score in pairs
    ]


def _safe_probability(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(features)[:, 1]


def train_bundle(data_path: Path | str = DATA_PATH) -> dict[str, Any]:
    data = load_dataset(data_path)
    features, target = prepare_training_data(data)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
        stratify=target,
    )

    model = build_model_pipeline()
    model.fit(X_train, y_train)

    holdout_probability = _safe_probability(model, X_test)
    holdout_prediction = (holdout_probability >= DEFAULT_THRESHOLD).astype(int)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        build_model_pipeline(),
        features,
        target,
        scoring="roc_auc",
        cv=cv,
        n_jobs=None,
    )

    bundle = {
        "model_name": MODEL_NAME,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": model,
        "metrics": {
            "accuracy": float(accuracy_score(y_test, holdout_prediction)),
            "precision": float(precision_score(y_test, holdout_prediction)),
            "recall": float(recall_score(y_test, holdout_prediction)),
            "f1": float(f1_score(y_test, holdout_prediction)),
            "roc_auc": float(roc_auc_score(y_test, holdout_probability)),
            "cv_roc_auc_mean": float(cv_scores.mean()),
            "cv_roc_auc_std": float(cv_scores.std()),
            "threshold": DEFAULT_THRESHOLD,
        },
        "feature_importance": _top_feature_importance(model),
        "dataset_summary": {
            "rows": int(len(data)),
            "churn_rate": float(target.mean()),
            "geographies": sorted(data["Geography"].dropna().astype(str).unique().tolist()),
            "genders": sorted(data["Gender"].dropna().astype(str).unique().tolist()),
        },
        "feature_columns": FEATURE_COLUMNS,
        "input_defaults": get_default_input(data),
    }
    return bundle


def save_bundle(bundle: dict[str, Any], path: Path | str = MODEL_PATH) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destination)
    return destination


def train_and_persist(
    data_path: Path | str = DATA_PATH,
    model_path: Path | str = MODEL_PATH,
) -> dict[str, Any]:
    bundle = train_bundle(data_path)
    save_bundle(bundle, model_path)
    return bundle


def load_bundle(path: Path | str = MODEL_PATH) -> dict[str, Any]:
    return joblib.load(path)


def _normalize_binary(value: Any) -> int:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "active"}:
            return 1
        if normalized in {"0", "false", "no", "n", "inactive"}:
            return 0
    if pd.isna(value):
        raise ValueError("Binary value is missing.")
    return int(value)


def prepare_inference_frame(records: pd.DataFrame) -> pd.DataFrame:
    frame = records.copy()
    frame = frame.drop(columns=[TARGET_COLUMN, *EXCLUDED_COLUMNS], errors="ignore")

    missing_columns = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing_columns:
        missing_display = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns: {missing_display}")

    frame = frame[FEATURE_COLUMNS].copy()
    for column in NUMERIC_COLUMNS:
        if column in {"HasCrCard", "IsActiveMember"}:
            frame[column] = frame[column].apply(_normalize_binary)
        else:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["Geography"] = frame["Geography"].astype(str).str.strip().str.title()
    frame["Gender"] = frame["Gender"].astype(str).str.strip().str.title()

    if frame.isna().any().any():
        invalid_columns = frame.columns[frame.isna().any()].tolist()
        invalid_display = ", ".join(invalid_columns)
        raise ValueError(f"Could not parse values for: {invalid_display}")

    return frame


def risk_band(probability: float) -> str:
    if probability >= 0.70:
        return "High"
    if probability >= 0.40:
        return "Medium"
    return "Low"


def score_records(
    records: pd.DataFrame,
    bundle: dict[str, Any],
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    prepared = prepare_inference_frame(records)
    probabilities = _safe_probability(bundle["pipeline"], prepared)
    predictions = (probabilities >= threshold).astype(int)

    scored = records.copy().reset_index(drop=True)
    scored["churn_probability"] = np.round(probabilities, 4)
    scored["risk_band"] = [risk_band(value) for value in probabilities]
    scored["prediction"] = np.where(predictions == 1, "Likely to churn", "Likely to stay")
    return scored
