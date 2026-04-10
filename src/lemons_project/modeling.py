from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    FIGURES_DIR,
    INFERENCE_SAMPLE_SIZE,
    LEXICON_COLUMNS,
    RANDOM_STATE,
    SAMPLE_SIZE,
    STRUCTURED_CATEGORICAL_COLUMNS,
    STRUCTURED_NUMERIC_COLUMNS,
    SUMMARY_PATH,
    TABLES_DIR,
)
from .features import NORMALIZED_STOPWORDS

matplotlib.use("Agg")
sns.set_theme(style="whitegrid")


def _ensure_output_dirs() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)


def _save_table(df: pd.DataFrame, stem: str, float_format: str = "%.4f") -> None:
    csv_path = TABLES_DIR / f"{stem}.csv"
    tex_path = TABLES_DIR / f"{stem}.tex"
    df.to_csv(csv_path, index=False)
    df.to_latex(tex_path, index=False, float_format=float_format.__mod__)


def evaluate_predictions(name: str, y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float | str]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"model": name, "rmse_log_price": rmse, "mae_log_price": mae, "r2": r2}


def build_structured_pipeline(feature_columns: list[str]) -> Pipeline:
    numeric_columns = [column for column in feature_columns if column not in STRUCTURED_CATEGORICAL_COLUMNS]
    categorical_columns = [column for column in feature_columns if column in STRUCTURED_CATEGORICAL_COLUMNS]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ],
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("regressor", Ridge(alpha=2.5)),
        ]
    )


def run_analysis(df: pd.DataFrame) -> dict[str, object]:
    _ensure_output_dirs()

    working = df.copy()
    sample_n = min(SAMPLE_SIZE, len(working))
    model_sample = working.sample(sample_n, random_state=RANDOM_STATE).reset_index(drop=True)

    train_df, test_df = train_test_split(model_sample, test_size=0.2, random_state=RANDOM_STATE)

    structured_columns = STRUCTURED_NUMERIC_COLUMNS + STRUCTURED_CATEGORICAL_COLUMNS
    lexicon_columns = structured_columns + LEXICON_COLUMNS

    structured_model = build_structured_pipeline(structured_columns)
    structured_model.fit(train_df[structured_columns], train_df["log_price"])
    structured_pred = structured_model.predict(test_df[structured_columns])

    lexicon_model = build_structured_pipeline(lexicon_columns)
    lexicon_model.fit(train_df[lexicon_columns], train_df["log_price"])
    lexicon_pred = lexicon_model.predict(test_df[lexicon_columns])

    train_residual = train_df["log_price"] - structured_model.predict(train_df[structured_columns])
    test_base = structured_model.predict(test_df[structured_columns])

    vectorizer = TfidfVectorizer(
        lowercase=False,
        preprocessor=None,
        token_pattern=r"(?u)\b[a-z0-9]{2,}\b",
        stop_words=NORMALIZED_STOPWORDS,
        ngram_range=(1, 2),
        min_df=20,
        max_features=15000,
    )
    text_train = vectorizer.fit_transform(train_df["listing_text_clean"])
    text_test = vectorizer.transform(test_df["listing_text_clean"])

    residual_model = SGDRegressor(
        loss="squared_error",
        penalty="elasticnet",
        alpha=1e-5,
        l1_ratio=0.15,
        max_iter=2000,
        random_state=RANDOM_STATE,
    )
    residual_model.fit(text_train, train_residual)
    hybrid_pred = test_base + residual_model.predict(text_test)

    metrics = pd.DataFrame(
        [
            evaluate_predictions("Structured baseline", test_df["log_price"], structured_pred),
            evaluate_predictions("Structured + disclosure lexicon", test_df["log_price"], lexicon_pred),
            evaluate_predictions("Structured + TF-IDF residual text", test_df["log_price"], hybrid_pred),
        ]
    )
    _save_table(metrics, "model_metrics")

    feature_names = np.array(vectorizer.get_feature_names_out())
    coefficients = residual_model.coef_
    positive_idx = coefficients.argsort()[-20:][::-1]
    negative_idx = coefficients.argsort()[:20]
    top_terms = pd.concat(
        [
            pd.DataFrame(
                {"term": feature_names[positive_idx], "coefficient": coefficients[positive_idx], "direction": "premium"}
            ),
            pd.DataFrame(
                {"term": feature_names[negative_idx], "coefficient": coefficients[negative_idx], "direction": "discount"}
            ),
        ],
        ignore_index=True,
    )
    _save_table(top_terms, "top_price_terms")

    inference_n = min(INFERENCE_SAMPLE_SIZE, len(working))
    inference_df = working.sample(inference_n, random_state=RANDOM_STATE).copy()
    inference_df["is_private"] = (inference_df["advertizer_type_clean"] == "Particular").astype(int)

    formula = (
        "log_price ~ disclosure_index_z * is_private + log_km + vehicle_age + power_cv + C(region)"
    )
    regression = smf.ols(formula=formula, data=inference_df).fit(cov_type="HC3")
    target_terms = pd.DataFrame(
        {
            "term": regression.params.index,
            "coefficient": regression.params.values,
            "std_error": regression.bse.values,
            "p_value": regression.pvalues.values,
        }
    )
    key_terms = target_terms[target_terms["term"].isin(["Intercept", "disclosure_index_z", "is_private", "disclosure_index_z:is_private"])].copy()
    _save_table(key_terms, "interaction_regression_key_terms")

    descriptive = pd.DataFrame(
        [
            {
                "n_listings": int(len(working)),
                "mean_price_eur": float(working["price_eur"].mean()),
                "median_price_eur": float(working["price_eur"].median()),
                "mean_km": float(working["km"].mean()),
                "mean_vehicle_age": float(working["vehicle_age"].mean()),
                "private_share": float((working["advertizer_type_clean"] == "Particular").mean()),
                "mean_disclosure_index": float(working["disclosure_index_z"].mean()),
            }
        ]
    )
    _save_table(descriptive, "descriptive_stats")

    seller_summary = (
        working.groupby("advertizer_type_clean", observed=False)
        .agg(
            mean_disclosure_index=("disclosure_index_z", "mean"),
            median_price_eur=("price_eur", "median"),
            listings=("ad_id", "count"),
        )
        .reset_index()
    )
    _save_table(seller_summary, "seller_summary")

    chart_df = test_df.copy()
    chart_df["baseline_residual"] = test_df["log_price"] - test_base
    chart_df["disclosure_quintile"] = pd.qcut(chart_df["disclosure_index_z"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])

    plt.figure(figsize=(7, 4.5))
    sns.boxplot(data=working.sample(min(20000, len(working)), random_state=RANDOM_STATE), x="advertizer_type_clean", y="disclosure_index_z")
    plt.xlabel("Seller type")
    plt.ylabel("Disclosure index (z-score)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "disclosure_by_seller.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    residual_quintiles = (
        chart_df.groupby(["disclosure_quintile", "advertizer_type_clean"], observed=False)["baseline_residual"]
        .mean()
        .reset_index()
    )
    sns.lineplot(data=residual_quintiles, x="disclosure_quintile", y="baseline_residual", hue="advertizer_type_clean", marker="o")
    plt.axhline(0.0, color="black", linewidth=1, linestyle="--")
    plt.xlabel("Disclosure quintile")
    plt.ylabel("Average log-price residual")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "residuals_by_disclosure_quintile.png", dpi=200)
    plt.close()

    premium_terms = top_terms[top_terms["direction"] == "premium"].head(10).iloc[::-1]
    discount_terms = top_terms[top_terms["direction"] == "discount"].head(10).iloc[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=False)
    axes[0].barh(premium_terms["term"], premium_terms["coefficient"], color="#2a9d8f")
    axes[0].set_title("Top premium terms")
    axes[1].barh(discount_terms["term"], discount_terms["coefficient"], color="#e76f51")
    axes[1].set_title("Top discount terms")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "top_price_terms.png", dpi=200)
    plt.close(fig)

    summary = {
        "dataset_rows_after_cleaning": int(len(working)),
        "model_sample_size": int(sample_n),
        "inference_sample_size": int(inference_n),
        "model_metrics": metrics.to_dict(orient="records"),
        "key_regression_terms": key_terms.to_dict(orient="records"),
        "top_premium_terms": top_terms[top_terms["direction"] == "premium"].head(10).to_dict(orient="records"),
        "top_discount_terms": top_terms[top_terms["direction"] == "discount"].head(10).to_dict(orient="records"),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary
