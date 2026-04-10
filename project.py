import json
import re
import unicodedata
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from spacy.lang.es.stop_words import STOP_WORDS


matplotlib.use("Agg")
sns.set_theme(style="whitegrid")


DATA_URL = "https://zenodo.org/records/4252636/files/dataset.csv?download=1"
RANDOM_STATE = 42
SAMPLE_SIZE = 60000
INFERENCE_SAMPLE_SIZE = 40000

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw"
DATA_PATH = DATA_DIR / "dataset.csv"
OUTPUTS_DIR = ROOT / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
FIGURES_DIR = OUTPUTS_DIR / "figures"
SUMMARY_PATH = OUTPUTS_DIR / "summary.json"

CATEGORICAL_COLUMNS = ["advertizer_type_clean", "region", "brand"]
LEXICON_COLUMNS = [
    "maintenance_mentions",
    "transparency_mentions",
    "defect_mentions",
    "promotional_mentions",
    "token_count",
    "number_count",
    "disclosure_index_z",
]


MAINTENANCE_PATTERNS = [
    r"\blibro de mantenimiento\b",
    r"\bhistorial de mantenimiento\b",
    r"\brevisiones? al dia\b",
    r"\bmantenimiento al dia\b",
    r"\bitv (recien )?pasad[ao]s?\b",
    r"\bcorrea de distribucion\b",
    r"\bdistribucion recien cambiada\b",
    r"\baceite (recien )?cambiad[oa]\b",
    r"\bembrague (nuevo|cambiad[oa])\b",
    r"\bru(ed|ed)as nuevas\b",
    r"\bneumaticos nuevos\b",
    r"\bfacturas?\b",
]

TRANSPARENCY_PATTERNS = [
    r"\bgarantia\b",
    r"\bse acepta prueba mecanica\b",
    r"\bprueba mecanica\b",
    r"\bkilometros reales\b",
    r"\bkm reales\b",
    r"\bunico dueno\b",
    r"\bun solo propietario\b",
    r"\bnacional\b",
    r"\btransferencia incluida\b",
    r"\bprecio transparente\b",
    r"\btodo incluido\b",
]

DEFECT_PATTERNS = [
    r"\brasgu[nn]o\b",
    r"\baranazo\b",
    r"\bgolpe\b",
    r"\baveria\b",
    r"\bfall[ao]\b",
    r"\broce\b",
    r"\breparar\b",
    r"\bdetalles?\b",
    r"\bnecesita\b",
    r"\bcambiar\b",
    r"\bsin garantia\b",
]

PROMOTIONAL_PATTERNS = [
    r"\boportunidad\b",
    r"\bimpecable\b",
    r"\bcomo nuevo\b",
    r"\bmejor ver\b",
    r"\burge\b",
    r"\burg[e]? venta\b",
    r"\bganga\b",
    r"\bestado perfecto\b",
    r"\bmejor precio\b",
    r"\bprecio negociable\b",
    r"\bperfecto estado\b",
]


def fix_mojibake(text):
    text = str(text or "")
    if not text:
        return ""

    if any(marker in text for marker in ("Ã", "Â", "Ð", "¤", "�")):
        try:
            repaired = text.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
            if repaired.count("Ã") + repaired.count("Â") <= text.count("Ã") + text.count("Â"):
                text = repaired
        except UnicodeError:
            pass

    return unicodedata.normalize("NFKC", text)


def strip_accents(text):
    text = unicodedata.normalize("NFKD", text)
    return "".join(char for char in text if not unicodedata.combining(char))


def normalize_text(text):
    text = fix_mojibake(text)
    text = text.replace("Leer más", " ")
    text = text.replace("leer más", " ")
    text = text.lower()
    text = strip_accents(text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_number(value):
    text = fix_mojibake(value).replace(".", "")
    match = re.search(r"(\d+)", text)
    return float(match.group(1)) if match else np.nan


def count_patterns(text, patterns):
    return int(sum(len(re.findall(pattern, text)) for pattern in patterns))


def save_table(df, name):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES_DIR / f"{name}.csv", index=False)
    df.to_latex(TABLES_DIR / f"{name}.tex", index=False, float_format="%.4f")


def prepare_model_data(train_df, test_df, feature_columns):
    numeric_columns = [column for column in feature_columns if column not in CATEGORICAL_COLUMNS]

    x_train = train_df[feature_columns].copy()
    x_test = test_df[feature_columns].copy()

    for column in numeric_columns:
        median = x_train[column].median()
        mean = x_train[column].mean()
        std = x_train[column].std(ddof=0)
        if pd.isna(std) or std == 0:
            std = 1.0
        x_train[column] = x_train[column].fillna(median)
        x_test[column] = x_test[column].fillna(median)
        x_train[column] = (x_train[column] - mean) / std
        x_test[column] = (x_test[column] - mean) / std

    for column in CATEGORICAL_COLUMNS:
        if column in x_train.columns:
            mode = x_train[column].mode(dropna=True)
            fill_value = mode.iat[0] if not mode.empty else "desconocida"
            x_train[column] = x_train[column].fillna(fill_value)
            x_test[column] = x_test[column].fillna(fill_value)

    x_train = pd.get_dummies(x_train, columns=[column for column in CATEGORICAL_COLUMNS if column in x_train.columns])
    x_test = pd.get_dummies(x_test, columns=[column for column in CATEGORICAL_COLUMNS if column in x_test.columns])
    x_test = x_test.reindex(columns=x_train.columns, fill_value=0)

    return x_train, x_test


def evaluate_predictions(name, y_true, y_pred):
    return {
        "model": name,
        "rmse_log_price": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae_log_price": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def download_dataset_if_needed():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.exists():
        print(f"Dataset already exists at {DATA_PATH}")
        return

    print("Downloading dataset...")
    urlretrieve(DATA_URL, DATA_PATH)
    print(f"Saved dataset to {DATA_PATH}")


def load_and_clean_data():
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={"Unnamed: 0": "row_id"})
    df = df.drop_duplicates(subset=["ad_id"])

    df["advertizer_type_clean"] = df["advertizer_type"].where(
        df["advertizer_type"].isin(["Profesional", "Particular"])
    )
    df = df.dropna(subset=["advertizer_type_clean", "car_desc", "car_price", "car_km", "car_year"])

    df["price_eur"] = df["car_price"].map(extract_number)
    df["km"] = df["car_km"].map(extract_number)
    df["registration_year"] = df["car_year"].map(extract_number)
    df["power_cv"] = df["car_power"].map(extract_number)
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")

    snapshot_year = int(df["ts"].dt.year.dropna().mode().iat[0])
    df["vehicle_age"] = snapshot_year - df["registration_year"]

    df["ad_title"] = df["ad_title"].fillna("").map(fix_mojibake)
    df["car_desc"] = df["car_desc"].fillna("").map(fix_mojibake)
    df["brand"] = df["ad_title"].str.split("-").str[0].fillna("").map(normalize_text).replace("", "desconocida")
    df["listing_text"] = df["car_desc"].str.strip()
    df["listing_text_clean"] = df["listing_text"].map(normalize_text)
    df["region"] = df["region"].fillna("desconocida").map(normalize_text)

    df = df[
        df["price_eur"].between(500, 100000)
        & df["km"].between(0, 500000)
        & df["registration_year"].between(1990, snapshot_year)
        & df["vehicle_age"].between(0, 35)
        & df["power_cv"].between(40, 500)
        & df["listing_text_clean"].str.len().ge(20)
    ].copy()

    df["log_km"] = np.log1p(df["km"])
    df["log_price"] = np.log(df["price_eur"])
    df = df.reset_index(drop=True)

    return df


def add_disclosure_features(df):
    df = df.copy()

    df["token_count"] = df["listing_text_clean"].str.split().str.len()
    df["number_count"] = df["listing_text_clean"].str.count(r"\b\d+\b")
    df["maintenance_mentions"] = df["listing_text_clean"].map(lambda text: count_patterns(text, MAINTENANCE_PATTERNS))
    df["transparency_mentions"] = df["listing_text_clean"].map(lambda text: count_patterns(text, TRANSPARENCY_PATTERNS))
    df["defect_mentions"] = df["listing_text_clean"].map(lambda text: count_patterns(text, DEFECT_PATTERNS))
    df["promotional_mentions"] = df["listing_text_clean"].map(lambda text: count_patterns(text, PROMOTIONAL_PATTERNS))

    score = (
        1.2 * df["maintenance_mentions"]
        + 1.4 * df["transparency_mentions"]
        + 1.0 * df["defect_mentions"]
        + 0.08 * df["number_count"]
        + 0.01 * df["token_count"]
        - 1.1 * df["promotional_mentions"]
    )

    df["disclosure_index_raw"] = score
    df["disclosure_index_z"] = (score - score.mean()) / score.std(ddof=0)

    return df


def run_analysis(df):
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    sample_n = min(SAMPLE_SIZE, len(df))
    model_df = df.sample(sample_n, random_state=RANDOM_STATE).reset_index(drop=True)
    train_df, test_df = train_test_split(model_df, test_size=0.2, random_state=RANDOM_STATE)

    structured_columns = ["log_km", "vehicle_age", "power_cv", "advertizer_type_clean", "region", "brand"]
    structured_plus_text_columns = structured_columns + LEXICON_COLUMNS

    x_train_base, x_test_base = prepare_model_data(train_df, test_df, structured_columns)
    baseline_model = Ridge(alpha=2.5)
    baseline_model.fit(x_train_base, train_df["log_price"])
    baseline_pred = baseline_model.predict(x_test_base)

    x_train_lex, x_test_lex = prepare_model_data(train_df, test_df, structured_plus_text_columns)
    disclosure_model = Ridge(alpha=2.5)
    disclosure_model.fit(x_train_lex, train_df["log_price"])
    disclosure_pred = disclosure_model.predict(x_test_lex)

    normalized_stopwords = sorted({normalize_text(word) for word in STOP_WORDS if normalize_text(word)})
    vectorizer = TfidfVectorizer(
        lowercase=False,
        preprocessor=None,
        token_pattern=r"(?u)\b[a-z0-9]{2,}\b",
        stop_words=normalized_stopwords,
        ngram_range=(1, 2),
        min_df=20,
        max_features=15000,
    )

    train_text = vectorizer.fit_transform(train_df["listing_text_clean"])
    test_text = vectorizer.transform(test_df["listing_text_clean"])

    residual_train = train_df["log_price"] - baseline_model.predict(x_train_base)
    residual_model = SGDRegressor(
        loss="squared_error",
        penalty="elasticnet",
        alpha=1e-5,
        l1_ratio=0.15,
        max_iter=2000,
        random_state=RANDOM_STATE,
    )
    residual_model.fit(train_text, residual_train)
    hybrid_pred = baseline_model.predict(x_test_base) + residual_model.predict(test_text)

    metrics = pd.DataFrame(
        [
            evaluate_predictions("Structured baseline", test_df["log_price"], baseline_pred),
            evaluate_predictions("Structured + disclosure lexicon", test_df["log_price"], disclosure_pred),
            evaluate_predictions("Structured + TF-IDF residual text", test_df["log_price"], hybrid_pred),
        ]
    )
    save_table(metrics, "model_metrics")

    feature_names = np.array(vectorizer.get_feature_names_out())
    coefficients = residual_model.coef_
    top_positive = coefficients.argsort()[-20:][::-1]
    top_negative = coefficients.argsort()[:20]
    top_terms = pd.concat(
        [
            pd.DataFrame({"term": feature_names[top_positive], "coefficient": coefficients[top_positive], "direction": "premium"}),
            pd.DataFrame({"term": feature_names[top_negative], "coefficient": coefficients[top_negative], "direction": "discount"}),
        ],
        ignore_index=True,
    )
    save_table(top_terms, "top_price_terms")

    inference_n = min(INFERENCE_SAMPLE_SIZE, len(df))
    inference_df = df.sample(inference_n, random_state=RANDOM_STATE).copy()
    inference_df["is_private"] = (inference_df["advertizer_type_clean"] == "Particular").astype(int)

    regression = smf.ols(
        formula="log_price ~ disclosure_index_z * is_private + log_km + vehicle_age + power_cv + C(region)",
        data=inference_df,
    ).fit(cov_type="HC3")

    regression_table = pd.DataFrame(
        {
            "term": regression.params.index,
            "coefficient": regression.params.values,
            "std_error": regression.bse.values,
            "p_value": regression.pvalues.values,
        }
    )
    key_terms = regression_table[
        regression_table["term"].isin(["Intercept", "disclosure_index_z", "is_private", "disclosure_index_z:is_private"])
    ].copy()
    save_table(key_terms, "interaction_regression_key_terms")

    descriptive_stats = pd.DataFrame(
        [
            {
                "n_listings": int(len(df)),
                "mean_price_eur": float(df["price_eur"].mean()),
                "median_price_eur": float(df["price_eur"].median()),
                "mean_km": float(df["km"].mean()),
                "mean_vehicle_age": float(df["vehicle_age"].mean()),
                "private_share": float((df["advertizer_type_clean"] == "Particular").mean()),
                "mean_disclosure_index": float(df["disclosure_index_z"].mean()),
            }
        ]
    )
    save_table(descriptive_stats, "descriptive_stats")

    seller_summary = (
        df.groupby("advertizer_type_clean", observed=False)
        .agg(
            mean_disclosure_index=("disclosure_index_z", "mean"),
            median_price_eur=("price_eur", "median"),
            listings=("ad_id", "count"),
        )
        .reset_index()
    )
    save_table(seller_summary, "seller_summary")

    plot_df = test_df.copy()
    plot_df["baseline_residual"] = test_df["log_price"] - baseline_model.predict(x_test_base)
    plot_df["disclosure_quintile"] = pd.qcut(plot_df["disclosure_index_z"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])

    plt.figure(figsize=(7, 4.5))
    sns.boxplot(
        data=df.sample(min(20000, len(df)), random_state=RANDOM_STATE),
        x="advertizer_type_clean",
        y="disclosure_index_z",
    )
    plt.xlabel("Seller type")
    plt.ylabel("Disclosure index (z-score)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "disclosure_by_seller.png", dpi=200)
    plt.close()

    residual_quintiles = (
        plot_df.groupby(["disclosure_quintile", "advertizer_type_clean"], observed=False)["baseline_residual"]
        .mean()
        .reset_index()
    )
    plt.figure(figsize=(7, 4.5))
    sns.lineplot(
        data=residual_quintiles,
        x="disclosure_quintile",
        y="baseline_residual",
        hue="advertizer_type_clean",
        marker="o",
    )
    plt.axhline(0.0, color="black", linewidth=1, linestyle="--")
    plt.xlabel("Disclosure quintile")
    plt.ylabel("Average log-price residual")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "residuals_by_disclosure_quintile.png", dpi=200)
    plt.close()

    premium_terms = top_terms[top_terms["direction"] == "premium"].head(10).iloc[::-1]
    discount_terms = top_terms[top_terms["direction"] == "discount"].head(10).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].barh(premium_terms["term"], premium_terms["coefficient"], color="#2a9d8f")
    axes[0].set_title("Top premium terms")
    axes[1].barh(discount_terms["term"], discount_terms["coefficient"], color="#e76f51")
    axes[1].set_title("Top discount terms")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "top_price_terms.png", dpi=200)
    plt.close(fig)

    summary = {
        "dataset_rows_after_cleaning": int(len(df)),
        "model_sample_size": int(sample_n),
        "inference_sample_size": int(inference_n),
        "model_metrics": metrics.to_dict(orient="records"),
        "key_regression_terms": key_terms.to_dict(orient="records"),
        "top_premium_terms": top_terms[top_terms["direction"] == "premium"].head(10).to_dict(orient="records"),
        "top_discount_terms": top_terms[top_terms["direction"] == "discount"].head(10).to_dict(orient="records"),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


def main():
    download_dataset_if_needed()
    df = load_and_clean_data()
    df = add_disclosure_features(df)
    summary = run_analysis(df)

    print("Done.")
    print(f"Clean listings: {summary['dataset_rows_after_cleaning']}")
    print(f"Summary file: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
