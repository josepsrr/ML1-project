import os
import re
import unicodedata
from urllib.request import urlretrieve

import matplotlib

matplotlib.use("Agg")

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


sns.set_theme(style="whitegrid")


def fix_text(text):
    text = str(text or "")
    if text == "":
        return ""

    if any(mark in text for mark in ["Ãƒ", "Ã‚", "Ã", "Â¤", "ï¿½"]):
        try:
            repaired = text.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
            if repaired.count("Ãƒ") + repaired.count("Ã‚") <= text.count("Ãƒ") + text.count("Ã‚"):
                text = repaired
        except UnicodeError:
            pass

    return unicodedata.normalize("NFKC", text)


def remove_accents(text):
    text = unicodedata.normalize("NFKD", text)
    return "".join(letter for letter in text if not unicodedata.combining(letter))


def normalise(text):
    text = fix_text(text)
    text = text.replace("Leer mÃ¡s", " ")
    text = text.replace("leer mÃ¡s", " ")
    text = text.lower()
    text = remove_accents(text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_number(value):
    text = fix_text(value).replace(".", "")
    match = re.search(r"(\d+)", text)
    if match:
        return float(match.group(1))
    return np.nan


def count_words(text, patterns):
    total = 0
    for pattern in patterns:
        total += len(re.findall(pattern, text))
    return total


def save_table(table, name, tables_dir):
    table.to_csv(os.path.join(tables_dir, name + ".csv"), index=False)
    table.to_latex(
        os.path.join(tables_dir, name + ".tex"),
        index=False,
        float_format="%.4f",
        escape=True,
    )


def prepare_x(train_df, test_df, columns):
    categorical_columns = ["advertizer_type_clean", "region", "brand"]
    numeric_columns = [col for col in columns if col not in categorical_columns]

    x_train = train_df[columns].copy()
    x_test = test_df[columns].copy()

    for col in numeric_columns:
        median_value = x_train[col].median()
        mean_value = x_train[col].mean()
        std_value = x_train[col].std(ddof=0)

        if pd.isna(std_value) or std_value == 0:
            std_value = 1.0

        x_train[col] = x_train[col].fillna(median_value)
        x_test[col] = x_test[col].fillna(median_value)
        x_train[col] = (x_train[col] - mean_value) / std_value
        x_test[col] = (x_test[col] - mean_value) / std_value

    for col in categorical_columns:
        if col in x_train.columns:
            if x_train[col].mode(dropna=True).empty:
                fill_value = "desconocida"
            else:
                fill_value = x_train[col].mode(dropna=True).iloc[0]

            x_train[col] = x_train[col].fillna(fill_value)
            x_test[col] = x_test[col].fillna(fill_value)

    x_train = pd.get_dummies(x_train, columns=[col for col in categorical_columns if col in x_train.columns])
    x_test = pd.get_dummies(x_test, columns=[col for col in categorical_columns if col in x_test.columns])
    x_test = x_test.reindex(columns=x_train.columns, fill_value=0)

    return x_train, x_test


# 1. Basic settings

data_url = "https://zenodo.org/records/4252636/files/dataset.csv?download=1"
random_state = 42
sample_size = 60000
inference_sample_size = 40000

root = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(root, "data", "raw")
data_path = os.path.join(data_folder, "dataset.csv")
outputs_folder = os.path.join(root, "outputs")
tables_dir = os.path.join(outputs_folder, "tables")
figures_dir = os.path.join(outputs_folder, "figures")
old_summary_path = os.path.join(outputs_folder, "summary.json")

os.makedirs(data_folder, exist_ok=True)
os.makedirs(outputs_folder, exist_ok=True)
os.makedirs(tables_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

if os.path.exists(old_summary_path):
    try:
        os.remove(old_summary_path)
    except PermissionError:
        pass


# 2. Download data

if not os.path.exists(data_path):
    print("Downloading dataset...")
    urlretrieve(data_url, data_path)
    print("Dataset downloaded.")
else:
    print("Dataset already exists.")


# 3. Read and clean data

print("Reading dataset...")
df = pd.read_csv(data_path)

df = df.rename(columns={"Unnamed: 0": "row_id"})
df = df.drop_duplicates(subset=["ad_id"])

df["advertizer_type_clean"] = df["advertizer_type"].where(
    df["advertizer_type"].isin(["Profesional", "Particular"])
)

df = df.dropna(subset=["advertizer_type_clean", "car_desc", "car_price", "car_km", "car_year"])

df["price_eur"] = df["car_price"].map(get_number)
df["km"] = df["car_km"].map(get_number)
df["registration_year"] = df["car_year"].map(get_number)
df["power_cv"] = df["car_power"].map(get_number)
df["ts"] = pd.to_datetime(df["ts"], errors="coerce")

snapshot_year = int(df["ts"].dt.year.dropna().mode().iloc[0])

df["vehicle_age"] = snapshot_year - df["registration_year"]
df["ad_title"] = df["ad_title"].fillna("").map(fix_text)
df["car_desc"] = df["car_desc"].fillna("").map(fix_text)
df["listing_text"] = df["car_desc"].str.strip()
df["listing_text_clean"] = df["listing_text"].map(normalise)
df["region"] = df["region"].fillna("desconocida").map(normalise)
df["brand"] = df["ad_title"].str.split("-").str[0].fillna("").map(normalise)
df["brand"] = df["brand"].replace("", "desconocida")

df = df[
    df["price_eur"].between(500, 100000)
    & df["km"].between(0, 500000)
    & df["registration_year"].between(1990, snapshot_year)
    & df["vehicle_age"].between(0, 35)
    & df["power_cv"].between(40, 500)
    & df["listing_text_clean"].str.len().ge(20)
].copy()

df["log_price"] = np.log(df["price_eur"])
df["log_km"] = np.log1p(df["km"])
df = df.reset_index(drop=True)

print("Clean listings:", len(df))


# 4. Disclosure score

maintenance_patterns = [
    r"\blibro de mantenimiento\b",
    r"\bhistorial de mantenimiento\b",
    r"\brevisiones? al dia\b",
    r"\bmantenimiento al dia\b",
    r"\bitv (recien )?pasad[ao]s?\b",
    r"\bcorrea de distribucion\b",
    r"\bdistribucion recien cambiada\b",
    r"\baceite (recien )?cambiad[oa]\b",
    r"\bembrague (nuevo|cambiad[oa])\b",
    r"\bruedas nuevas\b",
    r"\bneumaticos nuevos\b",
    r"\bfacturas?\b",
]

transparency_patterns = [
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

defect_patterns = [
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

promotional_patterns = [
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

df["token_count"] = df["listing_text_clean"].str.split().str.len()
df["number_count"] = df["listing_text_clean"].str.count(r"\b\d+\b")
df["maintenance_mentions"] = df["listing_text_clean"].map(lambda text: count_words(text, maintenance_patterns))
df["transparency_mentions"] = df["listing_text_clean"].map(lambda text: count_words(text, transparency_patterns))
df["defect_mentions"] = df["listing_text_clean"].map(lambda text: count_words(text, defect_patterns))
df["promotional_mentions"] = df["listing_text_clean"].map(lambda text: count_words(text, promotional_patterns))

df["disclosure_index_raw"] = (
    1.2 * df["maintenance_mentions"]
    + 1.4 * df["transparency_mentions"]
    + 1.0 * df["defect_mentions"]
    + 0.08 * df["number_count"]
    + 0.01 * df["token_count"]
    - 1.1 * df["promotional_mentions"]
)

df["disclosure_index_z"] = (
    df["disclosure_index_raw"] - df["disclosure_index_raw"].mean()
) / df["disclosure_index_raw"].std(ddof=0)


# 5. Simple descriptive tables

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
save_table(descriptive_stats, "descriptive_stats", tables_dir)

seller_summary = (
    df.groupby("advertizer_type_clean", observed=False)
    .agg(
        mean_disclosure_index=("disclosure_index_z", "mean"),
        median_price_eur=("price_eur", "median"),
        listings=("ad_id", "count"),
    )
    .reset_index()
)
save_table(seller_summary, "seller_summary", tables_dir)


# 6. Train / test split

model_df = df.sample(min(sample_size, len(df)), random_state=random_state).copy()
train_df, test_df = train_test_split(model_df, test_size=0.2, random_state=random_state)

structured_columns = ["log_km", "vehicle_age", "power_cv", "advertizer_type_clean", "region", "brand"]
text_columns = [
    "maintenance_mentions",
    "transparency_mentions",
    "defect_mentions",
    "promotional_mentions",
    "token_count",
    "number_count",
    "disclosure_index_z",
]


# 7. Baseline model

x_train_base, x_test_base = prepare_x(train_df, test_df, structured_columns)

baseline_model = Ridge(alpha=2.5)
baseline_model.fit(x_train_base, train_df["log_price"])
baseline_pred = baseline_model.predict(x_test_base)


# 8. Baseline + disclosure score

x_train_disc, x_test_disc = prepare_x(train_df, test_df, structured_columns + text_columns)

disclosure_model = Ridge(alpha=2.5)
disclosure_model.fit(x_train_disc, train_df["log_price"])
disclosure_pred = disclosure_model.predict(x_test_disc)


# 9. TF-IDF model on the residuals

stop_words = sorted({normalise(word) for word in STOP_WORDS if normalise(word)})

vectorizer = TfidfVectorizer(
    lowercase=False,
    token_pattern=r"(?u)\b[a-z0-9]{2,}\b",
    stop_words=stop_words,
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
    random_state=random_state,
)

residual_model.fit(train_text, residual_train)
hybrid_pred = baseline_model.predict(x_test_base) + residual_model.predict(test_text)


# 10. Model comparison

model_metrics = pd.DataFrame(
    [
        {
            "model": "Structured baseline",
            "rmse_log_price": float(np.sqrt(mean_squared_error(test_df["log_price"], baseline_pred))),
            "mae_log_price": float(mean_absolute_error(test_df["log_price"], baseline_pred)),
            "r2": float(r2_score(test_df["log_price"], baseline_pred)),
        },
        {
            "model": "Structured + disclosure lexicon",
            "rmse_log_price": float(np.sqrt(mean_squared_error(test_df["log_price"], disclosure_pred))),
            "mae_log_price": float(mean_absolute_error(test_df["log_price"], disclosure_pred)),
            "r2": float(r2_score(test_df["log_price"], disclosure_pred)),
        },
        {
            "model": "Structured + TF-IDF residual text",
            "rmse_log_price": float(np.sqrt(mean_squared_error(test_df["log_price"], hybrid_pred))),
            "mae_log_price": float(mean_absolute_error(test_df["log_price"], hybrid_pred)),
            "r2": float(r2_score(test_df["log_price"], hybrid_pred)),
        },
    ]
)
save_table(model_metrics, "model_metrics", tables_dir)


# 11. Top words

feature_names = np.array(vectorizer.get_feature_names_out())
coefs = residual_model.coef_

top_positive = coefs.argsort()[-20:][::-1]
top_negative = coefs.argsort()[:20]

top_terms = pd.concat(
    [
        pd.DataFrame(
            {
                "term": feature_names[top_positive],
                "coefficient": coefs[top_positive],
                "direction": "premium",
            }
        ),
        pd.DataFrame(
            {
                "term": feature_names[top_negative],
                "coefficient": coefs[top_negative],
                "direction": "discount",
            }
        ),
    ],
    ignore_index=True,
)
save_table(top_terms, "top_price_terms", tables_dir)


# 12. Regression for the paper

inference_df = df.sample(min(inference_sample_size, len(df)), random_state=random_state).copy()
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
    regression_table["term"].isin(
        ["Intercept", "disclosure_index_z", "is_private", "disclosure_index_z:is_private"]
    )
].copy()
save_table(key_terms, "interaction_regression_key_terms", tables_dir)


# 13. Figures

sample_plot_df = df.sample(min(20000, len(df)), random_state=random_state).copy()

plt.figure(figsize=(7, 4.5))
sns.boxplot(data=sample_plot_df, x="advertizer_type_clean", y="disclosure_index_z")
plt.xlabel("Seller type")
plt.ylabel("Disclosure index (z-score)")
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "disclosure_by_seller.png"), dpi=200)
plt.close()

plot_df = test_df.copy()
plot_df["baseline_residual"] = test_df["log_price"] - baseline_model.predict(x_test_base)
plot_df["disclosure_quintile"] = pd.qcut(
    plot_df["disclosure_index_z"],
    5,
    labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
)

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
plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
plt.xlabel("Disclosure quintile")
plt.ylabel("Average log-price residual")
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "residuals_by_disclosure_quintile.png"), dpi=200)
plt.close()

premium_terms = top_terms[top_terms["direction"] == "premium"].head(10).iloc[::-1]
discount_terms = top_terms[top_terms["direction"] == "discount"].head(10).iloc[::-1]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].barh(premium_terms["term"], premium_terms["coefficient"], color="#2a9d8f")
axes[0].set_title("Top premium terms")
axes[1].barh(discount_terms["term"], discount_terms["coefficient"], color="#e76f51")
axes[1].set_title("Top discount terms")
fig.tight_layout()
fig.savefig(os.path.join(figures_dir, "top_price_terms.png"), dpi=200)
plt.close(fig)


# 14. Final prints

print()
print("Done.")
print("Rows after cleaning:", len(df))
print()
print(model_metrics)
print()
print(key_terms)
