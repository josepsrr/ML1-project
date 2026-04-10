from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


def _fix_mojibake(text: str) -> str:
    """Repair the most common UTF-8/latin1 decoding glitches in the raw CSV."""
    if not text:
        return ""

    candidate = text
    if any(marker in text for marker in ("Ã", "Â", "Ð", "¤", "�")):
        try:
            repaired = text.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
            if repaired.count("Ã") + repaired.count("Â") <= text.count("Ã") + text.count("Â"):
                candidate = repaired
        except UnicodeError:
            pass

    return unicodedata.normalize("NFKC", candidate)


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(char for char in text if not unicodedata.combining(char))


def normalize_text(text: str) -> str:
    text = _fix_mojibake(str(text or ""))
    text = text.replace("Leer más", " ")
    text = text.replace("leer más", " ")
    text = text.lower()
    text = strip_accents(text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_number(value: object) -> float:
    text = str(value or "").strip()
    if not text:
        return np.nan
    text = _fix_mojibake(text)
    text = text.replace(".", "")
    match = re.search(r"(\d+)", text)
    return float(match.group(1)) if match else np.nan


def load_raw_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    cleaned = cleaned.rename(columns={"Unnamed: 0": "row_id"})
    cleaned = cleaned.drop_duplicates(subset=["ad_id"])

    cleaned["advertizer_type_clean"] = cleaned["advertizer_type"].where(
        cleaned["advertizer_type"].isin(["Profesional", "Particular"])
    )
    cleaned = cleaned.dropna(subset=["advertizer_type_clean", "car_desc", "car_price", "car_km", "car_year"])

    cleaned["price_eur"] = cleaned["car_price"].map(_extract_number)
    cleaned["km"] = cleaned["car_km"].map(_extract_number)
    cleaned["registration_year"] = cleaned["car_year"].map(_extract_number)
    cleaned["power_cv"] = cleaned["car_power"].map(_extract_number)
    cleaned["ts"] = pd.to_datetime(cleaned["ts"], errors="coerce")

    snapshot_year = int(cleaned["ts"].dt.year.dropna().mode().iat[0])
    cleaned["vehicle_age"] = snapshot_year - cleaned["registration_year"]

    cleaned["ad_title"] = cleaned["ad_title"].fillna("").map(_fix_mojibake)
    cleaned["car_desc"] = cleaned["car_desc"].fillna("").map(_fix_mojibake)
    cleaned["brand"] = (
        cleaned["ad_title"].str.split("-").str[0].fillna("").map(normalize_text).replace("", "desconocida")
    )
    cleaned["listing_text"] = cleaned["car_desc"].str.strip()
    cleaned["listing_text_clean"] = cleaned["listing_text"].map(normalize_text)

    cleaned["region"] = cleaned["region"].fillna("desconocida").map(normalize_text)

    cleaned = cleaned[
        cleaned["price_eur"].between(500, 100000)
        & cleaned["km"].between(0, 500000)
        & cleaned["registration_year"].between(1990, snapshot_year)
        & cleaned["vehicle_age"].between(0, 35)
        & cleaned["power_cv"].between(40, 500)
        & cleaned["listing_text_clean"].str.len().ge(20)
    ].copy()

    cleaned["log_km"] = np.log1p(cleaned["km"])
    cleaned["log_price"] = np.log(cleaned["price_eur"])

    cleaned = cleaned.reset_index(drop=True)
    return cleaned
