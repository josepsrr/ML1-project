from __future__ import annotations

import re

import numpy as np
import pandas as pd
from spacy.lang.es.stop_words import STOP_WORDS

from .data import normalize_text

NORMALIZED_STOPWORDS = sorted({normalize_text(word) for word in STOP_WORDS if normalize_text(word)})

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


def _count_patterns(text: str, patterns: list[str]) -> int:
    return int(sum(len(re.findall(pattern, text)) for pattern in patterns))


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()
    text = featured["listing_text_clean"].fillna("")

    featured["token_count"] = text.str.split().str.len()
    featured["number_count"] = text.str.count(r"\b\d+\b")
    featured["maintenance_mentions"] = text.map(lambda value: _count_patterns(value, MAINTENANCE_PATTERNS))
    featured["transparency_mentions"] = text.map(lambda value: _count_patterns(value, TRANSPARENCY_PATTERNS))
    featured["defect_mentions"] = text.map(lambda value: _count_patterns(value, DEFECT_PATTERNS))
    featured["promotional_mentions"] = text.map(lambda value: _count_patterns(value, PROMOTIONAL_PATTERNS))

    raw_score = (
        1.2 * featured["maintenance_mentions"]
        + 1.4 * featured["transparency_mentions"]
        + 1.0 * featured["defect_mentions"]
        + 0.08 * featured["number_count"]
        + 0.01 * featured["token_count"]
        - 1.1 * featured["promotional_mentions"]
    )
    featured["disclosure_index_raw"] = raw_score
    featured["disclosure_index_z"] = (raw_score - raw_score.mean()) / raw_score.std(ddof=0)

    return featured
