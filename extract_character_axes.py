
from __future__ import annotations

import pandas as pd
import numpy as np

# ============================================================
# OpenPsychometrics Character Dataset -> 4 Axis Projection
# ------------------------------------------------------------
# Bu script, 500 BAP içeren karakter datasetinden senin seçtiğin
# 12 BAP alanını çekip her karakter için 4 eksen skoru üretir.
#
# Beklenen çıktı:
# data/character_axis_profiles.csv
#
# Çıktı kolonları:
# - character_id
# - name (varsa)
# - source / universe / title benzeri kolonlar (varsa)
# - uyum
# - ahlak
# - varolus
# - karar
# ============================================================

INPUT_PATH = "data/characters-aggregated-scores.csv"
OUTPUT_PATH = "data/character_axis_profiles.csv"

# ------------------------------------------------------------
# Eksenleri tanımlayan BAP'lar
# ------------------------------------------------------------
AXIS_BAPS = {
    "uyum": ["BAP15", "BAP134", "BAP485"],
    "ahlak": ["BAP79", "BAP84", "BAP390"],
    "varolus": ["BAP188", "BAP476", "BAP187"],
    "karar": ["BAP35", "BAP104", "BAP13"],
}

# ------------------------------------------------------------
# Yön düzeltmeleri
# Eğer ilgili BAP'ın sağ ucu bizim eksenimizin 100 tarafına
# ters düşüyorsa burada reverse=True ile çeviriyoruz.
#
# Senin tanımına göre:
# - uyum: 0=Uyumlu, 100=Kaotik
# - ahlak: 0=Bencil, 100=Özgeci
# - varolus: 0=Determinist/Kinik, 100=Varoluşçu/İradeci
# - karar: 0=Duygusal/Dürtüsel, 100=Mantıksal/Stratejik
# ------------------------------------------------------------
REVERSE_BAPS = {
    "BAP485": True,   # Başına Buyruk/Uyumlu  -> 100'ün Kaotik olması için ters çevir
    "BAP390": True,   # Şeffaf/Makyavelist    -> 100'ün Özgeci olması için ters çevir
    "BAP187": True,   # İdealist/Realist      -> 100'ü İradeci/Varoluşçu tarafa yaklaştırmak için ters çevir
    "BAP13": True,    # Stoic/Expressive      -> 100 Mantıksal/Stratejik için ters çevir
}

# ------------------------------------------------------------
# Dataset'te karakter adı / kaynak adı olabilecek kolon adayları
# ------------------------------------------------------------
NAME_CANDIDATES = [
    "character",
    "character_name",
    "name",
    "char_name",
]
SOURCE_CANDIDATES = [
    "source",
    "universe",
    "franchise",
    "title",
    "work",
    "series",
]

def smart_read_dataset(path: str) -> pd.DataFrame:
    """
    Bazı OpenPsychometrics dosyaları tab-delimited gelebiliyor.
    Önce tab ile, sonra normal csv ile deniyoruz.
    """
    try:
        df = pd.read_csv(path, sep="\t", engine="python")
        if len(df.columns) > 10:
            return clean_columns(df)
    except Exception:
        pass

    df = pd.read_csv(path, engine="python")
    return clean_columns(df)

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df

def find_first_existing(columns: list[str], candidates: list[str]) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    return None

def infer_bap_scale(df: pd.DataFrame, bap_cols: list[str]) -> tuple[float, float]:
    """
    BAP değerleri bazı dosyalarda 0-100, bazılarında farklı olabilir.
    Veri aralığını tahmin edip normalize edeceğiz.
    """
    values = []
    for c in bap_cols:
        if c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce").dropna()
            if not vals.empty:
                values.append(vals.min())
                values.append(vals.max())

    if not values:
        return 0.0, 100.0

    min_v = float(np.nanmin(values))
    max_v = float(np.nanmax(values))

    # Zaten 0-100 civarıysa dokunma
    if min_v >= 0 and max_v <= 100:
        return 0.0, 100.0

    return min_v, max_v

def normalize_to_100(series: pd.Series, low: float, high: float) -> pd.Series:
    if high <= low:
        return pd.Series(np.full(len(series), 50.0), index=series.index)

    if low == 0.0 and high == 100.0:
        return series.clip(0, 100)

    scaled = (series - low) / (high - low) * 100.0
    return scaled.clip(0, 100)

def build_axis_profiles(input_path: str = INPUT_PATH, output_path: str = OUTPUT_PATH) -> pd.DataFrame:
    df = smart_read_dataset(input_path)

    required_baps = sorted({b for bap_list in AXIS_BAPS.values() for b in bap_list})
    missing = [b for b in required_baps if b not in df.columns]
    if missing:
        raise ValueError(f"Eksik BAP kolonları var: {missing}")

    # Sayısala çevir
    for col in required_baps:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    scale_min, scale_max = infer_bap_scale(df, required_baps)

    # Normalize + yön düzelt
    oriented = {}
    for col in required_baps:
        s = normalize_to_100(df[col], scale_min, scale_max)
        if REVERSE_BAPS.get(col, False):
            s = 100.0 - s
        oriented[col] = s

    oriented_df = pd.DataFrame(oriented)

    # 4 eksen
    axis_df = pd.DataFrame({
        "uyum": oriented_df[AXIS_BAPS["uyum"]].mean(axis=1),
        "ahlak": oriented_df[AXIS_BAPS["ahlak"]].mean(axis=1),
        "varolus": oriented_df[AXIS_BAPS["varolus"]].mean(axis=1),
        "karar": oriented_df[AXIS_BAPS["karar"]].mean(axis=1),
    })

    # Kimlik / açıklayıcı kolonları koru
    out = pd.DataFrame()
    out["character_id"] = np.arange(len(df))

    name_col = find_first_existing(df.columns.tolist(), NAME_CANDIDATES)
    source_col = find_first_existing(df.columns.tolist(), SOURCE_CANDIDATES)

    if name_col:
        out["name"] = df[name_col]
    if source_col:
        out["source"] = df[source_col]

    out = pd.concat([out, axis_df], axis=1)

    # İstersen debug için seçili BAP'ları da ekleyebilirsin
    for col in required_baps:
        out[col] = oriented_df[col]

    out.to_csv(output_path, index=False, encoding="utf-8")
    return out

if __name__ == "__main__":
    result = build_axis_profiles()
    print("Tamamlandı.")
    print(result.head())
    print(f"\nÇıktı dosyası yazıldı: {OUTPUT_PATH}")
