from pathlib import Path
import pandas as pd
import streamlit as st


def _find_processed_file():
    base = Path(__file__).resolve().parents[1]

    candidates = [
        base / "data" / "processed" / "X_train_hit.parquet",
        base / "data" / "processed" / "X_train_hidden_gem.parquet",
        base.parent / "steam-project" / "data" / "processed" / "X_train_hit.parquet",
        base.parent / "steam-project" / "data" / "processed" / "X_train_hidden_gem.parquet",
        base / "data" / "processed" / "Xtrainhit.parquet",
        base.parent / "steam-project" / "data" / "processed" / "Xtrainhit.parquet",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError("전처리된 parquet 파일을 찾지 못했습니다.")


@st.cache_data
def load_data():
    path = _find_processed_file()
    df = pd.read_parquet(path)

    if "release_year" not in df.columns:
        if "year" in df.columns:
            df["release_year"] = df["year"]
        elif "releaseyear" in df.columns:
            df["release_year"] = df["releaseyear"]

    if "is_free" not in df.columns:
        if "isfree" in df.columns:
            df["is_free"] = df["isfree"]
        else:
            df["is_free"] = 0

    if "required_age" not in df.columns:
        if "requiredage" in df.columns:
            df["required_age"] = df["requiredage"]
        else:
            df["required_age"] = 0

    if "price" not in df.columns:
        df["price"] = 0

    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0)
    df["is_free"] = pd.to_numeric(df["is_free"], errors="coerce").fillna(0).astype(int)
    df["required_age"] = pd.to_numeric(df["required_age"], errors="coerce").fillna(0).astype(int)

    if "release_year" in df.columns:
        df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
        df = df[df["release_year"].notna()].copy()
        df["release_year"] = df["release_year"].astype(int)

    return df


def loaddata():
    return load_data()