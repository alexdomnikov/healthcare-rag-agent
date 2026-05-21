# Loads the two 2026 CMS Star Ratings CSVs into Neon Postgres. Run once:
#   uv run python scripts/load_cms_tables.py

import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text

from healthcare_rag.core import get_engine

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data"
SUMMARY_PATH = DATA_PATH / "2026_Star_Ratings_Data_Table_Summary_Ratings_Oct_8_2025.csv"
DOMAIN_PATH  = DATA_PATH / "2026_Star_Ratings_Data_Table_Domain_Stars_Oct_8_2025.csv"


def normalize_col(name: str) -> str:
    # '2026 Part C Summary' → 'part_c_summary_2026'. A leading year would make
    # an invalid SQL identifier, so we move it to the end.
    s = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    s = re.sub(r"^(\d+)_(.+)$", r"\2_\1", s)
    return s


RATING_SENTINELS = {
    "not applicable",
    "not enough data available",
    "plan not required to report measure",
    "too few to report",
    "",
}


def to_numeric_or_null(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip().str.lower()
    return pd.to_numeric(
        series.where(~cleaned.isin(RATING_SENTINELS), other=pd.NA),
        errors="coerce",
    )


def load_summary_ratings() -> None:
    print("Loading Summary Ratings")
    df = pd.read_csv(SUMMARY_PATH, header=1, encoding="latin-1")
    df.columns = [normalize_col(c) for c in df.columns]

    for col in ["part_c_summary_2026", "part_d_summary_2026", "overall_2026"]:
        if col in df.columns:
            df[col] = to_numeric_or_null(df[col])

    for col in ["disaster_2023", "disaster_2024"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["contract_number"].notna()].copy()

    engine = get_engine()
    with engine.begin() as conn:
        df.to_sql("cms_summary_ratings", conn, if_exists="replace",
                  index=False, method="multi", chunksize=500)
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_sum_contract ON cms_summary_ratings (contract_number)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_sum_org_type ON cms_summary_ratings (organization_type)"))

    print(f"done: cms_summary_ratings ({len(df):,} rows)")
    print(f"columns: {list(df.columns)}")

def load_domain_stars() -> None:
    print("Loading Domain Stars")
    df = pd.read_csv(DOMAIN_PATH, header=1, encoding="latin-1")
    df.columns = [normalize_col(c) for c in df.columns]

    rating_cols = [c for c in df.columns if re.match(r"(hd|dd)\d", c)]
    for col in rating_cols:
        df[col] = to_numeric_or_null(df[col])

    df = df[df["contract_number"].notna()].copy()

    engine = get_engine()
    with engine.begin() as conn:
        df.to_sql("cms_domain_stars", conn, if_exists="replace",
                  index=False, method="multi", chunksize=500)
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_dom_contract ON cms_domain_stars (contract_number)"))

    print(f"done: cms_domain_stars ({len(df):,} rows)")
    print(f"columns: {list(df.columns)}")

def sanity_check() -> None:
    print("\nSanity checks")
    with get_engine().connect() as conn:
        for table, rating_col in [
            ("cms_summary_ratings", "overall_2026"),
            ("cms_domain_stars", "hd1_staying_healthy_screenings_tests_and_vaccines"),
        ]:
            n   = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            avg = conn.execute(
                text(f'SELECT ROUND(AVG("{rating_col}")::numeric, 2) FROM {table}')
            ).scalar()
            print(f"{table}: {n:,} rows | avg {rating_col}: {avg}")

if __name__ == "__main__":
    load_summary_ratings()
    load_domain_stars()
    sanity_check()