"""
ETL Script for LCA Data (Labor Condition Applications)
-------------------------------------------------------
This script:
1. Reads multiple Excel source files containing LCA disclosure data.
2. Normalizes columns (names, formats, and data types).
3. Adds calculated fields (PRIMARY_DATE, RECEIVED_YEAR, FISCAL_YEAR, locations).
4. Deduplicates records by CASE_NUMBER.
5. Saves the merged clean dataset as both CSV and Parquet for fast reloading.

Usage:
    python etl_lca.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ------------------- PATHS -------------------
DATA_DIR = Path("data")  # Directory containing raw and processed files
OUT_CSV = DATA_DIR / "lca_merged_clean.csv"
OUT_PARQUET = DATA_DIR / "lca_merged_clean.parquet"

# ------------------- TARGET COLUMNS -------------------
# These are the final columns we want in our cleaned dataset
TARGET_COLS = [
    "CASE_NUMBER","CASE_STATUS","RECEIVED_DATE","DECISION_DATE","ORIGINAL_CERT_DATE",
    "VISA_CLASS","JOB_TITLE","SOC_CODE","SOC_TITLE","FULL_TIME_POSITION",
    "BEGIN_DATE","END_DATE","TOTAL_WORKER_POSITIONS",
    "NEW_EMPLOYMENT","CONTINUED_EMPLOYMENT","CHANGE_PREVIOUS_EMPLOYMENT",
    "NEW_CONCURRENT_EMPLOYMENT","CHANGE_EMPLOYER","AMENDED_PETITION",
    "EMPLOYER_NAME","EMPLOYER_CITY","EMPLOYER_STATE","EMPLOYER_POSTAL_CODE",
    "EMPLOYER_COUNTRY","EMPLOYER_FEIN","NAICS_CODE",
    "WORKSITE_CITY","WORKSITE_COUNTY","WORKSITE_STATE","WORKSITE_POSTAL_CODE",
    "WAGE_RATE_OF_PAY_FROM","WAGE_RATE_OF_PAY_TO","WAGE_UNIT_OF_PAY",
    "PREVAILING_WAGE","PW_UNIT_OF_PAY","PW_WAGE_LEVEL","PW_OES_YEAR",
    "TOTAL_WORKSITE_LOCATIONS"
]

# Excel load optimization — FEIN may be missing in some datasets
USECOLS = [c for c in TARGET_COLS if c not in {"EMPLOYER_FEIN"}]

# ------------------- FUNCTION: Load a Single File -------------------
def load_one(path: Path, source_tag: str) -> pd.DataFrame:
    """
    Loads one Excel file, cleans and standardizes columns, ensures required fields exist,
    and calculates helper columns for downstream analysis.
    
    Args:
        path (Path): Path to Excel file
        source_tag (str): Label to indicate data source (e.g., FY2024_Q1)
    
    Returns:
        pd.DataFrame: Cleaned DataFrame with standard columns
    """
    # Load all columns from Excel (names may have different cases)
    df = pd.read_excel(path, engine="openpyxl", usecols=lambda c: True)
    df.columns = [str(c).strip().upper() for c in df.columns]

    # Force specific columns to string to preserve formatting (e.g., leading zeros in ZIP/SOC codes)
    TEXT_FORCE = [
        "CASE_NUMBER","CASE_STATUS","VISA_CLASS","JOB_TITLE","SOC_CODE","SOC_TITLE",
        "FULL_TIME_POSITION","EMPLOYER_NAME","EMPLOYER_CITY","EMPLOYER_STATE",
        "EMPLOYER_POSTAL_CODE","EMPLOYER_COUNTRY","EMPLOYER_FEIN","NAICS_CODE",
        "WORKSITE_CITY","WORKSITE_COUNTY","WORKSITE_STATE","WORKSITE_POSTAL_CODE",
        "WAGE_UNIT_OF_PAY","PW_UNIT_OF_PAY","PW_WAGE_LEVEL"
    ]
    for col in TEXT_FORCE:
        if col in df.columns:
            df[col] = df[col].astype("string")

    # Add missing TARGET_COLS as NaN if not present
    for col in TARGET_COLS:
        if col not in df.columns:
            df[col] = np.nan

    # Keep only TARGET_COLS in correct order
    df = df[TARGET_COLS].copy()

    # Convert date columns
    for col in ["RECEIVED_DATE","DECISION_DATE","ORIGINAL_CERT_DATE","BEGIN_DATE","END_DATE"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Convert numeric columns
    num_cols = [
        "TOTAL_WORKER_POSITIONS","NEW_EMPLOYMENT","CONTINUED_EMPLOYMENT",
        "CHANGE_PREVIOUS_EMPLOYMENT","NEW_CONCURRENT_EMPLOYMENT","CHANGE_EMPLOYER",
        "WAGE_RATE_OF_PAY_FROM","WAGE_RATE_OF_PAY_TO","PREVAILING_WAGE","PW_OES_YEAR",
        "TOTAL_WORKSITE_LOCATIONS"
    ]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Add source tag column
    df["SOURCE"] = source_tag

    # ------------------- CALCULATED FIELDS -------------------
    # Primary date = first available date from these columns
    date_cols = ["RECEIVED_DATE","DECISION_DATE","BEGIN_DATE","ORIGINAL_CERT_DATE","END_DATE"]
    df["PRIMARY_DATE"] = pd.NaT
    for c in date_cols:
        df["PRIMARY_DATE"] = df["PRIMARY_DATE"].fillna(df[c])

    # Calendar year and month
    df["RECEIVED_YEAR"]  = df["PRIMARY_DATE"].dt.year
    df["RECEIVED_MONTH"] = df["PRIMARY_DATE"].dt.to_period("M").astype("string")

    # Fiscal year (starts in October)
    m = df["PRIMARY_DATE"].dt.month
    y = df["PRIMARY_DATE"].dt.year
    df["FISCAL_YEAR"] = np.where(m >= 10, y + 1, y)

    # Combined location fields
    df["WORKSITE_LOCATION"] = (
        df["WORKSITE_CITY"].astype("string").fillna("") + ", " +
        df["WORKSITE_STATE"].astype("string").fillna("")
    )
    df["EMPLOYER_LOCATION"] = (
        df["EMPLOYER_CITY"].astype("string").fillna("") + ", " +
        df["EMPLOYER_STATE"].astype("string").fillna("")
    )

    return df

# ------------------- MAIN PIPELINE -------------------
def main():
    """
    Reads multiple quarterly datasets, merges them into one,
    deduplicates by CASE_NUMBER (keeping the most recent decision),
    and saves the clean dataset as CSV and Parquet.
    """
    # List of source files (modify this list as new files are added)
    srcs = [
        (DATA_DIR / "LCA_Disclosure_Data_FY2024_Q1.xlsx", "FY2024_Q1"),
        (DATA_DIR / "LCA_Disclosure_Data_FY2023_Q4.xlsx", "FY2023_Q4"),
    ]
    frames = []

    # Load all sources
    for path, tag in srcs:
        print(f"Reading: {path}")
        frames.append(load_one(path, tag))

    # Merge all datasets
    merged = pd.concat(frames, ignore_index=True)

    # Deduplicate by CASE_NUMBER, keeping the latest DECISION_DATE (or RECEIVED_DATE if missing)
    merged["_sort_key"] = merged["DECISION_DATE"].fillna(merged["RECEIVED_DATE"])
    merged = (
        merged.sort_values("_sort_key")
              .drop_duplicates(subset=["CASE_NUMBER"], keep="last")
              .drop(columns=["_sort_key"])
    )

    print(f"Rows after merge & deduplication: {len(merged):,}")

    # ------------------- SAVE OUTPUT -------------------
    # Save as CSV
    merged.to_csv(OUT_CSV, index=False)

    # Ensure string columns are proper pandas string dtype before saving to Parquet
    for c in merged.columns:
        if pd.api.types.is_object_dtype(merged[c]) or pd.api.types.is_string_dtype(merged[c]):
            merged[c] = merged[c].astype("string")

    # Save as compressed Parquet for faster future loads
    merged.to_parquet(OUT_PARQUET, index=False, compression="snappy")

    print(f"Saved: {OUT_CSV}")
    print(f"Saved: {OUT_PARQUET}")

# ------------------- SCRIPT ENTRY POINT -------------------
if __name__ == "__main__":
    main()
