import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path
import re
import numpy as np

# ------------------- APP CONFIGURATION -------------------
st.set_page_config(page_title="Talent Migration (LCA) MVP", layout="wide")
st.title("📈 Strategic Talent Migration — LCA MVP")

# ------------------- DATA FILE PATHS -------------------
DATA_PARQUET = Path("data/lca_merged_clean.parquet")  # Optimized binary file
DATA_CSV     = Path("data/lca_merged_clean.csv")      # Backup CSV file

# ------------------- LOAD DATA FUNCTION -------------------
@st.cache_data
def load_data():
    """
    Loads the LCA dataset from Parquet (preferred) or CSV (fallback).
    Also calculates missing helper columns like PRIMARY_DATE, RECEIVED_YEAR,
    FISCAL_YEAR, normalized employer names, and annual wages.
    """
    # Load from Parquet if available, otherwise fallback to CSV
    if DATA_PARQUET.exists():
        df = pd.read_parquet(DATA_PARQUET)
    else:
        # Read CSV and parse date columns
        date_cols = ["RECEIVED_DATE","DECISION_DATE","ORIGINAL_CERT_DATE","BEGIN_DATE","END_DATE"]
        df = pd.read_csv(
            DATA_CSV,
            parse_dates=[c for c in date_cols if c in pd.read_csv(DATA_CSV, nrows=0).columns]
        )

    # ------------------- ADD MISSING DATE COLUMNS -------------------
    # PRIMARY_DATE = the main date used for analysis (fallback chain of available date fields)
    if "PRIMARY_DATE" not in df.columns:
        date_cols = ["RECEIVED_DATE","DECISION_DATE","BEGIN_DATE","ORIGINAL_CERT_DATE","END_DATE"]
        df["PRIMARY_DATE"] = pd.NaT
        for c in date_cols:
            if c in df.columns:
                df["PRIMARY_DATE"] = df["PRIMARY_DATE"].fillna(df[c])

    # RECEIVED_YEAR and RECEIVED_MONTH for calendar-based grouping
    if "RECEIVED_YEAR" not in df.columns or df["RECEIVED_YEAR"].isna().all():
        df["RECEIVED_YEAR"]  = df["PRIMARY_DATE"].dt.year
        df["RECEIVED_MONTH"] = df["PRIMARY_DATE"].dt.to_period("M").astype("string")

    # FISCAL_YEAR calculation (Oct–Sep fiscal year, common for US government data)
    if "FISCAL_YEAR" not in df.columns or df["FISCAL_YEAR"].isna().all():
        m = df["PRIMARY_DATE"].dt.month
        y = df["PRIMARY_DATE"].dt.year
        df["FISCAL_YEAR"] = np.where(m >= 10, y + 1, y)

    # ------------------- NORMALIZE EMPLOYER NAMES -------------------
    if "EMPLOYER_NAME_NORM" not in df.columns and "EMPLOYER_NAME" in df.columns:
        def norm_company(x: str) -> str:
            """
            Uppercases, trims spaces, and removes punctuation from company names
            for better filtering and grouping.
            """
            if not isinstance(x, str): return ""
            import re
            x = x.upper().strip()
            x = re.sub(r"[\,\.\-]", "", x)
            x = re.sub(r"\s+", " ", x)
            return x
        df["EMPLOYER_NAME_NORM"] = df["EMPLOYER_NAME"].apply(norm_company)

    # ------------------- CALCULATE ANNUAL WAGE -------------------
    if "WAGE_ANNUAL_FROM" not in df.columns:
        def annualize(row):
            """
            Converts wage from hourly/weekly/monthly to annual salary (USD).
            If already in yearly format, returns as is.
            """
            v = row.get("WAGE_RATE_OF_PAY_FROM")
            if pd.isna(v): return pd.NA
            unit = str(row.get("WAGE_UNIT_OF_PAY", "")).lower()
            if "hour" in unit:   return float(v) * 2080
            if "week" in unit:   return float(v) * 52
            if "bi" in unit:     return float(v) * 26
            if "month" in unit:  return float(v) * 12
            if "year" in unit:   return float(v)
            return pd.NA
        df["WAGE_ANNUAL_FROM"] = df.apply(annualize, axis=1)

    return df

# ------------------- LOAD THE DATA -------------------
df = load_data()

# ------------------- SIDEBAR FILTERS -------------------
st.sidebar.header("Filters")

# Year type selection (Calendar vs Fiscal)
mode = st.sidebar.radio("Year type", ["Calendar Year", "Fiscal Year"])
year_col = "RECEIVED_YEAR" if mode.startswith("Calendar") else "FISCAL_YEAR"

# Available years
years = sorted([int(y) for y in df[year_col].dropna().unique()])
if not years:
    st.error("No valid years found in the dataset.")
    st.stop()

# Year filter
year = st.sidebar.selectbox("Year", options=years, index=len(years)-1)

# State filter
states = sorted(df["WORKSITE_STATE"].dropna().astype(str).unique())
state = st.sidebar.selectbox("State (WORKSITE_STATE)", options=["All"] + states)

# Employer name substring filter
emp_query = st.sidebar.text_input("Employer filter (substring)", "")

# ------------------- APPLY FILTERS -------------------
df_year = df[df[year_col] == year].copy()
if state != "All":
    df_year = df_year[df_year["WORKSITE_STATE"].astype(str) == state]
if emp_query.strip():
    q = emp_query.strip().upper()
    df_year = df_year[df_year["EMPLOYER_NAME_NORM"].str.contains(q, na=False)]

# ------------------- KPI METRICS -------------------
total_filings = len(df_year)
unique_employers = df_year["EMPLOYER_NAME_NORM"].nunique()
median_wage = pd.to_numeric(df_year["WAGE_ANNUAL_FROM"], errors="coerce").median()

k1, k2, k3 = st.columns(3)
k1.metric("Filings (after filter)", f"{total_filings:,}")
k2.metric("Unique employers", f"{unique_employers:,}")
k3.metric("Median annual wage (USD)", f"{int(median_wage):,}" if pd.notna(median_wage) else "—")

# ------------------- TOP EMPLOYERS -------------------
st.subheader("🔝 Top-25 Employers by LCA Filings")
top_emp = (df_year.groupby("EMPLOYER_NAME")
                 .size()
                 .reset_index(name="filings")
                 .sort_values("filings", ascending=False)
                 .head(25))
fig_emp = px.bar(top_emp, x="EMPLOYER_NAME", y="filings")
fig_emp.update_layout(xaxis_title="", yaxis_title="Filings", margin=dict(l=10,r=10,t=40,b=10))
st.plotly_chart(fig_emp, use_container_width=True)
st.dataframe(top_emp, use_container_width=True)

# ------------------- FILINGS BY STATE -------------------
st.subheader("🌎 Filings by State (Worksite)")
by_state = (df_year.groupby("WORKSITE_STATE")
                 .size()
                 .reset_index(name="filings")
                 .sort_values("filings", ascending=False))
fig_state = px.bar(by_state, x="WORKSITE_STATE", y="filings")
fig_state.update_layout(xaxis_title="State", yaxis_title="Filings", margin=dict(l=10,r=10,t=40,b=10))
st.plotly_chart(fig_state, use_container_width=True)
st.dataframe(by_state, use_container_width=True)

# ------------------- MONTHLY TREND (ALL YEARS) -------------------
st.subheader("📅 Monthly Filing Trend (All Years, Calendar Months)")
if "RECEIVED_MONTH" in df.columns:
    by_month = (df.dropna(subset=["RECEIVED_MONTH"])
                  .groupby("RECEIVED_MONTH")
                  .size()
                  .reset_index(name="filings"))
    by_month["_ts"] = pd.PeriodIndex(by_month["RECEIVED_MONTH"], freq="M").to_timestamp()
    by_month = by_month.sort_values("_ts")
    fig_month = px.line(by_month, x="RECEIVED_MONTH", y="filings")
    fig_month.update_layout(xaxis_title="Month", yaxis_title="Filings", margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig_month, use_container_width=True)
    st.dataframe(by_month[["RECEIVED_MONTH","filings"]], use_container_width=True)

# ------------------- DIAGNOSTIC INFO -------------------
with st.expander("ℹ️ Diagnostic — Available Years"):
    info_cal = (df.groupby("RECEIVED_YEAR").size().reset_index(name="filings").sort_values("RECEIVED_YEAR"))
    info_fy  = (df.groupby("FISCAL_YEAR").size().reset_index(name="filings").sort_values("FISCAL_YEAR"))
    st.write("Calendar Years:"); st.dataframe(info_cal, use_container_width=True)
    st.write("Fiscal Years:");  st.dataframe(info_fy,  use_container_width=True)
