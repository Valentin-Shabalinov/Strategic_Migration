import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path
import re
import numpy as np
from datetime import date

# ------------------- APP CONFIGURATION -------------------
st.set_page_config(page_title="Talent Migration (LCA)", layout="wide")
st.title("📈 Strategic Talent Migration — LCA")

# ------------------- DATA FILE PATHS -------------------
DATA_PARQUET = Path("data/lca_merged_clean.parquet")
DATA_CSV     = Path("data/lca_merged_clean.csv")

# ------------------- HELPERS -------------------
def fiscal_year_range(fy: int):
    """Return start/end dates (inclusive) for a given fiscal year FY.
       Example: FY2024 => Oct 1, 2023 — Sep 30, 2024"""
    return date(fy - 1, 10, 1), date(fy, 9, 30)

def fiscal_quarter_from_month(m: int) -> int:
    """Oct-Dec=Q1, Jan-Mar=Q2, Apr-Jun=Q3, Jul-Sep=Q4"""
    if m in (10, 11, 12): return 1
    if m in (1, 2, 3):    return 2
    if m in (4, 5, 6):    return 3
    return 4  # 7,8,9

def norm_text(x: str) -> str:
    if not isinstance(x, str): return ""
    x = x.strip().upper()
    x = re.sub(r"\s+", " ", x)
    return x

# ------------------- LOAD DATA FUNCTION -------------------
@st.cache_data
def load_data():
    """
    Loads the LCA dataset from Parquet (preferred) or CSV (fallback) and
    computes helpers (PRIMARY_DATE, RECEIVED_YEAR, FISCAL_YEAR, normalized fields, annual wage).
    """
    if DATA_PARQUET.exists():
        df = pd.read_parquet(DATA_PARQUET)
    else:
        date_cols = ["RECEIVED_DATE","DECISION_DATE","ORIGINAL_CERT_DATE","BEGIN_DATE","END_DATE"]
        df = pd.read_csv(
            DATA_CSV,
            parse_dates=[c for c in date_cols if c in pd.read_csv(DATA_CSV, nrows=0).columns]
        )

    # PRIMARY_DATE
    if "PRIMARY_DATE" not in df.columns:
        date_cols = ["RECEIVED_DATE","DECISION_DATE","BEGIN_DATE","ORIGINAL_CERT_DATE","END_DATE"]
        df["PRIMARY_DATE"] = pd.NaT
        for c in date_cols:
            if c in df.columns:
                df["PRIMARY_DATE"] = df["PRIMARY_DATE"].fillna(df[c])

    # Calendar helpers
    if "RECEIVED_YEAR" not in df.columns or df["RECEIVED_YEAR"].isna().all():
        df["RECEIVED_YEAR"]  = df["PRIMARY_DATE"].dt.year
        df["RECEIVED_MONTH"] = df["PRIMARY_DATE"].dt.to_period("M").astype("string")

    # Fiscal year helper
    if "FISCAL_YEAR" not in df.columns or df["FISCAL_YEAR"].isna().all():
        m = df["PRIMARY_DATE"].dt.month
        y = df["PRIMARY_DATE"].dt.year
        df["FISCAL_YEAR"] = np.where(m >= 10, y + 1, y)

    # Normalized names
    if "EMPLOYER_NAME" in df.columns and "EMPLOYER_NAME_NORM" not in df.columns:
        df["EMPLOYER_NAME_NORM"] = df["EMPLOYER_NAME"].apply(norm_text)
    if "JOB_TITLE" in df.columns and "JOB_TITLE_NORM" not in df.columns:
        df["JOB_TITLE_NORM"] = df["JOB_TITLE"].apply(norm_text)
    if "SOC_TITLE" in df.columns and "SOC_TITLE_NORM" not in df.columns:
        df["SOC_TITLE_NORM"] = df["SOC_TITLE"].apply(norm_text)

    # Annual wage
    if "WAGE_ANNUAL_FROM" not in df.columns:
        def annualize(row):
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

# Show year date range
if mode.startswith("Fiscal"):
    fy_start, fy_end = fiscal_year_range(year)
    st.sidebar.info(f"**Fiscal Year {year}**: {fy_start.strftime('%b %d, %Y')} — {fy_end.strftime('%b %d, %Y')}")
else:
    st.sidebar.info(f"**Calendar Year {year}**: Jan 01, {year} — Dec 31, {year}")

# State & Employer filters
states = sorted(df["WORKSITE_STATE"].dropna().astype(str).unique())
state = st.sidebar.selectbox("State (WORKSITE_STATE)", options=["All"] + states)
emp_query = st.sidebar.text_input("Employer filter (substring)", "")

# ---- Job/SOC filters
st.sidebar.markdown("---")
st.sidebar.subheader("Job / Occupation")
job_query = st.sidebar.text_input("Search in JOB_TITLE (substring)", "")
# top lists to help select
top_jobs = (df["JOB_TITLE_NORM"].dropna().value_counts().head(200).index.tolist()
            if "JOB_TITLE_NORM" in df.columns else [])
top_soc_titles = (df["SOC_TITLE_NORM"].dropna().value_counts().head(200).index.tolist()
                  if "SOC_TITLE_NORM" in df.columns else [])
job_pick = st.sidebar.multiselect("Popular JOB_TITLE", options=top_jobs)
soc_code_pick = st.sidebar.text_input("SOC_CODE (exact/prefix, e.g., 15-12)", "")
soc_title_pick = st.sidebar.multiselect("Popular SOC_TITLE", options=top_soc_titles)

# ------------------- APPLY FILTERS -------------------
df_year = df[df[year_col] == year].copy()
if state != "All":
    df_year = df_year[df_year["WORKSITE_STATE"].astype(str) == state]
if emp_query.strip():
    q = emp_query.strip().upper()
    df_year = df_year[df_year["EMPLOYER_NAME_NORM"].str.contains(q, na=False)]
if job_query.strip() and "JOB_TITLE_NORM" in df_year.columns:
    q = job_query.strip().upper()
    df_year = df_year[df_year["JOB_TITLE_NORM"].str.contains(q, na=False)]
if job_pick:
    df_year = df_year[df_year["JOB_TITLE_NORM"].isin(job_pick)]
if soc_code_pick.strip() and "SOC_CODE" in df_year.columns:
    q = soc_code_pick.strip()
    df_year = df_year[df_year["SOC_CODE"].astype(str).str.startswith(q, na=False)]
if soc_title_pick:
    df_year = df_year[df_year["SOC_TITLE_NORM"].isin(soc_title_pick)]

# ---- enrich with readable date & fiscal quarter
if "PRIMARY_DATE" in df_year.columns:
    dt = pd.to_datetime(df_year["PRIMARY_DATE"])
    df_year["PRIMARY_DATE_STR"] = dt.dt.strftime("%Y-%m-%d")
    df_year["FISCAL_QUARTER"] = dt.dt.month.map(fiscal_quarter_from_month)
    df_year["FY_LABEL"] = "FY" + df_year["FISCAL_YEAR"].astype("Int64").astype("string")

# ------------------- KPI METRICS -------------------
total_filings = len(df_year)
unique_employers = df_year["EMPLOYER_NAME_NORM"].nunique() if "EMPLOYER_NAME_NORM" in df_year else df_year["EMPLOYER_NAME"].nunique()
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

# ------------------- TOP JOB TITLES -------------------
st.subheader("🧑‍💻 Top-25 Job Titles")
if "JOB_TITLE_NORM" in df_year.columns:
    top_jobs_df = (df_year.groupby("JOB_TITLE_NORM")
                          .size()
                          .reset_index(name="filings")
                          .sort_values("filings", ascending=False)
                          .head(25))
    fig_jobs = px.bar(top_jobs_df, x="JOB_TITLE_NORM", y="filings")
    fig_jobs.update_layout(xaxis_title="", yaxis_title="Filings", margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig_jobs, use_container_width=True)
    st.dataframe(top_jobs_df, use_container_width=True)

# ------------------- TOP SOC OCCUPATIONS -------------------
st.subheader("🏷 Top-25 SOC Occupations")
if "SOC_TITLE_NORM" in df_year.columns:
    top_soc = (df_year.groupby(["SOC_CODE","SOC_TITLE_NORM"])
                      .size()
                      .reset_index(name="filings")
                      .sort_values("filings", ascending=False)
                      .head(25))
    fig_soc = px.bar(top_soc, x="SOC_TITLE_NORM", y="filings", hover_data=["SOC_CODE"])
    fig_soc.update_layout(xaxis_title="", yaxis_title="Filings", margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig_soc, use_container_width=True)
    st.dataframe(top_soc, use_container_width=True)

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

# ------------------- FISCAL QUARTER CHART (optional) -------------------
if mode.startswith("Fiscal") and "FISCAL_QUARTER" in df_year.columns:
    st.subheader("📆 Filings by Fiscal Quarter")
    by_q = (df_year.groupby("FISCAL_QUARTER").size()
            .reindex([1,2,3,4], fill_value=0)
            .reset_index(name="filings"))
    fig_q = px.bar(by_q, x="FISCAL_QUARTER", y="filings")
    fig_q.update_layout(xaxis_title="Fiscal Quarter", yaxis_title="Filings", margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig_q, use_container_width=True)

# ------------------- SAMPLE ROWS WITH FISCAL INFO -------------------
st.caption("Sample records (with fiscal date info)")
cols_to_show = [c for c in ["PRIMARY_DATE_STR", "FY_LABEL", "FISCAL_QUARTER",
                            "EMPLOYER_NAME","JOB_TITLE","SOC_CODE","SOC_TITLE",
                            "WORKSITE_STATE","WAGE_ANNUAL_FROM"] if c in df_year.columns]
if cols_to_show:
    st.dataframe(df_year[cols_to_show].head(50), use_container_width=True)

# ------------------- EXPORT CURRENT SELECTION -------------------
st.subheader("⬇️ Export current selection")
export_cols = [c for c in ["PRIMARY_DATE","FISCAL_YEAR","RECEIVED_YEAR","WORKSITE_STATE",
                           "EMPLOYER_NAME","JOB_TITLE","SOC_CODE","SOC_TITLE",
                           "WAGE_ANNUAL_FROM","WAGE_UNIT_OF_PAY"] if c in df_year.columns]
csv_bytes = df_year[export_cols].to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download CSV",
    data=csv_bytes,
    file_name=f"lca_selection_{year}_{'FY' if year_col=='FISCAL_YEAR' else 'CY'}.csv",
    mime="text/csv"
)
