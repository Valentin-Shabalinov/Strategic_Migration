import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ------------------- APP CONFIG -------------------
st.set_page_config(page_title="Strategic Talent Migration — LCA", layout="wide")
st.title("📈 Strategic Talent Migration — LCA")
st.caption("Interactive dashboard based on LCA disclosure data (Labor Condition Applications).")


# ------------------- DATA PATHS -------------------
DATA_PARQUET = Path("data/lca_merged_clean.parquet")
DATA_CSV = Path("data/lca_merged_clean.csv")


# ------------------- HELPERS -------------------
def fiscal_year_range(fy: int):
    return date(fy - 1, 10, 1), date(fy, 9, 30)


def fiscal_quarter_from_month(m: int) -> int:
    if m in (10, 11, 12):
        return 1
    if m in (1, 2, 3):
        return 2
    if m in (4, 5, 6):
        return 3
    return 4  # 7,8,9


def norm_series(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str).str.strip().str.upper()
    return s.str.replace(r"\s+", " ", regex=True)


@st.cache_data(show_spinner="Loading dataset…")
def load_data() -> pd.DataFrame:
    if DATA_PARQUET.exists():
        df = pd.read_parquet(DATA_PARQUET)
    elif DATA_CSV.exists():
        # Read once; then convert date columns safely
        df = pd.read_csv(DATA_CSV)
    else:
        st.error("Dataset not found. Put file into /data: lca_merged_clean.parquet or lca_merged_clean.csv")
        st.stop()

    # Ensure date columns
    date_cols = ["RECEIVED_DATE", "DECISION_DATE", "ORIGINAL_CERT_DATE", "BEGIN_DATE", "END_DATE", "PRIMARY_DATE"]
    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # PRIMARY_DATE (first available)
    if "PRIMARY_DATE" not in df.columns or df["PRIMARY_DATE"].isna().all():
        df["PRIMARY_DATE"] = pd.NaT
        for c in ["RECEIVED_DATE", "DECISION_DATE", "BEGIN_DATE", "ORIGINAL_CERT_DATE", "END_DATE"]:
            if c in df.columns:
                df["PRIMARY_DATE"] = df["PRIMARY_DATE"].fillna(df[c])

    # Calendar helpers
    if "RECEIVED_YEAR" not in df.columns or pd.isna(df["RECEIVED_YEAR"]).all():
        df["RECEIVED_YEAR"] = df["PRIMARY_DATE"].dt.year
    if "RECEIVED_MONTH" not in df.columns or pd.isna(df["RECEIVED_MONTH"]).all():
        df["RECEIVED_MONTH"] = df["PRIMARY_DATE"].dt.to_period("M").astype("string")

    # Fiscal year
    if "FISCAL_YEAR" not in df.columns or pd.isna(df["FISCAL_YEAR"]).all():
        m = df["PRIMARY_DATE"].dt.month
        y = df["PRIMARY_DATE"].dt.year
        df["FISCAL_YEAR"] = np.where(m >= 10, y + 1, y)

    # Normalized fields (vectorized)
    if "EMPLOYER_NAME" in df.columns and "EMPLOYER_NAME_NORM" not in df.columns:
        df["EMPLOYER_NAME_NORM"] = norm_series(df["EMPLOYER_NAME"])
    if "JOB_TITLE" in df.columns and "JOB_TITLE_NORM" not in df.columns:
        df["JOB_TITLE_NORM"] = norm_series(df["JOB_TITLE"])
    if "SOC_TITLE" in df.columns and "SOC_TITLE_NORM" not in df.columns:
        df["SOC_TITLE_NORM"] = norm_series(df["SOC_TITLE"])

    # Annual wage (vectorized)
    if "WAGE_ANNUAL_FROM" not in df.columns:
        v = pd.to_numeric(df.get("WAGE_RATE_OF_PAY_FROM"), errors="coerce")
        unit = df.get("WAGE_UNIT_OF_PAY")
        unit = unit.astype(str).str.lower() if unit is not None else pd.Series("", index=df.index)

        mult = np.select(
            [
                unit.str.contains("hour", na=False),
                unit.str.contains("week", na=False),
                unit.str.contains("bi", na=False),
                unit.str.contains("month", na=False),
                unit.str.contains("year", na=False),
            ],
            [2080, 52, 26, 12, 1],
            default=np.nan,
        )
        df["WAGE_ANNUAL_FROM"] = v * mult

    return df


df = load_data()


# ------------------- SIDEBAR FILTERS -------------------
st.sidebar.header("Filters")

mode = st.sidebar.radio("Year type", ["Calendar Year", "Fiscal Year"])
year_col = "RECEIVED_YEAR" if mode.startswith("Calendar") else "FISCAL_YEAR"

years = sorted([int(y) for y in df[year_col].dropna().unique()])
if not years:
    st.error("No valid years found in the dataset.")
    st.stop()

year = st.sidebar.selectbox("Year", options=years, index=len(years) - 1)

if mode.startswith("Fiscal"):
    fy_start, fy_end = fiscal_year_range(year)
    st.sidebar.info(f"**FY {year}**: {fy_start:%b %d, %Y} — {fy_end:%b %d, %Y}")
else:
    st.sidebar.info(f"**CY {year}**: Jan 01, {year} — Dec 31, {year}")

states = sorted(df.get("WORKSITE_STATE", pd.Series([], dtype="object")).dropna().astype(str).unique())
state = st.sidebar.selectbox("State (WORKSITE_STATE)", options=["All"] + states)

emp_query = st.sidebar.text_input("Employer filter (substring)", "")

st.sidebar.markdown("---")
st.sidebar.subheader("Job / Occupation")

job_query = st.sidebar.text_input("Search in JOB_TITLE (substring)", "")

top_jobs = (
    df["JOB_TITLE_NORM"].dropna().value_counts().head(200).index.tolist()
    if "JOB_TITLE_NORM" in df.columns
    else []
)
top_soc_titles = (
    df["SOC_TITLE_NORM"].dropna().value_counts().head(200).index.tolist()
    if "SOC_TITLE_NORM" in df.columns
    else []
)

job_pick = st.sidebar.multiselect("Popular JOB_TITLE", options=top_jobs)
soc_code_pick = st.sidebar.text_input("SOC_CODE (exact/prefix, e.g., 15-12)", "")
soc_title_pick = st.sidebar.multiselect("Popular SOC_TITLE", options=top_soc_titles)


# ------------------- APPLY FILTERS -------------------
df_year = df[df[year_col] == year].copy()

if state != "All" and "WORKSITE_STATE" in df_year.columns:
    df_year = df_year[df_year["WORKSITE_STATE"].astype(str) == state]

if emp_query.strip() and "EMPLOYER_NAME_NORM" in df_year.columns:
    q = emp_query.strip().upper()
    df_year = df_year[df_year["EMPLOYER_NAME_NORM"].str.contains(q, na=False)]

if job_query.strip() and "JOB_TITLE_NORM" in df_year.columns:
    q = job_query.strip().upper()
    df_year = df_year[df_year["JOB_TITLE_NORM"].str.contains(q, na=False)]

if job_pick and "JOB_TITLE_NORM" in df_year.columns:
    df_year = df_year[df_year["JOB_TITLE_NORM"].isin(job_pick)]

if soc_code_pick.strip() and "SOC_CODE" in df_year.columns:
    q = soc_code_pick.strip()
    df_year = df_year[df_year["SOC_CODE"].astype(str).str.startswith(q, na=False)]

if soc_title_pick and "SOC_TITLE_NORM" in df_year.columns:
    df_year = df_year[df_year["SOC_TITLE_NORM"].isin(soc_title_pick)]


# Enrich
if "PRIMARY_DATE" in df_year.columns:
    dt = pd.to_datetime(df_year["PRIMARY_DATE"], errors="coerce")
    df_year["PRIMARY_DATE_STR"] = dt.dt.strftime("%Y-%m-%d")
    df_year["FISCAL_QUARTER"] = dt.dt.month.map(fiscal_quarter_from_month)
    if "FISCAL_YEAR" in df_year.columns:
        df_year["FY_LABEL"] = "FY" + df_year["FISCAL_YEAR"].astype("Int64").astype("string")


# ------------------- KPI METRICS -------------------
total_filings = len(df_year)
unique_employers = (
    df_year["EMPLOYER_NAME_NORM"].nunique()
    if "EMPLOYER_NAME_NORM" in df_year.columns
    else df_year.get("EMPLOYER_NAME", pd.Series([])).nunique()
)
median_wage = pd.to_numeric(df_year.get("WAGE_ANNUAL_FROM"), errors="coerce").median()

k1, k2, k3 = st.columns(3)
k1.metric("Filings (after filter)", f"{total_filings:,}")
k2.metric("Unique employers", f"{unique_employers:,}")
k3.metric("Median annual wage (USD)", f"{int(median_wage):,}" if pd.notna(median_wage) else "—")


# ------------------- TOP EMPLOYERS -------------------
st.subheader("🔝 Top-25 Employers by LCA Filings")
if "EMPLOYER_NAME" in df_year.columns:
    top_emp = (
        df_year.groupby("EMPLOYER_NAME")
        .size()
        .reset_index(name="filings")
        .sort_values("filings", ascending=False)
        .head(25)
    )
    fig_emp = px.bar(top_emp, x="EMPLOYER_NAME", y="filings")
    fig_emp.update_layout(xaxis_title="", yaxis_title="Filings", margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_emp, use_container_width=True)
    st.dataframe(top_emp, use_container_width=True)


# ------------------- TOP JOB TITLES -------------------
st.subheader("🧑‍💻 Top-25 Job Titles")
if "JOB_TITLE_NORM" in df_year.columns:
    top_jobs_df = (
        df_year.groupby("JOB_TITLE_NORM")
        .size()
        .reset_index(name="filings")
        .sort_values("filings", ascending=False)
        .head(25)
    )
    fig_jobs = px.bar(top_jobs_df, x="JOB_TITLE_NORM", y="filings")
    fig_jobs.update_layout(xaxis_title="", yaxis_title="Filings", margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_jobs, use_container_width=True)
    st.dataframe(top_jobs_df, use_container_width=True)


# ------------------- TOP SOC OCCUPATIONS -------------------
st.subheader("🏷 Top-25 SOC Occupations")
if "SOC_TITLE_NORM" in df_year.columns and "SOC_CODE" in df_year.columns:
    top_soc = (
        df_year.groupby(["SOC_CODE", "SOC_TITLE_NORM"])
        .size()
        .reset_index(name="filings")
        .sort_values("filings", ascending=False)
        .head(25)
    )
    fig_soc = px.bar(top_soc, x="SOC_TITLE_NORM", y="filings", hover_data=["SOC_CODE"])
    fig_soc.update_layout(xaxis_title="", yaxis_title="Filings", margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_soc, use_container_width=True)
    st.dataframe(top_soc, use_container_width=True)


# ------------------- FILINGS BY STATE -------------------
st.subheader("🌎 Filings by State (Worksite)")
if "WORKSITE_STATE" in df_year.columns:
    by_state = (
        df_year.groupby("WORKSITE_STATE")
        .size()
        .reset_index(name="filings")
        .sort_values("filings", ascending=False)
    )
    fig_state = px.bar(by_state, x="WORKSITE_STATE", y="filings")
    fig_state.update_layout(xaxis_title="State", yaxis_title="Filings", margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_state, use_container_width=True)
    st.dataframe(by_state, use_container_width=True)


# ------------------- MONTHLY TREND (ALL YEARS) -------------------
st.subheader("📅 Monthly Filing Trend (All Years, Calendar Months)")
if "RECEIVED_MONTH" in df.columns:
    by_month = (
        df.dropna(subset=["RECEIVED_MONTH"])
        .groupby("RECEIVED_MONTH")
        .size()
        .reset_index(name="filings")
    )
    by_month["_ts"] = pd.PeriodIndex(by_month["RECEIVED_MONTH"], freq="M").to_timestamp()
    by_month = by_month.sort_values("_ts")
    fig_month = px.line(by_month, x="RECEIVED_MONTH", y="filings")
    fig_month.update_layout(xaxis_title="Month", yaxis_title="Filings", margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_month, use_container_width=True)
    st.dataframe(by_month[["RECEIVED_MONTH", "filings"]], use_container_width=True)


# ------------------- FISCAL QUARTER CHART -------------------
if mode.startswith("Fiscal") and "FISCAL_QUARTER" in df_year.columns:
    st.subheader("📆 Filings by Fiscal Quarter")
    by_q = (
        df_year.groupby("FISCAL_QUARTER")
        .size()
        .reindex([1, 2, 3, 4], fill_value=0)
        .reset_index(name="filings")
    )
    fig_q = px.bar(by_q, x="FISCAL_QUARTER", y="filings")
    fig_q.update_layout(xaxis_title="Fiscal Quarter", yaxis_title="Filings", margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_q, use_container_width=True)


# ------------------- SAMPLE ROWS -------------------
st.caption("Sample records (with fiscal date info)")
cols_to_show = [
    c
    for c in [
        "PRIMARY_DATE_STR",
        "FY_LABEL",
        "FISCAL_QUARTER",
        "EMPLOYER_NAME",
        "JOB_TITLE",
        "SOC_CODE",
        "SOC_TITLE",
        "WORKSITE_STATE",
        "WAGE_ANNUAL_FROM",
    ]
    if c in df_year.columns
]
if cols_to_show:
    st.dataframe(df_year[cols_to_show].head(50), use_container_width=True)


# ------------------- EXPORT CURRENT SELECTION -------------------
st.subheader("⬇️ Export current selection")
export_cols = [
    c
    for c in [
        "PRIMARY_DATE",
        "FISCAL_YEAR",
        "RECEIVED_YEAR",
        "WORKSITE_STATE",
        "EMPLOYER_NAME",
        "JOB_TITLE",
        "SOC_CODE",
        "SOC_TITLE",
        "WAGE_ANNUAL_FROM",
        "WAGE_UNIT_OF_PAY",
    ]
    if c in df_year.columns
]
csv_bytes = df_year[export_cols].to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download CSV",
    data=csv_bytes,
    file_name=f"lca_selection_{year}_{'FY' if year_col=='FISCAL_YEAR' else 'CY'}.csv",
    mime="text/csv",
)
