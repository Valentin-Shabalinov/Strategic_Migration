# 📈 Strategic Talent Migration — LCA Data Analysis MVP

---

## 📌 Project Overview

**Strategic Talent Migration — LCA MVP** is a **data processing and analytics platform** for exploring **Labor Condition Applications (LCA)** filed by U.S. employers when sponsoring foreign workers under **H-1B** and similar visa categories.

### **Main Components**
1. **ETL Pipeline** — [`etl_lca.py`](etl_lca.py)  
   Extracts, transforms, and cleans LCA datasets from Excel files into a standardized format.  
2. **Interactive Dashboard** — [`app.py`](app.py)  
   A **Streamlit-based** web interface for filtering, analyzing, and visualizing LCA data.

---

### **Purpose & Use Cases**
- 📍 Researching **talent migration patterns** across the United States.
- 🏢 Identifying **industries and regions** with high demand for international talent.
- 📊 Supporting **business intelligence** and consulting use cases in labor market analysis.
- 🏛 Assisting **policy makers** in workforce planning and immigration policy evaluation.

---

## 📊 Key Features

✅ Integration of multiple quarterly LCA disclosure datasets  
✅ Standardized formats:
- `PRIMARY_DATE`, `RECEIVED_YEAR`, `FISCAL_YEAR`
- Normalized employer names and worksite locations
- Annualized wage calculation  

✅ Deduplication by **CASE_NUMBER**  
✅ Interactive filtering by:
- Calendar year / fiscal year
- State
- Employer name (partial match)  

✅ KPIs:
- Total filings
- Unique employers
- Median annual wage  

✅ Visualizations:
- **Top-25 Employers** by number of filings
- **State-level distribution** of filings
- **Annual wage distribution**
- **Monthly filing trends**  

✅ Data diagnostics for available years

---


## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/strategic-talent-migration.git
```
cd strategic-talent-migration

### 2️⃣ Create a Virtual Environment

```bash
python3 -m venv .venv
```
#### Activate the virtual environment:
#### Mac/Linux:

```bash
source .venv/bin/activate
```

#### Windows (PowerShell):
```bash
.venv\Scripts\activate
```
### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```
#### If you don't have requirements.txt, generate it:

```bash
pip install pandas numpy streamlit plotly openpyxl pyarrow fastparquet pip freeze > requirements.txt
```

### 🛠 Running the Project
Step 1 — ETL Processing
Run the ETL script to:
- Load raw Excel LCA files from data/
- Standardize fields and formats
- Deduplicate cases
- Save cleaned data in CSV and Parquet
```bash
python etl_lca.py
```
Expected output:
```yaml
Reading: data/LCA_Disclosure_Data_FY2024_Q1.xlsx
Reading: data/LCA_Disclosure_Data_FY2023_Q4.xlsx
Rows after merge & dedup: 226,843
Saved: data/lca_merged_clean.csv
Saved: data/lca_merged_clean.parquet
```
Step 2 — Launch the Dashboard
```bash
streamlit run app.py
```
The interactive dashboard will open in your browser at:
```bash
http://localhost:8501
```
### 🖥 Dashboard Usage
- Select Year Type — Calendar or Fiscal.
- Choose Filters — State and/or employer name.
- View KPIs — Total filings, unique employers, median wage.
- Explore Visualizations:
  - Top-25 employers by LCA filings
  - Filings by state
  - Wage distribution histogram
  - Monthly filing trends
- Check Data Coverage — See available years in diagnostics.
### 📌 Notes
- Data source: U.S. Department of Labor — Office of Foreign Labor Certification (OFLC) LCA Disclosure Data.
- Place raw Excel files in the data/ directory before running ETL.
- Fiscal year logic:
  - Oct–Dec → next year
  - Jan–Sep → current year
### 📄 License
Provided for research and analysis purposes.
For commercial use, ensure compliance with U.S. Department of Labor data policies.



