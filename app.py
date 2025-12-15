import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import os

# =========================
# PAGE CONFIG (MUST BE FIRST)
# =========================
st.set_page_config(
    page_title="Automated Data Cleaning Agent",
    layout="wide"
)

# =========================
# HIDE STREAMLIT CLOUD UI
# =========================
st.markdown("""
<style>
/* Hide main Streamlit UI */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}

/* Dim floating cloud overlay (cannot fully remove) */
[data-testid="stToolbar"] {
    opacity: 0.05;
    pointer-events: none;
}
</style>
""", unsafe_allow_html=True)


# =========================
# DATE STANDARDIZATION (STRING-BASED)
# =========================
def normalize_dates_as_string(series):
    def format_date(v):
        if pd.isna(v):
            return "Unknown"

        v = str(v).strip()

        if v.lower() in ["", "nan", "none", "null"]:
            return "Unknown"

        try:
            dt = pd.to_datetime(v, errors="coerce")
            if pd.isna(dt):
                return v  # keep original text
            return dt.strftime("%d-%m-%y")
        except Exception:
            return v

    return series.apply(format_date)

# =========================
# COLUMN INTENT DETECTION
# =========================
def detect_column_intent(df):
    intents = {}
    for col in df.columns:
        name = col.lower()
        if name.endswith("id") or name == "id":
            intents[col] = "identifier"
        elif "date" in name or "year" in name:
            intents[col] = "date"
        elif pd.api.types.is_numeric_dtype(df[col]):
            intents[col] = "numeric"
        else:
            intents[col] = "categorical"
    return intents

# =========================
# OUTLIER DETECTION (READ ONLY)
# =========================
def detect_outliers(df, intents):
    rows = []
    for col in df.columns:
        if intents[col] != "numeric":
            continue

        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = ((df[col] < lower) | (df[col] > upper)).sum()

        rows.append([col, int(count), round(lower, 2), round(upper, 2)])

    return pd.DataFrame(
        rows,
        columns=["Column", "Outlier Count", "Lower Bound", "Upper Bound"]
    )

# =========================
# CLEANING + SUMMARY TABLE
# =========================
def clean_dataset_and_summary(df, intents):
    summary = []

    # Date columns
    date_cols = [c for c in df.columns if intents[c] == "date"]
    if date_cols:
        for c in date_cols:
            df[c] = normalize_dates_as_string(df[c])
        summary.append({
            "Column Type": "Date",
            "Columns Affected": ", ".join(date_cols),
            "Change Applied": "Standardized date format",
            "Value Used": "DD-MM-YY, missing → 'Unknown'"
        })

    # Numeric columns
    numeric_cols = [c for c in df.columns if intents[c] == "numeric"]
    if numeric_cols:
        for c in numeric_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].fillna(df[c].median())
        summary.append({
            "Column Type": "Numeric",
            "Columns Affected": ", ".join(numeric_cols),
            "Change Applied": "Missing values handled",
            "Value Used": "Median of each column"
        })

    # Categorical columns
    categorical_cols = [c for c in df.columns if intents[c] == "categorical"]
    if categorical_cols:
        for c in categorical_cols:
            df[c] = df[c].fillna("Unknown")
        summary.append({
            "Column Type": "Categorical",
            "Columns Affected": ", ".join(categorical_cols),
            "Change Applied": "Missing values handled",
            "Value Used": "'Unknown'"
        })

    # Duplicate handling
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)

    summary.append({
        "Column Type": "Duplicates",
        "Columns Affected": "All columns",
        "Change Applied": "Duplicate rows removed",
        "Value Used": f"{removed} rows removed"
    })

    return df, pd.DataFrame(summary)

# =========================
# UI
# =========================
st.title("🧹 Automated Data Cleaning Agent")

uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    ["csv", "xlsx"]
)

if uploaded_file:
    df = (
        pd.read_excel(uploaded_file)
        if uploaded_file.name.endswith(".xlsx")
        else pd.read_csv(uploaded_file)
    )

    intents = detect_column_intent(df)

    st.subheader("Raw Data")
    st.dataframe(df)

    st.subheader("Outlier Detection (Informational Only)")
    st.dataframe(detect_outliers(df, intents))

    cleaned_df, summary_df = clean_dataset_and_summary(df.copy(), intents)

    st.subheader("Cleaned Data")
    st.dataframe(cleaned_df)

    st.subheader("Cleaning Summary (Clear & Interpretable)")
    st.dataframe(summary_df)

    base = os.path.splitext(uploaded_file.name)[0]

    st.subheader("⬇️ Download Cleaned Dataset")

    st.download_button(
        "Download CSV",
        cleaned_df.to_csv(index=False),
        f"{base}_cleaned.csv"
    )

    excel_buf = BytesIO()
    cleaned_df.to_excel(excel_buf, index=False)
    st.download_button(
        "Download Excel",
        excel_buf.getvalue(),
        f"{base}_cleaned.xlsx"
    )


