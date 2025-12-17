import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import tempfile
import os

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="AI Data Cleaning Agent", layout="wide")

# =====================================================
# PROFILER AGENT
# =====================================================
def profiler_agent(df):
    return {
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Missing Values": df.isnull().sum(),
        "Duplicate Rows": df.duplicated().sum(),
        "Data Types": df.dtypes
    }

# =====================================================
# CLEANER AGENT (LOGIC UNCHANGED)
# =====================================================
def cleaner_agent(df):
    df = df.copy()

    report = []
    outlier_summary = {}
    imputation_summary = {}

    df.columns = df.columns.str.lower().str.replace(" ", "_")

    # 1️⃣ Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    report.append(f"Removed {before - len(df)} duplicate rows")

    # 2️⃣ Outlier fixing
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if df[col].dropna().nunique() <= 2:
            continue

        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        if IQR == 0 or pd.isna(IQR):
            continue

        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        mask = (df[col] < lower) | (df[col] > upper)
        count = int(mask.sum())
        if count == 0:
            continue

        if abs(df[col].skew()) > 1:
            df.loc[mask, col] = df[col].median()
            method, reason = "Median", "Skewed distribution"
        else:
            df.loc[mask, col] = df[col].mean()
            method, reason = "Mean", "Near-normal distribution"

        outlier_summary[col] = {
            "Outliers Fixed": count,
            "Method Used": method,
            "Reason": reason
        }

    report.append("Detected and fixed outliers using IQR")

    # 3️⃣ Missing values
    cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns
    for col in df.columns:
        missing = int(df[col].isnull().sum())
        if missing == 0:
            continue

        if col in cat_cols:
            if df[col].nunique(dropna=True) < 20:
                fill, method, reason = (
                    df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown",
                    "Mode",
                    "Low cardinality categorical column",
                )
            else:
                fill, method, reason = "Unknown", "Unknown", "High cardinality categorical column"
        else:
            if abs(df[col].skew()) > 1:
                fill, method, reason = df[col].median(), "Median", "Skewed numeric distribution"
            else:
                fill, method, reason = df[col].mean(), "Mean", "Near-normal numeric distribution"

        df[col] = df[col].fillna(fill)
        imputation_summary[col] = {
            "Missing Filled": missing,
            "Method Used": method,
            "Reason": reason
        }

    report.append("Filled missing values using intelligent strategies")
    return df, report, outlier_summary, imputation_summary

# =====================================================
# VALIDATOR
# =====================================================
def validator_agent(df):
    score = 100
    score -= int(df.isnull().sum().sum()) * 2
    score -= int(df.duplicated().sum()) * 5
    return max(score, 0)

# =====================================================
# FILE LOADER
# =====================================================
def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)

    elif file.name.endswith(".xlsx"):
        return pd.read_excel(file)

    elif file.name.endswith(".db"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            tmp.write(file.getbuffer())
            path = tmp.name
        conn = sqlite3.connect(path)
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
        table = st.selectbox("Select SQL Table", tables["name"])
        df = pd.read_sql(f"SELECT * FROM {table}", conn)
        conn.close()
        os.remove(path)
        return df

# =====================================================
# UI
# =====================================================
st.title("🧠 AI Data Cleaning Agent")
st.write("Explainable, Agentic, Industry-Grade Data Cleaning System")

uploaded_file = st.file_uploader(
    "📂 Upload Dataset (CSV / XLSX / SQLite)",
    type=["csv", "xlsx", "db"]
)

if uploaded_file:
    df = load_data(uploaded_file)
    profile = profiler_agent(df)

    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head(100))

    # =====================================================
    # FIX PROPOSALS
    # =====================================================
    st.subheader("🤖 Agent Fix Proposals")

    fix_duplicates = st.checkbox("Remove duplicate rows", True)
    fix_outliers = st.checkbox("Fix numeric outliers", True)
    fix_missing = st.checkbox("Fill missing values", True)

    if st.button("🚀 Apply Approved Fixes"):
        cleaned_df, report, outliers, imputations = cleaner_agent(df)
        score = validator_agent(cleaned_df)

        st.subheader("📊 Cleaning Report")
        for r in report:
            st.success(r)

        st.subheader("🚨 Outlier Fixing")
        st.dataframe(pd.DataFrame(outliers).T.reset_index(names=["Column"]))

        st.subheader("🩹 Missing Value Imputation")
        st.dataframe(pd.DataFrame(imputations).T.reset_index(names=["Column"]))

        # =====================================================
        # TRANSFORMATION RECOMMENDATIONS
        # =====================================================
        st.subheader("📌 Recommended Next Transformations")
        st.markdown("""
        - 📊 **Scaling**: Use StandardScaler for numeric features  
        - 🔤 **Encoding**: One-hot encode categorical variables  
        - 📉 **Log Transform**: Apply to highly skewed numeric columns  
        - 🧹 **Drop ID Columns** before ML  
        """)

        st.subheader("📈 Data Quality Score")
        st.progress(score / 100)
        st.write(f"Score: **{score}/100**")

        st.subheader("✅ Cleaned Dataset")
        st.dataframe(cleaned_df.head(100))

        st.download_button(
            "⬇ Download Cleaned Dataset",
            cleaned_df.to_csv(index=False).encode("utf-8"),
            "cleaned_dataset.csv",
            "text/csv"
        )

else:
    st.info("Please upload a dataset to start cleaning")
