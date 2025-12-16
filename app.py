import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, time
import warnings

warnings.filterwarnings("ignore")

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Intelligent Data Cleaning System",
    layout="wide"
)

st.title("🧠 Intelligent Data Cleaning System")



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
# HELPER FUNCTIONS
# =========================
def normalize_missing(df):
    return df.replace(["None", "none", "NULL", "null", ""], np.nan)

def safe_parse_date(val):
    """Parse date but keep as datetime internally"""
    if pd.isna(val):
        return pd.NaT
    try:
        if isinstance(val, (datetime, pd.Timestamp)):
            return val
        m, d, y = map(int, str(val).split("/"))
        y = y + 2000 if y < 25 else y + 1900
        return datetime(y, m, d)
    except:
        return pd.NaT

def safe_parse_time(val):
    if pd.isna(val):
        return None
    try:
        if isinstance(val, time):
            return val
        return datetime.strptime(str(val), "%H:%M").time()
    except:
        return None

def numeric_strategy(series):
    if series.nunique(dropna=True) <= 10:
        return "mode"
    if abs(series.skew()) > 1:
        return "median"
    return "mean"

def fill_numeric(series):
    method = numeric_strategy(series)
    if method == "mean":
        return series.fillna(series.mean()), method
    if method == "median":
        return series.fillna(series.median()), method
    mode = series.mode()
    return series.fillna(mode[0]) if not mode.empty else series, method

def make_arrow_safe(df):
    """Convert dataframe to Arrow-compatible types for display"""
    safe_df = df.copy()

    for col in safe_df.columns:
        if pd.api.types.is_datetime64_any_dtype(safe_df[col]):
            safe_df[col] = safe_df[col].dt.strftime("%Y-%m-%d")

        elif safe_df[col].apply(lambda x: isinstance(x, time)).any():
            safe_df[col] = safe_df[col].astype(str)

        elif safe_df[col].dtype == "object":
            safe_df[col] = safe_df[col].astype(str)

    return safe_df

# =========================
# MAIN CLEANING FUNCTION
# =========================
def clean_dataframe(df):
    df = normalize_missing(df)
    cleaned = df.copy()
    log = []

    for col in cleaned.columns:
        series = cleaned[col]

        # DATE
        if "date" in col.lower():
            cleaned[col] = series.apply(safe_parse_date)
            log.append(f"{col}: parsed as date")
            continue

        # TIME
        if "time" in col.lower():
            cleaned[col] = series.apply(safe_parse_time)
            log.append(f"{col}: parsed as time")
            continue

        # NUMERIC
        numeric_series = pd.to_numeric(series, errors="coerce")
        if numeric_series.notna().sum() >= len(series) * 0.7:
            filled, method = fill_numeric(numeric_series)
            cleaned[col] = filled
            log.append(f"{col}: numeric → filled using {method}")
            continue

        # CATEGORICAL
        if series.dtype == "object":
            mode = series.mode()
            if not mode.empty:
                cleaned[col] = series.fillna(mode[0])
                log.append(f"{col}: categorical → filled using mode")
            continue

    return cleaned, log

# =========================
# STREAMLIT UI
# =========================
uploaded_file = st.file_uploader("📤 Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Raw Data")
    st.dataframe(make_arrow_safe(df.head(30)))

    cleaned_df, cleaning_log = clean_dataframe(df)

    st.subheader("🧹 Cleaned Data")
    st.dataframe(make_arrow_safe(cleaned_df.head(30)))

    with st.expander("🧠 Cleaning Log"):
        for entry in cleaning_log:
            st.write(entry)

    st.subheader("📊 Summary Statistics")
    st.dataframe(make_arrow_safe(cleaned_df.describe(include="all")))
