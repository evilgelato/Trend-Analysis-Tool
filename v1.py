import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF
import tempfile
import openai
import json
import re

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Trend & Root Cause Analyzer", layout="wide")
openai.api_key = st.secrets["OPENAI_API_KEY"]

# ---------------- ANOMALY DETECTION ----------------
def detect_anomalies(df, threshold=3):
    anomalies = {}
    for col in df.select_dtypes(include=np.number).columns:
        mean = df[col].mean()
        std = df[col].std()
        if std == 0:
            continue
        z_scores = (df[col] - mean) / std
        anomaly_points = df.loc[np.abs(z_scores) > threshold, [col]]
        if not anomaly_points.empty:
            anomalies[col] = anomaly_points[col].tolist()
    return anomalies

# ---------------- GPT ANALYSIS ----------------
def gpt_analysis_with_causes(stats, trends, anomalies):
    prompt = f"""
    You are a statistical analyst and root cause expert.
    Analyze the dataset summary and trend slopes.
    Highlight any anomalies in the data: {anomalies}
    Return:
    1. A detailed human-readable analysis (trends, anomalies, inferences, 5 Whys, recommendations).
    2. A JSON dictionary mapping Fishbone categories to a list of possible causes.

    Dataset Summary:
    {stats.to_string()}
    Trend Slopes:
    {trends}

    Respond in the following exact format:
    ---
    ANALYSIS:
    <your human-readable analysis here>
    ---
    FISHBONE_JSON:
    {{"People": ["Cause 1", "Cause 2"], "Process": ["Cause 1"], "Equipment": ["Cause 1"], "Materials": ["Cause 1"], "Environment": ["Cause 1"], "Measurement": ["Cause 1"]}}
    """
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert in trend analysis and root cause investigation."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4
    )
    content = response.choices[0].message["content"]

    try:
        analysis_part = content.split("---")[1].replace("ANALYSIS:", "").strip()
        json_part = content.split("---")[2].replace("FISHBONE_JSON:", "").strip()
        causes_dict = json.loads(json_part)
    except Exception:
        analysis_part = content
        causes_dict = {
            "People": ["Could not parse AI output"],
            "Process": [],
            "Equipment": [],
            "Materials": [],
            "Environment": [],
            "Measurement": []
        }
    return analysis_part, causes_dict

# ---------------- VISUALIZATIONS ----------------
def plot_trends(df, time_col):
    fig, ax = plt.subplots()
    for col in df.select_dtypes(include=np.number).columns:
        ax.plot(df[time_col], df[col], marker='o', label=col)
    ax.set_title("Trend Over Time")
    ax.set_xlabel(time_col)
    ax.set_ylabel("Values")
    ax.legend()
    return fig

def pareto_chart(df, column):
    counts = df[column].value_counts().sort_values(ascending=False)
    cum_perc = counts.cumsum() / counts.sum() * 100
    fig, ax1 = plt.subplots()
    counts.plot(kind='bar', ax=ax1)
    ax1.set_ylabel('Frequency')
    ax2 = ax1.twinx()
    cum_perc.plot(marker='o', color='red', ax=ax2)
    ax2.set_ylabel('Cumulative %')
    ax2.axhline(80, color='gray', linestyle='--')
    ax1.set_title(f"Pareto Chart - {column}")
    return fig

def fishbone_diagram(causes_dict):
    fig, ax = plt.subplots(figsize=(10,6))
    ax.set_title("Fishbone Diagram", fontsize=14)
    center_y = 0.5
    ax.plot([0.2, 0.8], [center_y, center_y], color='black')
    categories = list(causes_dict.keys())
    for i, category in enumerate(categories):
        y = center_y + (0.3 if i % 2 == 0 else -0.3)
        ax.plot([0.5, 0.8 if i % 2 == 0 else 0.2], [center_y, y], color='black')
        ax.text(0.82 if i % 2 == 0 else 0.05, y, category, fontsize=10, va='center')
        for j, cause in enumerate(causes_dict[category]):
            offset_y = y + (0.05 * (j - len(causes_dict[category]) / 2))
            ax.plot([0.8 if i % 2 == 0 else 0.2, 0.85 if i % 2 == 0 else 0.15],
                    [y, offset_y], color='black')
            ax.text(0.87 if i % 2 == 0 else 0.02, offset_y, cause, fontsize=8, va='center')
    ax.axis('off')
    return fig

# ---------------- PARETO AUTO-SELECTION ----------------
def auto_select_pareto_column(df, time_col):
    keywords = ["defect", "error", "issue", "cause", "fail", "problem", "reject"]
    candidate_cols = [c for c in df.columns if c != time_col]
    for col in candidate_cols:
        if any(kw in col.lower() for kw in keywords):
            return col
    for col in candidate_cols:
        if df[col].dtype == object or (df[col].dtype in [np.int64, np.float64] and df[col].nunique() < 20):
            return col
    return candidate_cols[0] if candidate_cols else None

# ---------------- PDF REPORT ----------------
class PDFReport(FPDF):
    def header(self):
        self.set_font("Arial", 'B', 16)
        self.cell(0, 10, "Data Analysis Report", ln=True, align="C")
        self.ln(5)

    def chapter_title(self, title):
        self.set_font("Arial", 'B', 14)
        self.cell(0, 10, title, ln=True)
        self.ln(2)

    def chapter_body(self, body):
        self.set_font("Arial", '', 12)
        self.multi_cell(0, 8, body)
        self.ln()

def generate_pdf(analysis_text, chart_paths):
    pdf = PDFReport()
    pdf.add_page()
    pdf.chapter_title("Trend Analysis & Root Cause Investigation")
    pdf.chapter_body(analysis_text.encode('ascii', 'ignore').decode())
    for title, path in chart_paths:
        pdf.chapter_title(title)
        pdf.image(path, w=170)
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_pdf.name)
    return temp_pdf.name

# ---------------- STREAMLIT UI ----------------
st.title("📊 AI-Powered Spreadsheet Trend & Root Cause Analyzer (with Anomaly Detection)")

uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        xls = pd.ExcelFile(uploaded_file)
        sheet_choice = st.selectbox("Select sheet to analyze", xls.sheet_names)
        df = pd.read_excel(uploaded_file, sheet_name=sheet_choice)

    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Auto-detect time column
    time_col = None
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]) or re.search(r"date|time", col.lower()):
            try:
                df[col] = pd.to_datetime(df[col])
                time_col = col
                break
            except:
                pass
    if not time_col:
        time_col = df.columns[0]

    stats_summary = df.describe(include='all')
    trends = {}
    for col in df.select_dtypes(include=np.number).columns:
        slope = np.polyfit(range(len(df)), df[col], 1)[0]
        trends[col] = slope

    # Detect anomalies
    anomalies_found = detect_anomalies(df)

    # GPT Analysis
    with st.spinner("Generating GPT Analysis & Fishbone causes..."):
        analysis_text, causes_dict = gpt_analysis_with_causes(stats_summary, trends, anomalies_found)

    st.subheader("📄 GPT Analysis & Insights")
    st.text_area("Analysis", analysis_text, height=300)

    st.subheader("📈 Trend Chart")
    fig_trend = plot_trends(df, time_col)
    st.pyplot(fig_trend)

    # Auto Pareto
    fig_pareto = None
    pareto_col = auto_select_pareto_column(df, time_col)
    if pareto_col:
        st.subheader(f"📊 Pareto Chart - Auto-selected column: {pareto_col}")
        fig_pareto = pareto_chart(df, pareto_col)
        st.pyplot(fig_pareto)
    else:
        st.info("No suitable column found for Pareto analysis.")

    st.subheader("🦴 Fishbone Diagram")
    fig_fishbone = fishbone_diagram(causes_dict)
    st.pyplot(fig_fishbone)

    # PDF export
    chart_paths = []
    for title, fig in [("Trend Over Time", fig_trend), ("Pareto Chart", fig_pareto), ("Fishbone Diagram", fig_fishbone)]:
        if fig:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                fig.savefig(tmp_img.name)
                chart_paths.append((title, tmp_img.name))

    pdf_path = generate_pdf(analysis_text, chart_paths)
    with open(pdf_path, "rb") as f:
        st.download_button("📥 Download Full PDF Report", f, file_name="data_analysis_report.pdf")
