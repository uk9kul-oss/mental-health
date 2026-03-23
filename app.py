from __future__ import annotations

import base64
import io
import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from clean_data import DISORDER_COLUMNS, load_and_clean_data


DATA_FILE = Path("1- mental-illnesses-prevalence.csv")
LOGO_PATH = Path("assets/logo.png")
ILLUSTRATION_PATH = Path("assets/illustration.jpg")


@st.cache_data
def load_logo_base64() -> str:
    if not LOGO_PATH.exists():
        return ""
    return base64.b64encode(LOGO_PATH.read_bytes()).decode("utf-8")


@st.cache_data
def load_illustration_base64() -> str:
    if not ILLUSTRATION_PATH.exists():
        return ""
    return base64.b64encode(ILLUSTRATION_PATH.read_bytes()).decode("utf-8")

NEWS_ITEMS = [
    {
        "title": "World Health Organization - Mental health",
        "url": "https://www.who.int/health-topics/mental-health",
    },
    {
        "title": "NIMH - Mental Health Information",
        "url": "https://www.nimh.nih.gov/health",
    },
    {
        "title": "CDC - Mental Health",
        "url": "https://www.cdc.gov/mentalhealth/index.htm",
    },
]

AWARENESS_POINTS = [
    "Mental health conditions are common and treatable.",
    "Early support often improves long-term outcomes.",
    "Regular sleep, movement, and social support can reduce stress burden.",
    "Seeking professional help is a strength, not a weakness.",
    "Community stigma reduction improves care access and recovery.",
]

FOOD_FOR_THOUGHT = [
    "If your organization tracked stress like a business KPI, what would changeMH",
    "How do socioeconomic factors amplify mental health inequitiesMH",
    "What would prevention-first mental healthcare look like in schoolsMH",
    "Can digital tools improve access without reducing human connectionMH",
    "How can workplaces make mental wellbeing measurable and actionableMH",
]

DEFAULT_THEME = {
    "plotly_template": "plotly_white",
    "accent": "#5bb9c2",
    "background": "linear-gradient(135deg, #ffffff 0%, #f7fffd 45%, #ffffff 100%)",
    "card_bg": "#ffffff",
    "text": "#000000",
    "muted_text": "#6b7280",
    "visual_palette": ["#5bb9c2", "#8fd3c7", "#77bdf2", "#9ad9b6", "#5fa8d3", "#22c55e"],
    "heatmap_scale": [
        [0.0, "#eaf7fb"],
        [0.5, "#bfe8e4"],
        [1.0, "#5bb9c2"],
    ],
}

MENTAL_HEALTH_ANIM_SVG = """
<svg viewBox="0 0 420 260" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Mental health animation">
  <defs>
    <linearGradient id="bg" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0%" stop-color="#e6f7fb"/>
      <stop offset="100%" stop-color="#f7fffb"/>
    </linearGradient>
    <linearGradient id="pulse" x1="0" x2="1">
      <stop offset="0%" stop-color="#5bb9c2"/>
      <stop offset="100%" stop-color="#22c55e"/>
    </linearGradient>
  </defs>
  <rect x="0" y="0" width="420" height="260" rx="22" fill="url(#bg)"/>
  <g>
    <path d="M210 50c-42 0-76 34-76 76 0 31 18 58 44 70v34c0 10 8 18 18 18h28c10 0 18-8 18-18v-34c26-12 44-39 44-70 0-42-34-76-76-76z"
      fill="#cfeef2" stroke="#5bb9c2" stroke-width="2"/>
    <path d="M170 118h20l10-14 18 36 14-22h26" fill="none" stroke="url(#pulse)" stroke-width="4" stroke-linecap="round" stroke-linejoin="round">
      <animate attributeName="stroke-dasharray" values="0,160;80,80;0,160" dur="2.4s" repeatCount="indefinite"/>
    </path>
    <g transform="translate(210 150)">
      <path d="M0 16c-8-8-20-6-24 4-5 12 9 24 24 34 15-10 29-22 24-34-4-10-16-12-24-4z"
        fill="#ff8aa1">
        <animate attributeName="transform" values="scale(1);scale(1.08);scale(1)" dur="1.6s" repeatCount="indefinite"/>
      </path>
    </g>
  </g>
  <circle cx="58" cy="70" r="18" fill="#bfe8e4" opacity="0.8">
    <animate attributeName="cy" values="70;62;70" dur="3.2s" repeatCount="indefinite"/>
  </circle>
  <circle cx="358" cy="68" r="14" fill="#cfe6ff" opacity="0.7">
    <animate attributeName="cy" values="68;76;68" dur="2.8s" repeatCount="indefinite"/>
  </circle>
</svg>
"""


@st.cache_data
def get_clean_data(file_path: str) -> pd.DataFrame:
    return load_and_clean_data(file_path)


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="report")
    return output.getvalue()


def to_pdf_bytes(title: str, summary_rows: list[tuple[str, str]], table_df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, 760, title)

    c.setFont("Helvetica", 10)
    y = 735
    for key, value in summary_rows:
        c.drawString(40, y, f"{key}: {value}")
        y -= 16

    y -= 8
    c.setFont("Helvetica-Bold", 10)
    c.drawString(40, y, "Data preview")
    y -= 16

    c.setFont("Helvetica", 8)
    preview = table_df.head(15).copy()
    headers = " | ".join(preview.columns.astype(str).tolist())
    c.drawString(40, y, headers[:120])
    y -= 12
    for _, row in preview.iterrows():
        line = " | ".join(row.astype(str).tolist())
        c.drawString(40, y, line[:120])
        y -= 12
        if y < 50:
            c.showPage()
            c.setFont("Helvetica", 8)
            y = 760

    c.save()
    buffer.seek(0)
    return buffer.read()


def apply_theme(theme: dict[str, str]) -> None:
    st.markdown(
        f"""
        <style>
            @import url("https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&family=Roboto:wght@400;500;700&display=swap");

            .stApp {{
                background: {theme["background"]};
                color: {theme["text"]};
                background-attachment: fixed;
                font-family: "Poppins", "Roboto", sans-serif;
                font-weight: 700;
            }}
            .stApp::before {{
                content: "";
                position: fixed;
                inset: -10%;
                pointer-events: none;
                background:
                    radial-gradient(circle at 10% 15%, rgba(91, 185, 194, 0.2), transparent 40%),
                    radial-gradient(circle at 85% 20%, rgba(143, 211, 199, 0.18), transparent 36%),
                    radial-gradient(circle at 70% 85%, rgba(119, 189, 242, 0.16), transparent 40%);
                filter: blur(2px);
                z-index: 0;
                animation: floatGlow 16s ease-in-out infinite alternate;
            }}
            .stApp::after {{
                content: "";
                position: fixed;
                inset: 0;
                pointer-events: none;
                background: linear-gradient(120deg, rgba(255, 255, 255, 0.6), rgba(255, 255, 255, 0.2));
                z-index: 0;
            }}
            @keyframes floatGlow {{
                0% {{ transform: translateY(0px) scale(1); }}
                100% {{ transform: translateY(-12px) scale(1.02); }}
            }}
            .main .block-container {{
                position: relative;
                z-index: 1;
                background: rgba(255, 255, 255, 0.9);
                border: 1px solid rgba(91, 185, 194, 0.18);
                border-radius: 22px;
                padding: 1.4rem 1.4rem 1.2rem 1.4rem;
                box-shadow: 0 24px 50px rgba(31, 36, 48, 0.08);
                backdrop-filter: blur(8px);
                animation: fadeInUp 0.6s ease;
            }}
            @keyframes fadeInUp {{
                0% {{ opacity: 0; transform: translateY(12px); }}
                100% {{ opacity: 1; transform: translateY(0px); }}
            }}
            .portal-card {{
                padding: 14px;
                border-radius: 16px;
                border: 1px solid rgba(91, 185, 194, 0.18);
                background: {theme["card_bg"]};
                box-shadow: 0 14px 30px rgba(31, 36, 48, 0.08);
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }}
            .portal-card:hover {{
                transform: translateY(-3px);
                box-shadow: 0 18px 40px rgba(15, 23, 42, 0.18);
            }}
            h1, h2, h3 {{
                color: {theme["text"]};
                font-family: "Poppins", "Roboto", sans-serif;
                letter-spacing: 0.2px;
                font-weight: 700;
            }}
            p, label {{
                color: {theme["text"]};
                font-weight: 700;
            }}
            [data-testid="stSidebar"] {{
                background: linear-gradient(180deg, #ecf7fb 0%, #eaf7f3 100%);
                border-right: 1px solid rgba(91, 185, 194, 0.2);
            }}
            [data-testid="stSidebar"] * {{
                color: {theme["text"]};
                font-family: "Poppins", "Roboto", sans-serif;
            }}
            [data-testid="stSidebar"] [data-baseweb="select"] * {{
                color: {theme["text"]} !important;
                font-weight: 500 !important;
            }}
            [data-testid="stSidebar"] [data-baseweb="select"] > div {{
                background: #ffffff !important;
            }}
            [data-testid="stSidebar"] [data-baseweb="select"] svg {{
                color: {theme["text"]} !important;
            }}
            [data-baseweb="tab-list"] {{
                gap: 6px;
            }}
            [data-baseweb="tab-list"] button {{
                color: {theme["text"]} !important;
                font-weight: 700 !important;
                background: rgba(255, 255, 255, 0.95) !important;
                border: 1px solid rgba(91, 185, 194, 0.2) !important;
                border-radius: 999px !important;
                padding: 0.35rem 1rem !important;
                transition: all 0.2s ease;
            }}
            [data-baseweb="tab-list"] button[aria-selected="true"] {{
                background: rgba(91, 185, 194, 0.18) !important;
                border-color: rgba(91, 185, 194, 0.7) !important;
                box-shadow: 0 6px 18px rgba(91, 185, 194, 0.2);
            }}
            [data-testid="stAlert"] {{
                color: {theme["text"]} !important;
            }}
            [data-testid="stAlert"] * {{
                color: {theme["text"]} !important;
            }}
            [data-testid="stMetric"] {{
                background: {theme["card_bg"]};
                border: 1px solid rgba(91, 185, 194, 0.2);
                border-radius: 14px;
                padding: 12px;
                box-shadow: 0 12px 24px rgba(31, 36, 48, 0.08);
            }}
            [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {{
                color: {theme["text"]};
            }}
            [data-testid="stCaption"] {{
                color: {theme["muted_text"]};
            }}
            .stButton>button, .stDownloadButton>button {{
                border: 1px solid rgba(91, 185, 194, 0.5);
                border-radius: 999px;
                background: linear-gradient(120deg, rgba(91, 185, 194, 0.25), rgba(143, 211, 199, 0.25));
                color: {theme["text"]};
                font-weight: 700;
                padding: 0.4rem 1.1rem;
                transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
            }}
            .stButton>button:hover, .stDownloadButton>button:hover {{
                transform: translateY(-1px);
                border-color: rgba(91, 185, 194, 0.85);
                box-shadow: 0 10px 20px rgba(91, 185, 194, 0.2);
            }}
            .brand-bar {{
                display: flex;
                align-items: center;
                gap: 14px;
                padding: 12px 16px;
                border-radius: 18px;
                background: rgba(255, 255, 255, 0.95);
                border: 1px solid rgba(91, 185, 194, 0.2);
                box-shadow: 0 14px 28px rgba(31, 36, 48, 0.08);
                margin-bottom: 14px;
            }}
            .brand-logo {{
                width: 56px;
                height: 56px;
                border-radius: 14px;
                display: grid;
                place-items: center;
                background: #ffffff;
                border: 1px solid rgba(91, 185, 194, 0.35);
                overflow: hidden;
            }}
            .brand-logo img {{
                width: 100%;
                height: 100%;
                object-fit: contain;
                display: block;
            }}
            .login-wrap {{
                width: 100%;
                min-height: auto;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 0.5rem 0 1.25rem;
                margin-top: 0.2rem;
            }}
            .login-container {{
                width: min(980px, 100%);
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                padding: 0 12px;
                margin-top: 0;
            }}
            .mh-anim {{
                width: min(680px, 100%);
                margin: 0 auto 0.8rem;
                display: grid;
                place-items: center;
            }}
            .mh-anim svg {{
                width: 100%;
                height: auto;
                filter: drop-shadow(0 12px 24px rgba(31, 36, 48, 0.08));
            }}
            .hero {{
                width: min(980px, 100%);
                margin: 0.25rem auto 1rem;
                text-align: center;
                display: flex;
                flex-direction: column;
                gap: 10px;
            }}
            .hero-title {{
                margin: 0;
                font-size: 2.4rem;
                font-weight: 700;
                color: {theme["text"]};
            }}
            .hero-subtitle {{
                margin: 0;
                font-size: 1.05rem;
                color: {theme["text"]};
            }}
            .hero-actions {{
                display: flex;
                gap: 10px;
                justify-content: center;
                flex-wrap: wrap;
                margin-top: 4px;
            }}
            .login-header {{
                display: flex;
                align-items: center;
                gap: 12px;
                margin-bottom: 12px;
                text-align: center;
            }}
            .login-header-logo {{
                width: 52px;
                height: 52px;
                border-radius: 12px;
                border: 1px solid rgba(91, 185, 194, 0.25);
                background: #f2fbff;
                display: grid;
                place-items: center;
                overflow: hidden;
            }}
            .login-header-logo img {{
                width: 100%;
                height: 100%;
                object-fit: contain;
                display: block;
            }}
            .login-header-title {{
                margin: 0;
                font-size: 1.9rem;
                font-weight: 700;
            }}
            .login-card {{
                display: grid;
                grid-template-columns: 1.15fr 0.85fr;
                gap: 24px;
                width: 100%;
                padding: 28px;
                border-radius: 22px;
                background: #ffffff;
                border: 1px solid rgba(91, 185, 194, 0.2);
                box-shadow: 0 24px 50px rgba(31, 36, 48, 0.1);
            }}
            .login-left {{
                display: flex;
                flex-direction: column;
                gap: 14px;
                justify-content: center;
                align-items: center;
            }}
            .login-illustration {{
                width: 100%;
                border-radius: 18px;
                overflow: hidden;
                background: #f2fbff;
                border: 1px solid rgba(91, 185, 194, 0.15);
                display: grid;
                place-items: center;
                padding: 12px;
            }}
            .login-illustration img {{
                width: 100%;
                height: auto;
                display: block;
            }}
            .login-right {{
                display: flex;
                flex-direction: column;
                justify-content: center;
                gap: 6px;
                align-items: flex-start;
            }}
            .login-logo {{
                width: 88px;
                height: 88px;
                border-radius: 18px;
                border: 1px solid rgba(91, 185, 194, 0.25);
                background: #f2fbff;
                display: grid;
                place-items: center;
                overflow: hidden;
            }}
            .login-title {{
                margin: 0;
                font-size: 1.6rem;
                font-weight: 700;
            }}
            .login-subtitle {{
                margin: 0;
                color: {theme["muted_text"]};
            }}
            .login-footer {{
                margin-top: 10px;
                font-size: 0.9rem;
                color: {theme["muted_text"]};
            }}
            .login-field-label {{
                margin-top: 8px;
                font-size: 0.9rem;
                font-weight: 600;
                color: {theme["text"]};
            }}
            @media (max-width: 900px) {{
                .login-card {{
                    grid-template-columns: 1fr;
                }}
            }}
            .brand-title {{
                font-family: "Poppins", "Roboto", sans-serif;
                font-size: 1.7rem;
                margin: 0;
            }}
            .brand-subtitle {{
                margin: 2px 0 0;
                color: {theme["muted_text"]};
                font-size: 0.95rem;
            }}
            .welcome-card {{
                padding: 16px;
                border-radius: 18px;
                background: rgba(255, 255, 255, 0.95);
                border: 1px solid rgba(91, 185, 194, 0.2);
                box-shadow: 0 16px 32px rgba(31, 36, 48, 0.08);
            }}
            .stress-card {{
                padding: 16px;
                border-radius: 18px;
                background: linear-gradient(140deg, rgba(91, 185, 194, 0.18), rgba(143, 211, 199, 0.18));
                border: 1px solid rgba(91, 185, 194, 0.28);
                box-shadow: 0 14px 28px rgba(31, 36, 48, 0.08);
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def style_figure(fig, theme: dict[str, str]):
    fig.update_layout(
        template=theme["plotly_template"],
        font=dict(color=theme["text"]),
        title_font=dict(color=theme["text"], size=20),
        legend_font=dict(color=theme["text"]),
        plot_bgcolor="#ffffff",
        paper_bgcolor=theme["card_bg"],
    )
    return fig


def chart_jpeg_bytes(df: pd.DataFrame, country: str, metric: str, accent: str) -> bytes:
    chart_df = df[df["Entity"] == country][["Year", metric]].sort_values("Year")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(chart_df["Year"], chart_df[metric], color=accent, linewidth=2)
    ax.set_title(f"{country}: {metric.title()} trend")
    ax.set_xlabel("Year")
    ax.set_ylabel("Share of population")
    ax.grid(True, alpha=0.25)
    img_buffer = io.BytesIO()
    plt.tight_layout()
    fig.savefig(img_buffer, format="jpeg", dpi=150)
    plt.close(fig)
    img_buffer.seek(0)
    return img_buffer.read()


def stress_card_png(score: int | None, label: str, theme: dict[str, str]) -> bytes:
    fig, ax = plt.subplots(figsize=(4.2, 2.4), dpi=160)
    fig.patch.set_facecolor("#0b1225")
    ax.set_facecolor("#0b1225")
    ax.axis("off")

    title = "Stress Level Card"
    score_text = f"{score}/100" if score is not None else "Not calculated"
    fig.text(0.07, 0.75, title, color=theme["text"], fontsize=12, weight="bold")
    fig.text(0.07, 0.5, f"Latest score: {score_text}", color=theme["text"], fontsize=11)
    fig.text(0.07, 0.3, f"Status: {label}", color=theme["text"], fontsize=11)
    fig.text(0.07, 0.1, "Generated by Mental Health Prevalence", color=theme["muted_text"], fontsize=8)

    buffer = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buffer, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buffer.seek(0)
    return buffer.read()


def render_stress_calculator(theme: dict[str, str]) -> None:
    st.subheader("Stress Calculator")
    st.caption("Quick self-check. This is not a medical diagnosis.")

    c1, c2 = st.columns(2)
    points = ["good", "better", "average", "worst"]
    higher_worse = {"good": 2, "better": 4, "average": 6, "worst": 8}
    higher_better = {"good": 8, "better": 6, "average": 4, "worst": 2}

    with c1:
        stress_level_label = st.select_slider(
            "How stressed have you felt over the last 7 daysMH",
            options=points,
            value="average",
        )
        sleep_quality_label = st.select_slider(
            "How would you rate your sleep quality this weekMH",
            options=points,
            value="better",
        )
        energy_level_label = st.select_slider(
            "How steady has your energy been this weekMH",
            options=points,
            value="better",
        )
    with c2:
        mood_level_label = st.select_slider(
            "How stable has your mood been this weekMH",
            options=points,
            value="better",
        )
        workload_pressure_label = st.select_slider(
            "How heavy has your work or study pressure feltMH",
            options=points,
            value="average",
        )

    stress_level = higher_worse[stress_level_label]
    workload_pressure = higher_worse[workload_pressure_label]
    sleep_quality = higher_better[sleep_quality_label]
    energy_level = higher_better[energy_level_label]
    mood_level = higher_better[mood_level_label]

    # Higher stress/pressure increases score; higher wellbeing buffers reduce it.
    score = (
        stress_level * 20
        + workload_pressure * 18
        + (10 - sleep_quality) * 16
        + (10 - energy_level) * 14
        + (10 - mood_level) * 12
    )

    score = max(0, min(100, round(score / 8)))
    st.metric("Stress score (0-100)", score)

    if score >= 75:
        st.error("Worst: High stress indicated. Consider reaching out to a professional or trusted person.")
        st.session_state.stress_label = "Worst"
        st.markdown(
            "- Suggestions: pause non-urgent tasks, drink water and eat something light, take a 10-minute walk, "
            "and contact someone you trust today."
        )
    elif score >= 55:
        st.warning("Average: Elevated stress indicated. Add recovery time and reduce overload where possible.")
        st.session_state.stress_label = "Average"
        st.markdown(
            "- Suggestions: schedule two short breaks, lower multitasking, do 5-10 minutes of breathing, "
            "and aim for a consistent bedtime."
        )
    elif score >= 30:
        st.info("Better: Mild stress indicated. Keep balance and protect your routine.")
        st.session_state.stress_label = "Better"
        st.markdown(
            "- Suggestions: keep steady sleep, add one enjoyable activity, take a brief screen break, "
            "and do a quick stretch."
        )
    else:
        st.success("Best: Low stress indicated. Keep up the healthy habits.")
        st.session_state.stress_label = "Best"
        st.markdown(
            "- Suggestions: maintain your routine, plan a small reward, and check in with yourself weekly."
        )

    st.session_state.stress_score = score
    score_text = f"{score}/100"
    st.markdown(
        f"""
        <div class="stress-card">
            <h3>Stress Level Card</h3>
            <p><strong>Latest score:</strong> {score_text}</p>
            <p><strong>Status:</strong> {st.session_state.stress_label}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "If you feel overwhelmed or unsafe, seek help from local emergency services or a licensed professional."
    )

    card_png = stress_card_png(score, st.session_state.stress_label, theme)
    st.download_button(
        "Download stress card (PNG)",
        data=card_png,
        file_name="stress_level_card.png",
        mime="image/png",
    )


def build_prediction(country_df: pd.DataFrame, metric: str, horizon: int) -> pd.DataFrame:
    model_df = country_df[["Year", metric]].dropna().copy()
    x = model_df[["Year"]].to_numpy()
    y = model_df[metric].to_numpy()

    lr = LinearRegression()
    lr.fit(x, y)

    max_year = int(model_df["Year"].max())
    future_years = list(range(max_year + 1, max_year + horizon + 1))
    prediction = lr.predict(pd.Series(future_years).to_numpy().reshape(-1, 1))
    return pd.DataFrame({"Year": future_years, f"{metric}_predicted": prediction})


def build_ts_forecast(country_df: pd.DataFrame, metric: str, horizon: int) -> pd.DataFrame:
    ts = country_df[["Year", metric]].dropna().sort_values("Year")
    values = ts[metric].to_numpy()
    max_year = int(ts["Year"].max())

    if len(values) < 3:
        # Fallback forecast when not enough points for a stable TS fit.
        forecast = [float(values[-1])] * horizon
    else:
        model = ExponentialSmoothing(values, trend="add", seasonal=None, initialization_method="estimated")
        fit = model.fit(optimized=True)
        forecast = fit.forecast(horizon)
    years = list(range(max_year + 1, max_year + horizon + 1))

    return pd.DataFrame({"Year": years, f"{metric}_ts_forecast": forecast})


def main() -> None:
    st.set_page_config(page_title="Mental Health Prevalence", layout="wide")

    if not DATA_FILE.exists():
        st.error(f"Data file not found: {DATA_FILE}")
        return

    df = get_clean_data(str(DATA_FILE))
    apply_theme(DEFAULT_THEME)
    logo_b64 = load_logo_base64()
    logo_html = (
        f'<img src="data:image/png;base64,{logo_b64}" alt="Logo" />' if logo_b64 else "MH"
    )
    illustration_b64 = load_illustration_base64()
    illustration_html = (
        f'<img src="data:image/jpeg;base64,{illustration_b64}" alt="Illustration" />'
        if illustration_b64
        else ""
    )

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        st.markdown(
            f"""
            <div class="login-wrap">
                <div class="login-container">
                    <div class="login-header">
                        <div class="login-header-logo">{logo_html}</div>
                        <h2 class="login-header-title">Mental Health Prevalence</h2>
                    </div>
            """,
            unsafe_allow_html=True,
        )
        col_left, col_right = st.columns([1.15, 0.85], gap="large")
        with col_left:
            st.markdown(
                f"""
                <div class="login-left">
                    <div class="login-illustration">
                        {illustration_html}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_right:
            st.markdown(
                f"""
                <div class="login-right">
                    <h3 class="login-title">Log in</h3>
                    <p class="login-subtitle">Access your portal dashboard with your credentials.</p>
                """,
                unsafe_allow_html=True,
            )
            with st.form("login_form", clear_on_submit=False):
                st.markdown('<div class="login-field-label">Login name</div>', unsafe_allow_html=True)
                username = st.text_input(
                    "Username",
                    placeholder="Username",
                    label_visibility="collapsed",
                )
                password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Password",
                    label_visibility="collapsed",
                )
                remember = st.checkbox("Remember me")
                submitted = st.form_submit_button("Login")
        if submitted:
            if username.strip() and password.strip():
                st.session_state.logged_in = True
                st.success("Login successful. Welcome!")
                st.rerun()
            else:
                st.error("Please enter a username and password to continue.")
        st.markdown(
            """
            <div class="login-footer">By logging in you agree to the portal terms.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    countries = sorted(df["Entity"].unique().tolist())
    metrics = DISORDER_COLUMNS + ["total_burden"]

    st.sidebar.header("Filters")
    selected_country = st.sidebar.selectbox("Country", countries)
    selected_metric = st.sidebar.selectbox("Metric", metrics, index=1)
    year_min = int(df["Year"].min())
    year_max = int(df["Year"].max())
    forecast_end_year = 2030
    horizon_years = max(1, forecast_end_year - year_max)
    year_range = st.sidebar.slider("Year range", year_min, year_max, (year_min, year_max))
    st.sidebar.divider()
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    filtered = df[
        (df["Entity"] == selected_country)
        & (df["Year"] >= year_range[0])
        & (df["Year"] <= year_range[1])
    ].copy()

    if filtered.empty:
        st.warning("No data found for the selected filters. Adjust country or year range.")
        return

    st.markdown(
        f"""
        <div class="brand-bar">
            <div class="brand-logo">{logo_html}</div>
            <div>
                <h2 class="brand-title">Mental Health Prevalence</h2>
                <p class="brand-subtitle">Trends, forecasts, and wellbeing signals in one place</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        ["Home", "Dashboard", "Prediction", "Time Series", "Insights Hub", "Stress Calculator", "Help"]
    )

    with tab1:
        st.markdown(
            """
            <div class="hero">
                <h1 class="hero-title">Your Mental Health Matters 💙</h1>
                <p class="hero-subtitle">Track, analyze and improve your mental well-being</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.subheader("Welcome")
        st.markdown(
            """
            <div class="welcome-card">
                <strong>Welcome to your Mental Health Prevalence Portal.</strong><br/>
                Explore global trends, run forecasts, and track wellbeing signals in minutes.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        w1 = st.container()
        with w1:
            st.markdown(
                """
                <div class="portal-card">
                    <h3>Today's focus</h3>
                    <p>Use the Dashboard tab to explore country-level prevalence, then head to
                    Prediction or Time Series for forward-looking views.</p>
                    <p><strong>Tip:</strong> Try changing the metric to compare disorder categories.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab2:
        col1, col2, col3 = st.columns(3)
        latest = filtered.sort_values("Year").tail(1)
        if not latest.empty:
            col1.metric("Latest year", int(latest["Year"].iloc[0]))
            col2.metric(f"{selected_metric} (%)", f"{latest[selected_metric].iloc[0]:.3f}")
            col3.metric("Total burden (%)", f"{latest['total_burden'].iloc[0]:.3f}")

        fig = px.line(
            filtered,
            x="Year",
            y=selected_metric,
            title=f"{selected_country}: {selected_metric.title()} over time",
            markers=True,
        )
        fig = style_figure(fig, DEFAULT_THEME)
        fig.update_traces(line=dict(color=DEFAULT_THEME["accent"], width=3))
        st.plotly_chart(fig, use_container_width=True)

        v1, v2 = st.columns(2)
        with v1:
            st.markdown("<div class='portal-card'>", unsafe_allow_html=True)
            all_metric_trend = px.line(
                filtered,
                x="Year",
                y=DISORDER_COLUMNS,
                title="All disorders trend",
                color_discrete_sequence=DEFAULT_THEME["visual_palette"],
            )
            all_metric_trend = style_figure(all_metric_trend, DEFAULT_THEME)
            st.plotly_chart(all_metric_trend, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with v2:
            latest_row = filtered.sort_values("Year").tail(1)
            if not latest_row.empty:
                pie_df = latest_row[DISORDER_COLUMNS].T.reset_index()
                pie_df.columns = ["Disorder", "Value"]
                st.markdown("<div class='portal-card'>", unsafe_allow_html=True)
                donut = px.pie(
                    pie_df,
                    names="Disorder",
                    values="Value",
                    hole=0.45,
                    title=f"{int(latest_row['Year'].iloc[0])} composition",
                    color_discrete_sequence=DEFAULT_THEME["visual_palette"],
                )
                donut = style_figure(donut, DEFAULT_THEME)
                st.plotly_chart(donut, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

        heatmap_data = filtered.set_index("Year")[DISORDER_COLUMNS]
        heatmap_fig = px.imshow(
            heatmap_data.T,
            aspect="auto",
            color_continuous_scale=DEFAULT_THEME["heatmap_scale"],
            title="Disorder intensity heatmap by year",
            labels={"x": "Year", "y": "Disorder", "color": "Share (%)"},
        )
        heatmap_fig = style_figure(heatmap_fig, DEFAULT_THEME)
        st.plotly_chart(heatmap_fig, use_container_width=True)

        st.subheader("Cleaned data preview")
        st.dataframe(filtered, use_container_width=True)

    with tab3:
        st.subheader("Prediction")
        horizon = horizon_years
        st.caption(f"Forecasting through {forecast_end_year}.")

        pred_df = build_prediction(filtered, selected_metric, horizon)
        merged = filtered[["Year", selected_metric]].merge(pred_df, on="Year", how="outer")

        fig_pred = px.line(
            merged,
            x="Year",
            y=[selected_metric, f"{selected_metric}_predicted"],
            markers=True,
            color_discrete_sequence=[DEFAULT_THEME["accent"], DEFAULT_THEME["visual_palette"][1]],
        )
        fig_pred = style_figure(fig_pred, DEFAULT_THEME)
        st.plotly_chart(fig_pred, use_container_width=True)
        st.dataframe(pred_df, use_container_width=True)

    with tab4:
        st.subheader("Time Series Analysis")
        filtered["rolling_3y"] = filtered[selected_metric].rolling(3, min_periods=1).mean()
        filtered["yoy_change_pct"] = filtered[selected_metric].pct_change() * 100

        ts_horizon = horizon_years
        st.caption(f"Forecasting through {forecast_end_year}.")
        ts_forecast = build_ts_forecast(filtered, selected_metric, ts_horizon)
        combined = filtered[["Year", selected_metric, "rolling_3y"]].merge(ts_forecast, on="Year", how="outer")

        fig_ts = px.line(
            combined,
            x="Year",
            y=[selected_metric, "rolling_3y", f"{selected_metric}_ts_forecast"],
            markers=True,
            title="Historical trend + rolling mean + forecast",
            color_discrete_sequence=[
                DEFAULT_THEME["accent"],
                DEFAULT_THEME["visual_palette"][2],
                DEFAULT_THEME["visual_palette"][1],
            ],
        )
        fig_ts = style_figure(fig_ts, DEFAULT_THEME)
        st.plotly_chart(fig_ts, use_container_width=True)

        st.subheader("Year-over-year change (%)")
        st.dataframe(filtered[["Year", selected_metric, "yoy_change_pct"]], use_container_width=True)

    with tab5:
        st.subheader("News")
        for item in NEWS_ITEMS:
            st.markdown(f"- [{item['title']}]({item['url']})")

        st.subheader("Awareness")
        for point in AWARENESS_POINTS:
            st.write(f"- {point}")

        st.subheader("Food for Thought")
        st.info(random.choice(FOOD_FOR_THOUGHT))
        if st.button("New thought"):
            st.success(random.choice(FOOD_FOR_THOUGHT))

    with tab6:
        render_stress_calculator(DEFAULT_THEME)

    with tab7:
        st.subheader("Help & Guidance")
        st.markdown(
            """
            <div class="portal-card">
                <h3>How to use this portal</h3>
                <p><strong>Home:</strong> Overview and quick actions.</p>
                <p><strong>Dashboard:</strong> Explore prevalence trends and compositions.</p>
                <p><strong>Prediction:</strong> Forecast by selected metric.</p>
                <p><strong>Time Series:</strong> Rolling averages and change rates.</p>
                <p><strong>Stress Calculator:</strong> Personal check-in with suggestions.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown(
            """
            <div class="portal-card">
                <h3>Need support</h3>
                <p>If you feel overwhelmed or unsafe, reach out to local emergency services or a licensed professional.</p>
                <p>For non-urgent help, consider talking with a trusted friend, counselor, or healthcare provider.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("Export Reports")

    report_df = filtered.copy()
    csv_bytes = report_df.to_csv(index=False).encode("utf-8")
    excel_bytes = to_excel_bytes(report_df)
    pdf_bytes = to_pdf_bytes(
        title="Mental Health Report",
        summary_rows=[
            ("Country", selected_country),
            ("Metric", selected_metric),
            ("Year range", f"{year_range[0]} - {year_range[1]}"),
            ("Rows", str(len(report_df))),
        ],
        table_df=report_df,
    )
    jpeg_bytes = chart_jpeg_bytes(df, selected_country, selected_metric, DEFAULT_THEME["accent"])

    e1, e2, e3, e4 = st.columns(4)
    e1.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name=f"{selected_country}_{selected_metric}_report.csv",
        mime="text/csv",
    )
    e2.download_button(
        "Download Excel",
        data=excel_bytes,
        file_name=f"{selected_country}_{selected_metric}_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    e3.download_button(
        "Download PDF",
        data=pdf_bytes,
        file_name=f"{selected_country}_{selected_metric}_report.pdf",
        mime="application/pdf",
    )
    e4.download_button(
        "Download image.jpeg",
        data=jpeg_bytes,
        file_name="image.jpeg",
        mime="image/jpeg",
    )


if __name__ == "__main__":
    main()
