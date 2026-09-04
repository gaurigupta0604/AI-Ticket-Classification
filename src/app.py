# src/app.py

import sys
import html
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.predict import classify_ticket, vectorize_ticket
from src.recommend import load_ticket_corpus, recommend_resolution

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI-Powered Customer Support Assistant",
    page_icon="🛠️",
    layout="wide",
)

CATEGORY_COLORS = {
    "Account": "#5B8DEF",
    "Billing": "#E8A33D",
    "Fraud": "#E5484D",
    "General Inquiry": "#8B8FA3",
    "Technical": "#3ED6C5",
}
PRIORITY_COLORS = {
    "Low": "#3FB950",
    "Medium": "#E8A33D",
    "High": "#E5484D",
    "Critical": "#FF5C7A",
}
PIPELINE_STAGES = ["Ticket", "Classify", "Match", "Recommend"]

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg: #0F1424;
        --surface: #171F33;
        --border: #26304A;
        --text: #EDEFF7;
        --text-dim: #8B93AC;
        --accent: #E8A33D;
        --accent-2: #5B8DEF;
    }

    .stApp { background: var(--bg); color: var(--text); }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; }

    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 2rem; max-width: 980px; }

    .eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.14em;
        color: var(--accent);
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    .hero-title { font-size: 2.1rem; font-weight: 700; margin: 0 0 0.5rem 0; color: var(--text); }
    .hero-sub { color: var(--text-dim); font-size: 0.98rem; max-width: 640px; margin-bottom: 1.6rem; }

    .pipeline {
        display: flex;
        border: 1px solid var(--border);
        border-radius: 10px;
        overflow: hidden;
        margin-bottom: 1.6rem;
    }
    .pipeline-stage {
        flex: 1;
        padding: 0.7rem 0.8rem;
        background: var(--surface);
        border-right: 1px solid var(--border);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: var(--text-dim);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .pipeline-stage:last-child { border-right: none; }
    .pipeline-stage .num { color: var(--text-dim); font-weight: 500; }
    .pipeline-stage.active { background: rgba(232, 163, 61, 0.12); color: var(--accent); }
    .pipeline-stage.active .num { color: var(--accent); }

    .panel-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--text-dim);
        margin-bottom: 0.9rem;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--surface);
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        padding: 0.3rem 0.5rem;
        margin-bottom: 1.2rem;
    }

    .stTextArea textarea {
        background: var(--bg) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
    }
    .stTextArea textarea:focus {
        border-color: var(--accent-2) !important;
        box-shadow: 0 0 0 1px var(--accent-2) !important;
    }

    .stButton button {
        background: var(--accent) !important;
        color: #1A1200 !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.55rem 1.4rem !important;
    }
    .stButton button:hover { filter: brightness(1.08); }

    .stat-card {
        background: var(--bg);
        border: 1px solid var(--border);
        border-left: 3px solid var(--stat-color, var(--accent));
        border-radius: 8px;
        padding: 0.9rem 1rem;
    }
    .stat-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--text-dim);
        margin-bottom: 0.35rem;
    }
    .stat-value { font-size: 1.4rem; font-weight: 600; color: var(--text); }
    .meter-track {
        width: 100%; height: 5px; border-radius: 4px;
        background: var(--border); margin-top: 0.55rem; overflow: hidden;
    }
    .meter-fill { height: 100%; border-radius: 4px; }

    .badge {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 500;
        padding: 0.18rem 0.55rem;
        border-radius: 999px;
    }

    table.ticket-table { width: 100%; border-collapse: collapse; font-size: 0.86rem; }
    table.ticket-table th {
        text-align: left; font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem; letter-spacing: 0.06em; text-transform: uppercase;
        color: var(--text-dim); padding: 0.5rem 0.6rem; border-bottom: 1px solid var(--border);
    }
    table.ticket-table td {
        padding: 0.55rem 0.6rem; border-bottom: 1px solid var(--border); color: var(--text);
    }
    table.ticket-table tr:last-child td { border-bottom: none; }
    .mono { font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: var(--text-dim); }

    .empty-state {
        border: 1px dashed var(--border);
        border-radius: 12px;
        padding: 2.4rem 1.5rem;
        text-align: center;
        color: var(--text-dim);
    }

    .note-box {
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.8rem 1rem;
        color: var(--text-dim);
        font-size: 0.86rem;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Small render helpers
# ---------------------------------------------------------------------------
def badge(text: str, color: str) -> str:
    return (
        f'<span class="badge" style="background:{color}22; color:{color};">'
        f"{html.escape(str(text))}</span>"
    )


def pipeline_ribbon() -> str:
    stages_html = ""
    for i, stage in enumerate(PIPELINE_STAGES):
        stages_html += (
            f'<div class="pipeline-stage">'
            f'<span class="num">{i + 1:02d}</span><span>{stage}</span></div>'
        )
    return f'<div class="pipeline">{stages_html}</div>'


def stat_card(label: str, value: str, color: str = "#E8A33D", meter_pct=None) -> str:
    meter_html = ""
    if meter_pct is not None:
        meter_html = (
            f'<div class="meter-track"><div class="meter-fill" '
            f'style="width:{meter_pct}%; background:{color};"></div></div>'
        )
    return (
        f'<div class="stat-card" style="--stat-color:{color};">'
        f'<div class="stat-label">{html.escape(label)}</div>'
        f'<div class="stat-value">{html.escape(value)}</div>'
        f"{meter_html}</div>"
    )


def similar_tickets_table(df) -> str:
    rows = ""
    for _, row in df.iterrows():
        cat_color = CATEGORY_COLORS.get(row["Issue_Category"], "#8B8FA3")
        pri_color = PRIORITY_COLORS.get(row["Priority_Level"], "#8B8FA3")
        rows += (
            "<tr>"
            f'<td class="mono">{html.escape(str(row["Ticket_ID"]))}</td>'
            f'<td>{html.escape(str(row["Ticket_Subject"]))}</td>'
            f'<td>{badge(row["Issue_Category"], cat_color)}</td>'
            f'<td>{badge(row["Priority_Level"], pri_color)}</td>'
            f'<td class="mono">{row["Resolution_Time_Hours"]:.0f} hrs</td>'
            f'<td class="mono">{row["Satisfaction_Score"]:.1f}/5</td>'
            f'<td class="mono">{row["Similarity"]:.3f}</td>'
            "</tr>"
        )
    return (
        '<table class="ticket-table"><thead><tr>'
        "<th>Ticket ID</th><th>Subject</th><th>Category</th><th>Priority</th>"
        "<th>Resolution</th><th>Satisfaction</th><th>Similarity</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
    )


# ---------------------------------------------------------------------------
# Data (cached once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_ticket_corpus():
    return load_ticket_corpus()


df, X = get_ticket_corpus()

if "result" not in st.session_state:
    st.session_state.result = None

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown('<div class="eyebrow">Ticket Triage Console</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">AI-Powered Customer Support Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Paste in a ticket description and the system classifies it, '
    "finds the most similar resolved tickets, and estimates priority, "
    "resolution time, and satisfaction from that history.</div>",
    unsafe_allow_html=True,
)
st.markdown(pipeline_ribbon(), unsafe_allow_html=True)

with st.expander("About this model"):
    st.markdown(
        '<div class="note-box">Trained on TF-IDF features from ticket descriptions '
        "(5,000 features, 1-2 word n-grams) with Logistic Regression. Test-set accuracy on this "
        "dataset is near-perfect because the underlying tickets are template-generated; on a small "
        "hand-written set of realistically-phrased tickets, accuracy was 60% (6/10) — a more honest "
        "measure of real-world generalization, and the number reported here intentionally.</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Input panel
# ---------------------------------------------------------------------------
with st.container(border=True):
    st.markdown('<div class="panel-label">Ticket Input</div>', unsafe_allow_html=True)
    description = st.text_area(
        "Description",
        placeholder="Enter your customer support ticket...",
        height=140,
        label_visibility="collapsed",
    )
    analyze_clicked = st.button("Analyze Ticket", type="primary")

# ---------------------------------------------------------------------------
# Run analysis
# ---------------------------------------------------------------------------
if analyze_clicked:
    if not description.strip():
        st.warning("Please enter a ticket description before analyzing.")
    else:
        with st.spinner("Classifying ticket and matching historical cases..."):
            category, confidence = classify_ticket(description)
            ticket_vector = vectorize_ticket(description)
            rec = recommend_resolution(ticket_vector, category, df, X)
        st.session_state.result = {"category": category, "confidence": confidence, "rec": rec}

# ---------------------------------------------------------------------------
# Results / empty state
# ---------------------------------------------------------------------------
result = st.session_state.result

if result is None:
    st.markdown(
        '<div class="empty-state">Run a ticket through the pipeline above to see '
        "its predicted category, priority, and closest historical matches here.</div>",
        unsafe_allow_html=True,
    )
else:
    category = result["category"]
    confidence = result["confidence"]
    rec = result["rec"]
    cat_color = CATEGORY_COLORS.get(category, "#5B8DEF")
    pri_color = PRIORITY_COLORS.get(rec["suggested_priority"], "#E8A33D")

    with st.container(border=True):
        st.markdown('<div class="panel-label">Prediction</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(stat_card("Predicted Category", category, color=cat_color), unsafe_allow_html=True)
        with c2:
            st.markdown(
                stat_card("Confidence", f"{confidence:.2f}%", color=cat_color, meter_pct=confidence),
                unsafe_allow_html=True,
            )

    with st.container(border=True):
        st.markdown('<div class="panel-label">Resolution Recommendation</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(stat_card("Suggested Priority", rec["suggested_priority"], color=pri_color), unsafe_allow_html=True)
        with c2:
            st.markdown(stat_card("Expected Resolution", f"{rec['avg_resolution_time']:.2f} hrs"), unsafe_allow_html=True)
        with c3:
            st.markdown(stat_card("Avg. Satisfaction", f"{rec['avg_satisfaction']:.2f} / 5"), unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="panel-label">Similar Historical Tickets</div>', unsafe_allow_html=True)
        st.markdown(similar_tickets_table(rec["similar_tickets"]), unsafe_allow_html=True)