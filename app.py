"""
app.py — Disease Symptom Classifier
Streamlit web application.

Run: streamlit run app.py
"""

import pickle
import os
import sys
import numpy as np
import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(__file__))

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Disease Symptom Classifier",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ───────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Sidebar ─────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f1a2e 0%, #1a2d4a 100%);
    border-right: 1px solid rgba(99,179,237,0.15);
}
[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

/* ── Hero banner ─────────────────── */
.hero {
    background: linear-gradient(135deg, #0f1a2e 0%, #1a3a5c 50%, #0f2744 100%);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(99,179,237,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero h1 {
    color: #e2e8f0;
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 0.4rem;
    letter-spacing: -0.5px;
}
.hero p {
    color: #94a3b8;
    font-size: 1rem;
    margin: 0;
    max-width: 600px;
    line-height: 1.6;
}
.hero-badge {
    display: inline-block;
    background: rgba(99,179,237,0.15);
    border: 1px solid rgba(99,179,237,0.3);
    color: #63b3ed;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    margin-bottom: 1rem;
}

/* ── Metric cards ───────────────── */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.metric-card {
    flex: 1;
    background: #0f1a2e;
    border: 1px solid rgba(99,179,237,0.18);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
}
.metric-card .value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.8rem;
    font-weight: 600;
    color: #63b3ed;
    line-height: 1;
}
.metric-card .label {
    font-size: 0.78rem;
    color: #64748b;
    margin-top: 0.35rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}

/* ── Result card ─────────────────── */
.result-card {
    background: linear-gradient(135deg, #0d2137 0%, #0f2744 100%);
    border: 1.5px solid rgba(99,179,237,0.3);
    border-radius: 16px;
    padding: 2rem;
    margin-top: 1.5rem;
}
.result-card h2 {
    color: #e2e8f0;
    margin: 0 0 0.25rem;
    font-size: 1.5rem;
}
.result-card .confidence {
    font-family: 'JetBrains Mono', monospace;
    color: #48bb78;
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1;
    margin: 0.5rem 0 1rem;
}
.disclaimer {
    background: rgba(251,191,36,0.08);
    border: 1px solid rgba(251,191,36,0.25);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    font-size: 0.82rem;
    color: #f6c343;
    margin-top: 1.5rem;
}

/* ── Symptom tags ─────────────────── */
.symptom-tag {
    display: inline-block;
    background: rgba(99,179,237,0.12);
    border: 1px solid rgba(99,179,237,0.25);
    color: #90cdf4;
    font-size: 0.78rem;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    margin: 0.15rem;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Section heading ──────────────── */
.section-head {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #475569;
    margin: 1.5rem 0 0.75rem;
    border-bottom: 1px solid rgba(99,179,237,0.1);
    padding-bottom: 0.4rem;
}

/* ── Sidebar symptom search ───────── */
.stMultiSelect [data-baseweb="tag"] {
    background: rgba(99,179,237,0.2) !important;
    border-color: rgba(99,179,237,0.4) !important;
}
</style>
""", unsafe_allow_html=True)


# ── Load model ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

payload = load_model()
model        = payload["model"]
symptom_cols = payload["symptom_cols"]
classes      = payload["classes"]
acc          = payload["accuracy"]
f1           = payload["macro_f1"]
model_name   = payload["model_name"]

# Pretty-print symptom names for UI
def pretty(s):
    return s.replace("_", " ").title()

symptom_display = {pretty(s): s for s in symptom_cols}
symptom_options = sorted(symptom_display.keys())

# Disease descriptions (shown after prediction)
disease_info = {
    "Common Cold":         "A viral infection of the upper respiratory tract. Usually mild and self-limiting.",
    "Diabetes":            "A metabolic disease causing high blood sugar due to insulin issues.",
    "Hypertension":        "Chronically elevated blood pressure, a major risk factor for heart disease.",
    "Malaria":             "A mosquito-borne infectious disease caused by Plasmodium parasites.",
    "Dengue":              "A viral infection transmitted by Aedes mosquitoes, common in tropical regions.",
    "Tuberculosis":        "A bacterial infection primarily affecting the lungs, caused by Mycobacterium tuberculosis.",
    "Pneumonia":           "Infection that inflames air sacs in one or both lungs, which may fill with fluid.",
    "Typhoid":             "A bacterial infection caused by Salmonella typhi, spread through contaminated food/water.",
    "Hepatitis A":         "A viral liver infection spread through contaminated food and water.",
    "Hepatitis B":         "A serious liver infection caused by the hepatitis B virus, often chronic.",
    "Hepatitis C":         "A viral infection causing liver inflammation, sometimes leading to cirrhosis.",
    "Heart attack":        "Occurs when blood flow to the heart is blocked. A medical emergency.",
    "Migraine":            "A neurological condition characterised by intense, debilitating headaches.",
    "Chicken pox":         "A highly contagious viral infection causing an itchy, blister-like rash.",
    "Fungal infection":    "Infections caused by fungi, commonly affecting skin, nails, or mucous membranes.",
    "Allergy":             "An immune response to a normally harmless substance (allergen).",
    "Arthritis":           "Inflammation of one or more joints, causing pain and stiffness.",
    "Acne":                "A skin condition that occurs when hair follicles become plugged with oil and dead skin.",
    "Jaundice":            "Yellowing of the skin and eyes due to high bilirubin levels.",
    "Urinary tract infection": "An infection in any part of the urinary system, most commonly the bladder.",
}


# ── Hero ───────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">ML Portfolio Project · Naive Bayes from Scratch</div>
    <h1>🩺 Disease Symptom Classifier</h1>
    <p>Select the symptoms you are experiencing. The model will analyse them using
       a <strong style="color:#90cdf4">Bernoulli Naive Bayes</strong> classifier built entirely
       from scratch with NumPy — no scikit-learn used for the core algorithm.</p>
</div>
""", unsafe_allow_html=True)

# ── Metric cards ───────────────────────────────────────────────
st.markdown(f"""
<div class="metric-row">
    <div class="metric-card">
        <div class="value">{acc*100:.1f}%</div>
        <div class="label">Test Accuracy</div>
    </div>
    <div class="metric-card">
        <div class="value">{f1*100:.1f}%</div>
        <div class="label">Macro F1-Score</div>
    </div>
    <div class="metric-card">
        <div class="value">41</div>
        <div class="label">Diseases Covered</div>
    </div>
    <div class="metric-card">
        <div class="value">131</div>
        <div class="label">Symptoms Tracked</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 Select Symptoms")
    st.markdown("Search and select all symptoms you are experiencing:")
    selected_display = st.multiselect(
        "Symptoms",
        options=symptom_options,
        label_visibility="collapsed",
        placeholder="Type to search symptoms…",
    )

    st.markdown("---")
    st.markdown("**About the Model**")
    st.markdown(f"""
- **Algorithm:** {model_name}
- **Implementation:** NumPy only
- **Smoothing:** Laplace (α = 1.0)
- **Train/Test Split:** 80/20 stratified
- **Log-space inference** for numerical stability
    """)

    st.markdown("---")
    st.markdown("**Quick Test Sets**")
    presets = {
        "🦟 Malaria symptoms": ["Chills", "High Fever", "Sweating", "Headache", "Nausea", "Muscle Pain"],
        "🩸 Diabetes symptoms": ["Fatigue", "Weight Loss", "Excessive Hunger", "Polyuria", "Obesity"],
        "🫀 Heart attack symptoms": ["Chest Pain", "Breathlessness", "Sweating", "Vomiting"],
        "🤧 Common Cold": ["Continuous Sneezing", "Chills", "Fatigue", "Cough", "Runny Nose", "Headache"],
    }
    for label, symptoms in presets.items():
        if st.button(label, use_container_width=True):
            # Filter to symptoms that exist in our list
            valid = [s for s in symptoms if s in symptom_display]
            st.session_state["_preset"] = valid

# Apply preset if set
if "_preset" in st.session_state:
    selected_display = st.session_state.pop("_preset")

# ── Main content ───────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="section-head">Selected Symptoms</div>', unsafe_allow_html=True)

    if selected_display:
        tags_html = " ".join(f'<span class="symptom-tag">{s}</span>' for s in selected_display)
        st.markdown(tags_html, unsafe_allow_html=True)
        st.markdown(f"<br><small style='color:#64748b'>{len(selected_display)} symptom(s) selected</small>", unsafe_allow_html=True)
    else:
        st.info("Select symptoms from the sidebar to begin analysis.")

    st.markdown("")
    predict_btn = st.button("🔬 Analyse Symptoms", type="primary", use_container_width=True, disabled=len(selected_display) == 0)

with col_right:
    st.markdown('<div class="section-head">How It Works</div>', unsafe_allow_html=True)
    st.markdown("""
**Bernoulli Naive Bayes** treats each symptom as a binary feature (present/absent).

The model computes the posterior probability for each disease using Bayes' theorem:

$$P(disease \mid symptoms) \\propto P(disease) \\cdot \\prod_{j} P(x_j \\mid disease)$$

**Key implementation choices:**
- **Log-space computation** — prevents floating-point underflow when multiplying many small probabilities
- **Laplace smoothing (α=1)** — avoids zero-probability problem for unseen symptom-disease combinations
- **Stratified split** — preserves class balance in train/test sets
- **Bernoulli vs Multinomial** — Bernoulli models presence/absence of each symptom, better suited for this binary data
    """)

# ── Prediction output ──────────────────────────────────────────
if predict_btn and selected_display:
    selected_raw = [symptom_display[s] for s in selected_display]

    # Build feature vector
    x = np.zeros((1, len(symptom_cols)))
    for s in selected_raw:
        if s in symptom_cols:
            x[0, symptom_cols.index(s)] = 1.0

    # Predict
    top3 = model.top_k_predictions(x, k=5)[0]
    best_disease, best_prob = top3[0]

    st.markdown("---")
    st.markdown('<div class="section-head">Prediction Results</div>', unsafe_allow_html=True)

    res_col1, res_col2 = st.columns([1, 1], gap="large")

    with res_col1:
        info = disease_info.get(best_disease, "A medical condition identified by the selected symptom pattern.")
        prob_pct = f"{best_prob * 100:.1f}%"
        st.markdown(f"""
<div class="result-card">
    <div style="font-size:0.72rem;color:#64748b;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.5rem">Top Prediction</div>
    <h2>{best_disease}</h2>
    <div class="confidence">{prob_pct} confidence</div>
    <p style="color:#94a3b8;font-size:0.9rem;line-height:1.6;margin:0">{info}</p>
    <div class="disclaimer">⚠️ This tool is for educational purposes only. Always consult a qualified medical professional for diagnosis and treatment.</div>
</div>
""", unsafe_allow_html=True)

    with res_col2:
        st.markdown("**Top 5 Differential Diagnoses**")

        diseases_chart = [d for d, _ in top3][::-1]
        probs_chart    = [p * 100 for _, p in top3][::-1]

        colors = ["#2d4a6e"] * len(diseases_chart)
        colors[len(colors) - 1] = "#63b3ed"  # highlight top prediction

        fig = go.Figure(go.Bar(
            x=probs_chart,
            y=diseases_chart,
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(color="rgba(99,179,237,0.3)", width=1),
            ),
            text=[f"{p:.2f}%" for p in probs_chart],
            textposition="outside",
            textfont=dict(color="#94a3b8", size=11, family="JetBrains Mono"),
        ))

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,26,46,0.8)",
            font=dict(color="#94a3b8", family="Inter"),
            margin=dict(l=10, r=60, t=10, b=10),
            height=280,
            xaxis=dict(
                title="Probability (%)",
                gridcolor="rgba(99,179,237,0.1)",
                color="#64748b",
                range=[0, max(probs_chart) * 1.25],
            ),
            yaxis=dict(color="#94a3b8", tickfont=dict(size=11)),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Probability table
        st.markdown("| Rank | Disease | Probability |")
        st.markdown("|------|---------|------------|")
        for rank, (disease, prob) in enumerate(top3, 1):
            marker = "🎯" if rank == 1 else f"{rank}"
            st.markdown(f"| {marker} | {disease} | `{prob*100:.2f}%` |")


# ── About section ──────────────────────────────────────────────
st.markdown("---")
with st.expander("📐 Technical Implementation Details", expanded=False):
    st.markdown("""
### Implementation Breakdown

**`my_library/naive_bayes.py`** — Three classifiers built from scratch:
- `BernoulliNB` — for binary features (symptom present/absent). **Best for this dataset.**
- `MultinomialNB` — for count features
- `GaussianNB` — for continuous features

**Core formula (log-space):**
```
log P(disease | x) ∝ log P(disease)
                   + Σⱼ [ xⱼ · log P(xⱼ=1 | disease)
                         + (1-xⱼ) · log(1 - P(xⱼ=1 | disease)) ]
```

**`my_library/preprocessing.py`** — Stratified split, label encoding

**`my_library/metrics.py`** — Accuracy, confusion matrix, precision/recall/F1 — all from scratch

### Why Log-Space?
With 131 features, naively multiplying 131 probabilities (each < 1) would give numbers like 10⁻⁵⁰ — below float64 precision. Working in log-space converts multiplication to addition.

### Why Laplace Smoothing?
If a symptom never appears with a disease in training data, P(symptom | disease) = 0, making the entire posterior 0 regardless of other symptoms. Smoothing adds α=1 pseudo-count to every feature.

### Model Comparison Results
| Model | Accuracy | Macro F1 |
|-------|----------|---------|
| BernoulliNB (α=1.0) | 99.90% | 99.90% |
| BernoulliNB (α=0.5) | 99.90% | 99.90% |
| MultinomialNB (α=1.0) | 99.59% | 99.59% |
| GaussianNB | 76.83% | 76.62% |

BernoulliNB wins because the data is inherently binary — each symptom is present or absent, not a count.
    """)

with st.expander("📊 Dataset Information", expanded=False):
    st.markdown("""
- **4,920 samples** across 41 diseases (120 per disease)
- **131 binary symptom features** (1 = present, 0 = absent)
- **80/20 stratified train/test split** → 3,936 train / 984 test
- Balanced classes — no SMOTE needed
- Laplace smoothing handles rare symptom-disease combos
    """)

st.markdown("<br><div style='text-align:center;color:#334155;font-size:0.8rem'>Built by Sajid Ansari · Naive Bayes from Scratch · NumPy only · <a href='https://github.com/sajidansari7' style='color:#475569'>GitHub</a></div>", unsafe_allow_html=True)
