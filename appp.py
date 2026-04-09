
# ==============================
# IMPORTS
# ==============================
import streamlit as st
import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.inspection import PartialDependenceDisplay
import shap
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from lime.lime_tabular import LimeTabularExplainer

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "text.usetex": False,
    "text.parse_math": False,   # prevents $ in feature names triggering mathtext parser
})

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="MTT24.5 · Cognitive Prediction System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==============================
# GLOBAL CSS
# ==============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500&family=DM+Mono:wght@300;400&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
}

/* Hide default Streamlit header/footer */
#MainMenu, footer, header { visibility: hidden; }

/* App background */
.stApp { background: #f5f4f0; }

/* Remove default padding */
.block-container { padding-top: 0 !important; max-width: 100% !important; }

/* ── Hero banner ── */
.hero {
    background: #ffffff;
    border-bottom: 1px solid #e4e2da;
    padding: 1.75rem 2.5rem 1.5rem;
    display: flex;
    align-items: flex-start;
    gap: 1.25rem;
    margin-bottom: 0;
}
.hero-icon {
    width: 46px; height: 46px;
    background: #042C53;
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    font-size: 22px;
    line-height: 1;
}
.hero-title {
    font-size: 1.35rem;
    font-weight: 500;
    color: #1a1a1a;
    letter-spacing: -0.3px;
    margin: 0 0 3px;
}
.hero-sub {
    font-size: 0.8rem;
    color: #6b6b6b;
    font-weight: 300;
    margin: 0 0 8px;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #E6F1FB;
    color: #0C447C;
    font-size: 0.7rem;
    font-weight: 500;
    padding: 3px 10px;
    border-radius: 20px;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.3px;
}
.badge-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #378ADD;
    display: inline-block;
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.35} }

/* ── Section label ── */
.section-label {
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 1.3px;
    color: #999;
    text-transform: uppercase;
    margin: 1.75rem 0 0.75rem;
}

/* ── Pillar card ── */
.pillar-card {
    background: #ffffff;
    border: 0.5px solid #dedad2;
    border-radius: 12px;
    padding: 1rem;
    height: 100%;
}
.pillar-num {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #bbb;
    margin-bottom: 5px;
}
.pillar-title {
    font-size: 0.82rem;
    font-weight: 500;
    color: #1a1a1a;
    margin-bottom: 5px;
}
.pillar-desc {
    font-size: 0.75rem;
    color: #6b6b6b;
    font-weight: 300;
    line-height: 1.5;
}
.pillar-bar-wrap {
    height: 3px;
    background: #ede9e0;
    border-radius: 2px;
    margin-top: 10px;
    overflow: hidden;
}
.pillar-bar-fill {
    height: 100%;
    background: #378ADD;
    border-radius: 2px;
}

/* ── Insight banner ── */
.insight-bar {
    background: #042C53;
    border-radius: 10px;
    padding: 1rem 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: 1rem 0 1.75rem;
}
.insight-text {
    font-size: 0.8rem;
    color: #B5D4F4;
    font-weight: 300;
    line-height: 1.5;
}
.insight-text b { color: #E6F1FB; font-weight: 500; }

/* ── Metric card ── */
.metric-card {
    background: #ffffff;
    border: 0.5px solid #dedad2;
    border-radius: 10px;
    padding: 0.875rem 1rem;
}
.metric-label {
    font-size: 0.68rem;
    color: #999;
    font-weight: 400;
    margin-bottom: 3px;
}
.metric-val {
    font-size: 1.3rem;
    font-weight: 500;
    font-family: 'DM Mono', monospace;
    color: #1a1a1a;
    letter-spacing: -0.5px;
}
.metric-sub {
    font-size: 0.65rem;
    color: #bbb;
    font-weight: 300;
    margin-top: 2px;
}

/* ── Predict panels ── */
.panel {
    background: #ffffff;
    border: 0.5px solid #dedad2;
    border-radius: 12px;
    padding: 1.5rem;
}
.panel-title {
    font-size: 0.8rem;
    font-weight: 500;
    color: #1a1a1a;
    padding-bottom: 0.75rem;
    border-bottom: 0.5px solid #ede9e0;
    margin-bottom: 1.25rem;
}

/* ── Result card ── */
.result-card {
    background: #f5f4f0;
    border: 0.5px solid #dedad2;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.25rem;
}
.result-label { font-size: 0.75rem; color: #6b6b6b; }
.result-score {
    font-family: 'DM Mono', monospace;
    font-size: 2rem;
    font-weight: 400;
    color: #1a1a1a;
    letter-spacing: -1px;
}
.result-score span { font-size: 0.85rem; color: #999; font-weight: 300; }
.tier-high { background:#EAF3DE; color:#3B6D11; font-size:0.7rem; padding:4px 10px; border-radius:20px; font-weight:500; }
.tier-mid  { background:#FAEEDA; color:#854F0B; font-size:0.7rem; padding:4px 10px; border-radius:20px; font-weight:500; }
.tier-low  { background:#FCEBEB; color:#A32D2D; font-size:0.7rem; padding:4px 10px; border-radius:20px; font-weight:500; }

/* ── XAI section title ── */
.xai-title {
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.9px;
    color: #aaa;
    text-transform: uppercase;
    margin: 1.25rem 0 0.6rem;
}

/* ── RAG card ── */
.rag-item {
    display: flex;
    gap: 8px;
    padding: 7px 0;
    border-bottom: 0.5px solid #ede9e0;
    font-size: 0.75rem;
    color: #555;
    line-height: 1.5;
}
.rag-item:last-child { border-bottom: none; }
.rag-dot {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: #378ADD;
    flex-shrink: 0;
    margin-top: 5px;
}

/* ── Optimised features strip ── */
.feat-strip {
    background: #E6F1FB;
    border: 0.5px solid #B5D4F4;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    font-size: 0.72rem;
    color: #0C447C;
    font-family: 'DM Mono', monospace;
    margin-bottom: 1.5rem;
}

/* ── Footer ── */
.footer {
    font-size: 0.68rem;
    color: #bbb;
    font-weight: 300;
    margin-top: 2rem;
    padding-top: 1.25rem;
    border-top: 0.5px solid #dedad2;
}

/* ── Streamlit widget tweaks ── */
.stSlider > div > div > div { background: #378ADD !important; }
div[data-testid="stSlider"] label { font-size: 0.78rem !important; color: #444 !important; }
.stSelectbox label { font-size: 0.78rem !important; color: #444 !important; }
div[data-testid="stButton"] > button {
    background: #042C53 !important;
    color: #E6F1FB !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 0.65rem 1.5rem !important;
    width: 100% !important;
    transition: background 0.15s !important;
}
div[data-testid="stButton"] > button:hover { background: #185FA5 !important; }
</style>
""", unsafe_allow_html=True)


# ==============================
# HERO
# ==============================
st.markdown("""
<div class="hero">
  <div class="hero-icon">🧠</div>
  <div>
    <div class="hero-title">Cognitive Enhancement &amp; Prediction System</div>
    <div class="hero-sub">Research-driven cognitive training program · MTT24.5 · ML-powered outcome prediction</div>
    <span class="hero-badge"><span class="badge-dot"></span>Model active · GBR + GA feature selection</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ==============================
# LOAD DATA
# ==============================
@st.cache_data
def load_data():
    df = pd.read_csv("mtt245_real_dataset (1).csv")
    df = df.drop(columns=["id"], errors="ignore")
    df = df[df["improvement_ace_global"].notna()]
    df["intervention"] = df["group"].map({"CONTROL": 0, "INTERVENTION": 1})
    df["adherence_pct"] = df["adherence_pct"].fillna(0)
    return df

df = load_data()
features = [
    "intervention", "adherence_pct", "sex", "age",
    "education_years", "cognitive_reserve",
    "diabetes", "hypertension", "dyslipidemia",
    "pre_ace_global"
]
X = df[features]
y = df["improvement_ace_global"]


# ==============================
# GA OPTIMISATION
# ==============================
@st.cache_resource
def run_ga():
    def fitness(mask):
        selected = [f for f, m in zip(features, mask) if m == 1]
        if not selected:
            return -999
        model = Pipeline([
            ("imputer", SimpleImputer()),
            ("scaler", StandardScaler()),
            ("gb", GradientBoostingRegressor(random_state=42))
        ])
        cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)
        return cross_val_score(model, X[selected], y, cv=cv, scoring="r2").mean()

    population = [np.random.randint(0, 2, len(features)) for _ in range(8)]
    for _ in range(8):
        scores = [fitness(ind) for ind in population]
        pairs = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)
        population = [p[1] for p in pairs[:4]]
        while len(population) < 8:
            p1, p2 = random.sample(population[:4], 2)
            child = np.where(np.random.rand(len(features)) > 0.5, p1, p2)
            population.append(child)
    return [f for f, m in zip(features, population[0]) if m == 1]

selected_features = run_ga()


# ==============================
# MODEL
# ==============================
@st.cache_resource
def train_model():
    model = Pipeline([
        ("imputer", SimpleImputer()),
        ("scaler", StandardScaler()),
        ("gb", GradientBoostingRegressor(random_state=42))
    ])
    model.fit(X[selected_features], y)
    return model

model = train_model()


# ==============================
# RAG KNOWLEDGE BASE
# ==============================
knowledge_base = [
    "The MTT24.5 program significantly improves global cognitive performance compared to control groups.",
    "Memory shows the highest improvement among cognitive domains after intervention.",
    "Participants with lower baseline cognitive scores show greater improvement.",
    "The program is effective across age, sex, and clinical conditions such as diabetes and hypertension.",
    "Cognitive improvement is driven by neuroplasticity through acquisition of new knowledge and learning techniques.",
    "Multisensory and novel learning tasks stimulate brain adaptation more effectively than repetitive tasks."
]
vectorizer = TfidfVectorizer()
kb_vectors = vectorizer.fit_transform(knowledge_base)

def retrieve_context(user_input_df):
    query = (
        f"adherence {user_input_df['adherence_pct'].values[0]}, "
        f"baseline {user_input_df['pre_ace_global'].values[0]}, "
        f"age {user_input_df['age'].values[0]}, "
        f"cognitive reserve {user_input_df['cognitive_reserve'].values[0]}, "
        f"diseases {user_input_df['diabetes'].values[0]} {user_input_df['hypertension'].values[0]}"
    )
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, kb_vectors)[0]
    return [knowledge_base[i] for i in sims.argsort()[-3:][::-1]]


# ==============================
# METHODOLOGY SECTION
# ==============================
st.markdown('<div class="section-label">Training methodology</div>', unsafe_allow_html=True)

pillars = [
    ("01", "Multi-task training",
     "Simultaneous tasks combining thinking and coordination strengthen neural connections.", 92),
    ("02", "Skill acquisition",
     "Unfamiliar activities activate dormant brain regions and promote neuroplasticity.", 85),
    ("03", "Cognitive flexibility",
     "Task switching and non-dominant hand exercises enhance neural adaptability.", 78),
    ("04", "Adherence",
     "Regular practice is the single strongest predictor of cognitive improvement.", 97),
]

cols = st.columns(4)
for col, (num, title, desc, pct) in zip(cols, pillars):
    with col:
        st.markdown(f"""
        <div class="pillar-card">
          <div class="pillar-num">{num}</div>
          <div class="pillar-title">{title}</div>
          <div class="pillar-desc">{desc}</div>
          <div class="pillar-bar-wrap">
            <div class="pillar-bar-fill" style="width:{pct}%"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div class="insight-bar">
  <div style="font-size:1.2rem;flex-shrink:0;">💡</div>
  <div class="insight-text">
    <b>Key insight:</b> Participants with lower baseline ACE scores combined with high program
    adherence consistently show the greatest improvement margins across all cognitive domains.
  </div>
</div>
""", unsafe_allow_html=True)


# ==============================
# MODEL DIAGNOSTICS
# ==============================
st.markdown('<div class="section-label">Model diagnostics</div>', unsafe_allow_html=True)

mc1, mc2, mc3, mc4 = st.columns(4)
diagnostics = [
    ("Algorithm", "Gradient Boost", "GradientBoostingRegressor"),
    ("Feature selection", "Genetic Alg.", f"{len(selected_features)} of {len(features)} features"),
    ("Cross-validation", "RepeatedKFold", "5 splits · 2 repeats"),
    ("Optimised features", ", ".join(selected_features[:3]) + "…", "GA-selected subset"),
]
for col, (label, val, sub) in zip([mc1, mc2, mc3, mc4], diagnostics):
    with col:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-val" style="font-size:0.9rem;padding-top:3px;">{val}</div>
          <div class="metric-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown(f"""
<div class="feat-strip">
  ✦ &nbsp;Optimised feature set: &nbsp;{' · '.join(selected_features)}
</div>
""", unsafe_allow_html=True)


# ==============================
# PREDICTION INTERFACE
# ==============================
st.markdown('<div class="section-label">Prediction interface</div>', unsafe_allow_html=True)

left, right = st.columns([1, 1], gap="medium")

# ── LEFT: inputs ──────────────────────────────────────────────────────────────
with left:
    st.markdown('<div class="panel"><div class="panel-title">Patient parameters</div>', unsafe_allow_html=True)

    intervention = st.selectbox("Intervention group", options=[1, 0],
                                format_func=lambda x: "Active (intervention)" if x == 1 else "Control")
    adherence = st.slider("Adherence %", 0, 100, 80)
    baseline  = st.slider("Baseline ACE score", 50, 100, 85)
    age       = st.slider("Age", 20, 90, 60)
    education = st.slider("Education years", 0, 25, 15)

    c1, c2 = st.columns(2)
    with c1:
        sex      = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    with c2:
        reserve      = st.selectbox("Cognitive reserve", [1, 0], format_func=lambda x: "High" if x == 1 else "Low")
        hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    dyslipidemia = st.selectbox("Dyslipidemia", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    run = st.button("Run prediction model →")
    st.markdown("</div>", unsafe_allow_html=True)


# ── RIGHT: outputs ─────────────────────────────────────────────────────────────
with right:
    st.markdown('<div class="panel"><div class="panel-title">Model output &amp; explainability</div>', unsafe_allow_html=True)

    if run:
        full_input = pd.DataFrame([{
            "intervention": intervention,
            "adherence_pct": adherence,
            "sex": sex,
            "age": age,
            "education_years": education,
            "cognitive_reserve": reserve,
            "diabetes": diabetes,
            "hypertension": hypertension,
            "dyslipidemia": dyslipidemia,
            "pre_ace_global": baseline
        }])
        user_input = full_input[selected_features]

        prediction = model.predict(user_input)[0]

        if prediction >= 10:
            tier_html = f'<span class="tier-high">High responder</span>'
        elif prediction >= 5:
            tier_html = f'<span class="tier-mid">Moderate responder</span>'
        else:
            tier_html = f'<span class="tier-low">Low responder</span>'

        st.markdown(f"""
        <div class="result-card">
          <div>
            <div class="result-label">Predicted ACE improvement</div>
            <div class="result-score">{prediction:.2f}<span> pts</span></div>
          </div>
          {tier_html}
        </div>
        """, unsafe_allow_html=True)

        def safe_tight_layout(fig):
            try:
                fig.tight_layout()
            except Exception:
                pass

        # ── SHAP waterfall ────────────────────────────────────────────────────
        st.markdown('<div class="xai-title">SHAP — local feature attribution</div>', unsafe_allow_html=True)
        with st.spinner("Computing SHAP values…"):
            shap_explainer = shap.Explainer(model.predict, X[selected_features])
            shap_values    = shap_explainer(user_input)
            shap.plots.waterfall(shap_values[0], show=False)
            fig_shap = plt.gcf()
            fig_shap.patch.set_facecolor("#ffffff")
            fig_shap.set_size_inches(6, 3.5)
            safe_tight_layout(fig_shap)
            st.pyplot(fig_shap)
            plt.close(fig_shap)

        # ── Global SHAP ───────────────────────────────────────────────────────
        st.markdown('<div class="xai-title">SHAP — global feature importance</div>', unsafe_allow_html=True)
        with st.spinner("Computing global SHAP…"):
            sample_X    = X[selected_features].sample(min(200, len(X)), random_state=42)
            shap_global = shap_explainer(sample_X)
            shap.summary_plot(shap_global, sample_X, show=False)
            fig_glob = plt.gcf()
            fig_glob.patch.set_facecolor("#ffffff")
            fig_glob.set_size_inches(6, 3.5)
            safe_tight_layout(fig_glob)
            st.pyplot(fig_glob)
            plt.close(fig_glob)

        # ── LIME ─────────────────────────────────────────────────────────────
        st.markdown('<div class="xai-title">LIME — local surrogate explanation</div>', unsafe_allow_html=True)
        with st.spinner("Running LIME…"):
            lime_exp_obj = LimeTabularExplainer(
                training_data=X[selected_features].values,
                feature_names=selected_features,
                mode="regression"
            )
            lime_result = lime_exp_obj.explain_instance(user_input.values[0], model.predict)
            fig_lime    = lime_result.as_pyplot_figure()
            fig_lime.patch.set_facecolor("#ffffff")
            fig_lime.set_size_inches(6, 3.5)
            safe_tight_layout(fig_lime)
            st.pyplot(fig_lime)
            plt.close(fig_lime)

        # ── PDP / ICE ─────────────────────────────────────────────────────────
        feature_choice = st.selectbox("Select feature for PDP / ICE", selected_features)
        feat_idx       = selected_features.index(feature_choice)

        col_pdp, col_ice = st.columns(2)
        with col_pdp:
            st.markdown('<div class="xai-title">Partial dependence (PDP)</div>', unsafe_allow_html=True)
            fig_pdp, ax_pdp = plt.subplots(figsize=(3.5, 3))
            fig_pdp.patch.set_facecolor("#ffffff")
            PartialDependenceDisplay.from_estimator(model, X[selected_features], [feat_idx], ax=ax_pdp)
            ax_pdp.set_title("")
            safe_tight_layout(fig_pdp)
            st.pyplot(fig_pdp)
            plt.close(fig_pdp)

        with col_ice:
            st.markdown('<div class="xai-title">Individual conditional (ICE)</div>', unsafe_allow_html=True)
            fig_ice, ax_ice = plt.subplots(figsize=(3.5, 3))
            fig_ice.patch.set_facecolor("#ffffff")
            PartialDependenceDisplay.from_estimator(
                model, X[selected_features], [feat_idx], kind="individual", ax=ax_ice
            )
            ax_ice.set_title("")
            safe_tight_layout(fig_ice)
            st.pyplot(fig_ice)
            plt.close(fig_ice)

        # ── RAG ───────────────────────────────────────────────────────────────
        st.markdown('<div class="xai-title">Research-based context (RAG)</div>', unsafe_allow_html=True)
        context = retrieve_context(full_input)
        rag_html = "".join(
            f'<div class="rag-item"><div class="rag-dot"></div><span>{c}</span></div>'
            for c in context
        )
        st.markdown(f'<div style="background:#f5f4f0;border-radius:10px;padding:0.75rem 1rem;">{rag_html}</div>',
                    unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="display:flex;flex-direction:column;align-items:center;
                    justify-content:center;height:300px;gap:10px;color:#bbb;">
          <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
            <circle cx="24" cy="24" r="22" stroke="#dedad2" stroke-width="1.2"/>
            <circle cx="24" cy="24" r="12" stroke="#dedad2" stroke-width="0.8" stroke-dasharray="3 4"/>
            <circle cx="24" cy="24" r="4" fill="#dedad2"/>
          </svg>
          <div style="font-size:0.78rem;font-weight:300;text-align:center;max-width:200px;line-height:1.7;color:#aaa;">
            Configure patient parameters and press <b style="color:#888;">Run prediction model</b> to see output
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ==============================
# FOOTER
# ==============================
st.markdown("""
<div class="footer">
  ⓘ &nbsp;MTT24.5 research program · Predictions are model estimates for research purposes only ·
  Not intended for clinical decision-making
</div>
""", unsafe_allow_html=True)
