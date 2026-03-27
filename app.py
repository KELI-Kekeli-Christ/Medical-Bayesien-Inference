import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# MOTEUR BAYÉSIEN AMÉLIORÉ
# ─────────────────────────────────────────────

class SmartBayesianAI:
    def __init__(self):
        self.priors = {}
        self.likelihoods = {}
        self.symptoms = []
        self.maladies = []

    def fit(self, df: pd.DataFrame):
        df.columns = [c.strip() for c in df.columns]
        target_col = "prognosis"
        if "Unnamed: 133" in df.columns:
            df = df.drop("Unnamed: 133", axis=1)

        self.maladies = list(df[target_col].unique())
        self.symptoms = [c for c in df.columns if c != target_col]
        total_rows = len(df)

        for maladie in self.maladies:
            df_m = df[df[target_col] == maladie]
            self.priors[maladie] = len(df_m) / total_rows
            # Laplace smoothing
            self.likelihoods[maladie] = {
                s: (df_m[s].sum() + 1) / (len(df_m) + 2) for s in self.symptoms
            }

    def calculate_entropy(self, probs: dict) -> float:
        return -sum(p * math.log2(p + 1e-12) for p in probs.values())

    def get_best_symptom(self, current_probs: dict, excluded: list) -> str:
        """
        Sélectionne le symptôme avec le meilleur Information Gain réel.
        IG(S) = H(current) - [P(S=1)*H(post|S=1) + P(S=0)*H(post|S=0)]
        """
        H_current = self.calculate_entropy(current_probs)
        best_symptom, best_ig = None, -1.0

        for s in self.symptoms:
            if s in excluded:
                continue

            # P(S=1) marginale
            p_s1 = sum(
                self.likelihoods[m][s] * current_probs[m] for m in self.maladies
            )
            p_s0 = 1 - p_s1
            if p_s1 < 1e-9 or p_s0 < 1e-9:
                continue

            # Posteriors si S=1
            post1 = {
                m: current_probs[m] * self.likelihoods[m][s] for m in self.maladies
            }
            t1 = sum(post1.values())
            post1 = {m: v / t1 for m, v in post1.items()}

            # Posteriors si S=0
            post0 = {
                m: current_probs[m] * (1 - self.likelihoods[m][s])
                for m in self.maladies
            }
            t0 = sum(post0.values())
            post0 = {m: v / t0 for m, v in post0.items()}

            expected_H = p_s1 * self.calculate_entropy(post1) + p_s0 * self.calculate_entropy(post0)
            ig = H_current - expected_H

            if ig > best_ig:
                best_ig = ig
                best_symptom = s

        return best_symptom

    def update_probs(self, current_probs: dict, symptom: str, value: int) -> dict:
        """Mise à jour bayésienne + normalisation."""
        new_probs = {}
        for m in self.maladies:
            lk = self.likelihoods[m][symptom] if value == 1 else (1 - self.likelihoods[m][symptom])
            new_probs[m] = current_probs[m] * lk
        total = sum(new_probs.values())
        if total < 1e-15:
            return current_probs  # évite division par zéro
        return {m: p / total for m, p in new_probs.items()}

    def top_diagnoses(self, current_probs: dict, n=5):
        return sorted(current_probs.items(), key=lambda x: x[1], reverse=True)[:n]


# ─────────────────────────────────────────────
# INITIALISATION SESSION STATE
# ─────────────────────────────────────────────

def init_state(ai: SmartBayesianAI):
    if "current_probs" not in st.session_state:
        st.session_state.current_probs = ai.priors.copy()
    if "asked" not in st.session_state:
        st.session_state.asked = []
    if "answers" not in st.session_state:
        st.session_state.answers = {}
    if "finished" not in st.session_state:
        st.session_state.finished = False
    if "history" not in st.session_state:
        st.session_state.history = []  # liste (symptôme, réponse, top_diag)


def reset_state(ai: SmartBayesianAI):
    st.session_state.current_probs = ai.priors.copy()
    st.session_state.asked = []
    st.session_state.answers = {}
    st.session_state.finished = False
    st.session_state.history = []


# ─────────────────────────────────────────────
# GRAPHIQUES
# ─────────────────────────────────────────────

def plot_top_probs(current_probs: dict, n=8):
    top = sorted(current_probs.items(), key=lambda x: x[1], reverse=True)[:n]
    labels = [t[0].replace("_", " ") for t in top]
    values = [round(t[1] * 100, 2) for t in top]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker=dict(
            color=values,
            colorscale="Teal",
            showscale=False,
        ),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        margin=dict(l=0, r=60, t=10, b=10),
        xaxis=dict(range=[0, max(values) * 1.25], showgrid=False, zeroline=False,
                   ticksuffix="%"),
        yaxis=dict(autorange="reversed"),
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
    )
    return fig


def plot_entropy_history(history):
    if len(history) < 2:
        return None
    entropies = [h["entropy"] for h in history]
    steps = list(range(len(entropies)))
    fig = go.Figure(go.Scatter(
        x=steps, y=entropies,
        mode="lines+markers",
        line=dict(color="#00b4a0", width=2),
        marker=dict(size=6),
    ))
    fig.update_layout(
        xaxis_title="Questions posées",
        yaxis_title="Entropie (bits)",
        height=200,
        margin=dict(l=0, r=0, t=10, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ─────────────────────────────────────────────
# APP PRINCIPALE
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Diagnostic Bayésien",
    page_icon="🩺",
    layout="wide",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'DM Serif Display', serif; }

    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
    }
    .diagnosis-card {
        background: linear-gradient(135deg, #0f766e22, #0f766e08);
        border: 1px solid #0f766e55;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.5rem;
    }
    .badge {
        display: inline-block;
        background: #0f766e;
        color: white;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .entropy-badge {
        background: #64748b22;
        border: 1px solid #64748b55;
        border-radius: 8px;
        padding: 0.4rem 0.8rem;
        font-size: 0.85rem;
        color: #64748b;
    }
</style>
""", unsafe_allow_html=True)


# ── Chargement modèle ──
@st.cache_resource
def load_model():
    ai = SmartBayesianAI()
    return ai

@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)


ai = load_model()

# ── Sidebar ──
with st.sidebar:
    st.title("🩺 Diagnostic IA")
    st.markdown("Moteur **bayésien** avec sélection optimale des questions par *Information Gain*.")
    st.divider()

    uploaded = st.file_uploader("📂 Charger Training.csv", type="csv")
    if uploaded:
        df_data = load_data(uploaded)
        ai.fit(df_data)
        st.success(f"✅ {len(ai.maladies)} maladies · {len(ai.symptoms)} symptômes")
        if "model_loaded" not in st.session_state:
            st.session_state.model_loaded = True
            init_state(ai)

    if st.button("🔄 Recommencer", use_container_width=True):
        reset_state(ai)
        st.rerun()

    st.divider()
    MAX_QUESTIONS = st.slider("Questions max", 5, 20, 10)
    CONFIDENCE_THRESHOLD = st.slider("Seuil de confiance (%)", 70, 99, 95) / 100


# ── Corps principal ──
if "model_loaded" not in st.session_state:
    st.title("Diagnostic Médical Bayésien")
    st.info("👈 Chargez le fichier **Training.csv** dans la sidebar pour démarrer.")
    st.markdown("""
    **Comment ça marche ?**
    1. Chargez le dataset Kaggle *Disease Symptom Prediction*
    2. L'IA pose les questions les plus informatives en premier (**Information Gain** réel)
    3. Elle met à jour ses probabilités après chaque réponse (**Théorème de Bayes**)
    4. Elle s'arrête quand elle est suffisamment confiante
    """)
    st.stop()

init_state(ai)

col_main, col_viz = st.columns([3, 2], gap="large")

with col_main:
    st.title("Diagnostic Bayésien")

    n_asked = len(st.session_state.asked)
    entropy = ai.calculate_entropy(st.session_state.current_probs)
    top1 = ai.top_diagnoses(st.session_state.current_probs, 1)[0]

    # Métriques
    m1, m2, m3 = st.columns(3)
    m1.metric("Questions posées", f"{n_asked} / {MAX_QUESTIONS}")
    m2.metric("Entropie", f"{entropy:.2f} bits")
    m3.metric("Top diagnostic", f"{top1[1]*100:.1f}%")

    st.divider()

    # ── Fin de diagnostic ──
    if st.session_state.finished or top1[1] >= CONFIDENCE_THRESHOLD or n_asked >= MAX_QUESTIONS:
        st.session_state.finished = True
        top5 = ai.top_diagnoses(st.session_state.current_probs, 5)

        confidence = top5[0][1]
        if confidence >= CONFIDENCE_THRESHOLD:
            st.success(f"✅ Diagnostic établi avec **{confidence*100:.1f}%** de confiance")
        else:
            st.warning("⚠️ Limite de questions atteinte — diagnostic le plus probable :")

        for i, (maladie, prob) in enumerate(top5):
            rank_emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i]
            st.markdown(f"""
            <div class="diagnosis-card">
                {rank_emoji} <strong>{maladie.replace('_', ' ')}</strong>
                &nbsp;&nbsp;<span class="badge">{prob*100:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
            if i == 0:
                st.progress(prob)

        st.divider()
        st.markdown("**⚠️ Avertissement médical** : Cet outil est éducatif. Consultez un médecin pour tout diagnostic réel.")

    else:
        # ── Question suivante ──
        best_s = ai.get_best_symptom(st.session_state.current_probs, st.session_state.asked)

        if best_s is None:
            st.warning("Plus de symptômes disponibles.")
        else:
            symptom_display = best_s.replace("_", " ").capitalize()

            st.markdown(f"### Question {n_asked + 1}")
            st.markdown(f"**Présentez-vous le symptôme suivant ?**")
            st.markdown(f"## 🔍 {symptom_display}")

            # Calcul de l'IG pour info
            p_s1 = sum(ai.likelihoods[m][best_s] * st.session_state.current_probs[m] for m in ai.maladies)
            st.markdown(f'<span class="entropy-badge">Prévalence estimée : {p_s1*100:.0f}%</span>', unsafe_allow_html=True)

            col_oui, col_non, col_nsp = st.columns([1, 1, 1])

            def answer(val):
                new_probs = ai.update_probs(st.session_state.current_probs, best_s, val)
                st.session_state.history.append({
                    "symptom": best_s,
                    "answer": val,
                    "entropy": ai.calculate_entropy(new_probs),
                    "top": ai.top_diagnoses(new_probs, 1)[0][0],
                })
                st.session_state.current_probs = new_probs
                st.session_state.asked.append(best_s)
                st.session_state.answers[best_s] = val

            with col_oui:
                if st.button("✅ Oui", use_container_width=True, type="primary"):
                    answer(1)
                    st.rerun()
            with col_non:
                if st.button("❌ Non", use_container_width=True):
                    answer(0)
                    st.rerun()
            with col_nsp:
                if st.button("❓ Ne sais pas", use_container_width=True):
                    # On ignore ce symptôme sans mettre à jour
                    st.session_state.asked.append(best_s)
                    st.rerun()

        # Historique des réponses
        if st.session_state.history:
            st.divider()
            with st.expander("📋 Historique des réponses"):
                for h in reversed(st.session_state.history):
                    icon = "✅" if h["answer"] == 1 else "❌"
                    st.markdown(
                        f"{icon} **{h['symptom'].replace('_', ' ')}** → top : {h['top'].replace('_', ' ')}"
                    )


with col_viz:
    st.markdown("### 📊 Probabilités en temps réel")
    fig_probs = plot_top_probs(st.session_state.current_probs)
    st.plotly_chart(fig_probs, use_container_width=True)

    if len(st.session_state.history) >= 2:
        st.markdown("### 📉 Réduction d'entropie")
        fig_entropy = plot_entropy_history(
            [{"entropy": ai.calculate_entropy(ai.priors)}] + st.session_state.history
        )
        if fig_entropy:
            st.plotly_chart(fig_entropy, use_container_width=True)
        st.caption("L'entropie mesure l'incertitude : plus elle baisse, plus le diagnostic est précis.")

    # Top 3 en temps réel
    st.markdown("### 🎯 Top 3 actuel")
    for maladie, prob in ai.top_diagnoses(st.session_state.current_probs, 3):
        st.progress(prob, text=f"{maladie.replace('_', ' ')} — {prob*100:.1f}%")