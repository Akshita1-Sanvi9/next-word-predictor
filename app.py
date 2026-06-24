import streamlit as st
import numpy as np
import pickle
import os
from tf_keras.models import load_model
from tf_keras.preprocessing.sequence import pad_sequences

# Base directory — same folder as this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="✨",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main { background-color: #0f0f1a; }

/* Hero header */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}
.hero h1 {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.4rem;
}
.hero p {
    color: #94a3b8;
    font-size: 1rem;
    margin: 0;
}

/* Input card */
.input-card {
    background: #1e1e30;
    border: 1px solid #2d2d4e;
    border-radius: 16px;
    padding: 1.8rem;
    margin: 1.5rem 0;
}

/* Prediction pills */
.pill-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    margin-top: 1rem;
}
.pill {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: #fff;
    border-radius: 999px;
    padding: 0.45rem 1.1rem;
    font-size: 0.95rem;
    font-weight: 500;
    cursor: pointer;
    transition: transform 0.15s, box-shadow 0.15s;
    display: inline-block;
}
.pill:hover { transform: translateY(-2px); box-shadow: 0 4px 14px rgba(124,58,237,0.45); }

/* Section label */
.section-label {
    color: #94a3b8;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

/* Generated text box */
.gen-box {
    background: #13131f;
    border: 1px solid #2d2d4e;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    color: #e2e8f0;
    font-size: 1.05rem;
    line-height: 1.7;
    min-height: 80px;
    word-break: break-word;
}
.gen-box span.highlight {
    color: #a78bfa;
    font-weight: 600;
}

/* Streamlit overrides */
div[data-testid="stTextArea"] textarea {
    background: #13131f !important;
    color: #e2e8f0 !important;
    border: 1px solid #2d2d4e !important;
    border-radius: 10px !important;
    font-size: 1rem !important;
}
div[data-testid="stTextArea"] textarea:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.2) !important;
}
div[data-testid="stSlider"] { padding-top: 0.3rem; }
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.6rem 1.6rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }
</style>
""", unsafe_allow_html=True)

# ── Load model & artifacts ────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model     = load_model(os.path.join(BASE_DIR, "lstm_model (1).h5"))
    with open(os.path.join(BASE_DIR, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)
    with open(os.path.join(BASE_DIR, "max_len.pkl"), "rb") as f:
        max_len = pickle.load(f)
    return model, tokenizer, max_len

model, tokenizer, max_len = load_artifacts()

# ── Helper: predict next N words ──────────────────────────────────────────────
def predict_next_words(seed_text: str, n_words: int, top_k: int = 5):
    """Return (generated_text, list_of_top_k_next_word_candidates)."""
    generated = seed_text.strip()
    last_candidates = []

    for i in range(n_words):
        token_seq = tokenizer.texts_to_sequences([generated])[0]
        padded    = pad_sequences([token_seq], maxlen=max_len - 1, padding="pre")
        probs     = model.predict(padded, verbose=0)[0]

        # top-k candidates (only for the last step — shown as pills)
        top_indices = np.argsort(probs)[::-1][:top_k]
        if i == n_words - 1:
            last_candidates = [
                (w, float(probs[idx]))
                for idx, w in [
                    (idx, next((w for w, id_ in tokenizer.word_index.items() if id_ == idx), None))
                    for idx in top_indices
                ]
                if w is not None
            ]

        next_idx  = np.argmax(probs)
        next_word = next(
            (w for w, id_ in tokenizer.word_index.items() if id_ == next_idx), ""
        )
        if not next_word:
            break
        generated += " " + next_word

    return generated, last_candidates

# ── Session state ─────────────────────────────────────────────────────────────
if "generated" not in st.session_state:
    st.session_state.generated = ""
if "candidates" not in st.session_state:
    st.session_state.candidates = []
if "seed" not in st.session_state:
    st.session_state.seed = ""

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>✨ Next Word Predictor</h1>
  <p>Type a seed phrase and let the LSTM continue your thought.</p>
</div>
""", unsafe_allow_html=True)

# Input card
with st.container():
    seed_input = st.text_area(
        "Seed text",
        placeholder="e.g.  The quick brown fox",
        height=110,
        label_visibility="collapsed",
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        n_words = st.slider("Words to generate", min_value=1, max_value=20, value=5)
    with col2:
        top_k = st.slider("Top-K suggestions", min_value=3, max_value=10, value=5)

    predict_btn = st.button("Generate →", use_container_width=True)

# Run prediction
if predict_btn:
    if not seed_input.strip():
        st.warning("Please enter some seed text first.")
    else:
        with st.spinner("Predicting..."):
            generated, candidates = predict_next_words(seed_input, n_words, top_k)
        st.session_state.generated  = generated
        st.session_state.candidates = candidates
        st.session_state.seed       = seed_input.strip()

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.generated:
    st.markdown("---")

    # Generated text
    seed_part  = st.session_state.seed
    full_text  = st.session_state.generated
    new_part   = full_text[len(seed_part):].strip()

    st.markdown('<div class="section-label">Generated text</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="gen-box">{seed_part} <span class="highlight">{new_part}</span></div>',
        unsafe_allow_html=True,
    )

    # Top-K suggestions as pills
    if st.session_state.candidates:
        st.markdown('<div class="section-label" style="margin-top:1.4rem;">Top next-word suggestions</div>', unsafe_allow_html=True)
        pills_html = '<div class="pill-row">'
        for word, prob in st.session_state.candidates:
            pills_html += f'<span class="pill">{word} <span style="opacity:.65;font-size:.8em">{prob:.1%}</span></span>'
        pills_html += "</div>"
        st.markdown(pills_html, unsafe_allow_html=True)

    # Copy / Reset
    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.code(full_text, language=None)   # easy one-click copy
    with col_b:
        if st.button("🔄  Reset", use_container_width=True):
            st.session_state.generated  = ""
            st.session_state.candidates = []
            st.session_state.seed       = ""
            st.rerun()

