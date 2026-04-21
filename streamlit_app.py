"""
streamlit_app.py
Streamlit web app for Handwritten Math Equation Solver.
Upload a photo/scan of a handwritten equation → get the answer.

Run with:  streamlit run streamlit_app.py
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import os
import sys

# ── page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Handwritten Math Solver",
    page_icon="✏️",
    layout="centered",
)

# ── project modules ───────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from segment import get_segments
from predict import predict_sequence
from solver  import solve_from_symbols

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f0f23; }
    .block-container { padding-top: 2rem; }

    .title-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #4ecca3;
        border-radius: 12px;
        padding: 1.4rem 2rem;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .title-box h1 { color: #4ecca3; margin: 0; font-size: 2rem; }
    .title-box p  { color: #aaaaaa; margin: 0.4rem 0 0; font-size: 0.95rem; }

    .result-box {
        background: #1a1a2e;
        border-left: 4px solid #4ecca3;
        border-radius: 8px;
        padding: 1.2rem 1.6rem;
        margin-top: 1.2rem;
    }
    .result-box .label { color: #888888; font-size: 0.85rem; margin-bottom: 4px; }
    .result-box .expr  { color: #e2e2e2; font-size: 1.1rem; font-family: monospace; }
    .result-box .answer{ color: #4ecca3; font-size: 2.2rem; font-weight: 700; }

    .error-box {
        background: #2e1a1a;
        border-left: 4px solid #e94560;
        border-radius: 8px;
        padding: 1rem 1.4rem;
        color: #e94560;
        margin-top: 1rem;
    }

    .sym-pill {
        display: inline-block;
        background: #16213e;
        border: 1px solid #4ecca3;
        border-radius: 20px;
        padding: 3px 12px;
        margin: 3px;
        color: #4ecca3;
        font-family: monospace;
        font-size: 1rem;
    }
    .conf-low { border-color: #e94560; color: #e94560; }

    .step-card {
        background: #1a1a2e;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
        border: 1px solid #2a2a4e;
        font-size: 0.9rem;
        color: #cccccc;
    }
    .step-num {
        background: #4ecca3;
        color: #0f0f23;
        border-radius: 50%;
        width: 22px; height: 22px;
        display: inline-flex;
        align-items: center; justify-content: center;
        font-weight: 700; font-size: 0.8rem;
        margin-right: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ── header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-box">
  <h1>✏️ Handwritten Math Equation Solver</h1>
  <p>Upload a photo of your handwritten equation — get the answer instantly</p>
</div>
""", unsafe_allow_html=True)

# ── check model ───────────────────────────────────────────────────────────────
MODEL_READY = os.path.exists("model.h5") and os.path.exists("labels.npy")

if not MODEL_READY:
    st.error("⚠️ **Model not found.** Please run `python train_model.py` first, then restart the app.")
    st.stop()

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    conf_threshold = st.slider("Confidence threshold (%)",
                                min_value=10, max_value=95, value=50,
                                help="Symbols below this confidence are flagged")
    show_segments = st.checkbox("Show segmented symbols", value=True)
    show_steps    = st.checkbox("Show processing steps", value=False)

    st.markdown("---")
    st.markdown("### 📋 Supported Symbols")
    st.markdown("**Digits:** 0 1 2 3 4 5 6 7 8 9")
    st.markdown("**Operators:** `+`  `-`  `×`  `÷`")
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("""
- Write **large & clear**
- Leave **space** between symbols
- Use **dark ink on white** paper
- Good **lighting** for phone photos
    """)

# ── upload section ────────────────────────────────────────────────────────────
st.markdown("#### 📤 Upload your equation image")

col1, col2 = st.columns([3, 1])
with col1:
    uploaded = st.file_uploader(
        "Choose an image file",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
        label_visibility="collapsed",
    )
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    use_sample = st.button("🎲 Use sample", use_container_width=True)

# ── generate sample equation image ───────────────────────────────────────────
def make_sample_image():
    """Generate a synthetic '25 + 37' equation image for demo."""
    canvas = np.ones((80, 400), dtype=np.uint8) * 255
    font   = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "25 + 37", (30, 60), font, 2, 0, 4, cv2.LINE_AA)
    _, buf = cv2.imencode(".png", canvas)
    return buf.tobytes()

if use_sample:
    uploaded = io.BytesIO(make_sample_image())
    uploaded.name = "sample_25+37.png"
    st.info("Using sample equation: **25 + 37**")

# ── processing ────────────────────────────────────────────────────────────────
if uploaded is not None:
    # Read image
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img_color  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_pil    = Image.fromarray(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))

    # Show uploaded image
    st.markdown("#### 🖼️ Your equation")
    st.image(img_pil, use_container_width=True)

    # Save temp file
    tmp_path = "_streamlit_tmp.png"
    cv2.imwrite(tmp_path, img_color)

    # ── Solve button ──────────────────────────────────────────────────────────
    solve_clicked = st.button("⚡  Solve Equation", type="primary",
                               use_container_width=True)

    if solve_clicked:
        with st.spinner("Analysing handwriting…"):
            try:
                # Step 1 — Segment
                segs = get_segments(tmp_path)

                if show_steps:
                    st.markdown("**Step 1 — Segmentation**")
                    st.markdown(f'<div class="step-card"><span class="step-num">1</span>'
                                f'Found <b>{len(segs)}</b> symbol regions via contour detection</div>',
                                unsafe_allow_html=True)

                if not segs:
                    st.markdown('<div class="error-box">❌ No symbols detected. '
                                'Try a clearer image with darker ink.</div>',
                                unsafe_allow_html=True)
                    st.stop()

                # Step 2 — Predict
                predictions = predict_sequence(segs)
                symbols     = [sym for sym, _ in predictions]
                confs       = [c   for _, c   in predictions]

                if show_steps:
                    st.markdown("**Step 2 — CNN Prediction**")
                    st.markdown(f'<div class="step-card"><span class="step-num">2</span>'
                                f'CNN classified {len(symbols)} symbols using trained model</div>',
                                unsafe_allow_html=True)

                # Step 3 — Solve
                result = solve_from_symbols(symbols)

                if show_steps:
                    st.markdown("**Step 3 — Expression Parsing & Solving**")
                    st.markdown(f'<div class="step-card"><span class="step-num">3</span>'
                                f'SymPy evaluated: <code>{result["expression"]}</code></div>',
                                unsafe_allow_html=True)

                # ── Display symbols ───────────────────────────────────────────
                if show_segments:
                    st.markdown("#### 🔍 Recognised symbols")
                    pills = ""
                    for sym, conf in zip(symbols, confs):
                        low = "conf-low" if conf * 100 < conf_threshold else ""
                        pills += (f'<span class="sym-pill {low}">'
                                  f'{sym} <small>{conf*100:.0f}%</small></span>')
                    st.markdown(pills, unsafe_allow_html=True)

                    avg_conf = np.mean(confs) * 100
                    if avg_conf < conf_threshold:
                        st.warning(f"⚠️ Average confidence {avg_conf:.1f}% is low. "
                                   "The image may be unclear — try better lighting or larger writing.")

                # ── Result ────────────────────────────────────────────────────
                if result["error"]:
                    st.markdown(f'<div class="error-box">❌ {result["answer"]}</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-box">
                        <div class="label">Expression detected</div>
                        <div class="expr">{result['expression']}</div>
                        <br>
                        <div class="label">Answer</div>
                        <div class="answer">= {result['answer']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.balloons()

            except Exception as e:
                st.markdown(f'<div class="error-box">❌ Error: {e}</div>',
                            unsafe_allow_html=True)

            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

# ── footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='color:#444466;font-size:0.8rem'>"
    "Handwritten Math Equation Solver · Deep Learning Mini Project · "
    "CNN + OpenCV + SymPy + Streamlit"
    "</center>",
    unsafe_allow_html=True,
)
