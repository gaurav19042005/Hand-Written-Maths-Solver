"""
streamlit_app.py  —  Handwritten Math Equation Solver
Auto-trains the model on first run if model.h5 is missing.
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import os
import sys

st.set_page_config(
    page_title="Handwritten Math Solver",
    page_icon="✏️",
    layout="centered",
)

sys.path.insert(0, os.path.dirname(__file__))

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
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
.result-box .label  { color: #888888; font-size: 0.85rem; margin-bottom: 4px; }
.result-box .expr   { color: #e2e2e2; font-size: 1.1rem; font-family: monospace; }
.result-box .answer { color: #4ecca3; font-size: 2.2rem; font-weight: 700; }
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
.conf-low { border-color: #e94560 !important; color: #e94560 !important; }
</style>
""", unsafe_allow_html=True)

# ── header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-box">
  <h1>✏️ Handwritten Math Equation Solver</h1>
  <p>Upload a photo of your handwritten equation — get the answer instantly</p>
</div>
""", unsafe_allow_html=True)

# ── auto-train if model missing ───────────────────────────────────────────────
MODEL_PATH = "model.h5"
LABEL_PATH = "labels.npy"

def train_model_on_cloud():
    """Train and save the model. Called automatically if model.h5 is absent."""
    import numpy as np
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten,
                                         Dense, Dropout, BatchNormalization)
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.model_selection import train_test_split

    IMG_SIZE = 28
    DIGIT_CLASSES    = [str(i) for i in range(10)]
    OPERATOR_CLASSES = ['plus', 'minus', 'mul', 'div']
    ALL_CLASSES      = DIGIT_CLASSES + OPERATOR_CLASSES
    num_classes      = len(ALL_CLASSES)

    # Load MNIST
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X = np.concatenate([X_train, X_test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype("float32") / 255.0
    y = np.concatenate([y_train, y_test])

    # Synthetic operators
    def make_op(label, n=400):
        imgs, lbls = [], []
        idx = len(DIGIT_CLASSES) + OPERATOR_CLASSES.index(label)
        for _ in range(n):
            c = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
            if label == 'plus':
                cv2.line(c, (7,14),(21,14),(255,2)); cv2.line(c,(14,7),(14,21),(255,2))
            elif label == 'minus':
                cv2.line(c, (6,14),(22,14),(255,2))
            elif label == 'mul':
                cv2.line(c, (6,6),(22,22),(255,2)); cv2.line(c,(22,6),(6,22),(255,2))
            elif label == 'div':
                cv2.line(c,(6,14),(22,14),(255,2))
                cv2.circle(c,(14,8),2,(255,-1)); cv2.circle(c,(14,20),2,(255,-1))
            noise = np.random.randint(0,30,c.shape,dtype=np.uint8)
            c = np.clip(c.astype(int)+noise,0,255).astype(np.uint8)
            imgs.append(c.reshape(IMG_SIZE,IMG_SIZE,1).astype("float32")/255.0)
            lbls.append(idx)
        return np.array(imgs), np.array(lbls)

    Xo, yo = [], []
    for op in OPERATOR_CLASSES:
        xi, yi = make_op(op)
        Xo.append(xi); yo.append(yi)
    Xo = np.concatenate(Xo); yo = np.concatenate(yo)

    X_all = np.concatenate([X, Xo])
    y_all = np.concatenate([y, yo])

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_all, y_all, test_size=0.1, random_state=42, stratify=y_all)

    y_tr_c  = to_categorical(y_tr,  num_classes)
    y_val_c = to_categorical(y_val, num_classes)

    model = Sequential([
        Conv2D(32,(3,3),activation='relu',padding='same',input_shape=(IMG_SIZE,IMG_SIZE,1)),
        BatchNormalization(),
        Conv2D(32,(3,3),activation='relu',padding='same'),
        MaxPooling2D(2,2), Dropout(0.25),
        Conv2D(64,(3,3),activation='relu',padding='same'),
        BatchNormalization(),
        Conv2D(64,(3,3),activation='relu',padding='same'),
        MaxPooling2D(2,2), Dropout(0.25),
        Flatten(),
        Dense(256,activation='relu'), BatchNormalization(), Dropout(0.5),
        Dense(num_classes,activation='softmax'),
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    cb = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
    model.fit(X_tr, y_tr_c, validation_data=(X_val,y_val_c),
              epochs=15, batch_size=64, callbacks=[cb], verbose=0)

    model.save(MODEL_PATH)
    np.save(LABEL_PATH, np.array(ALL_CLASSES))
    return model

if not os.path.exists(MODEL_PATH):
    with st.status("🧠 Training model for the first time (takes ~3 min)...", expanded=True) as status:
        st.write("Loading MNIST digit dataset...")
        st.write("Generating synthetic operator samples...")
        st.write("Training CNN — please wait...")
        try:
            train_model_on_cloud()
            status.update(label="✅ Model trained and ready!", state="complete")
            st.rerun()
        except Exception as e:
            status.update(label=f"❌ Training failed: {e}", state="error")
            st.stop()

# ── load model (cached) ───────────────────────────────────────────────────────
@st.cache_resource
def load_model_cached():
    from tensorflow.keras.models import load_model as keras_load
    model  = keras_load(MODEL_PATH)
    labels = np.load(LABEL_PATH, allow_pickle=True)
    return model, labels

model, labels = load_model_cached()

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    conf_threshold = st.slider("Confidence threshold (%)", 10, 95, 50)
    show_segments  = st.checkbox("Show segmented symbols", value=True)
    show_steps     = st.checkbox("Show processing steps",  value=False)
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

# ── helper: predict symbols ───────────────────────────────────────────────────
OP_MAP = {'plus':'+','minus':'-','mul':'*','div':'/'}
IMG_SIZE = 28

def predict_symbol(img_28x28):
    x = img_28x28.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    probs = model.predict(x, verbose=0)[0]
    idx   = int(np.argmax(probs))
    lbl   = str(labels[idx])
    return OP_MAP.get(lbl, lbl), float(probs[idx])

def get_segments(img_color):
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(5,5),0)
    _, binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    dilated = cv2.dilate(binary,kernel,iterations=1)
    contours,_ = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    h_img, w_img = binary.shape
    segs = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if w<5 or h<5: continue
        x1,y1 = max(0,x-4), max(0,y-4)
        x2,y2 = min(w_img,x+w+4), min(h_img,y+h+4)
        crop = binary[y1:y2,x1:x2]
        sz = max(crop.shape)
        sq = np.zeros((sz,sz),dtype=np.uint8)
        dy,dx = (sz-crop.shape[0])//2,(sz-crop.shape[1])//2
        sq[dy:dy+crop.shape[0],dx:dx+crop.shape[1]] = crop
        resized = cv2.resize(sq,(IMG_SIZE,IMG_SIZE)).astype("float32")/255.0
        segs.append((x, resized))
    segs.sort(key=lambda s:s[0])
    return segs

def solve(symbols):
    import re
    from sympy import sympify
    syms = [s for s in symbols if s not in ('=','')]
    expr = "".join(syms)
    clean = re.sub(r'[^0-9+\-*/().\s]','',expr).strip()
    if not clean: return expr, "Error: empty expression", True
    try:
        result = sympify(clean)
        ans = str(int(result)) if result==int(result) else str(float(result))
        return expr, ans, False
    except Exception as e:
        return expr, f"Error: {e}", True

# ── upload section ────────────────────────────────────────────────────────────
st.markdown("#### 📤 Upload your equation image")
col1, col2 = st.columns([3,1])
with col1:
    uploaded = st.file_uploader("image", type=["png","jpg","jpeg","bmp","webp"],
                                 label_visibility="collapsed")
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    use_sample = st.button("🎲 Use sample", use_container_width=True)

def make_sample():
    canvas = np.ones((80,400),dtype=np.uint8)*255
    cv2.putText(canvas,"25 + 37",(30,60),cv2.FONT_HERSHEY_SIMPLEX,2,0,4,cv2.LINE_AA)
    _,buf = cv2.imencode(".png",canvas)
    return buf.tobytes()

if use_sample:
    uploaded = io.BytesIO(make_sample())
    st.info("Using sample equation: **25 + 37**")

# ── main processing ───────────────────────────────────────────────────────────
if uploaded is not None:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img_color  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_pil    = Image.fromarray(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))

    st.markdown("#### 🖼️ Your equation")
    st.image(img_pil, use_container_width=True)

    if st.button("⚡  Solve Equation", type="primary", use_container_width=True):
        with st.spinner("Analysing handwriting…"):
            try:
                segs = get_segments(img_color)

                if show_steps:
                    st.markdown(f"**Step 1 — Segmentation:** found `{len(segs)}` symbol regions")

                if not segs:
                    st.markdown('<div class="error-box">❌ No symbols detected. Try a clearer image.</div>',
                                unsafe_allow_html=True)
                    st.stop()

                predictions = [predict_symbol(img) for _,img in segs]
                symbols = [s for s,_ in predictions]
                confs   = [c for _,c in predictions]

                if show_steps:
                    st.markdown(f"**Step 2 — CNN Prediction:** `{'  '.join(symbols)}`")

                expr, answer, has_error = solve(symbols)

                if show_steps:
                    st.markdown(f"**Step 3 — Solver:** `{expr}` → `{answer}`")

                if show_segments:
                    st.markdown("#### 🔍 Recognised symbols")
                    pills = ""
                    for sym,conf in zip(symbols,confs):
                        low = "conf-low" if conf*100 < conf_threshold else ""
                        pills += f'<span class="sym-pill {low}">{sym} <small>{conf*100:.0f}%</small></span>'
                    st.markdown(pills, unsafe_allow_html=True)
                    avg = np.mean(confs)*100
                    if avg < conf_threshold:
                        st.warning(f"⚠️ Average confidence {avg:.1f}% is low — try a clearer image.")

                if has_error:
                    st.markdown(f'<div class="error-box">❌ {answer}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-box">
                        <div class="label">Expression detected</div>
                        <div class="expr">{expr}</div>
                        <br>
                        <div class="label">Answer</div>
                        <div class="answer">= {answer}</div>
                    </div>""", unsafe_allow_html=True)
                    st.balloons()

            except Exception as e:
                st.markdown(f'<div class="error-box">❌ Error: {e}</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<center style='color:#444466;font-size:0.8rem'>"
    "Handwritten Math Equation Solver · Deep Learning Mini Project · "
    "CNN + OpenCV + SymPy + Streamlit"
    "</center>", unsafe_allow_html=True)
