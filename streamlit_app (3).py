"""
streamlit_app.py  —  Handwritten Math Equation Solver
Trains model on first run, saves as model.keras (avoids h5 version mismatch).
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io, os, sys, re

st.set_page_config(page_title="Handwritten Math Solver", page_icon="✏️", layout="centered")
sys.path.insert(0, os.path.dirname(__file__))

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.block-container{padding-top:2rem}
.title-box{background:linear-gradient(135deg,#1a1a2e,#16213e);border:1px solid #4ecca3;
  border-radius:12px;padding:1.4rem 2rem;margin-bottom:1.5rem;text-align:center}
.title-box h1{color:#4ecca3;margin:0;font-size:2rem}
.title-box p{color:#aaa;margin:.4rem 0 0;font-size:.95rem}
.result-box{background:#1a1a2e;border-left:4px solid #4ecca3;border-radius:8px;
  padding:1.2rem 1.6rem;margin-top:1.2rem}
.result-box .label{color:#888;font-size:.85rem;margin-bottom:4px}
.result-box .expr{color:#e2e2e2;font-size:1.1rem;font-family:monospace}
.result-box .answer{color:#4ecca3;font-size:2.2rem;font-weight:700}
.error-box{background:#2e1a1a;border-left:4px solid #e94560;border-radius:8px;
  padding:1rem 1.4rem;color:#e94560;margin-top:1rem}
.sym-pill{display:inline-block;background:#16213e;border:1px solid #4ecca3;
  border-radius:20px;padding:3px 12px;margin:3px;color:#4ecca3;font-family:monospace}
.conf-low{border-color:#e94560!important;color:#e94560!important}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="title-box">
  <h1>✏️ Handwritten Math Equation Solver</h1>
  <p>Upload a photo of your handwritten equation — get the answer instantly</p>
</div>""", unsafe_allow_html=True)

# ── constants ─────────────────────────────────────────────────────────────────
IMG_SIZE   = 28
MODEL_PATH = "model.keras"          # NEW: .keras format avoids h5 issues
LABEL_PATH = "labels.npy"
DIGIT_CLASSES    = [str(i) for i in range(10)]
OPERATOR_CLASSES = ['plus','minus','mul','div']
ALL_CLASSES      = DIGIT_CLASSES + OPERATOR_CLASSES
OP_MAP           = {'plus':'+','minus':'-','mul':'*','div':'/'}

# ── build model architecture (shared by train + load) ─────────────────────────
def build_model():
    from tensorflow import keras
    return keras.Sequential([
        keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        keras.layers.Conv2D(32,(3,3),activation='relu',padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(32,(3,3),activation='relu',padding='same'),
        keras.layers.MaxPooling2D(2,2),
        keras.layers.Dropout(0.25),
        keras.layers.Conv2D(64,(3,3),activation='relu',padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(64,(3,3),activation='relu',padding='same'),
        keras.layers.MaxPooling2D(2,2),
        keras.layers.Dropout(0.25),
        keras.layers.Flatten(),
        keras.layers.Dense(256,activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(len(ALL_CLASSES),activation='softmax'),
    ])

# ── training ──────────────────────────────────────────────────────────────────
def train_and_save():
    from tensorflow import keras
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.model_selection import train_test_split

    # MNIST digits
    (Xt, yt),(Xv, yv) = mnist.load_data()
    X = np.concatenate([Xt,Xv]).reshape(-1,IMG_SIZE,IMG_SIZE,1).astype("float32")/255.0
    y = np.concatenate([yt,yv])

    # Synthetic operators
    def make_op(label, n=400):
        imgs, lbls = [], []
        idx = len(DIGIT_CLASSES) + OPERATOR_CLASSES.index(label)
        for _ in range(n):
            c = np.zeros((IMG_SIZE,IMG_SIZE),dtype=np.uint8)
            if label=='plus':
                cv2.line(c,(7,14),(21,14),255,2); cv2.line(c,(14,7),(14,21),255,2)
            elif label=='minus':
                cv2.line(c,(6,14),(22,14),255,2)
            elif label=='mul':
                cv2.line(c,(6,6),(22,22),255,2); cv2.line(c,(22,6),(6,22),255,2)
            elif label=='div':
                cv2.line(c,(6,14),(22,14),255,2)
                cv2.circle(c,(14,8),2,255,-1); cv2.circle(c,(14,20),2,255,-1)
            noise = np.random.randint(0,25,c.shape,dtype=np.uint8)
            c = np.clip(c.astype(int)+noise,0,255).astype(np.uint8)
            imgs.append(c.reshape(IMG_SIZE,IMG_SIZE,1).astype("float32")/255.0)
            lbls.append(idx)
        return np.array(imgs), np.array(lbls)

    Xo = np.concatenate([make_op(op)[0] for op in OPERATOR_CLASSES])
    yo = np.concatenate([make_op(op)[1] for op in OPERATOR_CLASSES])

    X_all = np.concatenate([X,Xo])
    y_all = np.concatenate([y,yo])

    X_tr,X_val,y_tr,y_val = train_test_split(
        X_all, y_all, test_size=0.1, random_state=42, stratify=y_all)

    nc = len(ALL_CLASSES)
    model = build_model()
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    cb = EarlyStopping(monitor='val_accuracy',patience=3,restore_best_weights=True)
    model.fit(X_tr, to_categorical(y_tr,nc),
              validation_data=(X_val,to_categorical(y_val,nc)),
              epochs=15, batch_size=64, callbacks=[cb], verbose=0)

    model.save(MODEL_PATH)                        # saves as .keras format
    np.save(LABEL_PATH, np.array(ALL_CLASSES))

# ── auto-train if model missing ───────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    with st.status("🧠 Training model — first run only, takes ~3 min…", expanded=True) as s:
        st.write("Downloading MNIST dataset…")
        st.write("Building CNN architecture…")
        st.write("Training — please wait…")
        try:
            train_and_save()
            s.update(label="✅ Model ready!", state="complete")
            st.rerun()
        except Exception as e:
            s.update(label=f"❌ Training failed: {e}", state="error")
            st.stop()

# ── load model (cached across reruns) ────────────────────────────────────────
@st.cache_resource
def load_assets():
    from tensorflow import keras
    m = build_model()                            # build fresh architecture
    tmp = keras.models.load_model(MODEL_PATH)   # load saved weights + config
    m.set_weights(tmp.get_weights())            # transfer weights safely
    lbl = np.load(LABEL_PATH, allow_pickle=True)
    return m, lbl

model, labels = load_assets()

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    conf_threshold = st.slider("Confidence threshold (%)", 10, 95, 50)
    show_segments  = st.checkbox("Show segmented symbols", True)
    show_steps     = st.checkbox("Show processing steps", False)
    st.markdown("---")
    st.markdown("### 📋 Supported Symbols")
    st.markdown("**Digits:** 0 1 2 3 4 5 6 7 8 9")
    st.markdown("**Operators:** `+`  `-`  `×`  `÷`")
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("- Write **large & clear**\n- Leave **space** between symbols\n- Use **dark ink on white** paper\n- Good **lighting** for phone photos")

# ── segment image into individual symbol crops ────────────────────────────────
def get_segments(img_color):
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(5,5),0)
    _,binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    dilated = cv2.dilate(binary,kernel,iterations=1)
    contours,_ = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    h_img,w_img = binary.shape
    segs = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if w<5 or h<5: continue
        x1,y1=max(0,x-4),max(0,y-4)
        x2,y2=min(w_img,x+w+4),min(h_img,y+h+4)
        crop=binary[y1:y2,x1:x2]
        sz=max(crop.shape)
        sq=np.zeros((sz,sz),dtype=np.uint8)
        dy,dx=(sz-crop.shape[0])//2,(sz-crop.shape[1])//2
        sq[dy:dy+crop.shape[0],dx:dx+crop.shape[1]]=crop
        resized=cv2.resize(sq,(IMG_SIZE,IMG_SIZE)).astype("float32")/255.0
        segs.append((x,resized))
    segs.sort(key=lambda s:s[0])
    return segs

# ── predict a single 28x28 symbol ────────────────────────────────────────────
def predict_symbol(img_28x28):
    probs = model.predict(img_28x28.reshape(1,IMG_SIZE,IMG_SIZE,1), verbose=0)[0]
    idx   = int(np.argmax(probs))
    lbl   = str(labels[idx])
    return OP_MAP.get(lbl,lbl), float(probs[idx])

# ── parse & solve expression ──────────────────────────────────────────────────
def solve(symbols):
    from sympy import sympify
    syms  = [s for s in symbols if s not in ('=','')]
    expr  = "".join(syms)
    clean = re.sub(r'[^0-9+\-*/().\s]','',expr).strip()
    if not clean: return expr,"Error: empty expression",True
    try:
        r   = sympify(clean)
        ans = str(int(r)) if r==int(r) else str(float(r))
        return expr,ans,False
    except Exception as e:
        return expr,f"Error: {e}",True

# ── sample image ──────────────────────────────────────────────────────────────
def make_sample():
    c=np.ones((80,400),dtype=np.uint8)*255
    cv2.putText(c,"25 + 37",(30,60),cv2.FONT_HERSHEY_SIMPLEX,2,0,4,cv2.LINE_AA)
    _,buf=cv2.imencode(".png",c)
    return buf.tobytes()

# ── upload UI ─────────────────────────────────────────────────────────────────
st.markdown("#### 📤 Upload your equation image")
c1,c2 = st.columns([3,1])
with c1:
    uploaded = st.file_uploader("img", type=["png","jpg","jpeg","bmp","webp"],
                                 label_visibility="collapsed")
with c2:
    st.markdown("<br>",unsafe_allow_html=True)
    if st.button("🎲 Use sample", use_container_width=True):
        uploaded = io.BytesIO(make_sample())
        st.info("Sample: **25 + 37**")

# ── solve ─────────────────────────────────────────────────────────────────────
if uploaded is not None:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img_color  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_pil    = Image.fromarray(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))

    st.markdown("#### 🖼️ Your equation")
    st.image(img_pil, use_container_width=True)

    if st.button("⚡  Solve Equation", type="primary", use_container_width=True):
        with st.spinner("Analysing…"):
            try:
                segs = get_segments(img_color)
                if show_steps:
                    st.markdown(f"**Step 1 — Segmentation:** `{len(segs)}` symbol(s) found")
                if not segs:
                    st.markdown('<div class="error-box">❌ No symbols detected — try a clearer image.</div>',
                                unsafe_allow_html=True)
                    st.stop()

                predictions = [predict_symbol(img) for _,img in segs]
                symbols = [s for s,_ in predictions]
                confs   = [c for _,c in predictions]
                if show_steps:
                    st.markdown(f"**Step 2 — CNN:** `{'  '.join(symbols)}`")

                expr,answer,has_error = solve(symbols)
                if show_steps:
                    st.markdown(f"**Step 3 — Solver:** `{expr}` → `{answer}`")

                if show_segments:
                    st.markdown("#### 🔍 Recognised symbols")
                    pills=""
                    for sym,conf in zip(symbols,confs):
                        low="conf-low" if conf*100<conf_threshold else ""
                        pills+=f'<span class="sym-pill {low}">{sym} <small>{conf*100:.0f}%</small></span>'
                    st.markdown(pills, unsafe_allow_html=True)
                    avg=np.mean(confs)*100
                    if avg<conf_threshold:
                        st.warning(f"⚠️ Avg confidence {avg:.1f}% is low — try a clearer image.")

                if has_error:
                    st.markdown(f'<div class="error-box">❌ {answer}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div class="result-box">
                        <div class="label">Expression detected</div>
                        <div class="expr">{expr}</div><br>
                        <div class="label">Answer</div>
                        <div class="answer">= {answer}</div>
                    </div>""", unsafe_allow_html=True)
                    st.balloons()

            except Exception as e:
                st.markdown(f'<div class="error-box">❌ Error: {e}</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("<center style='color:#444466;font-size:.8rem'>Handwritten Math Equation Solver · CNN + OpenCV + SymPy + Streamlit</center>",
            unsafe_allow_html=True)
