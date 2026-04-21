# Handwritten Math Equation Solver
> Deep Learning Mini Project | CNN + OpenCV + SymPy + Tkinter

---

## Project Structure

```
math_solver/
├── train_model.py   ← Step 1: Train CNN on digits + operators
├── segment.py       ← Step 2: OpenCV contour segmentation
├── predict.py       ← Step 3: CNN inference per symbol
├── solver.py        ← Step 4: Expression parser & evaluator
├── app.py           ← Step 5: Tkinter GUI (draw & solve)
├── requirements.txt ← Python dependencies
└── operators/       ← (optional) your own operator images
    ├── plus/
    ├── minus/
    ├── mul/
    └── div/
```

---

## Setup & Execution

### 1. Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Add your own operator images
Create folders and add 100+ PNG images of handwritten operators:
```
operators/plus/   → images of hand-drawn  +
operators/minus/  → images of hand-drawn  -
operators/mul/    → images of hand-drawn  ×
operators/div/    → images of hand-drawn  ÷
```
If you skip this, the script auto-generates synthetic operator images.

### 4. Train the model
```bash
python train_model.py
```
This creates:
- `model.h5`   — saved CNN weights
- `labels.npy` — class label mapping

Training takes ~3–5 minutes on CPU.

### 5. Run the app
```bash
python app.py
```

---

## How to Use the App

1. **Draw** your equation on the dark canvas using your mouse
   - Example: draw  `2 5 + 3 7`  for 25 + 37
2. Press **Enter** or click **⚡ Solve**
3. The predicted expression and answer appear below
4. Press **C** or click **🗑 Clear** to draw a new equation

---

## Supported Operations

| Symbol | What to draw |
|--------|-------------|
| 0–9    | Any digit   |
| +      | Plus sign   |
| -      | Minus/dash  |
| *      | X cross     |
| /      | Slash       |

### Supported equation types
- Single-line arithmetic:  `3 + 5 =`
- Multi-digit numbers:     `25 * 4`
- Mixed operations:        `10 + 2 * 3`  (BODMAS respected)

---

## Architecture

```
Input Image
    │
    ▼
OpenCV Preprocessing
(grayscale → blur → Otsu threshold)
    │
    ▼
Contour Segmentation
(bounding boxes → 28×28 crops)
    │
    ▼
CNN Model (Keras)
Conv2D → BatchNorm → MaxPool → Dense
    │
    ▼
Symbol Sequence
    │
    ▼
SymPy Expression Parser
    │
    ▼
Answer
```

---

## Tech Stack

| Component       | Library          |
|-----------------|------------------|
| Deep Learning   | TensorFlow/Keras |
| Image Processing| OpenCV           |
| Math Solving    | SymPy            |
| GUI             | Tkinter (built-in)|
| Dataset         | MNIST + synthetic |

---

## Troubleshooting

**`model.h5 not found`**
→ Run `python train_model.py` first.

**Low accuracy on operators**
→ Add real handwritten operator images in `operators/` folder.

**Canvas save fails**
→ Install Pillow: `pip install Pillow`
→ Or install Ghostscript: https://ghostscript.com/releases/

**TensorFlow install issues on Windows**
→ Use Python 3.9–3.11; TF doesn't support Python 3.12 yet.

---

## Sample Results

| Equation drawn | Predicted | Answer |
|----------------|-----------|--------|
| 3 + 5          | 3+5       | 8      |
| 12 * 4         | 12*4      | 48     |
| 100 - 37       | 100-37    | 63     |
| 8 / 2          | 8/2       | 4      |
