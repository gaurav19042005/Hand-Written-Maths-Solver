"""
predict.py
Loads trained CNN + class labels and predicts symbols from segmented images.
"""

import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE   = 28
MODEL_PATH = "model.h5"
LABEL_PATH = "labels.npy"

# Operator label → math character mapping
OP_MAP = {
    'plus':  '+',
    'minus': '-',
    'mul':   '*',
    'div':   '/',
}

_model  = None
_labels = None


def _load():
    global _model, _labels
    if _model is None:
        _model  = load_model(MODEL_PATH)
        _labels = np.load(LABEL_PATH, allow_pickle=True)


def predict_symbol(img_28x28: np.ndarray) -> tuple[str, float]:
    """
    img_28x28 : float32 array of shape (28,28) normalised 0-1
    Returns (label_string, confidence_float)
    """
    _load()
    x = img_28x28.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    probs = _model.predict(x, verbose=0)[0]
    idx   = int(np.argmax(probs))
    label = str(_labels[idx])
    conf  = float(probs[idx])
    # Map operator names to symbols
    symbol = OP_MAP.get(label, label)
    return symbol, conf


def predict_sequence(segments: list) -> list[tuple[str, float]]:
    """
    segments : list of (x_pos, img_28x28) from segment.py
    Returns list of (symbol, confidence) in order.
    """
    return [predict_symbol(img) for _, img in segments]


if __name__ == "__main__":
    import sys
    from segment import get_segments

    path = sys.argv[1] if len(sys.argv) > 1 else "test_eq.png"
    segs = get_segments(path)
    preds = predict_sequence(segs)
    print("Predicted symbols:", " ".join(s for s, _ in preds))
    for sym, conf in preds:
        print(f"  {sym}  ({conf*100:.1f}%)")
