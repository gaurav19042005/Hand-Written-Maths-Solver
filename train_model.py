import numpy as np
import os
import cv2
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

IMG_SIZE   = 28
EPOCHS     = 15
BATCH_SIZE = 64
MODEL_PATH = "model.h5"
LABEL_PATH = "labels.npy"

DIGIT_CLASSES    = [str(i) for i in range(10)]          # '0'..'9'
OPERATOR_CLASSES = ['plus', 'minus', 'mul', 'div']      # map to +  -  *  /
ALL_CLASSES      = DIGIT_CLASSES + OPERATOR_CLASSES


def load_mnist_digits():
    """Load MNIST, reshape to (N,28,28,1), normalise."""
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)
    X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype("float32") / 255.0
    return X, y   # labels are already ints 0-9


def load_operator_images():
    """
    Load operator images from  operators/<label>/  folders.
    If the folder doesn't exist we generate a tiny synthetic set so the
    script still runs end-to-end for demo purposes.
    """
    X_ops, y_ops = [], []
    for idx, label in enumerate(OPERATOR_CLASSES):
        class_idx = len(DIGIT_CLASSES) + idx      # 10, 11, 12, 13
        folder = os.path.join("operators", label)

        if os.path.isdir(folder):
            files = [f for f in os.listdir(folder)
                     if f.lower().endswith(('.png','.jpg','.jpeg'))]
            for fname in files:
                img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img.astype("float32") / 255.0
                X_ops.append(img.reshape(IMG_SIZE, IMG_SIZE, 1))
                y_ops.append(class_idx)
        else:
            # ── Synthetic fallback: draw the operator glyph with OpenCV ──────
            print(f"  [warn] operators/{label}/ not found – generating synthetic samples")
            glyphs = {
                'plus':  lambda c: cv2.line(cv2.line(c,(7,14),(21,14),(255,2)),
                                             (14,7),(14,21),(255,2)),
                'minus': lambda c: cv2.line(c,(6,14),(22,14),(255,2)),
                'mul':   lambda c: (cv2.line(cv2.line(c,(6,6),(22,22),(255,2)),
                                             (22,6),(6,22),(255,2))),
                'div':   lambda c: (cv2.circle(cv2.line(cv2.circle(
                                    c,(14,8),2,(255,-1)),(6,14),(22,14),(255,2)),
                                    (14,20),2,(255,-1))),
            }
            for _ in range(500):
                canvas = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
                glyphs[label](canvas)
                # random shift / noise for augmentation
                noise = np.random.randint(0, 30, canvas.shape, dtype=np.uint8)
                canvas = np.clip(canvas.astype(int) + noise, 0, 255).astype(np.uint8)
                img = canvas.astype("float32") / 255.0
                X_ops.append(img.reshape(IMG_SIZE, IMG_SIZE, 1))
                y_ops.append(class_idx)

    if X_ops:
        return np.array(X_ops), np.array(y_ops)
    return np.empty((0, IMG_SIZE, IMG_SIZE, 1)), np.empty((0,), dtype=int)


def build_model(num_classes):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        BatchNormalization(),
        Conv2D(32, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2,2),
        Dropout(0.25),

        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2,2),
        Dropout(0.25),

        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax'),
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def main():
    num_classes = len(ALL_CLASSES)
    print(f"Classes ({num_classes}): {ALL_CLASSES}")

    # Load data
    print("\nLoading MNIST digits...")
    X_digits, y_digits = load_mnist_digits()

    print("Loading operator images...")
    X_ops, y_ops = load_operator_images()

    # Combine
    if len(X_ops):
        X = np.concatenate([X_digits, X_ops], axis=0)
        y = np.concatenate([y_digits, y_ops], axis=0)
    else:
        X, y = X_digits, y_digits
        print("[warn] No operator data – training on digits only")

    print(f"Total samples: {len(X)}")

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y)

    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat   = to_categorical(y_val,   num_classes)

    # Build & train
    model = build_model(num_classes)
    model.summary()

    cb = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
    model.fit(X_train, y_train_cat,
              validation_data=(X_val, y_val_cat),
              epochs=EPOCHS, batch_size=BATCH_SIZE,
              callbacks=[cb])

    # Save
    model.save(MODEL_PATH)
    np.save(LABEL_PATH, np.array(ALL_CLASSES))
    print(f"\nModel saved → {MODEL_PATH}")
    print(f"Labels saved → {LABEL_PATH}")


if __name__ == "__main__":
    main()
