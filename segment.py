"""
segment.py
Given a grayscale image of a handwritten equation, returns a list of
(x, cropped_28x28_image) tuples sorted left-to-right.
"""

import cv2
import numpy as np

IMG_SIZE = 28
PAD      = 4          # padding around each bounding box


def preprocess(image_path: str) -> np.ndarray:
    """Load & binarise image. Works on white-on-black or black-on-white."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # Resize if very large
    h, w = img.shape
    if max(h, w) > 800:
        scale = 800 / max(h, w)
        img   = cv2.resize(img, (int(w*scale), int(h*scale)))

    # Gaussian blur to reduce noise
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Otsu threshold
    _, binary = cv2.threshold(img, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def get_segments(image_path: str, debug: bool = False):
    """
    Returns list of (x_position, 28x28 float32 array) sorted left-to-right.
    """
    binary = preprocess(image_path)

    # Dilate slightly to merge strokes of the same symbol
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated,
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    segments = []
    h_img, w_img = binary.shape

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Skip tiny noise
        if w < 5 or h < 5:
            continue

        # Crop from the original binary (not dilated)
        x1 = max(0, x - PAD)
        y1 = max(0, y - PAD)
        x2 = min(w_img, x + w + PAD)
        y2 = min(h_img, y + h + PAD)
        crop = binary[y1:y2, x1:x2]

        # Make square by padding
        sz = max(crop.shape)
        sq = np.zeros((sz, sz), dtype=np.uint8)
        dy = (sz - crop.shape[0]) // 2
        dx = (sz - crop.shape[1]) // 2
        sq[dy:dy+crop.shape[0], dx:dx+crop.shape[1]] = crop

        # Resize to 28×28 and normalise
        resized = cv2.resize(sq, (IMG_SIZE, IMG_SIZE))
        norm    = resized.astype("float32") / 255.0

        segments.append((x, norm))

        if debug:
            cv2.imshow(f"seg_x{x}", resized)

    # Sort left-to-right
    segments.sort(key=lambda s: s[0])

    if debug:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return segments


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "test_eq.png"
    segs = get_segments(path, debug=True)
    print(f"Found {len(segs)} segments at x positions: {[s[0] for s in segs]}")
