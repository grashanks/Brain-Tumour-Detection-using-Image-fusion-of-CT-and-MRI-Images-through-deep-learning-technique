"""
train_on_images.py
------------------
Trains the CNN on the 3 provided tumor images (with heavy augmentation),
then runs enhanced tumor marking on each image.

Usage:
    python train_on_images.py
"""

import os
import numpy as np
import cv2
import tensorflow as tf

from utils.preprocessing import preprocess_image, clahe_enhance, skull_strip
from models.cnn_model import build_custom_cnn, compile_model

TUMOR_IMAGES = [
    "tumor_4.jpg",
    "tumor_image2.jpg",
    "tumor_images.jpg",
]
OUT_DIR = "outputs/marked"
MODEL_PATH = "outputs/best_model.keras"
IMG_SIZE = (224, 224)


# ── 1. Build dataset ──────────────────────────────────────────────────────────

def load_and_preprocess(path):
    img = clahe_enhance(preprocess_image(path))
    img = skull_strip(img)
    return img  # float32 [0,1], shape (224,224)


def augment_image(img):
    """Return a list of augmented variants of a single (H,W) float image."""
    variants = [img]
    h, w = img.shape
    # Flips
    variants.append(cv2.flip(img, 1))
    variants.append(cv2.flip(img, 0))
    variants.append(cv2.flip(img, -1))
    # Rotations
    for angle in [15, -15, 30, -30, 45, -45, 90, 180, 270]:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        variants.append(cv2.warpAffine(img, M, (w, h)))
    # Brightness / contrast
    for alpha, beta in [(1.2, 0.05), (0.8, -0.05), (1.4, 0.1), (0.7, 0.0)]:
        v = np.clip(img * alpha + beta, 0, 1)
        variants.append(v.astype(np.float32))
    # Gaussian noise
    for sigma in [0.02, 0.04]:
        noisy = np.clip(img + np.random.normal(0, sigma, img.shape), 0, 1)
        variants.append(noisy.astype(np.float32))
    # Zoom crop
    for scale in [0.85, 0.75]:
        cy, cx = h // 2, w // 2
        nh, nw = int(h * scale), int(w * scale)
        cropped = img[cy - nh // 2:cy + nh // 2, cx - nw // 2:cx + nw // 2]
        variants.append(cv2.resize(cropped, (w, h)))
    return variants


def build_dataset():
    X, y = [], []
    base = os.path.dirname(os.path.abspath(__file__))

    for fname in TUMOR_IMAGES:
        path = os.path.join(base, fname)
        if not os.path.exists(path):
            print(f"[WARN] Not found: {fname}")
            continue
        img = load_and_preprocess(path)
        for aug in augment_image(img):
            X.append(aug)
            y.append(1.0)  # tumor

    # Synthetic no-tumor samples: uniform noise + dark images
    n_tumor = len(X)
    for _ in range(n_tumor):
        noise = np.random.uniform(0.05, 0.25, IMG_SIZE).astype(np.float32)
        X.append(noise)
        y.append(0.0)

    X = np.expand_dims(np.array(X, dtype=np.float32), axis=-1)  # (N,224,224,1)
    y = np.array(y, dtype=np.float32)

    # Shuffle
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


# ── 2. Train ──────────────────────────────────────────────────────────────────

def train():
    os.makedirs("outputs", exist_ok=True)
    print("[1/3] Building augmented dataset ...")
    X, y = build_dataset()
    print(f"      Total samples: {len(X)}  (tumor={int(y.sum())}, no_tumor={int((1-y).sum())})")

    split = int(len(X) * 0.85)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    print("[2/3] Building + training CNN ...")
    model = build_custom_cnn(input_shape=(224, 224, 1), num_classes=2)
    model = compile_model(model, num_classes=2, learning_rate=5e-4)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH, monitor="val_auc", save_best_only=True, mode="max", verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", patience=15, restore_best_weights=True, mode="max"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1
        ),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=60,
        batch_size=8,
        callbacks=callbacks,
        verbose=1,
    )
    print(f"      Model saved -> {MODEL_PATH}")
    return model


# ── 3. Enhanced tumor marking ─────────────────────────────────────────────────

def gradcam(inp, model):
    last_conv = model.get_layer("last_conv")
    grad_model = tf.keras.Model(inputs=model.input,
                                outputs=[last_conv.output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(inp, training=False)
        tape.watch(conv_out)
        loss = preds[:, 0]
    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    hm = tf.reduce_sum(conv_out[0] * pooled, axis=-1)
    hm = tf.maximum(hm, 0)
    mx = tf.reduce_max(hm)
    if mx > 0:
        hm = hm / mx
    return hm.numpy()


def mark_enhanced(img, heatmap, confidence, grade):
    """
    Enhanced marking:
    - Jet heatmap overlay (full color)
    - Bold red contour (5px) + filled region (40% opacity)
    - Bounding box with corner ticks
    - Confidence + grade banner
    """
    img_sq = np.squeeze(img)
    h, w = img_sq.shape
    img_u8 = np.uint8(np.clip(img_sq * 255, 0, 255))
    canvas = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)

    # ── Jet heatmap overlay ──
    hm = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    hm = np.clip(hm, 0, 1)
    hm_u8 = np.uint8(hm * 255)
    jet = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    canvas = cv2.addWeighted(jet, 0.40, canvas, 0.60, 0)

    # ── Contour detection ──
    hm_blur = cv2.GaussianBlur(hm, (15, 15), 0)
    nonzero = hm_blur[hm_blur > 0.05]
    thresh = float(np.percentile(nonzero, 50)) if len(nonzero) > 0 else 0.35
    binary = np.uint8(hm_blur >= thresh) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Draw ALL significant contours (not just largest)
        sig_contours = [c for c in contours if cv2.contourArea(c) > 100]
        largest = max(sig_contours, key=cv2.contourArea) if sig_contours else max(contours, key=cv2.contourArea)

        # Strong red fill (40% opacity)
        fill = canvas.copy()
        cv2.drawContours(fill, [largest], -1, (0, 0, 255), cv2.FILLED)
        canvas = cv2.addWeighted(fill, 0.40, canvas, 0.60, 0)

        # Bold red contour (5px) + white inner outline (1px)
        cv2.drawContours(canvas, [largest], -1, (255, 255, 255), 2)
        cv2.drawContours(canvas, [largest], -1, (0, 0, 255), 5)

        # Draw secondary contours in orange
        for c in sig_contours:
            if c is not largest and cv2.contourArea(c) > 200:
                cv2.drawContours(canvas, [c], -1, (0, 165, 255), 3)

        # Bounding box with corner ticks
        x, y_box, bw, bh = cv2.boundingRect(largest)
        cv2.rectangle(canvas, (x, y_box), (x + bw, y_box + bh), (0, 0, 255), 2)
        tick = 10
        for px, py in [(x, y_box), (x + bw, y_box), (x, y_box + bh), (x + bw, y_box + bh)]:
            dx = tick if px == x else -tick
            dy = tick if py == y_box else -tick
            cv2.line(canvas, (px, py), (px + dx, py), (255, 255, 0), 3)
            cv2.line(canvas, (px, py), (px, py + dy), (255, 255, 0), 3)

        # Label badge
        label = f"TUMOR  {confidence*100:.1f}%"
        lx, ly = x, max(y_box - 8, 20)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(canvas, (lx - 2, ly - th - 6), (lx + tw + 6, ly + 2), (0, 0, 200), cv2.FILLED)
        cv2.putText(canvas, label, (lx + 2, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # ── Grade banner ──
    banner = np.zeros((40, w, 3), dtype=np.uint8)
    banner[:] = (30, 30, 30)
    cv2.putText(banner, grade, (8, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2, cv2.LINE_AA)
    return np.vstack([canvas, banner])


def run_marking(model):
    os.makedirs(OUT_DIR, exist_ok=True)
    base = os.path.dirname(os.path.abspath(__file__))
    THRESHOLD = 0.30

    for fname in TUMOR_IMAGES:
        path = os.path.join(base, fname)
        if not os.path.exists(path):
            print(f"[SKIP] {fname}")
            continue

        img = load_and_preprocess(path)
        inp = np.expand_dims(img, axis=(0, -1)).astype(np.float32)

        prob = float(model.predict(inp, verbose=0)[0][0])
        confidence = prob
        label = "TUMOR" if prob >= THRESHOLD else "NO TUMOR"
        print(f"{fname:25s}  ->  {label}  (score: {prob:.3f})")

        heatmap = gradcam(inp, model)

        if prob < 0.4:   grade = "Grade 1 — Minimal"
        elif prob < 0.6: grade = "Grade 2 — Moderate"
        elif prob < 0.8: grade = "Grade 3 — High"
        else:            grade = "Grade 4 — Critical"

        marked = mark_enhanced(img, heatmap, confidence, f"{grade}  |  Score: {prob:.3f}")

        stem = os.path.splitext(fname)[0]
        out_path = os.path.join(OUT_DIR, f"{stem}_marked.png")
        cv2.imwrite(out_path, cv2.cvtColor(marked, cv2.COLOR_RGB2BGR))
        print(f"  Saved -> {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = train()
    print("[3/3] Running enhanced tumor marking ...")
    run_marking(model)
    print("Done.")
