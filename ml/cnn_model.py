"""
MediSense AI v2 — CNN Image Diagnosis Module
=============================================
Full pipeline:
  - MobileNetV2 transfer learning backbone (frozen → fine-tune)
  - Data augmentation (rotation, flip, zoom, brightness)
  - Train / validation split (80/20)
  - Callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
  - Inference: preprocess bytes → predict → human-readable report
  - Graceful degradation when TF/weights not available

Recommended datasets:
  - Chest X-ray (Pneumonia): https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
  - Skin Lesion (ISIC 2019): https://challenge.isic-archive.com/data
  - Retinal OCT:             https://www.kaggle.com/datasets/paultimothymooney/kermany2018

Training usage:
  python3 ml/cnn_model.py --train --dataset ./chest_xray --epochs 25

Inference usage (via Flask /api/predict/image):
  Instantiate CNNPredictor once at app startup; call predictor.predict(image_bytes)
"""

import os, json, io, argparse
import numpy as np

# ── Optional heavy imports ─────────────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, applications, callbacks, optimizers
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# ── Constants ──────────────────────────────────────────────────────────────────
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
BASE       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "models", "cnn_model.h5")
META_PATH  = os.path.join(BASE, "models", "cnn_classes.json")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Architecture
# ─────────────────────────────────────────────────────────────────────────────
def build_model(num_classes: int, use_transfer: bool = True) -> "tf.keras.Model":
    """
    Build the CNN.

    Transfer-learning mode (recommended):
      • Frozen MobileNetV2 pretrained on ImageNet as feature extractor
      • Custom classification head (Dense → Dropout → Dense → Dropout → Softmax)
      • Phase-1 trains only the head; Phase-2 fine-tunes the top-30 MobileNetV2 layers

    Scratch mode:
      • Four Conv2D blocks with BatchNorm + MaxPool
      • GlobalAveragePooling → Dense head
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is required. Install: pip install tensorflow-cpu")

    if use_transfer:
        base = applications.MobileNetV2(
            input_shape=(*IMG_SIZE, 3),
            include_top=False,
            weights="imagenet",
        )
        base.trainable = False   # freeze for Phase-1

        inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
        x = applications.mobilenet_v2.preprocess_input(inputs)  # [-1, 1] normalisation
        x = base(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(num_classes, activation="softmax")(x)

        model = tf.keras.Model(inputs, outputs)

    else:
        # Custom CNN from scratch
        model = models.Sequential([
            tf.keras.Input(shape=(*IMG_SIZE, 3)),
            # Block 1
            layers.Conv2D(32, 3, padding="same"), layers.BatchNormalization(), layers.Activation("relu"),
            layers.Conv2D(32, 3, padding="same"), layers.BatchNormalization(), layers.Activation("relu"),
            layers.MaxPooling2D(2), layers.Dropout(0.25),
            # Block 2
            layers.Conv2D(64, 3, padding="same"), layers.BatchNormalization(), layers.Activation("relu"),
            layers.Conv2D(64, 3, padding="same"), layers.BatchNormalization(), layers.Activation("relu"),
            layers.MaxPooling2D(2), layers.Dropout(0.25),
            # Block 3
            layers.Conv2D(128, 3, padding="same"), layers.BatchNormalization(), layers.Activation("relu"),
            layers.Conv2D(128, 3, padding="same"), layers.BatchNormalization(), layers.Activation("relu"),
            layers.MaxPooling2D(2), layers.Dropout(0.30),
            # Block 4
            layers.Conv2D(256, 3, padding="same"), layers.BatchNormalization(), layers.Activation("relu"),
            layers.GlobalAveragePooling2D(),
            # Head
            layers.Dense(512, activation="relu"), layers.Dropout(0.50),
            layers.Dense(256, activation="relu"), layers.Dropout(0.30),
            layers.Dense(num_classes, activation="softmax"),
        ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Training pipeline
# ─────────────────────────────────────────────────────────────────────────────
def train(
    dataset_dir: str,
    epochs_phase1: int = 15,
    epochs_phase2: int = 10,
    use_transfer:  bool = True,
    img_size: tuple = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
):
    """
    Two-phase training:
      Phase 1 — train only classification head (frozen backbone).
      Phase 2 — unfreeze top-30 MobileNetV2 layers and fine-tune at lower LR.

    Args:
        dataset_dir   : Path to root folder structured as class_name/image.jpg
        epochs_phase1 : Epochs for Phase 1 (head training)
        epochs_phase2 : Epochs for Phase 2 (fine-tuning); skipped if use_transfer=False
        use_transfer  : Whether to use MobileNetV2 backbone
        img_size      : (H, W) input dimensions
        batch_size    : Mini-batch size

    Returns:
        (model, class_names, history)
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is required for training.")

    os.makedirs(os.path.join(BASE, "models"), exist_ok=True)

    # ── Data generators ────────────────────────────────────────────────────────
    train_aug = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.20,
        rotation_range=15,
        width_shift_range=0.10,
        height_shift_range=0.10,
        shear_range=0.08,
        zoom_range=0.12,
        brightness_range=[0.85, 1.15],
        horizontal_flip=True,
        fill_mode="nearest",
    )
    val_aug = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.20)

    train_gen = train_aug.flow_from_directory(
        dataset_dir, target_size=img_size, batch_size=batch_size,
        class_mode="categorical", subset="training", shuffle=True,
    )
    val_gen = val_aug.flow_from_directory(
        dataset_dir, target_size=img_size, batch_size=batch_size,
        class_mode="categorical", subset="validation", shuffle=False,
    )

    class_names  = list(train_gen.class_indices.keys())
    num_classes  = len(class_names)
    print(f"\n  Classes ({num_classes}): {class_names}")
    print(f"  Train samples: {train_gen.samples}  |  Val samples: {val_gen.samples}\n")

    model = build_model(num_classes, use_transfer)

    # ── Shared callbacks ───────────────────────────────────────────────────────
    cbs = [
        callbacks.ModelCheckpoint(
            MODEL_PATH, save_best_only=True,
            monitor="val_accuracy", verbose=1,
        ),
        callbacks.EarlyStopping(
            patience=6, restore_best_weights=True,
            monitor="val_accuracy", verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            factor=0.4, patience=3, min_lr=1e-7,
            monitor="val_loss", verbose=1,
        ),
    ]

    # ── Phase 1: train classification head ────────────────────────────────────
    print("=" * 50)
    print("  Phase 1: Training classification head")
    print("=" * 50)
    h1 = model.fit(
        train_gen, epochs=epochs_phase1,
        validation_data=val_gen, callbacks=cbs,
    )

    # ── Phase 2: fine-tune top MobileNetV2 layers (transfer learning only) ────
    h2 = None
    if use_transfer and epochs_phase2 > 0:
        print("\n" + "=" * 50)
        print("  Phase 2: Fine-tuning top-30 MobileNetV2 layers")
        print("=" * 50)
        base_model = model.layers[3]          # MobileNetV2 is 4th layer in functional model
        base_model.trainable = True
        for layer in base_model.layers[:-30]:
            layer.trainable = False

        model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-5),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        h2 = model.fit(
            train_gen, epochs=epochs_phase2,
            validation_data=val_gen, callbacks=cbs,
        )

    # ── Save class names ───────────────────────────────────────────────────────
    with open(META_PATH, "w") as f:
        json.dump(class_names, f, indent=2)

    best_acc = max(h1.history["val_accuracy"] + (h2.history["val_accuracy"] if h2 else []))
    print(f"\n✅ Training complete. Best val_accuracy: {best_acc:.4f} ({best_acc*100:.1f}%)")
    print(f"   Model saved → {MODEL_PATH}")
    print(f"   Classes     → {META_PATH}")
    return model, class_names, h1


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_bytes(image_bytes: bytes, img_size: tuple = IMG_SIZE) -> np.ndarray:
    """
    Convert raw image bytes → normalised numpy array ready for model inference.
    Handles JPEG, PNG, and most common medical image formats that PIL supports.
    """
    if not PIL_AVAILABLE:
        raise RuntimeError("Pillow is required. Install: pip install Pillow")
    img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(img_size, PILImage.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)   # shape: (1, H, W, 3)


def preprocess_path(img_path: str, img_size: tuple = IMG_SIZE) -> np.ndarray:
    """Load an image from disk and preprocess it."""
    with open(img_path, "rb") as f:
        return preprocess_bytes(f.read(), img_size)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Inference class
# ─────────────────────────────────────────────────────────────────────────────

# Human-readable explanations for common medical image findings
FINDING_DESCRIPTIONS = {
    "NORMAL":    "No significant abnormalities detected. Lung fields appear clear.",
    "PNEUMONIA": "Findings consistent with pneumonia. Opacification visible in lung field(s). "
                 "Clinical correlation and radiologist review recommended.",
    "COVID19":   "Imaging findings may be consistent with COVID-19 pneumonia (bilateral ground-glass opacities). "
                 "PCR testing and specialist review required.",
    "TUBERCULOSIS": "Findings may suggest pulmonary tuberculosis. Upper-lobe involvement noted. "
                    "Sputum culture and specialist referral advised.",
    "BENIGN":    "Lesion appears consistent with benign characteristics. Follow-up recommended.",
    "MALIGNANT": "Lesion shows features that may warrant further investigation. "
                 "Biopsy and specialist review strongly recommended.",
}


class CNNPredictor:
    """
    Loads a trained Keras model and runs image classification.
    Designed to be instantiated once at app startup (lazy-loaded in Flask factory).
    """

    def __init__(self, model_path: str = MODEL_PATH, class_names: list = None):
        self.model_path  = model_path
        self.class_names = class_names or self._load_class_names()
        self.model       = None
        self.loaded      = False
        self._try_load()

    def _load_class_names(self) -> list:
        if os.path.exists(META_PATH):
            try:
                with open(META_PATH) as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def _try_load(self):
        if not TF_AVAILABLE:
            print("[CNNPredictor] TensorFlow not installed — image inference unavailable.")
            return
        if not os.path.exists(self.model_path):
            print(f"[CNNPredictor] Model file not found: {self.model_path}")
            print("  Train: python3 ml/cnn_model.py --train --dataset ./your_dataset")
            return
        try:
            self.model  = tf.keras.models.load_model(self.model_path)
            self.loaded = True
            print(f"[CNNPredictor] ✅ Loaded — {len(self.class_names)} classes")
        except Exception as e:
            print(f"[CNNPredictor] ⚠️  Load failed: {e}")

    # ── Public API ─────────────────────────────────────────────────────────────
    def predict(self, image_bytes: bytes, top_n: int = 3) -> dict:
        """
        Run inference on raw image bytes.

        Returns:
            {
              "loaded":      bool,
              "predictions": [{"label": str, "confidence": float}, ...],
              "top_finding": str,
              "report":      str,
              "disclaimer":  str,
            }
        """
        if not self.loaded:
            return self._not_loaded_response()

        arr   = preprocess_bytes(image_bytes)
        probs = self.model.predict(arr, verbose=0)[0]
        topN  = probs.argsort()[-top_n:][::-1]

        preds = [
            {
                "label":      self.class_names[i] if i < len(self.class_names) else f"Class {i}",
                "confidence": round(float(probs[i]) * 100, 2),
            }
            for i in topN
        ]

        top_label = preds[0]["label"]
        return {
            "loaded":      True,
            "predictions": preds,
            "top_finding": top_label,
            "report":      self._generate_report(preds),
            "description": FINDING_DESCRIPTIONS.get(top_label.upper(), ""),
            "disclaimer":  (
                "⚠️ This AI-generated analysis is for educational purposes only. "
                "It does NOT replace a radiologist's or specialist's interpretation. "
                "Always seek professional medical review."
            ),
        }

    def is_ready(self) -> bool:
        return self.loaded

    # ── Private helpers ────────────────────────────────────────────────────────
    def _generate_report(self, preds: list) -> str:
        top  = preds[0]
        rest = preds[1:]
        lines = [
            f"AI Visual Analysis Report",
            f"─" * 36,
            f"Primary Finding  : {top['label']} ({top['confidence']}% confidence)",
        ]
        for p in rest:
            lines.append(f"Alternative      : {p['label']} ({p['confidence']}%)")
        lines += [
            "",
            FINDING_DESCRIPTIONS.get(top["label"].upper(),
                "No specific description available for this finding."),
            "",
            "This report is generated by a Convolutional Neural Network "
            f"(MobileNetV2 backbone) and must be reviewed by a qualified "
            "medical professional before any clinical decision.",
        ]
        return "\n".join(lines)

    def _not_loaded_response(self) -> dict:
        steps = (
            "CNN model not loaded. To enable image diagnosis:\n\n"
            "1. Obtain a labeled medical image dataset (see README for links).\n"
            "2. Organise it as:  dataset/class_a/img.jpg  dataset/class_b/img.jpg\n"
            "3. Train the model:\n"
            "     python3 ml/cnn_model.py --train --dataset ./dataset --epochs 25\n"
            "4. Restart the Flask API.\n\n"
            "The trained model will be saved to ml/models/cnn_model.h5 automatically."
        )
        return {
            "loaded":      False,
            "predictions": [],
            "top_finding": None,
            "report":      steps,
            "description": "",
            "disclaimer":  "Model not loaded. Educational demonstration only.",
        }


# ─────────────────────────────────────────────────────────────────────────────
# 5.  CLI entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MediSense AI v2 — CNN Module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train (transfer learning, 25 epochs):
    python3 ml/cnn_model.py --train --dataset ./chest_xray --epochs 25

  Train from scratch (no pretrained weights):
    python3 ml/cnn_model.py --train --dataset ./skin_lesion --no-transfer --epochs 40

  Show model summary:
    python3 ml/cnn_model.py --summary --classes 2

  Run inference on a local image:
    python3 ml/cnn_model.py --infer --image ./test.jpg
        """,
    )
    parser.add_argument("--train",      action="store_true", help="Train the CNN model")
    parser.add_argument("--infer",      action="store_true", help="Run inference on a single image")
    parser.add_argument("--summary",    action="store_true", help="Print model architecture summary")
    parser.add_argument("--dataset",    type=str, default="./image_dataset",
                        help="Path to dataset directory (for --train)")
    parser.add_argument("--image",      type=str, help="Path to image file (for --infer)")
    parser.add_argument("--epochs",     type=int, default=15, help="Phase-1 training epochs")
    parser.add_argument("--epochs2",    type=int, default=10, help="Phase-2 fine-tune epochs")
    parser.add_argument("--classes",    type=int, default=2,  help="Number of output classes (for --summary)")
    parser.add_argument("--no-transfer",action="store_true", help="Train from scratch instead of MobileNetV2")
    parser.add_argument("--batch",      type=int, default=BATCH_SIZE, help="Batch size")
    args = parser.parse_args()

    if args.train:
        if not os.path.isdir(args.dataset):
            print(f"❌ Dataset directory not found: {args.dataset}")
            raise SystemExit(1)
        train(
            dataset_dir=args.dataset,
            epochs_phase1=args.epochs,
            epochs_phase2=args.epochs2 if not args.no_transfer else 0,
            use_transfer=not args.no_transfer,
            batch_size=args.batch,
        )

    elif args.infer:
        if not args.image:
            print("❌ --image path required for inference."); raise SystemExit(1)
        if not os.path.exists(args.image):
            print(f"❌ Image not found: {args.image}"); raise SystemExit(1)
        predictor = CNNPredictor()
        with open(args.image, "rb") as fh:
            result = predictor.predict(fh.read())
        print(result["report"])

    elif args.summary:
        if not TF_AVAILABLE:
            print("❌ TensorFlow not installed."); raise SystemExit(1)
        build_model(args.classes, use_transfer=not args.no_transfer).summary()

    else:
        parser.print_help()
