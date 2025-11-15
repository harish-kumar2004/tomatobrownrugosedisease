"""
Enhanced training pipeline for the tomato disease detector.

Key improvements:
    • Efficient input pipeline using tf.data (no need to load entire dataset into RAM)
    • Transfer learning with EfficientNetB0 + on-the-fly data augmentation
    • Class balancing via computed class weights
    • Early stopping, LR scheduling, best-checkpoint saving, and optional fine-tuning
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

tf.get_logger().setLevel("ERROR")
print(f"TensorFlow version: {tf.__version__}")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "Test Image"
IMG_SIZE = 224
BATCH_SIZE = 32
RANDOM_STATE = 42
AUTOTUNE = tf.data.AUTOTUNE
ALLOWED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def gather_dataset(root: Path) -> Tuple[List[str], List[int], List[str]]:
    """Return filepaths, numeric labels, and full class names."""
    if not root.exists():
        raise FileNotFoundError(f"Dataset folder not found: {root}")

    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if not class_dirs:
        raise RuntimeError(f"No class folders found inside {root}.")

    class_names = [f"Tomato-{d.name.replace(' ', '_')}" for d in class_dirs]
    filepaths: List[str] = []
    labels: List[int] = []

    print(f"\nDiscovered {len(class_dirs)} disease classes:")
    for idx, class_dir in enumerate(class_dirs):
        image_files: List[Path] = []
        for ext in ALLOWED_EXTENSIONS:
            image_files.extend(class_dir.glob(f"*{ext}"))

        if not image_files:
            print(f"  ⚠️  {class_dir.name}: no images found (skipping)")
            continue

        image_files = sorted(image_files)
        filepaths.extend([str(p.resolve()) for p in image_files])
        labels.extend([idx] * len(image_files))
        print(f"  {idx+1:02d}. {class_dir.name:<30} -> {len(image_files):5d} samples")

    if not filepaths:
        raise RuntimeError("No valid images were discovered. Check dataset structure.")

    return filepaths, labels, class_names


def describe_split(name: str, labels: Sequence[int]) -> None:
    """Pretty-print class distribution for a split."""
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    print(f"\n{name} set: {total} samples")
    for u, c in zip(unique, counts):
        pct = (c / total) * 100
        print(f"  Class {u:02d}: {c:5d} ({pct:4.1f}%)")


def decode_and_resize(path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Load an image file, decode, resize and normalize."""
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)  # matches EfficientNet preprocessing
    return image, label


def build_dataset(
    filepaths: Sequence[str], labels: Sequence[int], training: bool
) -> tf.data.Dataset:
    """Create tf.data pipeline."""
    ds = tf.data.Dataset.from_tensor_slices((list(filepaths), list(labels)))
    if training:
        ds = ds.shuffle(len(filepaths), seed=RANDOM_STATE, reshuffle_each_iteration=True)

    ds = ds.map(decode_and_resize, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


def create_model(num_classes: int) -> tf.keras.Model:
    """Build EfficientNet-based classifier with data augmentation."""
    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )

    base_model = EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    return model, base_model


def compute_weights(labels: Sequence[int], num_classes: int) -> Dict[int, float]:
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=np.array(labels),
    )
    return {cls: float(w) for cls, w in enumerate(weights)}


def main() -> None:
    filepaths, labels, class_names = gather_dataset(DATASET_ROOT)
    num_classes = len(class_names)

    # Split 80/10/10 with stratification
    (
        train_paths,
        temp_paths,
        train_labels,
        temp_labels,
    ) = train_test_split(
        filepaths,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=RANDOM_STATE,
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=0.5,
        stratify=temp_labels,
        random_state=RANDOM_STATE,
    )

    describe_split("Train", train_labels)
    describe_split("Validation", val_labels)
    describe_split("Test", test_labels)

    train_ds = build_dataset(train_paths, train_labels, training=True)
    val_ds = build_dataset(val_paths, val_labels, training=False)
    test_ds = build_dataset(test_paths, test_labels, training=False)

    class_weights = compute_weights(train_labels, num_classes)
    print("\nClass weights (to address imbalance):")
    for cls_idx, weight in class_weights.items():
        print(f"  Class {cls_idx:02d}: {weight:.3f}")

    model, base_model = create_model(num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    ckpt_path = "plant_disease_model.h5"
    cb_list = [
        callbacks.ModelCheckpoint(
            ckpt_path,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=7,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print("\nTraining feature-extractor head...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=25,
        class_weight=class_weights,
        callbacks=cb_list,
        verbose=1,
    )

    # Optional fine-tuning: unfreeze top layers of EfficientNet
    fine_tune_at = len(base_model.layers) - 60
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    for layer in base_model.layers[fine_tune_at:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("\nFine-tuning top EfficientNet blocks...")
    fine_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        class_weight=class_weights,
        callbacks=cb_list,
        verbose=1,
    )

    print("\nEvaluating best checkpoint on the independent test set...")
    best_model = tf.keras.models.load_model(ckpt_path)
    test_loss, test_acc = best_model.evaluate(test_ds, verbose=1)
    print(f"Test accuracy: {test_acc * 100:.2f}% | Test loss: {test_loss:.4f}")

    # Persist class names for inference
    with open("class_names.txt", "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(name + "\n")
    print("Class names saved to 'class_names.txt'")

    print("\nTraining complete. Best model stored at 'plant_disease_model.h5'.")


if __name__ == "__main__":
    main()
