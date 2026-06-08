"""
modelling.py — MLflow Project Entry Point
==========================================
Script training model deep learning (MobileNetV2) untuk klasifikasi
sampah daur ulang. Dijalankan via MLflow Project.

Terintegrasi dengan pipeline preprocessing di 1-Preprocessing/automate_Muhammad_Rahman.py

Usage (via MLflow):
    mlflow run MLProject --env-manager=local \
        -P data_path=sampah-daur-ulang \
        -P epochs=10 \
        -P learning_rate=0.001 \
        -P batch_size=32
"""

import argparse
import os
import sys
import random
import shutil
import subprocess
import zipfile
from pathlib import Path

import mlflow
import numpy as np
import tensorflow as tf

# ======================== CONFIGURATION ========================
SEED = 42
IMG_SIZE = (224, 224)
IMG_EXTS = {".jpg", ".jpeg", ".png"}
DATASET_SLUG = "fathurrahmanalfarizy/sampah-daur-ulang"
ZIP_NAME = "sampah-daur-ulang.zip"

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ======================== DATA FUNCTIONS ========================

def setup_kaggle():
    """Setup Kaggle credentials."""
    kaggle_dir = Path.home() / ".kaggle"
    target = kaggle_dir / "kaggle.json"
    if target.exists():
        os.chmod(target, 0o600)
        os.environ["KAGGLE_CONFIG_DIR"] = str(kaggle_dir)
        return True
    return False


def download_and_extract(output_dir="."):
    """Download and extract dataset from Kaggle if needed."""
    extract_dir = Path(output_dir)
    if extract_dir.exists() and any(extract_dir.rglob("*.jpg")):
        print(f"Dataset sudah ada di {extract_dir}")
        return extract_dir.resolve()

    if not setup_kaggle():
        raise FileNotFoundError("Kaggle credentials tidak ditemukan.")

    try:
        import kaggle  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kaggle"])

    kaggle_cli = Path(sys.executable).with_name("kaggle")
    if not kaggle_cli.exists():
        kaggle_cli = Path(shutil.which("kaggle") or "kaggle")

    zip_path = Path(ZIP_NAME)
    if not zip_path.exists():
        subprocess.check_call([
            str(kaggle_cli), "datasets", "download",
            "-d", DATASET_SLUG, "-p", ".", "--force",
        ])

    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    return extract_dir.resolve()


def find_class_root(root):
    """Find directory containing class subdirectories."""
    for p in root.rglob("*"):
        if p.is_dir():
            subdirs = [d for d in p.iterdir() if d.is_dir()]
            if len(subdirs) >= 2:
                has_imgs = sum(
                    1 for d in subdirs
                    if any(f.suffix.lower() in IMG_EXTS for f in d.iterdir() if f.is_file())
                )
                if has_imgs >= 2:
                    return p
    return None


def list_images(base_dir, class_names):
    """Collect all image paths and labels from class directories."""
    paths, labels = [], []
    for idx, cn in enumerate(class_names):
        for p in (base_dir / cn).rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                paths.append(str(p))
                labels.append(idx)
    return np.array(paths), np.array(labels)


def load_data(data_path, batch_size):
    """Load and prepare datasets — consistent with 1-Preprocessing pipeline."""
    dataset_root = Path(data_path).resolve()

    if not dataset_root.exists() or not any(dataset_root.rglob("*.jpg")):
        dataset_root = download_and_extract(data_path)

    class_root = find_class_root(dataset_root)
    if class_root is None:
        raise RuntimeError(f"Tidak menemukan folder kelas di {dataset_root}")

    class_names = sorted([d.name for d in class_root.iterdir() if d.is_dir()])
    num_classes = len(class_names)

    paths, labels = list_images(class_root, class_names)
    idx = np.arange(len(paths))
    np.random.shuffle(idx)
    paths, labels = paths[idx], labels[idx]

    # Split ratios consistent with 1-Preprocessing (70/15/15)
    n = len(paths)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    def decode(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, IMG_SIZE)
        return tf.cast(img, tf.float32), label

    def make_ds(p, l, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices((p, l))
        if shuffle:
            ds = ds.shuffle(min(len(p), 1000), seed=SEED)
        return ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train_ds = make_ds(paths[:n_train], labels[:n_train], shuffle=True)
    val_ds = make_ds(paths[n_train:n_train+n_val], labels[n_train:n_train+n_val])
    test_ds = make_ds(paths[n_train+n_val:], labels[n_train+n_val:])

    print(f"Classes: {class_names} | Train: {n_train} | Val: {n_val} | Test: {n - n_train - n_val}")
    return train_ds, val_ds, test_ds, class_names, num_classes


# ======================== MODEL ========================

def build_model(num_classes, learning_rate, dropout=0.3, dense_units=256):
    """Build MobileNetV2 Sequential model — consistent with 1-Preprocessing architecture."""
    base = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomContrast(0.2),
    ], name="data_augmentation")

    model = tf.keras.Sequential([
        tf.keras.Input(shape=IMG_SIZE + (3,)),
        aug,
        tf.keras.layers.Lambda(
            tf.keras.applications.mobilenet_v2.preprocess_input,
            name="preprocess_input",
        ),
        base,
        tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="post_conv"),
        tf.keras.layers.MaxPooling2D(name="post_pool"),
        tf.keras.layers.GlobalAveragePooling2D(name="post_gap"),
        tf.keras.layers.BatchNormalization(name="post_bn1"),
        tf.keras.layers.Dropout(dropout, name="post_dropout1"),
        tf.keras.layers.Dense(dense_units, activation="relu", name="post_dense1"),
        tf.keras.layers.BatchNormalization(name="post_bn2"),
        tf.keras.layers.Dropout(dropout, name="post_dropout2"),
        tf.keras.layers.Dense(dense_units // 2, activation="relu", name="post_dense2"),
        tf.keras.layers.Dropout(dropout * 0.67, name="post_dropout3"),
        tf.keras.layers.Dense(num_classes, activation="softmax", name="post_logits"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base


# ======================== MAIN ========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="sampah-daur-ulang")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    train_ds, val_ds, test_ds, class_names, num_classes = load_data(args.data_path, args.batch_size)
    model, base_model = build_model(num_classes, args.learning_rate)

    with mlflow.start_run():
        # Manual logging — parameters
        mlflow.log_param("architecture", "Sequential_MobileNetV2")
        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_param("class_names", str(class_names))
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("img_size", str(IMG_SIZE))
        mlflow.log_param("split_ratio", "70/15/15")

        # Phase 1: Head training
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        ]
        h1 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

        # Phase 2: Fine-tuning
        base_model.trainable = True
        ft_at = max(0, len(base_model.layers) - 50)
        for layer in base_model.layers[:ft_at]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        h2 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

        # Manual logging — metrics per epoch
        combined_loss = h1.history["loss"] + h2.history["loss"]
        combined_acc = h1.history["accuracy"] + h2.history["accuracy"]
        combined_vloss = h1.history["val_loss"] + h2.history["val_loss"]
        combined_vacc = h1.history["val_accuracy"] + h2.history["val_accuracy"]

        for i in range(len(combined_loss)):
            mlflow.log_metric("train_loss", combined_loss[i], step=i)
            mlflow.log_metric("train_accuracy", combined_acc[i], step=i)
            mlflow.log_metric("val_loss", combined_vloss[i], step=i)
            mlflow.log_metric("val_accuracy", combined_vacc[i], step=i)

        # Test evaluation
        test_loss, test_acc = model.evaluate(test_ds, verbose=1)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("best_val_accuracy", max(combined_vacc))

        # Log model artifact
        mlflow.tensorflow.log_model(model, artifact_path="model")

        print(f"\nTest Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")


if __name__ == "__main__":
    main()
