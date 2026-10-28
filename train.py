import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def main():
    parser = argparse.ArgumentParser(description="Train a CNN for cats vs dogs.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=160)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 32:
        parser.error("epochs and batch-size must be positive; image-size must be >= 32")

    data_url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
    archive = keras.utils.get_file(
        fname="cats_and_dogs_filtered.zip",
        origin=data_url,
        extract=True,
    )
    data_dir = Path(archive).parent / "cats_and_dogs_filtered"
    train_dir = data_dir / "train"
    validation_dir = data_dir / "validation"

    if not train_dir.is_dir() or not validation_dir.is_dir():
        raise SystemExit(
            f"Expected dataset folders under {data_dir}. "
            "Check that the dataset archive downloaded and extracted correctly."
        )

    image_size = (args.image_size, args.image_size)
    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        label_mode="binary",
        seed=123,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        validation_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        label_mode="binary",
        shuffle=False,
    )
    print("Class order:", train_ds.class_names)

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)

    model = keras.Sequential([
        layers.Input(shape=(args.image_size, args.image_size, 3)),
        layers.Rescaling(1.0 / 255),
        layers.RandomFlip("horizontal"),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)
    results = model.evaluate(val_ds, verbose=1)
    print(f"Validation loss: {results[0]:.4f}")
    print(f"Validation accuracy: {results[1]:.4f}")

    model_dir, output_dir = Path("models"), Path("outputs")
    model_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    model.save(model_dir / "cats_dogs_cnn.keras")

    epoch_range = range(1, len(history.history["accuracy"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epoch_range, history.history["accuracy"], label="Training")
    axes[0].plot(epoch_range, history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(epoch_range, history.history["loss"], label="Training")
    axes[1].plot(epoch_range, history.history["val_loss"], label="Validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_history.png", dpi=150)
    plt.close(fig)

    print(f"Saved model: {model_dir / 'cats_dogs_cnn.keras'}")
    print(f"Saved chart: {output_dir / 'training_history.png'}")


if __name__ == "__main__":
    main()
