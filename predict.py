import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow import keras


def main():
    parser = argparse.ArgumentParser(description="Predict whether an image is a cat or dog.")
    parser.add_argument("--image", required=True, help="Path to an image file.")
    parser.add_argument("--model", default="models/cats_dogs_cnn.keras")
    parser.add_argument("--image-size", type=int, default=160)
    args = parser.parse_args()

    image_path, model_path = Path(args.image), Path(args.model)
    if not image_path.is_file():
        raise SystemExit(f"Image not found: {image_path}")
    if not model_path.is_file():
        raise SystemExit(f"Model not found: {model_path}. Run train.py first.")

    model = keras.models.load_model(model_path)
    image = Image.open(image_path).convert("RGB").resize(
        (args.image_size, args.image_size)
    )
    array = np.asarray(image, dtype=np.float32)[None, ...]
    score = float(model.predict(array, verbose=0)[0][0])

    # Dataset labels are alphabetically ordered: cats=0, dogs=1.
    label = "Dog" if score >= 0.5 else "Cat"
    confidence = score if score >= 0.5 else 1.0 - score
    print(f"Prediction: {label}")
    print(f"Model confidence: {confidence * 100:.2f}%")
    print("Note: confidence is not a guarantee that the prediction is correct.")


if __name__ == "__main__":
    main()
