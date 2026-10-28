# CNN Image Classification: Cats vs Dogs

A beginner-friendly convolutional neural network (CNN) built with TensorFlow/Keras to classify images as cats or dogs.

## Features
- Downloads the Cats vs Dogs dataset automatically on the first training run
- Trains a CNN from scratch
- Saves the trained model and training-history chart
- Predicts a new image and prints a confidence score

## Requirements
- Python 3.10 or 3.11 recommended
- 8 GB RAM recommended
- CPU supported; GPU optional

## Setup (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train the CNN
```powershell
python train.py --epochs 5
```
The first run downloads the dataset. Start with 5 epochs to check that everything works. Training time depends on hardware.

## Predict an image
After training, use a local cat or dog image:
```powershell
python predict.py --image path\to\cat_or_dog.jpg
```
The trained model is saved at `models/cats_dogs_cnn.keras`.

## Output files
- `models/cats_dogs_cnn.keras` — trained model
- `outputs/training_history.png` — training and validation accuracy/loss plot

Generated model files and outputs are excluded from Git by default. Upload screenshots and measured results instead.

## Screenshot checklist
Capture actual runs using `Win + Shift + S`:
1. `screenshots/01_dataset.png` — dataset loading in the terminal.
2. `screenshots/02_training.png` — training epochs and metrics.
3. `screenshots/03_graph.png` — training-history chart.
4. `screenshots/04_prediction.png` — prediction for a test image.

## Important
Do not claim a particular accuracy until you have measured it on your own validation/test set. Results vary with dataset, initialization, and training settings.

## Resume description
**CNN Image Classification | Python, TensorFlow, Keras**
- Built and trained a CNN to classify cat and dog images.
- Implemented image loading, normalization, validation evaluation, model saving, and single-image inference.
- Visualized training and validation metrics to inspect model performance.

## References
- TensorFlow image classification tutorial: https://www.tensorflow.org/tutorials/images/classification
- Keras image classification from scratch: https://keras.io/examples/vision/image_classification_from_scratch/
