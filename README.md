# CNN Image Classification: Cats vs Dogs

A beginner-friendly convolutional neural network (CNN) built with TensorFlow/Keras to classify images as cats or dogs. Includes a graphical interface for training the model, viewing training logs, selecting an image, and running a prediction.

## Features
- **GUI with two tabs:** train the CNN and predict a new image
- Configure training epochs and batch size
- Live training output in the interface and stop-training control
- Choose an image and preview it before prediction
- Displays the predicted class and model confidence
- Saves the trained model and training-history chart

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

## Launch the graphical interface
```powershell
python gui.py
```
1. Open the **Train model** tab and choose the epoch count and batch size.
2. Click **Start training**. The first run downloads the Cats vs Dogs dataset.
3. Wait for training to finish; the model is saved to `models/cats_dogs_cnn.keras`.
4. Open the **Predict image** tab and choose a cat or dog image.
5. Click **Predict** to see the class and confidence estimate.

## Command-line alternatives
Train:
```powershell
python train.py --epochs 5
```
Predict:
```powershell
python predict.py --image path\to\cat_or_dog.jpg
```

## Output files
- `models/cats_dogs_cnn.keras` — trained model
- `outputs/training_history.png` — training and validation accuracy/loss plot

Generated model files and outputs are excluded from Git by default. Upload screenshots and measured results instead.

## Screenshot checklist
Capture actual runs using `Win + Shift + S`:
1. `screenshots/01_gui_training.png` — training tab showing logs.
2. `screenshots/02_training_graph.png` — training-history chart.
3. `screenshots/03_gui_prediction.png` — image and prediction result.
4. `screenshots/04_terminal_or_result.png` — successful completion details.

## Important
Do not claim a particular accuracy until you have measured it on your own validation/test set. Results vary with dataset, initialization, and training settings. Model confidence is not a guarantee of correctness.

## Resume description
**CNN Image Classification | Python, TensorFlow, Keras, Tkinter**
- Built a GUI application to train a CNN and classify cat/dog images.
- Added configurable training, live log display, image preview, model inference, and confidence output.
- Visualized training and validation metrics to inspect model performance.

## References
- TensorFlow image classification tutorial: https://www.tensorflow.org/tutorials/images/classification
- Keras image classification from scratch: https://keras.io/examples/vision/image_classification_from_scratch/
