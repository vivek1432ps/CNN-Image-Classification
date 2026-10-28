import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

class CNNClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CNN Image Classifier — Cats vs Dogs")
        self.root.geometry("1000x760")
        self.root.minsize(780, 600)
        self.events = queue.Queue()
        self.training_process = None
        self.training_thread = None
        self.selected_image = None
        self.preview_photo = None

        main = ttk.Frame(root, padding=14)
        main.pack(fill="both", expand=True)
        ttk.Label(main, text="CNN Image Classification", font=("Segoe UI", 20, "bold")).pack(anchor="w")
        ttk.Label(
            main, text="Train a TensorFlow CNN and classify a cat or dog image using the saved model.",
            wraplength=900
        ).pack(anchor="w", pady=(4, 12))

        tabs = ttk.Notebook(main)
        tabs.pack(fill="both", expand=True)
        self.train_tab = ttk.Frame(tabs, padding=12)
        self.predict_tab = ttk.Frame(tabs, padding=12)
        tabs.add(self.train_tab, text="Train model")
        tabs.add(self.predict_tab, text="Predict image")

        self.build_train_tab()
        self.build_predict_tab()
        self.root.after(100, self.poll_events)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def build_train_tab(self):
        ttk.Label(self.train_tab, text="Training settings", font=("Segoe UI", 13, "bold")).pack(anchor="w")
        options = ttk.Frame(self.train_tab)
        options.pack(fill="x", pady=10)
        ttk.Label(options, text="Epochs").grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.epochs = tk.StringVar(value="5")
        ttk.Entry(options, textvariable=self.epochs, width=8).grid(row=0, column=1, padx=(0, 18))
        ttk.Label(options, text="Batch size").grid(row=0, column=2, sticky="w", padx=(0, 6))
        self.batch_size = tk.StringVar(value="32")
        ttk.Entry(options, textvariable=self.batch_size, width=8).grid(row=0, column=3, padx=(0, 18))
        self.train_button = ttk.Button(options, text="Start training", command=self.start_training)
        self.train_button.grid(row=0, column=4, padx=4)
        self.stop_button = ttk.Button(options, text="Stop training", command=self.stop_training, state="disabled")
        self.stop_button.grid(row=0, column=5, padx=4)

        ttk.Label(
            self.train_tab,
            text="The first run downloads the Cats vs Dogs dataset. Training can take several minutes or longer depending on hardware.",
            wraplength=850,
        ).pack(anchor="w", pady=(0, 8))
        self.log = tk.Text(self.train_tab, height=20, wrap="word", state="disabled")
        self.log.pack(fill="both", expand=True)
        ttk.Label(
            self.train_tab, text="After successful training, the model is saved to models/cats_dogs_cnn.keras.",
            foreground="#555"
        ).pack(anchor="w", pady=(8, 0))

    def build_predict_tab(self):
        row = ttk.Frame(self.predict_tab)
        row.pack(fill="x")
        self.image_path_label = tk.StringVar(value="No image selected")
        ttk.Button(row, text="Choose cat/dog image…", command=self.choose_image).pack(side="left")
        ttk.Label(row, textvariable=self.image_path_label).pack(side="left", padx=10)
        self.predict_button = ttk.Button(row, text="Predict", command=self.predict_image, state="disabled")
        self.predict_button.pack(side="right")

        body = ttk.Frame(self.predict_tab)
        body.pack(fill="both", expand=True, pady=14)
        self.image_preview = ttk.Label(body, text="Select an image to preview", anchor="center")
        self.image_preview.pack(fill="both", expand=True)
        self.prediction_result = tk.StringVar(value="Train a model first, then choose an image.")
        ttk.Label(body, textvariable=self.prediction_result, font=("Segoe UI", 14, "bold"), wraplength=800).pack(pady=10)

    def append_log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def start_training(self):
        try:
            epochs = int(self.epochs.get())
            batch = int(self.batch_size.get())
            if epochs < 1 or batch < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid settings", "Epochs and batch size must be positive integers.")
            return

        self.train_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.append_log(f"Starting training: {epochs} epochs, batch size {batch}")
        self.training_thread = threading.Thread(
            target=self.training_worker, args=(epochs, batch), daemon=True
        )
        self.training_thread.start()

    def training_worker(self, epochs, batch):
        try:
            process = subprocess.Popen(
                [sys.executable, "-u", "train.py", "--epochs", str(epochs), "--batch-size", str(batch)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            self.training_process = process
            for line in process.stdout:
                self.events.put(("log", line.rstrip()))
            code = process.wait()
            self.events.put(("training_done", code))
        except Exception as exc:
            self.events.put(("error", f"Could not start training: {exc}"))

    def stop_training(self):
        process = self.training_process
        if process and process.poll() is None:
            process.terminate()
            self.append_log("Stop requested. Waiting for training process to exit…")
        else:
            self.append_log("No active training process was found.")

    def choose_image(self):
        path = filedialog.askopenfilename(
            title="Choose a cat or dog image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.webp *.bmp"), ("All files", "*.*")]
        )
        if not path:
            return
        self.selected_image = Path(path)
        self.image_path_label.set(self.selected_image.name)
        try:
            image = Image.open(path).convert("RGB")
            image.thumbnail((600, 420))
            self.preview_photo = ImageTk.PhotoImage(image.copy())
            self.image_preview.configure(image=self.preview_photo, text="")
        except Exception as exc:
            messagebox.showerror("Image error", str(exc))
            return
        model_path = Path("models/cats_dogs_cnn.keras")
        self.predict_button.configure(state="normal" if model_path.is_file() else "disabled")
        if not model_path.is_file():
            self.prediction_result.set("Model not found. Train the CNN first.")

    def predict_image(self):
        if not self.selected_image or not self.selected_image.is_file():
            messagebox.showerror("Select image", "Choose a valid image first.")
            return
        if not Path("models/cats_dogs_cnn.keras").is_file():
            messagebox.showerror("Model missing", "Train the CNN first using the Train model tab.")
            return
        self.predict_button.configure(state="disabled")
        self.prediction_result.set("Loading model and predicting…")
        threading.Thread(target=self.predict_worker, args=(self.selected_image,), daemon=True).start()

    def predict_worker(self, image_path):
        try:
            import numpy as np
            from tensorflow import keras
            model = keras.models.load_model("models/cats_dogs_cnn.keras")
            image = Image.open(image_path).convert("RGB").resize((160, 160))
            array = np.asarray(image, dtype="float32")[None, ...]
            score = float(model.predict(array, verbose=0)[0][0])
            label = "Dog" if score >= 0.5 else "Cat"
            confidence = score if score >= 0.5 else 1.0 - score
            self.events.put(("prediction", f"Prediction: {label} | Model confidence: {confidence * 100:.2f}%"))
        except Exception as exc:
            self.events.put(("error", f"Prediction failed: {exc}"))
        finally:
            self.events.put(("prediction_done", None))

    def poll_events(self):
        try:
            while True:
                event, value = self.events.get_nowait()
                if event == "log":
                    self.append_log(value)
                elif event == "training_done":
                    self.training_process = None
                    self.train_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                    self.append_log("Training finished successfully." if value == 0 else f"Training exited with code {value}.")
                    self.predict_button.configure(
                        state="normal" if self.selected_image and Path("models/cats_dogs_cnn.keras").is_file() else "disabled"
                    )
                elif event == "prediction":
                    self.prediction_result.set(value)
                elif event == "prediction_done":
                    self.predict_button.configure(state="normal")
                elif event == "error":
                    self.append_log(value)
                    messagebox.showerror("Error", value)
                    self.train_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                    self.predict_button.configure(state="normal" if Path("models/cats_dogs_cnn.keras").is_file() else "disabled")
        except queue.Empty:
            pass
        if self.root.winfo_exists():
            self.root.after(100, self.poll_events)

    def on_close(self):
        if self.training_process and self.training_process.poll() is None:
            self.training_process.terminate()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = CNNClassifierGUI(root)
    root.mainloop()
