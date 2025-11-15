from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import io
import contextlib
import traceback
import sys
import time
import os

# ==============================
# Flask setup
# ==============================
app = Flask(__name__)

# Path to your trained model file (.h5 or .keras)
MODEL_PATH = r"C:\Users\NVH Krishna\Desktop\mini_project\code\model.h5"

# ==============================
# Load the model once at startup
# ==============================
print("\n🚀 Starting Flask app...")
print("📦 Loading trained model...")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!\n")
except Exception as e:
    print("❌ Failed to load model:", e)
    model = None


# ==============================
# Helper function to run prediction
# ==============================
def run_prediction():
    """Runs a dummy or test prediction using the loaded model."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        try:
            print(">>> Running model prediction...\n")
            start = time.time()

            # Example input (you can replace this with real input)
            dummy_input = np.random.rand(1, 128, 128, 3)  # adjust shape as per your model
            pred = model.predict(dummy_input)

            predicted_class = np.argmax(pred)
            print(f"Predicted class index: {predicted_class}")
            print(f"Raw prediction vector: {pred}")
            print(f"\n✅ Prediction completed in {time.time() - start:.2f}s.")

        except Exception as e:
            print("❌ Error while running prediction:\n")
            traceback.print_exc(file=sys.stdout)

    return buffer.getvalue()


# ==============================
# Flask routes
# ==============================
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/run_model', methods=['POST'])
def run_model():
    if model is None:
        return render_template('index.html', output="❌ Model not loaded.")
    output = run_prediction()
    return render_template('index.html', output=output)


# ==============================
# Run Flask app
# ==============================
if __name__ == '__main__':
    app.run(debug=True)
