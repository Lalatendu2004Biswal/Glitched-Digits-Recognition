"""
app.py
------
Flask backend for Digit Recognition Stress Test Lab.
Serves predictions from CNN, SVM, KNN, and Random Forest models.

Usage:
    pip install flask flask-cors tensorflow scikit-learn numpy joblib pillow
    python app.py

Endpoints:
    POST /predict         → runs selected algorithm on clean + noisy image
    GET  /models/status   → returns which models are loaded
"""

import os
import io
import base64
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import joblib

app = Flask(__name__)
CORS(app)  # Allow requests from the HTML frontend

# ── Load models at startup ─────────────────────────────────────────────────────
models = {}

def load_models():
    print("Loading models...")

    # CNN via TensorFlow/Keras
    try:
        from tensorflow import keras
        models["cnn"] = keras.models.load_model("models/cnn_model.keras")
        print("  ✓ CNN loaded")
    except Exception as e:
        print(f"  ✗ CNN failed: {e}")

    # Sklearn models
    for name, path in [("svm", "models/svm_model.pkl"),
                        ("knn", "models/knn_model.pkl"),
                        ("rf",  "models/rf_model.pkl")]:
        try:
            models[name] = joblib.load(path)
            print(f"  ✓ {name.upper()} loaded")
        except Exception as e:
            print(f"  ✗ {name.upper()} failed: {e}")

    print(f"Ready. Loaded models: {list(models.keys())}")

# ── Helpers ────────────────────────────────────────────────────────────────────
def base64_to_array(b64_string):
    """Decode base64 PNG → normalized flat numpy array (784,) and 2D (28,28,1)"""
    img_bytes = base64.b64decode(b64_string)
    img = Image.open(io.BytesIO(img_bytes)).convert("L").resize((28, 28))
    arr = np.array(img).astype("float32")
    # Invert (drawn digit is black on white; MNIST is white on black)
    arr = 255.0 - arr
    arr_norm = arr / 255.0
    flat = arr_norm.flatten()           # (784,) for sklearn
    cnn_input = arr_norm.reshape(1, 28, 28, 1)  # (1,28,28,1) for CNN
    return flat, cnn_input

def predict_with_model(model_name, flat, cnn_input):
    """Run prediction and return digit, confidence, probabilities list"""
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' is not loaded.")

    model = models[model_name]

    if model_name == "cnn":
        probs = model.predict(cnn_input, verbose=0)[0]
    else:
        probs = model.predict_proba(flat.reshape(1, -1))[0]

    digit = int(np.argmax(probs))
    confidence = float(probs[digit])
    probabilities = [round(float(p), 4) for p in probs]

    return digit, confidence, probabilities

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/models/status", methods=["GET"])
def model_status():
    return jsonify({
        "loaded": list(models.keys()),
        "available": ["cnn", "svm", "knn", "rf"]
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    algorithm  = data.get("algorithm", "cnn")   # cnn | svm | knn | rf
    clean_b64  = data.get("clean_image")
    noisy_b64  = data.get("noisy_image")

    if not clean_b64 or not noisy_b64:
        return jsonify({"error": "Missing image data"}), 400

    if algorithm not in ["cnn", "svm", "knn", "rf"]:
        return jsonify({"error": f"Unknown algorithm: {algorithm}"}), 400

    try:
        clean_flat, clean_cnn = base64_to_array(clean_b64)
        noisy_flat, noisy_cnn = base64_to_array(noisy_b64)

        clean_digit, clean_conf, clean_probs = predict_with_model(algorithm, clean_flat, clean_cnn)
        noisy_digit, noisy_conf, noisy_probs = predict_with_model(algorithm, noisy_flat, noisy_cnn)

        reliability_drop = max(0.0, round((clean_conf - noisy_conf) * 100, 2))

        # Auto-generate a note
        if clean_digit != noisy_digit:
            note = f"Misclassification! Noise changed prediction from {clean_digit} → {noisy_digit}."
        elif reliability_drop > 20:
            note = f"Correct prediction but confidence dropped by {reliability_drop:.1f}% due to noise."
        elif reliability_drop > 5:
            note = f"Minor confidence degradation ({reliability_drop:.1f}%). Model is fairly robust."
        else:
            note = f"Model robust to applied noise. Confidence stable."

        return jsonify({
            "algorithm": algorithm,
            "clean": {
                "digit": clean_digit,
                "confidence": round(clean_conf, 4),
                "probabilities": clean_probs
            },
            "noisy": {
                "digit": noisy_digit,
                "confidence": round(noisy_conf, 4),
                "probabilities": noisy_probs
            },
            "reliability_drop": reliability_drop,
            "notes": note
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict/all", methods=["POST"])
def predict_all():
    """Run all loaded models on the same input and return a comparison."""
    data = request.get_json()
    clean_b64 = data.get("clean_image")
    noisy_b64 = data.get("noisy_image")

    if not clean_b64 or not noisy_b64:
        return jsonify({"error": "Missing image data"}), 400

    clean_flat, clean_cnn = base64_to_array(clean_b64)
    noisy_flat, noisy_cnn = base64_to_array(noisy_b64)

    results = {}
    for algo in ["cnn", "svm", "knn", "rf"]:
        if algo not in models:
            continue
        try:
            cd, cc, cp = predict_with_model(algo, clean_flat, clean_cnn)
            nd, nc, np_ = predict_with_model(algo, noisy_flat, noisy_cnn)
            results[algo] = {
                "clean": {"digit": cd, "confidence": round(cc, 4), "probabilities": cp},
                "noisy": {"digit": nd, "confidence": round(nc, 4), "probabilities": np_},
                "reliability_drop": max(0.0, round((cc - nc) * 100, 2))
            }
        except Exception as e:
            results[algo] = {"error": str(e)}

    return jsonify(results)


if __name__ == "__main__":
    load_models()
    print("\nStarting Flask server on http://localhost:5000")
    app.run(debug=True, port=5000)
