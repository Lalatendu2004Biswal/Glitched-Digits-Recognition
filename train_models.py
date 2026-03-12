"""
train_models.py
---------------
Trains CNN, SVM, KNN, and Random Forest on MNIST dataset.
Run this ONCE before starting the Flask server.

Usage:
    pip install tensorflow scikit-learn numpy joblib
    python train_models.py

Output:
    models/cnn_model.keras
    models/svm_model.pkl
    models/knn_model.pkl
    models/rf_model.pkl
"""

import os
import numpy as np
import joblib
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

os.makedirs("models", exist_ok=True)

# ── Load MNIST ─────────────────────────────────────────────────────────────────
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# For CNN: keep 28x28, normalize to [0,1]
x_train_cnn = x_train.astype("float32") / 255.0
x_test_cnn  = x_test.astype("float32")  / 255.0
x_train_cnn = x_train_cnn[..., np.newaxis]  # (60000, 28, 28, 1)
x_test_cnn  = x_test_cnn[..., np.newaxis]

# For sklearn models: flatten to (N, 784) and use a subset for speed
x_train_flat = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test_flat  = x_test.reshape(-1, 784).astype("float32")  / 255.0

# Use 20k samples for SVM/KNN/RF to keep training fast
SUBSET = 20000
x_sub = x_train_flat[:SUBSET]
y_sub = y_train[:SUBSET]

# ── CNN ────────────────────────────────────────────────────────────────────────
print("\n[1/4] Training CNN...")
cnn = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(10, activation="softmax"),
])
cnn.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
cnn.fit(x_train_cnn, y_train, epochs=10, batch_size=128,
        validation_split=0.1, verbose=1)
cnn_loss, cnn_acc = cnn.evaluate(x_test_cnn, y_test, verbose=0)
print(f"CNN Test Accuracy: {cnn_acc*100:.2f}%")
cnn.save("models/cnn_model.keras")
print("Saved → models/cnn_model.keras")

# ── SVM ────────────────────────────────────────────────────────────────────────
print("\n[2/4] Training SVM (this may take a few minutes)...")
svm_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=5, gamma="scale", probability=True))
])
svm_pipe.fit(x_sub, y_sub)
svm_acc = svm_pipe.score(x_test_flat, y_test)
print(f"SVM Test Accuracy: {svm_acc*100:.2f}%")
joblib.dump(svm_pipe, "models/svm_model.pkl")
print("Saved → models/svm_model.pkl")

# ── KNN ────────────────────────────────────────────────────────────────────────
print("\n[3/4] Training KNN...")
knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean", n_jobs=-1)
knn.fit(x_sub, y_sub)
knn_acc = knn.score(x_test_flat, y_test)
print(f"KNN Test Accuracy: {knn_acc*100:.2f}%")
joblib.dump(knn, "models/knn_model.pkl")
print("Saved → models/knn_model.pkl")

# ── Random Forest ──────────────────────────────────────────────────────────────
print("\n[4/4] Training Random Forest...")
rf = RandomForestClassifier(n_estimators=200, max_depth=None,
                             n_jobs=-1, random_state=42)
rf.fit(x_sub, y_sub)
rf_acc = rf.score(x_test_flat, y_test)
print(f"Random Forest Test Accuracy: {rf_acc*100:.2f}%")
joblib.dump(rf, "models/rf_model.pkl")
print("Saved → models/rf_model.pkl")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("Training complete! Model accuracies on MNIST test set:")
print(f"  CNN           : {cnn_acc*100:.2f}%")
print(f"  SVM           : {svm_acc*100:.2f}%")
print(f"  KNN           : {knn_acc*100:.2f}%")
print(f"  Random Forest : {rf_acc*100:.2f}%")
print("="*50)
print("\nAll models saved in /models folder.")
print("Now run: python app.py")
