"""
LAB 6: Neural Network (NN) on a Dataset of Your Choice
Dataset: Iris (เปลี่ยนเป็น dataset อื่นได้ตามต้องการ)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# ---------- 1. Load dataset ----------
data = load_iris()
X = data.data
y = data.target
feature_names = data.feature_names
target_names = data.target_names

# ---------- 2. Split train/test ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- 3. Standardize features ----------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------- 4. Build NN models with different configs ----------
# ทดลองหลาย configuration: จำนวน hidden layer / neuron ต่างกัน
# และจำนวน epoch (max_iter) ต่างกัน
configs = [
    {"name": "1 hidden layer (10), epoch=100", "hidden_layer_sizes": (10,), "max_iter": 100},
    {"name": "1 hidden layer (10), epoch=300", "hidden_layer_sizes": (10,), "max_iter": 300},
    {"name": "2 hidden layers (20,10), epoch=300", "hidden_layer_sizes": (20, 10), "max_iter": 300},
    {"name": "2 hidden layers (20,10), epoch=500", "hidden_layer_sizes": (20, 10), "max_iter": 500},
]

results = []

for cfg in configs:
    model = MLPClassifier(
        hidden_layer_sizes=cfg["hidden_layer_sizes"],
        max_iter=cfg["max_iter"],
        activation="relu",
        solver="adam",
        random_state=42,
        learning_rate_init=0.01,
        early_stopping=True,               # ใช้ validation set ภายในระหว่าง train
        validation_fraction=0.15,
        n_iter_no_change=cfg["max_iter"],  # ปิด early stop จริง ๆ แต่ยังเก็บ validation score ไว้ดู
    )

    model.fit(X_train_scaled, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
    test_acc = accuracy_score(y_test, model.predict(X_test_scaled))

    results.append({
        "name": cfg["name"],
        "model": model,
        "train_acc": train_acc,
        "test_acc": test_acc,
    })

    print(f"[{cfg['name']}]")
    print(f"  Train Accuracy: {train_acc:.4f}")
    print(f"  Test  Accuracy: {test_acc:.4f}")
    print(f"  Iterations run: {model.n_iter_}")
    print("-" * 50)

# ---------- 5. Compare results ----------
print("\n=== Comparison Table ===")
print(f"{'Configuration':45s} {'Train Acc':>10s} {'Test Acc':>10s}")
for r in results:
    print(f"{r['name']:45s} {r['train_acc']:>10.4f} {r['test_acc']:>10.4f}")

# ---------- 6. Plot training loss curve (best model) ----------
best = max(results, key=lambda r: r["test_acc"])
plt.figure(figsize=(7, 5))
plt.plot(best["model"].loss_curve_, label="Training loss")
if hasattr(best["model"], "validation_scores_"):
    plt.plot(best["model"].validation_scores_, label="Validation accuracy")
plt.title(f"Loss Curve - Best Model: {best['name']}")
plt.xlabel("Epoch")
plt.ylabel("Loss / Score")
plt.legend()
plt.tight_layout()
plt.savefig("nn_loss_curve.png")
plt.show()

# ---------- 7. Predictions on selected samples ----------
print("\n=== Predictions on test set (first 10 samples) ===")
y_pred = best["model"].predict(X_test_scaled)
for i in range(10):
    print(f"Actual: {target_names[y_test[i]]:12s} -> Predicted: {target_names[y_pred[i]]}")

print("\n=== Classification Report (Best Model) ===")
print(classification_report(y_test, y_pred, target_names=target_names))