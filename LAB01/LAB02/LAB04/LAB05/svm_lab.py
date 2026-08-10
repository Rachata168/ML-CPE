import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

df = pd.DataFrame(X, columns=feature_names)
df['target'] = y
print(df.head())
print(df.describe())
print(df['target'].value_counts())

sns.pairplot(df, hue='target')
plt.savefig('pairplot.png')
plt.close()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

kernels = ['linear', 'poly', 'rbf']
results = {}

for kernel in kernels:
    model = SVC(kernel=kernel, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    results[kernel] = acc

    print(f"\n=== Kernel: {kernel} ===")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=target_names))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Confusion Matrix - {kernel}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'confusion_matrix_{kernel}.png')
    plt.close()

print("\n=== Summary ===")
for kernel, acc in results.items():
    print(f"{kernel}: {acc:.4f}")

plt.figure()
plt.bar(results.keys(), results.values(), color=['skyblue', 'orange', 'green'])
plt.ylabel('Accuracy')
plt.title('SVM Kernel Comparison')
plt.ylim(0, 1)
plt.savefig('kernel_comparison.png')
plt.close()

sample = X_test_scaled[:5]
best_kernel = max(results, key=results.get)
best_model = SVC(kernel=best_kernel, random_state=42)
best_model.fit(X_train_scaled, y_train)
predictions = best_model.predict(sample)
print(f"\nPredictions using best kernel ({best_kernel}):")
for i, pred in enumerate(predictions):
    print(f"Sample {i+1}: Predicted = {target_names[pred]}, Actual = {target_names[y_test[i]]}")