import pandas as pd

data = pd.read_csv("data/creditcard.csv")

print(data.info())
print(data.isnull().sum().sum())

fraud = data[data["Class"] == 1]
normal = data[data["Class"] == 0]

print("Мошеннических:", len(fraud))
print("Обычных:", len(normal))
print("Доля мошеннических, %:", len(fraud) / len(data) * 100)

import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x="Class", data=data)
plt.title("Распределение классов")
plt.show()

X = data.drop("Class", axis=1)
y = data["Class"]

print("Размер X:", X.shape)
print("Размер y:", y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Размер X_train:", X_train.shape)
print("Размер X_test:", X_test.shape)

print("Классы в обучающей выборке:")
print(y_train.value_counts())

print("Классы в тестовой выборке:")
print(y_test.value_counts())

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

scaler = StandardScaler()

X_train_scaled[["Time", "Amount"]] = scaler.fit_transform(
    X_train[["Time", "Amount"]]
)

X_test_scaled[["Time", "Amount"]] = scaler.transform(
    X_test[["Time", "Amount"]]
)

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print("Первые 20 предсказаний модели:")
print(y_pred[:20])

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

y_proba = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
pr_auc = average_precision_score(y_test, y_proba)

print("\nМетрики базовой Logistic Regression:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("ROC-AUC:", roc_auc)
print("PR-AUC:", pr_auc)

print("\nClassification report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix — Logistic Regression")
plt.xlabel("Предсказанный класс")
plt.ylabel("Реальный класс")
plt.show()

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION С CLASS_WEIGHT='BALANCED'")
print("=" * 60)

model_balanced = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight="balanced"
)

model_balanced.fit(X_train_scaled, y_train)

y_pred_balanced = model_balanced.predict(X_test_scaled)
y_proba_balanced = model_balanced.predict_proba(X_test_scaled)[:, 1]

accuracy_balanced = accuracy_score(y_test, y_pred_balanced)
precision_balanced = precision_score(y_test, y_pred_balanced)
recall_balanced = recall_score(y_test, y_pred_balanced)
f1_balanced = f1_score(y_test, y_pred_balanced)
roc_auc_balanced = roc_auc_score(y_test, y_proba_balanced)
pr_auc_balanced = average_precision_score(y_test, y_proba_balanced)

print("\nМетрики Logistic Regression с балансировкой:")
print("Accuracy:", accuracy_balanced)
print("Precision:", precision_balanced)
print("Recall:", recall_balanced)
print("F1-score:", f1_balanced)
print("ROC-AUC:", roc_auc_balanced)
print("PR-AUC:", pr_auc_balanced)

print("\nClassification report:")
print(classification_report(y_test, y_pred_balanced))

cm_balanced = confusion_matrix(y_test, y_pred_balanced)

print("\nConfusion Matrix:")
print(cm_balanced)

sns.heatmap(
    cm_balanced,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix — Logistic Regression Balanced")
plt.xlabel("Предсказанный класс")
plt.ylabel("Реальный класс")
plt.show()

from imblearn.over_sampling import SMOTE

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION + SMOTE")
print("=" * 60)

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_scaled,
    y_train
)

print("\nКоличество классов после SMOTE:")
print(y_train_smote.value_counts())

model_smote = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model_smote.fit(X_train_smote, y_train_smote)

y_pred_smote = model_smote.predict(X_test_scaled)
y_proba_smote = model_smote.predict_proba(X_test_scaled)[:, 1]

accuracy_smote = accuracy_score(y_test, y_pred_smote)
precision_smote = precision_score(y_test, y_pred_smote)
recall_smote = recall_score(y_test, y_pred_smote)
f1_smote = f1_score(y_test, y_pred_smote)
roc_auc_smote = roc_auc_score(y_test, y_proba_smote)
pr_auc_smote = average_precision_score(y_test, y_proba_smote)

print("\nМетрики Logistic Regression + SMOTE:")
print("Accuracy:", accuracy_smote)
print("Precision:", precision_smote)
print("Recall:", recall_smote)
print("F1-score:", f1_smote)
print("ROC-AUC:", roc_auc_smote)
print("PR-AUC:", pr_auc_smote)

print("\nClassification report:")
print(classification_report(y_test, y_pred_smote))

cm_smote = confusion_matrix(y_test, y_pred_smote)

print("\nConfusion Matrix:")
print(cm_smote)

sns.heatmap(
    cm_smote,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix — Logistic Regression + SMOTE")
plt.xlabel("Предсказанный класс")
plt.ylabel("Реальный класс")
plt.show()

from imblearn.under_sampling import RandomUnderSampler

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION + RANDOM UNDERSAMPLING")
print("=" * 60)

undersampler = RandomUnderSampler(random_state=42)

X_train_under, y_train_under = undersampler.fit_resample(
    X_train_scaled,
    y_train
)

print("\nКоличество классов после RandomUnderSampler:")
print(y_train_under.value_counts())

model_under = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model_under.fit(X_train_under, y_train_under)

y_pred_under = model_under.predict(X_test_scaled)
y_proba_under = model_under.predict_proba(X_test_scaled)[:, 1]

accuracy_under = accuracy_score(y_test, y_pred_under)
precision_under = precision_score(y_test, y_pred_under)
recall_under = recall_score(y_test, y_pred_under)
f1_under = f1_score(y_test, y_pred_under)
roc_auc_under = roc_auc_score(y_test, y_proba_under)
pr_auc_under = average_precision_score(y_test, y_proba_under)

print("\nМетрики Logistic Regression + RandomUnderSampler:")
print("Accuracy:", accuracy_under)
print("Precision:", precision_under)
print("Recall:", recall_under)
print("F1-score:", f1_under)
print("ROC-AUC:", roc_auc_under)
print("PR-AUC:", pr_auc_under)

print("\nClassification report:")
print(classification_report(y_test, y_pred_under))

cm_under = confusion_matrix(y_test, y_pred_under)

print("\nConfusion Matrix:")
print(cm_under)

sns.heatmap(
    cm_under,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix — Logistic Regression + RandomUnderSampler")
plt.xlabel("Предсказанный класс")
plt.ylabel("Реальный класс")
plt.show()

print("\n" + "=" * 60)
print("ИТОГОВОЕ СРАВНЕНИЕ МЕТОДОВ")
print("=" * 60)

results = pd.DataFrame({
    "Метод": [
        "Baseline",
        "Class Weight",
        "SMOTE",
        "Random UnderSampling"
    ],
    "Accuracy": [
        accuracy,
        accuracy_balanced,
        accuracy_smote,
        accuracy_under
    ],
    "Precision": [
        precision,
        precision_balanced,
        precision_smote,
        precision_under
    ],
    "Recall": [
        recall,
        recall_balanced,
        recall_smote,
        recall_under
    ],
    "F1-score": [
        f1,
        f1_balanced,
        f1_smote,
        f1_under
    ],
    "ROC-AUC": [
        roc_auc,
        roc_auc_balanced,
        roc_auc_smote,
        roc_auc_under
    ],
    "PR-AUC": [
        pr_auc,
        pr_auc_balanced,
        pr_auc_smote,
        pr_auc_under
    ]
})

print("\n")
print(results.round(4))

results_plot = results.set_index("Метод")[
    ["Precision", "Recall", "F1-score", "PR-AUC"]
]

results_plot.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Сравнение методов борьбы с дисбалансом классов")
plt.ylabel("Значение метрики")
plt.xlabel("Метод")
plt.xticks(rotation=0)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

from sklearn.metrics import precision_recall_curve

precision_base_curve, recall_base_curve, _ = precision_recall_curve(
    y_test,
    y_proba
)

precision_balanced_curve, recall_balanced_curve, _ = precision_recall_curve(
    y_test,
    y_proba_balanced
)

precision_smote_curve, recall_smote_curve, _ = precision_recall_curve(
    y_test,
    y_proba_smote
)

precision_under_curve, recall_under_curve, _ = precision_recall_curve(
    y_test,
    y_proba_under
)

plt.figure(figsize=(10, 6))

plt.plot(
    recall_base_curve,
    precision_base_curve,
    label=f"Baseline (PR-AUC = {pr_auc:.3f})"
)

plt.plot(
    recall_balanced_curve,
    precision_balanced_curve,
    label=f"Class Weight (PR-AUC = {pr_auc_balanced:.3f})"
)

plt.plot(
    recall_smote_curve,
    precision_smote_curve,
    label=f"SMOTE (PR-AUC = {pr_auc_smote:.3f})"
)

plt.plot(
    recall_under_curve,
    precision_under_curve,
    label=f"Random UnderSampling (PR-AUC = {pr_auc_under:.3f})"
)

plt.title("Precision-Recall Curves")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

results.to_csv(
    "model_results.csv",
    index=False
)

print("\nРезультаты сохранены в model_results.csv")

from sklearn.ensemble import RandomForestClassifier

print("\n" + "=" * 60)
print("RANDOM FOREST")
print("=" * 60)

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

rf_model.fit(X_train_scaled, y_train)

y_pred_rf = rf_model.predict(X_test_scaled)
y_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_proba_rf)
pr_auc_rf = average_precision_score(y_test, y_proba_rf)

print("\nМетрики Random Forest:")
print("Accuracy:", accuracy_rf)
print("Precision:", precision_rf)
print("Recall:", recall_rf)
print("F1-score:", f1_rf)
print("ROC-AUC:", roc_auc_rf)
print("PR-AUC:", pr_auc_rf)

print("\nClassification report:")
print(classification_report(y_test, y_pred_rf))

cm_rf = confusion_matrix(y_test, y_pred_rf)

print("\nConfusion Matrix:")
print(cm_rf)

sns.heatmap(
    cm_rf,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix — Random Forest")
plt.xlabel("Предсказанный класс")
plt.ylabel("Реальный класс")
plt.show()

rf_result = pd.DataFrame({
    "Метод": ["Random Forest"],
    "Accuracy": [accuracy_rf],
    "Precision": [precision_rf],
    "Recall": [recall_rf],
    "F1-score": [f1_rf],
    "ROC-AUC": [roc_auc_rf],
    "PR-AUC": [pr_auc_rf]
})

results = pd.concat(
    [results, rf_result],
    ignore_index=True
)

print("\nИтоговая таблица с Random Forest:")
print(results.round(4))

results.to_csv(
    "model_results.csv",
    index=False
)

import numpy as np

print("\n" + "=" * 60)
print("ПОДБОР THRESHOLD ДЛЯ RANDOM FOREST")
print("=" * 60)

X_train_threshold, X_val, y_train_threshold, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

scaler_threshold = StandardScaler()

X_train_threshold_scaled = X_train_threshold.copy()
X_val_scaled = X_val.copy()

X_train_threshold_scaled[["Time", "Amount"]] = scaler_threshold.fit_transform(
    X_train_threshold[["Time", "Amount"]]
)

X_val_scaled[["Time", "Amount"]] = scaler_threshold.transform(
    X_val[["Time", "Amount"]]
)

rf_threshold_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

rf_threshold_model.fit(
    X_train_threshold_scaled,
    y_train_threshold
)

y_val_proba = rf_threshold_model.predict_proba(
    X_val_scaled
)[:, 1]

thresholds = np.arange(0.05, 0.51, 0.01)

best_threshold = 0.5
best_f1 = 0

for threshold in thresholds:
    y_val_pred = (y_val_proba >= threshold).astype(int)

    current_f1 = f1_score(
        y_val,
        y_val_pred
    )

    if current_f1 > best_f1:
        best_f1 = current_f1
        best_threshold = threshold

print("\nЛучший threshold:", round(best_threshold, 2))
print("луучший F1 на validation:", round(best_f1, 4))

y_pred_rf_threshold = (
    y_proba_rf >= best_threshold
).astype(int)

accuracy_rf_threshold = accuracy_score(
    y_test,
    y_pred_rf_threshold
)

precision_rf_threshold = precision_score(
    y_test,
    y_pred_rf_threshold
)

recall_rf_threshold = recall_score(
    y_test,
    y_pred_rf_threshold
)

f1_rf_threshold = f1_score(
    y_test,
    y_pred_rf_threshold
)

print("\nRandom Forest с подобранным threshold:")
print("Threshold:", round(best_threshold, 2))
print("Accuracy:", accuracy_rf_threshold)
print("Precision:", precision_rf_threshold)
print("Recall:", recall_rf_threshold)
print("F1-score:", f1_rf_threshold)

cm_rf_threshold = confusion_matrix(
    y_test,
    y_pred_rf_threshold
)

print("\nConfusion Matrix:")
print(cm_rf_threshold)

sns.heatmap(
    cm_rf_threshold,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title(
    f"Random Forest — Threshold {best_threshold:.2f}"
)
plt.xlabel("предсказанный класс")
plt.ylabel("Реальный класс")
plt.show()

rf_threshold_result = pd.DataFrame({
    "Метод": [
        f"Random Forest threshold={best_threshold:.2f}"
    ],
    "Accuracy": [
        accuracy_rf_threshold
    ],
    "Precision": [
        precision_rf_threshold
    ],
    "Recall": [
        recall_rf_threshold
    ],
    "F1-score": [
        f1_rf_threshold
    ],
    "ROC-AUC": [
        roc_auc_rf
    ],
    "PR-AUC": [
        pr_auc_rf
    ]
})

results = pd.concat(
    [results, rf_threshold_result],
    ignore_index=True
)

print("\nФАЙНАЛ ТАБЛИЦА:")
print(results.round(4))

results.to_csv(
    "model_results.csv",
    index=False
)
