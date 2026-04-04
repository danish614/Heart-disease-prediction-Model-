"""
Heart Disease Prediction - Complete Training Pipeline
=====================================================
This script handles:
1. Data Loading & Cleaning
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Model Training (Logistic Regression, Random Forest, SVM, XGBoost)
5. Cross-Validation
6. Model Evaluation (Confusion Matrix, Classification Report, ROC Curve)
7. Feature Importance
8. Save Best Model
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_score, recall_score, f1_score
)
import warnings
warnings.filterwarnings('ignore')

DATASET_PATH = r"C:\Users\Danish Ilyas\heart_disease_uci.csv"
SAVE_DIR = r"C:\Users\Danish Ilyas\Downloads"


# ══════════════════════════════════════════════════════════
# 1. DATA LOADING & CLEANING
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("  STEP 1: DATA LOADING & CLEANING")
print("=" * 60)

df = pd.read_csv(DATASET_PATH)
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")

# Drop unnecessary columns
df = df.drop(columns=["id", "dataset"])

# Rename columns for consistency
df = df.rename(columns={"thalch": "thalach", "num": "target"})

# Convert target to binary (0 = no disease, 1+ = disease)
df["target"] = (df["target"] > 0).astype(int)
print(f"\nTarget distribution:")
print(f"  No Disease (0): {(df['target'] == 0).sum()}")
print(f"  Disease (1):    {(df['target'] == 1).sum()}")

# Handle missing values
print(f"\nMissing values before cleaning:")
missing = df.isnull().sum()
print(missing[missing > 0].to_string())

# Fill numeric columns with median
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

# Fill categorical columns with mode
cat_cols = df.select_dtypes(include=["object", "bool"]).columns
for col in cat_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

print(f"\nMissing values after cleaning: {df.isnull().sum().sum()}")
print(f"Final dataset shape: {df.shape}")


# ══════════════════════════════════════════════════════════
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 2: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# --- Figure 1: Target Distribution ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

df["target"].value_counts().plot(kind="bar", ax=axes[0], color=["#2ecc71", "#e74c3c"])
axes[0].set_title("Target Distribution")
axes[0].set_xticklabels(["No Disease", "Disease"], rotation=0)
axes[0].set_ylabel("Count")

df["age"].hist(bins=20, ax=axes[1], color="#3498db", edgecolor="black")
axes[1].set_title("Age Distribution")
axes[1].set_xlabel("Age")

plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/eda_target_age.png", dpi=150, bbox_inches="tight")
plt.show()

# --- Figure 2: Numeric Features Distribution ---
numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
fig, axes = plt.subplots(1, len(numeric_features), figsize=(20, 4))

for i, col in enumerate(numeric_features):
    df.boxplot(column=col, by="target", ax=axes[i])
    axes[i].set_title(col)
    axes[i].set_xlabel("Target")

plt.suptitle("Numeric Features by Target", y=1.02, fontsize=14)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/eda_boxplots.png", dpi=150, bbox_inches="tight")
plt.show()

# --- Figure 3: Correlation Heatmap ---
# Encode categorical columns first for correlation
df_encoded = df.copy()
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    label_encoders[col] = le

plt.figure(figsize=(12, 10))
corr = df_encoded.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, square=True)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/eda_correlation.png", dpi=150, bbox_inches="tight")
plt.show()

# --- Figure 4: Categorical Features vs Target ---
cat_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for i, col in enumerate(cat_features):
    pd.crosstab(df[col], df["target"]).plot(kind="bar", ax=axes[i],
                                             color=["#2ecc71", "#e74c3c"])
    axes[i].set_title(f"{col} vs Target")
    axes[i].set_xlabel("")
    axes[i].legend(["No Disease", "Disease"], fontsize=8)
    axes[i].tick_params(axis='x', rotation=45)

axes[-1].axis("off")
plt.suptitle("Categorical Features vs Target", fontsize=14)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/eda_categorical.png", dpi=150, bbox_inches="tight")
plt.show()

print("EDA plots saved!")


# ══════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 3: FEATURE ENGINEERING")
print("=" * 60)

# Encode categorical features
cp_map = {"typical angina": 0, "atypical angina": 1, "non-anginal": 2, "asymptomatic": 3}
restecg_map = {"normal": 0, "st-t abnormality": 1, "lv hypertrophy": 2}
slope_map = {"upsloping": 0, "flat": 1, "downsloping": 2}
thal_map = {"normal": 1, "fixed defect": 2, "reversable defect": 3}

df["sex"] = df["sex"].map({"Male": 1, "Female": 0})
df["cp"] = df["cp"].map(cp_map)
df["restecg"] = df["restecg"].map(restecg_map)
df["slope"] = df["slope"].map(slope_map)
df["thal"] = df["thal"].map(thal_map)
df["fbs"] = df["fbs"].astype(int)
df["exang"] = df["exang"].astype(int)

# Create interaction features
df["age_chol"] = df["age"] * df["chol"]
df["trestbps_thalach_ratio"] = df["trestbps"] / (df["thalach"] + 1)
df["oldpeak_slope"] = df["oldpeak"] * df["slope"]
df["age_trestbps"] = df["age"] * df["trestbps"]
df["chol_thalach_ratio"] = df["chol"] / (df["thalach"] + 1)

print(f"Features after engineering: {df.shape[1] - 1}")
print(f"New features: age_chol, trestbps_thalach_ratio, oldpeak_slope, age_trestbps, chol_thalach_ratio")


# ══════════════════════════════════════════════════════════
# 4. TRAIN / TEST SPLIT & SCALING
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 4: TRAIN / TEST SPLIT")
print("=" * 60)

X = df.drop(columns=["target"])
y = df["target"]

feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set:     {X_test.shape[0]} samples")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ══════════════════════════════════════════════════════════
# 5. MODEL TRAINING & CROSS-VALIDATION
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 5: MODEL TRAINING & CROSS-VALIDATION")
print("=" * 60)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                              use_label_encoder=False, eval_metric="logloss", random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

print(f"\n{'Model':<25} {'CV Accuracy':<15} {'Test Accuracy':<15} {'Precision':<12} {'Recall':<12} {'F1':<12}")
print("-" * 90)

for name, model in models.items():
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="accuracy")

    # Train on full training set
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results[name] = {
        "model": model,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "test_acc": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "y_pred": y_pred,
    }

    print(f"{name:<25} {cv_scores.mean():.4f}±{cv_scores.std():.4f}  {acc:<15.4f} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f}")


# ══════════════════════════════════════════════════════════
# 6. MODEL EVALUATION
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 6: MODEL EVALUATION")
print("=" * 60)

# --- Figure 5: Model Comparison ---
model_names = list(results.keys())
test_accs = [results[m]["test_acc"] for m in model_names]
cv_accs = [results[m]["cv_mean"] for m in model_names]
f1_scores = [results[m]["f1"] for m in model_names]

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(model_names))
width = 0.25

bars1 = ax.bar(x - width, cv_accs, width, label="CV Accuracy", color="#3498db")
bars2 = ax.bar(x, test_accs, width, label="Test Accuracy", color="#2ecc71")
bars3 = ax.bar(x + width, f1_scores, width, label="F1 Score", color="#e74c3c")

ax.set_ylabel("Score")
ax.set_title("Model Comparison")
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=15)
ax.legend()
ax.set_ylim(0, 1.1)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)

plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# --- Figure 6: Confusion Matrices ---
fig, axes = plt.subplots(1, 4, figsize=(20, 4))

for i, (name, res) in enumerate(results.items()):
    cm = confusion_matrix(y_test, res["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"])
    axes[i].set_title(f"{name}\nAcc: {res['test_acc']:.3f}")
    axes[i].set_ylabel("Actual")
    axes[i].set_xlabel("Predicted")

plt.suptitle("Confusion Matrices", fontsize=14)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.show()

# --- Figure 7: ROC Curves ---
plt.figure(figsize=(8, 6))

for name, res in results.items():
    model = res["model"]
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {roc_auc:.3f})")

plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Model Comparison")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/roc_curves.png", dpi=150, bbox_inches="tight")
plt.show()

# Classification Reports
print("\nDetailed Classification Reports:")
for name, res in results.items():
    print(f"\n--- {name} ---")
    print(classification_report(y_test, res["y_pred"], target_names=["No Disease", "Disease"]))


# ══════════════════════════════════════════════════════════
# 7. BEST MODEL SELECTION & FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 7: BEST MODEL & FEATURE IMPORTANCE")
print("=" * 60)

# Select best model by F1 score
best_name = max(results, key=lambda m: results[m]["f1"])
best_model = results[best_name]["model"]
best_result = results[best_name]

print(f"\nBest Model: {best_name}")
print(f"  CV Accuracy:   {best_result['cv_mean']:.4f} ± {best_result['cv_std']:.4f}")
print(f"  Test Accuracy: {best_result['test_acc']:.4f}")
print(f"  Precision:     {best_result['precision']:.4f}")
print(f"  Recall:        {best_result['recall']:.4f}")
print(f"  F1 Score:      {best_result['f1']:.4f}")

# Feature importance
if hasattr(best_model, "feature_importances_"):
    importance = best_model.feature_importances_
elif hasattr(best_model, "coef_"):
    importance = np.abs(best_model.coef_[0])
else:
    importance = np.zeros(len(feature_names))

feature_importance = dict(zip(feature_names, importance))

# --- Figure 8: Feature Importance ---
sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
feat_names = [f[0] for f in sorted_features]
feat_values = [f[1] for f in sorted_features]

plt.figure(figsize=(10, 8))
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(feat_names)))
plt.barh(range(len(feat_names)), feat_values, color=colors)
plt.yticks(range(len(feat_names)), feat_names)
plt.xlabel("Importance")
plt.title(f"Feature Importance ({best_name})")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()


# ══════════════════════════════════════════════════════════
# 8. SAVE MODEL & ARTIFACTS
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 8: SAVING MODEL")
print("=" * 60)

# Save all models
for name, res in results.items():
    safe_name = name.lower().replace(" ", "_")
    joblib.dump(res["model"], f"{SAVE_DIR}/{safe_name}_model.pkl")
    print(f"  Saved: {safe_name}_model.pkl")

# Save best model separately
joblib.dump(best_model, f"{SAVE_DIR}/best_model.pkl")
joblib.dump(scaler, f"{SAVE_DIR}/scaler.pkl")
joblib.dump(feature_names, f"{SAVE_DIR}/feature_names.pkl")
joblib.dump(feature_importance, f"{SAVE_DIR}/feature_importance.pkl")

# Save model comparison results
comparison = pd.DataFrame({
    "Model": model_names,
    "CV Accuracy": [f"{results[m]['cv_mean']:.4f}±{results[m]['cv_std']:.4f}" for m in model_names],
    "Test Accuracy": [results[m]["test_acc"] for m in model_names],
    "Precision": [results[m]["precision"] for m in model_names],
    "Recall": [results[m]["recall"] for m in model_names],
    "F1 Score": [results[m]["f1"] for m in model_names],
})
comparison.to_csv(f"{SAVE_DIR}/model_comparison.csv", index=False)

print(f"\nBest model saved: best_model.pkl ({best_name})")
print(f"Scaler saved: scaler.pkl")
print(f"Feature names saved: feature_names.pkl")
print(f"Feature importance saved: feature_importance.pkl")
print(f"Model comparison saved: model_comparison.csv")

print("\n" + "=" * 60)
print("  TRAINING COMPLETE!")
print("=" * 60)
print(f"\nBest Model: {best_name}")
print(f"Test Accuracy: {best_result['test_acc']*100:.2f}%")
print(f"F1 Score: {best_result['f1']*100:.2f}%")
print(f"\nAb frontend chalane ke liye run karein:")
print(f"  streamlit run {SAVE_DIR}\\heart_disease_project.py")
