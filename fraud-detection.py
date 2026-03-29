
"""
Real-Time Fraud Detection System
Dataset: Kaggle Credit Card Fraud Detection
Tech: Python, scikit-learn, pandas, Flask, Streamlit
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc,
)
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import joblib
import os
import json
import warnings

warnings.filterwarnings("ignore")

# Set style
sns.set_style("darkgrid")
plt.rcParams["figure.figsize"] = (12, 6)


# ==================== CONFIG & DIRECTORIES ====================
def create_directories():
    """Create required directories"""
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("config", exist_ok=True)
    print("✓ Directories created: models/, outputs/, config/")


def save_model_params():
    """Save model hyperparameters to config/model_params.json"""
    model_params = {
        "random_forest": {
            "n_estimators": 100,
            "max_depth": 15,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "random_state": 42,
            "n_jobs": -1,
            "class_weight": "balanced",
        },
        "isolation_forest": {"contamination": 0.01, "random_state": 42, "n_jobs": -1},
        "preprocessing": {
            "test_size": 0.2,
            "random_state": 42,
            "smote": {"k_neighbors": 5, "random_state": 42},
        },
    }
    with open("config/model_params.json", "w") as f:
        json.dump(model_params, f, indent=4)
    print("✓ Hyperparameters saved to config/model_params.json")


# ==================== DATA LOADING & EDA ====================
def load_and_explore_data(file_path):
    """Load and explore fraud dataset"""
    df = pd.read_csv(file_path)

    print("=" * 60)
    print("FRAUD DETECTION SYSTEM - DATA EXPLORATION")
    print("=" * 60)
    print(f"\nDataset Shape: {df.shape}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")

    # Class distribution
    fraud_count = len(df[df["Class"] == 1])
    valid_count = len(df[df["Class"] == 0])
    fraud_percentage = (fraud_count / len(df)) * 100

    print(f"\nClass Distribution:")
    print(f" Fraudulent Cases: {fraud_count} ({fraud_percentage:.2f}%)")
    print(f" Valid Transactions: {valid_count} ({100 - fraud_percentage:.2f}%)")

    # Visualize class imbalance
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Pie chart
    class_counts = df["Class"].value_counts()
    axes[0].pie(
        class_counts,
        labels=["Valid", "Fraud"],
        autopct="%1.2f%%",
        colors=["#2ecc71", "#e74c3c"],
        startangle=90,
    )
    axes[0].set_title("Class Distribution (Imbalanced)", fontsize=12, fontweight="bold")

    # Count plot
    df["Class"].value_counts().plot(
        kind="bar", ax=axes[1], color=["#2ecc71", "#e74c3c"]
    )
    axes[1].set_title("Fraud vs Valid Transactions", fontsize=12, fontweight="bold")
    axes[1].set_xticklabels(["Valid (0)", "Fraud (1)"], rotation=0)
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig("outputs/class_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Class distribution saved to outputs/class_distribution.png")

    return df


# ==================== DATA PREPROCESSING ====================
def preprocess_data(df):
    """Preprocess and prepare data for modeling"""
    print("\n" + "=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)

    # Separate features and target
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame
    X_train_scaled = pd.DataFrame(
        X_train_scaled, columns=X.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

    print("✓ Features scaled using StandardScaler")

    # Save scaler
    joblib.dump(scaler, "models/scaler.pkl")
    print("✓ Scaler saved as 'models/scaler.pkl'")

    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

    print(f"✓ SMOTE applied. New training set size: {X_train_smote.shape}")
    print(f" Fraudulent: {sum(y_train_smote == 1)}, Valid: {sum(y_train_smote == 0)}")

    return X_train_smote, X_test_scaled, y_train_smote, y_test, scaler


# ==================== FEATURE IMPORTANCE ====================
def plot_feature_importance(model, X_train, top_n=10):
    """Plot and save top feature importances"""
    feature_importance = (
        pd.DataFrame(
            {"Feature": X_train.columns, "Importance": model.feature_importances_}
        )
        .sort_values("Importance", ascending=False)
        .head(top_n)
    )

    plt.figure(figsize=(10, 6))
    plt.barh(
        feature_importance["Feature"], feature_importance["Importance"], color="#3498db"
    )
    plt.title("Top 10 Feature Importances", fontweight="bold")
    plt.xlabel("Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Feature importance saved to outputs/feature_importance.png")
    return feature_importance


# ==================== MODEL TRAINING ====================
def train_random_forest(X_train, X_test, y_train, y_test):
    """Train Random Forest model"""
    print("\n" + "=" * 60)
    print("TRAINING RANDOM FOREST MODEL")
    print("=" * 60)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    print("\nTraining model...")
    model.fit(X_train, y_train)
    print("✓ Model trained successfully")

    # Feature importance plot
    feature_importance = plot_feature_importance(model, pd.DataFrame(X_train))

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Evaluation metrics
    print("\n" + "-" * 60)
    print("MODEL PERFORMANCE METRICS")
    print("-" * 60)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Valid", "Fraud"]))

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC-AUC Score: {roc_auc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f" True Negatives: {cm[0, 0]}")
    print(f" False Positives: {cm[0, 1]}")
    print(f" False Negatives: {cm[1, 0]}")
    print(f" True Positives: {cm[1, 1]}")

    # Calculate Precision & Recall
    precision = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0
    recall = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
    print(f"\nPrecision: {precision:.4f} (How many flagged are actually fraud)")
    print(f"Recall: {recall:.4f} (How many frauds are caught)")

    # Comprehensive performance plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Confusion Matrix
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[0, 0],
        xticklabels=["Valid", "Fraud"],
        yticklabels=["Valid", "Fraud"],
    )
    axes[0, 0].set_title("Confusion Matrix", fontweight="bold")
    axes[0, 0].set_ylabel("True Label")
    axes[0, 0].set_xlabel("Predicted Label")

    # Precision-Recall Curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall_curve, precision_curve)
    axes[1, 0].plot(
        recall_curve,
        precision_curve,
        label=f"PR Curve (AUC={pr_auc:.3f})",
        linewidth=2,
        color="#e74c3c",
    )
    axes[1, 0].fill_between(recall_curve, precision_curve, alpha=0.2, color="#e74c3c")
    axes[1, 0].set_xlabel("Recall")
    axes[1, 0].set_ylabel("Precision")
    axes[1, 0].set_title("Precision-Recall Curve", fontweight="bold")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Prediction Distribution
    axes[1, 1].hist(
        y_pred_proba[y_test == 0],
        bins=50,
        label="Valid (Actual)",
        alpha=0.7,
        color="#2ecc71",
    )
    axes[1, 1].hist(
        y_pred_proba[y_test == 1],
        bins=50,
        label="Fraud (Actual)",
        alpha=0.7,
        color="#e74c3c",
    )
    axes[1, 1].set_xlabel("Fraud Probability")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("Distribution of Predicted Probabilities", fontweight="bold")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig("outputs/model_performance.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Save model
    joblib.dump(model, "models/fraud_detection_model.pkl")
    print("\n✓ Model saved as 'models/fraud_detection_model.pkl'")

    return model, roc_auc


# ==================== ISOLATION FOREST (Anomaly Detection) ====================
def train_isolation_forest(X_train, X_test, y_test):
    """Alternative: Anomaly-based fraud detection"""
    print("\n" + "=" * 60)
    print("TRAINING ISOLATION FOREST (ANOMALY DETECTION)")
    print("=" * 60)

    iso_forest = IsolationForest(
        contamination=0.01, random_state=42, n_jobs=-1  # Estimate 1% fraud rate
    )
    iso_forest.fit(X_train)
    y_pred_iso = iso_forest.predict(X_test)
    # Convert -1 (anomaly) to 1 (fraud)
    y_pred_iso = np.where(y_pred_iso == -1, 1, 0)

    print("\nIsolation Forest Performance:")
    print(classification_report(y_test, y_pred_iso, target_names=["Valid", "Fraud"]))

    joblib.dump(iso_forest, "models/isolation_forest_model.pkl")
    print("✓ Isolation Forest saved as 'models/isolation_forest_model.pkl'")
    return iso_forest


# ==================== REAL-TIME PREDICTION ====================
def predict_fraud(transaction, model, scaler):
    """Predict if a transaction is fraudulent"""
    # Scale the transaction
    transaction_scaled = scaler.transform([transaction])

    # Get prediction and probability
    prediction = model.predict(transaction_scaled)[0]
    probability = model.predict_proba(transaction_scaled)[0]

    result = {
        "is_fraud": bool(prediction == 1),
        "fraud_probability": probability[1],
        "confidence": max(probability),
    }
    return result


# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    # Download dataset from: https://www.kaggle.com/mlg-ulb/creditcardfraud
    # Place in same directory as 'creditcard.csv'

    # Create directories first
    create_directories()
    save_model_params()

    try:
        # Load data
        df = load_and_explore_data("creditcard.csv")

        # Preprocess
        X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

        # Train Random Forest
        model, roc_auc = train_random_forest(X_train, X_test, y_train, y_test)

        # Train Isolation Forest (alternative)
        iso_forest = train_isolation_forest(X_train, X_test, y_test)

        # Example: Predict on new transaction
        print("\n" + "=" * 60)
        print("REAL-TIME PREDICTION EXAMPLE")
        print("=" * 60)

        # Get sample transactions
        valid_transaction = X_test.iloc[0].values
        fraud_indices = y_test[y_test == 1].index
        fraud_transaction = (
            X_test.loc[fraud_indices[0]].values if len(fraud_indices) > 0 else None
        )

        result_valid = predict_fraud(valid_transaction, model, scaler)
        print(f"\nValid Transaction Prediction:")
        print(f" Is Fraud: {result_valid['is_fraud']}")
        print(f" Fraud Probability: {result_valid['fraud_probability']:.4f}")
        print(f" Confidence: {result_valid['confidence']:.4f}")

        if fraud_transaction is not None:
            result_fraud = predict_fraud(fraud_transaction, model, scaler)
            print(f"\nFraud Transaction Prediction:")
            print(f" Is Fraud: {result_fraud['is_fraud']}")
            print(f" Fraud Probability: {result_fraud['fraud_probability']:.4f}")
            print(f" Confidence: {result_fraud['confidence']:.4f}")

        print("\n" + "=" * 60)
        print("🎉 FRAUD DETECTION SYSTEM COMPLETE!")
        print("Generated files:")
        print("- config/model_params.json")
        print("- models/fraud_detection_model.pkl")
        print("- models/scaler.pkl")
        print("- outputs/class_distribution.png")
        print("- outputs/feature_importance.png")
        print("- outputs/model_performance.png")
        print("=" * 60)

    except FileNotFoundError:
        print("❌ Error: 'creditcard.csv' not found!")
        print("Download from: https://www.kaggle.com/mlg-ulb/creditcardfraud")
    except KeyboardInterrupt:
        print(
            "\n⏹️ Process interrupted. Check generated files in models/, outputs/, config/"
        )
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("Files generated so far are still available!")
