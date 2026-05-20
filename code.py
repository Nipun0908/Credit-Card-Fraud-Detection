# CREDIT CARD FRAUD DETECTION


# SECTION 1: IMPORTS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
from sklearn.utils import resample

import warnings
warnings.filterwarnings('ignore')

# SECTION 2: LOAD AND EXPLORE DATA

def load_and_explore(filepath='creditcard.csv'):
    """
    Load the CSV dataset and print basic exploration statistics.
    Returns the raw DataFrame.
    """
    print("=" * 60)
    print("STEP 1: LOADING DATA")
    print("=" * 60)

    df = pd.read_csv(filepath)

    print(f"\nDataset Shape: {df.shape}")
    print(f"\nColumn Names:\n{df.columns.tolist()}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")

    print("\n--- Class Distribution ---")
    class_counts = df['Class'].value_counts()
    print(class_counts)
    print(f"\nFraud percentage: {class_counts[1] / len(df) * 100:.4f}%")

    return df


# SECTION 3: EXPLORATORY DATA ANALYSIS (EDA)

def perform_eda(df):
    """
    Generate visualizations to understand the data better:
    - Class distribution bar chart
    - Transaction amount distribution by class
    - Correlation heatmap of features
    """
    print("\n" + "=" * 60)
    print("STEP 2: EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Plot 1: Class Imbalance ---
    class_counts = df['Class'].value_counts()
    axes[0].bar(['Legit (0)', 'Fraud (1)'], class_counts.values,
                color=['steelblue', 'tomato'], edgecolor='black')
    axes[0].set_title('Class Distribution\n(Severe Imbalance)', fontsize=13)
    axes[0].set_ylabel('Number of Transactions')
    for i, v in enumerate(class_counts.values):
        axes[0].text(i, v + 500, f'{v:,}', ha='center', fontweight='bold')

    # --- Plot 2: Transaction Amount by Class ---
    df[df['Class'] == 0]['Amount'].plot(kind='hist', bins=50, alpha=0.6,
                                         ax=axes[1], label='Legit', color='steelblue')
    df[df['Class'] == 1]['Amount'].plot(kind='hist', bins=50, alpha=0.6,
                                         ax=axes[1], label='Fraud', color='tomato')
    axes[1].set_title('Transaction Amount Distribution\nby Class', fontsize=13)
    axes[1].set_xlabel('Amount ($)')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    axes[1].set_xlim(0, 2500)

    # --- Plot 3: Correlation Heatmap (top features) ---
    top_features = ['V1', 'V2', 'V3', 'V4', 'V5', 'V10', 'V11', 'V12',
                    'V14', 'V17', 'Amount', 'Class']
    corr = df[top_features].corr()
    sns.heatmap(corr, ax=axes[2], cmap='coolwarm', center=0,
                linewidths=0.5, annot=False, cbar=True)
    axes[2].set_title('Feature Correlation Heatmap\n(Selected Features)', fontsize=13)

    plt.tight_layout()
    plt.savefig('eda_plots.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("EDA plots saved as 'eda_plots.png'")


# SECTION 4: DATA PREPROCESSING

def preprocess(df):
    """
    Preprocessing steps:
    1. Scale 'Amount' and 'Time' using StandardScaler
    2. Drop original unscaled columns
    3. Handle class imbalance using under-sampling
    Returns: X_train, X_test, y_train, y_test
    """
    print("\n" + "=" * 60)
    print("STEP 3: DATA PREPROCESSING")
    print("=" * 60)

    # --- Scale 'Amount' and 'Time' ---
    scaler = StandardScaler()
    df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    df.drop(['Amount', 'Time'], axis=1, inplace=True)

    # Rearrange columns for clarity
    scaled_df = df[['scaled_amount', 'scaled_time'] +
                   [c for c in df.columns if c not in ['scaled_amount', 'scaled_time', 'Class']] +
                   ['Class']]

    print(f"Shape after scaling: {scaled_df.shape}")

    # --- Handle Class Imbalance via Under-sampling ---
    legit = scaled_df[scaled_df['Class'] == 0]
    fraud = scaled_df[scaled_df['Class'] == 1]

    print(f"\nBefore balancing — Legit: {len(legit)}, Fraud: {len(fraud)}")

    # Under-sample legit transactions to match fraud count
    legit_under = resample(legit,
                           replace=False,
                           n_samples=len(fraud),
                           random_state=42)

    balanced_df = pd.concat([legit_under, fraud])
    print(f"After balancing  — Legit: {len(legit_under)}, Fraud: {len(fraud)}")
    print(f"Balanced dataset shape: {balanced_df.shape}")

    # --- Train/Test Split ---
    X = balanced_df.drop('Class', axis=1)
    y = balanced_df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain set: {X_train.shape}, Test set: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# SECTION 5: MODEL TRAINING


def train_models(X_train, y_train):
    """
    Train two classifiers:
    - Logistic Regression: simple, interpretable baseline
    - Random Forest: ensemble method, handles non-linearity better
    Returns both trained model objects.
    """
    print("\n" + "=" * 60)
    print("STEP 4: MODEL TRAINING")
    print("=" * 60)

    # --- Logistic Regression ---
    print("\nTraining Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    print("Logistic Regression training complete.")

    # --- Random Forest ---
    print("\nTraining Random Forest Classifier...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    print("Random Forest training complete.")

    return lr_model, rf_model


# SECTION 6: MODEL EVALUATION


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a trained classifier:
    - Prints classification report (precision, recall, F1)
    - Prints ROC-AUC score
    - Plots confusion matrix and ROC curve
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n{'─'*50}")
    print(f"Results: {model_name}")
    print(f"{'─'*50}")
    print(classification_report(y_test, y_pred, target_names=['Legit', 'Fraud']))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'{model_name} — Evaluation', fontsize=14, fontweight='bold')

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Legit', 'Fraud'],
                yticklabels=['Legit', 'Fraud'])
    axes[0].set_title('Confusion Matrix')
    axes[0].set_ylabel('Actual')
    axes[0].set_xlabel('Predicted')

    # --- ROC Curve ---
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    axes[1].plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC Curve (AUC = {auc:.4f})')
    axes[1].plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve')
    axes[1].legend(loc='lower right')

    plt.tight_layout()
    filename = f"{model_name.replace(' ', '_').lower()}_evaluation.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Evaluation plots saved as '{filename}'")


def evaluate_both(lr_model, rf_model, X_test, y_test):
    """
    Run evaluation on both trained models.
    """
    print("\n" + "=" * 60)
    print("STEP 5: MODEL EVALUATION")
    print("=" * 60)

    evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    evaluate_model(rf_model, X_test, y_test, "Random Forest")

# SECTION 7: FEATURE IMPORTANCE

def plot_feature_importance(rf_model, feature_names):
    """
    Plot the top 15 most important features as determined by the
    Random Forest's built-in feature importance scores.
    """
    print("\n" + "=" * 60)
    print("STEP 6: FEATURE IMPORTANCE (Random Forest)")
    print("=" * 60)

    importances = rf_model.feature_importances_
    feat_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(15)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=feat_df, x='Importance', y='Feature',
                palette='viridis', edgecolor='black')
    plt.title('Top 15 Feature Importances (Random Forest)', fontsize=14)
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Feature importance plot saved as 'feature_importance.png'")
    print(f"\nTop 5 Features:\n{feat_df.head()}")


# SECTION 8: MAIN PIPELINE

if __name__ == "__main__":
    # Step 1 — Load data
    df = load_and_explore('creditcard.csv')

    # Step 2 — EDA
    perform_eda(df)

    # Step 3 — Preprocess (scale + balance + split)
    X_train, X_test, y_train, y_test = preprocess(df)

    # Step 4 — Train models
    lr_model, rf_model = train_models(X_train, y_train)

    # Step 5 — Evaluate models
    evaluate_both(lr_model, rf_model, X_test, y_test)

    # Step 6 — Feature importance
    plot_feature_importance(rf_model, X_train.columns.tolist())

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
