from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def evaluate_classifier(name, model, X_train, X_test, y_train, y_test):
    """
    Fit a classifier and print key metrics in a consistent format.
    Also plots the confusion matrix.
    """
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    print(f"\n=== {name} ===")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
    print(f"F1-score:  {f1_score(y_test, y_pred):.3f}")

    if y_proba is not None:
        print(f"ROC AUC:  {roc_auc_score(y_test, y_proba):.3f}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.show()
