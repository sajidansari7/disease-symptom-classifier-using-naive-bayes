"""
train.py — Train all three Naive Bayes variants and save the best model.

Usage:
    python train.py

Outputs:
    model.pkl   — serialized best model + metadata
    results.txt — full evaluation report
"""

import pickle
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from my_library import (
    BernoulliNB, MultinomialNB, GaussianNB,
    load_and_prepare,
    train_test_split_stratified,
    accuracy, classification_report,
    precision_recall_f1,
)


def train_and_evaluate():
    print("=" * 60)
    print("  Disease Symptom Classifier — Training Pipeline")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────
    X, y, symptom_cols, classes = load_and_prepare("data/disease_symptom_dataset.csv")
    print(f"\nDataset: {X.shape[0]} samples | {X.shape[1]} symptoms | {len(classes)} diseases")

    # ── Stratified split ───────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split_stratified(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # ── Train all three variants ───────────────────────────────
    models = {
        "BernoulliNB  (α=1.0)": BernoulliNB(alpha=1.0),
        "BernoulliNB  (α=0.5)": BernoulliNB(alpha=0.5),
        "MultinomialNB(α=1.0)": MultinomialNB(alpha=1.0),
        "GaussianNB             ": GaussianNB(),
    }

    results = {}
    print("\n{'Model':<30} {'Accuracy':>10} {'Macro-F1':>10}")
    print("-" * 52)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy(y_test, y_pred)
        metrics = precision_recall_f1(y_test, y_pred, classes, average="macro")
        f1 = metrics["f1"]
        results[name] = {"model": model, "acc": acc, "f1": f1, "y_pred": y_pred}
        print(f"  {name:<30} {acc:>10.4f} {f1:>10.4f}")

    # ── Pick best model (by accuracy) ─────────────────────────
    best_name = max(results, key=lambda n: results[n]["acc"])
    best = results[best_name]
    best_model = best["model"]
    print(f"\nBest model: {best_name.strip()} (Accuracy={best['acc']:.4f})")

    # ── Detailed report for best model ────────────────────────
    report = classification_report(y_test, best["y_pred"], classes)
    print("\n" + report)

    # ── Zero-probability demo ─────────────────────────────────
    print("\n--- Laplace Smoothing Demo ---")
    nb_no_smooth = BernoulliNB(alpha=0.0)
    try:
        nb_no_smooth.fit(X_train, y_train)
        y_demo = nb_no_smooth.predict(X_test[:5])
        print("No-smoothing predictions:", y_demo)
    except Exception as e:
        print(f"Without smoothing error (expected): {e}")

    # ── Top-3 predictions demo ────────────────────────────────
    print("\n--- Top-3 Prediction Demo (first 3 test samples) ---")
    top3 = best_model.top_k_predictions(X_test[:3], k=3)
    for i, preds in enumerate(top3):
        print(f"  Sample {i+1} (True: {y_test[i]})")
        for disease, prob in preds:
            marker = " ✓" if disease == y_test[i] else ""
            print(f"    {disease:<40} {prob:.4f}{marker}")

    # ── Save model ────────────────────────────────────────────
    payload = {
        "model": best_model,
        "model_name": best_name.strip(),
        "symptom_cols": symptom_cols,
        "classes": classes,
        "accuracy": best["acc"],
        "macro_f1": best["f1"],
    }
    with open("model.pkl", "wb") as f:
        pickle.dump(payload, f)
    print(f"\nModel saved to model.pkl")

    # Save text report
    with open("results.txt", "w") as f:
        f.write("Disease Symptom Classifier — Evaluation Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Best Model: {best_name.strip()}\n")
        f.write(f"Test Accuracy: {best['acc']:.4f}\n")
        f.write(f"Macro F1:     {best['f1']:.4f}\n\n")
        f.write("All Models:\n")
        for name, r in results.items():
            f.write(f"  {name:<30}  Acc={r['acc']:.4f}  F1={r['f1']:.4f}\n")
        f.write("\n" + report)
    print("Results saved to results.txt")
    print("\n✓ Training complete.")


if __name__ == "__main__":
    train_and_evaluate()
