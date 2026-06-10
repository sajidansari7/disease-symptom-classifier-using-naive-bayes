"""
Evaluation metrics — implemented from scratch using NumPy only.

Functions
---------
accuracy(y_true, y_pred)
precision_recall_f1(y_true, y_pred, average='macro')
confusion_matrix(y_true, y_pred, classes)
classification_report(y_true, y_pred, classes)

Author: Sajid Ansari
"""

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of correctly classified samples."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float((y_true == y_pred).sum() / len(y_true))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """
    Compute confusion matrix.

    Returns
    -------
    np.ndarray, shape (n_classes, n_classes)
        cm[i, j] = number of samples with true class i predicted as class j.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    n = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[class_to_idx[t], class_to_idx[p]] += 1
    return cm


def precision_recall_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: np.ndarray,
    average: str = "macro",
):
    """
    Compute precision, recall, and F1 for each class, then aggregate.

    Parameters
    ----------
    average : str
        'macro'  — unweighted mean across classes.
        'weighted' — weighted by class support.
        'per_class' — returns arrays of shape (n_classes,).

    Returns
    -------
    dict with keys 'precision', 'recall', 'f1'
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    cm = confusion_matrix(y_true, y_pred, classes)

    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    support = cm.sum(axis=1).astype(float)

    precision = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
    recall    = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
    f1        = np.where((precision + recall) > 0,
                         2 * precision * recall / (precision + recall), 0.0)

    if average == "per_class":
        return {"precision": precision, "recall": recall, "f1": f1, "support": support}

    if average == "macro":
        return {
            "precision": float(precision.mean()),
            "recall":    float(recall.mean()),
            "f1":        float(f1.mean()),
        }

    if average == "weighted":
        w = support / support.sum()
        return {
            "precision": float((precision * w).sum()),
            "recall":    float((recall    * w).sum()),
            "f1":        float((f1        * w).sum()),
        }

    raise ValueError(f"Unknown average: {average}")


def classification_report(
    y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray
) -> str:
    """Return a formatted classification report string."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    per_class = precision_recall_f1(y_true, y_pred, classes, average="per_class")
    macro     = precision_recall_f1(y_true, y_pred, classes, average="macro")
    weighted  = precision_recall_f1(y_true, y_pred, classes, average="weighted")

    col_w = max(len(c) for c in classes) + 2
    header = f"{'Class':<{col_w}} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}"
    lines = [header, "-" * len(header)]

    for i, c in enumerate(classes):
        lines.append(
            f"{c:<{col_w}} "
            f"{per_class['precision'][i]:>10.3f} "
            f"{per_class['recall'][i]:>10.3f} "
            f"{per_class['f1'][i]:>10.3f} "
            f"{int(per_class['support'][i]):>10}"
        )

    lines.append("")
    lines.append(
        f"{'macro avg':<{col_w}} "
        f"{macro['precision']:>10.3f} {macro['recall']:>10.3f} {macro['f1']:>10.3f}"
    )
    lines.append(
        f"{'weighted avg':<{col_w}} "
        f"{weighted['precision']:>10.3f} {weighted['recall']:>10.3f} {weighted['f1']:>10.3f}"
    )
    lines.append(f"\nOverall Accuracy: {accuracy(y_true, y_pred):.4f}")
    return "\n".join(lines)
