"""
Preprocessing utilities — from scratch.

Functions
---------
train_test_split_stratified   : stratified split preserving class balance
encode_labels                 : string labels → integer array + decoder
load_and_prepare              : load CSV, return (X, y, symptom_columns, label_encoder)

Author: Sajid Ansari
"""

import numpy as np
import pandas as pd
from collections import defaultdict


def train_test_split_stratified(X, y, test_size=0.2, random_state=42):
    """
    Stratified train/test split — preserves class proportions.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
    y : np.ndarray
    test_size : float
    random_state : int

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    rng = np.random.RandomState(random_state)
    X = np.array(X)
    y = np.array(y)

    classes, class_counts = np.unique(y, return_counts=True)
    train_idx, test_idx = [], []

    for c in classes:
        c_idx = np.where(y == c)[0]
        rng.shuffle(c_idx)
        n_test = max(1, int(len(c_idx) * test_size))
        test_idx.extend(c_idx[:n_test].tolist())
        train_idx.extend(c_idx[n_test:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    return (
        X[train_idx], X[test_idx],
        y[train_idx], y[test_idx],
    )


def encode_labels(y):
    """
    Encode string labels to integers.

    Returns
    -------
    y_encoded : np.ndarray of int
    classes   : np.ndarray of original class strings (decoder: classes[i] → label)
    """
    classes = np.unique(y)
    label_to_idx = {c: i for i, c in enumerate(classes)}
    y_encoded = np.array([label_to_idx[label] for label in y])
    return y_encoded, classes


def load_and_prepare(csv_path: str):
    """
    Load the disease-symptom CSV and return model-ready arrays.

    Returns
    -------
    X             : np.ndarray, shape (n_samples, n_symptoms) — binary features
    y             : np.ndarray of str — disease labels
    symptom_cols  : list of str — ordered symptom column names
    classes       : np.ndarray — unique disease names (label decoder)
    """
    df = pd.read_csv(csv_path)
    symptom_cols = [c for c in df.columns if c != "prognosis"]
    X = df[symptom_cols].values.astype(np.float64)
    y = df["prognosis"].values
    _, classes = encode_labels(y)
    return X, y, symptom_cols, classes
