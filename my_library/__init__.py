"""
my_library — Disease Symptom Classifier
========================================
Naive Bayes classifiers and utilities built from scratch (NumPy only).

Modules
-------
naive_bayes   : BernoulliNB, MultinomialNB, GaussianNB
preprocessing : data loading, label encoding, stratified split
metrics       : accuracy, confusion matrix, precision/recall/F1
"""

from .naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from .preprocessing import train_test_split_stratified, encode_labels, load_and_prepare
from .metrics import accuracy, confusion_matrix, precision_recall_f1, classification_report
""" the above lines go into the modules and bring the submodules to the my_library level so no need to give full path directly import from my_library level"""
__all__ = [
    "BernoulliNB", "MultinomialNB", "GaussianNB",
    "train_test_split_stratified", "encode_labels", "load_and_prepare",
    "accuracy", "confusion_matrix", "precision_recall_f1", "classification_report",
]
