# 🩺 Disease Symptom Classifier

> A multi-class disease prediction system built with **Bernoulli Naive Bayes from scratch** — no scikit-learn for the core algorithm. Trained on 41 diseases and 131 symptoms. Deployed as an interactive Streamlit web app.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://disease-symptom-classifier-using-naive-bayes-vqyzwsc8nxbsadvzu.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![NumPy](https://img.shields.io/badge/Built%20with-NumPy%20only-013243)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📸 Demo

| Symptom Selection | Prediction Results |
|---|---|
| Multi-select symptoms from 131 options | Top-5 differential diagnoses with confidence scores |

---

## 🎯 What This Project Does

Given a set of symptoms (e.g., fever + headache + chills + nausea), the model predicts the most likely disease and provides a ranked list of the top 5 differential diagnoses with probability scores.

**Why Naive Bayes for this task?**
- Symptoms are binary features (present/absent) → perfect for Bernoulli NB
- Fast inference — no iterative optimisation needed
- Naturally probabilistic → gives calibrated confidence scores
- Interpretable — each symptom's contribution is traceable

---

## 📊 Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| **BernoulliNB (α=1.0)** ← **best** | **99.90%** | **99.90%** |
| BernoulliNB (α=0.5) | 99.90% | 99.90% |
| MultinomialNB (α=1.0) | 99.59% | 99.59% |
| GaussianNB | 76.83% | 76.62% |

Evaluated on a held-out 20% stratified test set (984 samples).  
BernoulliNB outperforms GaussianNB because the features are inherently binary — Gaussian's continuous-distribution assumption is a poor fit.

---

## 🧠 Algorithm — Bernoulli Naive Bayes

### Core Formula (log-space)

```
log P(disease | x) ∝ log P(disease)
                   + Σⱼ [ xⱼ · log P(xⱼ=1 | disease)
                         + (1-xⱼ) · log(1 - P(xⱼ=1 | disease)) ]
```

### Feature Likelihood Estimation (with Laplace Smoothing)

```
P̂(xⱼ=1 | c) = (Nⱼc + α) / (Nc + 2α)
```

Where:
- `Nⱼc` = number of training samples in class `c` where symptom `j` = 1
- `Nc`  = total training samples in class `c`
- `α`   = smoothing parameter (1.0 in this project)

### Why Log-Space?

With 131 features, naively multiplying 131 probabilities risks **float64 underflow**.  
If each P(xⱼ|c) ≈ 0.05, the product ≈ 10⁻¹⁷⁸ — below representable range.  
Converting to log-space turns multiplication into addition, completely avoiding this.

### Why Laplace Smoothing?

Without smoothing, any symptom that never appears with a disease in training data gives P=0, making the **entire posterior 0** regardless of other evidence.  
Laplace smoothing adds α=1 pseudo-count to every feature, guaranteeing no zero probabilities.

---

## 🗂️ Project Structure

```
disease-symptom-classifier/
│
├── my_library/                  ← Custom ML library (NumPy only)
│   ├── __init__.py
│   ├── naive_bayes.py           ← BernoulliNB, MultinomialNB, GaussianNB
│   ├── preprocessing.py         ← Stratified split, label encoding
│   └── metrics.py               ← Accuracy, confusion matrix, F1
│
├── data/
│   ├── disease_symptom_dataset.csv   ← 4,920 samples, 131 symptoms, 41 diseases
│   └── generate_dataset.py           ← Dataset generation script
│
├── notebooks/
│   └── 01_eda_and_training.ipynb    ← Full EDA + training walkthrough
│
├── app.py                       ← Streamlit web application
├── train.py                     ← Training script → saves model.pkl
├── model.pkl                    ← Serialized trained model
├── results.txt                  ← Full evaluation report
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/sajidansari7/disease-symptom-classifier.git
cd disease-symptom-classifier
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Regenerate dataset

```bash
python data/generate_dataset.py
```

### 4. Train the model

```bash
python train.py
```

This prints a full evaluation report and saves `model.pkl`.

### 5. Run the Streamlit app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ☁️ Deployment (Streamlit Cloud)

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set `app.py` as the main file
4. Click **Deploy** — done!

> **Note:** `model.pkl` must be committed to the repo for the deployed app to work.  
> Run `python train.py` locally first, then `git add model.pkl && git commit`.

---

## 📐 Implementation Details

### `my_library/naive_bayes.py`

Three Naive Bayes classifiers, all from scratch:

| Class | Best for | Key parameter |
|---|---|---|
| `BernoulliNB` | Binary features (✓ this project) | `alpha` — Laplace smoothing |
| `MultinomialNB` | Count/TF-IDF features | `alpha` — Laplace smoothing |
| `GaussianNB` | Continuous features | `var_smoothing` — numerical stability |

All three implement:
- `fit(X, y)` — estimate priors and likelihoods
- `predict(X)` — return class labels
- `predict_proba(X)` — return calibrated probabilities
- `top_k_predictions(X, k)` — return top-k (disease, probability) pairs

### `my_library/metrics.py`

All from scratch using only NumPy:
- `accuracy(y_true, y_pred)`
- `confusion_matrix(y_true, y_pred, classes)`
- `precision_recall_f1(y_true, y_pred, classes, average)` — supports `'macro'`, `'weighted'`, `'per_class'`
- `classification_report(y_true, y_pred, classes)` — formatted table

### `my_library/preprocessing.py`

- `train_test_split_stratified` — custom stratified split preserving class proportions
- `encode_labels` — string labels → integer array + decoder
- `load_and_prepare` — full data loading pipeline

---

## 📋 Dataset

- **4,920 samples** across **41 diseases** (120 samples each — balanced)
- **131 binary symptom features** (1 = present, 0 = absent)
- Diseases covered: Common Cold, Malaria, Dengue, Typhoid, Tuberculosis, Diabetes, Hypertension, Hepatitis variants, Heart attack, and 31 more
- Based on the structure of the [Kaggle Disease Symptom Prediction Dataset](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset)

---

## 💡 Key Interview Talking Points

**Q: Why did you choose Bernoulli NB over Multinomial NB?**  
A: Each symptom is binary (present or absent) — the Bernoulli model explicitly models P(x=0|class) and P(x=1|class) separately, which is more appropriate than Multinomial's count-based approach. This is reflected in the results: BernoulliNB achieves 99.9% vs MultinomialNB's 99.6%.

**Q: What is Laplace smoothing and why is it critical here?**  
A: Without smoothing, if symptom j never appears with disease c in training, P(xⱼ|c) = 0, making the entire posterior 0 regardless of all other symptoms. Laplace smoothing adds a pseudo-count α to every feature, guaranteeing no probability collapses to zero. Setting α=1 is the standard "add-one" smoothing.

**Q: Why log-space computation?**  
A: With 131 features, multiplying 131 probabilities risks float64 underflow (numbers approaching 10⁻¹⁸⁰). Log-space converts products to sums, completely avoiding underflow while preserving the argmax (we only need relative values to pick the best class).

**Q: How does stratified splitting work?**  
A: For each class independently, shuffle the samples and allocate `test_size` fraction to test. This preserves class proportions in both splits, which is critical when classes are balanced — a uniform random split could accidentally under-represent some diseases.

---

## 🔗 Related Projects

- [House Price Prediction](https://github.com/sajidansari7/house-price-prediction) — Gradient Descent variants from scratch on Ames Housing dataset
- [Credit Risk Scoring Engine](https://github.com/sajidansari7/credit-risk-scoring) — Logistic Regression + PDO-based credit scorecard

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<div align="center">
  <strong>Built by Sajid Ansari</strong> · BTech CSE · ML Portfolio Project<br>
  <a href="https://github.com/sajidansari7">GitHub</a> ·
  <a href="https://linkedin.com/in/sajidansari7">LinkedIn</a>
</div>
