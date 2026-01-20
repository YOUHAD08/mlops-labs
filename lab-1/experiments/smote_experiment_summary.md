# Experiment: SMOTE vs class_weight for Class Imbalance

## 📋 Experiment Metadata

- **Date:** 1/20/2026
- **Branch:** feature/smote-comparison
- **Objective:** Compare SMOTE and class_weight approaches for handling imbalanced data
- **Dataset:** Bank Customer Churn (10,000 samples, 20% minority class)
- **Metric Focus:** F1 Score (more important than accuracy for imbalanced data)

---

## 🔬 Experiment Setup

### Baseline Configuration (Method 1)

```yaml
model:
  name: "logistic_regression"
  max_iter: 3000
  class_weight: "balanced"
```

- No synthetic data generation
- Model internally weights minority class higher

### SMOTE Configuration (Method 2)

```yaml
model:
  name: "logistic_regression"
  max_iter: 3000
  class_weight: null
```

- SMOTE applied after preprocessing
- Balanced minority class: 1629 → 6361 samples
- Synthetic samples created using k-nearest neighbors

---

## 📊 Results

| Method       | Accuracy | F1 Score | Training Samples (Class 1) |
| ------------ | -------- | -------- | -------------------------- |
| class_weight | 0.7102   | 0.4908   | 1629 (original)            |
| SMOTE        | 0.7142   | 0.4897   | 6361 (with synthetic)      |

### Winner: class_weight="balanced"

- **Reason:** Higher F1 score, simpler implementation, more efficient
- **Performance Gain:** +0.0011 F1 points

---

## 💡 Key Findings

### What Worked Well:

1. class_weight="balanced" achieved slightly better F1 score with simpler implementation
2. Both methods successfully handled class imbalance better than naive approach
3. SMOTE increased accuracy but slightly decreased F1 score

### What I Learned:

1. F1 score is crucial for imbalanced datasets
2. Simpler methods (class_weight) can outperform more complex approaches (SMOTE)
3. Synthetic data generation doesn't always guarantee better minority class prediction
4. The performance difference was minimal (0.0011), suggesting both methods are viable

### Challenges:

1. SMOTE requires preprocessing first (categorical → numeric)
2. SMOTE increases training time due to synthetic sample generation
3. Risk of overfitting to synthetic data with SMOTE

---

## 📁 Artifacts Generated

### Baseline (class_weight):

- `artifacts/baseline_classweight/model.joblib`
- `artifacts/baseline_classweight/metrics.json`
- `artifacts/baseline_classweight/confusion_matrix.png`
- `artifacts/baseline_classweight/run_info.json`

### SMOTE:

- `artifacts/smote_results/model.joblib`
- `artifacts/smote_results/metrics.json`
- `artifacts/smote_results/confusion_matrix.png`
- `artifacts/smote_results/run_info.json`

---

## 🔄 Code Changes

### Files Modified:

1. `src/train.py` - Added SMOTE preprocessing
2. `config/train.yaml` - Disabled class_weight
3. `requirements.txt` - Added imbalanced-learn==0.12.3

### Key Implementation Details:

```python
# Critical: Apply SMOTE AFTER preprocessing
pre = make_preprocess(numeric_cols, cat_cols)
Xtr_preprocessed = pre.fit_transform(Xtr)
smote = SMOTE(random_state=rs)
Xtr_balanced, ytr_balanced = smote.fit_resample(Xtr_preprocessed, ytr)
```

---
