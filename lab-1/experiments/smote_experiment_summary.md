# Experiment: SMOTE vs class_weight for Class Imbalance

## 📋 Experiment Metadata

- **Date:** [TODAY'S_DATE]
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
| class_weight | [VALUE]  | [VALUE]  | 1629 (original)            |
| SMOTE        | [VALUE]  | [VALUE]  | 6361 (with synthetic)      |

### Winner: [METHOD_NAME]

- **Reason:** [Higher F1 score / Simpler / More efficient]
- **Performance Gain:** [+/- X.XX] F1 points

---

## 💡 Key Findings

### What Worked Well:

1. [Write 2-3 observations about what worked]
2.
3.

### What We Learned:

1. F1 score is crucial for imbalanced datasets
2. [Add your learnings]
3.

### Challenges:

1. SMOTE requires preprocessing first (categorical → numeric)
2. [Add any other challenges]

---

## 🎯 Recommendation

**For Production:** Use [WINNING_METHOD]

**Reasoning:**

- [Explain why this method is better]
- [Trade-offs considered]
- [Business impact]

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

## 🚀 Next Steps (Optional)

1. ✅ Extension 1 Complete - SMOTE comparison done
2. ⏭️ Extension 2 - Deploy as API with FastAPI (optional)
3. 🔬 Further experiments:
   - Try SMOTE-Tomek or SMOTE-ENN variants
   - Test different sampling ratios
   - Experiment with other algorithms

---

## 📚 References

- [imbalanced-learn documentation](https://imbalanced-learn.org/)
- SMOTE paper: Chawla et al. (2002)
- MLOps best practices for experiment tracking

---

**Experiment Status:** ✅ COMPLETE
