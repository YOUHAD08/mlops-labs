# SMOTE vs Class Weight Comparison

## Experiment Goal

Compare two methods for handling class imbalance and determine which gives better F1 score.

## Dataset Info

- Total samples: ~10,000
- Class 0 (Stayed): ~80%
- Class 1 (Exited): ~20%
- Imbalance ratio: 4:1

---

## Results

### Method 1: class_weight="balanced" (Baseline)

**Configuration:**

```yaml
model:
  class_weight: "balanced"
```

**Metrics:**

- Accuracy: 0.7102102102102102
- F1 Score: 0.49076517150395776

**Date:** 1/20/2026

---

### Method 2: SMOTE (Coming Soon)

**Configuration:**

```yaml
model:
  class_weight: null
```

- SMOTE applied before training

**Metrics:**

- Accuracy: TBD
- F1 Score: TBD

**Date:** TBD

---

## Conclusion

TBD after both experiments complete
