# MLOps Lab 1 - Bank Customer Churn Prediction

## 📋 Project Overview

ML pipeline to predict bank customer churn with MLOps best practices: reproducibility, traceability, data quality, and containerization.

**Objective:** Predict customer exit (0/1) with:

- ✅ Reproducibility (seeds + versions + config)
- ✅ Traceability (artifacts + metadata)
- ✅ Data Quality (schema validation)
- ✅ Containerization (Docker)

---

## 📁 Project Structure

```
lab_1/
├── 📁 .pytest_cache/
├── 📁 artifacts/
│   ├── 📁 baseline_classweight/
│   ├── 📁 smote_results/
│   ├── 💾 model.joblib (latest)
│   ├── 📊 metrics.json (latest)
│   ├── 📈 confusion_matrix.png (latest)
│   └── 📝 run_info.json (latest)
├── 📁 config/
│   └── ⚙️ train.yaml
├── 📁 data/
│   └── 📄 dataset.csv
├── 📁 experiments/
│   └── 📝 smote_experiment_summary.md
├── 📁 src/
│   ├── 🐍 train.py
│   └── 🐍 validate_data.py
├── 📁 tests/
│   └── 🧪 test_data.py
├── 🚫 .gitignore
├── 🐳 Dockerfile
├── 📖 README.md
├── 📋 comparison_results.md
└── 📋 requirements.txt
```

---

## 🛠️ Setup

### Prerequisites

- Python 3.11+
- Docker (optional)

### Install

```bash
# Create environment
conda create -n mlops-lab1 python=3.11
conda activate mlops-lab1

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Local Execution

```bash
# 1. Validate data
python -m pytest -q

# 2. Train model
python src/train.py

# 3. Check results
ls artifacts/
cat artifacts/metrics.json
```

### Docker Execution

```bash
# Build
docker build -t mlops-lab1:1.0 .

# Run (Linux/macOS)
docker run --rm -v "$(pwd)/artifacts:/app/artifacts" mlops-lab1:1.0

# Run (Windows PowerShell)
docker run --rm -v "${PWD}/artifacts:/app/artifacts" mlops-lab1:1.0
```

---

## 🔮 Extensions

### ✅ Extension 1: SMOTE vs class_weight

**Status:** Complete  
**Winner:** class_weight="balanced"  
**Results:**

- class_weight: F1 = 0.4908, Accuracy = 0.7102
- SMOTE: F1 = 0.4897, Accuracy = 0.7142

**Details:** See `experiments/` folder

---

### ⏭️ Extension 2: FastAPI Deployment

**Status:** Pending

---

## 📊 Artifacts

| File                   | Description               |
| ---------------------- | ------------------------- |
| `model.joblib`         | Complete trained pipeline |
| `metrics.json`         | Accuracy & F1 scores      |
| `confusion_matrix.png` | Visualization             |
| `run_info.json`        | Full training metadata    |

---

## 🔄 Workflow

```
Data Validation → Load Config → Clean Data → Train/Test Split
→ Preprocessing → Model Training → Evaluation → Save Artifacts
```

---

## 📚 Key MLOps Concepts

1. **Reproducibility:** Fixed seeds, version locking, config files
2. **Traceability:** Artifacts, version tracking, timestamps
3. **Data Quality:** Schema validation, automated tests
4. **Containerization:** Docker isolation, volume persistence

---

**Last Updated:** January 2026
