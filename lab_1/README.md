# MLOps Lab 1 - Bank Customer Churn Prediction

## 📋 Project Overview

This project demonstrates MLOps best practices for training a machine learning model to predict bank customer churn. It includes data validation, reproducible training, artifact generation, and Docker containerization.

**Objective:** Train a classification model to predict whether a customer will exit (Exited = 0/1) while ensuring:

- ✅ Reproducibility (seeds + versions + config)
- ✅ Traceability (named and preserved artifacts)
- ✅ Data Quality (schema tests + validation)
- ✅ Industrialization (Docker execution)

---

## 📁 Project Structure

```
lab_1/
├── 📁 .pytest_cache/          # Pytest cache (auto-generated)
├── 📁 artifacts/              # Generated artifacts (DO NOT EDIT MANUALLY)
│   ├── 💾 model.joblib        # Trained model pipeline
│   ├── 📊 metrics.json        # Performance metrics (accuracy, F1)
│   ├── 📈 confusion_matrix.png # Confusion matrix visualization
│   └── 📝 run_info.json       # Complete training record (config + versions)
├── 📁 config/
│   └── ⚙️ train.yaml          # Training configuration
├── 📁 data/
│   └── 📄 dataset.csv         # Input dataset
├── 📁 src/
│   ├── 🐍 train.py            # Main training script
│   └── 🐍 validate_data.py    # Data validation script
├── 📁 tests/
│   └── 🧪 test_data.py        # Data validation tests
├── 🚫 .gitignore              # Git ignore rules
├── 🐳 Dockerfile              # Docker image definition
├── 📖 README.md               # This file
└── 📋 requirements.txt        # Python dependencies
```

---

## 🛠️ Setup Instructions

### Prerequisites

- Python 3.11+
- Conda (recommended) or venv
- Docker (for containerized execution)

### 1. Create Environment

**Using Conda (recommended):**

```bash
conda create -n mlops-lab1 python=3.11
conda activate mlops-lab1
```

**Using venv (alternative):**

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Setup

```bash
python --version  # Should show Python 3.11.x
```

---

## 🚀 Usage

### Option 1: Local Execution

#### Step 1: Validate Data

```bash
# Using pytest (recommended)
python -m pytest -q

# Or direct execution
python src/validate_data.py
```

#### Step 2: Train Model

```bash
python src/train.py
```

**Expected output:**

```
OK: {'accuracy': 0.8465, 'f1': 0.6234}
Artefacts -> artifacts/
```

#### Step 3: Check Artifacts

```bash
ls -l artifacts/
cat artifacts/metrics.json
```

---

### Option 2: Docker Execution (Recommended for Production)

#### Step 1: Build Docker Image

```bash
docker build -t mlops-lab1:1.0 .
```

#### Step 2: Run Training in Container

**Linux/macOS:**

```bash
docker run --rm -v "$(pwd)/artifacts:/app/artifacts" mlops-lab1:1.0
```

**Windows PowerShell:**

```bash
docker run --rm -v "${PWD}/artifacts:/app/artifacts" mlops-lab1:1.0
```

**Windows CMD:**

```bash
docker run --rm -v "%cd%/artifacts:/app/artifacts" mlops-lab1:1.0
```

#### Step 3: Verify Artifacts Persisted

```bash
ls artifacts/
# Should contain: model.joblib, metrics.json, confusion_matrix.png, run_info.json
```

---

## 📊 Understanding the Artifacts

### 1. `model.joblib` (Binary File)

- **What:** Complete trained ML pipeline (preprocessing + model)
- **Size:** ~100-500 KB
- **Use:** Load this file to make predictions on new data

### 2. `metrics.json`

```json
{
  "accuracy": 0.8465,
  "f1": 0.6234
}
```

- **accuracy:** Overall correctness (85% of predictions correct)
- **f1:** Balance between precision and recall (better for imbalanced data)

### 3. `confusion_matrix.png`

- Visual representation of model predictions vs actual values
- Shows True Positives, True Negatives, False Positives, False Negatives

### 4. `run_info.json`

- **Most important for MLOps!**
- Contains:
  - Timestamp of training
  - Complete configuration used
  - Library versions (Python, scikit-learn, pandas, numpy)
  - Detailed classification report
- **Purpose:** Ensures complete traceability and reproducibility

---

## 🔄 Workflow Summary

```
1. Data Validation (validate_data.py)
          ↓
2. Load Config (train.yaml)
          ↓
3. Load & Clean Data
          ↓
4. Train/Test Split (reproducible)
          ↓
5. Preprocessing Pipeline (numeric + categorical)
          ↓
6. Model Training (Logistic Regression)
          ↓
7. Evaluation (accuracy, F1, confusion matrix)
          ↓
8. Save Artifacts (model, metrics, visualization, run info)
```

---

## 📚 Key MLOps Concepts Demonstrated

1. **Reproducibility:**

   - Fixed random seeds
   - Version locking (requirements.txt)
   - Configuration files

2. **Traceability:**

   - Artifact generation
   - Version tracking (run_info.json)
   - Timestamped runs

3. **Data Quality:**

   - Schema validation
   - Value range checks
   - Automated tests

4. **Containerization:**
   - Docker for environment isolation
   - Volume persistence
   - Portable execution

---

## 📝 License

This is an educational project for MLOps learning.

---

## 👤 Author

Created as part of MLOps Lab 1 - DevOps & MLOps Course

---

**Last Updated:** January 2026
