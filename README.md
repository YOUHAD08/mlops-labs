# MLOps - Bank Customer Churn Prediction

## 📋 Project Overview

This project demonstrates comprehensive MLOps practices for predicting bank customer churn using machine learning. It includes data validation, reproducible training, artifact generation, Docker containerization, and model deployment via REST API.

**Business Objective:** Predict whether a customer will exit the bank (Exited = 0/1) to enable proactive retention strategies.

**MLOps Principles Applied:**

- ✅ **Reproducibility** - Fixed seeds, version control, configuration management
- ✅ **Traceability** - Artifact generation, experiment tracking, version logging
- ✅ **Data Quality** - Schema validation, automated testing, value checks
- ✅ **Industrialization** - Docker execution, API deployment, automated workflows
- ✅ **Experimentation** - Systematic comparison of ML approaches

---

## 🎯 What Makes This Project Special

This isn't just a machine learning project - it's a **production-ready MLOps pipeline** that demonstrates:

1. **Professional Workflow** - Git branching, feature development, experiment tracking
2. **Code Quality** - Automated testing, validation, error handling
3. **Reproducibility** - Anyone can recreate exact same results
4. **Deployment Ready** - REST API for real-time predictions
5. **Documentation** - Comprehensive guides and experiment summaries

---

## 📁 Project Structure

```
mlops-mini-project-churn/
├── 📁 .pytest_cache/          # Pytest cache (auto-generated)
├── 📁 artifacts/              # Generated artifacts (DO NOT EDIT MANUALLY)
│   ├── 📁 baseline_classweight/
│   │   ├── 💾 model.joblib        # Trained model (class_weight approach)
│   │   ├── 📊 metrics.json        # Performance metrics
│   │   ├── 📈 confusion_matrix.png # Confusion matrix visualization
│   │   └── 📝 run_info.json       # Complete training record
│   ├── 📁 smote_results/
│   │   ├── 💾 model.joblib        # Trained model (SMOTE approach)
│   │   ├── 📊 metrics.json        # Performance metrics
│   │   ├── 📈 confusion_matrix.png # Confusion matrix visualization
│   │   └── 📝 run_info.json       # Complete training record
│   ├── 💾 model.joblib            # Latest trained model
│   ├── 📊 metrics.json            # Latest metrics
│   ├── 📈 confusion_matrix.png    # Latest confusion matrix
│   └── 📝 run_info.json           # Latest run information
├── 📁 config/
│   └── ⚙️ train.yaml              # Training configuration
├── 📁 data/
│   └── 📄 dataset.csv             # Input dataset (10K customer records)
├── 📁 experiments/                # Experiment tracking and summaries
│   ├── 📝 smote_experiment_summary.md
│   └── 📝 api_deployment_summary.md
├── 📁 src/
│   ├── 🐍 train.py                # Main training script
│   ├── 🐍 validate_data.py        # Data validation script
│   └── 🐍 api.py                  # FastAPI application
├── 📁 tests/
│   └── 🧪 test_data.py            # Data validation tests
├── 🚫 .gitignore                  # Git ignore rules
├── 🐳 Dockerfile                  # Docker image for training
├── 📖 README.md                   # This file
├── 📋 comparison_results.md       # SMOTE vs class_weight comparison
├── 🐍 test_api_client.py          # API testing script
└── 📋 requirements.txt            # Python dependencies
```

---

## 🛠️ Setup Instructions

### Prerequisites

- **Python:** 3.11+ (3.10+ also works)
- **Conda:** Recommended for environment management
- **Docker:** For containerized execution (optional)
- **Git:** For version control

### 1. Clone and Navigate

```bash
git clone <your-repo-url>
cd lab_1
```

### 2. Create Environment

**Using Conda (recommended):**

```bash
conda create -n mlops-lab1 python=3.11
conda activate mlops-lab1
```

**Using venv (alternative):**

```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate

# Linux/Mac:
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key Dependencies:**

- `pandas` - Data manipulation
- `scikit-learn` - Machine learning
- `matplotlib` - Visualizations
- `pyyaml` - Configuration management
- `pytest` - Testing framework
- `imbalanced-learn` - Handling class imbalance
- `fastapi` - REST API framework
- `uvicorn` - ASGI server

### 4. Verify Setup

```bash
python --version  # Should show Python 3.11.x or 3.10.x
python -c "import sklearn, pandas, fastapi; print('✅ All dependencies installed')"
```

---

## 🚀 Quick Start

### Option 1: Train Model Locally

```bash
# Validate data first
python src/validate_data.py

# Train model
python src/train.py

# Check results
cat artifacts/metrics.json
```

### Option 2: Train with Docker

```bash
# Build image
docker build -t mlops-lab1:1.0 .

# Run training (artifacts persist to local folder)
docker run --rm -v "$(pwd)/artifacts:/app/artifacts" mlops-lab1:1.0

# Check results
ls artifacts/
```

### Option 3: Run API Server

```bash
# Start API
python src/api.py

# Visit interactive docs
# http://localhost:8000/docs
```

---

## 📚 Detailed Usage Guide

### 1. Data Validation

The project includes automated data quality checks to catch issues before training.

**What is validated:**

- ✅ Schema validation (all expected columns present)
- ✅ Target column completeness (no missing values in "Exited")
- ✅ Value ranges (Age: 0-120, CreditScore ≥ 0)
- ✅ Data types (numeric fields are numeric)

**Run validation:**

```bash
# Automated tests (recommended)
python -m pytest -v

# Manual validation
python src/validate_data.py
```

**Expected columns:**

```
RowNumber, CustomerId, Surname, CreditScore, Geography, Gender,
Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember,
EstimatedSalary, Exited
```

**If validation fails:**

- Check error message for specific issue
- Fix data quality problems before training
- Re-run validation

---

### 2. Model Training

**Configuration File:** `config/train.yaml`

```yaml
data:
  path: "data/dataset.csv"
  target: "Exited"

split:
  test_size: 0.2 # 20% for testing
  random_state: 42 # Reproducibility seed
  stratify: true # Balanced class distribution

model:
  name: "logistic_regression"
  max_iter: 3000
  class_weight: "balanced" # or null for SMOTE
```

**Training Process:**

```bash
python src/train.py
```

**What happens:**

1. ✅ Validates data quality
2. 📖 Loads and cleans dataset
3. ✂️ Splits into train/test (80/20)
4. 🔧 Preprocesses features (numeric scaling + one-hot encoding)
5. 🎓 Trains Logistic Regression model
6. 📊 Evaluates on test set
7. 💾 Saves 4 artifacts

**Output:**

```
OK: {'accuracy': 0.8465, 'f1': 0.6234}
Artefacts -> artifacts/
```

---

### 3. Understanding Artifacts

#### 📄 `model.joblib` (Binary File)

- **Content:** Complete trained ML pipeline
- **Size:** ~100-500 KB
- **Usage:** Load to make predictions on new data

```python
import joblib
model = joblib.load('artifacts/model.joblib')
# Use model for predictions
```

#### 📄 `metrics.json`

```json
{
  "accuracy": 0.8465,
  "f1": 0.6234
}
```

- **accuracy:** Overall correctness (84.65% of predictions correct)
- **f1:** Harmonic mean of precision and recall (better for imbalanced data)

**Why F1 is more important:**

- Dataset is imbalanced (80% stayed, 20% left)
- Accuracy can be misleading (always predicting "stay" gives 80% accuracy!)
- F1 considers both false positives and false negatives

#### 📄 `confusion_matrix.png`

Visual representation showing:

```
              Predicted
              Stay  Exit
Actual Stay   [TN]  [FP]
       Exit   [FN]  [TP]
```

- **TN (True Negative):** Correctly predicted "stay"
- **TP (True Positive):** Correctly predicted "exit"
- **FP (False Positive):** Predicted "exit" but actually stayed (lost opportunity)
- **FN (False Negative):** Predicted "stay" but actually left (missed at-risk customer!)

#### 📄 `run_info.json`

**Most important for MLOps!** Contains complete training record:

```json
{
  "timestamp": "2026-01-22T14:30:22Z",
  "config": { ... },  // Complete train.yaml
  "versions": {
    "python": "3.11.5",
    "scikit-learn": "1.3.0",
    "pandas": "2.0.3",
    "numpy": "1.24.3"
  },
  "report": { ... }  // Detailed classification metrics
}
```

**Purpose:**

- ✅ Reproducibility (exact versions used)
- ✅ Traceability (what configuration produced these results)
- ✅ Auditability (timestamp and full context)

---

### 4. Docker Execution

**Why Docker?**

- ✅ **Environment isolation** - No "works on my machine" issues
- ✅ **Reproducibility** - Exact same environment every time
- ✅ **Portability** - Run anywhere Docker is installed
- ✅ **Version control** - Environment captured in Dockerfile

**Build Image:**

```bash
docker build -t mlops-lab1:1.0 .
```

**Run Training:**

```bash
# Linux/macOS
docker run --rm -v "$(pwd)/artifacts:/app/artifacts" mlops-lab1:1.0

# Windows PowerShell
docker run --rm -v "${PWD}/artifacts:/app/artifacts" mlops-lab1:1.0

# Windows CMD
docker run --rm -v "%cd%/artifacts:/app/artifacts" mlops-lab1:1.0
```

**Volume Mount Explained:**

- `-v "$(pwd)/artifacts:/app/artifacts"` creates a bridge
- Files created inside container appear in your local `artifacts/` folder
- Without volume, files disappear when container stops!

---

### 5. REST API Usage

**Start API Server:**

```bash
python src/api.py
```

**Endpoints:**

| Endpoint      | Method | Purpose                   |
| ------------- | ------ | ------------------------- |
| `/`           | GET    | API information           |
| `/health`     | GET    | Health check              |
| `/predict`    | POST   | Make prediction           |
| `/model-info` | GET    | Model metadata            |
| `/docs`       | GET    | Interactive documentation |

**Interactive Documentation:**

Visit `http://localhost:8000/docs` for Swagger UI where you can:

- View all endpoints
- Test predictions directly in browser
- See request/response schemas
- Download OpenAPI specification

**Example Prediction Request:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "CreditScore": 650,
    "Geography": "France",
    "Gender": "Male",
    "Age": 35,
    "Tenure": 5,
    "Balance": 125000.0,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 75000.0
  }'
```

**Example Response:**

```json
{
  "prediction": 0,
  "will_exit": false,
  "probability_stay": 0.82,
  "probability_exit": 0.18,
  "risk_level": "Low"
}
```

**Using Python Client:**

```bash
python test_api_client.py
```

**Input Validation:**

The API automatically validates all inputs:

- ✅ CreditScore: 300-850
- ✅ Geography: France, Germany, Spain
- ✅ Gender: Male, Female
- ✅ Age: 18-100
- ✅ All fields required

Invalid inputs return clear error messages:

```json
{
  "detail": [
    {
      "loc": ["body", "Age"],
      "msg": "Input should be less than or equal to 100"
    }
  ]
}
```

---

## 🔬 Extension 1: SMOTE vs class_weight Comparison

### Objective

Compare two approaches for handling imbalanced data (20% exit, 80% stay):

1. **class_weight="balanced"** - Model internally weights minority class
2. **SMOTE** - Creates synthetic minority class samples

### Results

| Method       | Accuracy     | F1 Score     | Approach     |
| ------------ | ------------ | ------------ | ------------ |
| class_weight | [YOUR_VALUE] | [YOUR_VALUE] | Weight-based |
| SMOTE        | [YOUR_VALUE] | [YOUR_VALUE] | Oversampling |

**Winner:** [METHOD] based on F1 score

**Detailed Analysis:** See `experiments/smote_experiment_summary.md`

### Key Findings

**What We Learned:**

1. F1 score is more reliable than accuracy for imbalanced data
2. [WINNING_METHOD] better predicts customers who will exit
3. SMOTE requires preprocessing before application (categorical → numeric)

**Trade-offs:**

**class_weight:**

- ➕ Simpler implementation
- ➕ Faster training
- ➕ Original data unchanged
- ➖ Relies on model's internal mechanism

**SMOTE:**

- ➕ Explicit data balancing
- ➕ Model sees more minority examples
- ➖ More complex preprocessing
- ➖ Longer training time
- ➖ Risk of overfitting synthetic data

**Recommendation:** Use [WINNING_METHOD] for production deployment.

**Full Comparison:** See `comparison_results.md`

---

## 🌐 Extension 2: REST API Deployment

### Status

✅ **Local Implementation Complete**  
⏭️ Docker deployment pending

### Features Implemented

**API Framework:** FastAPI + Uvicorn

**Endpoints:**

- ✅ Health check (`/health`)
- ✅ Prediction (`/predict`)
- ✅ Model information (`/model-info`)
- ✅ Automatic documentation (`/docs`)

**Input Validation:**

- ✅ Pydantic schemas with field constraints
- ✅ Clear error messages for invalid inputs
- ✅ Type checking and range validation

**Response Format:**

```json
{
  "prediction": 0, // 0=Stay, 1=Exit
  "will_exit": false, // Human-readable boolean
  "probability_stay": 0.82, // Confidence (0-1)
  "probability_exit": 0.18, // Confidence (0-1)
  "risk_level": "Low" // Low/Medium/High
}
```

**Testing:**

- ✅ All endpoints tested locally
- ✅ Multiple customer scenarios validated
- ✅ Error handling verified
- ✅ Test client script provided

**Detailed Documentation:** See `experiments/api_deployment_summary.md`

---

## 📊 Model Performance

### Current Baseline

**Model:** Logistic Regression  
**Approach:** [class_weight / SMOTE]

**Metrics:**

- **Accuracy:** ~84-86%
- **F1 Score:** ~60-65%
- **Precision:** ~70-75% (for exit class)
- **Recall:** ~50-60% (for exit class)

**Interpretation:**

- Model correctly identifies ~85% of cases overall
- For customers who actually exit, catches ~55% of them
- When predicting exit, correct ~72% of the time

### Performance Factors

**Class Imbalance:**

- 80% customers stay, 20% exit
- Model naturally biased toward majority class
- SMOTE/class_weight help balance this

**Feature Importance:**

- Age, Balance, NumOfProducts are strong predictors
- Geography and Gender also contribute
- CreditScore has moderate impact

### Improvement Opportunities

**Better F1 Score:**

1. Feature engineering (create interaction features)
2. Try ensemble models (Random Forest, XGBoost)
3. Hyperparameter tuning
4. Collect more data on minority class

**Business Context:**

- False Negative (FN) is costly - miss at-risk customer
- False Positive (FP) is less costly - unnecessary retention offer
- May want to optimize for recall over precision

---

## 🧪 Testing

### Automated Tests

```bash
# Run all tests
python -m pytest -v

# Run specific test file
python -m pytest tests/test_data.py -v

# Run with coverage
python -m pytest --cov=src tests/
```

### Manual Testing

**Data Validation:**

```bash
python src/validate_data.py
```

**Training:**

```bash
python src/train.py
```

**API:**

```bash
# Start server
python src/api.py

# In another terminal, run tests
python test_api_client.py
```

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError"

**Problem:** Missing dependencies

**Solution:**

```bash
# Ensure environment is activated
conda activate mlops-lab1  # or source .venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

---

### Issue: Docker artifacts not persisting

**Problem:** Volume mount incorrect

**Solution:**

```bash
# Use absolute path
docker run --rm -v "/full/path/to/artifacts:/app/artifacts" mlops-lab1:1.0

# Or ensure you're in project root
pwd  # Should show .../lab_1
docker run --rm -v "$(pwd)/artifacts:/app/artifacts" mlops-lab1:1.0
```

---

### Issue: "Colonnes manquantes" error

**Problem:** Dataset missing required columns

**Solution:**

```bash
# Check your CSV has all required columns
python -c "import pandas as pd; print(pd.read_csv('data/dataset.csv').columns.tolist())"

# Compare with expected columns in validate_data.py
```

---

### Issue: API returns 503 "Model not available"

**Problem:** Model file not found

**Solution:**

```bash
# Check model exists
ls artifacts/smote_results/model.joblib

# Or train model first
python src/train.py

# Update MODEL_PATH in src/api.py if needed
```

---

### Issue: SMOTE "could not convert string to float"

**Problem:** Applying SMOTE before preprocessing

**Solution:** Already fixed in code - SMOTE applied AFTER preprocessing converts categorical to numeric.

---

### Issue: Matplotlib backend error

**Problem:** No display available in virtual environment

**Solution:** Already handled with `matplotlib.use('Agg')` at top of train.py

---

## 🔐 Best Practices Demonstrated

### MLOps Principles

1. **Reproducibility**
   - Fixed random seeds (`random_state=42`)
   - Version pinning in requirements.txt
   - Configuration files for settings
   - Complete run_info.json for traceability

2. **Data Quality**
   - Schema validation before training
   - Value range checks
   - Automated testing with pytest
   - Clear error messages

3. **Experiment Tracking**
   - Separate artifact folders for each approach
   - Detailed comparison documentation
   - Git branches for features/experiments
   - Tagged milestones

4. **Code Quality**
   - Modular design (separate validation, training, API)
   - Comprehensive documentation
   - Type hints and docstrings
   - Error handling

5. **Deployment Readiness**
   - Docker containerization
   - REST API for serving predictions
   - Health check endpoints
   - Input validation

### Git Workflow

**Branching Strategy:**

```
main
  └── dev
      ├── feature/smote-comparison (merged)
      └── feature/api-deployment (merged)
```

**Commit Conventions:**

- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation updates
- `chore:` Maintenance tasks
- `test:` Testing additions

**Tags:**

- `extension-1-complete` - SMOTE comparison done
- `extension-2-partial` - API implementation done

---

## 📖 Additional Resources

### Documentation Files

- **`comparison_results.md`** - Detailed SMOTE vs class_weight analysis
- **`experiments/smote_experiment_summary.md`** - Extension 1 summary
- **`experiments/api_deployment_summary.md`** - Extension 2 summary

### Learning Resources

**MLOps Concepts:**

- [Anthropic Documentation](https://docs.anthropic.com/)
- [MLOps Principles](https://ml-ops.org/)
- [Experiment Tracking Best Practices](https://neptune.ai/blog/ml-experiment-tracking)

**Technical Docs:**

- [scikit-learn](https://scikit-learn.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Docker](https://docs.docker.com/)
- [imbalanced-learn](https://imbalanced-learn.org/)

---

## 🚀 Future Enhancements

### Short Term

- [ ] Complete Docker deployment for API
- [ ] Add model versioning system
- [ ] Implement CI/CD pipeline
- [ ] Add monitoring and logging

### Medium Term

- [ ] Try other algorithms (Random Forest, XGBoost)
- [ ] Implement feature engineering pipeline
- [ ] Add model explainability (SHAP values)
- [ ] Create dashboard for predictions

### Long Term

- [ ] Deploy to cloud (AWS/GCP/Azure)
- [ ] Implement A/B testing framework
- [ ] Add real-time model monitoring
- [ ] Build automated retraining pipeline

---

## 👥 Project Structure Rationale

### Why This Organization?

**`artifacts/`** - Generated outputs (never edit manually)

- Proves what the model actually produced
- Enables comparison between runs
- Supports reproducibility

**`config/`** - Configuration files

- Separates settings from code
- Easy to modify without changing code
- Supports multiple configurations

**`data/`** - Input data

- Clear separation of data from code
- Easy to swap datasets
- Version control friendly (.gitignore for large files)

**`experiments/`** - Experiment tracking

- Documents decision-making process
- Enables learning from past experiments
- Supports knowledge sharing

**`src/`** - Source code

- Core business logic
- Reusable modules
- Clean separation of concerns

**`tests/`** - Test files

- Automated quality checks
- Regression prevention
- Documentation through examples

---

## 📊 Dataset Information

**Source:** Bank customer data  
**Size:** ~10,000 records  
**Target:** Exited (0 = Stayed, 1 = Left)  
**Class Distribution:** 80% stayed, 20% exited (imbalanced)

**Features:**

- **Demographic:** Age, Gender, Geography
- **Financial:** CreditScore, Balance, EstimatedSalary
- **Engagement:** Tenure, NumOfProducts, HasCrCard, IsActiveMember
- **Identifiers:** RowNumber, CustomerId, Surname (not used for prediction)

---

## ✅ Project Checklist

### Core Workshop

- [x] Project structure created
- [x] Environment setup (Conda/venv)
- [x] Dependencies installed
- [x] Configuration file created
- [x] Data validation implemented
- [x] Training pipeline developed
- [x] Artifacts generated
- [x] Docker execution working

### Extension 1: SMOTE Comparison

- [x] imbalanced-learn installed
- [x] Baseline (class_weight) results saved
- [x] SMOTE implementation
- [x] Both approaches compared
- [x] Results analyzed and documented
- [x] Merged to dev branch

### Extension 2: API Deployment

- [x] FastAPI installed
- [x] API implementation
- [x] Endpoints developed
- [x] Input validation added
- [x] Local testing complete
- [x] Test client created
- [x] Merged to dev branch
- [ ] Docker deployment (pending)
- [ ] Production documentation (pending)

---

## 🎓 Learning Outcomes

By completing this project, you've learned:

### MLOps Skills

✅ Experiment tracking and comparison  
✅ Artifact management and versioning  
✅ Data validation and quality checks  
✅ Model reproducibility techniques  
✅ Docker containerization  
✅ REST API development  
✅ Input validation and error handling

### ML/Data Science

✅ Handling imbalanced datasets  
✅ SMOTE vs class_weight approaches  
✅ Model evaluation metrics (accuracy vs F1)  
✅ Preprocessing pipelines  
✅ Feature engineering concepts

### Software Engineering

✅ Git branching strategies  
✅ Feature development workflow  
✅ Code organization and modularity  
✅ Testing and validation  
✅ Documentation best practices

---

## 📧 Support

For questions or issues:

1. Check troubleshooting section above
2. Review experiment documentation in `experiments/`
3. Check Git history for context: `git log --oneline`
4. Review inline code comments

---

## 📄 License

Educational project for MLOps learning.

---

## 🙏 Acknowledgments

- **Course:** MLOps / DevOps Lab (Prof. Soufiane HAMIDA)
- **Institution:** ENSET
- **Frameworks:** scikit-learn, FastAPI, Docker
- **Community:** Open-source ML and MLOps community

---

**Last Updated:** January 2026  
**Project Status:** Core Complete ✅ | Extension 1 Complete ✅ | Extension 2 Partial 🚧  
**Version:** 2.0.0

---

## 🎯 Quick Commands Reference

```bash
# Setup
conda create -n mlops-lab1 python=3.11
conda activate mlops-lab1
pip install -r requirements.txt

# Validation
python src/validate_data.py
python -m pytest -v

# Training
python src/train.py

# Docker
docker build -t mlops-lab1:1.0 .
docker run --rm -v "$(pwd)/artifacts:/app/artifacts" mlops-lab1:1.0

# API
python src/api.py
# Visit http://localhost:8000/docs

# Testing
python test_api_client.py

# Git
git log --oneline --graph --all
git tag
```

---
