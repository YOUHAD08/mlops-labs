"""ML Training + MLOps artifacts generation.
Artifacts produced in artifacts/:
- model.joblib
- metrics.json
- confusion_matrix.png
- run_info.json (traceability: config + versions)
"""
import json
from pathlib import Path
from datetime import datetime
import sys
from datetime import datetime, timezone

import joblib
import yaml
import pandas as pd
# FORCE NON-INTERACTIVE BACKEND (IMPORTANT FOR VIRTUAL ENV)
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import sklearn
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from validate_data import validate_dataset

ART = Path("artifacts")

def load_cfg(path="config/train.yaml"):
    return yaml.safe_load(open(path, "r", encoding="utf-8"))

def make_preprocess(numeric_cols, cat_cols):
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore"))
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop"
    )
    return pre

def save_confusion_matrix(y_true, y_pred, out_path: Path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Pred")
    plt.ylabel("True")
    for (i, j), v in zip([(0,0),(0,1),(1,0),(1,1)], cm.flatten()):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main():
    cfg = load_cfg()
    data_path = cfg["data"]["path"]
    target = cfg["data"]["target"]

    # 0) Data tests (stop if failed)
    validate_dataset(data_path, target)

    # 1) Read data
    df = pd.read_csv(data_path)

    # 2) Simple cleaning example (optional)
    df = df[df["Age"] <= 80].copy()

    # 3) Separate X/y
    y = df[target]
    X = df.drop(columns=[target])

    # 4) Remove identifier columns
    drop_cols = ["RowNumber", "CustomerId", "Surname"]
    X = X.drop(columns=[c for c in drop_cols if c in X.columns])

    numeric_cols = ["Age","CreditScore","Balance","EstimatedSalary","Tenure","NumOfProducts"]
    cat_cols = ["Gender","Geography"]

    # 5) Reproducible split
    rs = int(cfg["split"]["random_state"])
    strat = y if cfg["split"].get("stratify", True) else None
    Xtr, Xte, ytr, yte = train_test_split(
        X, y,
        test_size=float(cfg["split"]["test_size"]),
        random_state=rs,
        stratify=strat
    )

    # 6) Preprocessing pipeline + model
    pre = make_preprocess(numeric_cols=numeric_cols, cat_cols=cat_cols)

    model_cfg = cfg["model"]
    model = LogisticRegression(
        max_iter=int(model_cfg["max_iter"]),
        class_weight=(None if str(model_cfg.get("class_weight", "None")).lower() == "none" else model_cfg.get("class_weight", None)),
        solver=model_cfg.get("solver", "liblinear"),
        random_state=rs
    )

    pipe = Pipeline(steps=[("preprocess", pre), ("model", model)])

    ART.mkdir(exist_ok=True)

    # 7) Training + evaluation
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)

    acc = float(accuracy_score(yte, pred))
    f1 = float(f1_score(yte, pred))

    # 8) Artifacts
    joblib.dump(pipe, ART / "model.joblib")

    json.dump(
        {"accuracy": acc, "f1": f1},
        open(ART / "metrics.json", "w", encoding="utf-8"),
        indent=2
    )

    save_confusion_matrix(yte, pred, ART / "confusion_matrix.png")
    
    # 9) Run info (traceability / reproducibility)
    run_info = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "config": cfg,
        "versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "report": classification_report(yte, pred, output_dict=True),
    }
    json.dump(
        run_info,
        open(ART / "run_info.json", "w", encoding="utf-8"),
        indent=2
    )

    print("OK:", {"accuracy": acc, "f1": f1})
    print("Artifacts -> artifacts/")

if __name__ == "__main__":
    main()