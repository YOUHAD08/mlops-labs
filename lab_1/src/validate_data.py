"""Minimal data validation (data tests).
- Checks schema, target, and some simple constraints.
"""
import pandas as pd

EXPECTED_COLS = [
    "RowNumber","CustomerId","Surname","CreditScore","Geography","Gender","Age",
    "Tenure","Balance","NumOfProducts","HasCrCard","IsActiveMember",
    "EstimatedSalary","Exited"
]

def validate_dataset(path: str, target: str = "Exited") -> None:
    df = pd.read_csv(path)

    # 1) Schema: expected columns (at minimum)
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    assert len(missing) == 0, f"Missing columns: {missing}"

    # 2) Target not empty
    assert df[target].notna().all(), "The target column contains missing values."

    # 3) Simple checks (examples)
    assert (df["Age"] >= 0).all(), "Negative age detected."
    assert (df["Age"] <= 120).all(), "Age too high detected."
    assert (df["CreditScore"] >= 0).all(), "Negative CreditScore detected."

    # 4) Type/value checks (optional but useful)
    assert set(df[target].unique()).issubset({0, 1}), "Target expected to be binary {0,1}."

if __name__ == "__main__":
    print("Starting dataset validation...")
    validate_dataset("data/dataset.csv", "Exited")
    print("✓ OK: dataset is valid")
    print("All checks passed!")