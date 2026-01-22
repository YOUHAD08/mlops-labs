# tests/test_data.py
from src.validate_data import validate_dataset

def test_dataset_schema_and_target():
    validate_dataset("data/dataset.csv", "Exited")
