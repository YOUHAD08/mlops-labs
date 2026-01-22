"""
FastAPI application for Bank Customer Churn Prediction
Loads trained model and serves predictions via REST API
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any

# Initialize FastAPI app
app = FastAPI(
    title="Bank Customer Churn Prediction API",
    description="Predict whether a bank customer will exit (churn) based on their profile",
    version="1.0.0"
)

# Load the trained model and preprocessor
MODEL_PATH = Path("artifacts/smote_results/model.joblib")  
try:
    model_artifacts = joblib.load(MODEL_PATH)
    preprocessor = model_artifacts['preprocessor']
    model = model_artifacts['model']
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"❌ Model file not found at {MODEL_PATH}")
    print("⚠️ API will start but predictions will fail until model is available")
    preprocessor = None
    model = None
except Exception as e:
    print(f"❌ Error loading model: {e}")
    preprocessor = None
    model = None


# Define input data schema using Pydantic
class CustomerData(BaseModel):
    """
    Customer profile data for churn prediction
    All fields are required
    """
    CreditScore: int = Field(..., ge=300, le=850, description="Credit score (300-850)")
    Geography: str = Field(..., description="Country: France, Germany, or Spain")
    Gender: str = Field(..., description="Gender: Male or Female")
    Age: int = Field(..., ge=18, le=100, description="Age (18-100)")
    Tenure: int = Field(..., ge=0, le=10, description="Years as customer (0-10)")
    Balance: float = Field(..., ge=0, description="Account balance")
    NumOfProducts: int = Field(..., ge=1, le=4, description="Number of products (1-4)")
    HasCrCard: int = Field(..., ge=0, le=1, description="Has credit card: 1=Yes, 0=No")
    IsActiveMember: int = Field(..., ge=0, le=1, description="Active member: 1=Yes, 0=No")
    EstimatedSalary: float = Field(..., ge=0, description="Estimated salary")
    
    class Config:
        json_schema_extra = {
            "example": {
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
            }
        }


# Define output schema
class PredictionResponse(BaseModel):
    """
    Prediction result
    """
    prediction: int = Field(..., description="Predicted class: 0=Stay, 1=Exit")
    will_exit: bool = Field(..., description="Will customer exit? True/False")
    probability_stay: float = Field(..., description="Probability of staying (0-1)")
    probability_exit: float = Field(..., description="Probability of exiting (0-1)")
    risk_level: str = Field(..., description="Risk level: Low, Medium, High")


@app.get("/")
def read_root():
    """
    Root endpoint - API health check
    """
    return {
        "message": "Bank Customer Churn Prediction API",
        "status": "running",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)",
            "docs": "/docs (GET)"
        }
    }


@app.get("/health")
def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH)
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData) -> Dict[str, Any]:
    """
    Make churn prediction for a customer
    
    Args:
        customer: Customer profile data
    
    Returns:
        Prediction result with probabilities and risk level
    """
    # Check if model is loaded
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Please ensure model file exists."
        )
    
    try:
        # Convert input to DataFrame (model expects this format)
        input_data = pd.DataFrame([customer.model_dump()])
        
        # Preprocess the data
        input_preprocessed = preprocessor.transform(input_data)
        
        # Make prediction
        prediction = int(model.predict(input_preprocessed)[0])
        probabilities = model.predict_proba(input_preprocessed)[0]
        
        # Determine risk level based on exit probability
        prob_exit = float(probabilities[1])
        if prob_exit < 0.3:
            risk_level = "Low"
        elif prob_exit < 0.6:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return {
            "prediction": prediction,
            "will_exit": bool(prediction),
            "probability_stay": float(probabilities[0]),
            "probability_exit": prob_exit,
            "risk_level": risk_level
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.get("/model-info")
def model_info():
    """
    Get information about the loaded model
    """
    if model is None:
        return {"error": "Model not loaded"}
    
    return {
        "model_type": type(model).__name__,
        "model_path": str(MODEL_PATH),
        "preprocessor_type": type(preprocessor).__name__,
        "features_expected": [
            "CreditScore", "Geography", "Gender", "Age", "Tenure",
            "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
            "EstimatedSalary"
        ]
    }


# Example usage for testing
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting API server...")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔍 Alternative docs: http://localhost:8000/redoc")
    uvicorn.run(app, host="0.0.0.0", port=8000)