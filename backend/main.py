"""
Process:
    Frontend
        ↓
    main.py
        ↓
    predictor.py
        ↓
    model + preprocessor
"""
# Configuration
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.predictor import predict

from pathlib import Path
import joblib

#
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

METRICS_PATH = ARTIFACTS_DIR / "metrics.pkl"

app = FastAPI(
    title = "Student At-Risk Prediction API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers = ["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok"}

class StudentInput(BaseModel):
    # Numerical Features
    FirstTermGpa : float
    SecondTermGpa : float
    HighSchoolAvgMark : float
    MathScore : float
    EnglishGrade : float

    # Categorical Features
    FirstLanguage : int
    Funding : int
    FastTrack : int
    Coop : int
    Residency : int
    Gender : int
    PrevEducation : int
    AgeGroup : int

@app.post("/predict")
def predict_student(student : StudentInput):
    # 1. Pydantic object -> dict
    student_data = student.model_dump()
    # 2. Send data to predictor
    result = predict(student_data)
    # 3. Return prediction result
    return result

def load_metrics():
    return joblib.load(METRICS_PATH)

@app.get("/model-info")
def model_info():
    metrics = load_metrics()
    return {
        "model" : "PersistenceNN",
        "target" : "AtRisk",
        "threshold" : 0.5,
        "metrics" : metrics
    }