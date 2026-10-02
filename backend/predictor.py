from pathlib import Path

import joblib
import pandas as pd
import torch

from backend.model.network import PersistenceNN

# Address
BASE_DIR = Path(__file__).resolve().parent

ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.pt"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.pkl"

# Load fitted preprocessor
preprocessor = joblib.load(PREPROCESSOR_PATH)

# Get model input dimension from fitted preprocessor
input_dim = len(preprocessor.get_feature_names_out())
# print(input_dim)
# Create and load model
model = PersistenceNN(input_dim = input_dim)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location = "cpu")
)

model.eval()

def predict(data):
    # step 1: Load the student's data as a list rather than a dict
    df = pd.DataFrame([data])

    # step 2: User model preprocessor.pkl to transform the data
    processed = preprocessor.transform(df)

    # step 3: transfer to Tensor
    x = torch.tensor(processed, dtype = torch.float32)
    """
    Process:
    DataFrame   (1, 13)
        ↓
    preprocessor    
        ↓
    Numpy   (1, 36)
        ↓
    torch.tensor
        ↓
    Tensor  [1, 36]
    """

    with torch.no_grad():
        logit = model(x)
        probability = torch.sigmoid(logit)
        risk_probability = probability.item()   # Python Tensor -> Python float
        prediction = int(risk_probability >= 0.5)   # if risk_probability = 0.80 -> prediction = 1; if risk_probability = 0.23 -> prediction = 0

    return {
        "risk_probability" : risk_probability,
        "prediction": prediction
    }

    # temporary test
    # print(df.shape)
    # print(processed.shape)
    # print(x.shape)



if __name__ == "__main__":
    test_student = {
        "FirstTermGpa": 3.5,
        "SecondTermGpa": 3.2,
        "FirstLanguage": 1,
        "Funding": 2,
        "FastTrack": 1,
        "Coop": 1,
        "Residency": 1,
        "Gender": 1,
        "PrevEducation": 1,
        "AgeGroup": 2,
        "HighSchoolAvgMark": 80,
        "MathScore": 40,
        "EnglishGrade": 8
    }
    result = predict(test_student)
    print(result)