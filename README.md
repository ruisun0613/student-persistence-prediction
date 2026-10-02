# Student At-Risk Prediction System

## Overview
This project is a machine learning system that predicts whether a student is at risk of not persisting in their studies.
It uses PyTorch to train a neural network model based on student academic and demographic information as input.
The backend is built with FastAPI and provides API endpoints for model inference and model information.
A simple web interface allows users to enter student information and view the prediction results.

## Features
- Student at-risk prediction
- Automated data preprocessing
- Class imbalance handling
- Neural network training with early stopping
- Model evaluation with multiple metrics
- REST API for prediction
- Simple web interface

## Tech Stack
- Python
- PyTorch
- FastAPI
- Scikit-learn
- Pandas
- NumPy
- HTML / CSS / JavaScript

## Project Structure
```text
Student At-Risk Prediction System/
├── backend/
│   ├── data/
│   │   ├── data_loader.py              # Loads the student dataset
│   │   └── preprocessing.py            # Cleans and preprocesses data
│   │
│   ├── model/
│   │   ├── network.py                  # Defines the PyTorch neural network
│   │   └── train.py                    # Trains and evaluates the model
│   │
│   ├── artifacts/
│   │   ├── model.pt                    # Trained model weights
│   │   ├── preprocessor.pkl            # Fitted preprocessing pipeline
│   │   └── metrics.pkl                 # Saved test metrics
│   │
│   ├── main.py                         # FastAPI application
│   └── predictor.py                    # Loads artifacts and performs inference
│
├── frontend/
│   ├── index.html                      # Web interface structure
│   ├── style.css                       # Frontend styling
│   └── script.js                       # API requests and prediction display
│
├── .gitignore
├── README.md
├── pyproject.toml
└── requirements.txt
```

## Machine Learning Pipeline
1. **Data Loading**
    - Load the student dataset.

2. **Data Preprocessing**
    - Handle missing values.
    - Scale numerical features.
    - Encode categorical features.

3. **Data Splitting**
    - Split the dataset into training, validation, and test sets.

4. **Model Training**
    - Train a PyTorch neural network with weighted binary cross-entropy loss and early stopping.

5. **Model Evaluation**
    - Evaluate the model using accuracy, precision, recall, F1-score, and a confusion matrix.

6. **Model Saving**
    - Save the trained model, fitted preprocessor, and evaluation metrics for inference.

## Model Performance

The final model was evaluated on the held-out test set.

### Test Metrics

| Metric | Score |
|---|---:|
| Accuracy | 76.39% |
| At-Risk Precision | 46.25% |
| At-Risk Recall | 82.22% |
| At-Risk F1-Score | 59.20% |

### Confusion Matrix

| | Predicted Not At Risk | Predicted At Risk |
|---|---:|---:|
| Actual Not At Risk | 128 | 43 |
| Actual At Risk | 8 | 37 |

The model prioritizes identifying at-risk students by using a weighted loss to address class imbalance.

## Installation

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install the dependencies.

```bash
pip install -r requirements.txt
```

## Running

### Backend
```bash
python -m uvicorn backend.main:app --reload
```

### Frontend

Open another terminal and run:

```bash
python -m http.server 5500 --directory frontend
```

Then open the application at:

`http://127.0.0.1:5500`

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Check API status |
| POST | `/predict` | Predict student at-risk probability |
| GET | `/model-info` | Retrieve model information and evaluation metrics |

## Future Improvements
- Improve the web interface.
- Add input validation and user-friendly categorical selections.
- Experiment with additional models and hyperparameters.