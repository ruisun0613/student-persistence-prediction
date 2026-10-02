import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

TARGET_COLUMN = "Persistence"

DROP_COLUMNS = ["School"]

NUMERICAL_FEATURES = [
    "FirstTermGpa",
    "SecondTermGpa",
    "HighSchoolAvgMark",
    "MathScore",
    "EnglishGrade"
]

CATEGORICAL_FEATURES = [
    "FirstLanguage",
    "Funding",
    "FastTrack",
    "Coop",
    "Residency",
    "Gender",
    "PrevEducation",
    "AgeGroup"
]

def clean_data(df):
    df = df.copy()

    # 1. clean the "?"
    df = df.replace("?", np.nan)
    # 2. numerical columns -> numeric
    for col in NUMERICAL_FEATURES:
        df[col] = pd.to_numeric(df[col], errors = 'coerce')
    # 3. drop meaningless columns.
    df = df.drop(columns = DROP_COLUMNS)

    return df

numerical_pipeline = Pipeline([
    # 1. missing value -> median
    ("imputer",SimpleImputer(strategy = "median")),
    # 2. StandardScaler
    ("scaler",StandardScaler())
])

categorical_pipeline = Pipeline([
    # 1.missing value
    ("imputer", SimpleImputer(strategy = "constant", fill_value = "Unknown")),
    # 2.OneHotEncoder
    ("encoder", OneHotEncoder(handle_unknown = "ignore"))
])

preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, NUMERICAL_FEATURES),
    ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
])

if __name__ == "__main__":
    # test
    from data_loader import load_students_data
    raw_df = load_students_data()
    clean_df = clean_data(raw_df)

    print(clean_df.head())
    print(clean_df.dtypes)
    print(clean_df[NUMERICAL_FEATURES].dtypes)
    print(clean_df.isna().sum())
    print(clean_df.shape)