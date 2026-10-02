from pathlib import Path
import pandas as pd

# ========= Path settings =========
ROOT_DIR = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT_DIR / "dataset" / "Student_data.csv"

# Column definitions
columns = [
        "FirstTermGpa", "SecondTermGpa", "FirstLanguage", "Funding",
        "School", "FastTrack", "Coop", "Residency", "Gender",
        "PrevEducation", "AgeGroup", "HighSchoolAvgMark",
        "MathScore", "EnglishGrade", "Persistence"
    ]

# ========= Load data =========
def load_students_data():
    raw_lines = CSV_PATH.read_text(encoding = 'utf-8').splitlines()
    data_start = next(i for i, line in enumerate(raw_lines)
                      if line.strip() and line.strip()[0].isdigit())

    clean_data = "\n".join(raw_lines[data_start:])

    df = pd.read_csv(pd.io.common.StringIO(clean_data), names = columns)
    return df

if __name__ == "__main__":
    df = load_students_data()

    print(df.head())
    print(df.shape)
    print(df.dtypes)

    print("\nMissing (?) count:")
    print((df == "?").sum())

    print("\nUnique values:")
    for col in df.columns:
        print(f"\n{col}:")
        print(df[col].unique())

    print("\nPersistence distribution:")
    print(df["Persistence"].value_counts())