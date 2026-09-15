import numpy as np
import pandas as pd

# These columns use 0 as a placeholder for "not measured" in this dataset.
# A real patient can't have 0 blood pressure, 0 BMI, etc., so these zeros
# are missing values, not genuine measurements, and need to be treated
# as such before scaling/modeling.
ZERO_AS_MISSING = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]


def load_data():
    df = pd.read_csv("data/diabetes.csv")

    # Convert placeholder zeros to NaN so they get imputed instead of
    # being treated as the literal lowest value in the feature's range.
    df[ZERO_AS_MISSING] = df[ZERO_AS_MISSING].replace(0, np.nan)

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    return X, y