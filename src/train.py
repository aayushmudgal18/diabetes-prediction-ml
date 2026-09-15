import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

from preprocessing import load_data

# Load data
X, y = load_data()

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Machine Learning Pipeline
# Imputer runs first so the placeholder-zero-turned-NaN values (from
# preprocessing.py) are filled in before scaling and classification.
# CalibratedClassifierCV replaces SVC(probability=True), which is
# deprecated as of scikit-learn 1.9.
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", CalibratedClassifierCV(SVC(), ensemble=False))
])

# Hyperparameter tuning
# Params are nested under model__estimator__ because CalibratedClassifierCV
# wraps the underlying SVC.
param_grid = {
    "model__estimator__C": [0.1, 1, 10, 100],
    "model__estimator__kernel": ["linear", "rbf"],
    "model__estimator__gamma": ["scale", "auto"]
}

# GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="f1"
)

# Train model
grid_search.fit(X_train, y_train)

# Best parameters
print("Best Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation F1:")
print(grid_search.best_score_)

# Save trained model
with open("models/diabetes_model.pkl", "wb") as file:
    pickle.dump(grid_search.best_estimator_, file)

print("\nModel saved successfully!")