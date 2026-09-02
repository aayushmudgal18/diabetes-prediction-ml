import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from preprocessing import load_data

# Load data
X,y = load_data()

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split( X,y,test_size=0.2,random_state=42,stratify=y)

# Machine Learning Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC(probability=True))
])

# Hyperparameter tuning
param_grid = {
    "model__C": [0.1, 1, 10, 100],
    "model__kernel": ["linear", "rbf"],
    "model__gamma": ["scale", "auto"]
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