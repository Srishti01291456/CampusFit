from pathlib import Path
import os

import pandas as pd
from flask import Flask, render_template, request
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "CampusFit_Synthetic_Dataset_10000.csv"
TARGET = "Dropout_Risk"

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "campusfit-local-key")

# Form fields are grouped for a clear dashboard layout.
FIELD_GROUPS = {
    "Personal Details": ["Age", "Gender", "Height_cm", "Weight_kg", "BMI"],
    "Fitness Activity": [
        "Fitness_Goal", "Activity_Level", "Daily_Steps", "Workout_Days",
        "Workout_Duration_Min", "Calories_Burned", "Fitness_Level",
        "Workout_Type", "Streak_Days"
    ],
    "Lifestyle & Nutrition": [
        "Sleep_Hours", "Stress_Level", "Water_Intake_L", "Diet_Type",
        "Daily_Calories", "Protein_g", "Carbs_g", "Fat_g", "Mood",
        "Heart_Rate", "Body_Fat_Percent", "Recommended_Diet", "Hydration_Status"
    ],
}

NUMERIC_FIELDS = {
    "Age": (18, 80, 1), "Height_cm": (120, 220, 0.1),
    "Weight_kg": (30, 200, 0.1), "BMI": (10, 60, 0.1),
    "Daily_Steps": (0, 30000, 1), "Workout_Days": (0, 7, 1),
    "Workout_Duration_Min": (0, 240, 1), "Calories_Burned": (0, 3000, 1),
    "Sleep_Hours": (0, 16, 0.1), "Stress_Level": (1, 10, 1),
    "Water_Intake_L": (0, 10, 0.1), "Daily_Calories": (500, 6000, 1),
    "Protein_g": (0, 400, 1), "Carbs_g": (0, 700, 1),
    "Fat_g": (0, 250, 1), "Heart_Rate": (35, 220, 1),
    "Body_Fat_Percent": (2, 60, 0.1), "Streak_Days": (0, 365, 1),
}

SELECT_OPTIONS = {
    "Gender": ["Male", "Female", "Other"],
    "Fitness_Goal": ["Weight Loss", "Muscle Gain", "Stay Fit", "Improve Endurance"],
    "Activity_Level": ["Low", "Moderate", "High"],
    "Fitness_Level": ["Beginner", "Intermediate", "Advanced"],
    "Workout_Type": ["Yoga", "Cardio", "Strength", "HIIT", "Walking", "Cycling"],
    "Diet_Type": ["Veg", "Non-Veg", "Vegan"],
    "Mood": ["Low", "Neutral", "Happy"],
    "Recommended_Diet": ["Balanced", "Calorie Deficit", "High Protein", "Low Carb"],
    "Hydration_Status": ["Low", "Moderate", "Good"],
}


def load_and_train():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. Place CampusFit_Synthetic_Dataset_10000.csv next to app.py."
        )

    data = pd.read_csv(DATA_PATH)
    if TARGET not in data.columns:
        raise ValueError(f"Dataset must contain the target column '{TARGET}'.")

    # IDs and reward points are excluded to avoid using identifiers / unrelated leakage.
    excluded = [TARGET, "User_ID", "Reward_Points"]
    X = data.drop(columns=[c for c in excluded if c in data.columns])
    y = data[TARGET].astype(str)

    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_cols),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ],
        remainder="drop",
    )
    pipeline = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", DecisionTreeClassifier(max_depth=8, min_samples_leaf=8, class_weight="balanced", random_state=42)),
    ])
    pipeline.fit(X, y)

    # Map transformed feature names to the fitted tree's importance scores.
    try:
        names = pipeline.named_steps["preprocessing"].get_feature_names_out()
        importances = pipeline.named_steps["classifier"].feature_importances_
        pairs = sorted(zip(names, importances), key=lambda item: item[1], reverse=True)[:10]
        importance_labels = [str(name).replace("numeric__", "").replace("categorical__", "") for name, _ in pairs]
        importance_values = [round(float(value), 5) for _, value in pairs]
    except Exception:
