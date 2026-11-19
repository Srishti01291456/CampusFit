import os
import pandas as pd
from flask import Flask, render_template, request
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "CampusFit_Synthetic_Dataset_10000.csv")
df = pd.read_csv(DATA_PATH)
TARGET = "Dropout_Risk"
EXCLUDED = ["User_ID", "Reward_Points", TARGET]
FEATURES = [c for c in df.columns if c not in EXCLUDED]
X, y = df[FEATURES], df[TARGET]
cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()
preprocessor = ColumnTransformer([
    ("numeric", StandardScaler(), num_cols),
    ("categorical", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])
model = Pipeline([("preprocessor", preprocessor),
                  ("classifier", DecisionTreeClassifier(random_state=42, class_weight="balanced"))])
model.fit(X, y)

fields = {
    "Age": ("Age", "number", 20), "Gender": ("Gender", "select", ["Female", "Male"]),
    "Height_cm": ("Height (cm)", "number", 170), "Weight_kg": ("Weight (kg)", "number", 65),
    "BMI": ("BMI", "number", 22.5), "Fitness_Goal": ("Fitness Goal", "select", ["Weight Loss", "Stay Fit", "Muscle Gain"]),
    "Activity_Level": ("Activity Level", "select", ["Low", "Moderate", "High"]),
    "Daily_Steps": ("Daily Steps", "number", 7000), "Workout_Days": ("Workout Days per Week", "number", 3),
    "Workout_Duration_Min": ("Workout Duration (min)", "number", 45), "Calories_Burned": ("Calories Burned", "number", 400),
    "Sleep_Hours": ("Sleep Hours", "number", 7), "Stress_Level": ("Stress Level (1–10)", "number", 4),
    "Water_Intake_L": ("Water Intake (L)", "number", 2.5), "Diet_Type": ("Diet Type", "select", ["Veg", "Non-Veg"]),
    "Daily_Calories": ("Daily Calories", "number", 2200), "Protein_g": ("Protein (g)", "number", 100),
    "Carbs_g": ("Carbs (g)", "number", 250), "Fat_g": ("Fat (g)", "number", 70),
    "Mood": ("Mood", "select", ["Low", "Neutral", "Happy"]), "Heart_Rate": ("Heart Rate", "number", 75),
    "Body_Fat_Percent": ("Body Fat (%)", "number", 22), "Fitness_Level": ("Fitness Level", "select", ["Beginner", "Intermediate", "Advanced"]),
    "Workout_Type": ("Workout Type", "select", ["Yoga", "Strength", "Mixed", "Cardio", "HIIT"]),
    "Recommended_Diet": ("Recommended Diet", "select", ["Calorie Deficit", "Balanced", "High Protein"]),
    "Hydration_Status": ("Hydration Status", "select", ["Low", "Good"]), "Streak_Days": ("Current Workout Streak (days)", "number", 10)
}

# Dataset-level charts are computed from the supplied synthetic dataset.
risk_counts = df[TARGET].value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
class_counts = {str(k): int(v) for k, v in risk_counts.items()}
class_pct = {k: round(v / len(df) * 100, 1) for k, v in class_counts.items()}

# Global feature importance from the fitted Decision Tree pipeline.
feature_names = model.named_steps["preprocessor"].get_feature_names_out()
importances = model.named_steps["classifier"].feature_importances_
importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
importance_df["feature"] = importance_df["feature"].str.replace(r"^(numeric|categorical)__", "", regex=True)
importance_df = importance_df.sort_values("importance", ascending=False).head(8)
importance_labels = importance_df["feature"].tolist()
importance_values = [round(float(v), 5) for v in importance_df["importance"].tolist()]


def recommendations(row, risk):
    workouts, lifestyle, nutrition = [], [], []
    level = row.get("Fitness_Level", "Beginner")
    if row.get("Daily_Steps", 0) < 6000:
        workouts.append("Start with a 20–30 minute brisk walk on most days; increase steps gradually.")
    if row.get("Workout_Days", 0) < 3:
        workouts.append("Build consistency with 2–3 manageable sessions per week, with rest days between harder sessions.")
    if row.get("Workout_Duration_Min", 0) < 30:
        workouts.append("Begin with 15–20 minute low-impact sessions and increase duration as fitness improves.")
    if not workouts:
        workouts.append("Maintain your current routine. Combine cardio, strength training and mobility work across the week.")
    if level == "Beginner":
        workouts.append("Prefer beginner-friendly bodyweight strength exercises, cycling or low-impact cardio; avoid sudden intensity increases.")
    elif row.get("Workout_Type") == "Yoga":
        workouts.append("Complement yoga with light walking and two short strength sessions if suitable for you.")
    else:
        workouts.append("Vary your sessions and include recovery or mobility work to help maintain consistency.")

    if row.get("Sleep_Hours", 0) < 7:
        lifestyle.append("Your reported sleep is below 7 hours; consider a consistent sleep schedule and adequate recovery.")
    if row.get("Stress_Level", 0) >= 7:
        lifestyle.append("Your reported stress is high; consider gentle activity, relaxation breaks and support when needed.")
    if row.get("Streak_Days", 0) < 3:
        lifestyle.append("Set a small, realistic weekly activity goal and track progress rather than aiming for a long streak immediately.")
    if row.get("Water_Intake_L", 0) < 2:
        lifestyle.append("Review your hydration during the day; fluid needs vary with body size, climate and activity.")
    if not lifestyle:
        lifestyle.append("Keep tracking sleep, stress, hydration and activity so you can spot changes in your routine.")

    if row.get("Protein_g", 0) < 60:
        nutrition.append("Review whether your meals include suitable protein sources such as pulses, dairy, eggs, tofu or lean meats.")
    if row.get("Diet_Type") == "Veg":
        nutrition.append("For vegetarian meals, consider dal, beans, paneer, tofu, curd, nuts and seeds as varied protein sources.")
    else:
        nutrition.append("Aim for balanced meals with vegetables, whole grains and suitable protein sources.")
    nutrition.append("Avoid treating calorie or macro targets as medical advice; individual nutrition needs differ.")

    risk_advice = {
        "High": "The model assigned a high-risk category. Review the suggested habit changes and consider a realistic support plan.",
        "Medium": "The model assigned a medium-risk category. Focus on consistency and monitor the habits highlighted below.",
        "Low": "The model assigned a low-risk category. Maintain sustainable habits and continue monitoring your routine."
    }.get(str(risk), "Review the prediction alongside your own circumstances.")
    return workouts, lifestyle, nutrition, risk_advice


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    row = None
    probs = None
    workouts = lifestyle = nutrition = []
    risk_advice = None
    risk_score = None
    if request.method == "POST":
        row = {}
        for key in FEATURES:
            kind = fields[key][1]
            raw = request.form.get(key, str(fields[key][2]))
            row[key] = raw if kind == "select" else float(raw)
        sample = pd.DataFrame([row], columns=FEATURES)
        result = str(model.predict(sample)[0])
        raw_probs = model.predict_proba(sample)[0]
        probs = {str(label): round(float(p) * 100, 1) for label, p in zip(model.classes_, raw_probs)}
        risk_score = round(probs.get("Medium", 0) * 0.5 + probs.get("High", 0), 1)
        workouts, lifestyle, nutrition, risk_advice = recommendations(row, result)
    return render_template(
        "index.html", fields=fields, result=result, row=row, probs=probs,
        risk_score=risk_score, workouts=workouts, lifestyle=lifestyle,
        nutrition=nutrition, risk_advice=risk_advice,
        class_counts=class_counts, class_pct=class_pct,
        importance_labels=importance_labels, importance_values=importance_values,
        total_records=len(df),
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
