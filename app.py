
from pathlib import Path
import os

import pandas as pd
from flask import Flask, render_template, request
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


# --------------------------------------------------
# 1. APPLICATION CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "CampusFit_Synthetic_Dataset_10000.csv"
TARGET = "Dropout_Risk"

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get(
    "SECRET_KEY", "campusfit-local-key"
)


# --------------------------------------------------
# 2. DASHBOARD INPUT FIELDS
# --------------------------------------------------

FIELD_GROUPS = {
    "Personal Details": [
        "Age", "Gender", "Height_cm", "Weight_kg", "BMI"
    ],
    "Fitness Activity": [
        "Fitness_Goal", "Activity_Level", "Daily_Steps",
        "Workout_Days", "Workout_Duration_Min",
        "Calories_Burned", "Fitness_Level",
        "Workout_Type", "Streak_Days"
    ],
    "Lifestyle & Nutrition": [
        "Sleep_Hours", "Stress_Level", "Water_Intake_L",
        "Diet_Type", "Daily_Calories", "Protein_g",
        "Carbs_g", "Fat_g", "Mood", "Heart_Rate",
        "Body_Fat_Percent", "Recommended_Diet",
        "Hydration_Status"
    ],
}

NUMERIC_FIELDS = {
    "Age": (18, 80, 1),
    "Height_cm": (120, 220, 0.1),
    "Weight_kg": (30, 200, 0.1),
    "BMI": (10, 60, 0.1),
    "Daily_Steps": (0, 30000, 1),
    "Workout_Days": (0, 7, 1),
    "Workout_Duration_Min": (0, 240, 1),
    "Calories_Burned": (0, 3000, 1),
    "Sleep_Hours": (0, 16, 0.1),
    "Stress_Level": (1, 10, 1),
    "Water_Intake_L": (0, 10, 0.1),
    "Daily_Calories": (500, 6000, 1),
    "Protein_g": (0, 400, 1),
    "Carbs_g": (0, 700, 1),
    "Fat_g": (0, 250, 1),
    "Heart_Rate": (35, 220, 1),
    "Body_Fat_Percent": (2, 60, 0.1),
    "Streak_Days": (0, 365, 1),
}

SELECT_OPTIONS = {
    "Gender": ["Male", "Female", "Other"],
    "Fitness_Goal": [
        "Weight Loss", "Muscle Gain",
        "Stay Fit", "Improve Endurance"
    ],
    "Activity_Level": ["Low", "Moderate", "High"],
    "Fitness_Level": ["Beginner", "Intermediate", "Advanced"],
    "Workout_Type": [
        "Yoga", "Cardio", "Strength",
        "HIIT", "Walking", "Cycling"
    ],
    "Diet_Type": ["Veg", "Non-Veg", "Vegan"],
    "Mood": ["Low", "Neutral", "Happy"],
    "Recommended_Diet": [
        "Balanced", "Calorie Deficit",
        "High Protein", "Low Carb"
    ],
    "Hydration_Status": ["Low", "Moderate", "Good"],
}

ALL_FIELDS = [
    field
    for group in FIELD_GROUPS.values()
    for field in group
]

RISK_CLASSES = ["Low", "Medium", "High"]


def make_field_label(field):
    return field.replace("_", " ")


# The HTML template expects:
# fields[key] = (label, input_type, default_or_options)

FIELDS = {}

for field in ALL_FIELDS:
    if field in SELECT_OPTIONS:
        FIELDS[field] = (
            make_field_label(field),
            "select",
            SELECT_OPTIONS[field]
        )
    else:
        minimum, maximum, step = NUMERIC_FIELDS[field]
        FIELDS[field] = (
            make_field_label(field),
            "number",
            minimum
        )


# --------------------------------------------------
# 3. LOAD DATASET AND TRAIN MODEL
# --------------------------------------------------

def load_and_train():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. "
            "Keep the CSV file in the same folder as app.py."
        )

    data = pd.read_csv(DATA_PATH)

    if TARGET not in data.columns:
        raise ValueError(
            f"Dataset must contain the '{TARGET}' column."
        )

    missing_columns = [
        column for column in ALL_FIELDS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "The following dashboard fields are missing "
            f"from the dataset: {missing_columns}"
        )

    X = data[ALL_FIELDS].copy()
    y = data[TARGET].astype(str).str.strip().str.title()

    numeric_columns = [
        column for column in ALL_FIELDS
        if column in NUMERIC_FIELDS
    ]

    categorical_columns = [
        column for column in ALL_FIELDS
        if column in SELECT_OPTIONS
    ]

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                numeric_columns
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns
            ),
        ],
        remainder="drop"
    )

    pipeline = Pipeline([
        ("preprocessing", preprocessing),
        (
            "classifier",
            DecisionTreeClassifier(
                max_depth=8,
                min_samples_leaf=8,
                class_weight="balanced",
                random_state=42
            )
        )
    ])

    pipeline.fit(X, y)

    # Dataset risk distribution
    counts = y.value_counts()

    class_counts = {
        risk: int(counts.get(risk, 0))
        for risk in RISK_CLASSES
    }

    total_records = len(data)

    class_pct = {
        risk: round(
            class_counts[risk] / total_records * 100, 2
        ) if total_records else 0
        for risk in RISK_CLASSES
    }

    # Feature importance
    importance_labels = []
    importance_values = []

    try:
        names = (
            pipeline.named_steps["preprocessing"]
            .get_feature_names_out()
        )

        importances = (
            pipeline.named_steps["classifier"]
            .feature_importances_
        )

        pairs = sorted(
            zip(names, importances),
            key=lambda item: item[1],
            reverse=True
        )[:10]

        importance_labels = [
            str(name)
            .replace("numeric__", "")
            .replace("categorical__", "")
            for name, _ in pairs
        ]

        importance_values = [
            round(float(value), 5)
            for _, value in pairs
        ]

    except Exception as e:
        print("Feature importance calculation error:", e)

    return (
        pipeline,
        class_counts,
        class_pct,
        importance_labels,
        importance_values,
        total_records
    )


(
    model,
    class_counts,
    class_pct,
    importance_labels,
    importance_values,
    total_records
) = load_and_train()


# --------------------------------------------------
# 4. RECOMMENDATION FUNCTIONS
# --------------------------------------------------

def get_workout_recommendations(profile):

    recommendations = []

    activity = profile["Activity_Level"]
    workout_days = profile["Workout_Days"]
    duration = profile["Workout_Duration_Min"]
    goal = profile["Fitness_Goal"]

    if activity == "Low" or workout_days <= 2:
        recommendations.append(
            "Start with 20-30 minutes of walking on most days."
        )
        recommendations.append(
            "Gradually build up to three or four workout days per week."
        )

    elif activity == "Moderate":
        recommendations.append(
            "Maintain a consistent routine with cardio and strength training."
        )
        recommendations.append(
            "Include recovery days between demanding sessions."
        )

    else:
        recommendations.append(
            "Maintain a balanced combination of strength and cardio."
        )
        recommendations.append(
            "Monitor recovery and avoid excessive training."
        )

    if goal == "Weight Loss":
        recommendations.append(
            "Combine regular walking or cardio with strength exercises."
        )

    elif goal == "Muscle Gain":
        recommendations.append(
            "Include progressive resistance training and adequate recovery."
        )

    elif goal == "Improve Endurance":
        recommendations.append(
            "Gradually increase cardio duration and intensity."
        )

    else:
        recommendations.append(
            "Use a balanced routine involving mobility, cardio and strength."
        )

    if duration < 20:
        recommendations.append(
            "Increase workout duration gradually as your fitness improves."
        )

    return recommendations


def get_nutrition_recommendations(profile):

    recommendations = []

    diet = profile["Diet_Type"]
    water = profile["Water_Intake_L"]
    protein = profile["Protein_g"]

    if diet == "Vegan":
        recommendations.append(
            "Include plant-based protein sources such as beans, lentils and tofu."
        )
    elif diet == "Veg":
        recommendations.append(
            "Consider dairy, lentils, beans and other vegetarian protein sources."
        )
    else:
        recommendations.append(
            "Include a balanced variety of protein sources, vegetables and whole grains."
        )

    if water < 2:
        recommendations.append(
            "Review your hydration habits and drink water regularly throughout the day."
        )
    else:
        recommendations.append(
            "Continue maintaining regular hydration throughout the day."
        )

    if protein < 50:
        recommendations.append(
            "Review your daily protein intake and include suitable protein-rich foods."
        )
    else:
        recommendations.append(
            "Maintain a balanced intake of protein and other essential nutrients."
        )

    recommendations.append(
        "Prefer balanced meals and avoid relying on highly processed foods."
    )

    return recommendations


def get_lifestyle_recommendations(profile):

    recommendations = []

    sleep = profile["Sleep_Hours"]
    stress = profile["Stress_Level"]
    mood = profile["Mood"]
    steps = profile["Daily_Steps"]

    if sleep < 7:
        recommendations.append(
            "Work towards a consistent sleep schedule and sufficient rest."
        )
    else:
        recommendations.append(
            "Continue maintaining a consistent sleep routine."
        )

    if stress >= 7:
        recommendations.append(
            "Consider relaxation activities, breathing exercises or short breaks."
        )

    if mood == "Low":
        recommendations.append(
            "Try enjoyable physical activities and maintain social support."
        )

    if steps < 5000:
        recommendations.append(
            "Gradually increase daily movement through short walking breaks."
        )

    if not recommendations:
        recommendations.append(
            "Continue monitoring your daily habits and maintaining consistency."
        )

    return recommendations


def get_risk_advice(risk):
    advice = {
        "Low": (
            "Your profile is classified as low dropout risk. "
            "Continue maintaining consistent fitness habits."
        ),
        "Medium": (
            "Your profile indicates moderate dropout risk. "
            "Focus on building a consistent workout routine "
            "and reviewing your lifestyle habits."
        ),
        "High": (
            "Your profile indicates high dropout risk. "
            "Consider setting manageable fitness goals, "
            "building a regular routine and seeking appropriate support."
        )
    }

    return advice.get(
        risk,
        "Review your fitness habits and maintain a consistent routine."
    )


# --------------------------------------------------
# 5. HOMEPAGE AND PREDICTION ROUTE
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    row = None
    probs = {}

    risk_score = 0
    workouts = []
    lifestyle = []
    nutrition = []
    risk_advice = ""

    if request.method == "POST":

        try:
            submitted_values = {}

            # Read all form inputs
            for field in ALL_FIELDS:

                raw_value = request.form.get(field, "").strip()

                if not raw_value:
                    raise ValueError(
                        f"Please enter a value for {field.replace('_', ' ')}."
                    )

                if field in SELECT_OPTIONS:

                    if raw_value not in SELECT_OPTIONS[field]:
                        raise ValueError(
                            f"Invalid selection for {field}."
                        )

                    submitted_values[field] = raw_value

                else:
                    value = float(raw_value)

                    minimum, maximum, step = NUMERIC_FIELDS[field]

                    if not minimum <= value <= maximum:
                        raise ValueError(
                            f"{field.replace('_', ' ')} must be "
                            f"between {minimum} and {maximum}."
                        )

                    submitted_values[field] = value

            row = submitted_values

            # Build prediction dataframe in training column order
            input_data = pd.DataFrame(
                [submitted_values],
                columns=ALL_FIELDS
            )

            # Generate prediction
            prediction = model.predict(input_data)[0]
            result = str(prediction).strip().title()

            if result not in RISK_CLASSES:
                raise ValueError(
                    f"Unexpected model prediction: {result}"
                )

            # Calculate class probabilities
            raw_probabilities = model.predict_proba(input_data)[0]
            model_classes = model.named_steps["classifier"].classes_

            probs = {
                risk: round(
                    float(
                        sum(
                            probability
                            for class_name, probability
                            in zip(model_classes, raw_probabilities)
                            if str(class_name).strip().title() == risk
                        )
                    ) * 100,
                    2
                )
                for risk in RISK_CLASSES
            }

            # Probability-weighted indicator (not a clinical score)
            risk_score = round(
                probs["High"] + (0.5 * probs["Medium"]),
                1
            )

            # Generate recommendations
            workouts = get_workout_recommendations(row)
            nutrition = get_nutrition_recommendations(row)
            lifestyle = get_lifestyle_recommendations(row)

            risk_advice = get_risk_advice(result)

        except ValueError as e:
            # Show form validation errors without crashing the app
            app.logger.warning("Form validation: %s", e)
            return (
                f"Input error: {e}",
                400
            )

        except Exception:
            app.logger.exception("Prediction failed")
            return (
                "The prediction could not be completed. "
                "Please check the application logs.",
                500
            )

    return render_template(
        "index.html",
        fields=FIELDS,
        result=result,
        row=row,
        probs=probs,
        risk_score=risk_score,
        workouts=workouts,
        lifestyle=lifestyle,
        nutrition=nutrition,
        risk_advice=risk_advice,
        class_counts=class_counts,
        class_pct=class_pct,
        importance_labels=importance_labels,
        importance_values=importance_values,
        total_records=total_records
    )


# --------------------------------------------------
# 6. HEALTH CHECK ROUTE
# --------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return {
        "status": "healthy",
        "application": "CampusFit",
        "model": "Decision Tree"
    }, 200


# --------------------------------------------------
# 7. LOCAL EXECUTION
# --------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000)),
        debug=False
    )
