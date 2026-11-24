from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

app = Flask(__name__)
CORS(app)

# --- Load dataset ---
df = pd.read_csv("dataset.csv")

# --- Clean numeric columns (remove commas and convert to float) ---
numeric_cols = ["Dose_Gy", "Avg_Temp_C", "Rainfall_mm", "Growth_Days", "Yield_kg_per_acre"]
for col in numeric_cols:
    df[col] = df[col].astype(str).str.replace(",", "").astype(float)

# --- Fill missing categorical columns ---
categorical_features = ["Crop", "Radiation_Type", "Mutation_Trait", "Location", "Soil_Type"]
df['Radiation_Type'] = df['Radiation_Type'].fillna("None")
df['Mutation_Trait'] = df['Mutation_Trait'].fillna("None")

# --- One-Hot Encode categorical features ---
df_encoded = pd.get_dummies(df, columns=categorical_features)

# --- Split dataset ---
X = df_encoded.drop("Yield_kg_per_acre", axis=1)
y = df_encoded["Yield_kg_per_acre"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train model ---
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Calculate model accuracy ---
y_pred_test = model.predict(X_test)
model_mse = mean_squared_error(y_test, y_pred_test)
model_r2 = r2_score(y_test, y_pred_test)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    if not data:
        return jsonify({"error": "No input data provided"}), 400

    input_df = pd.DataFrame([data])

    # Fill missing categorical columns in input
    for cat in ["Radiation_Type", "Mutation_Trait"]:
        if cat not in input_df.columns or pd.isna(input_df[cat][0]):
            input_df[cat] = "None"

    # One-Hot Encode input
    input_df = pd.get_dummies(input_df, columns=categorical_features)

    # Align input columns with training data
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    try:
        prediction = model.predict(input_df)
        return jsonify({
            "Predicted_Yield_kg_per_acre": float(prediction[0]),
            "Model_Accuracy_R2": round(model_r2, 4),
            "Model_Accuracy_MSE": round(model_mse, 4)
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500


@app.route("/data", methods=["GET"])
def get_data():
    """Returns unique values for dropdowns."""
    unique_data = {}
    for feature in categorical_features:
        unique_values = df[feature].fillna("None").unique().tolist()
        unique_data[feature] = sorted(unique_values)
    return jsonify(unique_data)


if __name__ == "__main__":
    print("Model MSE:", model_mse)
    print("Model R2:", model_r2)
    app.run(debug=True, use_reloader=False)
