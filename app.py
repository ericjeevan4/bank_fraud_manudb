from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os
from pymongo import MongoClient

app = Flask(__name__)
CORS(app)

# =========================================================
# MONGODB CONNECTION
# =========================================================

MONGO_URI = os.environ.get("MONGO_URI")

if not MONGO_URI:
    raise ValueError("MONGO_URI environment variable is not set")

mongo_client = MongoClient(MONGO_URI)

mongo_db = mongo_client["bank_fraud_db"]

manual_accounts_collection = mongo_db["manual_accounts"]

# =========================================================
# LOAD SAVED FRAUD DETECTION MODEL
# =========================================================

MODEL_PATH = "models/lightgbm_fraud_pipeline.joblib"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

best_model = joblib.load(MODEL_PATH)


# =========================================================
# MODEL FEATURES
# =========================================================

trained_features = [
    "TransactionAmount",
    "TransactionType",
    "Location",
    "Channel",
    "CustomerAge",
    "CustomerOccupation",
    "AccountBalance",
    "AnnualIncome",
    "CurrentAddressMonthCount",
    "PreviousAddressMonthCount"
]

categorical_features = [
    "TransactionType",
    "Location",
    "Channel",
    "CustomerOccupation"
]


# =========================================================
# FRAUD PREDICTION FUNCTION
# =========================================================

def predict_account_fraud(input_data):

    # Extract only the features used by the model
    model_input = pd.DataFrame(
        [[input_data[feature] for feature in trained_features]],
        columns=trained_features
    )

    # Convert categorical features to string
    for col in categorical_features:
        model_input[col] = model_input[col].astype(str)

    # Prediction
    prediction = best_model.predict(model_input)[0]

    # Fraud probability
    fraud_probability = best_model.predict_proba(
        model_input
    )[0][1]

    # Final result
    if prediction == 1:
        result = "THIS IS A FRAUD ACCOUNT"
    else:
        result = "THIS IS A NON FRAUD ACCOUNT"

    return {
        "prediction": int(prediction),
        "result": result,
        "fraud_probability": float(fraud_probability),
        "fraud_percentage": float(fraud_probability * 100)
    }


# =========================================================
# HOME ROUTE
# =========================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message": "Fraud Detection API is running",
        "status": "success"
    })


# =========================================================
# PREDICT API
# =========================================================

@app.route("/api/predict", methods=["POST"])
def predict():

    try:

        # Get JSON data
        input_data = request.get_json()

        if not input_data:
            return jsonify({
                "error": "No JSON data provided"
            }), 400

        # Check required fields
        missing_features = [
            feature
            for feature in trained_features
            if feature not in input_data
        ]

        if missing_features:

            return jsonify({
                "error": "Missing required model features",
                "missing_features": missing_features
            }), 400

        # Predict
        result = predict_account_fraud(input_data)

        # Return API response
        return jsonify({
            "status": "success",
            "data": result
        })

    except Exception as e:

        return jsonify({
            "status": "error",
            "error": type(e).__name__,
            "message": str(e)
        }), 500


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
