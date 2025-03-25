import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load the pickled model
MODEL_PATH = "xgb_vomitoxin_model.pickle.pkl"

try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None


@app.route("/")
def home():
    """
    Home page with a simple interface for prediction
    """
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Prediction endpoint for vomitoxin prediction

    Expected input: Form data or JSON with features
    Returns predicted vomitoxin level
    """
    try:
        # Check if request is JSON or form data
        if request.is_json:
            data = request.get_json(force=True)
            features = np.array(data["features"]).reshape(1, -1)
        else:
            # Handle form data from HTML input
            features_list = []
            for i in range(1, len(request.form) + 1):
                features_list.append(float(request.form.get(f"feature{i}", 0)))
            features = np.array(features_list).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)

        # If called via AJAX/API
        if request.is_json:
            return jsonify({"prediction": float(prediction[0]), "status": "success"})

        # If called from web form
        return render_template("result.html", prediction=float(prediction[0]))

    except Exception as e:
        logger.error(f"Prediction error: {e}")

        # Handle different response types
        if request.is_json:
            return jsonify({"error": str(e), "status": "failed"}), 500
        else:
            return render_template("error.html", error=str(e))


@app.route("/health")
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "healthy", "model_loaded": model is not None})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
