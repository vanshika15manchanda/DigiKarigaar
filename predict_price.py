from pathlib import Path

import joblib
import pandas as pd
from flask import Blueprint, jsonify, request


price_bp = Blueprint("price_prediction", __name__)

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "artisan_price_model.pkl"
)

price_model = joblib.load(MODEL_PATH)


@price_bp.route("/api/predict-price", methods=["POST"])
def predict_price():
    data = request.get_json(silent=True) or {}

    required_fields = [
        "category",
        "material",
        "size"
    ]

    missing_fields = [
        field
        for field in required_fields
        if not data.get(field)
    ]

    if missing_fields:
        return jsonify({
            "error": "Required fields are missing.",
            "missing_fields": missing_fields
        }), 400

    product = pd.DataFrame([
        {
            "category": str(data["category"]).strip(),
            "material": str(data["material"]).strip(),
            "size": str(data["size"]).strip()
        }
    ])

    try:
        predicted_price = float(
            price_model.predict(product)[0]
        )

        minimum_price = predicted_price * 0.90
        maximum_price = predicted_price * 1.10

        return jsonify({
            "predicted_price": round(predicted_price),
            "price_range": [
                round(minimum_price),
                round(maximum_price)
            ]
        }), 200

    except Exception as error:
        return jsonify({
            "error": "Price prediction failed.",
            "details": str(error)
        }), 500