"""
MediSense AI v2 — Standard API Response Helpers
All endpoints return:
  { "status": "success"|"error", "data": {...}|null, "message": "...", "timestamp": "..." }
"""
from flask import jsonify
from datetime import datetime, timezone

DISCLAIMER = (
    "⚠️ This is an AI-assisted prediction for educational purposes only. "
    "It is NOT a medical diagnosis and should never replace professional medical advice. "
    "Please consult a certified and licensed healthcare professional for any health concerns."
)

def _ts():
    return datetime.now(timezone.utc).isoformat()

def success(data=None, message="OK", status_code=200):
    return jsonify({
        "status":    "success",
        "message":   message,
        "data":      data,
        "timestamp": _ts(),
    }), status_code

def error(message="An error occurred", status_code=400, details=None):
    body = {
        "status":    "error",
        "message":   message,
        "data":      None,
        "timestamp": _ts(),
    }
    if details:
        body["details"] = details
    return jsonify(body), status_code

def prediction_response(matched_symptoms, predictions):
    """Wrap a prediction result in the standard envelope with disclaimer."""
    return success(data={
        "matched_symptoms": matched_symptoms,
        "predictions":      predictions,
        "disclaimer":       DISCLAIMER,
    }, message="Prediction generated successfully")
