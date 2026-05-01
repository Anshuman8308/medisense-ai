"""
MediSense AI v2 — Prediction Routes
POST /api/predict/symptoms  — symptom list or free text
POST /api/extract-symptoms  — NLP extraction only
GET  /api/symptoms          — full symptoms list
GET  /api/diseases          — full disease list
GET  /api/model/info        — model metrics & metadata
"""
from flask import Blueprint, request, current_app
from backend.models.response import success, error, prediction_response

predict_bp = Blueprint("predict", __name__)


def _svc():
    return current_app.config["PREDICTION_SERVICE"]


@predict_bp.route("/api/symptoms", methods=["GET"])
def get_symptoms():
    svc = _svc()
    if not svc.ready:
        return error("Model not loaded. Run ml/train_model.py first.", 503)
    return success(data={"symptoms": svc.symptoms_list,
                         "count": len(svc.symptoms_list)})


@predict_bp.route("/api/diseases", methods=["GET"])
def get_diseases():
    svc = _svc()
    if not svc.ready:
        return error("Model not loaded.", 503)
    return success(data={"diseases": list(svc.label_encoder.classes_),
                         "count": len(svc.label_encoder.classes_)})


@predict_bp.route("/api/model/info", methods=["GET"])
def model_info():
    return success(data=_svc().get_model_info())


@predict_bp.route("/api/predict/symptoms", methods=["POST"])
def predict_from_symptoms():
    svc  = _svc()
    if not svc.ready:
        return error("Model not loaded.", 503)

    body     = request.get_json(silent=True) or {}
    symptoms = body.get("symptoms", [])
    text     = body.get("text", "").strip()

    # Validate input
    if not symptoms and not text:
        return error("Provide either 'symptoms' (list) or 'text' (string).", 400)

    # NLP extraction from free text
    if not symptoms and text:
        symptoms = svc.extract_symptoms_from_text(text)
        if not symptoms:
            return error(
                "Could not identify any known symptoms in your text. "
                "Try being more specific (e.g. 'I have fever, headache, and joint pain').",
                422,
                details={"input_text": text}
            )

    # Validate against known symptoms
    valid, invalid = svc.validate_symptoms(symptoms)
    if not valid:
        return error(
            "None of the provided symptoms are recognised by the model.",
            422,
            details={"invalid": invalid, "hint": "Use /api/symptoms to see all valid symptom names."}
        )

    try:
        predictions = svc.predict(valid, top_n=3)
    except Exception as e:
        return error(f"Prediction failed: {str(e)}", 500)

    return prediction_response(
        matched_symptoms=valid,
        predictions=predictions,
    )


@predict_bp.route("/api/extract-symptoms", methods=["POST"])
def extract_symptoms():
    svc  = _svc()
    body = request.get_json(silent=True) or {}
    text = body.get("text", "").strip()
    if not text:
        return error("'text' field is required.", 400)
    found = svc.extract_symptoms_from_text(text)
    return success(data={"symptoms": found, "count": len(found),
                         "hint": "Pass these to /api/predict/symptoms for a diagnosis."})


@predict_bp.route("/api/chat", methods=["POST"])
def chat():
    """
    Chat endpoint: extracts symptoms from a message, runs prediction if >= 2 found.
    Returns structured response for the frontend chat UI.
    """
    svc  = _svc()
    body = request.get_json(silent=True) or {}
    msg  = body.get("message", "").strip()

    if not msg:
        return error("'message' field is required.", 400)

    found = svc.extract_symptoms_from_text(msg)
    valid, _ = svc.validate_symptoms(found)

    result: dict = {
        "found_symptoms":  found,
        "valid_symptoms":  valid,
        "should_predict":  len(valid) >= 2,
        "predictions":     None,
        "matched_symptoms": [],
    }

    if len(valid) >= 2:
        try:
            result["predictions"]     = svc.predict(valid, top_n=3)
            result["matched_symptoms"] = valid
        except Exception:
            pass

    return success(data=result)
