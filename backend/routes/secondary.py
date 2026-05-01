"""
MediSense AI v2 — Secondary Routes
GET  /api/doctors           — doctor recommendations
POST /api/counsel           — mental wellness chatbot
GET  /api/reminders         — daily health reminders
GET  /api/trending          — trending health tips & alerts
POST /api/report            — generate downloadable PDF report
GET  /api/health            — system health check
"""
from flask import Blueprint, request, current_app, send_file
import io

from backend.models.response import success, error

secondary_bp = Blueprint("secondary", __name__)


# ── Doctor recommendations ─────────────────────────────────────────────────────
@secondary_bp.route("/api/doctors", methods=["GET"])
def get_doctors():
    svc  = current_app.config["DOCTOR_SERVICE"]
    spec = request.args.get("specialization", "").strip()
    try:
        lat  = float(request.args.get("lat"))  if request.args.get("lat")  else None
        lng  = float(request.args.get("lng"))  if request.args.get("lng")  else None
        radius = float(request.args.get("radius_km", 10))
    except ValueError:
        return error("lat, lng, and radius_km must be numeric.", 400)

    doctors = svc.find_doctors(
        specialization=spec,
        user_lat=lat, user_lng=lng,
        radius_km=radius,
    )
    return success(data={
        "doctors": doctors,
        "count":   len(doctors),
        "source":  "google_maps" if svc.use_live else "demo",
        "note":    "" if svc.use_live else (
            "Set GOOGLE_MAPS_API_KEY in .env for real nearby doctor search."
        ),
    })


# ── Counselling / Wellness ─────────────────────────────────────────────────────
@secondary_bp.route("/api/counsel", methods=["POST"])
def counsel():
    svc  = current_app.config["COUNSELLING_SERVICE"]
    body = request.get_json(silent=True) or {}
    msg  = body.get("message", "").strip()
    mood = body.get("mood", "").strip()

    if not msg:
        return error("'message' field is required.", 400)

    result = svc.respond(msg, mood=mood)
    return success(data=result, message="Wellness response generated.")


@secondary_bp.route("/api/reminders", methods=["GET"])
def get_reminders():
    svc = current_app.config["COUNSELLING_SERVICE"]
    return success(data={"reminders": svc.get_all_reminders()})


# ── Trending tips ──────────────────────────────────────────────────────────────
@secondary_bp.route("/api/trending", methods=["GET"])
def trending():
    return success(data={
        "tips": [
            {"title":"Stay Hydrated",    "body":"Drink 8 glasses of water daily for optimal body function.",                      "icon":"water_drop"},
            {"title":"Regular Exercise", "body":"30 min of moderate activity 5 days/week cuts chronic disease risk by 30%.",       "icon":"directions_run"},
            {"title":"Sleep Hygiene",    "body":"Adults need 7–9 hours of quality sleep. Poor sleep links to diabetes and depression.", "icon":"bedtime"},
            {"title":"Balanced Diet",    "body":"Eat 5 servings of fruits and vegetables daily. Cut processed sugars.",            "icon":"restaurant"},
            {"title":"Mental Wellness",  "body":"Practice mindfulness daily. Chronic stress weakens immunity significantly.",      "icon":"self_improvement"},
            {"title":"Hand Hygiene",     "body":"Washing hands for 20 seconds prevents up to 21% of respiratory infections.",     "icon":"clean_hands"},
        ],
        "trending": [
            {"disease":"Seasonal Flu",    "severity":"moderate","cases":"Rising",        "precaution":"Get vaccinated, wash hands frequently"},
            {"disease":"Dengue",          "severity":"high",    "cases":"High Alert",    "precaution":"Eliminate stagnant water, use mosquito repellent"},
            {"disease":"Common Cold",     "severity":"low",     "cases":"Seasonal Peak", "precaution":"Boost immunity with Vitamin C"},
            {"disease":"Conjunctivitis",  "severity":"low",     "cases":"Moderate",      "precaution":"Avoid touching eyes, wash hands regularly"},
            {"disease":"Typhoid",         "severity":"moderate","cases":"Monitoring",    "precaution":"Drink clean water, avoid street food"},
        ],
    })


# ── PDF Report ─────────────────────────────────────────────────────────────────
@secondary_bp.route("/api/report", methods=["POST"])
def generate_report():
    """
    Body: {
      "symptoms":     ["fever", "headache"],
      "predictions":  [...],       ← from /api/predict/symptoms
      "patient_name": "John Doe"   ← optional
    }
    Returns a PDF file download.
    """
    try:
        from backend.services.report_service import generate_report as _gen, REPORTLAB_AVAILABLE
        if not REPORTLAB_AVAILABLE:
            return error(
                "PDF generation requires reportlab. "
                "Install with: pip install reportlab", 501
            )
    except ImportError:
        return error("Report service unavailable.", 500)

    body         = request.get_json(silent=True) or {}
    symptoms     = body.get("symptoms", [])
    predictions  = body.get("predictions", [])
    patient_name = body.get("patient_name", "Patient")

    if not symptoms or not predictions:
        return error("'symptoms' and 'predictions' are required.", 400)

    try:
        pdf_bytes = _gen(symptoms, predictions, patient_name)
    except Exception as e:
        return error(f"PDF generation failed: {str(e)}", 500)

    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"medisense_report_{patient_name.replace(' ','_')}.pdf",
    )


# ── System health ──────────────────────────────────────────────────────────────
@secondary_bp.route("/api/health", methods=["GET"])
def health():
    svc = current_app.config["PREDICTION_SERVICE"]
    return success(data={
        "service":    "MediSense AI v2",
        "model_ready": svc.ready,
        "model_info":  svc.get_model_info(),
    })
