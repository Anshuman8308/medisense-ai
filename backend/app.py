"""
MediSense AI v2 — Flask Application Factory
============================================
Production-ready entry point with:
  - App factory pattern (create_app)
  - Service injection via app.config
  - Modular blueprints
  - CORS, error handlers, request logging
  - Environment-driven configuration

Run (development):
  python3 app.py

Run (production with gunicorn):
  gunicorn "app:create_app()" --bind 0.0.0.0:5000 --workers 2
"""

import os, sys, time, logging
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, jsonify, g, request
from flask_cors import CORS

# ── Load .env ─────────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# Ensure project root is on sys.path so 'backend.*' imports resolve
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.config import get_config
from backend.routes.predict   import predict_bp
from backend.routes.secondary import secondary_bp


def create_app(config_cls=None):
    app = Flask(__name__)
    cfg = config_cls or get_config()
    app.config.from_object(cfg)

    # ── Logging ───────────────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.DEBUG if cfg.DEBUG else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    log = logging.getLogger("medisense")

    # ── CORS ──────────────────────────────────────────────────────────────────
    CORS(app, origins=cfg.CORS_ORIGINS, supports_credentials=True)

    # ── Inject services ───────────────────────────────────────────────────────
    from backend.services.prediction_service  import PredictionService
    from backend.services.doctor_service      import DoctorService
    from backend.services.counselling_service import CounsellingService

    app.config["PREDICTION_SERVICE"]  = PredictionService(str(cfg.ML_DIR))
    app.config["DOCTOR_SERVICE"]      = DoctorService(cfg.GOOGLE_MAPS_API_KEY)
    app.config["COUNSELLING_SERVICE"] = CounsellingService()

    # ── Register blueprints ───────────────────────────────────────────────────
    app.register_blueprint(predict_bp)
    app.register_blueprint(secondary_bp)

    # ── Request timing middleware ─────────────────────────────────────────────
    @app.before_request
    def start_timer():
        g.start = time.time()

    @app.after_request
    def log_request(response):
        if hasattr(g, "start"):
            elapsed = (time.time() - g.start) * 1000
            log.debug(f"{request.method} {request.path} → {response.status_code} ({elapsed:.1f}ms)")
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"]        = "DENY"
        response.headers["X-Powered-By"]           = "MediSense AI v2"
        return response

    # ── Global error handlers ─────────────────────────────────────────────────
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"status":"error","message":f"Endpoint not found: {request.path}",
                        "data":None}), 404

    @app.errorhandler(405)
    def method_not_allowed(e):
        return jsonify({"status":"error","message":"Method not allowed","data":None}), 405

    @app.errorhandler(500)
    def internal_error(e):
        log.exception("Unhandled exception")
        return jsonify({"status":"error","message":"Internal server error","data":None}), 500

    # ── Root route ────────────────────────────────────────────────────────────
    @app.route("/")
    def root():
        return jsonify({
            "service": "MediSense AI v2",
            "status":  "running",
            "endpoints": [
                "GET  /api/health",
                "GET  /api/symptoms",
                "GET  /api/diseases",
                "GET  /api/model/info",
                "POST /api/predict/symptoms",
                "POST /api/extract-symptoms",
                "POST /api/chat",
                "GET  /api/doctors",
                "POST /api/counsel",
                "GET  /api/reminders",
                "GET  /api/trending",
                "POST /api/report",
            ]
        })

    log.info(f"MediSense AI v2 app created — env={os.getenv('FLASK_ENV','development')}")
    return app


if __name__ == "__main__":
    app = create_app()
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "true").lower() == "true"
    print(f"\n🚀 MediSense AI v2 running on http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
