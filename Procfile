# ── Procfile (Heroku / Railway / Render) ──────────────────────────────────────
# Railway / Heroku reads this automatically.
# Command: gunicorn with the Flask app factory pattern.
web: gunicorn "backend.app:create_app()" --bind 0.0.0.0:$PORT --workers 2 --timeout 120
