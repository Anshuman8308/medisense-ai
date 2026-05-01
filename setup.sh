#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# MediSense AI v2 — Setup & Launch
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh            full setup + start backend
#   ./setup.sh install    install Python deps only
#   ./setup.sh train      train Random Forest model only
#   ./setup.sh backend    start Flask API (assumes model already trained)
#   ./setup.sh prod       start with gunicorn (production)
# ─────────────────────────────────────────────────────────────────────────────
set -e

GREEN='\033[0;32m'; CYAN='\033[0;36m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

banner() {
  echo -e "${CYAN}"
  echo "  ╔═══════════════════════════════════════════════╗"
  echo "  ║       MediSense AI v2 — Setup & Launch        ║"
  echo "  ║   AI-Powered Disease Prediction & Healthcare  ║"
  echo "  ╚═══════════════════════════════════════════════╝"
  echo -e "${NC}"
}

step()  { echo -e "\n${YELLOW}▶ $1${NC}"; }
ok()    { echo -e "${GREEN}  ✅ $1${NC}"; }
fail()  { echo -e "${RED}  ❌ $1${NC}"; exit 1; }
info()  { echo -e "  ${CYAN}ℹ  $1${NC}"; }

MODE=${1:-all}
banner

# ── Python check ──────────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
  fail "Python 3.9+ is required. Install from https://python.org"
fi
PY=$(command -v python3)
PY_VER=$($PY -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
ok "Python $PY_VER found at $PY"

# ── .env setup ────────────────────────────────────────────────────────────────
if [[ ! -f backend/.env && -f backend/.env.example ]]; then
  cp backend/.env.example backend/.env
  info "Created backend/.env from .env.example — edit it to add your API keys."
fi

# ── Install dependencies ──────────────────────────────────────────────────────
if [[ "$MODE" == "all" || "$MODE" == "install" ]]; then
  step "Installing Python dependencies..."
  $PY -m pip install -r requirements.txt -q --disable-pip-version-check
  ok "Dependencies installed"
fi

# ── Train ML model ────────────────────────────────────────────────────────────
if [[ "$MODE" == "all" || "$MODE" == "train" ]]; then
  step "Training Random Forest pipeline..."
  if [[ ! -f ml/data/dataset.csv ]]; then
    fail "ml/data/dataset.csv not found. Copy the 4 CSV files into ml/data/"
  fi
  $PY ml/train_model.py
  ok "Model artefacts saved to ml/models/"
fi

# ── Start backend ─────────────────────────────────────────────────────────────
if [[ "$MODE" == "all" || "$MODE" == "backend" ]]; then
  step "Starting Flask development server..."
  if [[ ! -f ml/models/pipeline.pkl ]]; then
    info "Model not found — running training first..."
    $PY ml/train_model.py
  fi
  echo ""
  echo -e "  ${GREEN}🚀 Backend : http://localhost:5000${NC}"
  echo -e "  ${CYAN}   Frontend: open frontend/index.html in your browser${NC}"
  echo ""
  FLASK_ENV=development $PY backend/app.py
fi

# ── Production (gunicorn) ─────────────────────────────────────────────────────
if [[ "$MODE" == "prod" ]]; then
  step "Starting gunicorn production server..."
  if ! command -v gunicorn &>/dev/null; then
    $PY -m pip install gunicorn -q
  fi
  echo -e "\n  ${GREEN}🚀 Production server starting on port ${PORT:-5000}${NC}\n"
  gunicorn "backend.app:create_app()" \
    --bind "0.0.0.0:${PORT:-5000}" \
    --workers "${WEB_CONCURRENCY:-2}" \
    --timeout 120 \
    --log-level info \
    --access-logfile -
fi
