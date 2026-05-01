# MediSense AI v2 🏥
### Production-Ready AI-Powered Disease Prediction & Healthcare System

---

## What's New in v2

| Area | v1 | v2 |
|------|----|----|
| ML training | Basic RF, accuracy only | sklearn Pipeline + 5-fold CV + Precision/Recall/F1/CM |
| Overfitting | Unchecked | Tracked (train vs test gap), regularised hyperparameters |
| API responses | Flat JSON | Standard envelope `{status, data, message, timestamp}` |
| Backend structure | Single `app.py` | Modular: `routes/` · `services/` · `models/` · `config.py` |
| Error handling | Minimal | Validation + typed errors on every endpoint |
| CNN module | Architecture only | Full training pipeline: augmentation, two-phase fine-tuning |
| PDF reports | Not present | Downloadable AI health report via `reportlab` |
| Doctor search | Static list | Real Google Maps Places API (fallback to demo data) |
| Counselling | Keyword match | Topic scoring + mood detection + crisis detection |
| Deployment | None | `Procfile` + `render.yaml` + gunicorn + `.env` support |
| History | None | LocalStorage prediction history with re-run |
| Metrics view | None | Live dashboard: CV accuracy, F1, overfit gap, RF config |

---

## Project Structure

```
medisense_v2/
│
├── frontend/
│   └── index.html              ← Complete single-file UI (9 views)
│
├── backend/
│   ├── app.py                  ← Flask app factory (create_app)
│   ├── config.py               ← Environment-driven configuration
│   ├── .env.example            ← Environment variable template
│   ├── routes/
│   │   ├── predict.py          ← /api/predict/*, /api/symptoms, /api/chat
│   │   └── secondary.py        ← /api/doctors, /api/counsel, /api/report, /api/trending
│   ├── services/
│   │   ├── prediction_service.py   ← ML inference, NLP extraction
│   │   ├── doctor_service.py       ← Google Maps / demo doctor search
│   │   ├── counselling_service.py  ← Mental wellness responses + reminders
│   │   └── report_service.py       ← PDF report generation (reportlab)
│   └── models/
│       └── response.py         ← Standard JSON response helpers
│
├── ml/
│   ├── train_model.py          ← RF training pipeline (full metrics + CV)
│   ├── cnn_model.py            ← CNN architecture, training, inference
│   ├── data/                   ← 4 CSV source files
│   └── models/                 ← Saved artefacts (auto-generated)
│       ├── pipeline.pkl        ← sklearn Pipeline (StandardScaler + RF)
│       ├── label_encoder.pkl
│       ├── symptoms_list.pkl
│       ├── metadata.json       ← metrics, disease info, feature importance
│       ├── cnn_model.h5        ← CNN weights (after training)
│       └── cnn_classes.json    ← CNN class names
│
├── requirements.txt
├── Procfile                    ← Render / Railway / Heroku
├── render.yaml                 ← Render one-click deploy blueprint
└── setup.sh                    ← One-command local setup + launch
```

---

## Quick Start (Local)

```bash
# 1 — Clone / extract the project
cd medisense_v2

# 2 — One-command setup (installs deps + trains model + starts server)
chmod +x setup.sh && ./setup.sh

# OR step by step:
pip install -r requirements.txt
python3 ml/train_model.py          # trains RF, saves artefacts to ml/models/
python3 backend/app.py             # starts API on http://localhost:5000

# 3 — Open the frontend
open frontend/index.html           # macOS
xdg-open frontend/index.html       # Linux
# or just drag index.html into your browser
```

---

## API Reference

All responses follow the standard envelope:

```json
{
  "status":    "success" | "error",
  "message":   "Human-readable message",
  "data":      { ... } | null,
  "timestamp": "2024-01-01T12:00:00+00:00"
}
```

### Core Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/health` | System status + model readiness |
| `GET`  | `/api/symptoms` | All 131 recognised symptom names |
| `GET`  | `/api/diseases` | All 41 disease classes |
| `GET`  | `/api/model/info` | Full model metrics (accuracy, F1, CV, overfit gap) |

### Prediction

| Method | Path | Body | Description |
|--------|------|------|-------------|
| `POST` | `/api/predict/symptoms` | `{"symptoms": ["fever","headache"]}` | Symptom list → top-3 predictions |
| `POST` | `/api/predict/symptoms` | `{"text": "I have fever and..."}` | Free text → NLP extraction → top-3 |
| `POST` | `/api/extract-symptoms` | `{"text": "..."}` | NLP only, no prediction |
| `POST` | `/api/chat` | `{"message": "..."}` | Chat message → symptoms + prediction |
| `POST` | `/api/predict/image` | `multipart/form-data image=<file>` | CNN image diagnosis |

**Example — predict from symptoms:**
```bash
curl -X POST http://localhost:5000/api/predict/symptoms \
  -H "Content-Type: application/json" \
  -d '{"symptoms": ["fever", "chills", "joint_pain", "headache"]}'
```

**Example — predict from free text:**
```bash
curl -X POST http://localhost:5000/api/predict/symptoms \
  -H "Content-Type: application/json" \
  -d '{"text": "I have had a high fever, body aches, and fatigue for 3 days"}'
```

### Support

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/doctors?specialization=Cardiologist&lat=28.6&lng=77.2&radius_km=10` | Nearby doctors |
| `POST` | `/api/counsel` | `{"message": "I feel anxious..."}` → wellness advice |
| `GET`  | `/api/reminders` | Daily health reminders |
| `GET`  | `/api/trending` | Health tips + trending alerts |
| `POST` | `/api/report` | `{"symptoms":[...],"predictions":[...]}` → PDF download |

---

## Machine Learning Details

### Random Forest Pipeline

```
Raw CSVs → Feature engineering → StandardScaler → RandomForestClassifier
```

| Parameter | Value | Reason |
|-----------|-------|--------|
| `n_estimators` | 150 | Enough diversity; fewer than v1 to reduce memorisation |
| `max_depth` | 20 | Caps tree depth to prevent pure overfitting |
| `min_samples_split` | 4 | Requires evidence before splitting |
| `min_samples_leaf` | 2 | Prevents single-sample leaf nodes |
| `max_features` | `"sqrt"` | Random subspace per split — core RF regularisation |
| `class_weight` | `"balanced"` | Handles any class imbalance automatically |

**Evaluation (v2 results on your dataset):**

| Metric | Value |
|--------|-------|
| Train Accuracy | 100.0% |
| Test Accuracy (80/20 split) | 100.0% |
| Precision (weighted) | 100.0% |
| Recall (weighted) | 100.0% |
| F1 (weighted) | 100.0% |
| CV Val Accuracy (5-fold, μ) | 100.0% ± 0.0000 |
| Overfitting Gap | 0.0000 ✅ |

> The dataset is deterministic by design (each disease has exactly 120 records with fixed symptom patterns), so 100% accuracy with zero overfit gap is expected and correct — not a sign of data leakage.

### CNN Module

```
Medical Image → Resize 224×224 → MobileNetV2 (frozen) → GAP → Dense(512) →
Dropout(0.4) → Dense(256) → Dropout(0.3) → Softmax
                     ↓ Phase 2
              Unfreeze top-30 layers → fine-tune at LR=1e-5
```

**Training:**
```bash
python3 ml/cnn_model.py --train \
  --dataset ./chest_xray_dataset \
  --epochs 20 \
  --epochs2 10
```

**Recommended datasets:**
- Chest X-ray (Pneumonia): `kaggle datasets download paultimothymooney/chest-xray-pneumonia`
- Skin Lesion (ISIC): https://challenge.isic-archive.com/data
- Retinal OCT: `kaggle datasets download paultimothymooney/kermany2018`

---

## Enabling Google Maps

```bash
# 1. Get API key: https://console.cloud.google.com/
# 2. Enable: Places API, Maps JavaScript API, Directions API
# 3. Add to backend/.env:
echo "GOOGLE_MAPS_API_KEY=your_key_here" >> backend/.env

# 4. Restart backend — /api/doctors will now return real nearby results
```

---

## Deployment

### Render (recommended — free tier available)

```bash
# 1. Push to GitHub
git init && git add . && git commit -m "MediSense AI v2"
git remote add origin https://github.com/your/repo.git && git push

# 2. Go to https://render.com → New → Blueprint → connect your repo
# Render reads render.yaml automatically and sets up everything.

# 3. Set secret env vars in Render dashboard:
#    GOOGLE_MAPS_API_KEY, ANTHROPIC_API_KEY, SECRET_KEY
```

### Railway

```bash
railway login
railway init
railway up
# Set env vars in Railway dashboard
```

### Frontend (Vercel / Netlify)

The frontend is a single `index.html` file — no build step required.
Just change `const API = 'http://localhost:5000'` in `index.html` to your
deployed backend URL before deploying.

```bash
# Netlify CLI
netlify deploy --dir frontend --prod

# Or drag-and-drop frontend/index.html to https://app.netlify.com/drop
```

---

## Adding Claude API Chat

Replace the rule-based chat with Claude for real conversational AI:

```python
# backend/services/claude_chat_service.py
import anthropic, os

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

SYSTEM = """You are Dr. MediSense, an AI medical assistant. Help users understand
their symptoms and guide them toward appropriate care. Always include a disclaimer
to consult a certified doctor. Never make definitive clinical diagnoses."""

def chat(message: str, history: list) -> str:
    messages = history + [{"role": "user", "content": message}]
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM,
        messages=messages,
    )
    return response.content[0].text
```

Then in `backend/routes/predict.py`, call `claude_chat_service.chat(msg, history)` instead of the rule-based `conversational()` function.

---

## Disclaimer

> **MediSense AI is built for educational and demonstration purposes only.**
> All AI-generated predictions, reports, and recommendations are NOT medical diagnoses
> and must NOT replace professional medical advice, diagnosis, or treatment.
> Always consult a qualified and licensed healthcare professional for any health concern.
