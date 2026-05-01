"""
MediSense AI v2 — ML Prediction Service
Loads the trained pipeline + metadata once at startup,
exposes clean inference methods used by route handlers.
"""
import os, re, json
import numpy as np
import pandas as pd
import joblib

class PredictionService:
    """Singleton-style service — instantiate once in app factory."""

    def __init__(self, ml_dir: str):
        self.ml_dir = ml_dir
        self._load_artefacts()

    # ── Artefact loading ───────────────────────────────────────────────────────
    def _load_artefacts(self):
        pipeline_path = os.path.join(self.ml_dir, "pipeline.pkl")
        le_path       = os.path.join(self.ml_dir, "label_encoder.pkl")
        sym_path      = os.path.join(self.ml_dir, "symptoms_list.pkl")
        meta_path     = os.path.join(self.ml_dir, "metadata.json")

        # Gracefully fail — routes will return 503 if service not loaded
        try:
            self.pipeline      = joblib.load(pipeline_path)
            self.label_encoder = joblib.load(le_path)
            self.symptoms_list = joblib.load(sym_path)
            with open(meta_path) as f:
                self.meta = json.load(f)
            self.severity_map = self.meta.get("severity_map", {})
            self.ready = True
            print(f"[PredictionService] Loaded — {len(self.symptoms_list)} symptoms, "
                  f"{len(self.label_encoder.classes_)} diseases")
        except FileNotFoundError as e:
            self.ready = False
            print(f"[PredictionService] ⚠️  Model artefacts not found: {e}")
            print("  Run: python3 ml/train_model.py")

    # ── Symptom utilities ──────────────────────────────────────────────────────
    @staticmethod
    def clean(s: str) -> str:
        return s.strip().lower().replace(" ", "_").replace("-", "_")

    def extract_symptoms_from_text(self, text: str) -> list[str]:
        """
        NLP: match free-text input against all known symptoms.
        Strategy:
          1. Exact substring match (symptom phrase in text)
          2. Token-set match (all underscore-tokens of symptom present as words)
        Returns deduplicated list of matched symptom keys.
        """
        text_lower = text.lower()
        text_clean = re.sub(r"[^a-z0-9 ]", " ", text_lower)
        words = set(text_clean.split())
        found = set()

        for sym in self.symptoms_list:
            # Strategy 1: phrase match
            if sym.replace("_", " ") in text_lower:
                found.add(sym)
                continue
            # Strategy 2: token-set match
            tokens = set(sym.split("_"))
            if len(tokens) <= 3 and tokens.issubset(words):
                found.add(sym)

        return list(found)

    def validate_symptoms(self, symptoms: list[str]) -> tuple[list[str], list[str]]:
        """
        Split input symptom list into:
          valid   — known to the model
          invalid — not recognised
        """
        valid, invalid = [], []
        for s in symptoms:
            cs = self.clean(s)
            (valid if cs in self.symptoms_list else invalid).append(cs)
        return valid, invalid

    # ── Feature vector ─────────────────────────────────────────────────────────
    def _build_vector(self, symptoms: list[str]) -> pd.DataFrame:
        vec = {s: 0 for s in self.symptoms_list}
        for s in symptoms:
            cs = self.clean(s)
            if cs in vec:
                vec[cs] = self.severity_map.get(cs, 1)
        return pd.DataFrame([[vec[s] for s in self.symptoms_list]],
                             columns=self.symptoms_list)

    # ── Core prediction ────────────────────────────────────────────────────────
    def predict(self, symptoms: list[str], top_n: int = 3) -> list[dict]:
        """
        Run inference and return top-N predictions with:
          disease, probability, description, precautions, specialization
        """
        if not self.ready:
            raise RuntimeError("Model not loaded. Run train_model.py first.")
        if not symptoms:
            raise ValueError("Empty symptom list.")

        df    = self._build_vector(symptoms)
        probs = self.pipeline.predict_proba(df)[0]
        topN  = probs.argsort()[-top_n:][::-1]

        results = []
        for idx in topN:
            disease = self.label_encoder.classes_[idx]
            results.append({
                "disease":        disease.strip(),
                "probability":    round(float(probs[idx]) * 100, 1),
                "description":    self.meta["descriptions"].get(disease, "No description available."),
                "precautions":    self.meta["precautions"].get(disease, []),
                "specialization": self.meta["specializations"].get(disease, "General Physician"),
            })
        return results

    # ── Model info ─────────────────────────────────────────────────────────────
    def get_model_info(self) -> dict:
        if not self.ready:
            return {"ready": False}
        m = self.meta.get("metrics", {})
        return {
            "ready":           True,
            "version":         self.meta.get("version", "2.0"),
            "total_diseases":  len(self.label_encoder.classes_),
            "total_symptoms":  len(self.symptoms_list),
            "metrics":         m,
        }
