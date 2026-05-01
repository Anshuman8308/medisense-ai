"""
MediSense AI v2 — Counselling & Wellness Service
Improved topic detection, multi-turn conversation context,
reminder system, and mental health support responses.
"""
import random
from datetime import datetime

# ── Wellness knowledge base ────────────────────────────────────────────────────
TOPICS = {
    "stress": {
        "keywords": ["stress", "overwhelm", "pressure", "burnout", "overworked",
                     "exhausted", "tense", "strain"],
        "tips": [
            "Try box breathing: inhale 4s → hold 4s → exhale 4s → hold 4s. Repeat 4–6 cycles.",
            "Write down 3 things you're grateful for right now. Gratitude rewires cortisol pathways.",
            "Take a 10-minute walk outside — daylight and movement together cut stress hormones significantly.",
            "Set a 'worry window': 20 minutes daily where you're allowed to worry. Outside it, redirect thoughts.",
            "Limit news and social media to 30 minutes/day. Passive scrolling amplifies stress.",
            "Progressive muscle relaxation: tense each muscle group for 5s, then release from toes upward.",
        ],
        "empathy": [
            "That sounds really demanding. It's okay to acknowledge when things feel like too much.",
            "Chronic stress is serious — recognising it is the first step.",
            "You're not alone in feeling overwhelmed. Let's work through some strategies together.",
        ],
    },
    "anxiety": {
        "keywords": ["anxiety", "anxious", "worry", "panic", "nervous", "fear",
                     "dread", "uneasy", "apprehensive"],
        "tips": [
            "Use the 5-4-3-2-1 grounding technique: name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste.",
            "Diaphragmatic breathing: place one hand on your belly and breathe so it rises. Exhale slowly through pursed lips.",
            "Anxiety peaks and then naturally subsides — ride it like a wave rather than fighting it.",
            "Cold water on your face triggers the dive reflex and slows heart rate within seconds.",
            "Name what you're anxious about specifically. Vague anxiety is harder to manage than a named concern.",
            "Avoid caffeine and alcohol — both significantly worsen anxiety symptoms.",
        ],
        "empathy": [
            "Anxiety can feel very overwhelming. What you're experiencing is real and valid.",
            "I hear you. Anxiety is one of the most common experiences people face.",
            "It takes courage to talk about anxiety. Let me share some evidence-based strategies.",
        ],
    },
    "sleep": {
        "keywords": ["sleep", "insomnia", "can't sleep", "tired", "fatigue",
                     "exhausted", "rest", "awake", "drowsy", "waking up"],
        "tips": [
            "Avoid screens 60 minutes before bed — blue light suppresses melatonin by up to 50%.",
            "Keep your bedroom cool (18–20°C / 64–68°F) — core body temperature must drop to initiate sleep.",
            "Set a consistent sleep-wake time, even on weekends. Regularity is the #1 sleep improvement factor.",
            "Avoid caffeine after 2 PM — it has a 5–7 hour half-life and disrupts sleep architecture.",
            "If you can't sleep after 20 minutes, get up and do something calm in dim light. Don't lie awake frustrated.",
            "The 4-7-8 technique: inhale 4s, hold 7s, exhale 8s. Activates parasympathetic nervous system.",
        ],
        "empathy": [
            "Poor sleep affects everything — mood, immunity, cognition. You deserve restful nights.",
            "Sleep difficulties are frustrating and exhausting. Let's look at what might help.",
            "Many people struggle with sleep. Some targeted changes can make a significant difference.",
        ],
    },
    "diet": {
        "keywords": ["diet", "eating", "food", "nutrition", "weight", "meal",
                     "hungry", "appetite", "calories", "unhealthy"],
        "tips": [
            "Eat protein with every meal — it stabilises blood sugar and reduces cravings throughout the day.",
            "Chew slowly: it takes 20 minutes for satiety signals to reach the brain. Rushed meals cause overeating.",
            "Hydrate before meals — thirst is frequently mistaken for hunger.",
            "Aim for a colourful plate: different pigments represent different antioxidants and phytonutrients.",
            "Limit ultra-processed foods. If a product has more than 5 ingredients you can't recognise, reconsider.",
            "Plan meals weekly. Decision fatigue leads to poor food choices when you're hungry.",
        ],
        "empathy": [
            "Nutrition can feel complicated with so much conflicting advice. Let's simplify it.",
            "Your relationship with food is important. Small, consistent changes beat extreme diets.",
            "It's okay to have imperfect eating days. Progress, not perfection.",
        ],
    },
    "exercise": {
        "keywords": ["exercise", "workout", "gym", "activity", "sedentary",
                     "fit", "move", "walk", "running", "inactive"],
        "tips": [
            "Even 10 minutes of movement counts — perfect consistency with short sessions beats occasional marathons.",
            "Walking 10 minutes after meals reduces post-meal blood sugar spikes by up to 12%.",
            "Strength training 2× per week preserves muscle mass and metabolic rate as you age.",
            "Find an activity you genuinely enjoy — sustainability matters more than intensity.",
            "Exercise is one of the most effective interventions for both depression and anxiety.",
            "Start with the '2-minute rule': commit to just 2 minutes. Momentum usually takes over.",
        ],
        "empathy": [
            "Getting started is often the hardest part. Small steps genuinely lead to big changes.",
            "It's okay if exercise feels difficult right now. Let's find something that suits your life.",
            "Movement is medicine — and it doesn't have to look like a gym session.",
        ],
    },
    "depression": {
        "keywords": ["depressed", "depression", "hopeless", "empty", "numb",
                     "worthless", "sad", "crying", "no motivation", "pointless"],
        "tips": [
            "Reach out to someone you trust — isolation worsens depression. Even a short message counts.",
            "Maintain basic structure: get up, shower, eat, go outside briefly. Routine anchors mood.",
            "Physical activity — even a 15-minute walk — has clinically proven antidepressant effects.",
            "Depression lies to you. Thoughts like 'nothing will help' are symptoms, not facts.",
            "Professional support is important for depression. Please consider speaking to a doctor or therapist.",
        ],
        "empathy": [
            "What you're feeling sounds really difficult. Depression is a real illness, not a weakness.",
            "Thank you for sharing this. Please know you're not alone, and things can get better.",
            "I'm concerned about what you've described. Please consider speaking to a mental health professional.",
        ],
        "urgent": True,
    },
    "general": {
        "keywords": [],
        "tips": [
            "Your mental and physical health are deeply connected — care for both intentionally.",
            "Small, consistent habits outperform intense, unsustainable ones every time.",
            "Rest is not laziness — recovery is where growth, healing, and resilience are built.",
            "Seeking support is a sign of strength, not weakness.",
            "Hydration, sleep, movement, and social connection are the four pillars of baseline wellbeing.",
        ],
        "empathy": [
            "I'm here to support your wellbeing journey.",
            "How are you doing today? I can help with stress, sleep, diet, exercise, or mental wellness.",
        ],
    },
}

# ── Daily reminders ────────────────────────────────────────────────────────────
REMINDERS = [
    {"id": "water",      "text": "Time to drink a glass of water 💧",         "icon": "water_drop",    "interval_hours": 2},
    {"id": "walk",       "text": "Stand up and take a 5-minute walk 🚶",       "icon": "directions_walk","interval_hours": 3},
    {"id": "breathe",    "text": "Take 3 deep breaths right now 🌬️",           "icon": "air",           "interval_hours": 1},
    {"id": "posture",    "text": "Check your posture and roll your shoulders", "icon": "accessibility", "interval_hours": 2},
    {"id": "eyes",       "text": "Look away from the screen for 20 seconds 👀","icon": "visibility",    "interval_hours": 1},
    {"id": "medication", "text": "Time for your medication (if prescribed) 💊", "icon": "medication",    "interval_hours": 12},
    {"id": "meal",       "text": "Have a balanced meal or healthy snack 🥗",   "icon": "restaurant",    "interval_hours": 4},
    {"id": "sleep_prep", "text": "Start winding down for sleep 😴",            "icon": "bedtime",       "interval_hours": 24},
]


class CounsellingService:

    def detect_topic(self, text: str) -> str:
        text_lower = text.lower()
        # Score each topic by keyword hits (more hits = better match)
        scores = {}
        for topic, data in TOPICS.items():
            if topic == "general":
                continue
            score = sum(1 for kw in data["keywords"] if kw in text_lower)
            if score > 0:
                scores[topic] = score
        if not scores:
            return "general"
        return max(scores, key=scores.get)

    def respond(self, message: str, mood: str = None) -> dict:
        topic = self.detect_topic(message)
        data  = TOPICS[topic]

        empathy = random.choice(data["empathy"])
        tips    = random.sample(data["tips"], min(3, len(data["tips"])))
        urgent  = data.get("urgent", False)

        response = {
            "topic":      topic,
            "empathy":    empathy,
            "tips":       tips,
            "urgent":     urgent,
            "reminders":  self._get_active_reminders(),
            "disclaimer": "If you are experiencing a mental health crisis, please contact a professional immediately.",
        }

        if urgent:
            response["crisis_note"] = (
                "Based on what you've shared, I'd strongly encourage you to speak "
                "with a mental health professional or call a helpline. "
                "In India: iCall — 9152987821 | Vandrevala Foundation — 1860-2662-345"
            )

        return response

    def _get_active_reminders(self) -> list[dict]:
        """Return a contextual subset of reminders based on time of day."""
        hour = datetime.now().hour
        active = []
        for r in REMINDERS:
            # Sleep prep only in evening
            if r["id"] == "sleep_prep" and not (20 <= hour <= 23):
                continue
            # Medication reminder at standard times
            if r["id"] == "medication" and hour not in (8, 13, 20):
                continue
            active.append(r)
        return active[:4]   # return at most 4

    def get_all_reminders(self) -> list[dict]:
        return REMINDERS
