"""
MediSense AI v2 — Doctor Recommendation Service
================================================
Two modes:
  1. LIVE  — calls Google Maps Places API (requires GOOGLE_MAPS_API_KEY)
  2. DEMO  — returns curated sample data (default when no key is set)

Google Maps Places API docs:
  https://developers.google.com/maps/documentation/places/web-service/nearby-search
"""
import os, math, requests


# ── Specialisation → type mapping for Places API ─────────────────────────────
SPEC_TO_QUERY = {
    "Cardiologist":                  "cardiologist",
    "Endocrinologist":               "endocrinologist",
    "Pulmonologist":                 "pulmonologist",
    "General Physician":             "general physician",
    "Allergist":                     "allergist",
    "Neurologist":                   "neurologist",
    "Orthopedist":                   "orthopedic doctor",
    "Rheumatologist":                "rheumatologist",
    "Infectious Disease Specialist": "infectious disease doctor",
    "Gastroenterologist":            "gastroenterologist",
    "Proctologist":                  "proctologist",
    "Urologist":                     "urologist",
    "Dermatologist":                 "dermatologist",
    "Vascular Surgeon":              "vascular surgeon",
}

SAMPLE_DOCTORS = [
    {"name":"Dr. Priya Sharma",   "specialization":"General Physician",      "rating":4.8,"distance_km":1.2,"phone":"+91 98765 43210","availability":"Mon–Sat  9 AM–6 PM","lat":28.6139,"lng":77.2090,"address":"12 Civil Lines, New Delhi"},
    {"name":"Dr. Rahul Mehta",    "specialization":"Cardiologist",           "rating":4.9,"distance_km":2.1,"phone":"+91 98765 11223","availability":"Tue–Sun 10 AM–5 PM","lat":28.6200,"lng":77.2150,"address":"Sector 5, Rohini, New Delhi"},
    {"name":"Dr. Anita Rao",      "specialization":"Dermatologist",          "rating":4.7,"distance_km":3.5,"phone":"+91 99887 65432","availability":"Mon–Fri  8 AM–4 PM","lat":28.6080,"lng":77.2200,"address":"Green Park Extension, New Delhi"},
    {"name":"Dr. Suresh Kumar",   "specialization":"Neurologist",            "rating":4.6,"distance_km":4.2,"phone":"+91 99776 55443","availability":"Mon–Sat 10 AM–7 PM","lat":28.6300,"lng":77.2050,"address":"Pitampura, New Delhi"},
    {"name":"Dr. Meera Pillai",   "specialization":"Gastroenterologist",     "rating":4.8,"distance_km":5.0,"phone":"+91 98654 32109","availability":"Tue–Sat  9 AM–5 PM","lat":28.6050,"lng":77.1980,"address":"Lajpat Nagar, New Delhi"},
    {"name":"Dr. Vikram Singh",   "specialization":"Endocrinologist",        "rating":4.5,"distance_km":6.3,"phone":"+91 97845 61234","availability":"Mon–Fri 11 AM–6 PM","lat":28.6400,"lng":77.2100,"address":"Vasant Kunj, New Delhi"},
    {"name":"Dr. Kavya Nair",     "specialization":"General Physician",      "rating":4.9,"distance_km":1.8,"phone":"+91 96543 21098","availability":"Mon–Sun  8 AM–8 PM","lat":28.6160,"lng":77.2000,"address":"Karol Bagh, New Delhi"},
    {"name":"Dr. Arjun Patel",    "specialization":"Cardiologist",           "rating":4.7,"distance_km":7.1,"phone":"+91 98321 54321","availability":"Mon–Sat  9 AM–5 PM","lat":28.6250,"lng":77.2300,"address":"Dwarka Sector 10, New Delhi"},
    {"name":"Dr. Sneha Iyer",     "specialization":"Pulmonologist",          "rating":4.6,"distance_km":3.2,"phone":"+91 97654 32198","availability":"Mon–Fri  9 AM–5 PM","lat":28.6100,"lng":77.2180,"address":"Saket, New Delhi"},
    {"name":"Dr. Ravi Krishnan",  "specialization":"Infectious Disease Specialist","rating":4.8,"distance_km":4.7,"phone":"+91 98123 45678","availability":"Mon–Sat 10 AM–6 PM","lat":28.6220,"lng":77.1950,"address":"Mayur Vihar Phase 1, Delhi"},
    {"name":"Dr. Pooja Verma",    "specialization":"Rheumatologist",         "rating":4.5,"distance_km":5.5,"phone":"+91 96789 01234","availability":"Tue–Sat  9 AM–4 PM","lat":28.6350,"lng":77.2250,"address":"Janakpuri, New Delhi"},
    {"name":"Dr. Amit Gupta",     "specialization":"Orthopedist",            "rating":4.7,"distance_km":2.8,"phone":"+91 99012 34567","availability":"Mon–Sat  8 AM–6 PM","lat":28.6180,"lng":77.2120,"address":"Rajouri Garden, New Delhi"},
    {"name":"Dr. Nisha Kapoor",   "specialization":"Urologist",              "rating":4.6,"distance_km":6.0,"phone":"+91 97890 12345","availability":"Mon–Fri 10 AM–5 PM","lat":28.6280,"lng":77.2070,"address":"Nehru Place, New Delhi"},
    {"name":"Dr. Sanjay Bose",    "specialization":"Allergist",              "rating":4.4,"distance_km":3.9,"phone":"+91 96012 34567","availability":"Tue–Sun  9 AM–5 PM","lat":28.6130,"lng":77.2160,"address":"Laxmi Nagar, Delhi"},
    {"name":"Dr. Deepa Menon",    "specialization":"Gastroenterologist",     "rating":4.8,"distance_km":2.3,"phone":"+91 97123 45000","availability":"Mon–Sat  9 AM–5 PM","lat":28.6090,"lng":77.2010,"address":"Patel Nagar, New Delhi"},
]


def haversine_km(lat1, lng1, lat2, lng2) -> float:
    """Calculate great-circle distance in km between two lat/lng points."""
    R = 6371.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lng2 - lng1)
    a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def maps_nav_url(dest_lat, dest_lng) -> str:
    return f"https://www.google.com/maps/dir/?api=1&destination={dest_lat},{dest_lng}"


class DoctorService:

    def __init__(self, api_key: str = ""):
        self.api_key  = api_key
        self.use_live = bool(api_key)
        if self.use_live:
            print("[DoctorService] Google Maps API key detected — live mode active.")
        else:
            print("[DoctorService] No API key — using demo doctor data.")

    def find_doctors(
        self,
        specialization: str = "",
        user_lat: float = None,
        user_lng: float = None,
        radius_km: float = 10.0,
    ) -> list[dict]:
        if self.use_live and user_lat and user_lng:
            return self._live_search(specialization, user_lat, user_lng, radius_km)
        return self._demo_search(specialization, user_lat, user_lng, radius_km)

    # ── Live Google Maps Places search ─────────────────────────────────────────
    def _live_search(self, specialization, lat, lng, radius_km) -> list[dict]:
        query  = SPEC_TO_QUERY.get(specialization, "doctor")
        radius = int(min(radius_km, 50) * 1000)   # metres, max 50 km

        url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        params = {
            "query":    f"{query} near me",
            "location": f"{lat},{lng}",
            "radius":   radius,
            "type":     "doctor",
            "key":      self.api_key,
        }

        try:
            resp  = requests.get(url, params=params, timeout=8)
            data  = resp.json()
            places = data.get("results", [])
        except Exception as e:
            print(f"[DoctorService] Places API error: {e} — falling back to demo")
            return self._demo_search(specialization, lat, lng, radius_km)

        doctors = []
        for p in places[:10]:
            plat = p["geometry"]["location"]["lat"]
            plng = p["geometry"]["location"]["lng"]
            dist = haversine_km(lat, lng, plat, plng)

            # Fetch phone via Place Details (extra call)
            phone = self._get_phone(p.get("place_id", ""))

            doctors.append({
                "name":           p.get("name", "Unknown"),
                "specialization": specialization or "General Physician",
                "rating":         p.get("rating", 0),
                "distance_km":    round(dist, 1),
                "phone":          phone,
                "availability":   "Check with clinic",
                "lat":            plat,
                "lng":            plng,
                "address":        p.get("formatted_address", ""),
                "maps_url":       maps_nav_url(plat, plng),
                "source":         "google_maps",
            })

        doctors.sort(key=lambda d: d["distance_km"])
        return doctors

    def _get_phone(self, place_id: str) -> str:
        if not place_id or not self.api_key:
            return ""
        url = "https://maps.googleapis.com/maps/api/place/details/json"
        params = {"place_id": place_id, "fields": "formatted_phone_number", "key": self.api_key}
        try:
            data = requests.get(url, params=params, timeout=5).json()
            return data.get("result", {}).get("formatted_phone_number", "")
        except Exception:
            return ""

    # ── Demo data search ───────────────────────────────────────────────────────
    def _demo_search(self, specialization, user_lat, user_lng, radius_km) -> list[dict]:
        docs = list(SAMPLE_DOCTORS)

        if specialization:
            docs = [d for d in docs if d["specialization"].lower() == specialization.lower()]

        # Recalculate distance if real coordinates provided
        for d in docs:
            if user_lat and user_lng:
                d = dict(d)
                d["distance_km"] = round(haversine_km(user_lat, user_lng, d["lat"], d["lng"]), 1)
            d["maps_url"] = maps_nav_url(d["lat"], d["lng"])
            d["source"]   = "demo"

        docs = [d for d in docs if d["distance_km"] <= radius_km]
        docs.sort(key=lambda d: d["distance_km"])
        return docs
