from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import pickle, os, numpy as np
from datetime import datetime
from functools import wraps

# ── Paths — works locally AND on Render ───────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
TMPL_DIR  = os.path.join(BASE_DIR, "templates")
DATA_DIR  = os.path.join(BASE_DIR, "data")

app = Flask(__name__, template_folder=TMPL_DIR)
app.secret_key = os.environ.get("SECRET_KEY", "pilgrimage_secret_key_2024")

# ── Fix Render's postgres:// → postgresql:// ──────────────────────────────────
database_url = os.environ.get(
    "DATABASE_URL",
    "sqlite:///" + os.path.join(BASE_DIR, "pilgrimage.db")
)
if database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)

app.config["SQLALCHEMY_DATABASE_URI"] = database_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# ── Load ML model ──────────────────────────────────────────────────────────────
with open(os.path.join(DATA_DIR, "model.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(DATA_DIR, "model_meta.pkl"), "rb") as f:
    meta = pickle.load(f)

# ── Database Models ────────────────────────────────────────────────────────────
class User(db.Model):
    id            = db.Column(db.Integer, primary_key=True)
    name          = db.Column(db.String(100), nullable=False)
    email         = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)
    searches      = db.relationship("SearchHistory", backref="user", lazy=True)

class SearchHistory(db.Model):
    id              = db.Column(db.Integer, primary_key=True)
    user_id         = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    location        = db.Column(db.String(200))
    day             = db.Column(db.String(20))
    festival        = db.Column(db.String(50))
    weather         = db.Column(db.String(30))
    predicted_crowd = db.Column(db.Integer)
    crowd_level     = db.Column(db.String(20))
    wait_time       = db.Column(db.Integer)
    created_at      = db.Column(db.DateTime, default=datetime.utcnow)

# ── Auth decorator ─────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

# ── ML helpers ────────────────────────────────────────────────────────────────
DAY_ORDER      = meta["day_order"]
WEATHER_ORDER  = meta["weather_order"]
FESTIVAL_ORDER = meta["festival_order"]

LOCATION_MULTIPLIERS = {
    "kashi vishwanath": 1.20, "tirupati": 1.35, "vaishno devi": 1.15,
    "golden temple":    1.10, "shirdi":   1.18, "char dham":    1.25,
    "ajmer sharif":     1.12, "kedarnath":0.95, "badrinath":    0.90,
    "puri jagannath":   1.10, "mathura":  1.08, "haridwar":     1.12,
    "mahakaleshwar":    1.10, "somnath":  1.05, "rameshwaram":  1.08,
    "dwarka":           1.05,
}

def predict_crowd(day, festival, weather, month, location=""):
    day_num      = DAY_ORDER.index(day) if day in DAY_ORDER else 0
    is_weekend   = 1 if day in ["Saturday", "Sunday"] else 0
    festival_num = FESTIVAL_ORDER.index(festival) if festival in FESTIVAL_ORDER else 0
    weather_num  = WEATHER_ORDER.index(weather) if weather in WEATHER_ORDER else 5
    X     = np.array([[day_num, month, is_weekend, festival_num, weather_num]])
    crowd = int(model.predict(X)[0])
    loc   = location.lower().strip()
    mult  = next((v for k, v in LOCATION_MULTIPLIERS.items() if k in loc), 1.0)
    crowd = max(3000, int(crowd * mult))
    wait  = max(5, round(crowd / (420 * 6)))
    if festival == "major":
        wait = int(wait * 1.4)
    level = "High" if crowd > 50000 else "Medium" if crowd > 25000 else "Low"
    return {"crowd": crowd, "level": level, "wait_time": wait, "entry_rate": 420}

CHATBOT_KB = {
    "crowd":    "Crowd levels depend on day (+50% weekends), festival type (major = 3.5x), and weather. High crowd means 50,000+ pilgrims expected.",
    "wait":     "Wait time formula: Wait = Predicted Crowd ÷ (Entry Rate × Gates) = Crowd ÷ 2,520. Currently ~420 pilgrims/hr per gate across 6 gates.",
    "festival": "Major festivals (Maha Shivaratri, Ram Navami, Diwali) see 3–4x normal footfall. Minor festivals see ~2x. Plan 2–3 days ahead.",
    "weather":  "Rain and storms reduce footfall by 45–70%. Pleasant winter (Oct–Feb) has highest crowds. Extreme heat reduces pilgrims by ~25%.",
    "best time":"Best time: Early morning 4–7 AM — lowest wait, most peaceful experience. Avoid festival evenings and weekend afternoons.",
    "gate":     "If Gate 1 (Main) shows HIGH, use Gate 3 (East) or Gate 5 (VIP/Divyang) — typically 60–70% lower wait time.",
    "location": "Our model adjusts predictions per site: Tirupati (+35%), Kashi Vishwanath (+20%), Char Dham (+25%), Shirdi (+18%), Vaishno Devi (+15%).",
    "model":    "We use a Random Forest model trained on 731 days of data. It achieves R² = 0.87, meaning 87% of crowd variation is correctly predicted.",
    "default":  "I can help with: crowd predictions, wait times, best visit timing, festival schedules, gate recommendations, and location-specific advice.",
}

def chatbot_reply(message):
    msg = message.lower()
    if any(w in msg for w in ["hi", "hello", "hey", "namaste"]):
        return "Namaste! 🙏 I'm your Pilgrimage Planning Assistant. Ask me about crowds, wait times, gates, festivals, or the best time to visit!"
    if any(w in msg for w in ["thank", "thanks", "shukriya", "dhanyawad"]):
        return "You're most welcome! Have a safe and blessed pilgrimage. Jai Mata Di! 🙏"
    for key, reply in CHATBOT_KB.items():
        if key in msg:
            return reply
    return CHATBOT_KB["default"]

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return redirect(url_for("dashboard") if "user_id" in session else url_for("login"))

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        data     = request.get_json()
        name     = data.get("name", "").strip()
        email    = data.get("email", "").strip().lower()
        password = data.get("password", "")
        if not name or not email or not password:
            return jsonify({"success": False, "error": "All fields are required"})
        if len(password) < 6:
            return jsonify({"success": False, "error": "Password must be at least 6 characters"})
        if User.query.filter_by(email=email).first():
            return jsonify({"success": False, "error": "Email already registered. Please login."})
        user = User(name=name, email=email,
                    password_hash=generate_password_hash(password))
        db.session.add(user)
        db.session.commit()
        session["user_id"]   = user.id
        session["user_name"] = user.name
        return jsonify({"success": True})
    return render_template("auth.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        data     = request.get_json()
        email    = data.get("email", "").strip().lower()
        password = data.get("password", "")
        user     = User.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password_hash, password):
            return jsonify({"success": False, "error": "Invalid email or password"})
        session["user_id"]   = user.id
        session["user_name"] = user.name
        return jsonify({"success": True})
    return render_template("auth.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/dashboard")
@login_required
def dashboard():
    user    = User.query.get(session["user_id"])
    history = SearchHistory.query.filter_by(user_id=user.id)\
                .order_by(SearchHistory.created_at.desc()).limit(5).all()
    return render_template("dashboard.html", user=user, history=history)

@app.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    data     = request.get_json()
    day      = data.get("day", "Monday")
    festival = data.get("festival", "none")
    weather  = data.get("weather", "Pleasant")
    month    = int(data.get("month", datetime.now().month))
    location = data.get("location", "")
    result   = predict_crowd(day, festival, weather, month, location)
    hist = SearchHistory(
        user_id=session["user_id"], location=location,
        day=day, festival=festival, weather=weather,
        predicted_crowd=result["crowd"], crowd_level=result["level"],
        wait_time=result["wait_time"],
    )
    db.session.add(hist)
    db.session.commit()
    return jsonify(result)

@app.route("/api/chat", methods=["POST"])
@login_required
def api_chat():
    data = request.get_json()
    return jsonify({"reply": chatbot_reply(data.get("message", ""))})

@app.route("/api/history")
@login_required
def api_history():
    records = SearchHistory.query\
        .filter_by(user_id=session["user_id"])\
        .order_by(SearchHistory.created_at.desc()).limit(10).all()
    return jsonify([{
        "location": r.location, "day": r.day,
        "festival": r.festival, "weather": r.weather,
        "crowd": r.predicted_crowd, "level": r.crowd_level,
        "wait": r.wait_time,
        "date": r.created_at.strftime("%d %b %Y"),
    } for r in records])

@app.route("/health")
def health():
    return jsonify({"status": "ok", "app": "PilgrimAI"})

# ── Create tables & run ───────────────────────────────────────────────────────
with app.app_context():
    db.create_all()  # ✅ Runs on Render with gunicorn too

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
