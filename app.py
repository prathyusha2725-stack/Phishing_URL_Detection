
from pathlib import Path
from urllib.parse import urlsplit
import ipaddress
import re

import joblib
from flask import Flask, render_template, request, jsonify


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "phishing_model.joblib"

app = Flask(__name__)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Model not found. Run python train_model.py first."
    )

model = joblib.load(MODEL_PATH)


def validate_url(value):
    if not isinstance(value, str):
        return None

    value = value.strip()

    if not value or len(value) > 2048:
        return None

    # Require an explicit HTTP or HTTPS URL.
    if not re.match(r"^https?://", value, re.IGNORECASE):
        return None

    try:
        parsed = urlsplit(value)

        if parsed.scheme.lower() not in ("http", "https"):
            return None

        if not parsed.hostname or not parsed.netloc:
            return None

        # Reject malformed ports and invalid IP address literals.
        _ = parsed.port

        try:
            ipaddress.ip_address(parsed.hostname)
        except ValueError:
            pass

        # Reject whitespace and control characters.
        if any(ord(char) <= 32 for char in value):
            return None

        return value

    except (ValueError, UnicodeError):
        return None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    submitted_url = data.get("url", "")
    url = validate_url(submitted_url)

    if url is None:
        return jsonify({
            "error": (
                "Enter a valid URL beginning with "
                "http:// or https://."
            )
        }), 400

    # The application classifies the URL string only.
    # It does not visit or download the submitted website.
    prediction = int(model.predict([url])[0])

    if prediction == 1:
        result = "Likely Legitimate"
        risk = "Lower predicted risk"
    else:
        result = "Likely Phishing"
        risk = "Higher predicted risk"

    return jsonify({
        "url": url,
        "prediction": prediction,
        "result": result,
        "risk": risk,
        "message": (
            "This is an ML prediction, not a guarantee "
            "of website safety."
        )
    })


if __name__ == "__main__":
    app.run(debug=True)
