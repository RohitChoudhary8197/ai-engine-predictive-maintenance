import os
from datetime import datetime

import joblib
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash

from database import init_db, save_prediction, get_predictions, get_stats

app = Flask(__name__)
app.secret_key = "dev-key"

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL = joblib.load(os.path.join(BASE, "predictive_maintenance_model.pkl"))

TYPE_MAP = {"L": 0, "M": 1, "H": 2}
TYPE_LABELS = {"L": "Light", "M": "Medium", "H": "Heavy"}

init_db()


def predict_engine(form):
    air = float(form["air_temp"])
    proc = float(form["process_temp"])
    rpm = float(form["rpm"])
    torque = float(form["torque"])
    wear = float(form["tool_wear"])

    row = {
        "Type": TYPE_MAP.get(form.get("engine_type", "M"), 1),
        "Air temperature [K]": air,
        "Process temperature [K]": proc,
        "Rotational speed [rpm]": rpm,
        "Torque [Nm]": torque,
        "Tool wear [min]": wear,
        "Temp_Difference": proc - air,
        "Power_Index": rpm * torque,
        "Torque_RPM_Ratio": torque / rpm,
    }

    df = pd.DataFrame([row])[list(MODEL.feature_names_in_)]
    prob = float(MODEL.predict_proba(df)[0][1])
    fail = int(MODEL.predict(df)[0])

    if prob >= 0.6:
        risk, msg = "danger", "Turant service karwao — engine kharab ho sakta hai!"
        days = max(3, int(15 * (1 - prob)))
        health = max(10, int(100 - prob * 100))
    elif prob >= 0.3:
        risk, msg = "warning", "Jaldi service schedule karo."
        days = max(15, int(40 * (1 - prob)))
        health = max(40, int(75 - prob * 50))
    else:
        risk, msg = "safe", "Engine healthy hai — maintenance continue rakho."
        days = max(60, int(120 * (1 - prob)))
        health = max(70, int(95 - prob * 30))

    return {
        "fail": fail,
        "prob": round(prob * 100, 1),
        "risk": risk,
        "msg": msg,
        "days": days,
        "health": health,
        "time": datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "rpm": rpm,
        "torque": torque,
        "wear": wear,
        "engine": TYPE_LABELS.get(form.get("engine_type", "M"), "Medium"),
    }


@app.route("/")
def home():
    stats = get_stats()
    return render_template("index.html", stats=stats)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        result = predict_engine(request.form)
        save_prediction(request.form, result)
        return render_template("result.html", r=result)
    except Exception as e:
        flash(f"Error: {e}")
        return redirect(url_for("home"))


@app.route("/history")
def history():
    records = get_predictions()
    stats = get_stats()
    return render_template("history.html", records=records, stats=stats)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/project")
def project():
    return render_template("project.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404


if __name__ == "__main__":
    print("\n>>> Server: http://127.0.0.1:5000")
    print(">>> Database: data/pridectiveengine.db\n")
    app.run(host="127.0.0.1", port=5000, debug=True)
