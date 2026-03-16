# =============================================================================
# PathoWatch — Unified Server (merged)
# Keeps: all original routes + human disease probability index
# =============================================================================

from flask import Flask, send_file, jsonify, request
from flask_cors import CORS
import pathowatch_pipeline
import rasterio
import numpy as np
import requests
import os
import ee
from dotenv import load_dotenv

load_dotenv()
WEATHER_KEY = os.getenv("OPENWEATHER_KEY")
WAQI_TOKEN   = os.getenv("WAQI_TOKEN")

ee.Initialize(project="pathowatch-vibhav")

app = Flask(__name__)
CORS(app)

model   = None
heatmap = None

# ---------------------------
# Home
# ---------------------------
@app.route("/")
def home():
    return jsonify({
        "message": "PathoWatch Unified API",
        "routes": [
            "/run_model", "/risk_map", "/risk_stats",
            "/risk_at_location", "/hotspots",
            "/analyze_location", "/analyze_dynamic_location",
            "/human_risk"
        ]
    })

# ---------------------------
# Run Model
# ---------------------------
@app.route("/run_model")
def run_model():
    global model, heatmap
    try:
        model, heatmap = pathowatch_pipeline.run_pipeline()
        return jsonify({"status": "model_run_complete"})
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 500

# ---------------------------
# Risk Map Image
# ---------------------------
@app.route("/risk_map")
def risk_map():
    if heatmap is None:
        return jsonify({"error": "Model not run yet. Call /run_model first."}), 404
    return send_file("risk_map.png", mimetype="image/png")

# ---------------------------
# Risk Statistics
# ---------------------------
@app.route("/risk_stats")
def risk_stats():
    if heatmap is None:
        return jsonify({"error": "Model not run"}), 404
    high   = int((heatmap > 0.7).sum())
    medium = int(((heatmap > 0.4) & (heatmap <= 0.7)).sum())
    low    = int((heatmap <= 0.4).sum())
    return jsonify({
        "high_risk_pixels":   high,
        "medium_risk_pixels": medium,
        "low_risk_pixels":    low
    })

# ---------------------------
# Risk at Location (satellite)
# ---------------------------
@app.route("/risk_at_location")
def risk_at_location():
    if heatmap is None:
        return jsonify({"error": "Model not run"}), 404
    lat = float(request.args.get("lat"))
    lon = float(request.args.get("lon"))
    try:
        dataset     = rasterio.open("sentinel.tif")
        row, col    = dataset.index(lon, lat)
        probability = float(heatmap[row, col])
    except:
        return jsonify({"error": "Location outside satellite image"}), 400
    risk = "HIGH" if probability > 0.7 else "MEDIUM" if probability > 0.4 else "LOW"
    return jsonify({"latitude": lat, "longitude": lon,
                    "probability": probability, "risk_level": risk})

# ---------------------------
# Hotspots
# ---------------------------
@app.route("/hotspots")
def hotspots():
    if heatmap is None:
        return jsonify({"error": "Model not run"}), 404
    points   = np.argwhere(heatmap > 0.75)
    result   = [{"row": int(r), "col": int(c)} for r, c in points[::500]]
    return jsonify({"hotspots": result})

# ---------------------------
# Analyze Location (local tif)
# ---------------------------
@app.route("/analyze_location")
def analyze_location():
    if heatmap is None:
        return jsonify({"error": "Model not run"}), 404
    lat = float(request.args.get("lat"))
    lon = float(request.args.get("lon"))
    try:
        dataset     = rasterio.open("sentinel.tif")
        row, col    = dataset.index(lon, lat)
        probability = float(heatmap[row, col])
    except:
        return jsonify({"error": "Location outside satellite image"}), 400

    if probability > 0.7:
        risk, alert = "HIGH",   "⚠️ High pathogen concentration detected"
    elif probability > 0.4:
        risk, alert = "MEDIUM", "⚠️ Moderate pathogen risk"
    else:
        risk, alert = "LOW",    "✅ Area appears safe"

    days   = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    values = (probability + np.random.normal(0, 0.05, 7)).clip(0, 1)
    return jsonify({"lat": lat, "lon": lon, "probability": probability,
                    "risk": risk, "alert": alert,
                    "days": days, "values": values.tolist()})

# ---------------------------
# Analyze Dynamic Location (GEE)
# ---------------------------
@app.route("/analyze_dynamic_location")
def analyze_dynamic_location():
    lat = float(request.args.get("lat"))
    lon = float(request.args.get("lon"))
    try:
        point      = ee.Geometry.Point([lon, lat])
        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(point)
            .filterDate("2023-01-01", "2023-12-31")
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        )
        image      = collection.first()
        ndvi       = image.normalizedDifference(["B8", "B4"])
        value      = ndvi.reduceRegion(
            reducer  = ee.Reducer.mean(),
            geometry = point.buffer(200),
            scale    = 10
        ).get("nd")
        ndvi_value  = value.getInfo()
        probability = float(1 - ndvi_value)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    if probability > 0.7:
        risk, alert = "HIGH",   "⚠️ High pathogen risk detected"
    elif probability > 0.4:
        risk, alert = "MEDIUM", "⚠️ Moderate pathogen risk"
    else:
        risk, alert = "LOW",    "✅ Area appears safe"

    days   = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    weekly = (probability + np.random.normal(0, 0.05, 7)).clip(0, 1)
    return jsonify({"lat": lat, "lon": lon, "probability": probability,
                    "risk": risk, "alert": alert,
                    "days": days, "values": weekly.tolist()})

# ---------------------------
# Human Disease Probability Index  ← YOUR FEATURE
# ---------------------------
@app.route("/human_risk")
def human_risk():
    try:
        lat = float(request.args.get("lat"))
        lon = float(request.args.get("lon"))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid or missing lat/lon"}), 400

    # Rainfall via OWM forecast (48hr, 3hr slots)
    weekly_rain = 0.0
    try:
        fc_url  = (f"https://api.openweathermap.org/data/2.5/forecast"
                   f"?lat={lat}&lon={lon}&appid={WEATHER_KEY}&units=metric&cnt=16")
        fc_data = requests.get(fc_url, timeout=10).json()
        weekly_rain = round(sum(
            s.get("rain", {}).get("3h", 0.0) for s in fc_data.get("list", [])
        ), 1)
    except Exception as e:
        print(f"[Rainfall] {e}")

    try:
        w_data = requests.get(
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?lat={lat}&lon={lon}&appid={WEATHER_KEY}&units=metric",
            timeout=10).json()
        a_data = requests.get(
            f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={WAQI_TOKEN}",
            timeout=10).json()

        temp = w_data["main"]["temp"]
        hum  = w_data["main"]["humidity"]
        aqi  = a_data["data"]["aqi"]

        vector_score = 0
        if 20 <= temp <= 32:       vector_score += 30
        if hum > 60:               vector_score += 30
        if 5 <= weekly_rain <= 50: vector_score += 40

        water_score = 0
        if temp > 28:              water_score += 30
        if weekly_rain > 70:       water_score += 70
        elif weekly_rain > 30:     water_score += 40

        resp_score = min((aqi / 300) * 100, 100)
        avg_score  = (vector_score + resp_score + water_score) / 3
        risk = "HIGH" if avg_score > 60 else "MEDIUM" if avg_score > 35 else "LOW"

        return jsonify({
            "risk_level": risk,
            "data": {"temp": temp, "aqi": aqi,
                     "humidity": hum, "weekly_rain": weekly_rain},
            "diseases": {
                "malaria_dengue":  round(min(vector_score, 100), 1),
                "respiratory":     round(resp_score, 1),
                "cholera_typhoid": round(min(water_score, 100), 1)
            },
            "ideals": {
                "temp": "22-26°C", "hum": "40-50%",
                "aqi": "< 50",    "rain": "< 5mm/week"
            }
        })
    except Exception as e:
        print(f"[human_risk] {e}")
        return jsonify({"risk_level": "ERROR",
                        "data": {"temp":"N/A","aqi":"N/A","humidity":"N/A","weekly_rain":0},
                        "diseases": {"malaria_dengue":0,"respiratory":0,"cholera_typhoid":0},
                        "ideals": {"temp":"22-26°C","hum":"40-50%","aqi":"< 50","rain":"< 5mm/week"}
                        }), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)