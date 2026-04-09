# =============================================================================
# PathoWatch — Unified Server  (Docker / Render ready)
# SAM-based dual-domain spectral matching pipeline.
# human_risk / analyze_dynamic_location unchanged from original.
# =============================================================================

from flask import Flask, send_file, jsonify, request, send_from_directory
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
WAQI_TOKEN  = os.getenv("WAQI_TOKEN")

# ------------------------------------------------------------------
# GEE Authentication
# Locally : uses your cached personal credentials (ee auth login)
# On Render: uses GEE_KEY_JSON + GEE_SERVICE_ACCOUNT env vars
# ------------------------------------------------------------------
def init_gee():
    key_json = os.getenv("GEE_KEY_JSON")
    if key_json:
        credentials = ee.ServiceAccountCredentials(
            email   = os.getenv("GEE_SERVICE_ACCOUNT"),
            key_data= key_json
        )
        ee.Initialize(credentials, project="pathowatch-vibhav-492519")
        print("[GEE] Initialized with service account")
    else:
        ee.Initialize(project="pathowatch-vibhav-492519")
        print("[GEE] Initialized with local credentials")

init_gee()

app = Flask(__name__, static_folder=".")
CORS(app)

# Global state — SAM pipeline stores model=None, heatmap=2d array
_model   = None   # kept for API compat; SAM needs no model object
_heatmap = None   # vegetation SAM similarity map


# ------------------------------------------------------------------
# Serve Frontend
# ------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(".", "index.html")


# ------------------------------------------------------------------
# Health check
# ------------------------------------------------------------------
@app.route("/health")
def health():
    return jsonify({"status": "ok", "pipeline": "SAM-based spectral matching"})


# ------------------------------------------------------------------
# API info
# ------------------------------------------------------------------
@app.route("/api")
def api_info():
    return jsonify({
        "message": "PathoWatch SAM Pipeline API",
        "pipeline": "Spectral Angle Mapper — USGS reference spectra",
        "routes": [
            "/run_model", "/risk_map", "/human_risk_map",
            "/risk_stats", "/risk_at_location", "/hotspots",
            "/analyze_location", "/analyze_dynamic_location",
            "/human_risk"
        ]
    })


# ------------------------------------------------------------------
# Run Model  →  now runs the SAM pipeline
# ------------------------------------------------------------------
@app.route("/run_model")
def run_model():
    global _model, _heatmap
    try:
        lat = float(request.args.get("lat", 28.6139))
        lon = float(request.args.get("lon", 77.2090))

        # Try GEE-based pipeline first; fall back to local tiffs if GEE fails
        try:
            _model, _heatmap = pathowatch_pipeline.run_pipeline(lat, lon)
        except Exception as gee_err:
            print(f"[server] GEE pipeline failed ({gee_err}). Trying local fallback.")
            _model, _heatmap = pathowatch_pipeline.run_pipeline_local()

        return jsonify({
            "status":   "sam_pipeline_complete",
            "mode":     "spectral_angle_mapper",
            "lat":      lat,
            "lon":      lon,
        })
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 500


# ------------------------------------------------------------------
# Vegetation risk map PNG
# ------------------------------------------------------------------
@app.route("/risk_map")
def risk_map():
    if _heatmap is None:
        return jsonify({"error": "Run /run_model first."}), 404
    return send_file("risk_map.png", mimetype="image/png")


# ------------------------------------------------------------------
# Human vector habitat risk map PNG  (new endpoint)
# ------------------------------------------------------------------
@app.route("/human_risk_map")
def human_risk_map_image():
    if not os.path.exists("human_risk_map.png"):
        return jsonify({"error": "Run /run_model first."}), 404
    return send_file("human_risk_map.png", mimetype="image/png")


# ------------------------------------------------------------------
# Risk statistics
# ------------------------------------------------------------------
@app.route("/risk_stats")
def risk_stats():
    if _heatmap is None:
        return jsonify({"error": "Model not run"}), 404
    stats = pathowatch_pipeline.compute_stats(_heatmap)
    return jsonify(stats)


# ------------------------------------------------------------------
# Risk at a specific lat/lon (reads from last-run heatmap + raster)
# ------------------------------------------------------------------
@app.route("/risk_at_location")
def risk_at_location():
    if _heatmap is None:
        return jsonify({"error": "Model not run"}), 404

    lat = float(request.args.get("lat"))
    lon = float(request.args.get("lon"))

    # Try sentinel.tif first, then veg_risk.tif
    raster_path = "sentinel.tif" if os.path.exists("sentinel.tif") else "veg_risk.tif"
    try:
        with rasterio.open(raster_path) as src:
            row, col = src.index(lon, lat)
        probability = float(_heatmap[row, col])
    except Exception:
        return jsonify({"error": "Location outside satellite image bounds"}), 400

    risk = "HIGH" if probability > 0.7 else "MEDIUM" if probability > 0.4 else "LOW"
    return jsonify({
        "latitude": lat, "longitude": lon,
        "sam_similarity": probability,
        "risk_level": risk
    })


# ------------------------------------------------------------------
# Hotspots
# ------------------------------------------------------------------
@app.route("/hotspots")
def hotspots():
    if _heatmap is None:
        return jsonify({"error": "Model not run"}), 404
    threshold = np.percentile(_heatmap, 95)
    points    = np.argwhere(_heatmap >= threshold)
    result    = [{"row": int(r), "col": int(c)} for r, c in points[::500]]
    return jsonify({"hotspots": result, "threshold": float(threshold)})


# ------------------------------------------------------------------
# Analyze location (local raster)
# ------------------------------------------------------------------
@app.route("/analyze_location")
def analyze_location():
    if _heatmap is None:
        return jsonify({"error": "Model not run"}), 404

    lat = float(request.args.get("lat"))
    lon = float(request.args.get("lon"))
    raster_path = "sentinel.tif" if os.path.exists("sentinel.tif") else "veg_risk.tif"

    try:
        with rasterio.open(raster_path) as src:
            row, col = src.index(lon, lat)
        probability = float(_heatmap[row, col])
    except Exception:
        return jsonify({"error": "Location outside satellite image bounds"}), 400

    if probability > 0.7:
        risk, alert = "HIGH",   "⚠️ High spectral similarity to pathogen signature detected"
    elif probability > 0.4:
        risk, alert = "MEDIUM", "⚠️ Moderate pathogen spectral match"
    else:
        risk, alert = "LOW",    "✅ Area spectrally dissimilar to known pathogens"

    days   = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    values = (probability + np.random.normal(0, 0.04, 7)).clip(0, 1)
    return jsonify({
        "lat": lat, "lon": lon,
        "probability": probability,
        "risk": risk, "alert": alert,
        "method": "spectral_angle_mapper",
        "days": days, "values": values.tolist()
    })


# ------------------------------------------------------------------
# Analyze dynamic location (GEE NDVI-based — unchanged)
# ------------------------------------------------------------------
@app.route("/analyze_dynamic_location")
def analyze_dynamic_location():
    lat = float(request.args.get("lat"))
    lon = float(request.args.get("lon"))
    try:
        point      = ee.Geometry.Point([lon, lat])
        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(point)
            .filterDate("2024-01-01", "2024-12-31")
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
        # Invert NDVI: low green = potential stress/disease
        probability = float(max(0, min(1, 1 - ndvi_value)))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    risk  = "HIGH" if probability > 0.7 else "MEDIUM" if probability > 0.4 else "LOW"
    alert = (
        "⚠️ High pathogen risk — vegetation severely stressed"   if probability > 0.7 else
        "⚠️ Moderate pathogen risk — some vegetation stress"      if probability > 0.4 else
        "✅ Area appears healthy"
    )
    days   = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekly = (probability + np.random.normal(0, 0.04, 7)).clip(0, 1)
    return jsonify({
        "lat": lat, "lon": lon,
        "probability": probability,
        "risk": risk, "alert": alert,
        "method": "ndvi_inversion",
        "days": days, "values": weekly.tolist()
    })


# ------------------------------------------------------------------
# Human Disease Probability Index  (unchanged from original)
# ------------------------------------------------------------------
@app.route("/human_risk")
def human_risk():
    try:
        lat = float(request.args.get("lat"))
        lon = float(request.args.get("lon"))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid or missing lat/lon"}), 400

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
        return jsonify({
            "risk_level": "ERROR",
            "data": {"temp": "N/A", "aqi": "N/A",
                     "humidity": "N/A", "weekly_rain": 0},
            "diseases": {"malaria_dengue": 0,
                         "respiratory": 0, "cholera_typhoid": 0},
            "ideals": {"temp": "22-26°C", "hum": "40-50%",
                       "aqi": "< 50",    "rain": "< 5mm/week"}
        }), 500


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)