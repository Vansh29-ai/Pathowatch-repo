from flask import Flask, send_file, jsonify, request
from flask_cors import CORS 
import pathowatch 
import requests
import os
import numpy as np
from dotenv import load_dotenv
from datetime import datetime, timedelta 
from meteostat import Point, daily 


# Load the .env file
load_dotenv()

WEATHER_KEY = os.getenv("OPENWEATHER_KEY")
WAQI_TOKEN = os.getenv("WAQI_TOKEN")

app = Flask(__name__)
CORS(app)

model, dataset, features, b2, b3, b4, b8 = None, None, None, None, None, None, None

# ---------------------------
# Load System on Startup
# ---------------------------
print("Initializing PathoWatch ML System...")
try:
    model, dataset, features, b2, b3, b4, b8 = pathowatch.load_model_system()
    print("System Ready: Vegetation Model Loaded.")
except Exception as e:
    print(f"Error loading model: {e}")

# ---------------------------
# Routes
# ---------------------------
@app.route("/")
def home():
    return {"status": "PathoWatch Unified Server Running"}

@app.route("/run_model")
def run_model():
    global model, dataset, features, b2, b3, b4, b8
    model, dataset, features, b2, b3, b4, b8 = pathowatch.load_model_system()
    pathowatch.generate_heatmap(model, features, b2)
    return {"status": "model_run_complete"}

@app.route("/risk_map")
def risk_map():
    return send_file("heatmap.png", mimetype="image/png")

@app.route("/human_risk")
def human_risk():
    lat_raw = request.args.get("lat")
    lon_raw = request.args.get("lon")
    
    if not lat_raw or not lon_raw:
        return jsonify({"error": "Missing coordinates"}), 400
    
    lat, lon = float(lat_raw), float(lon_raw)

    # 1. --- Meteostat Rainfall Fetch with Guard ---
    try:
        start = datetime.now() - timedelta(days=7)
        end = datetime.now()
        location = Point(lat, lon)
        rain_data = daily(location, start, end).fetch()
        
        # Check if rain_data exists before checking .empty
        if rain_data is not None and not rain_data.empty:
            weekly_rain = round(float(rain_data['prcp'].sum()), 1)
        else:
            weekly_rain = 0.0
            
        if np.isnan(weekly_rain): weekly_rain = 0.0
    except Exception as e:
        print(f"Meteostat Error: {e}")
        weekly_rain = 0.0

    # 2. --- External API Calls ---
    url_w = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_KEY}&units=metric"
    url_a = f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={WAQI_TOKEN}"
    
    try:
        w_data = requests.get(url_w).json()
        a_data = requests.get(url_a).json()
        
        temp = w_data['main']['temp']
        hum = w_data['main']['humidity']
        aqi = a_data['data']['aqi']
        
        # --- DISEASE RISK LOGIC ---
        vector_score = 0
        if 20 <= temp <= 32: vector_score += 30
        if hum > 60: vector_score += 30
        if 5 <= weekly_rain <= 50: vector_score += 40
        
        water_score = 0
        if temp > 28: water_score += 30
        if weekly_rain > 70: water_score += 70  
        elif weekly_rain > 30: water_score += 40 
        
        resp_score = min((aqi / 300) * 100, 100)
        
        avg_score = (vector_score + resp_score + water_score) / 3
        risk = "HIGH (RED)" if avg_score > 60 else "MEDIUM (YELLOW)" if avg_score > 35 else "LOW (GREEN)"
        
        return jsonify({
            "risk_level": risk,
            "data": {"temp": temp, "aqi": aqi, "humidity": hum, "weekly_rain": weekly_rain},
            "diseases": {
                "malaria_dengue": round(min(vector_score, 100), 1),
                "respiratory": round(resp_score, 1),
                "cholera_typhoid": round(min(water_score, 100), 1)
            },
            "ideals": {
                "temp": "22-26°C", "hum": "40-50%", "aqi": "< 50", "rain": "< 5mm"
            }
        })
    except Exception as e:
        print(f"API Fetch Error: {e}") 
        return jsonify({
            "risk_level": "ERROR", 
            "data": {"temp": "N/A", "aqi": "N/A", "humidity": "N/A", "weekly_rain": 0},
            "diseases": {"malaria_dengue": "Error", "respiratory": "Error", "cholera_typhoid": "Error"},
            "ideals": {"temp": "22-26°C", "hum": "40-50%", "aqi": "< 50"}
        })

# ADDED BACK: This fixes the 404 error on your map click
@app.route("/risk_at_location")
def risk_at_location():
    global model, dataset, features, b2, b3, b4, b8
    try:
        lat = float(request.args.get("lat"))
        lon = float(request.args.get("lon"))
        result = pathowatch.detect_dual_risk(model, dataset, features, b2, b3, b4, b8, lat, lon)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True, use_reloader=False)