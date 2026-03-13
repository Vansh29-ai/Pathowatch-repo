import rasterio
import numpy as np
import matplotlib
matplotlib.use('Agg')  # MUST be before pyplot import
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import requests
import os
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.ndimage import gaussian_filter
from datetime import datetime, timedelta
from meteostat import Point, daily 

# ---------------------------
# Secure API Configuration
# ---------------------------
load_dotenv()
WEATHER_KEY = os.getenv("OPENWEATHER_KEY")
WAQI_TOKEN = os.getenv("WAQI_TOKEN")

# ---------------------------
# Load Sentinel Bands
# ---------------------------
def load_bands():
    # Adjusted to ensure the folder path is consistent
    dataset = rasterio.open("Browser_images/B02.tiff")
    b2 = dataset.read(1)
    b3 = rasterio.open("Browser_images/B03.tiff").read(1)
    b4 = rasterio.open("Browser_images/B04.tiff").read(1)
    b8 = rasterio.open("Browser_images/B08.tiff").read(1)
    return dataset, b2, b3, b4, b8

# ---------------------------
# Feature Extraction
# ---------------------------
def extract_features(b2, b3, b4, b8):
    ndvi = (b8 - b4) / (b8 + b4 + 1e-10)
    ndwi = (b3 - b8) / (b3 + b8 + 1e-10)
    spectral_slope = (b8 - b2) / (b8 + b2 + 1e-10)
    absorption_depth = np.minimum.reduce([b2, b3, b4, b8])
    spectral_variance = np.var(np.stack([b2, b3, b4, b8], axis=0), axis=0)
    spectral_gradient = np.gradient(np.stack([b2, b3, b4, b8], axis=0), axis=0)[0]

    features = np.stack([
        b2, b3, b4, b8, ndvi, ndwi, 
        absorption_depth, spectral_variance, 
        spectral_slope, spectral_gradient
    ], axis=-1)

    return features, ndvi

# ---------------------------
# Train Machine Learning Model
# ---------------------------
def train_model(features, ndvi):
    psi = (1 - ndvi)
    X = features.reshape(-1, 10)
    threshold = np.mean(psi)
    labels = (psi > threshold).astype(int)
    y = labels.flatten()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=300, max_depth=20, 
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

# ---------------------------
# Human Disease Risk (Water, Vector, & Respiratory)
# ---------------------------
def get_human_environmental_risk(lat, lon):
    try:
        # 1. Fetch Rainfall with NoneType Guard
        try:
            start = datetime.now() - timedelta(days=7)
            end = datetime.now()
            location = Point(lat, lon)
            rain_data = daily(location, start, end).fetch()
            
            if rain_data is not None and not rain_data.empty:
                weekly_rain = round(float(rain_data['prcp'].sum()), 1)
            else:
                weekly_rain = 0.0
                
            if np.isnan(weekly_rain): weekly_rain = 0.0
        except:
            weekly_rain = 0.0

        # 2. Fetch Live Weather and AQI
        w_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_KEY}&units=metric"
        weather = requests.get(w_url).json()
        
        a_url = f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={WAQI_TOKEN}"
        aqi_data = requests.get(a_url).json()

        temp = weather["main"]["temp"]
        hum = weather["main"]["humidity"]
        aqi = aqi_data["data"]["aqi"]

        # --- SCORING LOGIC ---

        # Vector-Borne (Malaria/Dengue)
        vector_score = 0
        if 20 <= temp <= 32: vector_score += 30
        if hum > 60: vector_score += 30
        if 5 <= weekly_rain <= 50: vector_score += 40 
        
        # Water-Borne (Cholera/Typhoid)
        water_score = 0
        if temp > 28: water_score += 30 
        if weekly_rain > 70: water_score += 70 
        elif weekly_rain > 30: water_score += 40 
        
        # Respiratory (Flu/Asthma)
        resp_score = min((aqi / 300) * 100, 100)
        
        avg_score = (vector_score + resp_score + water_score) / 3
        risk_level = "HIGH (RED)" if avg_score > 60 else "MEDIUM (YELLOW)" if avg_score > 35 else "LOW (GREEN)"
        
        return {
            "risk_level": risk_level,
            "data": {"temp": temp, "aqi": aqi, "humidity": hum, "weekly_rain": weekly_rain},
            "diseases": {
                "malaria_dengue": round(min(vector_score, 100), 1),
                "respiratory": round(resp_score, 1),
                "cholera_typhoid": round(min(water_score, 100), 1)
            },
            "ideals": {
                "temp": "22-26 C", "hum": "40-50%", "aqi": "< 50", "rain": "< 5mm"
            }
        }
    except Exception as e:
        print(f"Logic Error in pathowatch.py: {e}")
        return {
            "risk_level": "ERROR", 
            "data": {"temp": "N/A", "aqi": "N/A", "humidity": "N/A", "weekly_rain": 0},
            "diseases": {"malaria_dengue": 0, "respiratory": 0, "cholera_typhoid": 0},
            "ideals": {"temp": "22-26 C", "hum": "40-50%", "aqi": "< 50", "rain": "< 5mm"}
        }

# ---------------------------
# Unified Detection
# ---------------------------
def detect_dual_risk(model, dataset, features, b2, b3, b4, b8, lat, lon):
    try:
        row, col = dataset.index(lon, lat)
        if 0 <= row < features.shape[0] and 0 <= col < features.shape[1]:
            pixel = features[row, col].reshape(1, -1)
            prob = float(model.predict_proba(pixel)[0][1])
            veg_risk = "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.4 else "LOW"
        else:
            veg_risk = "OUTSIDE BOUNDS"
            prob = 0.0
    except Exception:
        veg_risk = "ERROR"
        prob = 0.0

    human_info = get_human_environmental_risk(lat, lon)

    return {
        "veg_risk": veg_risk,
        "veg_prob": prob,
        "risk_level": human_info["risk_level"],
        "data": human_info["data"],
        "diseases": human_info["diseases"],
        "ideals": human_info["ideals"]
    }

# ---------------------------
# System Initialization & Heatmap
# ---------------------------
def load_model_system():
    print("Loading satellite bands...")
    dataset, b2, b3, b4, b8 = load_bands()
    
    print(f"\n🚀 COPY THESE COORDINATES: {dataset.bounds}\n")
    
    print("Extracting features...")
    features, ndvi = extract_features(b2, b3, b4, b8)
    
    print("Training Random Forest model...")
    model = train_model(features, ndvi)
    
    return model, dataset, features, b2, b3, b4, b8

def generate_heatmap(model, features, b2):
    print("Generating disease risk heatmap...")
    X_full = features.reshape(-1, 10)
    
    probs = model.predict_proba(X_full)[:, 1]
    prob_map = probs.reshape(features.shape[0], features.shape[1])
    
    smoothed_map = gaussian_filter(prob_map, sigma=2)

    plt.figure(figsize=(10, 10))
    plt.imshow(smoothed_map, cmap="jet", alpha=0.5)
    plt.axis("off")
    plt.savefig("heatmap.png", bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close()
    print("Heatmap saved successfully.")