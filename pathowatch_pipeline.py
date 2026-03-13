import rasterio
import numpy as np
import matplotlib.pyplot as plt
import ee
import geemap
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.ndimage import gaussian_filter

# ---------------------------
# 1. Download Sentinel Data (Vegetation)
# ---------------------------
def download_satellite(lat, lon):
    try:
        ee.Initialize(project="pathowatch-vibhav")
    except:
        ee.Authenticate()
        ee.Initialize(project="pathowatch-vibhav")

    point = ee.Geometry.Point([lon, lat])
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(point)
        .filterDate("2024-01-01", "2026-03-11")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
    )

    image = collection.first()
    bands = image.select(["B2", "B3", "B4", "B8"])

    # This is the base file for your ML processing
    filename = "sentinel.tif"
    geemap.ee_export_image(
        bands,
        filename=filename,
        scale=10,
        region=point.buffer(5000).bounds()
    )
    return filename

# ---------------------------
# 2. Human Disease Risk (WAQI + OpenWeather)
# ---------------------------
# ---------------------------
# 2. Human Disease Risk (WAQI + OpenWeather)
# ---------------------------
def get_human_risk(lat, lon, weather_key, waqi_token):
    try:
        # Fetch Weather
        w_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={weather_key}&units=metric"
        weather = requests.get(w_url).json()
        
        # Fetch AQI
        a_url = f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={waqi_token}"
        aqi_data = requests.get(a_url).json()

        temp = weather['main']['temp']
        hum = weather['main']['humidity']
        aqi = aqi_data['data']['aqi']

        # --- COMPARISON LOGIC VS IDEAL CONDITIONS ---
        
        # 1. Vector-Borne Risk (Malaria/Dengue)
        # Ideal breeding: 24-29°C, Humidity 60-80%
        vector_score = 0
        if 20 <= temp <= 32: vector_score += 40  # Temperature comfort zone for vectors
        if hum > 60: vector_score += 40          # Humidity threshold for survival
        if hum > 80: vector_score += 20          # Exponential breeding increase
        
        # 2. Respiratory Risk (AQI Thresholds)
        # Ideal: AQI < 50
        resp_score = min((aqi / 300) * 100, 100) # Scales risk up to 100% at AQI 300

        # Weighted Overall Risk for the Badge
        avg_score = (vector_score + resp_score) / 2
        risk_level = "HIGH (RED)" if avg_score > 70 else "MEDIUM (YELLOW)" if avg_score > 40 else "LOW (GREEN)"
        
        return {
            "risk_level": risk_level,
            "data": {"temp": temp, "aqi": aqi, "humidity": hum},
            "diseases": {
                "malaria_dengue": round(vector_score, 1),
                "respiratory": round(resp_score, 1)
            },
            "ideals": {
                "temp": "22-26°C",
                "hum": "40-50%",
                "aqi": "< 50"
            }
        }
    except Exception as e:
        return {"error": "API Fetch Failed", "details": str(e)}
# ---------------------------
# 3. Feature Extraction (Vegetation)
# ---------------------------
def extract_veg_features(b2, b3, b4, b8):
    # Standard Vegetation Index
    ndvi = (b8 - b4) / (b8 + b4 + 1e-10)
    # Water Stress Index
    ndwi = (b3 - b8) / (b3 + b8 + 1e-10)
    
    # Stack original bands + indices
    features = np.stack([b2, b3, b4, b8, ndvi, ndwi], axis=-1)
    return features, ndvi

# ---------------------------
# 4. Train & Generate Heatmap
# ---------------------------
def run_unified_pipeline(lat, lon, w_key, waqi_token):
    # Step A: Human Analysis
    human_results = get_human_risk(lat, lon, w_key, waqi_token)

    # Step B: Vegetation Analysis
    filename = download_satellite(lat, lon)
    dataset = rasterio.open(filename)
    # Reading the 4 sentinel bands
    b2, b3, b4, b8 = dataset.read(1), dataset.read(2), dataset.read(3), dataset.read(4)
    
    features, ndvi = extract_veg_features(b2, b3, b4, b8)
    
    # Train Vegetation Model (Pathogen Stress)
    psi = (1 - ndvi)
    X = features.reshape(-1, 6)
    # Define "High Risk" as the top 25% of stressed areas
    y = (psi > np.percentile(psi, 75)).astype(int).flatten()
    
    model = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1)
    model.fit(X, y)
    
    # Generate Heatmap
    probs = model.predict_proba(X)[:, 1]
    heatmap = gaussian_filter(probs.reshape(b2.shape), sigma=2)
    
    # --- IMPORTANT: Name must be 'heatmap.png' for server2.py to find it ---
    plt.imsave("heatmap.png", heatmap, cmap="RdYlGn_r")

    return {
        "model": model,
        "heatmap": heatmap,
        "human_risk": human_results,
        "dataset": dataset
    }