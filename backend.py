from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathowatch import load_model_system, detect_dual_risk # Use the dual_risk function
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# --- CONFIGURATION (Use your working keys) ---
WEATHER_KEY = "d508e108be1c3f69dadcff2e2a91ab0f"
WAQI_TOKEN = "246de2c62ba017da669b1929f8497d0650afc46a"

print("Initializing PathoWatch ML System...")
model, dataset, features, b2, b3, b4, b8 = load_model_system()
print("System Ready")

# --- 1. VEGETATION DETECTION ---
@app.get("/risk_at_location")
def risk_at_location(lat: float, lon: float):
    # Calling the updated function from pathowatch.py
    result = detect_dual_risk(
        model, dataset, features, b2, b3, b4, b8, lat, lon
    )
    return result

# --- 2. HUMAN DISEASE RISK (Disease Probability Index) ---
@app.get("/human_risk")
def human_risk(lat: float, lon: float):
    try:
        # Fetch Weather
        w_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_KEY}&units=metric"
        weather = requests.get(w_url).json()
        
        # Fetch AQI
        a_url = f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={WAQI_TOKEN}"
        aqi_data = requests.get(a_url).json()

        temp = weather['main']['temp']
        hum = weather['main']['humidity']
        aqi = aqi_data['data']['aqi']

        # --- Disease Logic vs Ideal Conditions ---
        # Malaria/Dengue (Ideal: 24-29°C, Humidity 60-80%)
        vector_score = 0
        if 20 <= temp <= 32: vector_score += 40
        if hum > 60: vector_score += 40
        if hum > 80: vector_score += 20
        
        # Respiratory (Ideal AQI < 50)
        resp_score = min((aqi / 300) * 100, 100)
        
        avg_score = (vector_score + resp_score) / 2
        risk = "HIGH (RED)" if avg_score > 70 else "MEDIUM (YELLOW)" if avg_score > 40 else "LOW (GREEN)"

        # Return format matching index.html
        return {
            "risk_level": risk,
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
        print(f"API Error: {e}")
        return {"risk_level": "DATA ERROR", "data": {"temp": 0, "aqi": 0}}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)