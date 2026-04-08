# 🦠 PathoWatch — Pathogen Risk Intelligence Platform

> Real-time vegetation pathogen detection and human disease risk analysis using Sentinel-2 satellite imagery, Google Earth Engine, and live environmental data.

---

## 🚨 Problem Statement

Every year, millions of people in India and across South Asia are affected by preventable diseases — malaria, dengue, cholera, typhoid, and respiratory illnesses caused by air pollution. At the same time, crop diseases silently devastate agricultural yields before farmers even notice symptoms on the ground.

The core problem is a **lack of early, location-specific risk intelligence**:

- Farmers have no way to detect vegetation pathogen stress before it becomes visible
- Public health workers lack real-time, hyperlocal disease risk data
- Existing systems are either too expensive, too slow, or too coarse-grained to be actionable
- No single platform combines satellite vegetation analysis with ground-level human health risk

**Vulnerable populations — farmers, rural communities, and urban residents in high-density areas — make decisions every day without access to the environmental risk data that could protect them.**

---

## 💡 Solution

PathoWatch is a unified web platform that combines:

1. **Satellite-based vegetation pathogen detection** using Sentinel-2 multispectral imagery processed via Google Earth Engine and a Random Forest ML model
2. **Real-time human disease risk scoring** using live weather (temperature, humidity, rainfall) and air quality index data
3. **Actionable recommendations** tailored to the specific risk profile of any location on Earth

A user simply clicks a location on the map or searches by name — PathoWatch instantly returns vegetation health status, disease probability scores for malaria/dengue, respiratory illness, and cholera/typhoid, plus specific protective actions to take.

---

## 🌍 How It Works

```
User selects location
        │
        ▼
┌─────────────────────────────────────────────┐
│           PathoWatch Backend (Flask)         │
│                                             │
│  ┌─────────────────┐  ┌──────────────────┐  │
│  │ Vegetation Path │  │  Human Disease   │  │
│  │ (GEE + Sentinel)│  │  Risk Engine     │  │
│  └────────┬────────┘  └────────┬─────────┘  │
│           │                    │            │
│  NDVI from Copernicus S2  OpenWeather API  │
│  Random Forest Classifier  WAQI AQI API   │
│  Pathogen Probability      Rainfall 48hr  │
└─────────────────────────────────────────────┘
        │
        ▼
   Risk Score + Alert + Weekly Trend + Recommendations
        │
        ▼
   Interactive Map Dashboard (Leaflet.js)
```

### Vegetation Pathogen Detection

- Fetches Sentinel-2 SR Harmonized imagery from Google Earth Engine for the selected location
- Computes **NDVI** (Normalized Difference Vegetation Index) — healthy vegetation absorbs more red light and reflects near-infrared; stressed or pathogen-infected vegetation shows reduced NDVI
- `probability = 1 - NDVI` — higher pathogen stress = lower NDVI = higher risk probability
- Risk thresholds: `> 0.7` = HIGH, `> 0.4` = MEDIUM, `≤ 0.4` = LOW

### Human Disease Risk Engine

Three disease categories are scored independently using live data:

| Disease | Key Signals | Scoring Logic |
|---|---|---|
| Vector-borne (Malaria/Dengue) | Temperature 20–32°C, Humidity > 60%, Rainfall 5–50mm | Additive score up to 100 |
| Water-borne (Cholera/Typhoid) | Temperature > 28°C, Rainfall > 30mm | Heavy rain = contamination risk |
| Respiratory (Flu/Asthma/Pollution) | AQI score | `min((AQI / 300) × 100, 100)` |

Overall risk = average of the three scores. `> 60` = HIGH, `> 35` = MEDIUM, else LOW.

---

## 🏗️ Project Architecture

```
Pathowatch-repo/
│
├── server.py                  # Flask backend — all API routes
├── pathowatch_pipeline.py     # ML pipeline — satellite download, feature extraction, model
├── pathowatch.py              # Standalone analysis utilities
├── index.html                 # Frontend — single-file UI (Leaflet, Chart.js)
│
├── Dockerfile                 # Container definition
├── .dockerignore              # Files excluded from Docker image
├── requirements.txt           # Python dependencies
├── Procfile                   # Render/Heroku process definition
│
├── Browser_images/            # Local Sentinel band TIFFs (B02, B03, B04, B08)
│   ├── B02.tiff
│   ├── B03.tiff
│   ├── B04.tiff
│   └── B08.tiff
│
└── .env                       # Local secrets (never committed)
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Map | Leaflet.js + OpenStreetMap |
| Charts | Chart.js |
| Geocoding | Nominatim (OpenStreetMap) |
| Backend | Python 3.11, Flask, Flask-CORS |
| ML Model | Scikit-learn Random Forest Classifier |
| Satellite Data | Google Earth Engine — Copernicus/S2_SR_HARMONIZED |
| Satellite Processing | rasterio, numpy, scipy |
| Weather API | OpenWeatherMap (current weather + 48hr forecast) |
| Air Quality API | WAQI (World Air Quality Index) |
| Server | Gunicorn (WSGI) |
| Containerization | Docker |
| Deployment | Render (Docker-based web service) |

---

## 📡 API Routes

| Route | Method | Description |
|---|---|---|
| `/` | GET | Serves the frontend (index.html) |
| `/health` | GET | Health check |
| `/analyze_dynamic_location` | GET | GEE-based NDVI analysis for lat/lon |
| `/human_risk` | GET | Human disease probability scores for lat/lon |
| `/run_model` | GET | Runs full local ML pipeline (requires TIFF files) |
| `/risk_map` | GET | Returns heatmap PNG from local pipeline |
| `/risk_stats` | GET | Pixel-level risk distribution stats |
| `/analyze_location` | GET | Local TIFF-based analysis |
| `/risk_at_location` | GET | Risk value at specific pixel |
| `/hotspots` | GET | High-risk pixel coordinates |

**Primary routes used by the frontend:** `/analyze_dynamic_location` and `/human_risk`

---

## ⚙️ Environment Variables

Create a `.env` file in the project root:

```env
OPENWEATHER_KEY=your_openweathermap_api_key
WAQI_TOKEN=your_waqi_api_token
GEE_SERVICE_ACCOUNT=your-sa@your-project.iam.gserviceaccount.com
GEE_KEY_JSON={"type":"service_account","project_id":"..."}
```

| Variable | Where to get it |
|---|---|
| `OPENWEATHER_KEY` | [openweathermap.org/api](https://openweathermap.org/api) — free tier |
| `WAQI_TOKEN` | [aqicn.org/data-platform/token](https://aqicn.org/data-platform/token/) — free |
| `GEE_SERVICE_ACCOUNT` | Google Cloud Console → IAM → Service Accounts |
| `GEE_KEY_JSON` | Download JSON key from the service account |

---

## 🚀 Running Locally

### Option A — Python directly

```bash
pip install -r requirements.txt
python server.py
# Open http://localhost:5000
```

### Option B — Docker (recommended)

```bash
# Build
docker build -t pathowatch .

# Run
docker run --env-file .env -p 8080:8080 pathowatch

# Open http://localhost:8080
```

---

## ☁️ Deployment (Render)

### Prerequisites

1. GitHub repo with all files pushed
2. Google Cloud service account with these IAM roles:
   - `roles/earthengine.viewer`
   - `roles/serviceusage.serviceUsageConsumer`
3. Service account registered with Earth Engine

### Steps

1. Go to [render.com](https://render.com) → **New → Web Service**
2. Connect your GitHub repository
3. Settings:
   - **Runtime:** Docker
   - **Region:** Singapore
   - **Branch:** main
4. Add environment variables (same 4 as your `.env`)
5. Click **Deploy**

Build takes 5–8 minutes. Your app will be live at `https://your-app.onrender.com`.

> **Note:** Render free tier spins down after 15 minutes of inactivity. First request after sleep takes ~30 seconds. Upgrade to $7/month for always-on.

---

## 🔐 Google Earth Engine Service Account Setup

```
1. console.cloud.google.com → your project
2. IAM & Admin → Service Accounts → Create Service Account
3. Download JSON key
4. IAM → Grant Access → add both roles listed above
5. Paste JSON contents into GEE_KEY_JSON env var
```

---

## 📊 Features

- **Interactive map** — click anywhere or search by city/village name
- **Real-time GEE analysis** — Sentinel-2 NDVI computed on demand via Google Earth Engine
- **Weekly trend chart** — 7-day pathogen concentration visualization
- **Human disease dashboard** — probability scores for 3 disease categories
- **Environmental metrics** — live AQI, temperature, humidity, 48hr rainfall
- **Actionable recommendations** — specific protective measures based on risk level
- **Dark mode UI** — cyberpunk-themed dashboard designed for field use
- **Dual input modes** — search by name or enter coordinates directly

---

## 🧠 ML Model Details

| Parameter | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Estimators | 200 trees |
| Max depth | 20 |
| Features | NDVI, NDWI, B2, B3, B4, B8 (6 spectral features) |
| Labels | Top 75th percentile PSI = pathogen stress |
| Train/test split | 80/20 |
| Image source | Copernicus Sentinel-2 SR Harmonized, 2023, cloud < 20% |
| Resolution | 10m per pixel |

The model is trained on-the-fly from downloaded satellite data when `/run_model` is called. For the primary use case (clicking on map), GEE computes NDVI directly without needing local model training.

---

## 🙌 Team

Built for **PathoWatch** — Pathogen Risk Intelligence Platform

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.