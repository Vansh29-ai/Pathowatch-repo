# 🦠 PathoWatch — Pathogen Risk Intelligence Platform

> Real-time vegetation pathogen detection and human disease risk analysis using satellite imagery, spectral intelligence, and live environmental data.

---

## 🚨 Problem Statement

Every year, millions of people in India and across South Asia are affected by preventable diseases — malaria, dengue, cholera, typhoid, and respiratory illnesses caused by air pollution. At the same time, crop diseases silently devastate agricultural yields before farmers even notice symptoms on the ground.

The core problem is a **lack of early, location-specific risk intelligence**:

* Farmers cannot detect vegetation stress before visible symptoms
* Public health systems lack hyperlocal, real-time insights
* Existing tools are expensive, slow, or too coarse
* No unified platform combines environmental + health intelligence

---

## 💡 Solution

PathoWatch is a unified platform that combines:

1. **Satellite-based vegetation pathogen detection**
2. **Dynamic spectral intelligence using real scientific datasets**
3. **Human disease risk scoring using live environmental data**
4. **Actionable recommendations for users**

Users simply select a location → PathoWatch returns:

* Vegetation stress probability
* Disease risk scores
* Environmental insights
* Preventive actions

---

# 🧠 Major Upgrade — Spectral Intelligence Pipeline (v2)

> ⚡ PathoWatch now uses a **fully dynamic, data-driven pipeline**

---

## 🔄 What Changed

### ❌ Removed (Old System)

* Hardcoded spectral dictionaries
* Fixed wavelength arrays
* Static thresholds
* Limited feature extraction
* SAM-only detection
* Fixed disease targets

---

### ✅ Added (New System)

* 🌐 Real **USGS spectral library integration**
* 📡 **Dynamic wavelength extraction**
* 📊 **Statistical thresholding (per image)**
* 🧠 **Adaptive feature engineering**
* 🤖 **Hybrid ML pipeline (RF + SVM)**
* ⚙️ **Runtime configurable targets**

---

## 🧬 New Pipeline Flow

```
Input Raster (Sentinel / PRISMA / Hyperspectral)
        │
        ▼
Extract Wavelengths (metadata-driven)
        │
        ▼
Fetch USGS Reference Spectrum
        │
        ▼
Spectral Matching (SAM)
        │
        ▼
Dynamic Thresholding
        │
        ▼
Feature Extraction (adaptive)
        │
        ▼
ML Training (RF + SVM)
        │
        ▼
Ensemble Prediction
        │
        ▼
Risk Map + Stats + Evaluation
```

---

## 📡 Core Components

### 1. `fetch_usgs_spectrum()`

* Downloads real spectral data from USGS splib07
* Converts µm → nm
* Caches locally (.npz)
* Falls back to literature if offline

---

### 2. `extract_wavelengths_from_raster()`

* Reads wavelengths directly from raster metadata
* Works with Sentinel-2, PRISMA, and hyperspectral data
* Fallback: interpolation if metadata missing

---

### 3. `extract_spectral_features()`

Dynamic feature builder:

* NDVI, NDWI, NDII, EVI, NBR, RECl
* Mean, variance, spectral slope
* Raw band intensities

✔ Automatically adapts to band count

---

### 4. `build_training_labels()`

* Uses percentile-based thresholds
* Derived from SAM similarity distribution
* No hardcoded values

---

### 5. ML Pipeline (Upgraded 🚀)

#### Step 1 — SAM (Spectral Matching)

* Measures similarity between pixel and reference spectrum

#### Step 2 — ML Ensemble

| Model                     | Role                        |
| ------------------------- | --------------------------- |
| Random Forest (300 trees) | Nonlinear learning          |
| SVM (RBF kernel)          | High-dimensional separation |

**Final Prediction:**

```
Final Probability = (RF + SVM) / 2
```

---

### 6. `evaluate_model()`

Outputs:

* Accuracy
* Precision / Recall / F1
* Full classification report

---

### 7. `run_pipeline_custom(target)`

🔥 Fully configurable system:

```python
run_pipeline_custom("vegetation_stress")
run_pipeline_custom("algal_bloom")
run_pipeline_custom("soil_moisture")
```

---

## 🌍 How It Works

```
User selects location
        │
        ▼
Backend (Flask)
   │
   ├── Vegetation Engine (Satellite + ML)
   └── Human Risk Engine (Weather + AQI)
        │
        ▼
Risk Scores + Recommendations
        │
        ▼
Interactive Dashboard
```

---

## 🏗️ Project Architecture

```
Pathowatch-repo/
│
├── server.py
├── pathowatch_pipeline.py
├── pathowatch.py
├── index.html
│
├── Dockerfile
├── requirements.txt
├── Procfile
│
├── Browser_images/
│   ├── B02.tiff
│   ├── B03.tiff
│   ├── B04.tiff
│   └── B08.tiff
│
└── .env
```

---

## 🛠️ Tech Stack

| Layer      | Technology          |
| ---------- | ------------------- |
| Backend    | Flask, Python       |
| ML         | Scikit-learn        |
| Satellite  | Google Earth Engine |
| Processing | rasterio, numpy     |
| Frontend   | HTML, JS, Leaflet   |
| APIs       | OpenWeather, WAQI   |
| Deployment | Docker, Render      |

---

## 📊 Features

* 🌍 Interactive map
* 📡 Real-time satellite analysis
* 📈 Weekly trends
* 🧠 ML-powered detection
* 🌫️ AQI + weather integration
* ⚠️ Risk alerts + recommendations
* 🌙 Dark mode UI

---

## 📡 API Routes

| Route                     | Description             |
| ------------------------- | ----------------------- |
| /analyze_dynamic_location | Satellite NDVI analysis |
| /human_risk               | Disease risk scoring    |
| /run_model                | Full ML pipeline        |
| /risk_map                 | Heatmap output          |
| /risk_stats               | Statistics              |

---

## ⚙️ Environment Variables

```env
OPENWEATHER_KEY=your_key
WAQI_TOKEN=your_token
GEE_SERVICE_ACCOUNT=your_account
GEE_KEY_JSON={...}
```

---

## 🚀 Running Locally

### Python

```bash
pip install -r requirements.txt
python server.py
```

---

### Docker

```bash
docker build -t pathowatch .
docker run --env-file .env -p 8080:8080 pathowatch
```

---

## ☁️ Deployment (Render)

* Runtime: Docker
* Region: Singapore
* Add environment variables
* Deploy

---

## 🧠 Why This Upgrade Matters

### Old System

* Not scalable
* Not robust
* Sensor-dependent

---

### New System

* ✅ Sensor-agnostic
* ✅ Research-grade
* ✅ Data-driven
* ✅ Extensible
* ✅ ML-enhanced

---

## 📊 Old vs New

| Feature     | Old       | New          |
| ----------- | --------- | ------------ |
| Spectra     | Hardcoded | USGS         |
| Wavelengths | Fixed     | Dynamic      |
| Thresholds  | Static    | Adaptive     |
| Features    | Limited   | Dynamic      |
| Detection   | SAM       | SAM + ML     |
| Targets     | Fixed     | Configurable |

---

## 🚀 Future Scope

* Hyperspectral drone integration
* Deep learning models
* Global-scale disease mapping
* Mobile deployment
* Real-time streaming

---

## 🧪 Research-Grade Platform

PathoWatch is now:

> 🎓 Scalable • Reproducible • Sensor-agnostic • Scientifically grounded

---

## 🙌 Team

Built for **PathoWatch**

---

## 📄 License

MIT License
