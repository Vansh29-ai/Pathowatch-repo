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

1. Satellite-based vegetation pathogen detection
2. Dynamic spectral intelligence using real scientific datasets
3. Human disease risk scoring using live environmental data
4. Actionable recommendations

---

# 🧠 PathoWatch – Complete Workflow (Simplified + Logical Flow)

---

## 🔷 0. System Initialization (Backend Start)

* Flask server starts
* `PathogenMonitoringSystem` is initialized
* Spectral library + detection models are loaded

👉 This prepares the system to process satellite images

---

# 🌍 MAIN WORKFLOW (5-Phase Pipeline)

---

## 🟢 PHASE 1: Data Infrastructure & Spectral Library

### 🎯 Goal:

Create reference “fingerprints” of diseases

### ⚙️ What happens:

* Fetch real spectral data from **USGS library**
* Generate synthetic pathogen signatures using:

  * Chlorophyll loss
  * Water stress
  * Leaf damage

### 📦 Stored Signatures:

* `wheat_rust`
* `rice_blast`
* `late_blight`
* `bacterial_blight`

👉 Each pathogen has unique wavelength behavior

---

## 🟡 PHASE 2: Preprocessing (Cleaning Satellite Data)

### 🎯 Goal:

Make raw satellite data usable

### ⚙️ Steps:

#### 1. Atmospheric Correction

* Removes haze, dust, sunlight distortion
* Uses **Dark Object Subtraction**

#### 2. Noise Reduction

* Smooth spectral values
* Spatial filtering (Gaussian)

#### 3. Dimensionality Reduction

* Uses **PCA**
* Reduces hundreds of bands → important features

#### 4. Feature Extraction

Calculates indices:

* NDVI (vegetation health)
* NDWI (water stress)
* NDRE (disease stress)

👉 Converts raw image → meaningful scientific data

---

## 🔵 PHASE 3: Detection (Core Intelligence)

### 🎯 Goal:

Detect disease patterns in pixels

### ⚙️ Techniques Used:

### 🧩 A. Spectral Angle Mapper (SAM)

* Compares pixel spectrum with pathogen signature
* Measures similarity (0 → 1)

👉 If similar → possible disease

---

### 🧠 B. 3D CNN (Deep Learning)

Uses:

* Spatial info (image)
* Spectral info (bands)

👉 Learns patterns automatically

---

### 🔁 Hybrid Approach

* SAM → initial detection
* CNN → refinement

👉 Combines physics + AI

---

## 🟠 PHASE 4: Prediction & Monitoring

### 🎯 Goal:

Generate disease risk maps

### ⚙️ Steps:

* Process hyperspectral image
* Create similarity maps (per pathogen)

### 📊 Risk Levels:

* 🔴 High risk (> 0.7)
* 🟡 Medium (0.4 – 0.7)
* 🟢 Low (< 0.4)

### 📦 Stored Outputs:

* Risk maps (PNG)
* Statistics
* Trends

👉 Output = visual + numerical insights

---

## 🔴 PHASE 5: Validation & Quality Control

### 🎯 Goal:

Ensure reliability

### ⚙️ Metrics:

* Spatial coherence
* Signal-to-noise ratio (SNR)
* Statistical consistency

### 🧠 Output:

* Quality score
* Recommendations:

  * Improve data
  * Increase resolution
  * Reduce noise

👉 Prevents false detections

---

# 🔄 ADDITIONAL PIPELINE (Advanced ML Version)

### ⚙️ Enhancements:

* SAM → generates training labels
* Feature extraction + PCA
* Train models:

  * Random Forest
  * SVM
* Ensemble prediction

👉 More accurate + adaptive system

---

# 🧬 Major Upgrade — Spectral Intelligence Pipeline (v2)

## ❌ Removed (Old System)

* Hardcoded spectral dictionaries
* Fixed wavelength arrays
* Static thresholds
* SAM-only detection

---

## ✅ Added (New System)

* 🌐 USGS spectral integration
* 📡 Dynamic wavelength extraction
* 📊 Adaptive thresholding
* 🤖 Hybrid ML (RF + SVM)
* ⚙️ Configurable targets

---

## 🧬 Pipeline Flow

```
Input Raster
   ↓
Extract Wavelengths
   ↓
Fetch USGS Spectrum
   ↓
SAM Matching
   ↓
Dynamic Thresholding
   ↓
Feature Extraction
   ↓
ML Training
   ↓
Ensemble Prediction
   ↓
Risk Map Output
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

## 🚀 Running Locally

### Python

```bash
pip install -r requirements.txt
python server.py
```

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

## 🚀 Future Scope

* Hyperspectral drones
* Deep learning models
* Global disease tracking
* Mobile app

---

## 📄 License

MIT License
