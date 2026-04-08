import rasterio
import numpy as np
import matplotlib
matplotlib.use('Agg')  # MUST be before pyplot import
import matplotlib.pyplot as plt
import ee
import geemap
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ---------------------------
# NOTE: GEE is initialized in server.py before this module is used.
# Do NOT call ee.Initialize() here again — it causes a conflict.
# ---------------------------

# ---------------------------
# Download Sentinel Image via GEE
# ---------------------------
def download_satellite(lat, lon):
    point = ee.Geometry.Point([lon, lat])

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(point)
        .filterDate("2023-01-01", "2023-12-31")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
    )

    image = collection.first()
    bands = image.select(["B2", "B3", "B4", "B8"])

    geemap.ee_export_image(
        bands,
        filename="sentinel.tif",
        scale=10,
        region=point.buffer(5000)
    )

    return "sentinel.tif"

# ---------------------------
# Load Bands
# ---------------------------
def load_bands(filename):
    dataset = rasterio.open(filename)
    b2 = dataset.read(1)
    b3 = dataset.read(2)
    b4 = dataset.read(3)
    b8 = dataset.read(4)
    return dataset, b2, b3, b4, b8

# ---------------------------
# Feature Extraction
# ---------------------------
def extract_features(b2, b3, b4, b8):
    ndvi = (b8 - b4) / (b8 + b4 + 1e-10)
    ndwi = (b3 - b8) / (b3 + b8 + 1e-10)
    features = np.stack([b2, b3, b4, b8, ndvi, ndwi], axis=-1)
    return features, ndvi

# ---------------------------
# Train Model
# ---------------------------
def train_model(features, ndvi):
    psi    = (1 - ndvi)
    X      = features.reshape(-1, 6)
    labels = (psi > np.percentile(psi, 75)).astype(int)
    y      = labels.flatten()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

# ---------------------------
# Generate Risk Map
# ---------------------------
def generate_heatmap(model, features, b2):
    X             = features.reshape(-1, 6)
    probabilities = model.predict_proba(X)[:, 1]
    heatmap       = probabilities.reshape(b2.shape)

    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap, cmap="RdYlGn_r", vmin=0, vmax=1)
    cbar = plt.colorbar()
    cbar.set_label("Pathogen Risk Probability")
    plt.title("Pathogen Risk Map")
    plt.axis("off")
    plt.savefig("risk_map.png", dpi=300, bbox_inches="tight")
    plt.close()

    return heatmap

# ---------------------------
# Run Pipeline
# Called by /run_model route in server.py
# GEE is already initialized by server.py — no ee.Initialize() here
# ---------------------------
def run_pipeline(lat, lon):
    filename              = download_satellite(lat, lon)
    dataset, b2, b3, b4, b8 = load_bands(filename)
    features, ndvi        = extract_features(b2, b3, b4, b8)
    model                 = train_model(features, ndvi)
    heatmap               = generate_heatmap(model, features, b2)
    return model, heatmap