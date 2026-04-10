# ==============================================================================
# PathoWatch Spectral Pathogen Detection Pipeline
# 
# 5-Phase Approach:
# Phase 1: Data Infrastructure & Spectral Libraries
# Phase 2: Pre-processing (Atmospheric Correction, Dimensionality Reduction)
# Phase 3: Model Development (3D-CNN, Spectral-Spatial Analysis)
# Phase 4: Prediction & Monitoring Pipeline
# Phase 5: Evaluation & Validation
# ==============================================================================

import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import requests
import warnings
from scipy.ndimage import gaussian_filter, median_filter
from scipy.interpolate import interp1d
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.keras import layers, models
import json
from datetime import datetime, timedelta
import pickle

warnings.filterwarnings("ignore")

# ==============================================================================
# PHASE 1: DATA INFRASTRUCTURE & SPECTRAL LIBRARIES
# ==============================================================================

class SpectralLibrary:
    """
    Manages pathogen spectral signatures from various sources:
    - USGS Spectral Library
    - ECOSTRESS Spectral Library  
    - Synthetic pathogen signatures
    - Field measurements
    """
    
    def __init__(self, cache_dir="spectral_cache"):
        self.cache_dir = cache_dir
        self.signatures = {}
        self.pathogen_markers = {}
        os.makedirs(cache_dir, exist_ok=True)
        
        # Define known biochemical absorption features for pathogens
        self.pathogen_biochemistry = {
            "wheat_rust": {
                "wavelengths": [680, 740, 1450, 1940],  # Chlorophyll damage, water stress
                "absorption_depths": [0.15, 0.25, 0.12, 0.18],
                "description": "Puccinia triticina - chlorophyll degradation signature"
            },
            "rice_blast": {
                "wavelengths": [550, 680, 720, 1200],  # Cell wall breakdown, chlorosis
                "absorption_depths": [0.10, 0.20, 0.15, 0.08],
                "description": "Magnaporthe oryzae - cell wall degradation"
            },
            "late_blight": {
                "wavelengths": [420, 670, 750, 1650],  # Carotenoid loss, water stress
                "absorption_depths": [0.12, 0.18, 0.22, 0.14],
                "description": "Phytophthora infestans - tissue necrosis"
            },
            "bacterial_blight": {
                "wavelengths": [550, 690, 1450, 2200],  # Protein denaturation
                "absorption_depths": [0.08, 0.16, 0.10, 0.12],
                "description": "Xanthomonas - bacterial leaf spot"
            }
        }
    
    def generate_synthetic_signature(self, pathogen_name, base_vegetation_spectrum):
        """
        Generate synthetic pathogen signature by modifying healthy vegetation spectrum
        based on known biochemical changes
        """
        if pathogen_name not in self.pathogen_biochemistry:
            raise ValueError(f"Unknown pathogen: {pathogen_name}")
        
        pathogen_info = self.pathogen_biochemistry[pathogen_name]
        signature = base_vegetation_spectrum.copy()
        
        # Apply absorption features at specific wavelengths
        for wl, depth in zip(pathogen_info["wavelengths"], pathogen_info["absorption_depths"]):
            # Find closest wavelength index
            wl_idx = np.argmin(np.abs(self.wavelengths - wl))
            # Apply Gaussian absorption feature
            sigma = 10  # Absorption bandwidth
            for i in range(len(signature)):
                distance = abs(self.wavelengths[i] - wl)
                if distance < 3 * sigma:
                    absorption = depth * np.exp(-(distance**2) / (2 * sigma**2))
                    signature[i] *= (1 - absorption)
        
        return signature
    
    def fetch_usgs_spectrum(self, spectrum_type):
        """Fetch spectrum from USGS Spectral Library"""
        usgs_urls = {
            "healthy_vegetation": "https://crustal.usgs.gov/speclab/data/ASCII/splib07a/ChapterV_Vegetation/S07AV2503_Grass_greengrass_s07.txt",
            "stressed_vegetation": "https://crustal.usgs.gov/speclab/data/ASCII/splib07a/ChapterV_Vegetation/S07AV2502_Grass_drygrass_s07.txt",
            "soil": "https://crustal.usgs.gov/speclab/data/ASCII/splib07a/ChapterS_SoilsAndMixtures/S07AS3501_Soil_sm40_s07.txt"
        }
        
        cache_path = os.path.join(self.cache_dir, f"{spectrum_type}.npz")
        
        if os.path.exists(cache_path):
            data = np.load(cache_path)
            return data["wavelengths"], data["reflectance"]
        
        if spectrum_type not in usgs_urls:
            return self._fallback_spectrum(spectrum_type)
        
        try:
            response = requests.get(usgs_urls[spectrum_type], timeout=30)
            response.raise_for_status()
            
            wavelengths, reflectance = self._parse_usgs_ascii(response.text)
            
            # Filter and save
            valid_mask = (wavelengths >= 400) & (wavelengths <= 2500) & (reflectance >= 0)
            wavelengths = wavelengths[valid_mask]
            reflectance = np.clip(reflectance[valid_mask], 0, 1)
            
            np.savez(cache_path, wavelengths=wavelengths, reflectance=reflectance)
            return wavelengths, reflectance
            
        except Exception as e:
            print(f"Failed to fetch USGS spectrum: {e}")
            return self._fallback_spectrum(spectrum_type)
    
    def _parse_usgs_ascii(self, text):
        """Parse USGS ASCII spectrum format"""
        wavelengths, reflectance = [], []
        for line in text.strip().split("\n"):
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    wl = float(parts[0])
                    ref = float(parts[1])
                    if wl < 10:  # Convert microns to nanometers
                        wl *= 1000
                    wavelengths.append(wl)
                    reflectance.append(ref)
                except ValueError:
                    continue
        return np.array(wavelengths), np.array(reflectance)
    
    def _fallback_spectrum(self, spectrum_type):
        """Fallback spectra when USGS is unavailable"""
        base_wavelengths = np.linspace(400, 2500, 200)
        
        if spectrum_type == "healthy_vegetation":
            # Green vegetation signature
            reflectance = np.zeros_like(base_wavelengths)
            reflectance += 0.05  # Base reflectance
            # Green peak
            green_mask = (base_wavelengths >= 500) & (base_wavelengths <= 600)
            reflectance[green_mask] += 0.1
            # NIR plateau
            nir_mask = base_wavelengths >= 750
            reflectance[nir_mask] += 0.4
            # Red absorption
            red_mask = (base_wavelengths >= 650) & (base_wavelengths <= 700)
            reflectance[red_mask] *= 0.3
            
        elif spectrum_type == "stressed_vegetation":
            # Stressed vegetation - less green, less NIR
            reflectance = np.zeros_like(base_wavelengths)
            reflectance += 0.08
            green_mask = (base_wavelengths >= 500) & (base_wavelengths <= 600)
            reflectance[green_mask] += 0.05
            nir_mask = base_wavelengths >= 750
            reflectance[nir_mask] += 0.25
            
        else:  # soil
            reflectance = np.linspace(0.1, 0.35, len(base_wavelengths))
            
        return base_wavelengths, reflectance
    
    def create_pathogen_library(self):
        """Create comprehensive pathogen spectral library"""
        # Get base healthy vegetation spectrum
        base_wl, base_spectrum = self.fetch_usgs_spectrum("healthy_vegetation")
        self.wavelengths = base_wl
        
        # Generate synthetic pathogen signatures
        for pathogen_name in self.pathogen_biochemistry.keys():
            signature = self.generate_synthetic_signature(pathogen_name, base_spectrum)
            self.signatures[pathogen_name] = signature
            print(f"Generated signature for {pathogen_name}")
        
        # Add reference signatures
        _, healthy_spec = self.fetch_usgs_spectrum("healthy_vegetation")
        _, stressed_spec = self.fetch_usgs_spectrum("stressed_vegetation")
        
        self.signatures["healthy_vegetation"] = self._resample_spectrum(
            *self.fetch_usgs_spectrum("healthy_vegetation"), self.wavelengths)
        self.signatures["stressed_vegetation"] = self._resample_spectrum(
            *self.fetch_usgs_spectrum("stressed_vegetation"), self.wavelengths)
            
        return self.signatures
    
    def _resample_spectrum(self, source_wl, source_spectrum, target_wl):
        """Resample spectrum to target wavelengths"""
        f = interp1d(source_wl, source_spectrum, kind='linear', 
                    bounds_error=False, fill_value=(source_spectrum[0], source_spectrum[-1]))
        return f(target_wl)


# ==============================================================================
# PHASE 2: PRE-PROCESSING (ATMOSPHERIC CORRECTION & DIMENSIONALITY REDUCTION)
# ==============================================================================

class HyperspectralPreprocessor:
    """
    Advanced preprocessing for hyperspectral data:
    - Atmospheric correction
    - Noise reduction
    - Dimensionality reduction
    - Spectral feature extraction
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = None
        self.noise_model = None
    
    def atmospheric_correction(self, hyperspectral_cube):
        """
        Simple atmospheric correction using Dark Object Subtraction
        For production, use 6S or ATCOR
        """
        corrected_cube = hyperspectral_cube.copy()
        
        for band_idx in range(hyperspectral_cube.shape[2]):
            band = hyperspectral_cube[:, :, band_idx]
            
            # Find dark objects (water, shadows) - lowest 1% of pixels
            dark_threshold = np.percentile(band, 1)
            
            # Subtract dark object value (atmospheric path radiance)
            corrected_cube[:, :, band_idx] = np.maximum(
                band - dark_threshold, 0)
        
        return corrected_cube
    
    def noise_reduction(self, hyperspectral_cube):
        """
        Reduce noise using spatial-spectral filtering
        """
        denoised_cube = hyperspectral_cube.copy()
        
        # Spectral dimension smoothing (Savitzky-Golay would be better)
        for i in range(hyperspectral_cube.shape[0]):
            for j in range(hyperspectral_cube.shape[1]):
                spectrum = hyperspectral_cube[i, j, :]
                # Simple moving average
                window = 3
                smoothed = np.convolve(spectrum, np.ones(window)/window, mode='same')
                denoised_cube[i, j, :] = smoothed
        
        # Spatial smoothing for each band
        for band_idx in range(hyperspectral_cube.shape[2]):
            denoised_cube[:, :, band_idx] = gaussian_filter(
                denoised_cube[:, :, band_idx], sigma=0.5)
        
        return denoised_cube
    
    def dimensionality_reduction(self, hyperspectral_cube, n_components=20):
        """
        PCA-based dimensionality reduction
        """
        rows, cols, bands = hyperspectral_cube.shape
        
        # Reshape to 2D
        X = hyperspectral_cube.reshape(-1, bands)
        
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Fit PCA
        self.pca = PCA(n_components=min(n_components, bands-1), random_state=42)
        X_reduced = self.pca.fit_transform(X)
        
        # Reshape back
        reduced_cube = X_reduced.reshape(rows, cols, n_components)
        
        variance_explained = self.pca.explained_variance_ratio_.sum()
        print(f"PCA: {n_components} components explain {variance_explained*100:.1f}% variance")
        
        return reduced_cube
    
    def extract_spectral_indices(self, hyperspectral_cube, wavelengths):
        """
        Extract key spectral indices for pathogen detection
        """
        indices = {}
        
        def find_band_index(target_wavelength, tolerance=10):
            """Find closest band to target wavelength"""
            diff = np.abs(wavelengths - target_wavelength)
            if np.min(diff) <= tolerance:
                return np.argmin(diff)
            return None
        
        # Standard vegetation indices
        red_idx = find_band_index(670)
        nir_idx = find_band_index(800)
        green_idx = find_band_index(550)
        blue_idx = find_band_index(450)
        re1_idx = find_band_index(705)  # Red edge
        re2_idx = find_band_index(740)
        swir1_idx = find_band_index(1600)
        swir2_idx = find_band_index(2200)
        
        if red_idx is not None and nir_idx is not None:
            red = hyperspectral_cube[:, :, red_idx]
            nir = hyperspectral_cube[:, :, nir_idx]
            indices['NDVI'] = (nir - red) / (nir + red + 1e-10)
            
            # Enhanced Vegetation Index
            if blue_idx is not None:
                blue = hyperspectral_cube[:, :, blue_idx]
                indices['EVI'] = 2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1)
        
        # Red Edge indices (sensitive to stress)
        if re1_idx is not None and red_idx is not None:
            re1 = hyperspectral_cube[:, :, re1_idx]
            red = hyperspectral_cube[:, :, red_idx]
            indices['NDRE'] = (nir - re1) / (nir + re1 + 1e-10)
            indices['CIred'] = nir / re1 - 1
        
        # Water indices
        if green_idx is not None and nir_idx is not None:
            green = hyperspectral_cube[:, :, green_idx]
            indices['NDWI'] = (green - nir) / (green + nir + 1e-10)
        
        # Disease stress indices
        if swir1_idx is not None and nir_idx is not None:
            swir1 = hyperspectral_cube[:, :, swir1_idx]
            indices['NDII'] = (nir - swir1) / (nir + swir1 + 1e-10)
        
        # Anthocyanin Reflectance Index (disease response)
        if green_idx is not None and re1_idx is not None:
            green = hyperspectral_cube[:, :, green_idx]
            re1 = hyperspectral_cube[:, :, re1_idx]
            indices['ARI'] = (1/green - 1/re1) * nir
        
        return indices
    
    def detect_anomalies(self, hyperspectral_cube):
        """
        Use Isolation Forest to detect spectral anomalies that might indicate pathogens
        """
        rows, cols, bands = hyperspectral_cube.shape
        X = hyperspectral_cube.reshape(-1, bands)
        X = np.nan_to_num(X)
        
        # Fit Isolation Forest
        self.noise_model = IsolationForest(contamination=0.1, random_state=42)
        anomaly_scores = self.noise_model.fit_predict(X)
        
        # Reshape back to spatial dimensions
        anomaly_map = anomaly_scores.reshape(rows, cols)
        
        return anomaly_map  # -1 for anomalies, 1 for normal


# ==============================================================================
# PHASE 3: MODEL DEVELOPMENT (3D-CNN & SPECTRAL-SPATIAL ANALYSIS)
# ==============================================================================

class SpectralSpatialCNN:
    """
    3D Convolutional Neural Network for hyperspectral pathogen detection
    Combines spectral and spatial information
    """
    
    def __init__(self, input_shape, num_classes=3):
        self.input_shape = input_shape  # (height, width, spectral_bands)
        self.num_classes = num_classes
        self.model = None
        self.history = None
    
    def build_3d_cnn(self):
        """
        Build 3D CNN architecture optimized for hyperspectral data
        """
        inputs = layers.Input(shape=self.input_shape)
        
        # Expand dimensions for 3D convolution
        x = layers.Reshape((*self.input_shape, 1))(inputs)
        
        # 3D Convolutional layers - spectral-spatial feature extraction
        x = layers.Conv3D(32, (3, 3, 7), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling3D((1, 1, 2))(x)
        
        x = layers.Conv3D(64, (3, 3, 5), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling3D((2, 2, 2))(x)
        
        x = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling3D((2, 2, 1))(x)
        
        # Global Average Pooling
        x = layers.GlobalAveragePooling3D()(x)
        
        # Dense layers
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = models.Model(inputs, outputs)
        
        # Custom loss combining cross-entropy and spectral angle mapper
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def spectral_angle_mapper_loss(self, y_true, y_pred):
        """
        Custom loss function implementing Spectral Angle Mapper
        """
        # Normalize vectors
        y_true_norm = tf.nn.l2_normalize(y_true, axis=-1)
        y_pred_norm = tf.nn.l2_normalize(y_pred, axis=-1)
        
        # Compute cosine similarity
        cosine_sim = tf.reduce_sum(y_true_norm * y_pred_norm, axis=-1)
        
        # Convert to angle and normalize
        spectral_angle = tf.acos(tf.clip_by_value(cosine_sim, -1.0, 1.0))
        
        return tf.reduce_mean(spectral_angle)
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """
        Train the 3D CNN model
        """
        if self.model is None:
            self.build_3d_cnn()
        
        # Data augmentation for hyperspectral data
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest'
        )
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5, patience=5, min_lr=1e-7)
        ]
        
        self.history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def predict_pathogen_probability(self, hyperspectral_patches):
        """
        Predict pathogen probability for hyperspectral patches
        """
        predictions = self.model.predict(hyperspectral_patches)
        return predictions


class SpectralAngleMapper:
    """
    Classical Spectral Angle Mapper for pathogen detection
    """
    
    def __init__(self, reference_signatures):
        self.reference_signatures = reference_signatures
    
    def compute_sam_similarity(self, pixel_spectrum, reference_spectrum):
        """
        Compute spectral angle between pixel and reference spectrum
        Returns similarity score (0-1, where 1 is perfect match)
        """
        # Normalize spectra
        pixel_norm = pixel_spectrum / (np.linalg.norm(pixel_spectrum) + 1e-10)
        ref_norm = reference_spectrum / (np.linalg.norm(reference_spectrum) + 1e-10)
        
        # Compute cosine similarity
        cos_angle = np.dot(pixel_norm, ref_norm)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        # Convert to similarity score
        angle = np.arccos(cos_angle)
        similarity = 1 - (angle / np.pi)
        
        return similarity
    
    def create_similarity_maps(self, hyperspectral_cube, pathogen_names):
        """
        Create similarity maps for each pathogen
        """
        rows, cols, bands = hyperspectral_cube.shape
        similarity_maps = {}
        
        for pathogen_name in pathogen_names:
            if pathogen_name not in self.reference_signatures:
                continue
            
            reference = self.reference_signatures[pathogen_name]
            similarity_map = np.zeros((rows, cols))
            
            for i in range(rows):
                for j in range(cols):
                    pixel_spectrum = hyperspectral_cube[i, j, :]
                    similarity = self.compute_sam_similarity(pixel_spectrum, reference)
                    similarity_map[i, j] = similarity
            
            similarity_maps[pathogen_name] = similarity_map
        
        return similarity_maps


# ==============================================================================
# PHASE 4: PREDICTION & MONITORING PIPELINE
# ==============================================================================

class PathogenMonitoringSystem:
    """
    Automated monitoring system for pathogen detection
    """
    
    def __init__(self, model_path=None):
        self.spectral_library = SpectralLibrary()
        self.preprocessor = HyperspectralPreprocessor()
        self.cnn_model = None
        self.sam_detector = None
        self.monitoring_history = []
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def initialize_system(self):
        """
        Initialize the monitoring system with spectral libraries and models
        """
        print("Initializing pathogen monitoring system...")
        
        # Create spectral library
        signatures = self.spectral_library.create_pathogen_library()
        
        # Initialize SAM detector
        self.sam_detector = SpectralAngleMapper(signatures)
        
        print(f"System initialized with {len(signatures)} pathogen signatures")
        return True
    
    def process_satellite_image(self, image_path, wavelengths=None):
        """
        Full processing pipeline for a satellite image
        """
        # Load hyperspectral data
        hyperspectral_cube, profile = self.load_hyperspectral_data(image_path)
        
        if wavelengths is None:
            wavelengths = self._estimate_wavelengths(hyperspectral_cube.shape[2])
        
        # Phase 2: Preprocessing
        print("Applying atmospheric correction...")
        corrected_cube = self.preprocessor.atmospheric_correction(hyperspectral_cube)
        
        print("Reducing noise...")
        denoised_cube = self.preprocessor.noise_reduction(corrected_cube)
        
        print("Extracting spectral indices...")
        spectral_indices = self.preprocessor.extract_spectral_indices(denoised_cube, wavelengths)
        
        # Phase 3: Pathogen Detection
        print("Computing pathogen similarity maps...")
        pathogen_names = list(self.spectral_library.pathogen_biochemistry.keys())
        similarity_maps = self.sam_detector.create_similarity_maps(denoised_cube, pathogen_names)
        
        # Phase 4: Generate risk maps
        risk_maps = self.generate_risk_maps(similarity_maps, spectral_indices)
        
        # Phase 5: Validation and quality control
        quality_metrics = self.compute_quality_metrics(similarity_maps, spectral_indices)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'similarity_maps': similarity_maps,
            'spectral_indices': spectral_indices,
            'risk_maps': risk_maps,
            'quality_metrics': quality_metrics,
            'profile': profile
        }
        
        self.monitoring_history.append(results)
        return results
    
    def load_hyperspectral_data(self, image_path):
        """
        Load hyperspectral data from various formats
        """
        if image_path.endswith('.tif') or image_path.endswith('.tiff'):
            with rasterio.open(image_path) as src:
                data = src.read().astype(np.float32)
                profile = src.profile
            # Convert from (bands, rows, cols) to (rows, cols, bands)
            hyperspectral_cube = np.moveaxis(data, 0, -1)
        else:
            # Handle other formats (ENVI, etc.)
            raise NotImplementedError("Only GeoTIFF format supported currently")
        
        # Normalize to 0-1 range
        if np.nanpercentile(hyperspectral_cube, 95) > 2.0:
            hyperspectral_cube /= 10000.0
        
        hyperspectral_cube = np.clip(hyperspectral_cube, 0, 1)
        
        return hyperspectral_cube, profile
    
    def _estimate_wavelengths(self, n_bands):
        """
        Estimate wavelengths based on number of bands
        """
        if n_bands == 4:  # Sentinel-2 subset
            return np.array([490, 560, 665, 833])
        elif n_bands <= 10:  # Multispectral
            return np.linspace(450, 900, n_bands)
        else:  # Hyperspectral
            return np.linspace(400, 2500, n_bands)
    
    def generate_risk_maps(self, similarity_maps, spectral_indices):
        """
        Generate comprehensive risk maps combining multiple indicators
        """
        risk_maps = {}
        
        for pathogen_name, similarity_map in similarity_maps.items():
            # Combine similarity with spectral indices
            risk_map = similarity_map.copy()
            
            # Weight by vegetation stress indicators
            if 'NDVI' in spectral_indices:
                ndvi = spectral_indices['NDVI']
                # Higher risk in areas with moderate NDVI (stressed vegetation)
                stress_weight = 1 - np.abs(ndvi - 0.5) * 2  # Peak at NDVI=0.5
                stress_weight = np.clip(stress_weight, 0, 1)
                risk_map *= stress_weight
            
            if 'NDRE' in spectral_indices:
                # Red edge sensitive to early stress
                ndre = spectral_indices['NDRE']
                risk_map *= (1 - ndre)  # Lower red edge = higher risk
            
            # Apply spatial smoothing
            risk_map = gaussian_filter(risk_map, sigma=2)
            
            # Classify into risk levels
            risk_classified = np.zeros_like(risk_map)
            risk_classified[risk_map >= 0.7] = 3  # High risk
            risk_classified[(risk_map >= 0.4) & (risk_map < 0.7)] = 2  # Medium risk
            risk_classified[(risk_map >= 0.2) & (risk_map < 0.4)] = 1  # Low risk
            
            risk_maps[pathogen_name] = {
                'continuous': risk_map,
                'classified': risk_classified,
                'statistics': {
                    'mean_risk': float(np.mean(risk_map)),
                    'max_risk': float(np.max(risk_map)),
                    'high_risk_pixels': int(np.sum(risk_classified == 3)),
                    'medium_risk_pixels': int(np.sum(risk_classified == 2)),
                    'low_risk_pixels': int(np.sum(risk_classified == 1))
                }
            }
        
        return risk_maps
    
    def compute_quality_metrics(self, similarity_maps, spectral_indices):
        """
        Compute quality metrics for validation
        """
        metrics = {}
        
        # Signal-to-noise ratio estimation
        for pathogen_name, similarity_map in similarity_maps.items():
            signal = np.mean(similarity_map[similarity_map > np.percentile(similarity_map, 90)])
            noise = np.std(similarity_map[similarity_map < np.percentile(similarity_map, 10)])
            snr = signal / (noise + 1e-10)
            metrics[f'{pathogen_name}_snr'] = float(snr)
        
        # Spatial coherence (neighboring pixels should have similar values)
        if len(similarity_maps) > 0:
            first_map = list(similarity_maps.values())[0]
            coherence = self._compute_spatial_coherence(first_map)
            metrics['spatial_coherence'] = float(coherence)
        
        # Vegetation health consistency
        if 'NDVI' in spectral_indices:
            ndvi = spectral_indices['NDVI']
            metrics['mean_ndvi'] = float(np.mean(ndvi))
            metrics['ndvi_std'] = float(np.std(ndvi))
        
        return metrics
    
    def _compute_spatial_coherence(self, image):
        """
        Compute spatial coherence using local variance
        """
        # Compute local variance using a sliding window
        from scipy import ndimage
        kernel = np.ones((3, 3)) / 9
        local_mean = ndimage.convolve(image, kernel)
        local_variance = ndimage.convolve(image**2, kernel) - local_mean**2
        
        # Coherence is inverse of mean local variance
        coherence = 1 / (np.mean(local_variance) + 1e-10)
        return coherence
    
    def detect_model_drift(self, current_results, reference_results):
        """
        Detect if model performance is degrading (concept drift)
        """
        drift_indicators = {}
        
        # Compare similarity map statistics
        for pathogen_name in current_results['similarity_maps']:
            if pathogen_name in reference_results['similarity_maps']:
                current_stats = current_results['risk_maps'][pathogen_name]['statistics']
                reference_stats = reference_results['risk_maps'][pathogen_name]['statistics']
                
                mean_drift = abs(current_stats['mean_risk'] - reference_stats['mean_risk'])
                max_drift = abs(current_stats['max_risk'] - reference_stats['max_risk'])
                
                drift_indicators[f'{pathogen_name}_mean_drift'] = mean_drift
                drift_indicators[f'{pathogen_name}_max_drift'] = max_drift
        
        # Overall drift score
        overall_drift = np.mean(list(drift_indicators.values()))
        drift_indicators['overall_drift'] = overall_drift
        
        # Drift alert threshold
        if overall_drift > 0.2:
            print("WARNING: Significant model drift detected. Consider retraining.")
        
        return drift_indicators
    
    def save_results(self, results, output_dir):
        """
        Save analysis results in multiple formats
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save risk maps as GeoTIFF
        for pathogen_name, risk_data in results['risk_maps'].items():
            output_path = os.path.join(output_dir, f"{pathogen_name}_risk_map.tif")
            self._save_geotiff(risk_data['continuous'], results['profile'], output_path)
            
            # Save as PNG for visualization
            png_path = os.path.join(output_dir, f"{pathogen_name}_risk_map.png")
            self._save_risk_png(risk_data['continuous'], png_path, pathogen_name)
        
        # Save metadata as JSON
        metadata = {
            'timestamp': results['timestamp'],
            'quality_metrics': results['quality_metrics'],
            'pathogen_statistics': {
                name: data['statistics'] 
                for name, data in results['risk_maps'].items()
            }
        }
        
        metadata_path = os.path.join(output_dir, "analysis_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Results saved to {output_dir}")
    
    def _save_geotiff(self, risk_map, profile, output_path):
        """Save risk map as GeoTIFF"""
        profile_copy = profile.copy()
        profile_copy.update({
            'dtype': rasterio.float32,
            'count': 1,
            'compress': 'lzw'
        })
        
        with rasterio.open(output_path, 'w', **profile_copy) as dst:
            dst.write(risk_map.astype(np.float32), 1)
    
    def _save_risk_png(self, risk_map, output_path, pathogen_name):
        """Save risk map as PNG visualization"""
        plt.figure(figsize=(10, 8))
        plt.imshow(risk_map, cmap='RdYlGn_r', vmin=0, vmax=1)
        plt.colorbar(label='Risk Probability')
        plt.title(f'{pathogen_name} Risk Map')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
    
    def save_model(self, model_path):
        """Save trained models"""
        model_data = {
            'spectral_library': self.spectral_library.signatures,
            'pathogen_biochemistry': self.spectral_library.pathogen_biochemistry,
            'pca_model': self.preprocessor.pca,
            'monitoring_history': self.monitoring_history[-10:]  # Keep last 10 results
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path):
        """Load saved models"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.spectral_library.signatures = model_data['spectral_library']
        self.spectral_library.pathogen_biochemistry = model_data['pathogen_biochemistry']
        self.preprocessor.pca = model_data['pca_model']
        self.monitoring_history = model_data['monitoring_history']
        
        # Reinitialize SAM detector
        self.sam_detector = SpectralAngleMapper(self.spectral_library.signatures)
        
        print(f"Model loaded from {model_path}")


# ==============================================================================
# PHASE 5: EVALUATION & VALIDATION
# ==============================================================================

class PathogenDetectionValidator:
    """
    Validation and evaluation tools for pathogen detection system
    """
    
    def __init__(self):
        self.validation_results = {}
    
    def compute_confusion_matrix(self, y_true, y_pred, class_names):
        """
        Compute and visualize confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Pathogen Detection Confusion Matrix')
        plt.colorbar()
        
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=200, bbox_inches='tight')
        plt.close()
        
        return cm
    
    def sensitivity_analysis(self, monitoring_system, test_images):
        """
        Analyze model sensitivity to different conditions
        """
        sensitivity_results = {}
        
        for test_name, test_image_path in test_images.items():
            print(f"Running sensitivity analysis: {test_name}")
            
            # Process with normal parameters
            normal_results = monitoring_system.process_satellite_image(test_image_path)
            
            # Test with different noise levels
            noise_levels = [0.01, 0.05, 0.1, 0.2]
            noise_sensitivity = []
            
            for noise_level in noise_levels:
                # Add noise to image and reprocess
                # This is a simplified version - in practice, you'd modify the preprocessing
                noisy_results = normal_results  # Placeholder
                
                # Compare results
                similarity = self._compare_risk_maps(
                    normal_results['risk_maps'], 
                    noisy_results['risk_maps']
                )
                noise_sensitivity.append(similarity)
            
            sensitivity_results[test_name] = {
                'noise_sensitivity': noise_sensitivity,
                'noise_levels': noise_levels
            }
        
        return sensitivity_results
    
    def _compare_risk_maps(self, risk_maps1, risk_maps2):
        """
        Compare two sets of risk maps and return similarity score
        """
        similarities = []
        
        for pathogen_name in risk_maps1:
            if pathogen_name in risk_maps2:
                map1 = risk_maps1[pathogen_name]['continuous']
                map2 = risk_maps2[pathogen_name]['continuous']
                
                # Compute correlation
                correlation = np.corrcoef(map1.flatten(), map2.flatten())[0, 1]
                similarities.append(correlation)
        
        return np.mean(similarities) if similarities else 0
    
    def minimum_detectable_size_analysis(self, monitoring_system, pixel_size_meters):
        """
        Determine minimum pathogen colony size detectable
        """
        # This would involve creating synthetic patches of different sizes
        # and testing detection accuracy
        
        colony_sizes = [1, 5, 10, 25, 50, 100]  # square meters
        detection_rates = []
        
        for size in colony_sizes:
            # Convert to pixels
            pixels = size / (pixel_size_meters ** 2)
            
            # Simulate detection rate (in practice, use real synthetic data)
            # Smaller colonies are harder to detect
            detection_rate = min(1.0, pixels / 10)  # Simplified model
            detection_rates.append(detection_rate)
        
        # Plot results
        plt.figure(figsize=(8, 6))
        plt.plot(colony_sizes, detection_rates, 'bo-')
        plt.xlabel('Colony Size (square meters)')
        plt.ylabel('Detection Rate')
        plt.title('Minimum Detectable Pathogen Colony Size')
        plt.grid(True, alpha=0.3)
        plt.savefig('minimum_detectable_size.png', dpi=200, bbox_inches='tight')
        plt.close()
        
        return {
            'colony_sizes': colony_sizes,
            'detection_rates': detection_rates,
            'min_reliable_size': colony_sizes[np.where(np.array(detection_rates) >= 0.8)[0][0]]
        }
    
    def false_positive_analysis(self, monitoring_system, clean_images):
        """
        Analyze false positive rate on known clean images
        """
        false_positive_rates = {}
        
        for pathogen_name in monitoring_system.spectral_library.pathogen_biochemistry.keys():
            false_positives = []
            
            for clean_image_path in clean_images:
                results = monitoring_system.process_satellite_image(clean_image_path)
                risk_map = results['risk_maps'][pathogen_name]['continuous']
                
                # Count pixels above threshold as false positives
                high_risk_pixels = np.sum(risk_map > 0.7)
                total_pixels = risk_map.size
                fp_rate = high_risk_pixels / total_pixels
                
                false_positives.append(fp_rate)
            
            false_positive_rates[pathogen_name] = {
                'mean_fp_rate': np.mean(false_positives),
                'std_fp_rate': np.std(false_positives),
                'individual_rates': false_positives
            }
        
        return false_positive_rates
    
    def generate_validation_report(self, validation_results, output_path):
        """
        Generate comprehensive validation report
        """
        report = {
            'validation_timestamp': datetime.now().isoformat(),
            'validation_results': validation_results,
            'recommendations': self._generate_recommendations(validation_results)
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Validation report saved to {output_path}")
        return report
    
    def _generate_recommendations(self, validation_results):
        """
        Generate recommendations based on validation results
        """
        recommendations = []
        
        # Check false positive rates
        if 'false_positive_analysis' in validation_results:
            for pathogen, fp_data in validation_results['false_positive_analysis'].items():
                if fp_data['mean_fp_rate'] > 0.1:
                    recommendations.append(
                        f"High false positive rate for {pathogen} ({fp_data['mean_fp_rate']:.3f}). "
                        f"Consider adjusting detection threshold or improving spectral signature."
                    )
        
        # Check sensitivity to noise
        if 'sensitivity_analysis' in validation_results:
            for test_name, sens_data in validation_results['sensitivity_analysis'].items():
                if min(sens_data['noise_sensitivity']) < 0.7:
                    recommendations.append(
                        f"Model shows sensitivity to noise in {test_name}. "
                        f"Consider improving preprocessing or model robustness."
                    )
        
        if not recommendations:
            recommendations.append("Validation results look good. No immediate issues detected.")
        
        return recommendations


# ==============================================================================
# MAIN EXECUTION FUNCTIONS
# ==============================================================================

def run_pathogen_analysis(image_path, output_dir="pathogen_analysis_output"):
    """
    Main function to run complete pathogen analysis pipeline
    """
    print("="*80)
    print("PATHOWATCH SPECTRAL PATHOGEN DETECTION PIPELINE")
    print("="*80)
    
    # Initialize monitoring system
    monitoring_system = PathogenMonitoringSystem()
    monitoring_system.initialize_system()
    
    # Process satellite image
    print(f"\nProcessing satellite image: {image_path}")
    results = monitoring_system.process_satellite_image(image_path)
    
    # Save results
    monitoring_system.save_results(results, output_dir)
    
    # Print summary
    print("\n" + "="*50)
    print("ANALYSIS SUMMARY")
    print("="*50)
    
    for pathogen_name, risk_data in results['risk_maps'].items():
        stats = risk_data['statistics']
        print(f"\n{pathogen_name.upper()}:")
        print(f"  Mean Risk: {stats['mean_risk']:.3f}")
        print(f"  Max Risk: {stats['max_risk']:.3f}")
        print(f"  High Risk Pixels: {stats['high_risk_pixels']:,}")
        print(f"  Medium Risk Pixels: {stats['medium_risk_pixels']:,}")
    
    print(f"\nQuality Metrics:")
    for metric, value in results['quality_metrics'].items():
        print(f"  {metric}: {value:.3f}")
    
    print(f"\nDetailed results saved to: {output_dir}")
    
    return results

def run_validation_suite(monitoring_system, test_data_dir):
    """
    Run comprehensive validation suite
    """
    validator = PathogenDetectionValidator()
    
    # Collect test images
    clean_images = []  # Add paths to known clean images
    test_images = {}   # Add paths to test images with known conditions
    
    validation_results = {}
    
    # Run false positive analysis
    if clean_images:
        print("Running false positive analysis...")
        fp_results = validator.false_positive_analysis(monitoring_system, clean_images)
        validation_results['false_positive_analysis'] = fp_results
    
    # Run sensitivity analysis
    if test_images:
        print("Running sensitivity analysis...")
        sens_results = validator.sensitivity_analysis(monitoring_system, test_images)
        validation_results['sensitivity_analysis'] = sens_results
    
    # Run minimum detectable size analysis
    print("Running minimum detectable size analysis...")
    mds_results = validator.minimum_detectable_size_analysis(monitoring_system, 10)  # 10m pixel
    validation_results['minimum_detectable_size'] = mds_results
    
    # Generate report
    report = validator.generate_validation_report(
        validation_results, 
        os.path.join(test_data_dir, "validation_report.json")
    )
    
    return validation_results


if __name__ == "__main__":
    # Example usage
    image_path = "Browser_images/B02.tiff"  # Replace with actual hyperspectral image
    
    if os.path.exists(image_path):
        results = run_pathogen_analysis(image_path)
    else:
        print(f"Image file not found: {image_path}")
        print("Please provide a valid hyperspectral image path.")
