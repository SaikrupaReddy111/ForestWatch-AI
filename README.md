# 🌳 ForestWatch AI
### Global Satellite-Based Deforestation Detection and Forest Monitoring System

ForestWatch AI is an AI-powered forest monitoring system that uses satellite imagery, deep learning, and geospatial analysis to detect forest cover changes and identify potential deforestation.

The system uses **Sentinel-2 satellite imagery, a U-Net deep learning model, and Google Earth Engine** to analyze forest cover across different geographical regions.

It includes an interactive Streamlit dashboard to visualize satellite imagery, forest cover predictions, deforestation estimates, and forest gain.

---

## 🌍 Project Overview

Deforestation is a major environmental challenge that contributes to biodiversity loss, climate change, and ecosystem degradation.

Traditional forest monitoring methods often require extensive manual analysis and may not provide timely information about changes in forest cover.

ForestWatch AI addresses this challenge by combining satellite-based remote sensing with deep learning to automate forest cover classification and change detection.

### Key Objectives

- Detect forest and non-forest regions from satellite imagery.
- Monitor forest cover changes across different years.
- Identify potential deforestation and forest gain areas.
- Support monitoring across different geographical locations.
- Visualize results through an interactive dashboard.

---

## ✨ Key Features

- **Satellite Image Processing:** Processes multispectral Sentinel-2 imagery.
- **AI-Based Forest Classification:** Uses a U-Net model to classify forest and non-forest pixels.
- **Deforestation Detection:** Compares forest masks from different years to identify potential tree-cover loss.
- **Forest Gain Monitoring:** Identifies areas showing potential forest gain.
- **Dynamic Geographic Selection:** Supports location-based analysis using geocoding and ecological boundaries.
- **Google Earth Engine Integration:** Retrieves satellite imagery and Dynamic World labels.
- **Interactive Dashboard:** Displays maps, statistics, charts, and downloadable results.
- **Spatial Analysis:** Uses geospatial processing to maintain raster alignment and analyze geographical regions.

---

## 🏗️ System Architecture

The project follows a satellite-data-to-dashboard processing pipeline.

```text
          Geographic Location
                  |
                  v
       Geocoding and Region Selection
                  |
                  v
         Google Earth Engine
                  |
                  v
       Sentinel-2 Satellite Imagery
                  |
                  v
       Preprocessing and Tiling
                  |
                  v
          U-Net Deep Learning
             Model
                  |
                  v
        Forest / Non-Forest Masks
                  |
                  v
         Multi-Year Comparison
                  |
                  v
       Deforestation and Forest Gain
               Detection
                  |
                  v
        Geospatial Output Generation
                  |
                  v
         Streamlit Dashboard
```

### Processing Workflow

1. **Location Selection:** The user selects a geographical region for monitoring.
2. **Data Acquisition:** Sentinel-2 satellite imagery is retrieved using Google Earth Engine.
3. **Preprocessing:** Images are processed, normalized, and divided into manageable tiles.
4. **Forest Classification:** The trained U-Net model predicts forest and non-forest regions.
5. **Change Detection:** Predictions from two different years are compared.
6. **Visualization:** Generated maps and statistical summaries are displayed through Streamlit.

---

## 🛰️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Segmentation Architecture | U-Net |
| Encoder | ResNet34 |
| Segmentation Library | Segmentation Models PyTorch |
| Satellite Imagery | Sentinel-2 |
| Satellite Data Platform | Google Earth Engine |
| Reference Labels | Dynamic World |
| Geospatial Processing | Rasterio, GeoPandas, Shapely, PyProj |
| Image Processing | NumPy |
| Geographic Search | Geopy, Nominatim |
| Visualization | Streamlit |
| Charts | Plotly |
| Testing | Pytest |
| Version Control | Git and GitHub |

---

## 🤖 Deep Learning Model

ForestWatch AI uses a **U-Net semantic segmentation architecture with a ResNet34 encoder**.

### Why U-Net?

U-Net is designed for pixel-level image segmentation. It is suitable for identifying spatial features such as forest boundaries and vegetation patterns in satellite imagery.

### Model Architecture

- Architecture: U-Net
- Encoder: ResNet34
- Framework: PyTorch
- Input: Six-band Sentinel-2 imagery
- Output: Binary forest/non-forest segmentation mask

### Sentinel-2 Input Bands

The model uses six spectral bands:

| Band | Description |
|---|---|
| B2 | Blue |
| B3 | Green |
| B4 | Red |
| B8 | Near Infrared |
| B11 | Short-Wave Infrared |
| B12 | Short-Wave Infrared |

These bands provide spectral information useful for distinguishing vegetation and other land-cover types.

### Training Data

The model was trained using satellite imagery from the Amazon region of Brazil.

Dynamic World forest/non-forest classifications were used as pseudo-labels for model training.

The dataset was divided using spatially separated training, validation, and testing regions to reduce spatial leakage.

---

## 📊 Results and Performance

### Amazon Brazil – Model Evaluation

The model was evaluated on a held-out test set using Dynamic World pseudo-labels.

| Metric | Result |
|---|---:|
| Validation IoU | 0.9509 |
| Test IoU | 0.9717 |
| Dice Score | 0.9857 |
| Precision | 0.9905 |
| Recall | 0.9808 |

**Important:** These metrics measure agreement with held-out Dynamic World pseudo-labels. They should not be interpreted as independently verified ground-truth accuracy.

### Amazon Brazil – Change Detection (2020–2023)

| Classification | Area |
|---|---:|
| Stable Non-Forest | 343,246.80 ha |
| Stable Forest | 216,460.68 ha |
| Potential Tree-Cover Loss | 16,595.71 ha |
| Potential Forest Gain | 7,920.65 ha |

Valid monitored area: approximately 584,223.84 hectares.

### Western Ghats, India – Pilot Analysis

A separate pilot analysis was conducted using satellite imagery from 2020 and 2023.

| Classification | Area |
|---|---:|
| Stable Non-Forest | 80.41 ha |
| Stable Forest | 1,048.56 ha |
| Potential Tree-Cover Loss | 45.37 ha |
| Potential Forest Gain | 85.53 ha |
| NoData | 348.13 ha |

Valid monitored area: approximately 1,259.87 hectares.

The Western Ghats pilot demonstrates the experimental application of the Amazon-trained model to another geographical region.

**Scientific limitation:** The Western Ghats results are experimental model-based estimates. The model was trained on Amazon imagery using Dynamic World pseudo-labels and has not been independently validated for the Western Ghats. The NoData area is excluded from valid change estimates.

---

## 📸 Dashboard Screenshots

The Streamlit dashboard provides five main sections:

1. Overview
2. Satellite Imagery
3. Change Detection
4. AI Model Insights
5. Methodology

Add your actual dashboard screenshots in the `screenshots/` directory.

Suggested screenshot organization:

```text
screenshots/
├── overview.png
├── satellite-imagery.png
├── change-detection.png
├── model-insights.png
└── methodology.png
```

Example Markdown for displaying screenshots:

```markdown
![ForestWatch AI Dashboard](screenshots/overview.png)

![Change Detection](screenshots/change-detection.png)

![AI Model Insights](screenshots/model-insights.png)
```

---

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.10 or later
- Git
- VS Code or another Python IDE
- Google Earth Engine account (for live satellite data acquisition)

### 1. Clone the Repository

```bash
git clone https://github.com/SaikrupaReddy111/ForestWatch-AI.git
```

### 2. Navigate to the Project

```bash
cd ForestWatch-AI
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Launch the Dashboard

```bash
python -m streamlit run app.py
```

The dashboard will open in your browser.

---

## 🛰️ Google Earth Engine Configuration

To retrieve new satellite imagery, configure Google Earth Engine authentication.

Install the Earth Engine API if it is not already installed:

```bash
pip install earthengine-api
```

Authenticate:

```bash
earthengine authenticate
```

Initialize using your Google Cloud project:

```python
import ee

ee.Initialize(project="YOUR_GOOGLE_CLOUD_PROJECT_ID")
```

Replace the project ID with your configured Google Cloud project.

---

## 📁 Project Structure

```text
ForestWatch-AI/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── config.yaml
│   └── regions.yaml
│
├── src/
│   ├── alignment.py
│   ├── change_detection.py
│   ├── config.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── forest_mask.py
│   ├── gee_export.py
│   ├── geocoder.py
│   ├── mock_data.py
│   ├── model.py
│   ├── ndvi.py
│   ├── pilot_export.py
│   ├── predict.py
│   ├── tiling.py
│   └── train.py
│
├── tests/
│   ├── test_alignment.py
│   ├── test_change_detection.py
│   ├── test_config.py
│   ├── test_dataset.py
│   ├── test_evaluate.py
│   ├── test_forest_mask.py
│   ├── test_gee_export.py
│   ├── test_mock_data.py
│   ├── test_model.py
│   ├── test_ndvi.py
│   ├── test_predict.py
│   ├── test_tiling.py
│   └── test_train.py
│
├── data/
│   ├── raw/
│   ├── synthetic/
│   ├── tiles/
│   └── outputs/
│
└── screenshots/
```

Large satellite datasets, generated raster outputs, and model checkpoints are excluded from the Git repository.

---

## 🧪 Testing

The project includes automated tests using Pytest.

Run the test suite:

```bash
python -m pytest tests/ -v
```

---

## 🚀 Future Enhancements

- Extend monitoring to more geographical regions.
- Improve cross-region model generalization.
- Validate predictions using independently verified ground-truth datasets.
- Integrate additional satellite sources such as Landsat.
- Introduce near-real-time forest monitoring.
- Improve cloud and shadow handling.
- Add automated alerts for significant forest-cover changes.
- Explore temporal deep learning architectures for long-term forest monitoring.

---

## ⚠️ Limitations

- Dynamic World labels are pseudo-labels and may contain classification errors.
- Model evaluation metrics represent agreement with pseudo-labels rather than independent ground truth.
- Cross-region predictions require additional validation.
- Satellite image quality and NoData regions can affect monitoring coverage.
- Detected tree-cover loss is a potential change signal and does not independently establish human-caused deforestation.

---

## 🌱 Conclusion

ForestWatch AI demonstrates how satellite remote sensing, deep learning, and geospatial analysis can be combined to build an automated forest monitoring system.

By integrating U-Net-based image segmentation, multi-year change detection, Google Earth Engine, and interactive visualization, the system provides an extensible foundation for monitoring forest-cover changes across geographical regions.

---

## 👩‍💻 Author

**Sai Krupa Reddy**

GitHub: [@SaikrupaReddy111](https://github.com/SaikrupaReddy111)

Project: [ForestWatch AI](https://github.com/SaikrupaReddy111/ForestWatch-AI)

---

*Built with Python, PyTorch, Sentinel-2, Google Earth Engine, and Streamlit.*
