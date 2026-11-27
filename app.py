import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import rasterio
import streamlit as st
from matplotlib.colors import BoundaryNorm, ListedColormap

BASE = Path(__file__).resolve().parent
OUT = BASE / "data" / "outputs"

REGIONS = {
    "Amazon Brazil": {
        "summary": "amazon_brazil_deforestation_summary_2020_2023.json",
        "change": "amazon_brazil_deforestation_change_2020_2023.tif",
        "forest_2020": "amazon_brazil_forest_2020.tif",
        "forest_2023": "amazon_brazil_forest_2023.tif",
        "model": "amazon_brazil_real_best_model.pth",
        "history": "amazon_brazil_real_training_history.json",
        "description": "Regional monitoring using the Amazon Brazil satellite dataset.",
        "experimental": False,
    },
    "Western Ghats, India (Pilot)": {
        "summary": "western_ghats_pilot_summary_2020_2023.json",
        "change": "western_ghats_pilot_change_2020_2023.tif",
        "forest_2020": "western_ghats_pilot_forest_2020.tif",
        "forest_2023": "western_ghats_pilot_forest_2023.tif",
        "model": "amazon_brazil_real_best_model.pth",
        "history": "amazon_brazil_real_training_history.json",
        "description": (
            "Experimental 2 km-radius pilot around the resolved location in "
            "the South Western Ghats montane rain forests ecoregion. This is "
            "not a map of the entire Western Ghats."
        ),
        "experimental": True,
    },
}

st.set_page_config(
    page_title="ForestWatch AI",
    page_icon="🌳",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
:root { --forest:#102b20; --green:#26734d; --ink:#182b22; --muted:#718078; --line:#e4ebe5; --paper:#f4f7f4; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:var(--paper); color:var(--ink); }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#102b20,#0b2118); }
[data-testid="stSidebar"] * { color:#eaf3ed; }
[data-testid="stMetric"] { background:white; border:1px solid var(--line); padding:16px; border-radius:13px; min-width:0; }
[data-testid="stMetricValue"] { color:#173d2c; font-weight:800; font-size:clamp(.9rem,1.2vw,1.3rem)!important; white-space:normal!important; overflow-wrap:anywhere; }
[data-testid="stMetricLabel"] { white-space:normal!important; }
.block-container { max-width:1500px; padding-top:1.5rem; }
.hero { background:linear-gradient(112deg,#123b29,#1d6241); border-radius:16px; padding:25px; color:white; margin:20px 0; }
.hero h2 { color:white; }
.hero p { color:#d0e5d6; }
.eyebrow { color:#548067; font-size:10px; font-weight:800; letter-spacing:2px; text-transform:uppercase; }
.footer { border-top:1px solid #dfe8e1; margin-top:35px; padding-top:15px; color:#829087; font-size:11px; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def read_json(path_string):
    path = Path(path_string)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def read_raster(path_string, max_dimension=1200):
    path = Path(path_string)
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        scale = min(1.0, max_dimension / max(src.height, src.width))
        height = max(1, int(src.height * scale))
        width = max(1, int(src.width * scale))
        arr = src.read(
            1,
            out_shape=(height, width),
            resampling=rasterio.enums.Resampling.nearest,
        ).astype(np.float32)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        arr[~np.isfinite(arr)] = np.nan
        return arr


@st.cache_data
def read_raster_metadata(path_string):
    path = Path(path_string)
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        return {
            "width": src.width,
            "height": src.height,
            "crs": str(src.crs),
            "transform": tuple(src.transform),
            "nodata": src.nodata,
            "bounds": tuple(src.bounds),
            "pixel_area_m2": abs(src.transform.a * src.transform.e),
        }


def section(eyebrow, title, subtitle=None):
    st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)


def format_ha(value):
    return f"{value:,.2f} ha"


def show_raster(title, filename, kind="forest"):
    path = OUT / filename
    arr = read_raster(str(path))
    if arr is None:
        st.warning(f"Raster not found: {path}")
        return

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#edf2ed")

    if kind == "change":
        colors = ["#cbd5ce", "#26734d", "#dc594d", "#7ebd83"]
        cmap = ListedColormap(colors)
        cmap.set_bad("white")
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
        ax.imshow(arr, cmap=cmap, norm=norm, interpolation="nearest")
        labels = [
            ("Stable non-forest", colors[0]),
            ("Stable forest", colors[1]),
            ("Potential loss", colors[2]),
            ("Potential gain", colors[3]),
        ]
        handles = [
            plt.Line2D([0], [0], marker="s", linestyle="",
                       markerfacecolor=color, markeredgecolor="none",
                       markersize=8, label=label)
            for label, color in labels
        ]
        ax.legend(handles=handles, loc="upper center",
                  bbox_to_anchor=(0.5, -0.07), ncol=2,
                  frameon=False, fontsize=8)
    else:
        ax.imshow(arr, cmap="YlGn", interpolation="nearest", vmin=0, vmax=1)
        ax.set_title("1 = Forest | 0 = Non-forest",
                     fontsize=8, color="#718078")

    ax.set_title(title, loc="left", fontsize=11,
                 fontweight="bold", color="#1c3d2b", pad=12)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# Sidebar: choose a region while retaining the existing five-page workspace.
with st.sidebar:
    st.markdown(
        '<div style="font-size:24px;font-weight:800;color:white">'
        '🌳 Forest<span style="color:#72d49b">Watch</span></div>'
        '<div style="font-size:9px;letter-spacing:2px;color:#9bb8a5">'
        'GEOSPATIAL INTELLIGENCE</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    region_name = st.selectbox("Monitoring region", list(REGIONS.keys()))
    config = REGIONS[region_name]
    page = st.radio(
        "Workspace",
        ["Overview", "Satellite imagery", "Change detection",
         "AI model insights", "Methodology"],
    )
    st.markdown("---")
    st.caption("Comparison period")
    st.markdown("**2020 – 2023**")
    st.markdown("---")
    st.success("Region selected")
    st.caption("Sentinel-2 imagery · U-Net predictions")

summary_path = OUT / config["summary"]
summary = read_json(str(summary_path))

st.markdown(
    '<div class="eyebrow">FORESTWATCH INTELLIGENCE / WORKSPACE</div>',
    unsafe_allow_html=True,
)
st.markdown("""
<div class="hero">
  <div style="font-size:10px;letter-spacing:2px;font-weight:800">
    SATELLITE-BASED FOREST MONITORING
  </div>
  <h2>A clearer view of changing forest landscapes.</h2>
  <p>AI-powered forest-cover classification and potential change monitoring
  using Sentinel-2 imagery.</p>
</div>
""", unsafe_allow_html=True)

st.markdown(f"**Selected region:** {region_name}")
st.caption(config["description"])

if config["experimental"]:
    st.warning(
        "Experimental cross-region pilot: the U-Net was trained on Amazon "
        "Brazil data using Dynamic World pseudo-labels and applied here. "
        "These results are not independently validated ground-truth "
        "deforestation measurements."
    )
else:
    st.warning(
        "Model-based estimates: the U-Net was trained using Dynamic World "
        "pseudo-labels. Potential forest loss is not independently verified "
        "ground-truth deforestation."
    )

if summary is None:
    st.error(f"Summary file not found: {summary_path}")
    st.info("Generate the selected region's change-detection summary first.")
    st.stop()

areas = summary["areas_hectares"]
counts = summary["pixel_counts"]
loss_ha = areas["potential_tree_cover_loss_ha"]
forest_ha = areas["stable_forest_ha"]
gain_ha = areas["potential_forest_gain_ha"]
monitored_ha = areas["valid_monitored_area_ha"]

change_file = config["change"]
forest_2020_file = config["forest_2020"]
forest_2023_file = config["forest_2023"]

if page == "Overview":
    section("01 / EXECUTIVE SUMMARY", f"{region_name} monitoring overview",
            "Model comparison for 2020 to 2023.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Potential tree-cover loss", format_ha(loss_ha))
    c2.metric("Stable forest", format_ha(forest_ha))
    c3.metric("Potential forest gain", format_ha(gain_ha))
    c4.metric("Valid monitored area", format_ha(monitored_ha))

    st.markdown("---")
    section("02 / GEOSPATIAL INTELLIGENCE", "Forest change map",
            "U-Net prediction comparison between 2020 and 2023.")
    show_raster(f"{region_name} · potential change · 2020–2023",
                change_file, "change")

    st.markdown("---")
    section("03 / LAND-COVER BREAKDOWN", "Change category statistics")
    names = ["Stable forest", "Stable non-forest",
             "Potential loss", "Potential gain"]
    values = [
        counts["stable_forest"], counts["stable_non_forest"],
        counts["potential_tree_cover_loss"], counts["potential_forest_gain"],
    ]
    colors = ["#26734d", "#bdc9c0", "#dc594d", "#7ebd83"]
    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h", marker_color=colors,
        text=[f"{v:,}" for v in values], textposition="outside",
    ))
    fig.update_layout(
        height=350, margin=dict(l=5, r=50, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Pixel count", yaxis=dict(autorange="reversed"),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

elif page == "Satellite imagery":
    section("04 / SATELLITE OBSERVATIONS", "Forest-cover comparison",
            "U-Net classification results for both observation years.")
    tab1, tab2 = st.tabs(["2020 forest cover", "2023 forest cover"])
    with tab1:
        show_raster("Predicted forest cover · 2020", forest_2020_file, "forest")
    with tab2:
        show_raster("Predicted forest cover · 2023", forest_2023_file, "forest")
    st.info("Green represents pixels classified as forest; low/grey values "
            "represent non-forest. White areas are NoData.")

elif page == "Change detection":
    section("05 / TEMPORAL ANALYSIS", "Forest-cover change",
            "Pixel-level comparison of U-Net forest masks.")
    show_raster(f"Forest change classification · {region_name} · 2020–2023",
                change_file, "change")
    c1, c2, c3 = st.columns(3)
    c1.metric("Potential loss", format_ha(loss_ha))
    c2.metric("Stable forest", format_ha(forest_ha))
    c3.metric("Potential gain", format_ha(gain_ha))

    st.markdown("#### Classification statistics")
    stats_table = {
        "Class": ["Stable non-forest", "Stable forest",
                  "Potential tree-cover loss", "Potential forest gain",
                  "Invalid / NoData pixels"],
        "Pixel count": [
            counts["stable_non_forest"], counts["stable_forest"],
            counts["potential_tree_cover_loss"], counts["potential_forest_gain"],
            counts["invalid_pixels"],
        ],
    }
    st.dataframe(stats_table, use_container_width=True, hide_index=True)

    meta = read_raster_metadata(str(OUT / change_file))
    if meta:
        total_pixels = meta["width"] * meta["height"]
        nodata_pct = counts["invalid_pixels"] / total_pixels * 100
        st.caption(
            f"Raster: {meta['width']} × {meta['height']} pixels · "
            f"CRS: {meta['crs']} · NoData: {nodata_pct:.2f}% of raster pixels"
        )

    st.markdown("#### Download results")
    change_path = OUT / change_file
    if change_path.exists():
        with change_path.open("rb") as f:
            st.download_button(
                "Download change detection GeoTIFF", data=f.read(),
                file_name=change_file, mime="image/tiff",
            )
    if summary_path.exists():
        with summary_path.open("rb") as f:
            st.download_button(
                "Download analysis summary JSON", data=f.read(),
                file_name=summary_path.name, mime="application/json",
            )

elif page == "AI model insights":
    section("06 / MACHINE LEARNING", "AI model insights",
            "U-Net ResNet34 model trained on Amazon Brazil satellite tiles.")
    model_path = OUT / "models" / "real_amazon" / config["model"]
    history_path = OUT / "models" / "real_amazon" / config["history"]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Trained model checkpoint**")
        if model_path.exists():
            st.success("Model checkpoint available")
            st.metric("Model size",
                      f"{model_path.stat().st_size / (1024 * 1024):.1f} MB")
        else:
            st.warning("Model checkpoint not found")
    with c2:
        st.markdown("**Prediction outputs**")
        for filename in [forest_2020_file, forest_2023_file, change_file]:
            if (OUT / filename).exists():
                st.success(filename)
            else:
                st.warning(f"Missing: {filename}")

    history = read_json(str(history_path))
    if history is not None:
        st.markdown("#### Training history")
        st.json(history)

    st.markdown("#### Evaluation interpretation")
    st.info(
        "Amazon test metrics were evaluated against held-out geographic "
        "Dynamic World pseudo-labels, not independent ground truth. "
        "Western Ghats agreement with Dynamic World is a preliminary "
        "cross-region comparison, not independent accuracy validation."
    )

elif page == "Methodology":
    section("07 / SCIENTIFIC FRAMEWORK", "Methodology and data provenance",
            "ForestWatch AI processing pipeline.")
    stages = [
        ("01 · Satellite data acquisition",
         "Sentinel-2 multispectral imagery for 2020 and 2023."),
        ("02 · Data preparation",
         "Six spectral bands are aligned, tiled and spatially split."),
        ("03 · Pseudo-label preparation",
         "Dynamic World land-cover classifications are used as training labels."),
        ("04 · Deep learning",
         "A U-Net architecture with ResNet34 encoder classifies forest cover."),
        ("05 · Prediction",
         "The trained model generates forest masks for both years."),
        ("06 · Change detection",
         "Forest masks are compared to identify stable forest, potential loss and potential gain."),
        ("07 · Area estimation",
         "Pixel counts are converted into hectares using geospatial resolution."),
    ]
    for title, description in stages:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(description)

    st.markdown("#### Change-map classes")
    st.dataframe({
        "Code": [0, 1, 2, 3, 255],
        "Class": ["Stable non-forest", "Stable forest",
                  "Potential tree-cover loss", "Potential forest gain", "NoData"],
    }, use_container_width=True, hide_index=True)

    st.markdown("#### Selected analysis configuration")
    st.json({
        "Region": summary["region"],
        "Period": summary["period"],
        "Method": summary["method"],
        "Pixel area (m²)": summary["pixel_area_m2"],
        "Minimum patch size (ha)": summary["minimum_patch_ha"],
        "Model": "U-Net ResNet34",
        "Training labels": "Dynamic World pseudo-labels",
        "Experimental cross-region pilot": config["experimental"],
    })
    st.warning(summary["scientific_note"])

st.markdown(f"""
<div class="footer">
  <b style="color:#315c42">FORESTWATCH AI</b>
  · Geospatial intelligence for forest monitoring.
  <br><br>
  {region_name} · Sentinel-2 · U-Net · 2020–2023
</div>
""", unsafe_allow_html=True)
