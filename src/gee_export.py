"""Google Earth Engine acquisition and export pipeline for ForestWatch AI.

Supports predefined regions and dynamically resolved polygon locations.
Dynamic locations use full-calendar-year composites; predefined regions retain
configured dry-season windows. Dynamic World labels are pseudo-labels, not
independently verified ground truth.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
import zipfile
from datetime import date, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import ee  # type: ignore
    EE_AVAILABLE = True
except ImportError:  # pragma: no cover
    ee = None  # type: ignore
    EE_AVAILABLE = False

from src.config import MasterConfig, RegionConfig

EARTH_ENGINE_PROJECT = "lucky-reactor-515812-h9"
DRIVE_EXPORT_FOLDER = "EarthEngineExports"


def initialize_earth_engine(project: Optional[str] = None,
                             service_account: Optional[str] = None,
                             key_file: Optional[str] = None) -> Tuple[bool, str]:
    """Initialize Earth Engine using interactive credentials or a service account."""
    if not EE_AVAILABLE:
        return False, "earthengine-api is not installed. Install with: python -m pip install earthengine-api"
    try:
        if service_account and key_file:
            credentials = ee.ServiceAccountCredentials(service_account, key_file)
            ee.Initialize(credentials, project=project)
        else:
            ee.Initialize(project=project)
        return True, "Earth Engine initialized successfully."
    except Exception as exc:
        return False, (f"Earth Engine initialization failed: {type(exc).__name__}: {exc}. "
                       "Authenticate with: earthengine authenticate")


def mask_sentinel2_sr(image: Any, scl_classes: Optional[List[int]] = None,
                      scale_factor: float = 10000.0) -> Any:
    """Mask cloud/shadow/cirrus/snow SCL classes and scale optical reflectance."""
    if scl_classes is None:
        scl_classes = [3, 8, 9, 10, 11]
    scl = image.select("SCL")
    mask = ee.Image(1)
    for class_value in scl_classes:
        mask = mask.And(scl.neq(class_value))
    bands = ["B2", "B3", "B4", "B8", "B11", "B12"]
    return image.select(bands).divide(scale_factor).updateMask(mask)


def _season_dates(region_config: RegionConfig, year: int) -> Tuple[str, str]:
    """Return configured seasonal start and exclusive end dates."""
    season = region_config.dry_season
    start = f"{year}-{season.start_month:02d}-{season.start_day:02d}"
    end_inclusive = date(year, season.end_month, season.end_day)
    return start, (end_inclusive + timedelta(days=1)).isoformat()


def _full_year_dates(year: int) -> Tuple[str, str]:
    """Return a full calendar year as an inclusive-start/exclusive-end range."""
    if year < 2015 or year > 2050:
        raise ValueError(f"Year must be between 2015 and 2050. Got {year}.")
    return f"{year}-01-01", f"{year + 1}-01-01"


def _build_sentinel2_for_roi(roi: Any, start_date: str, end_date: str,
                             config: MasterConfig) -> Any:
    collection = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                  .filterBounds(roi).filterDate(start_date, end_date)
                  .map(lambda image: mask_sentinel2_sr(
                      image, config.data.scl_mask_classes)))
    return collection.median().select(config.data.sentinel2_bands).clip(roi)


def _build_dynamic_world_for_roi(roi: Any, start_date: str, end_date: str,
                                 config: MasterConfig) -> Any:
    """Build Dynamic World tree pseudo-labels: 1 tree, 0 other, 255 unobserved."""
    collection = (ee.ImageCollection(config.data.dynamic_world_collection)
                  .filterBounds(roi).filterDate(start_date, end_date))
    labels = collection.select("label").mode()
    tree_probability = collection.select("trees").mean()
    valid_observations = collection.select("label").count().gt(0).unmask(0)
    return (labels.eq(config.data.dynamic_world_tree_class_index)
            .And(tree_probability.gte(config.data.pseudo_label_confidence_threshold))
            .unmask(0).where(valid_observations.Not(), 255)
            .toByte().clip(roi))


def build_sentinel2_composite(region_config: RegionConfig, year: int,
                              config: MasterConfig) -> Any:
    """Build a dry-season Sentinel-2 composite for a predefined region."""
    try:
        roi = ee.Geometry.BBox(*region_config.bbox)
        start, end = _season_dates(region_config, year)
        return _build_sentinel2_for_roi(roi, start, end, config)
    except Exception:
        return None


def build_dynamic_world_forest_mask(region_config: RegionConfig, year: int,
                                    config: MasterConfig) -> Any:
    """Build dry-season Dynamic World pseudo-labels for a predefined region."""
    try:
        roi = ee.Geometry.BBox(*region_config.bbox)
        start, end = _season_dates(region_config, year)
        return _build_dynamic_world_for_roi(roi, start, end, config)
    except Exception:
        return None


def get_download_url(image: Any, region_bbox: List[float], crs: str,
                     scale_meters: float = 10.0,
                     file_format: str = "GEO_TIFF") -> str:
    """Generate a direct Earth Engine download URL for a bounding box."""
    roi = ee.Geometry.BBox(*region_bbox)
    return image.getDownloadURL({"region": roi, "scale": scale_meters,
                                  "crs": crs, "format": file_format})


def download_geotiff_from_url(url: str, output_path: str | Path) -> Path:
    """Download a GeoTIFF or extract the first GeoTIFF from a ZIP response."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:
        content = response.read()
    if content[:2] == b"PK":
        with zipfile.ZipFile(BytesIO(content)) as archive:
            tif_names = [n for n in archive.namelist()
                         if n.lower().endswith((".tif", ".tiff"))]
            if not tif_names:
                raise ValueError("Downloaded ZIP archive did not contain a GeoTIFF.")
            with archive.open(tif_names[0]) as source:
                output_file.write_bytes(source.read())
    else:
        output_file.write_bytes(content)
    return output_file


def _geometry_bbox(geometry: Dict[str, Any]) -> List[float]:
    """Calculate [west, south, east, north] from GeoJSON coordinates."""
    coordinates = geometry.get("coordinates")
    if not coordinates:
        raise ValueError("GeoJSON geometry has no coordinates.")
    lons: List[float] = []
    lats: List[float] = []

    def visit(value: Any) -> None:
        if (isinstance(value, (list, tuple)) and len(value) >= 2
                and isinstance(value[0], (int, float))
                and isinstance(value[1], (int, float))):
            lons.append(float(value[0]))
            lats.append(float(value[1]))
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(coordinates)
    if not lons or not lats:
        raise ValueError("Could not extract coordinates from GeoJSON.")
    bbox = [min(lons), min(lats), max(lons), max(lats)]
    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        raise ValueError("GeoJSON polygon has an invalid bounding box.")
    return bbox


def _select_export_crs(bbox: List[float]) -> str:
    """Use local UTM for compact regions and global equal-area for broad ones."""
    west, south, east, north = bbox
    lon, lat = (west + east) / 2.0, (south + north) / 2.0
    if east - west > 6.0 or north - south > 8.0 or lat < -80 or lat > 84:
        return "EPSG:6933"
    zone = max(1, min(60, int((lon + 180.0) // 6.0) + 1))
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def _schedule_exports(region_id: str, region_name: str, roi: Any,
                      bbox: List[float], crs: str, years: Tuple[int, int],
                      date_ranges: Dict[int, Tuple[str, str]],
                      config: MasterConfig, output_dir: str | Path,
                      dry_run: bool = False,
                      metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build and schedule four Sentinel-2/Dynamic World Drive exports."""
    year1, year2 = years
    if year1 >= year2:
        raise ValueError("The baseline year must be earlier than the monitoring year.")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "t1_raster": str(out_dir / f"{region_id}_s2_{year1}.tif"),
        "t2_raster": str(out_dir / f"{region_id}_s2_{year2}.tif"),
        "t1_mask": str(out_dir / f"{region_id}_dw_mask_{year1}.tif"),
        "t2_mask": str(out_dir / f"{region_id}_dw_mask_{year2}.tif"),
    }
    manifest: Dict[str, Any] = {
        "region_id": region_id, "region_name": region_name,
        "target_years": {"t1": year1, "t2": year2},
        "date_ranges": {str(y): {"start_inclusive": date_ranges[y][0],
                                  "end_exclusive": date_ranges[y][1]} for y in years},
        "bbox": bbox, "projected_crs": crs,
        "resolution_meters": float(config.data.target_resolution_meters),
        "files": files, "label_source": "Google Dynamic World pseudo-labels",
    }
    if metadata:
        manifest["location_metadata"] = metadata
    if dry_run:
        manifest["status"] = "dry_run_success"
        return manifest

    initialized, message = initialize_earth_engine(project=EARTH_ENGINE_PROJECT)
    if not initialized:
        raise RuntimeError(f"Cannot export satellite imagery: {message}")

    images: Dict[str, Any] = {}
    for key, year in (("t1_raster", year1), ("t2_raster", year2)):
        start, end = date_ranges[year]
        images[key] = _build_sentinel2_for_roi(roi, start, end, config)
    for key, year in (("t1_mask", year1), ("t2_mask", year2)):
        start, end = date_ranges[year]
        images[key] = _build_dynamic_world_for_roi(roi, start, end, config)

    missing = [key for key, image in images.items() if image is None]
    if missing:
        raise RuntimeError("Earth Engine failed to build: " + ", ".join(missing)
                           + ". No export tasks were scheduled.")

    descriptions = {
        "t1_raster": f"{region_id} Sentinel-2 {year1}",
        "t2_raster": f"{region_id} Sentinel-2 {year2}",
        "t1_mask": f"{region_id} DynamicWorld mask {year1}",
        "t2_mask": f"{region_id} DynamicWorld mask {year2}",
    }
    tasks: Dict[str, Any] = {}
    for key, image in images.items():
        task = ee.batch.Export.image.toDrive(
            image=image, description=descriptions[key],
            fileNamePrefix=Path(files[key]).stem,
            folder=DRIVE_EXPORT_FOLDER, region=roi, crs=crs,
            scale=float(config.data.target_resolution_meters),
            fileFormat="GeoTIFF", maxPixels=1e13,
        )
        task.start()
        tasks[key] = task

    manifest["tasks"] = {
        key: {"task_id": getattr(task, "id", None),
              "description": getattr(task, "description", descriptions[key])}
        for key, task in tasks.items()
    }
    manifest["drive_folder"] = DRIVE_EXPORT_FOLDER
    manifest["status"] = "export_tasks_created"
    return manifest


def fetch_and_export_region(region_id: str, config: MasterConfig,
                            regions: Dict[str, RegionConfig],
                            output_dir: str | Path = "data/raw",
                            dry_run: bool = False) -> Dict[str, Any]:
    """Export a predefined region using its configured dry-season dates."""
    if region_id not in regions:
        raise ValueError(f"Unknown region_id '{region_id}'. Available: {list(regions)}")
    region = regions[region_id]
    roi = None if dry_run else ee.Geometry.BBox(*region.bbox)
    y1, y2 = region.target_years.t1, region.target_years.t2
    return _schedule_exports(
        region_id, region.name, roi, region.bbox, region.projected_utm_crs,
        (y1, y2), {y1: _season_dates(region, y1),
                   y2: _season_dates(region, y2)},
        config, output_dir, dry_run,
        {"country": region.country, "continent": region.continent,
         "boundary_source": "configs/regions.yaml",
         "composite_window": "configured dry season"},
    )


def fetch_and_export_dynamic_location(place_name: str, config: MasterConfig,
                                      output_dir: str | Path = "data/raw",
                                      year1: int = 2020, year2: int = 2023,
                                      dry_run: bool = False) -> Dict[str, Any]:
    """Resolve a place to a polygon and export full-calendar-year composites."""
    from src.geocoder import search_location

    location = search_location(place_name)
    geometry = location.get("geojson")
    geometry_type = location.get("geometry_type")
    if not geometry or geometry_type not in ("Polygon", "MultiPolygon"):
        raise ValueError("Location resolver did not return a Polygon/MultiPolygon boundary.")
    bbox = _geometry_bbox(geometry)
    roi = None if dry_run else ee.Geometry(geometry)
    crs = _select_export_crs(bbox)
    date_ranges = {year1: _full_year_dates(year1), year2: _full_year_dates(year2)}
    metadata = {
        "search_query": place_name, "matched_name": location.get("name"),
        "country": location.get("country"),
        "ecoregion_name": location.get("ecoregion_name"),
        "ecoregion_id": location.get("ecoregion_id"),
        "biome": location.get("biome"), "realm": location.get("realm"),
        "boundary_source": location.get("boundary_source"),
        "geometry_type": geometry_type,
        "composite_window": "full calendar year",
    }
    return _schedule_exports(
        location.get("region_id", "custom_location"),
        location.get("name", place_name), roi, bbox, crs,
        (year1, year2), date_ranges, config, output_dir, dry_run, metadata,
    )


def main() -> None:
    from src.config import load_full_pipeline_config

    parser = argparse.ArgumentParser(
        description="ForestWatch AI: Earth Engine satellite data acquisition")
    parser.add_argument("--region", type=str, default="amazon_brazil",
                        help="Predefined region ID from configs/regions.yaml")
    parser.add_argument("--location", type=str, default=None,
                        help="Dynamically resolve a place name and polygon boundary")
    parser.add_argument("--year1", type=int, default=2020)
    parser.add_argument("--year2", type=int, default=2023)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--regions", type=str, default="configs/regions.yaml")
    parser.add_argument("--output-dir", type=str, default="data/raw")
    args = parser.parse_args()
    config, regions = load_full_pipeline_config(args.config, args.regions)

    try:
        if args.location:
            print(f"Resolving dynamic location: {args.location}")
            manifest = fetch_and_export_dynamic_location(
                args.location, config, args.output_dir,
                args.year1, args.year2, args.dry_run)
        else:
            print(f"Preparing predefined region: {args.region}")
            manifest = fetch_and_export_region(
                args.region, config, regions, args.output_dir, args.dry_run)
        print("Export manifest:")
        print(json.dumps(manifest, indent=2))
        if manifest.get("status") == "export_tasks_created":
            print(f"\nExport tasks submitted. Open Google Drive > {DRIVE_EXPORT_FOLDER}, "
                  f"wait for completion, then download the four GeoTIFFs into "
                  f"{args.output_dir}.")
    except Exception as exc:
        print(f"Export failed: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
