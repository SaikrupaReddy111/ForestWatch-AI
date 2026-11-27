"""Small pilot-area Earth Engine export for ForestWatch AI.

Exports 2020 and 2023 Sentinel-2 composites plus Dynamic World masks
for a circular pilot area around the resolved location's polygon centroid.
"""
from __future__ import annotations

import argparse
import json

import ee

from src.config import load_full_pipeline_config
from src.geocoder import search_location
from src.gee_export import (
    EARTH_ENGINE_PROJECT,
    _full_year_dates,
    _geometry_bbox,
    _schedule_exports,
    _select_export_crs,
    initialize_earth_engine,
)


def export_pilot(location_name: str, radius_km: float = 2.0,
                 year1: int = 2020, year2: int = 2023,
                 output_dir: str = "data/raw") -> dict:
    if radius_km <= 0 or radius_km > 10:
        raise ValueError("Pilot radius must be greater than 0 and at most 10 km.")

    config, _ = load_full_pipeline_config(
        "configs/config.yaml", "configs/regions.yaml"
    )

    resolved = search_location(location_name)
    geojson = resolved.get("geojson")
    if not geojson or resolved.get("geometry_type") not in ("Polygon", "MultiPolygon"):
        raise ValueError("The location resolver did not return a usable polygon.")

    initialized, message = initialize_earth_engine(project=EARTH_ENGINE_PROJECT)
    if not initialized:
        raise RuntimeError(message)

    # Use a small circular pilot around the resolved boundary's centroid.
    source_geometry = ee.Geometry(geojson)
    center = source_geometry.centroid(maxError=1)
    roi = center.buffer(radius_km * 1000, maxError=1)
    roi_geojson = roi.bounds(maxError=1).getInfo()
    bbox = _geometry_bbox(roi_geojson)

    dates = {
        year1: _full_year_dates(year1),
        year2: _full_year_dates(year2),
    }
    metadata = {
        "search_query": location_name,
        "matched_name": resolved.get("name"),
        "country": resolved.get("country"),
        "ecoregion_name": resolved.get("ecoregion_name"),
        "ecoregion_id": resolved.get("ecoregion_id"),
        "biome": resolved.get("biome"),
        "boundary_source": resolved.get("boundary_source"),
        "geometry_type": resolved.get("geometry_type"),
        "pilot_area": "circle around resolved polygon centroid",
        "pilot_radius_km": radius_km,
        "composite_window": "full calendar year",
    }

    return _schedule_exports(
        region_id=f"{resolved.get('region_id', 'forest')}_pilot_{radius_km:g}km",
        region_name=f"{resolved.get('name', location_name)} — {radius_km:g} km pilot",
        roi=roi,
        bbox=bbox,
        crs=_select_export_crs(bbox),
        years=(year1, year2),
        date_ranges=dates,
        config=config,
        output_dir=output_dir,
        dry_run=False,
        metadata=metadata,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a small ForestWatch AI pilot area.")
    parser.add_argument("--location", default="Western Ghats, India")
    parser.add_argument("--radius-km", type=float, default=2.0)
    parser.add_argument("--year1", type=int, default=2020)
    parser.add_argument("--year2", type=int, default=2023)
    parser.add_argument("--output-dir", default="data/raw")
    args = parser.parse_args()

    result = export_pilot(
        args.location, args.radius_km, args.year1, args.year2, args.output_dir
    )
    print(json.dumps(result, indent=2))
    if result.get("status") == "export_tasks_created":
        print("\nFour pilot export tasks have been submitted to Google Drive.")
        print("Wait for all tasks to finish, then download the four GeoTIFFs")
        print(f"into: {args.output_dir}")


if __name__ == "__main__":
    main()
