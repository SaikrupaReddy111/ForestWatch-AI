
"""Automatic geographic boundary resolver for ForestWatch AI."""

import re
import time
from typing import Any, Dict

import requests
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderServiceError, GeocoderTimedOut


class LocationNotFoundError(Exception):
    pass


class LocationGeocoder:

    ECOREGION_URL = (
        "https://services.arcgis.com/P3ePLMYs2RVChkJx/"
        "ArcGIS/rest/services/Resolve_Ecoregions/FeatureServer/0/query"
    )

    def __init__(self):
        self.geolocator = Nominatim(
            user_agent="forestwatch-ai-project",
            timeout=20
        )
        self.session = requests.Session()
        self.last_request = 0.0

    def _geocode(self, place_name: str) -> Dict[str, Any]:

        elapsed = time.monotonic() - self.last_request

        if elapsed < 1.1:
            time.sleep(1.1 - elapsed)

        try:
            results = self.geolocator.geocode(
                place_name,
                exactly_one=False,
                limit=5,
                addressdetails=True,
                geometry="geojson"
            )
            self.last_request = time.monotonic()

        except (GeocoderTimedOut, GeocoderServiceError) as exc:
            raise RuntimeError(
                "Location search is temporarily unavailable."
            ) from exc

        if not results:
            raise LocationNotFoundError(
                f"Could not find location: {place_name}"
            )

        # Prefer a polygon result from OpenStreetMap.
        for result in results:
            raw = result.raw
            geometry = raw.get("geojson")

            if geometry and geometry.get("type") in (
                "Polygon", "MultiPolygon"
            ):
                return {
                    "latitude": float(result.latitude),
                    "longitude": float(result.longitude),
                    "name": result.address,
                    "geojson": geometry,
                    "boundary_source": "OpenStreetMap",
                    "geometry_type": geometry["type"],
                    "osm_id": raw.get("osm_id"),
                    "country": raw.get("address", {}).get(
                        "country", "Unknown"
                    )
                }

        # No polygon was found. Use the first geocoded point
        # to identify its containing ecological region.
        result = results[0]

        return {
            "latitude": float(result.latitude),
            "longitude": float(result.longitude),
            "name": result.address,
            "country": result.raw.get("address", {}).get(
                "country", "Unknown"
            )
        }

    def _find_ecoregion(
        self,
        latitude: float,
        longitude: float
    ) -> Dict[str, Any]:

        params = {
            "where": "1=1",
            "geometry": f"{longitude},{latitude}",
            "geometryType": "esriGeometryPoint",
            "inSR": 4326,
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "ECO_ID,ECO_NAME,BIOME_NAME,REALM",
            "returnGeometry": "true",
            "outSR": 4326,
            "f": "geojson"
        }

        try:
            response = self.session.get(
                self.ECOREGION_URL,
                params=params,
                timeout=60
            )
            response.raise_for_status()
            data = response.json()

        except (requests.RequestException, ValueError) as exc:
            raise RuntimeError(
                "Ecological boundary service is unavailable."
            ) from exc

        features = data.get("features", [])

        if not features:
            raise LocationNotFoundError(
                "No matching ecological polygon was found "
                "for this location."
            )

        feature = features[0]
        geometry = feature.get("geometry")

        if not geometry or geometry.get("type") not in (
            "Polygon", "MultiPolygon"
        ):
            raise LocationNotFoundError(
                "The ecological service returned no usable polygon."
            )

        properties = feature.get("properties", {})

        return {
            "geojson": geometry,
            "geometry_type": geometry["type"],
            "ecoregion_name": properties.get("ECO_NAME"),
            "ecoregion_id": properties.get("ECO_ID"),
            "biome": properties.get("BIOME_NAME"),
            "realm": properties.get("REALM"),
            "boundary_source": "RESOLVE Ecoregions 2017"
        }

    def search(self, place_name: str) -> Dict[str, Any]:

        if not place_name or not place_name.strip():
            raise ValueError("Please enter a location name.")

        place_name = place_name.strip()

        location = self._geocode(place_name)

        if "geojson" in location:
            boundary = {
                "geojson": location["geojson"],
                "geometry_type": location["geometry_type"],
                "boundary_source": location["boundary_source"]
            }
        else:
            boundary = self._find_ecoregion(
                location["latitude"],
                location["longitude"]
            )

        region_id = re.sub(
            r"[^a-z0-9]+",
            "_",
            place_name.lower()
        ).strip("_")

        return {
            "region_id": region_id,
            "search_query": place_name,
            "name": location["name"],
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "country": location["country"],
            **boundary
        }


def search_location(place_name: str) -> Dict[str, Any]:
    return LocationGeocoder().search(place_name)


if __name__ == "__main__":

    place = input("Enter a location: ")

    try:
        result = search_location(place)

        print("\nLocation successfully resolved:")

        for key, value in result.items():
            if key == "geojson":
                print("geojson: Polygon boundary retrieved")
            else:
                print(f"{key}: {value}")

    except (ValueError, RuntimeError, LocationNotFoundError) as exc:
        print(f"\nError: {exc}")
