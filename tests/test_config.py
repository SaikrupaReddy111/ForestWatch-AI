"""Unit tests for configuration loading and Pydantic validation schemas."""

import pytest
from pathlib import Path
from pydantic import ValidationError

from src.config import (
    load_config,
    load_regions,
    load_full_pipeline_config,
    RegionConfig,
    DrySeasonWindow,
    TargetYears,
    MasterConfig,
    SplittingConfig,
    LossWeights,
    ChangeDetectionConfig,
)


def test_load_real_configs():
    """Verify that the repository's YAML config files load and validate without error."""
    cfg, regions = load_full_pipeline_config("configs/config.yaml", "configs/regions.yaml")
    
    assert cfg.project_name == "ForestWatch AI"
    assert "prototype" in cfg.prototype_disclaimer.lower()
    assert cfg.seed == 42
    assert len(cfg.data.sentinel2_bands) == 6
    assert cfg.model.in_channels == 6
    assert cfg.model.first_conv_init == "channel_mapping_and_average"
    
    # Check that initial three regions are present
    assert "amazon_brazil" in regions
    assert "congo_basin" in regions
    assert "southeast_asia_kalimantan" in regions
    
    # Check Amazon bounds
    amazon = regions["amazon_brazil"]
    assert amazon.continent == "South America"
    assert amazon.bbox[0] < amazon.bbox[2]  # west < east
    assert amazon.bbox[1] < amazon.bbox[3]  # south < north
    assert amazon.target_years.t1 < amazon.target_years.t2


def test_bbox_validation():
    """Test latitude and longitude bounding box validation."""
    valid_data = {
        "region_id": "test_region",
        "name": "Test Region",
        "continent": "Test Continent",
        "country": "Test Country",
        "bbox": [-60.0, -10.0, -50.0, -5.0],
        "dry_season": {"start_month": 7, "start_day": 1, "end_month": 8, "end_day": 31},
        "target_years": {"t1": 2020, "t2": 2023},
        "projected_utm_crs": "EPSG:32720",
    }
    # Valid region
    reg = RegionConfig(**valid_data)
    assert reg.region_id == "test_region"

    # Inverted longitude: west > east
    invalid_bbox_lon = valid_data.copy()
    invalid_bbox_lon["bbox"] = [-50.0, -10.0, -60.0, -5.0]
    with pytest.raises(ValidationError, match="BBox west"):
        RegionConfig(**invalid_bbox_lon)

    # Inverted latitude: south > north
    invalid_bbox_lat = valid_data.copy()
    invalid_bbox_lat["bbox"] = [-60.0, -5.0, -50.0, -10.0]
    with pytest.raises(ValidationError, match="BBox south"):
        RegionConfig(**invalid_bbox_lat)

    # Out of range latitude (> 90)
    out_of_range = valid_data.copy()
    out_of_range["bbox"] = [-60.0, -10.0, -50.0, 95.0]
    with pytest.raises(ValidationError, match="Latitudes must be between"):
        RegionConfig(**out_of_range)


def test_target_years_validation():
    """Test validation of chronological target years."""
    # Invalid: t1 == t2
    with pytest.raises(ValidationError, match="must be strictly earlier"):
        TargetYears(t1=2020, t2=2020)

    # Invalid: t1 > t2
    with pytest.raises(ValidationError, match="must be strictly earlier"):
        TargetYears(t1=2023, t2=2020)

    # Valid
    ty = TargetYears(t1=2018, t2=2022)
    assert ty.t1 == 2018
    assert ty.t2 == 2022


def test_dry_season_window_validation():
    """Test dry season window validation."""
    # Valid
    w = DrySeasonWindow(start_month=6, start_day=15, end_month=8, end_day=15)
    assert w.start_month == 6

    # Invalid: same month but start_day >= end_day
    with pytest.raises(ValidationError, match="end_day must be after start_day"):
        DrySeasonWindow(start_month=7, start_day=20, end_month=7, end_day=10)


def test_split_ratios_validation():
    """Test that train/val/test ratios must sum to 1.0."""
    # Valid
    s = SplittingConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    assert s.train_ratio == 0.70

    # Invalid sum > 1.0
    with pytest.raises(ValidationError, match="Split ratios must sum to 1.0"):
        SplittingConfig(train_ratio=0.80, val_ratio=0.20, test_ratio=0.10)


def test_loss_weights_validation():
    """Test that BCE and Dice loss weights must sum to 1.0."""
    # Valid
    lw = LossWeights(bce=0.4, dice=0.6)
    assert lw.bce == 0.4

    # Invalid sum != 1.0
    with pytest.raises(ValidationError, match="must sum to 1.0"):
        LossWeights(bce=0.5, dice=0.7)


def test_change_detection_config():
    """Test change detection thresholds."""
    cd = ChangeDetectionConfig(min_patch_size_ha=0.75, connectivity=8)
    assert cd.min_patch_size_ha == 0.75
    assert cd.loss_event_label == "tree_cover_loss"

    # Non-positive patch size
    with pytest.raises(ValidationError):
        ChangeDetectionConfig(min_patch_size_ha=-0.5)


def test_missing_config_file_raises():
    """Verify that attempting to load a non-existent configuration raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config("configs/non_existent_file.yaml")


def test_dynamic_region_addition():
    """Verify that new regions can be registered dynamically via config without modifying code."""
    regions = load_regions("configs/regions.yaml")
    initial_count = len(regions)
    
    # Add new study region (e.g. Romanian Carpathians)
    carpathians = RegionConfig(
        region_id="romanian_carpathians",
        name="Romanian Carpathians (Logging Hotspot)",
        continent="Europe",
        country="Romania",
        bbox=[25.0, 45.5, 26.0, 46.5],
        dry_season=DrySeasonWindow(start_month=6, start_day=1, end_month=8, end_day=31),
        target_years=TargetYears(t1=2019, t2=2022),
        projected_utm_crs="EPSG:32635",
        equal_area_crs="ESRI:54034",
        notes="Temperate mixed/coniferous forest undergoing illegal sanitation cutting and logging.",
    )
    regions["romanian_carpathians"] = carpathians
    assert len(regions) == initial_count + 1
    assert "romanian_carpathians" in regions
