"""Configuration loading and schema validation for ForestWatch AI.

Defines Pydantic models to strictly validate regional boundaries, seasonal
windows, band combinations, training hyperparameters, and change detection
thresholds across all pipeline stages.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# ============================================================================
# Regional Configuration Schemas
# ============================================================================

class DrySeasonWindow(BaseModel):
    """Specifies the dry-season calendar window for cloud-free composite imagery."""
    start_month: int = Field(..., ge=1, le=12, description="Start month (1-12)")
    start_day: int = Field(..., ge=1, le=31, description="Start day of month")
    end_month: int = Field(..., ge=1, le=12, description="End month (1-12)")
    end_day: int = Field(..., ge=1, le=31, description="End day of month")

    @model_validator(mode="after")
    def validate_dates(self) -> DrySeasonWindow:
        if self.start_month == self.end_month and self.start_day >= self.end_day:
            raise ValueError("end_day must be after start_day within the same month.")
        return self


class TargetYears(BaseModel):
    """Specifies multi-temporal comparison years."""
    t1: int = Field(..., ge=2015, le=2050, description="Baseline observation year")
    t2: int = Field(..., ge=2015, le=2050, description="Monitoring observation year")

    @model_validator(mode="after")
    def validate_chronology(self) -> TargetYears:
        if self.t1 >= self.t2:
            raise ValueError(f"t1 ({self.t1}) must be strictly earlier than t2 ({self.t2}).")
        return self


class RegionConfig(BaseModel):
    """Defines a geographic study region for forest monitoring."""
    region_id: str = Field(..., pattern=r"^[a-z0-9_]+$", description="Unique identifier")
    name: str = Field(..., min_length=3, description="Descriptive name of the study region")
    continent: str = Field(..., min_length=2)
    country: str = Field(..., min_length=2)
    # [west, south, east, north] in EPSG:4326 degrees
    bbox: List[float] = Field(..., min_length=4, max_length=4, description="[west, south, east, north]")
    dry_season: DrySeasonWindow
    target_years: TargetYears
    projected_utm_crs: str = Field(..., description="Local UTM projection (e.g., EPSG:32720)")
    equal_area_crs: str = Field("ESRI:54034", description="Equal-area CRS for ground area computations")
    notes: Optional[str] = Field(None, description="Biome description and deforestation drivers")

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, v: List[float]) -> List[float]:
        west, south, east, north = v
        if not (-180.0 <= west <= 180.0 and -180.0 <= east <= 180.0):
            raise ValueError(f"Longitudes must be between -180 and 180. Got west={west}, east={east}")
        if not (-90.0 <= south <= 90.0 and -90.0 <= north <= 90.0):
            raise ValueError(f"Latitudes must be between -90 and 90. Got south={south}, north={north}")
        if west >= east:
            raise ValueError(f"BBox west ({west}) must be strictly less than east ({east}).")
        if south >= north:
            raise ValueError(f"BBox south ({south}) must be strictly less than north ({north}).")
        return v


# ============================================================================
# Master Pipeline Configuration Schemas
# ============================================================================

class PathsConfig(BaseModel):
    """Filesystem directory paths."""
    raw_dir: str = "data/raw"
    tiles_dir: str = "data/tiles"
    synthetic_dir: str = "data/synthetic"
    outputs_dir: str = "data/outputs"
    models_dir: str = "data/outputs/models"
    reports_dir: str = "data/outputs/reports"
    experiments_csv: str = "data/outputs/experiments.csv"

    def ensure_directories(self, base_path: Optional[Path] = None) -> None:
        """Create configured directories if they do not exist."""
        base = base_path or Path.cwd()
        for path_str in [
            self.raw_dir,
            self.tiles_dir,
            self.synthetic_dir,
            self.outputs_dir,
            self.models_dir,
            self.reports_dir,
        ]:
            (base / path_str).mkdir(parents=True, exist_ok=True)


class DataConfig(BaseModel):
    """Satellite imagery and label specifications."""
    sentinel2_bands: List[str] = Field(
        default=["B2", "B3", "B4", "B8", "B11", "B12"],
        min_length=6,
        max_length=6,
        description="Sentinel-2 L2A multispectral bands",
    )
    band_descriptions: Optional[Dict[str, str]] = None
    target_resolution_meters: float = Field(10.0, gt=0.0)
    tile_size: int = Field(256, ge=64, le=1024)
    nodata_value: float = -9999.0
    mask_nodata_value: int = 255
    scl_mask_classes: List[int] = Field(default=[3, 8, 9, 10, 11])
    dynamic_world_collection: str = "GOOGLE/DYNAMICWORLD/V1"
    dynamic_world_tree_class_index: int = 1
    pseudo_label_confidence_threshold: float = Field(0.50, ge=0.0, le=1.0)


class SplittingConfig(BaseModel):
    """Spatial dataset splitting parameters."""
    strategy: Literal["spatial_block", "leave_one_region_out"] = "spatial_block"
    train_ratio: float = Field(0.70, gt=0.0, lt=1.0)
    val_ratio: float = Field(0.15, gt=0.0, lt=1.0)
    test_ratio: float = Field(0.15, gt=0.0, lt=1.0)
    buffer_tiles: int = Field(1, ge=0, description="Tile buffer width between geographic splits")

    @model_validator(mode="after")
    def validate_ratios(self) -> SplittingConfig:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Split ratios must sum to 1.0. Got sum={total:.3f}")
        return self


class BaselineConfig(BaseModel):
    """NDVI baseline parameters."""
    nir_band: str = "B8"
    red_band: str = "B4"
    epsilon: float = Field(1e-6, gt=0.0)
    default_threshold: float = Field(0.60, ge=-1.0, le=1.0)
    threshold_tuning_range: List[float] = Field(default=[0.45, 0.75, 0.05])


class ModelConfig(BaseModel):
    """Deep learning architecture parameters."""
    architecture: str = "unet"
    encoder_name: str = "resnet34"
    encoder_weights: Optional[str] = "imagenet"
    in_channels: int = Field(6, ge=1, le=12)
    num_classes: int = Field(1, ge=1)
    activation: Optional[str] = None
    first_conv_init: Literal["channel_mapping_and_average", "random", "repeat_rgb"] = "channel_mapping_and_average"


class LossWeights(BaseModel):
    bce: float = Field(0.50, ge=0.0, le=1.0)
    dice: float = Field(0.50, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_weights(self) -> LossWeights:
        if abs((self.bce + self.dice) - 1.0) > 1e-4:
            raise ValueError(f"BCE and Dice loss weights must sum to 1.0. Got sum={self.bce + self.dice}")
        return self


class SchedulerConfig(BaseModel):
    name: str = "cosine"
    t_max: int = 30
    eta_min: float = 1e-6


class EarlyStoppingConfig(BaseModel):
    enabled: bool = True
    patience: int = Field(7, ge=1)
    monitor: str = "val_iou"
    mode: Literal["min", "max"] = "max"


class TrainingConfig(BaseModel):
    batch_size: int = Field(16, ge=1)
    num_epochs: int = Field(30, ge=1)
    learning_rate: float = Field(0.0003, gt=0.0)
    weight_decay: float = Field(0.0001, ge=0.0)
    loss_weights: LossWeights = Field(default_factory=LossWeights)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)


class ChangeDetectionConfig(BaseModel):
    """Multi-temporal change detection and area calculation."""
    min_patch_size_ha: float = Field(0.50, gt=0.0, description="Minimum patch area threshold in hectares")
    connectivity: Literal[4, 8] = 8
    require_dual_validity: bool = True
    area_calculation_method: Literal["ellipsoidal_geodesic", "projected_equal_area"] = "ellipsoidal_geodesic"
    loss_event_label: str = "tree_cover_loss"


class EvaluationConfig(BaseModel):
    metrics: List[str] = Field(default=["iou", "dice", "precision", "recall"])
    error_analysis_top_k_tiles: int = Field(5, ge=1)


class MasterConfig(BaseModel):
    """Root configuration for ForestWatch AI pipeline."""
    project_name: str = "ForestWatch AI"
    prototype_disclaimer: str
    seed: int = Field(42, ge=0)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    splitting: SplittingConfig = Field(default_factory=SplittingConfig)
    baseline: BaselineConfig = Field(default_factory=BaselineConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    change_detection: ChangeDetectionConfig = Field(default_factory=ChangeDetectionConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)


# ============================================================================
# Loader and Registry Helper Functions
# ============================================================================

def load_yaml(file_path: str | Path) -> Dict[str, Any]:
    """Safely loads a YAML file from disk."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML content at {path} must parse into a dictionary.")
    return data


def load_regions(regions_path: str | Path = "configs/regions.yaml") -> Dict[str, RegionConfig]:
    """Loads and validates study region configurations."""
    raw = load_yaml(regions_path)
    if "regions" not in raw or not isinstance(raw["regions"], dict):
        raise KeyError(f"'regions' dictionary key missing in {regions_path}")

    regions = {}
    for region_id, reg_data in raw["regions"].items():
        if "region_id" not in reg_data:
            reg_data["region_id"] = region_id
        regions[region_id] = RegionConfig(**reg_data)
    return regions


def load_config(config_path: str | Path = "configs/config.yaml") -> MasterConfig:
    """Loads and validates master pipeline configuration."""
    raw = load_yaml(config_path)
    return MasterConfig(**raw)


def load_full_pipeline_config(
    config_path: str | Path = "configs/config.yaml",
    regions_path: str | Path = "configs/regions.yaml",
) -> Tuple[MasterConfig, Dict[str, RegionConfig]]:
    """Loads both the pipeline configuration and regional definitions."""
    cfg = load_config(config_path)
    regions = load_regions(regions_path)
    return cfg, regions
