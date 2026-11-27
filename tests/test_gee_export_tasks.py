
'''Tests for the non-dry-run Earth Engine export workflow using batch export tasks.'''

from pathlib import Path
from unittest.mock import MagicMock, patch

from src.gee_export import fetch_and_export_region
from src.config import load_full_pipeline_config


def test_fetch_and_export_region_batch_tasks(tmp_path: Path):
    """Verify that fetch_and_export_region creates EE batch export tasks."""

    cfg, regions = load_full_pipeline_config(
        "configs/config.yaml",
        "configs/regions.yaml",
    )

    exported_calls = []

    def mock_toDrive(**kwargs):
        exported_calls.append(kwargs)
        task = MagicMock()
        task.id = f"task_{kwargs.get('fileNamePrefix')}"
        task.description = kwargs.get("description")
        return task

    mock_init = patch(
        "src.gee_export.initialize_earth_engine",
        return_value=(True, ""),
    )
    mock_bbox = patch(
        "src.gee_export.ee.Geometry.BBox",
        return_value="dummy_bbox",
    )
    mock_export = patch(
        "src.gee_export.ee.batch.Export.image.toDrive",
        side_effect=mock_toDrive,
    )
    mock_sentinel = patch(
        "src.gee_export.build_sentinel2_composite",
        return_value=MagicMock(name="sentinel_image"),
    )
    mock_dynamic = patch(
        "src.gee_export.build_dynamic_world_forest_mask",
        return_value=MagicMock(name="dynamic_world_image"),
    )

    with mock_init, mock_bbox, mock_export, mock_sentinel, mock_dynamic:
        manifest = fetch_and_export_region(
            region_id="amazon_brazil",
            config=cfg,
            regions=regions,
            output_dir=tmp_path,
            dry_run=False,
        )

    assert manifest.get("status") == "export_tasks_created"

    tasks = manifest.get("tasks", {})
    expected_keys = {"t1_raster", "t2_raster", "t1_mask", "t2_mask"}

    assert set(tasks.keys()) == expected_keys

    for key, info in tasks.items():
        assert info["task_id"] is not None
        assert info["description"] is not None

    prefixes = {call["fileNamePrefix"] for call in exported_calls}

    expected_prefixes = {
        f"amazon_brazil_s2_{regions['amazon_brazil'].target_years.t1}",
        f"amazon_brazil_s2_{regions['amazon_brazil'].target_years.t2}",
        f"amazon_brazil_dw_mask_{regions['amazon_brazil'].target_years.t1}",
        f"amazon_brazil_dw_mask_{regions['amazon_brazil'].target_years.t2}",
    }

    assert prefixes == expected_prefixes
    assert len(exported_calls) == 4
