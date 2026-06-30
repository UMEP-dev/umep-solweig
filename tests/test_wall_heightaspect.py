from pathlib import Path
import sys
import pytest

# Project root for finding /src
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Pointing to the correct filename in your src/umep_solweig/ directory
from umep_solweig.wallalgorithm import processAlgorithm

# FIXED: Look in the tests/ folder instead of the project root
TEST_DIR = Path(__file__).resolve().parent
DSM_PATH = TEST_DIR / "dsm.tif"

# Keep outputs in the root or tests folder (your choice, let's keep them in tests for clean-up)
OUTPUT_HEIGHT_PATH = TEST_DIR / "wall_height_test.tif"
OUTPUT_ASPECT_PATH = TEST_DIR / "wall_aspect_test.tif"


def test_wall_height_algorithm_runs_on_dsm_file():
    if not DSM_PATH.exists():
        pytest.fail(f"Missing test file! It was expected at: {DSM_PATH}")

    result = processAlgorithm(
        dsm_input=str(DSM_PATH),
        wall_limit=3.0,
        use_gpu=False,
        output_height_path=str(OUTPUT_HEIGHT_PATH),
        calculate_aspect=True,
        output_aspect_path=str(OUTPUT_ASPECT_PATH)
    )

    assert result["height"] is not None
    assert result["height"].shape[0] > 0
    assert result["height"].shape[1] > 0
    assert OUTPUT_HEIGHT_PATH.exists()