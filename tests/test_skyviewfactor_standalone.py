from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

import numpy as np
from osgeo import gdal

from umep_solweig.skyviewfactor import processAlgorithm


@pytest.fixture
def test_paths():
    """Fixture to manage paths cleanly across tests."""
    test_dir = Path(__file__).resolve().parent
    return {
        "data_dir": test_dir / "tests_data",
        "output_dir": test_dir / "tests_out",
        "output_file": test_dir / "tests_out" / "skyviewfactor_standalone.tif",
    }


@pytest.fixture(autouse=True)
def setup_and_teardown(test_paths):
    """Automatically cleans up outputs before and after tests."""
    test_paths["output_dir"].mkdir(parents=True, exist_ok=True)
    if test_paths["output_file"].exists():
        test_paths["output_file"].unlink()
    yield
    if test_paths["output_file"].exists():
        test_paths["output_file"].unlink()


# ----------------------------------------------------------------------
# 1. Integration / Functional Tests
# ----------------------------------------------------------------------


@pytest.mark.parametrize("use_gpu", [False, True])
def test_process_algorithm_standard_and_wall_scheme(test_paths, use_gpu):
    """Tests the function with standard parameters and wallScheme enabled on both CPU and GPU."""
    # Patch torch.cuda.is_available to match our current parametrization matrix
    with patch("torch.cuda.is_available", return_value=use_gpu):
        result = processAlgorithm(
            input_dsm=test_paths["data_dir"] / "dsm.tif",
            input_cdsm=test_paths["data_dir"] / "cdsm.tif",
            wallScheme=True,
            kmeans=True,
            clusters=5,
            input_dem=test_paths["data_dir"] / "dem.tif",
            svf_height=2.0,
            use_gpu=use_gpu,
            outputDir=str(test_paths["output_dir"]),
            outputFile=str(test_paths["output_file"]),
        )

    assert isinstance(result, dict)
    assert result["output_file"] == str(test_paths["output_file"])
    assert test_paths["output_file"].exists()
    assert isinstance(result["svf"], np.ndarray)

    dsm_dataset = gdal.Open(str(test_paths["data_dir"] / "dsm.tif"))
    assert dsm_dataset is not None
    dsm_array = dsm_dataset.ReadAsArray().astype(float)

    assert result["svf"].shape == dsm_array.shape
    assert np.isfinite(result["svf"]).all()
    assert result["svf"].min() >= 0.0
    assert result["svf"].max() <= 1.0


# @pytest.mark.parametrize("use_gpu", [False, True])
# def test_process_algorithm_minimal_inputs(test_paths, use_gpu):
#     """Tests execution with only mandatory inputs on CPU and GPU pipelines."""
#     with patch("torch.cuda.is_available", return_value=use_gpu):
#         result = processAlgorithm(
#             input_dsm=test_paths["data_dir"] / "dsm.tif",
#             use_gpu=use_gpu,
#             outputDir=str(test_paths["output_dir"]),
#             outputFile=str(test_paths["output_file"]),
#         )
#     assert isinstance(result, dict)
#     assert test_paths["output_file"].exists()
#     assert "svf" in result


# # ----------------------------------------------------------------------
# # 2. Parameter Matrix & Edge Cases
# # ----------------------------------------------------------------------


# @pytest.mark.parametrize("use_gpu", [False, True])
# @pytest.mark.parametrize("transVeg", [0, 50, 100])
# @pytest.mark.parametrize("aniso", [True, False])
# def test_process_algorithm_vegetation_and_shading_variants(
#     test_paths, use_gpu, transVeg, aniso
# ):
#     """Verifies transmissivity and shadow schemes across CPU and GPU configurations."""
#     with patch("torch.cuda.is_available", return_value=use_gpu):
#         result = processAlgorithm(
#             input_dsm=test_paths["data_dir"] / "dsm.tif",
#             input_cdsm=test_paths["data_dir"] / "cdsm.tif",
#             transVeg=transVeg,
#             aniso=aniso,
#             wallScheme=False,
#             use_gpu=use_gpu,
#             outputDir=str(test_paths["output_dir"]),
#             outputFile=str(test_paths["output_file"]),
#         )
#     assert test_paths["output_file"].exists()


# @pytest.mark.parametrize("use_gpu", [False, True])
# def test_process_algorithm_kmeans_disabled(test_paths, use_gpu):
#     """Verifies behavior when K-Means wall parameterization is turned off for CPU/GPU."""
#     with patch("torch.cuda.is_available", return_value=use_gpu):
#         result = processAlgorithm(
#             input_dsm=test_paths["data_dir"] / "dsm.tif",
#             input_dem=test_paths["data_dir"] / "dem.tif",
#             wallScheme=True,
#             kmeans=False,
#             use_gpu=use_gpu,
#             outputDir=str(test_paths["output_dir"]),
#             outputFile=str(test_paths["output_file"]),
#         )
#     assert test_paths["output_file"].exists()


# # ----------------------------------------------------------------------
# # 3. Code Path Execution via Mocking
# # ----------------------------------------------------------------------


# @pytest.mark.parametrize("use_gpu", [False, True])
# def test_process_algorithm_with_progress_feedback(test_paths, use_gpu):
#     """Passes a mock feedback object to ensure progress tracking is architecture independent."""
#     mock_feedback = MagicMock()

#     with patch("torch.cuda.is_available", return_value=use_gpu):
#         processAlgorithm(
#             input_dsm=test_paths["data_dir"] / "dsm.tif",
#             use_gpu=use_gpu,
#             outputDir=str(test_paths["output_dir"]),
#             outputFile=str(test_paths["output_file"]),
#             feedback=mock_feedback,
#         )
#     assert test_paths["output_file"].exists()


# # ----------------------------------------------------------------------
# # 4. Error Handling & Validation Tests
# # ----------------------------------------------------------------------


# @pytest.mark.parametrize("use_gpu", [False, True])
# def test_process_algorithm_missing_dem_for_wallscheme(test_paths, use_gpu):
#     """Ensures a ValueError/Exception is thrown regardless of hardware target when missing a DEM."""
#     with patch("torch.cuda.is_available", return_value=use_gpu):
#         with pytest.raises((ValueError, RuntimeError, TypeError)):
#             processAlgorithm(
#                 input_dsm=test_paths["data_dir"] / "dsm.tif",
#                 wallScheme=True,
#                 input_dem=None,
#                 use_gpu=use_gpu,
#                 outputDir=str(test_paths["output_dir"]),
#                 outputFile=str(test_paths["output_file"]),
#             )


# @pytest.mark.parametrize("use_gpu", [False, True])
# def test_process_algorithm_invalid_input_path(test_paths, use_gpu):
#     """Ensures input validation failures crash gracefully on both CPU and GPU execution loops."""
#     with patch("torch.cuda.is_available", return_value=use_gpu):
#         with pytest.raises((FileNotFoundError, Exception)):
#             processAlgorithm(
#                 input_dsm=test_paths["data_dir"] / "non_existent_file.tif",
#                 use_gpu=use_gpu,
#                 outputDir=str(test_paths["output_dir"]),
#                 outputFile=str(test_paths["output_file"]),
#             )
