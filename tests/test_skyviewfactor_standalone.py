from pathlib import Path

import numpy as np
from osgeo import gdal

from umep_solweig.skyviewfactor import processAlgorithm


def test_process_algorithm_uses_sample_rasters():
    test_dir = Path(__file__).resolve().parent
    data_dir = test_dir / "tests_data"
    output_dir = test_dir / "tests_out"
    output_file = output_dir / "skyviewfactor_standalone.tif"

    output_dir.mkdir(parents=True, exist_ok=True)
    if output_file.exists():
        output_file.unlink()

    result = processAlgorithm(
        input_dsm=data_dir / "dsm.tif",
        input_cdsm=data_dir / "cdsm.tif",
        wallScheme=True,
        kmeans=True,
        clusters=5,
        input_dem=data_dir / "dem.tif",
        svf_height=2.0,
        outputDir=str(output_dir),
        outputFile=str(output_file),
    )

    assert isinstance(result, dict)
    assert result["output_file"] == str(output_file)
    assert output_file.exists()
    assert isinstance(result["svf"], np.ndarray)

    dsm_dataset = gdal.Open(str(data_dir / "dsm.tif"))
    assert dsm_dataset is not None
    dsm_array = dsm_dataset.ReadAsArray().astype(float)
    assert result["svf"].shape == dsm_array.shape
    assert np.isfinite(result["svf"]).all()
    assert result["svf"].min() >= 0.0
    assert result["svf"].max() <= 1.0
