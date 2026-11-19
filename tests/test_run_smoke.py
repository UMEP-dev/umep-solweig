"""The model runs end to end on the bundled test rasters, without torch.

The sky view factors are computed first with the package's own SVF
algorithm and then passed to the model, as a user would do.
"""

from configparser import ConfigParser
from pathlib import Path

import numpy as np
from osgeo import gdal

from umep_solweig.skyviewfactor import processAlgorithm as sky_view_factor
from umep_solweig.solweig_run import solweig_run

DATA = Path(__file__).resolve().parent / "tests_data"
BUNDLED_PREFIX = "./tests/tests_data/"


def _config_with_absolute_paths(tmp_path, svf_zip):
    """Copy the bundled config, so the run does not depend on the working
    directory and writes its outputs to a temporary folder."""
    config = ConfigParser()
    config.read(DATA / "configsolweig.ini")
    for key, value in config.defaults().items():
        if value.startswith(BUNDLED_PREFIX):
            config["DEFAULT"][key] = str(DATA / value[len(BUNDLED_PREFIX) :])
    config["DEFAULT"]["input_svf"] = str(svf_zip)
    # The 2026a ground scheme is switched off here. With these inputs it
    # gives a median upwelling longwave of about 930 W m-2 at noon and
    # 525 W m-2 at 01:00, against 430 and 363 W m-2 without it.
    config["DEFAULT"]["groundmodel"] = "0"
    config["DEFAULT"]["outgoinglongwave"] = "0"
    config["DEFAULT"]["output_dir"] = str(tmp_path)
    config["DEFAULT"]["working_dir"] = str(tmp_path)
    path = tmp_path / "configsolweig.ini"
    with open(path, "w") as f:
        config.write(f)
    return path


def test_svf_then_run_writes_a_plausible_mean_radiant_temperature(tmp_path):
    svf_dir = tmp_path / "svf"
    svf_dir.mkdir()
    sky_view_factor(
        input_dsm=DATA / "dsm.tif",
        input_cdsm=DATA / "cdsm.tif",
        use_gpu=False,
        outputDir=str(svf_dir),
        outputFile=str(svf_dir / "svf.tif"),
    )
    svf_zip = svf_dir / "svfs.zip"
    assert svf_zip.exists()

    solweig_run(str(_config_with_absolute_paths(tmp_path, svf_zip)), None)

    out = tmp_path / "Tmrt_average.tif"
    assert out.exists()
    tmrt = gdal.Open(str(out)).ReadAsArray()
    dsm = gdal.Open(str(DATA / "dsm.tif")).ReadAsArray()

    assert tmrt.shape == dsm.shape
    assert np.isfinite(tmrt).all()
    # Degrees Celsius, averaged over 23 May 2010 in Gothenburg (air
    # temperature 13 to 20 degrees). The run gives 16 to 33.
    assert tmrt.min() > 0.0
    assert tmrt.max() < 60.0
    # Sunlit and shaded ground differ, so the field is not constant.
    assert tmrt.max() - tmrt.min() > 1.0
