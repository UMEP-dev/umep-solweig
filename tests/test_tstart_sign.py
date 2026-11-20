"""Ground Tstart is the same with and without a land-cover grid.

Without a land-cover grid the model treats all ground as cobblestone. The
Tstart it passes to the first timestep should then be the value Tgmaps_v1
gives for a cobblestone pixel.
"""

import inspect
import json
from configparser import ConfigParser
from pathlib import Path

import numpy as np
import pytest

from umep_solweig.functions.SOLWEIGpython import Solweig_run as runner
from umep_solweig.functions.SOLWEIGpython.Tgmaps_v1 import Tgmaps_v1

DATA = Path(__file__).resolve().parent / "tests_data"
BUNDLED_PREFIX = "./tests/tests_data/"
PARAMS = json.loads((DATA / "parametersforsolweig.json").read_text())


class _FirstTimestep(Exception):
    """Carries the arguments of the first timestep and ends the run there."""


def _first_timestep_arguments(tmp_path, monkeypatch, landcover):
    config = ConfigParser()
    config.read(DATA / "configsolweig.ini")
    for key, value in config.defaults().items():
        if value.startswith(BUNDLED_PREFIX):
            config["DEFAULT"][key] = str(DATA / value[len(BUNDLED_PREFIX) :])
    config["DEFAULT"]["landcover"] = str(landcover)
    config["DEFAULT"]["groundmodel"] = "0"
    config["DEFAULT"]["outgoinglongwave"] = "0"
    config["DEFAULT"]["output_dir"] = str(tmp_path)
    config["DEFAULT"]["working_dir"] = str(tmp_path)
    path = tmp_path / "configsolweig.ini"
    with open(path, "w") as f:
        config.write(f)

    signature = inspect.signature(runner.so.Solweig_2026a_calc)

    def stop_at_first_timestep(*args, **kwargs):
        raise _FirstTimestep(signature.bind(*args, **kwargs).arguments)

    monkeypatch.setattr(runner.so, "Solweig_2026a_calc", stop_at_first_timestep)
    with pytest.raises(_FirstTimestep) as stopped:
        runner.solweig_run(str(path), None)
    return stopped.value.args[0]


def test_tstart_without_land_cover_is_the_cobblestone_value(tmp_path, monkeypatch):
    arguments = _first_timestep_arguments(tmp_path, monkeypatch, landcover=0)

    cobblestone = np.zeros_like(arguments["dsm"])  # class 0 is cobblestone
    assert PARAMS["Names"]["Value"]["0"] == "Cobble_stone_2014a"
    _, tstart_from_land_cover, *_ = Tgmaps_v1(cobblestone, PARAMS)

    np.testing.assert_array_equal(arguments["Tstart"], tstart_from_land_cover)
