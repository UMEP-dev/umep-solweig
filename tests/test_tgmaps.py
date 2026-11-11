"""Land-cover parameter grids from Tgmaps_v1."""

import json
import time
from pathlib import Path

import numpy as np
import pytest

from umep_solweig.functions.SOLWEIGpython.Tgmaps_v1 import Tgmaps_v1

PARAMS = json.loads(
    (Path(__file__).parent / "tests_data" / "solweig_parameters.json").read_text()
)
NAMES = PARAMS["Names"]["Value"]


def _expected(table, grid):
    return np.array([[table[NAMES[str(int(c))]] for c in row] for row in grid])


@pytest.mark.parametrize(
    "grid",
    [
        [[0.0, 7.0, 5.0]],
        [[7.0, 0.0, 5.0]],  # water before cobblestone
        [[7.0, 6.0, 5.0, 2.0, 1.0, 0.0]],
    ],
)
def test_each_class_gets_its_own_parameters_in_any_order(grid):
    lc = np.array(grid)
    TgK, Tstart, alb, emis, _, _, TmaxLST, _ = Tgmaps_v1(lc.copy(), PARAMS)
    np.testing.assert_array_equal(TgK, _expected(PARAMS["Ts_deg"]["Value"], grid))
    np.testing.assert_array_equal(Tstart, _expected(PARAMS["Tstart"]["Value"], grid))
    np.testing.assert_array_equal(
        alb, _expected(PARAMS["Albedo"]["Effective"]["Value"], grid)
    )
    np.testing.assert_array_equal(
        emis, _expected(PARAMS["Emissivity"]["Value"], grid)
    )
    np.testing.assert_array_equal(
        TmaxLST, _expected(PARAMS["TmaxLST"]["Value"], grid)
    )


def test_wall_classes_map_to_roofs():
    lc = np.array([[100.0, 101.0, 2.0]])
    TgK, *_ = Tgmaps_v1(lc.copy(), PARAMS)
    roofs = PARAMS["Ts_deg"]["Value"][NAMES["2"]]
    np.testing.assert_array_equal(TgK, np.full((1, 3), roofs))


def test_run_time_does_not_grow_with_the_square_of_the_pixel_count():
    lc = np.zeros((400, 400))
    lc[:200] = 5.0
    start = time.perf_counter()
    Tgmaps_v1(lc, PARAMS)
    assert time.perf_counter() - start < 1.0  # was about 30 s with one pass per pixel
