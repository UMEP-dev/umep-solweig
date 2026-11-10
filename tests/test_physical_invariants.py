"""Physical-invariant tests on synthetic scenes.

Each test states a property that must hold for any correct implementation
(shadow geometry, sky view factor bounds, patch solid angles, luminance
normalisation). The scenes are small numpy arrays built in the test, so no
raster files are read and each test runs in well under a second.

Conventions used below: row 0 is the northern edge of the grid, column 0 the
western edge, azimuth is in degrees clockwise from north, and a shadow value
of 1 means sunlit.
"""

import numpy as np
import pytest

from umep_solweig.functions import svf_functions
from umep_solweig.functions.SOLWEIGpython import patch_radiation
from umep_solweig.util import shadowingfunctions
from umep_solweig.util.SEBESOLWEIGCommonFiles.create_patches import create_patches
from umep_solweig.util.SEBESOLWEIGCommonFiles.Perez_v3 import Perez_v3

GRID = 60
BUILDING_HEIGHT = 10.0
BUILDING = (slice(25, 35), slice(25, 35))  # rows 25-34, columns 25-34


def _flat():
    return np.zeros((GRID, GRID))


def _with_building():
    dsm = _flat()
    dsm[BUILDING] = BUILDING_HEIGHT
    return dsm


def _shadow(dsm, azimuth, altitude):
    return shadowingfunctions.shadowingfunctionglobalradiation(dsm, azimuth, altitude, 1.0, None, 1)


@pytest.fixture(scope="module")
def svf_building():
    """SVF and shadow matrices for the single-building scene, without vegetation."""
    dsm = _with_building()
    zeros = np.zeros_like(dsm)
    return svf_functions.svfForProcessing153(dsm, zeros, zeros, 1.0, 0, 1.0, False, None, None)


# ---------------------------------------------------------------------------
# Shadows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("azimuth", [45.0, 135.0, 180.0, 270.0])
@pytest.mark.parametrize("altitude", [10.0, 35.0, 70.0])
def test_flat_ground_is_fully_sunlit(azimuth, altitude):
    sh = _shadow(_flat(), azimuth, altitude)
    assert np.all(sh == 1)


@pytest.mark.parametrize("altitude", [25.0, 30.0, 45.0, 60.0])
def test_shadow_length_follows_sun_altitude(altitude):
    """A wall of height h casts a shadow of length h / tan(altitude).

    The lowest altitude is chosen so the shadow (21 m) still fits in the 25 m
    of ground north of the building.
    """
    sh = _shadow(_with_building(), 180.0, altitude)  # sun due south, shadow to the north
    column = sh[: BUILDING[0].start, 30]  # ground north of the building, through its centre
    shaded_pixels = int((column == 0).sum())
    expected = BUILDING_HEIGHT / np.tan(np.radians(altitude))
    assert abs(shaded_pixels - expected) <= 1.5


@pytest.mark.parametrize(
    ("azimuth", "shaded_side", "sunlit_side"),
    [
        (180.0, (20, 30), (40, 30)),  # sun south: shade north of the building
        (90.0, (30, 20), (30, 40)),  # sun east: shade west
        (270.0, (30, 40), (30, 20)),  # sun west: shade east
    ],
)
def test_shadow_falls_on_the_side_away_from_the_sun(azimuth, shaded_side, sunlit_side):
    sh = _shadow(_with_building(), azimuth, 30.0)
    assert sh[shaded_side] == 0
    assert sh[sunlit_side] == 1


def test_lower_sun_never_shrinks_the_shadow():
    dsm = _with_building()
    high = _shadow(dsm, 180.0, 60.0)
    low = _shadow(dsm, 180.0, 25.0)
    assert np.all(low <= high)


# ---------------------------------------------------------------------------
# Sky view factor
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def svf_flat():
    flat = _flat()
    return svf_functions.svfForProcessing153(flat, flat, flat, 1.0, 0, 1.0, False, None, None)


def test_flat_ground_sees_the_whole_sky(svf_flat):
    np.testing.assert_allclose(svf_flat["svf"], 1.0, atol=1e-6)
    assert np.all(svf_flat["shmat"] == 1)


_HALF_SKY_XFAIL = pytest.mark.xfail(
    strict=True,
    reason=(
        "Directional SVF on flat ground is N 0.976, E 1.000, S 0.948, W 0.924. "
        "The half-sky weight assumes ceil(n / 2) patches of each altitude band "
        "fall in every half-plane, which the band azimuth offsets do not give."
    ),
)


@pytest.mark.parametrize(
    "key",
    [
        pytest.param("svfN", marks=_HALF_SKY_XFAIL),
        "svfE",
        pytest.param("svfS", marks=_HALF_SKY_XFAIL),
        pytest.param("svfW", marks=_HALF_SKY_XFAIL),
    ],
)
def test_flat_ground_sees_the_whole_half_sky_in_every_direction(svf_flat, key):
    np.testing.assert_allclose(svf_flat[key], 1.0, atol=0.01)


def test_svf_is_bounded(svf_building):
    for key in ("svf", "svfE", "svfS", "svfW", "svfN"):
        assert svf_building[key].min() >= 0.0, key
        assert svf_building[key].max() <= 1.0 + 1e-9, key


def test_svf_drops_beside_a_building(svf_building):
    svf = svf_building["svf"]
    beside_wall = svf[30, 23]  # 2 m west of the west wall
    open_ground = svf[5, 5]
    assert beside_wall < open_ground - 0.1
    assert open_ground > 0.9


def test_directional_svf_is_lowest_towards_the_wall(svf_building):
    """West of the building the eastern half-sky is the obstructed one."""
    r, c = 30, 23
    assert svf_building["svfE"][r, c] < svf_building["svfW"][r, c]
    # and the mirror-image pixel east of the building
    r, c = 30, 36
    assert svf_building["svfW"][r, c] < svf_building["svfE"][r, c]


def test_shadow_matrix_blocks_more_patches_closer_to_the_building(svf_building):
    shmat = svf_building["shmat"]
    assert set(np.unique(shmat)) <= {0.0, 1.0}
    blocked_beside_wall = int((shmat[30, 23] == 0).sum())
    blocked_far_corner = int((shmat[5, 5] == 0).sum())
    assert blocked_beside_wall > 10
    assert blocked_far_corner < blocked_beside_wall


# ---------------------------------------------------------------------------
# Sky patches
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("patch_option", [1, 2, 3])
def test_patch_solid_angles_sum_to_the_hemisphere(patch_option):
    skyvaultalt, skyvaultazi, *_ = create_patches(patch_option)
    patches = np.column_stack([skyvaultalt.ravel(), skyvaultazi.ravel(), np.zeros(skyvaultalt.size)])
    steradians, _, _ = patch_radiation.patch_steradians(patches)
    assert np.all(steradians > 0)
    np.testing.assert_allclose(steradians.sum(), 2 * np.pi, rtol=0.01)


@pytest.mark.parametrize(
    ("zenith", "rad_d", "rad_i"),
    [(30.0, 120.0, 800.0), (60.0, 150.0, 600.0), (50.0, 250.0, 0.0)],
)
def test_perez_luminance_is_normalised(zenith, rad_d, rad_i):
    lv, _, _ = Perez_v3(zenith, 180.0, rad_d, rad_i, 180, 1, 2)
    assert np.all(np.isfinite(lv))
    assert np.all(lv[:, 2] >= 0)
    np.testing.assert_allclose(lv[:, 2].sum(), 1.0, rtol=1e-6)


def test_perez_clear_sky_is_brightest_near_the_sun():
    sun_azimuth, zenith = 180.0, 40.0
    lv, _, _ = Perez_v3(zenith, sun_azimuth, 100.0, 800.0, 180, 1, 2)
    brightest = lv[np.argmax(lv[:, 2])]
    assert abs(brightest[0] - (90.0 - zenith)) <= 15.0  # altitude within one patch band
    azimuth_gap = abs((brightest[1] - sun_azimuth + 180.0) % 360.0 - 180.0)
    assert azimuth_gap <= 30.0
