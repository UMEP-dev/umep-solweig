"""Anisotropic sky without vegetation: which emitter each sky patch is given.

`anisotropic_sky` assigns every patch to one emitter from three shadow
matrices: sky where `shmat` and `vegshmat` are 1, vegetation where `vegshmat`
or `vbshvegshmat` is 0, and buildings where `shmat` is 0 and `vbshvegshmat`
is 1. Without vegetation nothing is blocked by it, so both vegetation matrices
must be 1 on every patch. If they are 0, every patch, open sky included, is
treated as vegetation at air temperature and the wall temperature is unused.
"""

import numpy as np
import pytest

from umep_solweig.functions import svf_functions
from umep_solweig.functions.SOLWEIGpython import patch_radiation
from umep_solweig.functions.SOLWEIGpython.anisotropic_sky import anisotropic_sky
from umep_solweig.util.SEBESOLWEIGCommonFiles.create_patches import create_patches
from umep_solweig.util.SEBESOLWEIGCommonFiles.Perez_v3 import Perez_v3

GRID = 60
BESIDE_WALL = (30, 23)  # 2 m west of the west wall
OPEN_GROUND = (5, 5)
SUN_ALTITUDE, SUN_AZIMUTH = 40.0, 250.0  # afternoon sun on the west wall
AIR_TEMPERATURE = 25.0


@pytest.fixture(scope="module")
def svf():
    """One 10 m building on flat ground, no vegetation, wall scheme off."""
    dsm = np.zeros((GRID, GRID))
    dsm[25:35, 25:35] = 10.0
    zeros = np.zeros_like(dsm)
    return svf_functions.svfForProcessing153(dsm, zeros, zeros, 1.0, 0, 1.0, False, None, None)


def _side_longwave(svf, wall_temperature_excess):
    """Run the anisotropic sky and return its side-longwave components."""
    skyvaultalt, skyvaultazi, *_ = create_patches(2)
    patches = np.column_stack([skyvaultalt.ravel(), skyvaultazi.ravel(), np.zeros(skyvaultalt.size)])
    steradians, _, _ = patch_radiation.patch_steradians(patches)
    lv, _, _ = Perez_v3(90.0 - SUN_ALTITUDE, SUN_AZIMUTH, 120.0, 700.0, 180, 1, 2)
    shape = (GRID, GRID)
    zeros = np.zeros(shape)
    result = anisotropic_sky(
        svf["shmat"],
        svf["vegshmat"],
        svf["vbshvegshmat"],
        SUN_ALTITUDE,
        SUN_AZIMUTH,
        np.arccos(np.sqrt(svf["svf"])),  # asvf
        1,  # cyl
        0.75,  # esky
        patches,
        0,  # wallScheme
        None,  # voxelTable
        None,  # voxelMaps
        steradians,
        AIR_TEMPERATURE,
        wall_temperature_excess,  # Tgwall
        0.9,  # ewall
        np.full(shape, 450.0),  # Lup
        700.0,  # radI
        120.0,  # radD
        570.0,  # radG
        lv,
        0.2,  # albedo
        0,  # anisotropic_diffuse
        svf["shmat"],  # diffsh
        np.ones(shape),  # shadow
        zeros,
        zeros,
        zeros,
        zeros,
        0,  # current_step
    )
    names = ("Ldown", "Lside", "Lside_sky", "Lside_veg", "Lside_sh", "Lside_sun", "Lside_ref")
    return dict(zip(names, result))


def test_vegetation_matrices_are_all_visible_without_vegetation(svf):
    assert np.all(svf["vegshmat"] == 1)
    assert np.all(svf["vbshvegshmat"] == 1)


def test_no_patch_is_treated_as_vegetation(svf):
    components = _side_longwave(svf, wall_temperature_excess=10.0)
    assert np.all(components["Lside_veg"] == 0)


def test_open_ground_receives_sky_longwave(svf):
    components = _side_longwave(svf, wall_temperature_excess=10.0)
    assert components["Lside_sky"][OPEN_GROUND] > 0


def test_building_patches_emit_beside_the_wall(svf):
    components = _side_longwave(svf, wall_temperature_excess=10.0)
    from_buildings = components["Lside_sun"] + components["Lside_sh"]
    assert from_buildings[BESIDE_WALL] > 0
    assert from_buildings[BESIDE_WALL] > from_buildings[OPEN_GROUND]


def test_wall_temperature_reaches_the_receptor_beside_the_wall(svf):
    """A warmer sunlit wall must raise the side longwave next to it."""
    cool = _side_longwave(svf, wall_temperature_excess=0.0)["Lside"]
    warm = _side_longwave(svf, wall_temperature_excess=15.0)["Lside"]
    assert warm[BESIDE_WALL] > cool[BESIDE_WALL] + 1.0
