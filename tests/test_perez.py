"""Perez sky model: clearness, optical air mass and clearness bins.

References:
    Perez, R., Seals, R., Michalsky, J. (1993). All-weather model for sky
    luminance distribution. Solar Energy 50(3), 235-245.
    Kasten, F., Young, A. T. (1989). Revised optical air mass tables and
    approximation formula. Applied Optics 28(22), 4735-4738.
"""

import numpy as np
import pytest

from umep_solweig.util.SEBESOLWEIGCommonFiles.Perez_v3 import Perez_v3

JDAY = 180


def _extraterrestrial(jday):
    day_angle = jday * 2 * np.pi / 365
    return 1367 * (
        1.00011
        + 0.034221 * np.cos(day_angle)
        + 0.00128 * np.sin(day_angle)
        + 0.000719 * np.cos(2 * day_angle)
        + 0.000077 * np.sin(2 * day_angle)
    )


@pytest.mark.parametrize("zenith", [20.0, 50.0, 80.0])
def test_clearness_is_one_for_an_overcast_sky(zenith):
    """With no direct beam the clearness is 1 at every zenith angle."""
    _, clearness, _ = Perez_v3(zenith, 180.0, 150.0, 0.0, JDAY, 1, 2)
    assert clearness == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize(
    ("zenith", "rad_d", "rad_i"),
    [(30.0, 120.0, 800.0), (60.0, 150.0, 600.0), (80.0, 60.0, 400.0)],
)
def test_clearness_matches_the_published_formula(zenith, rad_d, rad_i):
    z = np.radians(zenith)
    expected = ((rad_d + rad_i) / rad_d + 1.041 * z**3) / (1 + 1.041 * z**3)
    _, clearness, _ = Perez_v3(zenith, 180.0, rad_d, rad_i, JDAY, 1, 2)
    assert clearness == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("altitude", [1.0, 2.0, 5.0, 9.0])
def test_low_sun_air_mass_follows_kasten_and_young(altitude):
    """Brightness is air mass * diffuse / extraterrestrial, so it gives the air mass."""
    rad_d = 60.0
    expected = 1 / (np.sin(np.radians(altitude)) + 0.50572 * (altitude + 6.07995) ** -1.6364)
    _, _, brightness = Perez_v3(90.0 - altitude, 180.0, rad_d, 200.0, JDAY, 1, 2)
    air_mass = brightness * _extraterrestrial(JDAY) / rad_d
    assert air_mass == pytest.approx(expected, rel=1e-6)
    assert air_mass < 1 / np.sin(np.radians(altitude))  # refraction shortens the path


def test_air_mass_is_continuous_across_ten_degrees():
    rad_d = 60.0
    below = Perez_v3(90.0 - 9.999, 180.0, rad_d, 200.0, JDAY, 1, 2)[2]
    above = Perez_v3(90.0 - 10.001, 180.0, rad_d, 200.0, JDAY, 1, 2)[2]
    assert below == pytest.approx(above, rel=0.05)


@pytest.mark.parametrize("edge", [1.065, 1.230, 1.500, 1.950, 2.800, 4.500, 6.200])
def test_clearness_on_a_bin_edge_is_assigned_a_bin(edge):
    """At zenith 0 the clearness is (Idh + Ibn) / Idh, so the edge can be hit exactly."""
    rad_d = 1000.0
    rad_i = edge * rad_d - rad_d
    lv, clearness, _ = Perez_v3(0.0, 180.0, rad_d, rad_i, JDAY, 1, 2)
    assert clearness == pytest.approx(edge, rel=1e-12)
    assert np.all(np.isfinite(lv))


def test_zero_diffuse_radiation_does_not_fail():
    lv, _, brightness = Perez_v3(40.0, 180.0, 0.0, 700.0, JDAY, 1, 2)
    assert brightness == 0
    assert np.all(np.isfinite(lv))
    np.testing.assert_allclose(lv[:, 2].sum(), 1.0, rtol=1e-6)
