import torch

""" This function calculates whether a point is sunlit or shaded
    based on a sky view factor (in a cylinder), solar altitude, solar azimuth """


def shaded_or_sunlit(
    solar_altitude, solar_azimuth, patch_altitude, patch_azimuth, asvf
):
    device = asvf.device
    solar_altitude = torch.as_tensor(solar_altitude, device=device)
    solar_azimuth = torch.as_tensor(solar_azimuth, device=device)
    patch_altitude = torch.as_tensor(patch_altitude, device=device)
    patch_azimuth = torch.as_tensor(patch_azimuth, device=device)

    # Patch azimuth in relation to sun azimuth
    patch_to_sun_azi = torch.abs(solar_azimuth - patch_azimuth)

    # Degrees to radians
    deg2rad = torch.pi / 180

    # Radians to degrees
    rad2deg = 180 / torch.pi

    #
    xi = torch.cos(patch_to_sun_azi * deg2rad)

    #
    yi = 2 * xi * torch.tan(solar_altitude * deg2rad)

    hsvf = torch.tan(asvf)

    if yi > 0:
        yi_ = 0
    else:
        yi_ = yi

    #
    tan_delta = hsvf + yi_

    # Degrees where below is in shade and above is sunlit
    sunlit_degrees = torch.arctan(tan_delta) * rad2deg

    # Boolean for pixels where patch is sunlit
    sunlit_patches = sunlit_degrees < patch_altitude
    # Boolean for pixels where patch is shaded
    shaded_patches = sunlit_degrees > patch_altitude

    return sunlit_patches, shaded_patches