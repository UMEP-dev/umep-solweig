import numpy as np
import torch

def shortwave_from_sky(
    sky, angle_of_incidence, lumChi, steradian, patch_azimuth, cyl
):
    """Calculates the amount of diffuse shortwave radiation from the sky for a patch with:
    angle of incidence = angle_of_incidence
    luminance = lumChi
    steradian = steradian"""

    # Diffuse vertical radiation
    diffuse_shortwave_radiation = sky * lumChi * angle_of_incidence * steradian

    return diffuse_shortwave_radiation


def longwave_from_sky(sky, Lsky_side, Lsky_down, patch_azimuth):
    device = sky.device

    # Degrees to radians
    deg2rad = torch.pi / 180

    # Longwave radiation from sky to vertical surface
    Ldown_sky = sky * Lsky_down

    # Longwave radiation from sky to horizontal surface
    Lside_sky = sky * Lsky_side

    #
    Least = torch.zeros((sky.shape[0], sky.shape[1]), device=device)
    Lsouth = torch.zeros((sky.shape[0], sky.shape[1]), device=device)
    Lwest = torch.zeros((sky.shape[0], sky.shape[1]), device=device)
    Lnorth = torch.zeros((sky.shape[0], sky.shape[1]), device=device)

    # Portion into cardinal directions to be used for standing box or POI output
    if (patch_azimuth > 360) or (patch_azimuth < 180):
        Least = sky * Lsky_side * torch.cos(torch.as_tensor((90 - patch_azimuth) * deg2rad, device=device))
    if (patch_azimuth > 90) and (patch_azimuth < 270):
        Lsouth = sky * Lsky_side * torch.cos(torch.as_tensor((180 - patch_azimuth) * deg2rad, device=device))
    if (patch_azimuth > 180) and (patch_azimuth < 360):
        Lwest = sky * Lsky_side * torch.cos(torch.as_tensor((270 - patch_azimuth) * deg2rad, device=device))
    if (patch_azimuth > 270) or (patch_azimuth < 90):
        Lnorth = sky * Lsky_side * torch.cos(torch.as_tensor((0 - patch_azimuth) * deg2rad, device=device))

    return Lside_sky, Ldown_sky, Least, Lsouth, Lwest, Lnorth


def longwave_from_veg(
    vegetation, steradian, angle_of_incidence, angle_of_incidence_h,
    patch_altitude, patch_azimuth, ewall, Ta,
):
    device = vegetation.device
    SBC = 5.67051e-8
    deg2rad = torch.pi / 180

    vegetation_surface = (ewall * SBC * ((Ta + 273.15) ** 4)) / torch.pi

    Lside_veg = vegetation_surface * steradian * angle_of_incidence * vegetation
    Ldown_veg = vegetation_surface * steradian * angle_of_incidence_h * vegetation

    Least = torch.zeros((vegetation.shape[0], vegetation.shape[1]), device=device)
    Lsouth = torch.zeros((vegetation.shape[0], vegetation.shape[1]), device=device)
    Lwest = torch.zeros((vegetation.shape[0], vegetation.shape[1]), device=device)
    Lnorth = torch.zeros((vegetation.shape[0], vegetation.shape[1]), device=device)

    patch_altitude_rad = torch.as_tensor(patch_altitude * deg2rad, device=device)

    if (patch_azimuth > 360) or (patch_azimuth < 180):
        Least = vegetation_surface * steradian * torch.cos(patch_altitude_rad) * vegetation * torch.cos(torch.as_tensor((90 - patch_azimuth) * deg2rad, device=device))
    if (patch_azimuth > 90) and (patch_azimuth < 270):
        Lsouth = vegetation_surface * steradian * torch.cos(patch_altitude_rad) * vegetation * torch.cos(torch.as_tensor((180 - patch_azimuth) * deg2rad, device=device))
    if (patch_azimuth > 180) and (patch_azimuth < 360):
        Lwest = vegetation_surface * steradian * torch.cos(patch_altitude_rad) * vegetation * torch.cos(torch.as_tensor((270 - patch_azimuth) * deg2rad, device=device))
    if (patch_azimuth > 270) or (patch_azimuth < 90):
        Lnorth = vegetation_surface * steradian * torch.cos(patch_altitude_rad) * vegetation * torch.cos(torch.as_tensor((0 - patch_azimuth) * deg2rad, device=device))

    return Lside_veg, Ldown_veg, Least, Lsouth, Lwest, Lnorth


def longwave_from_buildings(
    building, steradian, angle_of_incidence, angle_of_incidence_h, patch_azimuth,
    sunlit_patches, shaded_patches, azimuth_difference, solar_altitude,
    ewall, Ta, Tgwall,
):
    device = building.device
    SBC = 5.67051e-8
    deg2rad = torch.pi / 180

    Least = torch.zeros((building.shape[0], building.shape[1]), device=device)
    Lsouth = torch.zeros((building.shape[0], building.shape[1]), device=device)
    Lwest = torch.zeros((building.shape[0], building.shape[1]), device=device)
    Lnorth = torch.zeros((building.shape[0], building.shape[1]), device=device)

    sunlit_surface = (ewall * SBC * ((Ta + Tgwall + 273.15) ** 4)) / torch.pi
    shaded_surface = (ewall * SBC * ((Ta + 273.15) ** 4)) / torch.pi

    if (azimuth_difference > 90) and (azimuth_difference < 270) and (solar_altitude > 0):
        Lside_sun = sunlit_surface * sunlit_patches * steradian * angle_of_incidence * building
        Lside_sh = shaded_surface * shaded_patches * steradian * angle_of_incidence * building
        Ldown_sun = sunlit_surface * sunlit_patches * steradian * angle_of_incidence_h * building
        Ldown_sh = shaded_surface * shaded_patches * steradian * angle_of_incidence_h * building

        if (patch_azimuth > 360) or (patch_azimuth < 180):
            Least = sunlit_surface * sunlit_patches * steradian * angle_of_incidence * building * torch.cos(torch.as_tensor((90 - patch_azimuth) * deg2rad, device=device))
            Least = Least + shaded_surface * shaded_patches * steradian * angle_of_incidence * building * torch.cos(torch.as_tensor((90 - patch_azimuth) * deg2rad, device=device))
        if (patch_azimuth > 90) and (patch_azimuth < 270):
            Lsouth = sunlit_surface * sunlit_patches * steradian * angle_of_incidence * building * torch.cos(torch.as_tensor((180 - patch_azimuth) * deg2rad, device=device))
            Lsouth = Lsouth + shaded_surface * shaded_patches * steradian * angle_of_incidence * building * torch.cos(torch.as_tensor((180 - patch_azimuth) * deg2rad, device=device))
        if (patch_azimuth > 180) and (patch_azimuth < 360):
            Lwest = sunlit_surface * sunlit_patches * steradian * angle_of_incidence * building * torch.cos(torch.as_tensor((270 - patch_azimuth) * deg2rad, device=device))
            Lwest = Lwest + shaded_surface * shaded_patches * steradian * angle_of_incidence * building * torch.cos(torch.as_tensor((270 - patch_azimuth) * deg2rad, device=device))
        if (patch_azimuth > 270) or (patch_azimuth < 90):
            Lnorth = sunlit_surface * sunlit_patches * steradian * angle_of_incidence * building * torch.cos(torch.as_tensor((0 - patch_azimuth) * deg2rad, device=device))
            Lnorth = Lnorth + shaded_surface * shaded_patches * steradian * angle_of_incidence * building * torch.cos(torch.as_tensor((0 - patch_azimuth) * deg2rad, device=device))
    else:
        Lside_sh = shaded_surface * steradian * angle_of_incidence * building
        Lside_sun = torch.zeros((Lside_sh.shape[0], Lside_sh.shape[1]), device=device)
        Ldown_sh = shaded_surface * steradian * angle_of_incidence_h * building
        Ldown_sun = torch.zeros((Lside_sh.shape[0], Lside_sh.shape[1]), device=device)

        if (patch_azimuth > 360) or (patch_azimuth < 180):
            Least = shaded_surface * steradian * angle_of_incidence * building * torch.cos(torch.as_tensor((90 - patch_azimuth) * deg2rad, device=device))
        if (patch_azimuth > 90) and (patch_azimuth < 270):
            Lsouth = shaded_surface * steradian * angle_of_incidence * building * torch.cos(torch.as_tensor((180 - patch_azimuth) * deg2rad, device=device))
        if (patch_azimuth > 180) and (patch_azimuth < 360):
            Lwest = shaded_surface * steradian * angle_of_incidence * building * torch.cos(torch.as_tensor((270 - patch_azimuth) * deg2rad, device=device))
        if (patch_azimuth > 270) or (patch_azimuth < 90):
            Lnorth = shaded_surface * steradian * angle_of_incidence * building * torch.cos(torch.as_tensor((0 - patch_azimuth) * deg2rad, device=device))

    return Lside_sun, Lside_sh, Ldown_sun, Ldown_sh, Least, Lsouth, Lwest, Lnorth

def longwave_from_buildings_wallScheme(
    voxelMaps, voxelTable, steradian, angle_of_incidence, angle_of_incidence_h, patch_azimuth,
):
    device = voxelMaps.device
    deg2rad = torch.pi / 180
    shape = (voxelMaps.shape[0], voxelMaps.shape[1])

    Lside = torch.zeros(shape, device=device)
    Lside_sh = torch.zeros(shape, device=device)
    Ldown = torch.zeros(shape, device=device)
    Ldown_sh = torch.zeros(shape, device=device)
    Least = torch.zeros(shape, device=device)
    Lsouth = torch.zeros(shape, device=device)
    Lwest = torch.zeros(shape, device=device)
    Lnorth = torch.zeros(shape, device=device)

    voxel_ids = torch.unique(voxelMaps)
    ids = voxel_ids[1:].long()
    max_id = int(voxel_ids.max().item())
    lookup = torch.zeros(max_id + 1, device=device, dtype=torch.float64)
    if ids.numel() > 0:
        # was: a Python loop doing one voxelTable.loc[int(i), ...] call per id -
        # replaced with a single vectorized .loc[] call for every id at once.
        # Verified pandas preserves the requested id order in the result, so
        # this aligns correctly with `ids` for the lookup[ids] = values below.
        ids_np = ids.cpu().numpy()
        values_np = voxelTable.loc[ids_np, "LongwaveRadiation"].to_numpy()
        values = torch.tensor(values_np, device=device, dtype=torch.float64)
        lookup[ids] = values
    patch_radiation = lookup[voxelMaps.long()]

    Lside = Lside + patch_radiation * steradian * angle_of_incidence
    Ldown = Ldown + patch_radiation * steradian * angle_of_incidence_h

    if (patch_azimuth > 360) or (patch_azimuth < 180):
        Least = patch_radiation * steradian * angle_of_incidence * torch.cos(torch.as_tensor((90 - patch_azimuth) * deg2rad, device=device))
    if (patch_azimuth > 90) and (patch_azimuth < 270):
        Lsouth = patch_radiation * steradian * angle_of_incidence * torch.cos(torch.as_tensor((180 - patch_azimuth) * deg2rad, device=device))
    if (patch_azimuth > 180) and (patch_azimuth < 360):
        Lwest = patch_radiation * steradian * angle_of_incidence * torch.cos(torch.as_tensor((270 - patch_azimuth) * deg2rad, device=device))
    if (patch_azimuth > 270) or (patch_azimuth < 90):
        Lnorth = patch_radiation * steradian * angle_of_incidence * torch.cos(torch.as_tensor((0 - patch_azimuth) * deg2rad, device=device))

    return Lside, Lside_sh, Ldown, Ldown_sh, Least, Lsouth, Lwest, Lnorth

# def longwave_from_buildings_wallScheme(
#     voxelMaps, voxelTable, steradian, angle_of_incidence, angle_of_incidence_h, patch_azimuth,
# ):
#     device = voxelMaps.device
#     deg2rad = torch.pi / 180
#     shape = (voxelMaps.shape[0], voxelMaps.shape[1])

#     Lside = torch.zeros(shape, device=device)
#     Lside_sh = torch.zeros(shape, device=device)
#     Ldown = torch.zeros(shape, device=device)
#     Ldown_sh = torch.zeros(shape, device=device)
#     Least = torch.zeros(shape, device=device)
#     Lsouth = torch.zeros(shape, device=device)
#     Lwest = torch.zeros(shape, device=device)
#     Lnorth = torch.zeros(shape, device=device)

#     # np.vectorize(dict.get) has no torch equivalent - it's an elementwise Python-level
#     # dict lookup, which can't run on GPU (and isn't really vectorized in numpy either,
#     # it's a disguised Python loop). Build a dense lookup tensor once and gather instead.
#     voxel_ids = torch.unique(voxelMaps)
#     ids = voxel_ids[1:].long()          # same [1:] convention as original: excludes the sentinel (lowest id)
#     max_id = int(voxel_ids.max().item())
#     lookup = torch.zeros(max_id + 1, device=device, dtype=torch.float64)
#     if ids.numel() > 0:
#         values = torch.tensor(
#             [voxelTable.loc[int(i), "LongwaveRadiation"] for i in ids],
#             device=device, dtype=torch.float64,
#         )
#         lookup[ids] = values
#     patch_radiation = lookup[voxelMaps.long()]

#     Lside = Lside + patch_radiation * steradian * angle_of_incidence
#     Ldown = Ldown + patch_radiation * steradian * angle_of_incidence_h

#     if (patch_azimuth > 360) or (patch_azimuth < 180):
#         Least = patch_radiation * steradian * angle_of_incidence * torch.cos(torch.as_tensor((90 - patch_azimuth) * deg2rad, device=device))
#     if (patch_azimuth > 90) and (patch_azimuth < 270):
#         Lsouth = patch_radiation * steradian * angle_of_incidence * torch.cos(torch.as_tensor((180 - patch_azimuth) * deg2rad, device=device))
#     if (patch_azimuth > 180) and (patch_azimuth < 360):
#         Lwest = patch_radiation * steradian * angle_of_incidence * torch.cos(torch.as_tensor((270 - patch_azimuth) * deg2rad, device=device))
#     if (patch_azimuth > 270) or (patch_azimuth < 90):
#         Lnorth = patch_radiation * steradian * angle_of_incidence * torch.cos(torch.as_tensor((0 - patch_azimuth) * deg2rad, device=device))

#     return Lside, Lside_sh, Ldown, Ldown_sh, Least, Lsouth, Lwest, Lnorth


def reflected_longwave(
    reflecting_surface, steradian, angle_of_incidence, angle_of_incidence_h,
    patch_azimuth, Ldown_sky, Lup, ewall,
):
    device = reflecting_surface.device
    deg2rad = torch.pi / 180

    reflected_radiation = ((Ldown_sky + Lup) * (1 - ewall) * 0.5) / torch.pi

    Lside_ref = reflected_radiation * steradian * angle_of_incidence * reflecting_surface
    Ldown_ref = reflected_radiation * steradian * angle_of_incidence_h * reflecting_surface

    Least = torch.zeros((reflecting_surface.shape[0], reflecting_surface.shape[1]), device=device)
    Lsouth = torch.zeros((reflecting_surface.shape[0], reflecting_surface.shape[1]), device=device)
    Lwest = torch.zeros((reflecting_surface.shape[0], reflecting_surface.shape[1]), device=device)
    Lnorth = torch.zeros((reflecting_surface.shape[0], reflecting_surface.shape[1]), device=device)

    if (patch_azimuth > 360) or (patch_azimuth < 180):
        Least = reflected_radiation * steradian * angle_of_incidence * reflecting_surface * torch.cos(torch.as_tensor((90 - patch_azimuth) * deg2rad, device=device))
    if (patch_azimuth > 90) and (patch_azimuth < 270):
        Lsouth = reflected_radiation * steradian * angle_of_incidence * reflecting_surface * torch.cos(torch.as_tensor((180 - patch_azimuth) * deg2rad, device=device))
    if (patch_azimuth > 180) and (patch_azimuth < 360):
        Lwest = reflected_radiation * steradian * angle_of_incidence * reflecting_surface * torch.cos(torch.as_tensor((270 - patch_azimuth) * deg2rad, device=device))
    if (patch_azimuth > 270) or (patch_azimuth < 90):
        Lnorth = reflected_radiation * steradian * angle_of_incidence * reflecting_surface * torch.cos(torch.as_tensor((0 - patch_azimuth) * deg2rad, device=device))

    return Lside_ref, Ldown_ref, Least, Lsouth, Lwest, Lnorth


def patch_steradians(L_patches):
    """This function calculates the steradians of the patches"""
    device = L_patches.device

    # Degrees to radians
    deg2rad = torch.pi / 180

    # Unique altitudes for patches
    skyalt, skyalt_c = torch.unique(L_patches[:, 0], return_counts=True)

    # Altitudes of the Robinson & Stone patches
    patch_altitude = L_patches[:, 0]

    # Calculation of steradian for each patch
    steradian = torch.zeros(patch_altitude.shape[0], device=device)
    for i in range(patch_altitude.shape[0]):
        # If there are more than one patch in a band
        if skyalt_c[skyalt == patch_altitude[i]] > 1:
            steradian[i] = (
                (360 / skyalt_c[skyalt == patch_altitude[i]]) * deg2rad
            ) * (
                torch.sin((patch_altitude[i] + patch_altitude[0]) * deg2rad)
                - torch.sin((patch_altitude[i] - patch_altitude[0]) * deg2rad)
            )
        # If there is only one patch in band, i.e. 90 degrees
        else:
            steradian[i] = (
                (360 / skyalt_c[skyalt == patch_altitude[i]]) * deg2rad
            ) * (
                torch.sin((patch_altitude[i]) * deg2rad)
                - torch.sin((patch_altitude[i - 1] + patch_altitude[0]) * deg2rad)
            )

    return steradian, skyalt, patch_altitude