import torch
from .sunonsurface_2018a_torch import sunonsurface_2018a


def gvf_2018a(
    wallsun,
    walls,
    buildings,
    scale,
    shadow,
    first,
    second,
    dirwalls,
    Tg,
    Tgwall,
    Ta,
    emis_grid,
    ewall,
    alb_grid,
    SBC,
    albedo_b,
    rows,
    cols,
    Twater,
    lc_grid,
    landcover,
    debug=False
):
    device = walls.device

    # Search directions for Ground View Factors (GVF) - fixed scalar azimuths, not raster
    # data, so plain range() rather than np.arange/torch equivalent
    azimuthA = range(5, 359, 20)

    #### Ground View Factors ####
    gvfLup = torch.zeros((rows, cols), device=device)
    gvfalb = torch.zeros((rows, cols), device=device)
    gvfalbnosh = torch.zeros((rows, cols), device=device)
    gvfLupE = torch.zeros((rows, cols), device=device)
    gvfLupS = torch.zeros((rows, cols), device=device)
    gvfLupW = torch.zeros((rows, cols), device=device)
    gvfLupN = torch.zeros((rows, cols), device=device)
    gvfalbE = torch.zeros((rows, cols), device=device)
    gvfalbS = torch.zeros((rows, cols), device=device)
    gvfalbW = torch.zeros((rows, cols), device=device)
    gvfalbN = torch.zeros((rows, cols), device=device)
    gvfalbnoshE = torch.zeros((rows, cols), device=device)
    gvfalbnoshS = torch.zeros((rows, cols), device=device)
    gvfalbnoshW = torch.zeros((rows, cols), device=device)
    gvfalbnoshN = torch.zeros((rows, cols), device=device)
    gvfSum = torch.zeros((rows, cols), device=device)

    #  sunwall=wallinsun_2015a(buildings,azimuth(i),shadow,psi(i),dirwalls,walls);
    sunwall = (wallsun / walls * buildings) == 1  # new as from 2015a

    for azimuth in azimuthA:
        _, gvfLupi, gvfalbi, gvfalbnoshi, gvf2 = sunonsurface_2018a(
            azimuth,
            scale,
            buildings,
            shadow,
            sunwall,
            first,
            second,
            dirwalls * torch.pi / 180,
            walls,
            Tg,
            Tgwall,
            Ta,
            emis_grid,
            ewall,
            alb_grid,
            SBC,
            albedo_b,
            Twater,
            lc_grid,
            landcover,
        )

        gvfLup = gvfLup + gvfLupi
        gvfalb = gvfalb + gvfalbi
        gvfalbnosh = gvfalbnosh + gvfalbnoshi
        gvfSum = gvfSum + gvf2

        if (azimuth >= 0) and (azimuth < 180):
            gvfLupE = gvfLupE + gvfLupi
            gvfalbE = gvfalbE + gvfalbi
            gvfalbnoshE = gvfalbnoshE + gvfalbnoshi

        if (azimuth >= 90) and (azimuth < 270):
            gvfLupS = gvfLupS + gvfLupi
            gvfalbS = gvfalbS + gvfalbi
            gvfalbnoshS = gvfalbnoshS + gvfalbnoshi

        if (azimuth >= 180) and (azimuth < 360):
            gvfLupW = gvfLupW + gvfLupi
            gvfalbW = gvfalbW + gvfalbi
            gvfalbnoshW = gvfalbnoshW + gvfalbnoshi

        if (azimuth >= 270) or (azimuth < 90):
            gvfLupN = gvfLupN + gvfLupi
            gvfalbN = gvfalbN + gvfalbi
            gvfalbnoshN = gvfalbnoshN + gvfalbnoshi

    n_az = len(azimuthA)

    gvfLup = gvfLup / n_az + SBC * emis_grid * (Ta + 273.15) ** 4
    gvfalb = gvfalb / n_az
    gvfalbnosh = gvfalbnosh / n_az

    gvfLupE = gvfLupE / (n_az / 2) + SBC * emis_grid * (Ta + 273.15) ** 4
    gvfLupS = gvfLupS / (n_az / 2) + SBC * emis_grid * (Ta + 273.15) ** 4
    gvfLupW = gvfLupW / (n_az / 2) + SBC * emis_grid * (Ta + 273.15) ** 4
    gvfLupN = gvfLupN / (n_az / 2) + SBC * emis_grid * (Ta + 273.15) ** 4

    gvfalbE = gvfalbE / (n_az / 2)
    gvfalbS = gvfalbS / (n_az / 2)
    gvfalbW = gvfalbW / (n_az / 2)
    gvfalbN = gvfalbN / (n_az / 2)

    gvfalbnoshE = gvfalbnoshE / (n_az / 2)
    gvfalbnoshS = gvfalbnoshS / (n_az / 2)
    gvfalbnoshW = gvfalbnoshW / (n_az / 2)
    gvfalbnoshN = gvfalbnoshN / (n_az / 2)

    gvfNorm = gvfSum / n_az
    gvfNorm[buildings == 0] = 1

    return (
        gvfLup,
        gvfalb,
        gvfalbnosh,
        gvfLupE,
        gvfalbE,
        gvfalbnoshE,
        gvfLupS,
        gvfalbS,
        gvfalbnoshS,
        gvfLupW,
        gvfalbW,
        gvfalbnoshW,
        gvfLupN,
        gvfalbN,
        gvfalbnoshN,
        gvfSum,
        gvfNorm,
    )