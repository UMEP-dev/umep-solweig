import math
import torch


def sunonsurface_2018a(
    azimuthA,
    scale,
    buildings,
    shadow,
    sunwall,
    first,
    second,
    aspect,
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
):
    device = walls.device
    sizex = walls.shape[0]
    sizey = walls.shape[1]

    wallbol = (walls > 0) * 1
    sunwall[sunwall > 0] = 1  # test 20160910

    # conversion into radians - azimuthA is a scalar solar azimuth for this timestep, not a
    # raster, so this and everything derived from it below stays plain Python (math module)
    azimuth = azimuthA * (math.pi / 180)

    # loop parameters
    index = 0
    f = buildings
    Lup = (
        SBC * emis_grid * (Tg * shadow + Ta + 273.15) ** 4
        - SBC * emis_grid * (Ta + 273.15) ** 4
    )
    if landcover == 1:
        Tg[lc_grid == 3] = Twater - Ta

    Lwall = (
        SBC * ewall * (Tgwall + Ta + 273.15) ** 4
        - SBC * ewall * (Ta + 273.15) ** 4
    )
    albshadow = alb_grid * shadow
    alb = alb_grid

    tempsh = torch.zeros((sizex, sizey), device=device)
    tempbu = torch.zeros((sizex, sizey), device=device)
    tempbub = torch.zeros((sizex, sizey), device=device)
    tempbubwall = torch.zeros((sizex, sizey), device=device)
    tempwallsun = torch.zeros((sizex, sizey), device=device)
    weightsumsh = torch.zeros((sizex, sizey), device=device)
    weightsumwall = torch.zeros((sizex, sizey), device=device)

    # first/second control loop iteration count - scalars, not rasters, so plain round()
    # first = round(first * scale)
    # if first < 1:
    #     first = 1
    # second = round(second * scale)

    def _scalar(x):
        return x.item() if torch.is_tensor(x) else x

    # first/second control loop iteration count - scalars, not rasters, so plain round().
    # _scalar() unwraps a tensor to a plain Python number first, since round() has no
    # __round__ defined for Tensor - this handles first/scale arriving as either a tensor
    # or a plain float, whichever the caller happens to pass.
    first = round(_scalar(first * scale))
    if first < 1:
        first = 1
    second = round(_scalar(second * scale))

    weightsumLupsh = torch.zeros((sizex, sizey), device=device)
    weightsumLwall = torch.zeros((sizex, sizey), device=device)
    weightsumalbsh = torch.zeros((sizex, sizey), device=device)
    weightsumalbwall = torch.zeros((sizex, sizey), device=device)
    weightsumalbnosh = torch.zeros((sizex, sizey), device=device)
    weightsumalbwallnosh = torch.zeros((sizex, sizey), device=device)
    tempLupsh = torch.zeros((sizex, sizey), device=device)
    tempalbsh = torch.zeros((sizex, sizey), device=device)
    tempalbnosh = torch.zeros((sizex, sizey), device=device)

    # other loop parameters - all pure scalar geometry, never touches torch
    pibyfour = math.pi / 4
    threetimespibyfour = 3 * pibyfour
    fivetimespibyfour = 5 * pibyfour
    seventimespibyfour = 7 * pibyfour
    sinazimuth = math.sin(azimuth)
    cosazimuth = math.cos(azimuth)
    tanazimuth = math.tan(azimuth)
    # matches np.sign's exact-zero semantics (math.copysign would give ±1 instead of 0 at x==0)
    signsinazimuth = (sinazimuth > 0) - (sinazimuth < 0)
    signcosazimuth = (cosazimuth > 0) - (cosazimuth < 0)

    ## The Shadow casting algoritm
    for n in range(int(second)):
        if (pibyfour <= azimuth and azimuth < threetimespibyfour) or (
            fivetimespibyfour <= azimuth and azimuth < seventimespibyfour
        ):
            dy = signsinazimuth * index
            dx = -1 * signcosazimuth * abs(round(index / tanazimuth))
        else:
            dy = signsinazimuth * abs(round(index * tanazimuth))
            dx = -1 * signcosazimuth * index

        absdx = abs(dx)
        absdy = abs(dy)

        xc1 = (dx + absdx) / 2
        xc2 = sizex + (dx - absdx) / 2
        yc1 = (dy + absdy) / 2
        yc2 = sizey + (dy - absdy) / 2

        xp1 = -((dx - absdx) / 2)
        xp2 = sizex - (dx + absdx) / 2
        yp1 = -((dy - absdy) / 2)
        yp2 = sizey - (dy + absdy) / 2

        tempbu[int(xp1) : int(xp2), int(yp1) : int(yp2)] = buildings[
            int(xc1) : int(xc2), int(yc1) : int(yc2)
        ]
        tempsh[int(xp1) : int(xp2), int(yp1) : int(yp2)] = shadow[
            int(xc1) : int(xc2), int(yc1) : int(yc2)
        ]
        tempLupsh[int(xp1) : int(xp2), int(yp1) : int(yp2)] = Lup[
            int(xc1) : int(xc2), int(yc1) : int(yc2)
        ]
        tempalbsh[int(xp1) : int(xp2), int(yp1) : int(yp2)] = albshadow[
            int(xc1) : int(xc2), int(yc1) : int(yc2)
        ]
        tempalbnosh[int(xp1) : int(xp2), int(yp1) : int(yp2)] = alb[
            int(xc1) : int(xc2), int(yc1) : int(yc2)
        ]
        f = torch.minimum(f, tempbu)  # elementwise min of two tensors directly

        shadow2 = tempsh * f
        weightsumsh = weightsumsh + shadow2

        Lupsh = tempLupsh * f
        weightsumLupsh = weightsumLupsh + Lupsh

        albsh = tempalbsh * f
        weightsumalbsh = weightsumalbsh + albsh

        albnosh = tempalbnosh * f
        weightsumalbnosh = weightsumalbnosh + albnosh

        tempwallsun[int(xp1) : int(xp2), int(yp1) : int(yp2)] = sunwall[
            int(xc1) : int(xc2), int(yc1) : int(yc2)
        ]
        tempb = tempwallsun * f
        tempbwall = f * -1 + 1
        tempbub = ((tempb + tempbub) > 0) * 1
        tempbubwall = ((tempbwall + tempbubwall) > 0) * 1
        weightsumLwall = weightsumLwall + tempbub * Lwall
        weightsumalbwall = weightsumalbwall + tempbub * albedo_b
        weightsumwall = weightsumwall + tempbub
        weightsumalbwallnosh = weightsumalbwallnosh + tempbubwall * albedo_b

        ind = 1
        if (n + 1) <= first:
            weightsumwall_first = weightsumwall / ind
            weightsumsh_first = weightsumsh / ind
            wallsuninfluence_first = weightsumwall_first > 0
            weightsumLwall_first = (weightsumLwall) / ind
            weightsumLupsh_first = weightsumLupsh / ind

            weightsumalbwall_first = weightsumalbwall / ind
            weightsumalbsh_first = weightsumalbsh / ind
            weightsumalbwallnosh_first = weightsumalbwallnosh / ind
            weightsumalbnosh_first = weightsumalbnosh / ind
            wallinfluence_first = weightsumalbwallnosh_first > 0
            ind += 1
        index += 1

    wallsuninfluence_second = weightsumwall > 0
    wallinfluence_second = weightsumalbwallnosh > 0

    # Removing walls in shadow due to selfshadowing
    azilow = azimuth - math.pi / 2
    azihigh = azimuth + math.pi / 2
    if azilow >= 0 and azihigh < 2 * math.pi:  # 90 to 270  (SHADOW)
        facesh = (
            torch.logical_or(aspect < azilow, aspect >= azihigh).float()
            - wallbol
            + 1
        )
    elif azilow < 0 and azihigh <= 2 * math.pi:  # 0 to 90
        azilow = azilow + 2 * math.pi
        facesh = torch.logical_or(aspect > azilow, aspect <= azihigh) * -1 + 1
    elif azilow > 0 and azihigh >= 2 * math.pi:  # 270 to 360
        azihigh = azihigh - 2 * math.pi
        facesh = torch.logical_or(aspect > azilow, aspect <= azihigh) * -1 + 1

    # removing walls in self shadoing - torch refuses subtraction where EITHER side is a
    # bool tensor (numpy auto-promotes and allows it), so cast the comparison first
    keep = (weightsumwall == second).float() - facesh
    keep[keep == -1] = 0

    # gvf from shadow only
    gvf1 = (
        (weightsumwall_first + weightsumsh_first) / (first + 1)
    ) * wallsuninfluence_first + (weightsumsh_first) / (first) * (
        wallsuninfluence_first * -1 + 1
    )
    weightsumwall[keep == 1] = 0
    gvf2 = (
        (weightsumwall + weightsumsh) / (second + 1)
    ) * wallsuninfluence_second + (weightsumsh) / (second) * (
        wallsuninfluence_second * -1 + 1
    )
    gvf2[gvf2 > 1.0] = 1.0

    # gvf from shadow and Lup
    gvfLup1 = (
        (weightsumLwall_first + weightsumLupsh_first) / (first + 1)
    ) * wallsuninfluence_first + (weightsumLupsh_first) / (first) * (
        wallsuninfluence_first * -1 + 1
    )
    weightsumLwall[keep == 1] = 0
    gvfLup2 = (
        (weightsumLwall + weightsumLupsh) / (second + 1)
    ) * wallsuninfluence_second + (weightsumLupsh) / (second) * (
        wallsuninfluence_second * -1 + 1
    )

    # gvf from shadow and albedo
    gvfalb1 = (
        (weightsumalbwall_first + weightsumalbsh_first) / (first + 1)
    ) * wallsuninfluence_first + (weightsumalbsh_first) / (first) * (
        wallsuninfluence_first * -1 + 1
    )
    weightsumalbwall[keep == 1] = 0
    gvfalb2 = (
        (weightsumalbwall + weightsumalbsh) / (second + 1)
    ) * wallsuninfluence_second + (weightsumalbsh) / (second) * (
        wallsuninfluence_second * -1 + 1
    )

    # gvf from albedo only
    gvfalbnosh1 = (
        (weightsumalbwallnosh_first + weightsumalbnosh_first) / (first + 1)
    ) * wallinfluence_first + (weightsumalbnosh_first) / (first) * (
        wallinfluence_first * -1 + 1
    )
    gvfalbnosh2 = (
        (weightsumalbwallnosh + weightsumalbnosh) / (second)
    ) * wallinfluence_second + (weightsumalbnosh) / (second) * (
        wallinfluence_second * -1 + 1
    )

    # Weighting
    gvf = (gvf1 * 0.5 + gvf2 * 0.4) / 0.9
    gvfLup = (gvfLup1 * 0.5 + gvfLup2 * 0.4) / 0.9
    gvfLup = gvfLup + (
        (SBC * emis_grid * (Tg * shadow + Ta + 273.15) ** 4)
        - SBC * emis_grid * (Ta + 273.15) ** 4
    ) * (
        buildings * -1 + 1
    )
    gvfalb = (gvfalb1 * 0.5 + gvfalb2 * 0.4) / 0.9
    gvfalb = gvfalb + alb_grid * (buildings * -1 + 1) * shadow
    gvfalbnosh = (gvfalbnosh1 * 0.5 + gvfalbnosh2 * 0.4) / 0.9
    gvfalbnosh = gvfalbnosh * buildings + alb_grid * (buildings * -1 + 1)

    return gvf, gvfLup, gvfalb, gvfalbnosh, gvf2