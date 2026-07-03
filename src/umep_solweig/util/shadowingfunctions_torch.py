# -*- coding: utf-8 -*-
# Ready for python action!
import numpy as np
from math import radians

try:
    import torch
except:
    pass

# from numba import jit


def shadowingfunctionglobalradiation(
    a, azimuth, altitude, scale, feedback, forsvf, device
):

    # %This m.file calculates shadows on a DEM
    # % conversion
    degrees = torch.pi / 180.0
    azimuth = azimuth * degrees
    altitude = altitude * degrees
    # % measure the size of the image
    sizex = a.shape[0]
    sizey = a.shape[1]
    if forsvf == 0:
        barstep = torch.max([sizex, sizey])
        total = 100.0 / barstep  # dlg.progressBar.setRange(0, barstep)
    # % initialise parameters
    f = a
    dx = torch.tensor(0.0, device=device)
    dy = torch.tensor(0.0, device=device)
    dz = torch.tensor(0.0, device=device)
    temp = torch.zeros((sizex, sizey), device=device)
    index = torch.tensor(1.0, device=device)
    # % other loop parameters
    amaxvalue = a.max()
    pibyfour = torch.pi / 4.0
    threetimespibyfour = 3.0 * pibyfour
    fivetimespibyfour = 5.0 * pibyfour
    seventimespibyfour = 7.0 * pibyfour
    sinazimuth = torch.sin(azimuth)
    cosazimuth = torch.cos(azimuth)
    tanazimuth = torch.tan(azimuth)
    signsinazimuth = torch.sign(sinazimuth)
    signcosazimuth = torch.sign(cosazimuth)
    dssin = torch.abs((1.0 / sinazimuth))
    dscos = torch.abs((1.0 / cosazimuth))
    tanaltitudebyscale = torch.tan(altitude) / scale
    # % main loop
    while amaxvalue >= dz and torch.abs(dx) < sizex and torch.abs(dy) < sizey:
        if forsvf == 0:
            feedback.setProgress(int(index * total))
            # dlg.progressBar.setValue(index)
        # while torch.logical_and(torch.logical_and(amaxvalue >= dz, torch.abs(dx) <= sizex), torch.abs(dy) <= sizey):(torch.logical_and(amaxvalue >= dz, torch.abs(dx) <= sizex), torch.abs(dy) <= sizey):
        # if torch.logical_or(torch.logical_and(pibyfour <= azimuth, azimuth <
        # threetimespibyfour), torch.logical_and(fivetimespibyfour <= azimuth,
        # azimuth < seventimespibyfour)):
        if (
            pibyfour <= azimuth
            and azimuth < threetimespibyfour
            or fivetimespibyfour <= azimuth
            and azimuth < seventimespibyfour
        ):
            dy = signsinazimuth * index
            dx = (
                -1.0
                * signcosazimuth
                * torch.abs(torch.round(index / tanazimuth))
            )
            ds = dssin
        else:
            dy = signsinazimuth * torch.abs(torch.round(index * tanazimuth))
            dx = -1.0 * signcosazimuth * index
            ds = dscos

        # % note: dx and dy represent absolute values while ds is an incremental value
        dz = ds * index * tanaltitudebyscale
        temp[0:sizex, 0:sizey] = 0.0
        absdx = torch.abs(dx)
        absdy = torch.abs(dy)
        xc1 = (dx + absdx) / 2.0 + 1.0
        xc2 = sizex + (dx - absdx) / 2.0
        yc1 = (dy + absdy) / 2.0 + 1.0
        yc2 = sizey + (dy - absdy) / 2.0
        xp1 = -((dx - absdx) / 2.0) + 1.0
        xp2 = sizex - (dx + absdx) / 2.0
        yp1 = -((dy - absdy) / 2.0) + 1.0
        yp2 = sizey - (dy + absdy) / 2.0
        temp[int(xp1) - 1 : int(xp2), int(yp1) - 1 : int(yp2)] = (
            a[int(xc1) - 1 : int(xc2), int(yc1) - 1 : int(yc2)] - dz
        )
        # f = torch.maximum(f, temp)  # bad performance in python3. Replaced with
        # fmax
        f = torch.maximum(f, temp)
        index += 1.0

    f = f - a
    f = torch.logical_not(f)
    sh = f.double()

    return sh


# @jit(nopython=True)


def shadowingfunction_20(
    a,
    vegdem,
    vegdem2,
    azimuth,
    altitude,
    scale,
    amaxvalue,
    bush,
    feedback,
    forsvf,
    device,
):

    # conversion
    degrees = torch.pi / 180.0
    azimuth = azimuth * degrees
    altitude = altitude * degrees

    # measure the size of grid
    sizex = a.shape[0]
    sizey = a.shape[1]

    # progressbar for svf plugin
    if forsvf == 0:
        barstep = torch.max([sizex, sizey])
        total = 100.0 / barstep
        feedback.setProgress(0)

    # initialise parameters
    dx = torch.tensor(0.0, device=device)
    dy = torch.tensor(0.0, device=device)
    dz = torch.tensor(0.0, device=device)
    temp = torch.zeros((sizex, sizey), device=device)
    tempvegdem = torch.zeros((sizex, sizey), device=device)
    tempvegdem2 = torch.zeros((sizex, sizey), device=device)
    templastfabovea = torch.zeros((sizex, sizey), device=device)
    templastgabovea = torch.zeros((sizex, sizey), device=device)
    bushplant = bush > 1.0
    sh = torch.zeros((sizex, sizey), device=device)  # shadows from buildings
    vbshvegsh = torch.zeros(
        (sizex, sizey), device=device
    )  # vegetation blocking buildings
    vegsh = torch.add(
        torch.zeros((sizex, sizey), device=device).float(), bushplant
    ).float()  # vegetation shadow
    f = a

    pibyfour = torch.pi / 4.0
    threetimespibyfour = 3.0 * pibyfour
    fivetimespibyfour = 5.0 * pibyfour
    seventimespibyfour = 7.0 * pibyfour
    sinazimuth = torch.sin(azimuth)
    cosazimuth = torch.cos(azimuth)
    tanazimuth = torch.tan(azimuth)
    signsinazimuth = torch.sign(sinazimuth)
    signcosazimuth = torch.sign(cosazimuth)
    dssin = torch.abs((1.0 / sinazimuth))
    dscos = torch.abs((1.0 / cosazimuth))
    tanaltitudebyscale = torch.tan(altitude) / scale
    # index = 1
    index = torch.tensor(0.0, device=device)

    # new case with pergola (thin vertical layer of vegetation), August 2021
    dzprev = torch.tensor(0.0, device=device)

    # main loop
    while (
        (amaxvalue >= dz)
        and (torch.abs(dx) < sizex)
        and (torch.abs(dy) < sizey)
    ):
        if forsvf == 0:
            # dlg.progressBar.setValue(index)
            feedback.setProgress(int(index * total))
        if (
            (pibyfour <= azimuth)
            and (azimuth < threetimespibyfour)
            or (fivetimespibyfour <= azimuth)
            and (azimuth < seventimespibyfour)
        ):
            dy = signsinazimuth * index
            dx = (
                -1.0
                * signcosazimuth
                * torch.abs(torch.round(index / tanazimuth))
            )
            ds = dssin
        else:
            dy = signsinazimuth * torch.abs(torch.round(index * tanazimuth))
            dx = -1.0 * signcosazimuth * index
            ds = dscos
        # note: dx and dy represent absolute values while ds is an incremental
        # value
        dz = (ds * index) * tanaltitudebyscale
        tempvegdem[0:sizex, 0:sizey] = 0.0
        tempvegdem2[0:sizex, 0:sizey] = 0.0
        temp[0:sizex, 0:sizey] = 0.0
        templastfabovea[0:sizex, 0:sizey] = 0.0
        templastgabovea[0:sizex, 0:sizey] = 0.0
        absdx = torch.abs(dx)
        absdy = torch.abs(dy)
        xc1 = int((dx + absdx) / 2.0)
        xc2 = int(sizex + (dx - absdx) / 2.0)
        yc1 = int((dy + absdy) / 2.0)
        yc2 = int(sizey + (dy - absdy) / 2.0)
        xp1 = int(-((dx - absdx) / 2.0))
        xp2 = int(sizex - (dx + absdx) / 2.0)
        yp1 = int(-((dy - absdy) / 2.0))
        yp2 = int(sizey - (dy + absdy) / 2.0)

        tempvegdem[xp1:xp2, yp1:yp2] = vegdem[xc1:xc2, yc1:yc2] - dz
        tempvegdem2[xp1:xp2, yp1:yp2] = vegdem2[xc1:xc2, yc1:yc2] - dz
        temp[xp1:xp2, yp1:yp2] = a[xc1:xc2, yc1:yc2] - dz

        f = torch.maximum(f, temp)  # Moving building shadow
        sh[(f > a)] = 1.0
        sh[(f <= a)] = 0.0
        fabovea = tempvegdem > a  # vegdem above DEM
        gabovea = tempvegdem2 > a  # vegdem2 above DEM

        # new pergola condition
        templastfabovea[xp1:xp2, yp1:yp2] = vegdem[xc1:xc2, yc1:yc2] - dzprev
        templastgabovea[xp1:xp2, yp1:yp2] = vegdem2[xc1:xc2, yc1:yc2] - dzprev
        lastfabovea = templastfabovea > a
        lastgabovea = templastgabovea > a
        dzprev = dz
        vegsh2 = torch.add(
            torch.add(torch.add(fabovea, gabovea), lastfabovea).float(),
            lastgabovea,
        ).float()
        vegsh2[vegsh2 == 4] = 0.0
        # vegsh2[vegsh2 == 1] = 0. # This one is the ultimate question...
        vegsh2[vegsh2 > 0] = 1.0

        vegsh = torch.maximum(vegsh, vegsh2)
        vegsh[(vegsh * sh > 0.0)] = 0.0
        vbshvegsh = vegsh + vbshvegsh  # removing shadows 'behind' buildings

        index += 1.0

    sh = 1.0 - sh
    vbshvegsh[(vbshvegsh > 0.0)] = 1.0
    vbshvegsh = vbshvegsh - vegsh
    vegsh = 1.0 - vegsh
    vbshvegsh = 1.0 - vbshvegsh

    shadowresult = {"sh": sh, "vegsh": vegsh, "vbshvegsh": vbshvegsh}

    return shadowresult


def shadowingfunction_findwallID(
    dsm,
    azimuth,
    altitude,
    scale,
    walls,
    uniqueWallIDs,
    dem,
    wall2d_id,
    voxel_height,
    voxelId_list,
    facesh,
    wall_dict,
    sh,
    device,
):
    """
    This function identifies what wall id and voxel height that is seen from a ground pixel

    INPUTS:
    dsm = Digital surface model
    azimuth and altitude = sun position in degrees
    scale= scale of DSM (1 meter pixels=1, 2 meter pixels=0.5)
    uniqueWallIDs = pixel row 'outside' buildings. will be calculated if empty
    walls = height of walls
    dem = Digital elevation model. (Should be excluded in future to incorporate ground elevation)

    OUTPUT:
    buildIDSeen = ID seen from ground pixel
    voxelHeight = Wall height shadow volume

    Fredrik Lindberg 2023-02-16
    fredrikl@gvc.gu.se

    """

    # Remove ground heights
    dsm = dsm - dem
    # buildings = 1 - ((dsm) > 0)
    dsm[dsm < 0.5] = 0

    # conversion, degrees to radians
    azimuth = radians(azimuth)
    azimuth = torch.tensor(azimuth, device=device)
    altitude = radians(altitude)
    altitude = torch.tensor(altitude, device=device)

    # measure the size of the image
    rows, cols = dsm.shape

    # initialise parameters
    f = torch.clone(dsm)
    buildIDSeen = torch.zeros((rows, cols), device=device)

    dx = torch.tensor(0.0, device=device)
    dy = torch.tensor(0.0, device=device)
    dz = torch.tensor(0.0, device=device)
    temp = torch.zeros((rows, cols), device=device)
    temp2 = torch.zeros((rows, cols), device=device)  # walls
    tempwallID = torch.zeros((rows, cols), device=device)

    voxelHeight = torch.zeros((rows, cols), device=device)
    temp3 = torch.ones((rows, cols), device=device)

    # other loop parameters
    amaxvalue = torch.max(dsm)
    pibyfour = torch.pi / 4
    threetimespibyfour = 3 * pibyfour
    fivetimespibyfour = 5 * pibyfour
    seventimespibyfour = 7 * pibyfour
    sinazimuth = torch.sin(azimuth)
    cosazimuth = torch.cos(azimuth)
    tanazimuth = torch.tan(azimuth)
    signsinazimuth = torch.sign(sinazimuth)
    signcosazimuth = torch.sign(cosazimuth)
    dssin = torch.abs(1 / sinazimuth)
    dscos = torch.abs(1 / cosazimuth)
    tanaltitudebyscale = torch.tan(altitude) / scale

    max_wall_id = int(max(wall_dict.keys())) if wall_dict else 0
    wall_lookup = torch.zeros(max_wall_id + 1, device=device)
    for k, v in wall_dict.items():
        wall_lookup[k] = v

    index = 1

    # main loop
    while (
        (amaxvalue >= dz) and (torch.abs(dx) < rows) and (torch.abs(dy) < cols)
    ):

        if (pibyfour <= azimuth and azimuth < threetimespibyfour) or (
            fivetimespibyfour <= azimuth and azimuth < seventimespibyfour
        ):
            dy = signsinazimuth * index
            dx = (
                -1
                * signcosazimuth
                * torch.abs(torch.round(index / tanazimuth))
            )
            ds = dssin
        else:
            dy = signsinazimuth * torch.abs(torch.round(index * tanazimuth))
            dx = -1 * signcosazimuth * index
            ds = dscos

        # note: dx and dy represent absolute values while ds is an incremental
        # value
        dz = ds * index * tanaltitudebyscale
        temp[0:rows, 0:cols] = 0
        temp2[0:rows, 0:cols] = 0

        absdx = torch.abs(dx)
        absdy = torch.abs(dy)

        xc1 = int((dx + absdx) / 2)
        xc2 = int(rows + (dx - absdx) / 2)
        yc1 = int((dy + absdy) / 2)
        yc2 = int(cols + (dy - absdy) / 2)

        xp1 = int(-((dx - absdx) / 2))
        xp2 = int(rows - (dx + absdx) / 2)
        yp1 = int(-((dy - absdy) / 2))
        yp2 = int(cols - (dy + absdy) / 2)

        wallSeen = facesh
        uniqueWallIDs = uniqueWallIDs * wallSeen

        # # Moving wall shadow
        # Moving wall id
        tempwallID[xp1:xp2, yp1:yp2] = uniqueWallIDs[xc1:xc2, yc1:yc2]

        # Get wall height from wall id
        temp_wallHeight = wall_lookup[tempwallID.long()]

        # Descending wall, how much of the wall that is still above ground
        # level
        temp2 = temp_wallHeight - dz

        # buildIDSeen = Wall pixels/voxels seen, i.e. only voxels that are positive (above ground level) (temp2 > 0).
        # temp3 indicates those pixels that the walls have not progressed into
        # yet (saved in previous iteration).
        buildIDSeen = (temp2 > 0) * temp3 * tempwallID + buildIDSeen

        # voxelHeight = the elevation on a wall that is seen from a pixel with the given altitude and azimuth (only above ground leve, i.e. (temp2 > 0)).
        # voxelHeight = wall height - descending wall, i.e. temp_wallHeight -
        # temp2. Only applicable to pixels where there is no value from
        # previous iterations (temp3).
        voxelHeight = (temp2 > 0) * temp3 * (
            temp_wallHeight - temp2
        ) + voxelHeight

        # Remember pixels previous iteration that walls have not progressed
        # into yet.
        temp3 = torch.clone(temp2 <= 0) * (buildIDSeen == 0)

        index += 1

    # Ceil voxel height values to integers
    voxelHeight_ceil = torch.ceil(voxelHeight)
    # voxelHeight_ceil = torch.round(voxelHeight)

    # Empty raster to fill with voxel IDs
    voxelId = torch.zeros(
        (rows, cols), device=device, dtype=voxelId_list.dtype
    )
    # Convert wall2d_id from list to numpy array
    wall2d_id = torch.tensor(wall2d_id, device=device)
    # Convert voxel_height from list to numpy array
    voxel_height = torch.tensor(voxel_height, device=device)
    # Convert voxelId_list from list to numpy array
    voxelId_list = torch.tensor(voxelId_list, dtype=int, device=device)

    # Flatten buildIDseen from matrix to array
    a = buildIDSeen.flatten()
    # Flatten voxelHeight_ceil from matrix to array
    b = voxelHeight_ceil.flatten()
    # Combine the two above arrays into an n by 2 array
    c = torch.column_stack([a, b])
    # Find unique values in c
    d = torch.unique(c, dim=0)
    # Remove rows where both columns are zero
    d = d[~torch.all(d == 0, dim=1)]

    not_in_list = 0
    in_list = 0
    # Fill voxelId matrix with unique voxel IDs
    for temp_id, temp_height in d:
        # print(str(temp_id) + ' ' + str(temp_height))
        temp_fill_id = voxelId_list[
            ((wall2d_id == temp_id) & (voxel_height == temp_height))
        ]
        if temp_fill_id.__len__() > 0:
            # print('temp_fill_id = ' + str(temp_fill_id))
            voxelId[
                (buildIDSeen == temp_id) & (voxelHeight_ceil == temp_height),
            ] = temp_fill_id
            in_list += 1
        else:
            not_in_list += 1
            buildIDSeen[
                (buildIDSeen == temp_id) & (voxelHeight_ceil == temp_height)
            ] = 0
            voxelHeight_ceil[
                (buildIDSeen == temp_id) & (voxelHeight_ceil == temp_height)
            ] = 0

    # Correct for shadows, i.e. remove weird pixels on top of buildings etc
    buildIDSeen = buildIDSeen * (1 - sh)
    voxelHeight = voxelHeight * (1 - sh)
    voxelId = voxelId * (1 - sh)

    return buildIDSeen, voxelHeight, voxelId
