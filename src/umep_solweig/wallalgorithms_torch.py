from builtins import range

# -*- coding: utf-8 -*-
__author__ = "xlinfr"

import math
import torch
# import scipy.misc as sc
import scipy.ndimage as sc
from scipy.ndimage import maximum_filter


def findwalls_sp(arr_dsm, walllimit, device, footprint=False):
    # This function identifies walls based on a DSM and a wall height limit.
    # arr_dsm = DSM
    # walllimit = wall height limit
    # footprint = footprint for maximum filter, default = torch.array([[0, 1, 0],
    # [1, 0, 1], [0, 1, 0]])

    # Get the shape of the input array
    col, row = arr_dsm.shape
    walls = torch.zeros((col, row), device=device)

    # Create a padded version of the array
    padded_a = torch.nn.functional.pad(arr_dsm, (1, 1, 1, 1), mode="edge")

    # Default footprint for cardinal points
    if footprint is False:
        footprint = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], device=device)

    # Use maximum_filter with the custom footprint
    max_neighbors = maximum_filter(
        padded_a, footprint=footprint, mode="constant", cval=0
    )

    # Identify wall pixels: walls are where the max neighbors are greater than
    # the original DSM
    walls = max_neighbors[1:-1, 1:-1] - arr_dsm

    # Apply wall height limit
    walls[walls < walllimit] = 0

    # Set the edges to zero
    walls[0 : walls.shape[0], 0] = 0
    walls[0 : walls.shape[0], walls.shape[1] - 1] = 0
    walls[0, 0 : walls.shape[1]] = 0
    walls[walls.shape[0] - 1, 0 : walls.shape[1]] = 0

    return walls



def filter1Goodwin_as_aspect_v3(walls_for_dir, scale, a, feedback, total, device):
    """
    tThis function applies the filter processing presented in Goodwin et al (2010) but instead for removing
    linear fetures it calculates wall aspect based on a wall pixels grid, a dsm (a) and a scale factor

    Fredrik Lindberg, 2012-02-14
    fredrikl@gvc.gu.se

    Translated: 2015-09-15

    :param walls:
    :param scale:
    :param a:
    :return: dirwalls
    """

    walls = walls_for_dir.copy()

    row = a.shape[0]
    col = a.shape[1]

    filtersize = torch.floor((scale + 0.0000000001) * 9)
    if filtersize <= 2:
        filtersize = 3
    else:
        if filtersize != 9:
            if filtersize % 2 == 0:
                filtersize = filtersize + 1

    filthalveceil = int(torch.ceil(filtersize / 2.0))
    filthalvefloor = int(torch.floor(filtersize / 2.0))

    filtmatrix = torch.zeros((int(filtersize), int(filtersize)), device=device)
    buildfilt = torch.zeros((int(filtersize), int(filtersize)), device=device)

    filtmatrix[:, filthalveceil - 1] = 1
    n = filtmatrix.shape[0] - 1
    buildfilt[filthalveceil - 1, 0:filthalvefloor] = 1
    buildfilt[filthalveceil - 1, filthalveceil : int(filtersize)] = 2

    y = torch.zeros((row, col), device=device)  # final direction
    z = torch.zeros((row, col), device=device)  # temporary direction
    x = torch.zeros((row, col), device=device)  # building side
    walls[walls > 0] = 1

    for h in range(
        0, 180
    ):  # =0:1:180 #%increased resolution to 1 deg 20140911
        if feedback is not None:
            feedback.setProgress(int(h * total))
            if feedback.isCanceled():
                feedback.setProgressText("Calculation cancelled")
                break
        filtmatrix1temp = sc.rotate(
            filtmatrix, h, order=1, reshape=False, mode="nearest"
        )  # bilinear
        filtmatrix1 = torch.round(filtmatrix1temp)
        # filtmatrix1temp = sc.imrotate(filtmatrix, h, 'bilinear')
        # filtmatrix1 = torch.round(filtmatrix1temp / 255.)
        # filtmatrixbuildtemp = sc.imrotate(buildfilt, h, 'nearest')
        filtmatrixbuildtemp = sc.rotate(
            buildfilt, h, order=0, reshape=False, mode="nearest"
        )  # Nearest neighbor
        # filtmatrixbuild = torch.round(filtmatrixbuildtemp / 127.)
        filtmatrixbuild = torch.round(filtmatrixbuildtemp)
        index = 270 - h
        if h == 150:
            filtmatrixbuild[:, n] = 0
        if h == 30:
            filtmatrixbuild[:, n] = 0
        if index == 225:
            # n = filtmatrix.shape[0] - 1  # length(filtmatrix);
            filtmatrix1[0, 0] = 1
            filtmatrix1[n, n] = 1
        if index == 135:
            # n = filtmatrix.shape[0] - 1  # length(filtmatrix);
            filtmatrix1[0, n] = 1
            filtmatrix1[n, 0] = 1

        # i=filthalveceil:sizey-filthalveceil
        for i in range(int(filthalveceil) - 1, row - int(filthalveceil) - 1):
            # (j=filthalveceil:sizex-filthalveceil
            for j in range(
                int(filthalveceil) - 1, col - int(filthalveceil) - 1
            ):
                if walls[i, j] == 1:
                    wallscut = (
                        walls[
                            i - filthalvefloor : i + filthalvefloor + 1,
                            j - filthalvefloor : j + filthalvefloor + 1,
                        ]
                        * filtmatrix1
                    )
                    dsmcut = a[
                        i - filthalvefloor : i + filthalvefloor + 1,
                        j - filthalvefloor : j + filthalvefloor + 1,
                    ]
                    if z[i, j] < wallscut.sum():  # sum(sum(wallscut))
                        z[i, j] = wallscut.sum()  # sum(sum(wallscut));
                        if torch.sum(dsmcut[filtmatrixbuild == 1]) > torch.sum(
                            dsmcut[filtmatrixbuild == 2]
                        ):
                            x[i, j] = 1
                        else:
                            x[i, j] = 2

                        y[i, j] = index

    y[(x == 1)] = y[(x == 1)] - 180
    y[(y < 0)] = y[(y < 0)] + 360

    grad, asp = get_ders(a, scale)

    y = y + ((walls == 1) * 1) * ((y == 0) * 1) * (asp / (math.pi / 180.0))

    dirwalls = y

    return dirwalls


def cart2pol(x, y, units="deg"):
    radius = torch.sqrt(x**2 + y**2)
    theta = torch.arctan2(y, x)
    if units in ["deg", "degs"]:
        theta = theta * 180 / torch.pi
    return theta, radius


def get_ders(dsm, scale):
    # dem,_,_=read_dem_grid(dem_file)
    dx = 1 / scale
    # dx=0.5
    fy, fx = torch.gradient(dsm, dx, dx)
    asp, grad = cart2pol(fy, fx, "rad")
    grad = torch.arctan(grad)
    asp = asp * -1
    asp = asp + (asp < 0) * (torch.pi * 2)
    return grad, asp
