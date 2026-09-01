import math
import numpy as np
import torch


def get_wall_cover(voxelTable, lcgrid, dsm, lc_params):
    """Sets thermal properties of wall pixels used in the wall surface
    temperature scheme.

    Called once from load_walls (setup, not per-timestep). The per-voxel
    loop is driven by Python dict lookups in lc_params, which can't be
    vectorized onto GPU without a bigger restructuring - so lcgrid/dsm are
    pulled to CPU once here rather than doing thousands of tiny individual
    CUDA reads. wallCode/wallTu/etc. stay plain numpy for the same reason:
    they're filled one Python scalar at a time regardless of backend.
    """
    lcgrid_cpu = lcgrid.cpu() if torch.is_tensor(lcgrid) else lcgrid
    dsm_cpu = dsm.cpu() if torch.is_tensor(dsm) else dsm

    ypos = voxelTable["ypos"].to_numpy().astype(int)
    xpos = voxelTable["xpos"].to_numpy().astype(int)

    wallCode = np.zeros((voxelTable.shape[0]), dtype=np.float32)
    wallTu = np.zeros((voxelTable.shape[0]), dtype=np.float32)
    wallTd = np.zeros((voxelTable.shape[0]), dtype=np.float32)
    wallAlbedo = np.zeros((voxelTable.shape[0]), dtype=np.float32)
    wallEmissivity = np.zeros((voxelTable.shape[0]), dtype=np.float32)
    wallThickness = np.zeros((voxelTable.shape[0]), dtype=np.float32)

    # dtype derived from lcgrid rather than hardcoded, so this doesn't
    # silently mismatch depending on how far the float32 migration has
    # landed by the time this runs
    domain = torch.tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=lcgrid_cpu.dtype)

    for i in range(voxelTable.shape[0]):
        # Preserved as-is from the numpy version: a wall pixel with ypos or
        # xpos == 0 gives a slice start of -1, which both numpy and torch
        # interpret as "count from the end" rather than "out of bounds" -
        # same pre-existing edge-pixel quirk either way, not something this
        # port changes or introduces.
        temp_lc = (
            lcgrid_cpu[ypos[i] - 1 : ypos[i] + 2, xpos[i] - 1 : xpos[i] + 2]
            * domain
        )
        temp_dsm = (
            dsm_cpu[ypos[i] - 1 : ypos[i] + 2, xpos[i] - 1 : xpos[i] + 2]
            * domain
        )
        # boolean-mask indexing always copies, same in torch as numpy -
        # no aliasing risk on temp_code below
        temp_code = temp_lc[(temp_lc > 99) & (temp_dsm == temp_dsm.max())]

        # Original had two branches (len>1 / len==1) doing the identical
        # thing - merged here, no behavior change, just fewer lines.
        # .item() is the important part: it forces a plain Python int
        # rather than leaving a 0-d tensor, since str(0-d tensor) prints as
        # "tensor(101)" rather than "101" - that would silently break every
        # lc_params dict lookup below with a KeyError.
        if len(temp_code) >= 1:
            temp_code = int(temp_code[0].item())
        elif temp_code.numel() == 0:
            temp_code = 101  # no wall type specified in landcover -> concrete

        wallCode[i] = temp_code

        temp_Tc = lc_params["Specific_heat"]["Value"][
            lc_params["Names"]["Value"][str(temp_code)]
        ]
        temp_Tk = lc_params["Thermal_conductivity"]["Value"][
            lc_params["Names"]["Value"][str(temp_code)]
        ]
        temp_D = lc_params["Density"]["Value"][
            lc_params["Names"]["Value"][str(temp_code)]
        ]
        wallTu[i] = math.sqrt(temp_Tc * temp_D * temp_Tk)
        wallTd[i] = temp_Tk / (temp_Tc * temp_D)

        wallAlbedo[i] = lc_params["Albedo"]["Material"]["Value"][
            lc_params["Names"]["Value"][str(temp_code)]
        ]
        wallEmissivity[i] = lc_params["Emissivity"]["Value"][
            lc_params["Names"]["Value"][str(temp_code)]
        ]
        wallThickness[i] = lc_params["Wall_thickness"]["Value"][
            lc_params["Names"]["Value"][str(temp_code)]
        ]

    voxelTable["thermalEffusivity"] = wallTu
    voxelTable["thermalDiffusivity"] = wallTd
    voxelTable["wallAlbedo"] = wallAlbedo
    voxelTable["wallEmissivity"] = wallEmissivity
    voxelTable["wallThickness"] = wallThickness

    return voxelTable