import math
import numpy as np
import pandas as pd
import torch
from ...functions.SOLWEIGpython.wall_cover import get_wall_cover
from .cylindric_wedge_torch import cylindric_wedge_voxel

SBC = 5.67051e-8


def load_walls(
    voxelTable,
    solweig_parameters,
    wall_type,
    wallaspect,
    Ta,
    timeStep,
    albedo_b,
    emissivity_b,
    albedo_grid,
    landcover,
    lcgrid,
    dsm,
):
    """Loads the voxel data created in the sky view factor calculator into a
    Pandas DataFrame. Runs once at setup, not per-timestep, so this stays
    pandas/numpy throughout - wallaspect/albedo_grid/dsm/lcgrid are the only
    torch tensors involved, and only because that's what the rest of the
    per-timestep pipeline keeps them as. Every value pulled out of them
    below is unwrapped with .item() before it reaches the DataFrame, so
    nothing torch-specific leaks into voxelTable's columns.
    """
    if torch.is_tensor(voxelTable):
        voxelTable = voxelTable.cpu().numpy()

    voxelTable = pd.DataFrame(
        voxelTable,
        columns=[
            "voxelId", "voxelHeight", "wallHeight", "wallHeight_exact",
            "wallId", "ypos", "xpos", "SVF_height", "SVF", "SVF_fix",
            "svfbu", "svfveg", "svfaveg",
        ],
    )

    voxelTable["wallTemperature"] = Ta
    voxelTable["timeStep"] = timeStep

    columns_to_add = [
        "SVF_ground", "svfbu_at_ground", "svfaveg_at_ground", "wallAspect",
        "wallEmissivity", "wallThickness", "wallAlbedo", "thermalEffusivity",
        "thermalDiffusivity", "groundAlbedo", "wallShade", "wallShadeHeight",
        "LongwaveRadiation", "K_in", "L_in", "Lwallsun", "Lwallsh", "Lrefl",
        "Lveg", "Lground", "Lsky", "esky", "voxelHeightMasl",
    ]
    for col in columns_to_add:
        voxelTable[col] = 0.0

    tmp = voxelTable["SVF_fix"].to_numpy() + voxelTable["svfveg"].to_numpy() - 1.0
    tmp[tmp < 0.0] = 0.0
    voxelTable["svfalfa"] = np.arcsin(np.exp((np.log((1.0 - tmp)) / 2.0)))

    temp_table = np.column_stack(
        [
            voxelTable["wallId"].to_numpy(),
            voxelTable["ypos"].to_numpy(),
            voxelTable["xpos"].to_numpy(),
        ]
    )
    temp_table = np.unique(temp_table, axis=0)

    for i in np.arange(temp_table.shape[0]):
        y = int(temp_table[i, 1])
        x = int(temp_table[i, 2])
        # indexing a torch raster with plain ints returns a 0-d VIEW tensor -
        # .item() unwraps it to a plain float before it goes anywhere near
        # the DataFrame, same pattern as everywhere else in this port
        temp_aspect = wallaspect[y, x].item()
        voxelTable.loc[
            voxelTable["wallId"] == temp_table[i, 0], "wallAspect"
        ] = temp_aspect

        temp_building = (
            voxelTable.loc[
                (
                    (voxelTable["wallId"] == temp_table[i, 0])
                    & (voxelTable["voxelHeight"] == voxelTable["voxelHeight"].min())
                ),
                "svfbu",
            ]
            .copy()
            .to_numpy()[0]
        )
        temp_veg = (
            voxelTable.loc[
                (
                    (voxelTable["wallId"] == temp_table[i, 0])
                    & (voxelTable["voxelHeight"] == voxelTable["voxelHeight"].min())
                ),
                "svfaveg",
            ]
            .copy()
            .to_numpy()[0]
        )
        temp_albedo = albedo_grid[y, x].item()
        temp_dsm = dsm[y, x].item()

        voxelTable.loc[
            voxelTable["wallId"] == temp_table[i, 0], "svfbu_at_ground"
        ] = temp_building
        voxelTable.loc[
            voxelTable["wallId"] == temp_table[i, 0], "svfaveg_at_ground"
        ] = temp_veg
        voxelTable.loc[
            voxelTable["wallId"] == temp_table[i, 0], "groundAlbedo"
        ] = temp_albedo
        voxelTable.loc[
            voxelTable["wallId"] == temp_table[i, 0], "voxelHeightMasl"
        ] = (
            voxelTable.loc[
                voxelTable["wallId"] == temp_table[i, 0], "voxelHeight"
            ].to_numpy()
            + temp_dsm
        )

    voxelTable = voxelTable.set_index("voxelId")

    building_fraction = 1 - voxelTable["svfbu_at_ground"].to_numpy() - 0.5
    building_fraction[building_fraction < 0] = 0.0
    veg_fraction = 1 - voxelTable["svfaveg_at_ground"].to_numpy() - 0.5
    veg_fraction[veg_fraction < 0] = 0.0
    voxelTable["building_fraction"] = building_fraction
    voxelTable["veg_fraction"] = veg_fraction
    sky_fraction = voxelTable["SVF_fix"].to_numpy()
    ground_fraction = 1 - sky_fraction - building_fraction - veg_fraction
    voxelTable["ground_fraction"] = ground_fraction
    voxelTable["total_fraction"] = (
        building_fraction + sky_fraction + ground_fraction + veg_fraction
    )

    if landcover == 1:
        # np.unique doesn't accept a CUDA tensor - route through torch.unique
        # first when lcgrid is a tensor, otherwise fall back to plain numpy
        if torch.is_tensor(lcgrid):
            unique_landcover = torch.unique(lcgrid).cpu().numpy()
        else:
            unique_landcover = np.unique(lcgrid)
        unique_walls = unique_landcover[unique_landcover > 99].astype(int)

        if unique_walls.size > 1:
            voxelTable = get_wall_cover(voxelTable, lcgrid, dsm, solweig_parameters)
        elif unique_walls.size == 1:
            wallTc = solweig_parameters["Specific_heat"]["Value"][
                solweig_parameters["Names"]["Value"][str(unique_walls[0])]
            ]
            wallTk = solweig_parameters["Thermal_conductivity"]["Value"][
                solweig_parameters["Names"]["Value"][str(unique_walls[0])]
            ]
            wallD = solweig_parameters["Density"]["Value"][
                solweig_parameters["Names"]["Value"][str(unique_walls[0])]
            ]
            wallTu = np.sqrt(wallTc * wallD * wallTk)
            wallTd = wallTk / (wallTc * wallD)
            voxelTable["thermalEffusivity"] = wallTu
            voxelTable["thermalDiffusivity"] = wallTd
            voxelTable["wallAlbedo"] = albedo_b
            voxelTable["wallEmissivity"] = emissivity_b
            voxelTable["wallThickness"] = solweig_parameters["Wall_thickness"][
                "Value"
            ][solweig_parameters["Names"]["Value"][str(unique_walls[0])]]
        else:
            landcover = 0

    if landcover == 0:
        wallTc = solweig_parameters["Specific_heat"]["Value"][
            solweig_parameters["Names"]["Value"][wall_type]
        ]
        wallTk = solweig_parameters["Thermal_conductivity"]["Value"][
            solweig_parameters["Names"]["Value"][wall_type]
        ]
        wallD = solweig_parameters["Density"]["Value"][
            solweig_parameters["Names"]["Value"][wall_type]
        ]
        wallTu = np.sqrt(wallTc * wallD * wallTk)
        voxelTable["thermalEffusivity"] = wallTu
        wallTd = wallTk / (wallTc * wallD)
        voxelTable["thermalDiffusivity"] = wallTd
        voxelTable["wallAlbedo"] = albedo_b
        voxelTable["wallEmissivity"] = emissivity_b
        voxelTable["wallThickness"] = solweig_parameters["Wall_thickness"]["Value"][
            solweig_parameters["Names"]["Value"][wall_type]
        ]

    eqTime = True
    if eqTime:
        voxelTable["timeStep"] = voxelTable["wallThickness"].to_numpy() ** 2 / (
            np.pi**2 * voxelTable["thermalDiffusivity"].to_numpy()
        )

    return voxelTable, wallaspect


def step_heating(q, e, t):
    """Delta surface temperature from heat flux (q), thermal effusivity (e),
    and time (t). q/e are always tensors here; t may arrive as either a
    plain Python float or a tensor (see surface_temperature_calc), so it's
    normalized once at the top - torch.sqrt refuses a bare Python float the
    same way torch.exp/abs/cos do elsewhere in this codebase."""
    t = torch.as_tensor(t, device=q.device, dtype=q.dtype)
    return (2 * q) / e * torch.sqrt(t / torch.pi)


def surface_temperature_calc(effusivity, t, Kin, Lin, Ta, wall_emissivity, Ts_previous):
    """Two-pass step-heating estimate of wall surface temperature."""
    Lout_temp = wall_emissivity * SBC * (Ts_previous + 273.15) ** 4
    energy_in_temp = Kin + Lin - Lout_temp
    dT = step_heating(energy_in_temp, effusivity, t)

    Ts_current = Ta + dT
    Lout_temp = wall_emissivity * SBC * (Ts_current + 273.15) ** 4
    energy_in_temp = Kin + Lin - Lout_temp
    dT = step_heating(energy_in_temp, effusivity, t)
    Ts = Ta + dT

    return Ts, dT

def build_static_wall_tensors(voxelTable, device, dtype=torch.float32):
    """Call ONCE, immediately after load_walls - not per-timestep. Bundles
    every per-voxel column that never changes across the run into tensors
    resident on `device`, so wall_surface_temperature stops re-uploading
    the same unchanging data from CPU on every single call."""

    def _col(name):
        return torch.tensor(voxelTable[name].to_numpy(), device=device, dtype=dtype)

    static = {
        "svfalfa": _col("svfalfa"),
        "wall_aspect": _col("wallAspect"),
        "building_fraction": _col("building_fraction"),
        "veg_fraction": _col("veg_fraction"),
        "ground_fraction": _col("ground_fraction"),
        "svf_fix": _col("SVF_fix"),
        "wall_emissivity": _col("wallEmissivity"),
        "wall_albedo": _col("wallAlbedo"),
        "ground_albedo": _col("groundAlbedo"),
        "thermal_effusivity": _col("thermalEffusivity"),
        "wall_height_exact": _col("wallHeight_exact"),
        "voxel_height": _col("voxelHeight"),
    }

    wall_ids = voxelTable["wallId"].to_numpy()
    unique_ids, first_idx, inverse_idx = np.unique(
        wall_ids, return_index=True, return_inverse=True
    )
    ypos_np = voxelTable["ypos"].to_numpy()
    xpos_np = voxelTable["xpos"].to_numpy()
    static["wall_y_per_unique"] = torch.tensor(ypos_np[first_idx], device=device, dtype=torch.long)
    static["wall_x_per_unique"] = torch.tensor(xpos_np[first_idx], device=device, dtype=torch.long)
    static["wall_inverse_idx"] = torch.tensor(inverse_idx, device=device, dtype=torch.long)

    return static


def wall_surface_temperature(
    voxelTable, static, wallsh, altitude, azimuth, timeStep,
    K_direct, K_diff, K_down, Ldown, Lup, Ta, esky, device, debug=False,
):
    deg2rad = torch.pi / 180

    def _scalar(x):
        return x.item() if torch.is_tensor(x) else x

    altitude_s = _scalar(altitude)
    azimuth_s = _scalar(azimuth)
    n_voxel = static["svfalfa"].shape[0]

    if altitude_s > 0:
        temp_sh_per_wall = wallsh[static["wall_y_per_unique"], static["wall_x_per_unique"]]
        wall_shade_height = temp_sh_per_wall[static["wall_inverse_idx"]]
        wall_shade_mask = (static["voxel_height"] >= wall_shade_height) & (
            static["wall_height_exact"] > wall_shade_height
        )
    else:
        wall_shade_height = static["wall_height_exact"]
        wall_shade_mask = torch.zeros(n_voxel, dtype=torch.bool, device=device)

    if debug:   # <-- add this block
            print(f"GPU sunlit voxel count: {wall_shade_mask.sum().item()} / {n_voxel}")


    voxelTable["wallShadeHeight"] = wall_shade_height.cpu().numpy()
    voxelTable["wallShade"] = wall_shade_mask.cpu().numpy().astype(int)

    ypos_idx = torch.tensor(voxelTable["ypos"].to_numpy(), device=device, dtype=torch.long)
    xpos_idx = torch.tensor(voxelTable["xpos"].to_numpy(), device=device, dtype=torch.long)
    Ldown_array = Ldown[ypos_idx, xpos_idx]
    Lup_array = Lup[ypos_idx, xpos_idx]

    def _col(name, dtype=torch.float32):
        return torch.tensor(voxelTable[name].to_numpy(), device=device, dtype=dtype)

    svfalfa = static["svfalfa"]
    wall_aspect = static["wall_aspect"]
    building_fraction = static["building_fraction"]
    veg_fraction = static["veg_fraction"]
    ground_fraction = static["ground_fraction"]
    svf_fix = static["svf_fix"]
    wall_emissivity = static["wall_emissivity"]
    wall_albedo = static["wall_albedo"]
    ground_albedo = static["ground_albedo"]
    thermal_effusivity = static["thermal_effusivity"]

    wall_temperature_prev = _col("wallTemperature")

    if altitude_s > 0:
        F_sh = cylindric_wedge_voxel((90 - altitude_s) * deg2rad, svfalfa)
        F_sh = torch.nan_to_num(F_sh, nan=0.5)
        F_sh = 2.0 * F_sh - 1.0

        wallSun = torch.abs(wall_aspect - azimuth_s) / 180.0
        wallSun = torch.where(wallSun > 1.0, 2 - wallSun, wallSun)
        wallSun = 0.2 + wallSun * 0.6

        ts_shade = wall_temperature_prev[~wall_shade_mask].mean()
        ts_sun = wall_temperature_prev[wall_shade_mask].mean()

        Lwallsun = (
            SBC * wall_emissivity * ((ts_sun + 273.15) ** 4)
            * building_fraction * (1.0 - F_sh)
        ) * wallSun
        Lwallsh = (
            SBC * wall_emissivity * ((ts_shade + 273.15) ** 4)
            * building_fraction * (1.0 - F_sh)
        ) * (1 - wallSun)
        Lwallsh = Lwallsh + (
            SBC * wall_emissivity * ((ts_shade + 273.15) ** 4)
            * building_fraction * F_sh
        )
    else:
        ts_shade = wall_temperature_prev[~wall_shade_mask].mean()
        Lwallsun = torch.zeros(n_voxel, device=device, dtype=torch.float32)
        Lwallsh = SBC * wall_emissivity * (ts_shade + 273.15) ** 4 * building_fraction
        F_sh = torch.zeros(n_voxel, device=device, dtype=torch.float32)
        wallSun = torch.zeros(n_voxel, device=device, dtype=torch.float32)

    Lveg = SBC * wall_emissivity * (Ta + 273.15) ** 4 * veg_fraction
    Lsky = SBC * esky * (Ta + 273.15) ** 4 * svf_fix
    Lrefl = (1.0 - wall_emissivity) * (Ldown_array + Lup_array) * building_fraction
    Lground = Lup_array * ground_fraction
    L_in = Lwallsun + Lwallsh + Lrefl + Lveg + Lground + Lsky

    voxelTable["Lwallsun"] = Lwallsun.cpu().numpy()
    voxelTable["Lwallsh"] = Lwallsh.cpu().numpy()
    voxelTable["Lrefl"] = Lrefl.cpu().numpy()
    voxelTable["Lveg"] = Lveg.cpu().numpy()
    voxelTable["Lground"] = Lground.cpu().numpy()
    voxelTable["Lsky"] = Lsky.cpu().numpy()
    voxelTable["esky"] = esky
    voxelTable["F_sh"] = F_sh.cpu().numpy() if altitude_s > 0 else 0.0
    voxelTable["wallSun"] = wallSun.cpu().numpy() if altitude_s > 0 else 0.0

    sun_x = (
        math.cos(math.radians(altitude_s)) * math.cos(math.radians(azimuth_s))
        * torch.cos(deg2rad * wall_aspect)
    )
    sun_y = (
        math.cos(math.radians(altitude_s)) * math.sin(math.radians(azimuth_s))
        * torch.sin(deg2rad * wall_aspect)
    )
    cf = torch.clamp(sun_x + sun_y, min=0.0)

    if altitude_s > 0:
        K_in = (1 - wall_albedo) * (
            K_direct * cf * wall_shade_mask.float()
            + K_diff * svf_fix
            + K_down * wall_albedo * building_fraction
            + (K_down * ground_albedo) * ground_fraction
        )
    else:
        K_in = torch.zeros(n_voxel, device=device, dtype=torch.float32)

    voxelTable["K_in"] = K_in.cpu().numpy()
    voxelTable["L_in"] = L_in.cpu().numpy()

    if voxelTable["timeStep"].unique().size == 1:
        timeStep_val = float(voxelTable.iloc[0]["timeStep"])
    else:
        timeStep_val = _col("timeStep")

    Ts, dT = surface_temperature_calc(
        thermal_effusivity, timeStep_val, K_in, L_in, Ta, wall_emissivity, wall_temperature_prev
    )
    voxelTable["wallTemperature"] = Ts.cpu().numpy()
    voxelTable["LongwaveRadiation"] = (
        (wall_emissivity * SBC * (Ts + 273.15) ** 4) / torch.pi
    ).cpu().numpy()

    voxelTable["sunAltitude"] = altitude_s
    voxelTable["sunAzimuth"] = azimuth_s

    return voxelTable

# def wall_surface_temperature(
#     voxelTable, wallsh, altitude, azimuth, timeStep,
#     K_direct, K_diff, K_down, Ldown, Lup, Ta, esky, device,
# ):
#     """Wall surface temperature parameterization. voxelTable stays a
#     DataFrame for structure/bookkeeping; per-voxel numeric work runs on
#     `device` as torch tensors, written back to the DataFrame at the end."""
#     deg2rad = torch.pi / 180

#     def _scalar(x):
#         return x.item() if torch.is_tensor(x) else x

#     altitude_s = _scalar(altitude)
#     azimuth_s = _scalar(azimuth)
#     n_voxel = voxelTable.shape[0]

#     voxelTable["wallShade"] = 0.0

#     if altitude_s > 0:
#         voxelTable["wallShadeHeight"] = 0.0
#         unique_walls = np.unique(voxelTable["wallId"])
#         for unique_wall in unique_walls:
#             rows = voxelTable.wallId == unique_wall
#             y0 = int(voxelTable.loc[rows, "ypos"].to_numpy()[0])
#             x0 = int(voxelTable.loc[rows, "xpos"].to_numpy()[0])
#             temp_sh = wallsh[y0, x0].item()
#             voxelTable.loc[
#                 rows
#                 & (voxelTable["voxelHeight"] >= temp_sh)
#                 & (voxelTable["wallHeight_exact"] > temp_sh),
#                 "wallShade",
#             ] = 1
#             voxelTable.loc[rows, "wallShadeHeight"] = temp_sh
#         # This loops over unique WALLS, not voxels, so it's much cheaper
#         # than the old per-voxel loop below was - but if it shows up in
#         # profiling, it can be fully vectorized by having load_walls
#         # precompute a static wallId -> (ypos, xpos) lookup once, since that
#         # mapping never changes between timesteps. Say the word if you want
#         # that done.
#     else:
#         voxelTable["wallShadeHeight"] = voxelTable["wallHeight_exact"]

#     # Fully vectorized replacement for the original per-voxel Python loop -
#     # same gather, one indexing op instead of one Python iteration per voxel
#     ypos_idx = torch.as_tensor(voxelTable["ypos"].to_numpy(), device=device, dtype=torch.long)
#     xpos_idx = torch.as_tensor(voxelTable["xpos"].to_numpy(), device=device, dtype=torch.long)
#     Ldown_array = Ldown[ypos_idx, xpos_idx]
#     Lup_array = Lup[ypos_idx, xpos_idx]

#     # def _col(name, dtype=torch.float32):
#     #     return torch.as_tensor(voxelTable[name].to_numpy(), device=device, dtype=dtype)

#     def _col(name, dtype=torch.float32):
#         return torch.tensor(voxelTable[name].to_numpy(), device=device, dtype=dtype)

#     svfalfa = _col("svfalfa")
#     wall_aspect = _col("wallAspect")
#     building_fraction = _col("building_fraction")
#     veg_fraction = _col("veg_fraction")
#     ground_fraction = _col("ground_fraction")
#     svf_fix = _col("SVF_fix")
#     wall_emissivity = _col("wallEmissivity")
#     wall_albedo = _col("wallAlbedo")
#     ground_albedo = _col("groundAlbedo")
#     wall_temperature_prev = _col("wallTemperature")
#     thermal_effusivity = _col("thermalEffusivity")
#     wall_shade_mask = _col("wallShade", dtype=torch.bool)

#     if altitude_s > 0:
#         # cylindric_wedge_voxel needs its own torch version, imported above
#         # as cylindric_wedge_torch - I haven't seen that file, so this
#         # assumes it takes/returns a 1-D per-voxel tensor the same shape-for-
#         # shape as the numpy version's 1-D array. Worth confirming.
#         F_sh = cylindric_wedge_voxel((90 - altitude_s) * deg2rad, svfalfa)
#         F_sh = torch.nan_to_num(F_sh, nan=0.5)
#         F_sh = 2.0 * F_sh - 1.0

#         wallSun = torch.abs(wall_aspect - azimuth_s) / 180.0
#         wallSun = torch.where(wallSun > 1.0, 2 - wallSun, wallSun)
#         wallSun = 0.2 + wallSun * 0.6

#         ts_shade = wall_temperature_prev[~wall_shade_mask].mean()
#         ts_sun = wall_temperature_prev[wall_shade_mask].mean()

#         Lwallsun = (
#             SBC * wall_emissivity * ((ts_sun + 273.15) ** 4)
#             * building_fraction * (1.0 - F_sh)
#         ) * wallSun
#         Lwallsh = (
#             SBC * wall_emissivity * ((ts_shade + 273.15) ** 4)
#             * building_fraction * (1.0 - F_sh)
#         ) * (1 - wallSun)
#         Lwallsh = Lwallsh + (
#             SBC * wall_emissivity * ((ts_shade + 273.15) ** 4)
#             * building_fraction * F_sh
#         )
#     else:
#         ts_shade = wall_temperature_prev[~wall_shade_mask].mean()
#         Lwallsun = torch.zeros(n_voxel, device=device, dtype=torch.float32)
#         Lwallsh = SBC * wall_emissivity * (ts_shade + 273.15) ** 4 * building_fraction
#         F_sh = torch.zeros(n_voxel, device=device, dtype=torch.float32)
#         wallSun = torch.zeros(n_voxel, device=device, dtype=torch.float32)

#     Lveg = SBC * wall_emissivity * (Ta + 273.15) ** 4 * veg_fraction
#     Lsky = SBC * esky * (Ta + 273.15) ** 4 * svf_fix
#     Lrefl = (1.0 - wall_emissivity) * (Ldown_array + Lup_array) * building_fraction
#     Lground = Lup_array * ground_fraction
#     L_in = Lwallsun + Lwallsh + Lrefl + Lveg + Lground + Lsky

#     voxelTable["Lwallsun"] = Lwallsun.cpu().numpy()
#     voxelTable["Lwallsh"] = Lwallsh.cpu().numpy()
#     voxelTable["Lrefl"] = Lrefl.cpu().numpy()
#     voxelTable["Lveg"] = Lveg.cpu().numpy()
#     voxelTable["Lground"] = Lground.cpu().numpy()
#     voxelTable["Lsky"] = Lsky.cpu().numpy()
#     voxelTable["esky"] = esky
#     voxelTable["F_sh"] = F_sh.cpu().numpy() if altitude_s > 0 else 0.0
#     voxelTable["wallSun"] = wallSun.cpu().numpy() if altitude_s > 0 else 0.0

#     sun_x = (
#         math.cos(math.radians(altitude_s)) * math.cos(math.radians(azimuth_s))
#         * torch.cos(deg2rad * wall_aspect)
#     )
#     sun_y = (
#         math.cos(math.radians(altitude_s)) * math.sin(math.radians(azimuth_s))
#         * torch.sin(deg2rad * wall_aspect)
#     )
#     cf = torch.clamp(sun_x + sun_y, min=0.0)

#     if altitude_s > 0:
#         wall_shade_col = _col("wallShade")
#         K_in = (1 - wall_albedo) * (
#             K_direct * cf * wall_shade_col
#             + K_diff * svf_fix
#             + K_down * wall_albedo * building_fraction
#             + (K_down * ground_albedo) * ground_fraction
#         )
#     else:
#         K_in = torch.zeros(n_voxel, device=device, dtype=torch.float32)

#     voxelTable["K_in"] = K_in.cpu().numpy()
#     voxelTable["L_in"] = L_in.cpu().numpy()

#     Ts_previous = wall_temperature_prev
#     if voxelTable["timeStep"].unique().size == 1:
#         timeStep_val = float(voxelTable.iloc[0]["timeStep"])
#     else:
#         timeStep_val = _col("timeStep")

#     Ts, dT = surface_temperature_calc(
#         thermal_effusivity, timeStep_val, K_in, L_in, Ta, wall_emissivity, Ts_previous
#     )
#     voxelTable["wallTemperature"] = Ts.cpu().numpy()
#     voxelTable["LongwaveRadiation"] = (
#         (wall_emissivity * SBC * (Ts + 273.15) ** 4) / torch.pi
#     ).cpu().numpy()

#     voxelTable["sunAltitude"] = altitude_s
#     voxelTable["sunAzimuth"] = azimuth_s

#     return voxelTable