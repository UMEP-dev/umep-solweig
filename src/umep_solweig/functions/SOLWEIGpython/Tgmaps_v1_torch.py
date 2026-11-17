try:
    import torch
except ImportError:
    torch = None


def Tgmaps_v1(lc_grid, solweig_parameters):

    # Tgmaps_v1 Populates grids with cooeficients for Tg wave
    #   Detailed explanation goes here
    lc_grid[lc_grid >= 100] = 2
    # One pass per land-cover class. The masks come from the class grid, not
    # from the grids being filled: a value already written (water has
    # Ts_deg = 0 and Tstart = 0) must not be read as a class id later.
    id = torch.unique(lc_grid)
    id = id[id <= 7].to(int)
    TgK = torch.clone(lc_grid)
    Tstart = torch.clone(lc_grid)
    alb_grid = torch.clone(lc_grid)
    emis_grid = torch.clone(lc_grid)
    TmaxLST = torch.clone(lc_grid)

    for i in id:
        mask = lc_grid == i
        name = solweig_parameters["Names"]["Value"][str(int(i.item()))]
        Tstart[mask] = solweig_parameters["Tstart"]["Value"][name]
        alb_grid[mask] = solweig_parameters["Albedo"]["Effective"]["Value"][
            name
        ]
        emis_grid[mask] = solweig_parameters["Emissivity"]["Value"][name]
        TmaxLST[mask] = solweig_parameters["TmaxLST"]["Value"][name]
        TgK[mask] = solweig_parameters["Ts_deg"]["Value"][name]

    TgK_wall = solweig_parameters["Ts_deg"]["Value"]["Walls"]
    Tstart_wall = solweig_parameters["Tstart"]["Value"]["Walls"]
    TmaxLST_wall = solweig_parameters["TmaxLST"]["Value"]["Walls"]

    return (
        TgK,
        Tstart,
        alb_grid,
        emis_grid,
        TgK_wall,
        Tstart_wall,
        TmaxLST,
        TmaxLST_wall,
    )
