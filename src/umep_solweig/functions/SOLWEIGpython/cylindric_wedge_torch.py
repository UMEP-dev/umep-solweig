try:
    import torch
except:
    pass


def cylindric_wedge(zen, svfalfa, rows, cols):

    # Fraction of sunlit walls based on sun altitude and svf wieghted building angles
    # input:
    # sun zenith angle "beta"
    # svf related angle "alfa"
    try:
        import torch
    except:
        Exception("Error, pytorch must be imported.")

    device = (
        svfalfa.device
        if isinstance(svfalfa, torch.Tensor)
        else torch.device(
            "cuda"
            if torch.cuda.is_available()
            else (
                "xpu"
                if (hasattr(torch, "xpu") and torch.xpu.is_available())
                else "cpu"
            )
        )
    )
    beta = zen
    # alfa=svfalfa
    alfa = torch.zeros((rows, cols), device=device) + svfalfa
    # measure the size of the image
    # sizex=size(svfalfa,2)
    # sizey=size(svfalfa,1)

    xa = 1 - 2.0 / (torch.tan(alfa) * torch.tan(beta))
    ha = 2.0 / (torch.tan(alfa) * torch.tan(beta))
    ba = 1.0 / torch.tan(alfa)
    hkil = 2.0 * ba * ha

    qa = torch.zeros((rows, cols), device=device)
    # qa(length(svfalfa),length(svfalfa))=0;
    qa[xa < 0] = torch.tan(beta) / 2

    Za = torch.zeros((rows, cols), device=device)
    # Za(length(svfalfa),length(svfalfa))=0;
    Za[xa < 0] = (((ba[xa < 0] ** 2) - ((qa[xa < 0] ** 2) / 4)) ** 0.5).float()

    phi = torch.zeros((rows, cols), device=device)
    # phi(length(svfalfa),length(svfalfa))=0;
    phi[xa < 0] = torch.arctan(Za[xa < 0] / qa[xa < 0])

    A = torch.zeros((rows, cols), device=device)
    # A(length(svfalfa),length(svfalfa))=0;
    A[xa < 0] = (
        torch.sin(phi[xa < 0]) - phi[xa < 0] * torch.cos(phi[xa < 0])
    ) / (1 - torch.cos(phi[xa < 0]))

    ukil = torch.zeros((rows, cols), device=device)
    # ukil(length(svfalfa),length(svfalfa))=0
    ukil[xa < 0] = (2 * ba[xa < 0] * xa[xa < 0] * A[xa < 0]).float()

    Ssurf = hkil + ukil

    F_sh = (2 * torch.pi * ba - Ssurf) / (2 * torch.pi * ba)  # Xa
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "xpu":
        torch.xpu.empty_cache()
    return F_sh

def cylindric_wedge_voxel(zen, svfalfa):
    """Fraction of sunlit walls based on sun zenith angle and svf-weighted
    building angles, per voxel.

    zen is normalized to a tensor up front - it arrives as a plain Python
    float from wall_surface_temperature's ((90 - altitude) * deg2rad) call,
    and torch.tan/atan/sin/cos all reject bare floats the same way
    torch.exp/abs/cos do elsewhere in this codebase.

    No torch equivalent of np.seterr() is needed - confirmed separately
    that torch never raises a warning for divide-by-zero or 0/0 in the
    first place (both silently produce inf/nan), so there's nothing to
    suppress. The nan this can produce at edge-case angles is expected;
    wall_surface_temperature already handles it via
    torch.nan_to_num(F_sh, nan=0.5) right after calling this.
    """
    beta = torch.as_tensor(zen, device=svfalfa.device, dtype=svfalfa.dtype)

    xa = 1 - 2.0 / (torch.tan(svfalfa) * torch.tan(beta))
    ha = 2.0 / (torch.tan(svfalfa) * torch.tan(beta))
    ba = 1.0 / torch.tan(svfalfa)
    hkil = 2.0 * ba * ha

    # computed once and reused, rather than re-evaluating `xa < 0` five
    # times like the numpy original does - same result, less redundant work
    mask = xa < 0

    qa = torch.zeros_like(svfalfa)
    qa[mask] = torch.tan(beta) / 2

    Za = torch.zeros_like(svfalfa)
    Za[mask] = ((ba[mask] ** 2) - ((qa[mask] ** 2) / 4)) ** 0.5

    phi = torch.zeros_like(svfalfa)
    phi[mask] = torch.atan(Za[mask] / qa[mask])

    A = torch.zeros_like(svfalfa)
    A[mask] = (torch.sin(phi[mask]) - phi[mask] * torch.cos(phi[mask])) / (
        1 - torch.cos(phi[mask])
    )

    ukil = torch.zeros_like(svfalfa)
    ukil[mask] = 2 * ba[mask] * xa[mask] * A[mask]

    Ssurf = hkil + ukil
    F_sh = (2 * torch.pi * ba - Ssurf) / (2 * torch.pi * ba)

    return F_sh
