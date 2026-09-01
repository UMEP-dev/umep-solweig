import torch

""" Model 1 is based on Unsworth & Monteith, 1975 """


def model1(sky_patches, esky, Ta):
    device = sky_patches.device

    # Degrees to radians
    deg2rad = torch.pi / 180

    # Unique altitudes in lv, i.e. unique altitude for the patches
    skyalt, skyalt_c = torch.unique(sky_patches[:, 0], return_counts=True)

    # Unique zeniths for the patches
    skyzen = 90 - skyalt

    # Cosine of the zenith angles
    cosskyzen = torch.cos(skyzen * deg2rad)

    # Estimate emissivities at different altitudes/zenith angles

    # Constants?
    a_c = 0.67
    b_c = 0.094

    # Natural log of the reduced depth of precipitable water
    ln_u_prec = esky / b_c - a_c / b_c - 0.5

    # Reduced depth of precipitable water
    u_prec = torch.exp(torch.as_tensor(ln_u_prec, device=device))

    # Optical water depth
    owp = u_prec / cosskyzen

    # Natural log of optical water depth
    log_owp = torch.log(owp)

    # Emissivity of each zenith angle, i.e. the zenith angle of each band of patches
    esky_band = a_c + b_c * log_owp

    # Altitudes of the Robinson & Stone patches
    p_alt = sky_patches[:, 0]

    # Empty array with length based on number of patches
    patch_emissivity = torch.zeros(p_alt.shape[0], device=device)

    # Fill with emissivities
    for idx in skyalt:
        temp_emissivity = esky_band[skyalt == idx]
        patch_emissivity[p_alt == idx] = temp_emissivity

    # Normalize
    patch_emissivity_normalized = patch_emissivity / torch.sum(patch_emissivity)

    return patch_emissivity_normalized, esky_band


""" Model 2 is based on Martin & Berdhal, 1984
    Ez = 1 - (1 - Es)*e**(b2*(1.7 - (1/np.cos(z)))) """


def model2(sky_patches, esky, Ta):
    device = sky_patches.device

    # Degrees to radians
    deg2rad = torch.pi / 180

    # Unique altitudes in lv, i.e. unique altitude for the patches
    skyalt, skyalt_c = torch.unique(sky_patches[:, 0], return_counts=True)

    # Unique zeniths for the patches
    skyzen = 90 - skyalt

    # Constant (Ångström, 1915), proposed by Nahon et al., 2019
    b_c = 0.308
    # b_c = 0.1

    # Estimate emissivites at different altitudes/zenith angles
    esky_band = 1 - (1 - esky) * torch.exp(
        b_c * (1.7 - (1 / torch.cos(skyzen * deg2rad)))
    )

    # Altitudes of the Robinson & Stone patches
    p_alt = sky_patches[:, 0]

    # Empty array with length based on number of patches
    patch_emissivity = torch.zeros(p_alt.shape[0], device=device)

    # Fill with emissivities
    for idx in skyalt:
        temp_emissivity = esky_band[skyalt == idx]
        patch_emissivity[p_alt == idx] = temp_emissivity

    # Normalize
    patch_emissivity_normalized = patch_emissivity / torch.sum(patch_emissivity)

    return patch_emissivity_normalized, esky_band


""" Model 3 is based on Bliss, 1961.
    Ez = 1 - (1 - Es)**(1/(b1*np.cos(z))) """


def model3(sky_patches, esky, Ta):
    device = sky_patches.device

    # Degrees to radians
    deg2rad = torch.pi / 180

    # Unique altitudes in lv, i.e. unique altitude for the patches
    skyalt, skyalt_c = torch.unique(sky_patches[:, 0], return_counts=True)

    # Unique zeniths for the patches
    skyzen = 90 - skyalt

    # Constant, can (should?) be changed. Model gave unsatisfactory results in Nahon et al., 2019
    b_c = 1.8

    # Estimate emissivites at different altitudes/zenith angles
    esky_band = 1 - (1 - esky) ** (1 / (b_c * torch.cos(skyzen * deg2rad)))

    # Estimating longwave radiation for each patch to a horizontal or vertical surface

    # Altitudes of the Robinson & Stone patches
    p_alt = sky_patches[:, 0]

    # Empty array with length based on number of patches
    patch_emissivity = torch.zeros(p_alt.shape[0], device=device)

    # Fill with emissivities
    for idx in skyalt:
        temp_emissivity = esky_band[skyalt == idx]
        patch_emissivity[p_alt == idx] = temp_emissivity

    # Normalize
    patch_emissivity_normalized = patch_emissivity / torch.sum(patch_emissivity)

    return patch_emissivity_normalized, esky_band