import numpy as np
from . import sunlit_shaded_patches


""" This function creates a hemispheric image based on patch characteristics """


def hemispheric_image(poi, sh, vegsh, vbshvegsh, voxelmat, wallScheme):
    patch_characteristics = np.zeros((sh.shape[2], poi.shape[0]))
    for idx in range(poi.shape[0]):
        for idy in range(sh.shape[2]):
            # Calculations for patches on sky, shmat = 1 = sky is visible
            temp_sky = (sh[:, :, idy] == 1) & (vegsh[:, :, idy] == 1)
            # Calculations for patches that are vegetation, vegshmat = 0 = shade from vegetation
            temp_vegsh = (vegsh[:, :, idy] == 0) | (vbshvegsh[:, :, idy] == 0)
            # Calculations for patches that are buildings, shmat = 0 = shade from buildings
            temp_vbsh = (1 - sh[:, :, idy]) * vbshvegsh[:, :, idy]
            temp_sh = temp_vbsh == 1
            if wallScheme:
                temp_sh_w = temp_sh * voxelmat[:, :, idy]
                temp_sh_roof = temp_sh * (voxelmat[:, :, idy] == 0)
            else:
                temp_sh_w = 0
                temp_sh_roof = 0
            # Sky patch
            if temp_sky[int(poi[idx, 2]), int(poi[idx, 1])]:
                patch_characteristics[idy, idx] = 1.8
            # Vegetation patch
            elif temp_vegsh[int(poi[idx, 2]), int(poi[idx, 1])]:
                patch_characteristics[idy, idx] = 2.5
            # Building patch
            elif temp_sh[int(poi[idx, 2]), int(poi[idx, 1])]:
                if wallScheme:
                    if temp_sh_w[int(poi[idx, 2]), int(poi[idx, 1])]:
                        patch_characteristics[idy, idx] = 4.5
                    elif temp_sh_roof[int(poi[idx, 2]), int(poi[idx, 1])]:
                        # patch_characteristics[idy, idx] = 6.0
                        patch_characteristics[idy, idx] = 4.5
                else:
                    patch_characteristics[idy, idx] = 4.5
            # Roof patch
            # elif
    return patch_characteristics
