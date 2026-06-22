import numpy as np

def patch_steradians(L_patches):
    """'This function calculates the steradians of the patches"""

    # Degrees to radians
    deg2rad = np.pi / 180

    # Unique altitudes for patches
    skyalt, skyalt_c = np.unique(L_patches[:, 0], return_counts=True)

    # Altitudes of the Robinson & Stone patches
    patch_altitude = L_patches[:, 0]

    # Calculation of steradian for each patch
    steradian = np.zeros((patch_altitude.shape[0]))
    for i in range(patch_altitude.shape[0]):
        # If there are more than one patch in a band
        if skyalt_c[skyalt == patch_altitude[i]] > 1:
            steradian[i] = (
                (360 / skyalt_c[skyalt == patch_altitude[i]]) * deg2rad
            ) * (
                np.sin((patch_altitude[i] + patch_altitude[0]) * deg2rad)
                - np.sin((patch_altitude[i] - patch_altitude[0]) * deg2rad)
            )
        # If there is only one patch in band, i.e. 90 degrees
        else:
            steradian[i] = (
                (360 / skyalt_c[skyalt == patch_altitude[i]]) * deg2rad
            ) * (
                np.sin((patch_altitude[i]) * deg2rad)
                - np.sin((patch_altitude[i - 1] + patch_altitude[0]) * deg2rad)
            )

    return steradian, skyalt, patch_altitude
