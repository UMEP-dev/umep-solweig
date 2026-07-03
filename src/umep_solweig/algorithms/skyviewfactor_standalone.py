# -*- coding: utf-8 -*-

"""Standalone Sky View Factor processing without QGIS dependencies.

This module mirrors the behavior of the QGIS processing algorithm while using
ordinary Python inputs and the existing SVF implementations in the package.
"""

from __future__ import annotations

import gc
import os
import warnings
from pathlib import Path
from typing import Any, Optional, Union
import sys
import zipfile

import numpy as np
from osgeo import gdal

from ..functions import svf_functions_torch as svf_torch
from ..functions import svf_for_voxels_torch as svfv_torch
from ..functions import svf_functions as svf
from ..functions import svf_for_voxels as svfv
from ..util import misc

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

__author__ = "Fredrik Lindberg"
__copyright__ = "(C) 2020-2026 by Fredrik Lindberg"


def _resolve_raster_input(
    raster_input: Union[str, os.PathLike, Any],
    *,
    name: str,
) -> tuple[Any, np.ndarray, Optional[Any]]:
    """Load a raster from a path, GDAL dataset, or NumPy array.

    The standalone implementation keeps the original GDAL-based import flow so
    it behaves like the QGIS processing algorithm for ordinary raster files.
    """
    if raster_input is None:
        raise ValueError(f"A {name} input is required")

    if isinstance(raster_input, (str, os.PathLike)):
        dataset = gdal.Open(str(raster_input))
        if dataset is None:
            raise ValueError(f"Unable to open {name} raster: {raster_input}")
        array = dataset.ReadAsArray().astype(float)
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        return dataset, array

    if hasattr(raster_input, "ReadAsArray") and hasattr(
        raster_input, "GetGeoTransform"
    ):
        dataset = raster_input
        array = dataset.ReadAsArray().astype(float)
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        return dataset, array, dataset

    array = np.asarray(raster_input, dtype=float)
    return None, array


def processAlgorithm(
    input_dsm: Union[str, os.PathLike, Any],
    input_cdsm: Optional[Union[str, os.PathLike, Any]] = None,
    transVeg: int = 3,
    input_tdsm: Optional[Union[str, os.PathLike, Any]] = None,
    input_theight: float = 25.0,
    use_gpu: bool = False,
    aniso: bool = True,
    wallScheme: bool = False,
    kmeans: bool = True,
    clusters: int = 5,
    input_dem: Optional[Union[str, os.PathLike, Any]] = None,
    svf_height: float = 1.0,
    outputDir: Optional[Union[str, os.PathLike]] = None,
    outputFile: Optional[Union[str, os.PathLike]] = None,
    feedback: Any = None,
) -> dict[str, Any]:
    """Run SVF processing using the existing SVF implementations.

    Parameters
    ----------
    input_dsm:
        DSM raster path, GDAL dataset, or NumPy array for the building and
        ground surface model.
    input_cdsm:
        Optional vegetation canopy DSM raster path, GDAL dataset, or NumPy
        array used to include vegetation in the SVF calculation.
    transVeg:
        Transmissivity of light through vegetation, expressed as a percentage.
    input_tdsm:
        Optional vegetation trunk-zone DSM raster path, GDAL dataset, or NumPy
        array. If omitted, the trunk zone is derived from the canopy DSM using
        input_theight.
    input_theight:
        Trunk-zone height as a percentage of canopy height when input_tdsm is
        not supplied.
    use_gpu:
        When True, attempt to run the torch-based implementation on GPU.
    aniso:
        When True, use the 153-shadow-image scheme; otherwise use the
        655-shadow-image scheme.
    wallScheme:
        When True, enable the wall-surface-temperature parameterization path,
        which requires input_dem.
    kmeans:
        When True, use K-Means clustering for wall SVF calculations.
    clusters:
        Number of clusters used by K-Means when kmeans is enabled.
    input_dem:
        DEM raster path, GDAL dataset, or NumPy array required for wallScheme.
    svf_height:
        Elevation step in meters used for the wall-scheme SVF calculations.
    outputDir:
        Output folder used to store generated raster and auxiliary files.
    outputFile:
        Output sky-view-factor raster path.
    feedback:
        Optional progress object used by the underlying SVF implementation.
    """
    if input_dsm is None:
        raise ValueError("A DSM input is required")

    device = None
    if use_gpu:
        # Safely identify if we are dealing with the local mock class
        if (
            type(torch).__name__ == "MetaMock"
            or getattr(torch, "__name__", "") == "LocalMockTorch"
        ):
            raise RuntimeError(
                "\n[UMEP Error] PyTorch is required to run GPU mode.\n"
                "Please install it using: pip install torch or with osgeo4w.\n"
                "Note:  setup for intel GPU require a more complex setup :\n"
                "pip install torch --index-url https://download.pytorch.org/whl/xpu"
            )

        if torch.cuda.is_available():
            device = torch.device("cuda")
            if feedback is not None:
                print("PyTorch and GPU found. Initiating GPU mode...")
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            device = torch.device("xpu")
            if feedback is not None:
                print("PyTorch and GPU found. Initiating GPU mode...")
        else:
            device = torch.device("cpu")
            if feedback is not None:
                print(
                    "Pytorch found but GPU not found. Initiating CPU mode..."
                )
    else:
        # Fall back to standard CPU processing
        if feedback is not None:
            print("Running in CPU mode...")

    gdal_dsm, dsm = _resolve_raster_input(input_dsm, name="DSM")
    if gdal_dsm is None:
        raise ValueError("Unable to open DSM raster")

    # response to issue #85
    nd = gdal_dsm.GetRasterBand(1).GetNoDataValue()
    dsm = np.array(dsm, dtype=float, copy=True)
    if nd is not None:
        dsm[dsm == nd] = 0.0
    if dsm.size and dsm.min() < 0:
        dsm = dsm + np.abs(dsm.min())

    sizex = dsm.shape[0]
    sizey = dsm.shape[1]

    geotransform = gdal_dsm.GetGeoTransform()
    pixel_resolution = geotransform[1]
    scale = 1 / pixel_resolution

    if use_gpu:
        dsm = torch.from_numpy(dsm).to(device)
        scale = torch.tensor(scale, device=device)

    if wallScheme:
        if input_dem is None:
            raise ValueError("DEM layer required for wall surface scheme!")
        gdal_dem, dem = _resolve_raster_input(input_dem, name="DEM")
        if gdal_dem is None:
            raise ValueError("Unable to open DEM raster")

        demsizex = dem.shape[0]
        demsizey = dem.shape[1]

        if not ((demsizex == sizex) and (demsizey == sizey)):
            raise Exception(
                "Error in DEM: All rasters must be of same extent and resolution"
            )
    else:
        dem = None

    trans = transVeg / 100.0

    if input_cdsm is not None:
        usevegdem = 1
        print("Vegetation scheme activated")

        _, vegdsm = _resolve_raster_input(
            input_cdsm, name="Vegetation Canopy DSM"
        )
        vegdsm = np.array(vegdsm, dtype=float, copy=True)

        vegsizex = vegdsm.shape[0]
        vegsizey = vegdsm.shape[1]

        if not ((vegsizex == sizex) and (vegsizey == sizey)):
            raise Exception(
                "Error in Vegetation Canopy DSM: All rasters must be of same extent and resolution"
            )

        if input_tdsm is not None:
            _, vegdsm2 = _resolve_raster_input(
                input_tdsm, name="Vegetation Trunk Zone DSM"
            )
            vegdsm2 = np.array(vegdsm2, dtype=float, copy=True)
        else:
            trunkratio = input_theight / 100.0
            vegdsm2 = vegdsm * trunkratio

        vegsizex = vegdsm2.shape[0]
        vegsizey = vegdsm2.shape[1]

        if not ((vegsizex == sizex) and (vegsizey == sizey)):
            raise Exception(
                "Error in Trunk Zone DSM: All rasters must be of same extent and resolution"
            )
    else:
        rows = dsm.shape[0]
        cols = dsm.shape[1]
        vegdsm = np.zeros([rows, cols])
        if use_gpu:
            vegdsm = torch.from_numpy(vegdsm).to(device)
        vegdsm2 = 0.0
        usevegdem = 0

    if aniso == 1:

        if use_gpu:
            print("gpu version")

            ret = svf_torch.svfForProcessing153(
                dsm,
                vegdsm,
                vegdsm2,
                scale,
                usevegdem,
                pixel_resolution,
                wallScheme,
                dem,
                feedback,
                device,
            )
        else:
            print("cpu version")
            ret = svf.svfForProcessing153(
                dsm,
                vegdsm,
                vegdsm2,
                scale,
                usevegdem,
                pixel_resolution,
                wallScheme,
                dem,
                feedback,
            )

    else:

        if use_gpu:
            print("gpu version")

            ret = svf_torch.svfForProcessing655(
                dsm,
                vegdsm,
                vegdsm2,
                scale,
                usevegdem,
                feedback,
                device,
            )

            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "xpu":
                torch.xpu.empty_cache()
        else:
            print("cpu version")
            ret = svf.svfForProcessing655(
                dsm,
                vegdsm,
                vegdsm2,
                scale,
                usevegdem,
                feedback,
            )



    # print('Time to finish first SVF calculation = ' + str(run_time))
    if wallScheme == 1:
        voxelTable = ret["voxelTable"]
        voxelTable = voxelTable[
            voxelTable[:, 2] != 0, :
        ]  # Remove where wall height is zero, i.e. there is no wall...
        wallHeights = ret["walls"]
        svfbu = ret["svf"]
        if usevegdem == 0:
            svftotal = svfbu
            svfveg = ret["svfveg"]
            svfaveg = ret["svfaveg"]
        else:
            svfveg = ret["svfveg"]
            svfaveg = ret["svfaveg"]
            trans = transVeg / 100.0
            svftotal = svfbu - (1 - svfveg) * (1 - trans)
        # Lägg till loop för att lägga till i tabellen
        svf_array = np.zeros((voxelTable.shape[0]))
        svf_height_array = np.zeros((voxelTable.shape[0]))
        svfbu_array = np.zeros((voxelTable.shape[0]))
        svfveg_array = np.zeros((voxelTable.shape[0]))
        svfaveg_array = np.zeros((voxelTable.shape[0]))

        if use_gpu:
            svf_array = torch.from_numpy(svf_array).to(device)
            svf_height_array = torch.from_numpy(svf_height_array).to(device)
            svfbu_array = torch.from_numpy(svfbu_array).to(device)
            svfveg_array = torch.from_numpy(svfveg_array).to(device)
            svfaveg_array = torch.from_numpy(svfaveg_array).to(device)

        voxel_y = None
        if use_gpu:
            voxel_y = torch.where(voxelTable[:, 2] != 0)
        else:
            voxel_y = np.where(voxelTable[:, 2] != 0)

        for temp_y in voxel_y[0]:
            svf_array[temp_y] = svftotal[
                int(voxelTable[temp_y, 5]), int(voxelTable[temp_y, 6])
            ]
            svfbu_array[temp_y] = svfbu[
                int(voxelTable[temp_y, 5]), int(voxelTable[temp_y, 6])
            ]
            svfveg_array[temp_y] = svfveg[
                int(voxelTable[temp_y, 5]), int(voxelTable[temp_y, 6])
            ]
            svfaveg_array[temp_y] = svfaveg[
                int(voxelTable[temp_y, 5]), int(voxelTable[temp_y, 6])
            ]
            svf_height_array[temp_y] = svf_height

        # Clean up large SVF arrays after extraction
        if use_gpu:
            del svftotal, svfbu, svfveg, svfaveg, voxel_y
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "xpu":
                torch.xpu.empty_cache()

        if kmeans:

            if use_gpu:
                print("gpu version")

                voxelTable, cluster_heights = svfv_torch.svf_kmeans(
                    dsm,
                    dem,
                    vegdsm,
                    vegdsm2,
                    wallHeights,
                    transVeg,
                    scale,
                    usevegdem,
                    pixel_resolution,
                    voxelTable,
                    clusters,
                    svf_height,
                    svf_array,
                    svfbu_array,
                    svfveg_array,
                    svfaveg_array,
                    svf_height_array,
                    feedback,
                    device=device,
                )

                # Interpolate for voxels where SVF has not been calculated
                voxelTable = svfv_torch.interpolate_svf(voxelTable)
            else:

                voxelTable, cluster_heights = svfv.svf_kmeans(
                    dsm,
                    dem,
                    vegdsm,
                    vegdsm2,
                    wallHeights,
                    transVeg,
                    scale,
                    usevegdem,
                    pixel_resolution,
                    voxelTable,
                    clusters,
                    svf_height,
                    svf_array,
                    svfbu_array,
                    svfveg_array,
                    svfaveg_array,
                    svf_height_array,
                    feedback,
                )

                # Interpolate for voxels where SVF has not been calculated
                voxelTable = svfv.interpolate_svf(voxelTable)

        # Clean up k-means result tensors
        if use_gpu:
            del (
                svf_array,
                svfbu_array,
                svfveg_array,
                svfaveg_array,
                svf_height_array,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "xpu":
                torch.xpu.empty_cache()

        # Loop for exact SVF at heights (increase DEM)
        # if demlayer:
        else:

            if use_gpu:
                print("gpu version")
                voxelTable = svfv_torch.svf_for_voxels(
                    dsm,
                    dem,
                    vegdsm,
                    vegdsm2,
                    transVeg,
                    scale,
                    usevegdem,
                    pixel_resolution,
                    voxelTable,
                    svf_height,
                    svf_array,
                    svfbu_array,
                    svfveg_array,
                    svfaveg_array,
                    svf_height_array,
                    feedback,
                    device=device,
                )
            else:
                print("cpu version")
                voxelTable = svfv.svf_for_voxels(
                    dsm,
                    dem,
                    vegdsm,
                    vegdsm2,
                    transVeg,
                    scale,
                    usevegdem,
                    pixel_resolution,
                    voxelTable,
                    svf_height,
                    svf_array,
                    svfbu_array,
                    svfveg_array,
                    svfaveg_array,
                    svf_height_array,
                    feedback,
                )

            # Clean up voxel processing result tensors
            if use_gpu:
                del (
                    svf_array,
                    svfbu_array,
                    svfveg_array,
                    svfaveg_array,
                    svf_height_array,dsm, vegdsm, vegdsm2
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                elif device.type == "xpu":
                    torch.xpu.empty_cache()
            # Remove rows where svfbu, sfveg and svfaveg is zero
            if usevegdem == 1:
                voxelTable = voxelTable[
                    (
                        (voxelTable[:, -3] > 0.0)
                        & (voxelTable[:, -2] > 0.0)
                        & (voxelTable[:, -1] > 0.0)
                    ),
                    :,
                ]
            else:
                voxelTable = voxelTable[((voxelTable[:, -3] > 0.0)), :]

        # Store voxelTable, necessary?
        ret["voxelTable"] = voxelTable

    filename = outputFile

    if outputDir is not None:
        outputDir = str(outputDir)
        if not os.path.exists(outputDir):
            os.makedirs(outputDir)

    if ret is not None:
        svfbu = ret["svf"]
        svfbuE = ret["svfE"]
        svfbuS = ret["svfS"]
        svfbuW = ret["svfW"]
        svfbuN = ret["svfN"]

        if use_gpu:

            misc.saveraster(
                gdal_dsm,
                outputDir + "/" + "svf.tif",
                svfbu.cpu().detach().numpy(),
            )
            misc.saveraster(
                gdal_dsm,
                outputDir + "/" + "svfE.tif",
                svfbuE.cpu().detach().numpy(),
            )
            misc.saveraster(
                gdal_dsm,
                outputDir + "/" + "svfS.tif",
                svfbuS.cpu().detach().numpy(),
            )
            misc.saveraster(
                gdal_dsm,
                outputDir + "/" + "svfW.tif",
                svfbuW.cpu().detach().numpy(),
            )
            misc.saveraster(
                gdal_dsm,
                outputDir + "/" + "svfN.tif",
                svfbuN.cpu().detach().numpy(),
            )

        else:
            misc.saveraster(
                gdal_dsm,
                outputDir + "/" + "svf.tif",
                svfbu,
            )
            misc.saveraster(
                gdal_dsm,
                outputDir + "/" + "svfE.tif",
                svfbuE,
            )
            misc.saveraster(
                gdal_dsm,
                outputDir + "/" + "svfS.tif",
                svfbuS,
            )
            misc.saveraster(
                gdal_dsm,
                outputDir + "/" + "svfW.tif",
                svfbuW,
            )
            misc.saveraster(
                gdal_dsm,
                outputDir + "/" + "svfN.tif",
                svfbuN,
            )

        # Clean up main SVF tensors after saving
        if use_gpu:
            del svfbuE, svfbuS, svfbuW, svfbuN
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "xpu":
                torch.xpu.empty_cache()
        if os.path.isfile(outputDir + "/" + "svfs.zip"):
            os.remove(outputDir + "/" + "svfs.zip")

        zippo = zipfile.ZipFile(outputDir + "/" + "svfs.zip", "a")
        zippo.write(outputDir + "/" + "svf.tif", "svf.tif")
        zippo.write(outputDir + "/" + "svfE.tif", "svfE.tif")
        zippo.write(outputDir + "/" + "svfS.tif", "svfS.tif")
        zippo.write(outputDir + "/" + "svfW.tif", "svfW.tif")
        zippo.write(outputDir + "/" + "svfN.tif", "svfN.tif")
        zippo.close()

        os.remove(outputDir + "/" + "svf.tif")
        os.remove(outputDir + "/" + "svfE.tif")
        os.remove(outputDir + "/" + "svfS.tif")
        os.remove(outputDir + "/" + "svfW.tif")
        os.remove(outputDir + "/" + "svfN.tif")

        if usevegdem == 0:
            svftotal = svfbu
        else:
            # report the result
            svfveg = ret["svfveg"]
            svfEveg = ret["svfEveg"]
            svfSveg = ret["svfSveg"]
            svfWveg = ret["svfWveg"]
            svfNveg = ret["svfNveg"]
            svfaveg = ret["svfaveg"]
            svfEaveg = ret["svfEaveg"]
            svfSaveg = ret["svfSaveg"]
            svfWaveg = ret["svfWaveg"]
            svfNaveg = ret["svfNaveg"]

            if use_gpu:

                misc.saveraster(
                    gdal_dsm,
                    outputDir + "/" + "svfveg.tif",
                    svfveg.cpu().detach().numpy(),
                )
                misc.saveraster(
                    gdal_dsm,
                    outputDir + "/" + "svfEveg.tif",
                    svfEveg.cpu().detach().numpy(),
                )
                misc.saveraster(
                    gdal_dsm,
                    outputDir + "/" + "svfSveg.tif",
                    svfSveg.cpu().detach().numpy(),
                )
                misc.saveraster(
                    gdal_dsm,
                    outputDir + "/" + "svfWveg.tif",
                    svfWveg.cpu().detach().numpy(),
                )
                misc.saveraster(
                    gdal_dsm,
                    outputDir + "/" + "svfNveg.tif",
                    svfNveg.cpu().detach().numpy(),
                )
                misc.saveraster(
                    gdal_dsm,
                    outputDir + "/" + "svfaveg.tif",
                    svfaveg.cpu().detach().numpy(),
                )
                misc.saveraster(
                    gdal_dsm,
                    outputDir + "/" + "svfEaveg.tif",
                    svfEaveg.cpu().detach().numpy(),
                )
                misc.saveraster(
                    gdal_dsm,
                    outputDir + "/" + "svfSaveg.tif",
                    svfSaveg.cpu().detach().numpy(),
                )
                misc.saveraster(
                    gdal_dsm,
                    outputDir + "/" + "svfWaveg.tif",
                    svfWaveg.cpu().detach().numpy(),
                )
                misc.saveraster(
                    gdal_dsm,
                    outputDir + "/" + "svfNaveg.tif",
                    svfNaveg.cpu().detach().numpy(),
                )
            else:
                misc.saveraster(
                    gdal_dsm,
                    outputDir + "/" + "svfveg.tif",
                    svfveg,
                )
                misc.saveraster(
                    gdal_dsm,
                    outputDir + "/" + "svfEveg.tif",
                    svfEveg,
                )
                misc.saveraster(
                    gdal_dsm,
                    outputDir + "/" + "svfSveg.tif",
                    svfSveg,
                )
                misc.saveraster(
                    gdal_dsm,
                    outputDir + "/" + "svfWveg.tif",
                    svfWveg,
                )
                misc.saveraster(
                    gdal_dsm,
                    outputDir + "/" + "svfNveg.tif",
                    svfNveg,
                )
                misc.saveraster(
                    gdal_dsm,
                    outputDir + "/" + "svfaveg.tif",
                    svfaveg,
                )
                misc.saveraster(
                    gdal_dsm,
                    outputDir + "/" + "svfEaveg.tif",
                    svfEaveg,
                )
                misc.saveraster(
                    gdal_dsm,
                    outputDir + "/" + "svfSaveg.tif",
                    svfSaveg,
                )
                misc.saveraster(
                    gdal_dsm,
                    outputDir + "/" + "svfWaveg.tif",
                    svfWaveg,
                )
                misc.saveraster(
                    gdal_dsm,
                    outputDir + "/" + "svfNaveg.tif",
                    svfNaveg,
                )

            # Clean up vegetation SVF tensors after saving
            if use_gpu:
                del (
                    svfEveg,
                    svfSveg,
                    svfWveg,
                    svfNveg,
                    svfaveg,
                    svfEaveg,
                    svfSaveg,
                    svfWaveg,
                    svfNaveg,
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                elif device.type == "xpu":
                    torch.xpu.empty_cache()
            zippo = zipfile.ZipFile(outputDir + "/" + "svfs.zip", "a")
            zippo.write(outputDir + "/" + "svfveg.tif", "svfveg.tif")
            zippo.write(outputDir + "/" + "svfEveg.tif", "svfEveg.tif")
            zippo.write(outputDir + "/" + "svfSveg.tif", "svfSveg.tif")
            zippo.write(outputDir + "/" + "svfWveg.tif", "svfWveg.tif")
            zippo.write(outputDir + "/" + "svfNveg.tif", "svfNveg.tif")
            zippo.write(outputDir + "/" + "svfaveg.tif", "svfaveg.tif")
            zippo.write(outputDir + "/" + "svfEaveg.tif", "svfEaveg.tif")
            zippo.write(outputDir + "/" + "svfSaveg.tif", "svfSaveg.tif")
            zippo.write(outputDir + "/" + "svfWaveg.tif", "svfWaveg.tif")
            zippo.write(outputDir + "/" + "svfNaveg.tif", "svfNaveg.tif")
            zippo.close()

            os.remove(outputDir + "/" + "svfveg.tif")
            os.remove(outputDir + "/" + "svfEveg.tif")
            os.remove(outputDir + "/" + "svfSveg.tif")
            os.remove(outputDir + "/" + "svfWveg.tif")
            os.remove(outputDir + "/" + "svfNveg.tif")
            os.remove(outputDir + "/" + "svfaveg.tif")
            os.remove(outputDir + "/" + "svfEaveg.tif")
            os.remove(outputDir + "/" + "svfSaveg.tif")
            os.remove(outputDir + "/" + "svfWaveg.tif")
            os.remove(outputDir + "/" + "svfNaveg.tif")

            trans = transVeg / 100.0
            svftotal = svfbu - (1 - svfveg) * (1 - trans)

        if use_gpu:
            misc.saveraster(
                gdal_dsm, filename, svftotal.cpu().detach().numpy()
            )
            del svftotal, svfbu
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "xpu":
                torch.xpu.empty_cache()
        else:
            misc.saveraster(gdal_dsm, filename, svftotal)

        # Save shadow images for SOLWEIG 2019a
        if aniso == 1:
            shmat = ret["shmat"]
            vegshmat = ret["vegshmat"]
            vbshvegshmat = ret["vbshvegshmat"]

            if use_gpu:
                np.savez_compressed(
                    outputDir + "/" + "shadowmats.npz",
                    shadowmat=shmat.cpu().detach().numpy(),
                    vegshadowmat=vegshmat.cpu().detach().numpy(),
                    vbshmat=vbshvegshmat.cpu().detach().numpy(),
                )  # ,
                del shmat, vegshmat, vbshvegshmat
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                elif device.type == "xpu":
                    torch.xpu.empty_cache()
            else:
                np.savez_compressed(
                    outputDir + "/" + "shadowmats.npz",
                    shadowmat=shmat,
                    vegshadowmat=vegshmat,
                    vbshmat=vbshvegshmat,
                )  # ,

        if wallScheme == 1:
            voxelId = ret["voxelIds"]
            voxelTable = ret["voxelTable"]

            if use_gpu:
                np.savez_compressed(
                    outputDir + "/" + "wallScheme.npz",
                    voxelId=voxelId.cpu().detach().numpy(),
                    voxelTable=voxelTable.cpu().detach().numpy(),
                )
                del voxelId, voxelTable
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                elif device.type == "xpu":
                    torch.xpu.empty_cache()
            else:
                np.savez_compressed(
                    outputDir + "/" + "wallScheme.npz",
                    voxelId=voxelId,
                    voxelTable=voxelTable,
                )

    # Final GPU memory cleanup
    if use_gpu:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "xpu":
            torch.xpu.empty_cache()

    # Aggressive GPU memory cleanup
    if use_gpu and torch.cuda.is_available():
        torch.cuda.synchronize()  # Ensure all GPU operations are complete
        torch.cuda.empty_cache()  # Clear unused GPU memory
        torch.cuda.reset_peak_memory_stats()  # Reset peak memory tracking
        torch.cuda.empty_cache()  # Clear again to be sure
        gc.collect()  # Force Python garbage collection
    elif use_gpu and torch.xpu.is_available():
        torch.xpu.synchronize()  # Ensure all GPU operations are complete
        torch.xpu.empty_cache()  # Clear unused GPU memory
        torch.xpu.reset_peak_memory_stats()  # Reset peak memory tracking
        torch.xpu.empty_cache()  # Clear again to be sure
        gc.collect()  # Force Python garbage collection

    print("Sky View Factor: SVF grid(s) successfully generated")
    

    if use_gpu:
        return {
            "svf": ret.get("svf").cpu().detach().numpy() if isinstance(ret, dict) else ret,
            "svfE": ret.get("svfE").cpu().detach().numpy() if isinstance(ret, dict) else None,
            "svfS": ret.get("svfS").cpu().detach().numpy() if isinstance(ret, dict) else None,
            "svfW": ret.get("svfW").cpu().detach().numpy() if isinstance(ret, dict) else None,
            "svfN": ret.get("svfN").cpu().detach().numpy() if isinstance(ret, dict) else None,
            "output_dir": str(outputDir) if outputDir is not None else None,
            "output_file": str(outputFile) if outputFile is not None else None,
            "wallScheme": wallScheme,
            "use_gpu": use_gpu,
            "aniso": aniso,
        }
        
    else:
            return {
            "svf": ret.get("svf") if isinstance(ret, dict) else ret,
            "svfE": ret.get("svfE") if isinstance(ret, dict) else None,
            "svfS": ret.get("svfS") if isinstance(ret, dict) else None,
            "svfW": ret.get("svfW") if isinstance(ret, dict) else None,
            "svfN": ret.get("svfN") if isinstance(ret, dict) else None,
            "output_dir": str(outputDir) if outputDir is not None else None,
            "output_file": str(outputFile) if outputFile is not None else None,
            "wallScheme": wallScheme,
            "use_gpu": use_gpu,
            "aniso": aniso,
        }
