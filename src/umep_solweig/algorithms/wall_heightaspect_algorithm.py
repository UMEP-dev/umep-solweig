# -*- coding: utf-8 -*-

"""Standalone wall-height and wall-aspect processing for UMEP.

This module is intentionally free of QGIS-specific plumbing. The main entry
point is :func:`processAlgorithm`, which accepts ordinary Python arguments with
sensible defaults so it can be used directly from a script or notebook.

Methodology:
Identifies wall pixels and their heights from a Digital Surface Model (DSM)
using a filter as presented by Lindberg et al. (2015a). Optionally, wall aspect
is estimated using a specific linear filter adapted from Goodwin et al. (2009)
and further developed by Lindberg et al. (2015b). Wall aspect is given in degrees
where a north-facing wall pixel has a value of zero.

References:
- Goodwin NR, Coops NC, Tooke TR, Christen A, Voogt JA (2009) Characterizing
  urban surface cover and structure with airborne lidar technology. Can J Remote Sens 35:297–309
- Lindberg F., Grimmond, C.S.B. and Martilli, A. (2015a) Sunlit fractions on urban
  facets - Impact of spatial resolution and approach. Urban Climate.
- Lindberg F., Jonsson, P., Honjo, T. and Wästberg, D. (2015b) Solar energy on
  building envelopes - 3D modelling in a 2D environment. Solar Energy 115 369–378.
"""

from __future__ import annotations

import gc
import os
import warnings
from typing import Any, Optional, Union

import numpy as np
from osgeo import gdal

from ..util.misc import saverasternd

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from ..functions import wallalgorithms as wa
from ..functions import wallalgorithms_torch as wa_torch

__author__ = "Fredrik Lindberg"
__copyright__ = "(C) 2020-2026 by Fredrik Lindberg"


def _notify(feedback: bool, message: str) -> None:
    if feedback:
        print(message)


def processAlgorithm(
    dsm_input: Union[str, os.PathLike, Any],
    wall_limit: float = 3.0,
    use_gpu: bool = False,
    output_height_path: Optional[Union[str, os.PathLike]] = None,
    calculate_aspect: bool = True,
    output_aspect_path: Optional[Union[str, os.PathLike]] = None,
    feedback = None,
) -> dict[str, Any]:
    """Run wall height and optional wall aspect extraction.

    Parameters
    ----------
    dsm_input:
        DSM raster path, GDAL dataset, or NumPy array.
    wall_limit:
        Minimum wall height threshold (usually in meters).
    use_gpu:
        When True, try the torch-based implementation.
    output_height_path:
        Optional GeoTIFF path for the wall height output.
    calculate_aspect:
        Whether to compute wall aspect.
    output_aspect_path:
        Optional GeoTIFF path for the wall aspect output.
    feedback:
        Optional parameter to print processing checkpoints in the terminal.
    """
    if dsm_input is None:
        raise ValueError("A DSM input is required")

    # 1. Device Orchestration
    device = None
    if use_gpu and torch is not None:
        if (
            type(torch).__name__ == "MetaMock"
            or getattr(torch, "__name__", "") == "LocalMockTorch"
        ):
            raise RuntimeError("PyTorch is required for GPU mode")
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            device = torch.device("xpu")
        else:
            device = torch.device("cpu")
    else:
        if use_gpu:
            _notify(
                feedback, "PyTorch not available. Falling back to CPU mode."
            )
        else:
            _notify(feedback, "Running CPU mode")

    # 2. Input Loading & Resolution Extraction
    source_dataset = None
    dsm_scale = 1.0  # Default pixel scale multiplier

    if isinstance(dsm_input, (str, os.PathLike)):
        source_dataset = gdal.Open(str(dsm_input))
        if source_dataset is None:
            raise ValueError(f"Unable to open DSM raster: {dsm_input}")
        dsm_array = source_dataset.ReadAsArray().astype(float)

        # Dynamic Resolution Extraction: pixel width is index 1 of GeoTransform
        transform = source_dataset.GetGeoTransform()
        if transform and transform[1] > 0:
            dsm_scale = 1.0 / transform[1]

    elif hasattr(dsm_input, "ReadAsArray") and hasattr(
        dsm_input, "GetGeoTransform"
    ):
        source_dataset = dsm_input
        dsm_array = dsm_input.ReadAsArray().astype(float)
        transform = dsm_input.GetGeoTransform()
        if transform and transform[1] > 0:
            dsm_scale = 1.0 / transform[1]
    else:
        # Fallback for raw NumPy arrays
        dsm_array = np.asarray(dsm_input, dtype=float)
        if output_height_path or output_aspect_path:
            warnings.warn(
                "A raw NumPy array was provided. Output paths will be ignored "
                "because geospatial metadata (source_dataset) is unavailable for saving.",
                UserWarning,
            )

    # 3. Calculate Wall Height
    total = 100.0 / (int(dsm_array.shape[0] * dsm_array.shape[1]))
    _notify(feedback, "Calculating wall height")
    if use_gpu and device is not None:
        walls = wa_torch.findwalls_sp(dsm_array, wall_limit, device, False)
        wall_output = walls.cpu().detach().numpy()
    else:
        walls = wa.findwalls_sp(dsm_array, wall_limit, False)
        wall_output = np.asarray(walls, dtype=float)

    # 4. Calculate Wall Aspect
    aspect_output = None
    if calculate_aspect:
        _notify(feedback, "Calculating wall aspect")
        total = 100.0 / 180.0
        if use_gpu and device is not None:
            # Safely pass parameters into the PyTorch runtime
            aspect = wa_torch.filter1Goodwin_as_aspect_v3(
                walls,
                torch.tensor(1, dtype=torch.float32, device=device),
                torch.tensor(dsm_array, dtype=torch.float32, device=device),
                feedback,
                torch.tensor(total, dtype=torch.float32, device=device),
                device,
            )
            aspect_output = aspect.cpu().detach().numpy()
        else:
            aspect_output = wa.filter1Goodwin_as_aspect_v3(
                walls,
                1,
                dsm_array,
                feedback,
                total,
            )

    # 5. File Persistence
    if output_height_path is not None and source_dataset is not None:
        saverasternd(source_dataset, str(output_height_path), wall_output)

    if (
        output_aspect_path is not None
        and aspect_output is not None
        and source_dataset is not None
    ):
        saverasternd(source_dataset, str(output_aspect_path), aspect_output)

    # 6. Memory Cleanup
    if use_gpu and torch is not None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.synchronize()
            torch.xpu.empty_cache()

    gc.collect()

    return {
        "height": wall_output,
        "aspect": aspect_output,
        "height_path": (
            str(output_height_path)
            if (output_height_path and source_dataset)
            else None
        ),
        "aspect_path": (
            str(output_aspect_path)
            if (output_aspect_path and source_dataset)
            else None
        ),
    }


__all__ = ["processAlgorithm"]
