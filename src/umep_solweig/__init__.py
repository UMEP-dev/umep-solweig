"""
SOLWEIG - Solar and Long-wave Environmental Irradiance Geometry model

A comprehensive Python package for simulating solar and thermal radiation in urban environments,
accounting for building geometry, vegetation, and urban surfaces.

Main modules:
    - SOLWEIGpython: Core SOLWEIG calculations and radiation modeling
    - util: Utility functions for morphometric parameters and data processing
    - COMFA: Comfort and microclimate analysis tools
"""

__version__ = "2026.1.0"
__author__ = "UMEP Development Team"
__email__ = "umep@geo.su.se"
__license__ = "GPL-3.0"

# Version info for compatibility checks
__version_info__ = tuple(map(int, __version__.split(".")[:3]))

# Core imports from main modules
try:
    from .SOLWEIGpython import (
        Solweig_run,
        Solweig_2026a_calc_forprocessing,
        PET_calculations,
        UTCI_calculations,
    )
except ImportError:
    pass



# Optional GPU/torch imports
try:
    from .SOLWEIGpython import (
        Solweig_2026a_calc_forprocessing_torch,
    )
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Package-level convenient submodules
from .SOLWEIGpython import Solweig_run_torch

from . import solweig_run
from . import solweig_run_gpu

# Public API
__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "HAS_TORCH",
    # Core functions
    "Solweig_run",
    "Solweig_2026a_calc_forprocessing",
    "PET_calculations",
    "UTCI_calculations",
    # Convenience modules
    "solweig_run",
    "solweig_run_gpu",
]

# Add torch variants if available
if HAS_TORCH:
    __all__.extend([
        "Solweig_run_torch",
        "Solweig_2026a_calc_forprocessing_torch",
    ])
