# src/ingkit/signals/__init__.py
# Functions for signal processing.

from .filters import lowpass_filter, highpass_filter, bandpass_filter, bandstop_filter
from .analysis import ens_N, nperseg_from_ens, coherence_analysis

__all__ = ["lowpass_filter", "highpass_filter", "bandpass_filter", "bandstop_filter",
           "ens_N", "nperseg_from_ens", "coherence_analysis"]
__version__ = "0.1.2"
