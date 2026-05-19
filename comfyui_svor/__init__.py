"""
ComfyUI custom nodes for SVOR (Stable Video Object Removal).

Installation:
    Clone the ComfyUI-SVOR repository into ComfyUI/custom_nodes/.
"""

import os
import sys

# Resolve the ComfyUI-SVOR repository root.
SVOR_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

# Add SVOR project root to Python path so videox_fun can be imported.
if SVOR_ROOT not in sys.path:
    sys.path.insert(0, SVOR_ROOT)

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "SVOR_ROOT"]
