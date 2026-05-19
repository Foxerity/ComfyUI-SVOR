"""ComfyUI entrypoint for ComfyUI-SVOR.

The repository is intended to be cloned directly under ComfyUI/custom_nodes/.
ComfyUI loads this file, then this file delegates to the real node package.
"""

import os
import sys

SVOR_ROOT = os.path.dirname(os.path.realpath(__file__))

if SVOR_ROOT not in sys.path:
    sys.path.insert(0, SVOR_ROOT)

from comfyui_svor import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "SVOR_ROOT"]
