from .model_loader import SVORModelLoader
from .sampler import SVORSampler
from .video_io import SVORLoadVideo, SVORSaveVideo

NODE_CLASS_MAPPINGS = {
    "SVORModelLoader": SVORModelLoader,
    "SVORSampler": SVORSampler,
    "SVORLoadVideo": SVORLoadVideo,
    "SVORSaveVideo": SVORSaveVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SVORModelLoader": "SVOR Model Loader",
    "SVORSampler": "SVOR Sampler",
    "SVORLoadVideo": "SVOR Load Video",
    "SVORSaveVideo": "SVOR Save Video",
}
