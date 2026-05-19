"""Video I/O nodes for ComfyUI.

SVORLoadVideo  -- read a video file into ComfyUI IMAGE + MASK tensors.
SVORSaveVideo  -- write a ComfyUI IMAGE tensor sequence to an mp4 file.

Both nodes support in-browser video preview via ComfyUI's ``ui`` mechanism.
"""

import os
import time

import cv2
import folder_paths
import imageio
import numpy as np
import torch

from comfyui_svor import SVOR_ROOT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _unique_path(directory: str, prefix: str, ext: str = ".mp4") -> str:
    """Generate a non-colliding file path using a timestamp suffix."""
    os.makedirs(directory, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = f"{prefix}_{stamp}{ext}"
    path = os.path.join(directory, name)
    # unlikely collision guard
    idx = 0
    while os.path.exists(path):
        idx += 1
        name = f"{prefix}_{stamp}_{idx}{ext}"
        path = os.path.join(directory, name)
    return path


def _save_preview_video(
    frames_np: np.ndarray, fps: float, prefix: str, temp: bool = True,
) -> dict:
    """Write frames to ComfyUI's output/temp dir and return a ui-result dict.

    Parameters
    ----------
    frames_np : np.ndarray
        ``[T, H, W, C]`` uint8 RGB frames.
    fps : float
        Playback frame rate.
    prefix : str
        Filename prefix.
    temp : bool
        If True, save to ComfyUI's *temp* directory; otherwise *output*.

    Returns
    -------
    dict
        ``{"filename": ..., "subfolder": ..., "type": "temp"|"output"}``
        ready for the ``"images"`` list in the ``ui`` return dict.
    """
    target_dir = (
        folder_paths.get_temp_directory() if temp
        else folder_paths.get_output_directory()
    )
    os.makedirs(target_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{stamp}.mp4"
    filepath = os.path.join(target_dir, filename)
    # guard against rare collision
    idx = 0
    while os.path.exists(filepath):
        idx += 1
        filename = f"{prefix}_{stamp}_{idx}.mp4"
        filepath = os.path.join(target_dir, filename)

    frame_list = [frames_np[i] for i in range(frames_np.shape[0])]
    imageio.mimsave(filepath, frame_list, fps=fps)

    return {
        "filename": filename,
        "subfolder": "",
        "type": "temp" if temp else "output",
    }


# ---------------------------------------------------------------------------
# Load Video
# ---------------------------------------------------------------------------
class SVORLoadVideo:
    """Load a video file and output frames (IMAGE) + grayscale masks (MASK).

    Use one instance to load the RGB video and another to load the mask video.
    The MASK output converts each frame to single-channel [0,1] using luminance,
    which is useful when loading binary mask videos directly.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {
                    "default": "",
                    "tooltip": "Absolute or relative path to a video file",
                }),
            },
            "optional": {
                "max_frames": ("INT", {
                    "default": 0, "min": 0, "max": 9999,
                    "tooltip": "Limit on frames to load (0 = load all)",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "FLOAT", "INT")
    RETURN_NAMES = ("frames", "masks", "fps", "frame_count")
    FUNCTION = "load"
    CATEGORY = "SVOR"
    OUTPUT_NODE = True  # required for ui preview

    def load(self, video_path: str, max_frames: int = 0):
        # Resolve relative paths against SVOR project root
        if not os.path.isabs(video_path):
            video_path = os.path.join(SVOR_ROOT, video_path)

        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 16.0
        rgb_frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if 0 < max_frames <= len(rgb_frames):
                break
        cap.release()

        if not rgb_frames:
            raise ValueError(f"No frames could be read from: {video_path}")

        # IMAGE: [T, H, W, C] float32 [0, 1]
        arr = np.stack(rgb_frames, axis=0).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(arr)

        # MASK: [T, H, W] float32 {0, 1}  (binarized luminance)
        # For mask videos (black bg, white = region to remove), this gives a
        # clean binary mask.  For RGB videos the output is less meaningful but
        # harmless — users simply don't connect it.
        gray = (
            0.2989 * image_tensor[..., 0]
            + 0.5870 * image_tensor[..., 1]
            + 0.1140 * image_tensor[..., 2]
        )
        mask_tensor = (gray > 0.5).float()  # [T, H, W]

        # -- Video preview in ComfyUI UI --
        arr_uint8 = (arr * 255).clip(0, 255).astype(np.uint8)
        preview = _save_preview_video(arr_uint8, fps, "svor_input_preview", temp=True)

        return {
            "ui": {"images": [preview], "animated": (True,)},
            "result": (image_tensor, mask_tensor, fps, image_tensor.shape[0]),
        }


# ---------------------------------------------------------------------------
# Save Video
# ---------------------------------------------------------------------------
class SVORSaveVideo:
    """Save a sequence of ComfyUI IMAGE frames as an mp4 video file."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "fps": ("FLOAT", {"default": 16.0, "min": 1.0, "max": 120.0}),
            },
            "optional": {
                "filename_prefix": ("STRING", {"default": "svor_output"}),
                "output_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Output directory (empty = SVOR/samples/SVOR/)",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "save"
    CATEGORY = "SVOR"
    OUTPUT_NODE = True

    def save(
        self,
        frames: torch.Tensor,
        fps: float,
        filename_prefix: str = "svor_output",
        output_dir: str = "",
    ):
        if not output_dir:
            output_dir = os.path.join(SVOR_ROOT, "samples", "SVOR")

        path = _unique_path(output_dir, filename_prefix)

        # frames: [T, H, W, C] float32 [0, 1]
        np_frames = (frames.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        frame_list = [np_frames[i] for i in range(np_frames.shape[0])]
        imageio.mimsave(path, frame_list, fps=fps)

        print(f"[SVOR] Video saved to: {path}")

        # -- Video preview in ComfyUI UI --
        # Copy the saved file to ComfyUI's output dir for the /view endpoint.
        preview = _save_preview_video(np_frames, fps, filename_prefix, temp=False)

        return {
            "ui": {"images": [preview], "animated": (True,)},
            "result": (path,),
        }
