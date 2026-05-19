"""SVOR Sampler node for ComfyUI.

Performs the full video object removal inference:
  1. Preprocessing  -- resize, align frame count, apply mask with dilation
  2. Denoising      -- call SVORPipeline
  3. Postprocessing -- decode latents, convert back to ComfyUI IMAGE format
"""

import gc

import numpy as np
import scipy.ndimage
import torch
import torch.nn.functional as F

# Default negative prompt shipped with SVOR
_DEFAULT_NEGATIVE = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

# Mapping from user-friendly labels to (height, width) target areas
_RESOLUTION_PRESETS = {
    "720p (720x1280)": (720, 1280),
    "480p (480x832)": (480, 832),
    "Keep Original": None,
}


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------
def _align_video_length(n_frames: int, temporal_ratio: int = 4) -> int:
    """Return the largest valid video length <= *n_frames*.

    Valid lengths satisfy ``(length - 1) % temporal_ratio == 0``,
    i.e. 1, 5, 9, 13, ... , 77, 81, ...
    """
    if n_frames <= 1:
        return 1
    return ((n_frames - 1) // temporal_ratio) * temporal_ratio + 1


def _compute_resolution(
    orig_h: int, orig_w: int, target_h: int, target_w: int
) -> tuple[int, int]:
    """Compute aspect-ratio-preserving resolution within a target pixel area,
    rounded up to multiples of 16."""
    max_area = target_h * target_w
    aspect = orig_h / orig_w
    new_h = round(np.sqrt(max_area * aspect))
    new_h = ((new_h + 15) // 16) * 16
    new_w = round(np.sqrt(max_area / aspect))
    new_w = ((new_w + 15) // 16) * 16
    return int(new_h), int(new_w)


def _resize_frames(frames: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Spatially resize a ``[T, C, H, W]`` tensor (bilinear)."""
    return F.interpolate(frames, size=(h, w), mode="bilinear", align_corners=False)


def _resize_masks(masks: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Spatially resize a ``[T, 1, H, W]`` tensor (nearest)."""
    return F.interpolate(masks, size=(h, w), mode="nearest")


def _dilate_masks(masks: torch.Tensor, iterations: int) -> torch.Tensor:
    """Apply binary dilation to each frame in a ``[T, 1, H, W]`` tensor."""
    if iterations <= 0:
        return masks
    out = masks.clone()
    for i in range(masks.shape[0]):
        m = masks[i, 0].cpu().numpy().astype(np.uint8)
        m = scipy.ndimage.binary_dilation(m, iterations=iterations).astype(np.float32)
        out[i, 0] = torch.from_numpy(m)
    return out


def _pad_or_trim(tensor: torch.Tensor, target_len: int, dim: int = 0) -> torch.Tensor:
    """Pad by repeating last frame, or trim, along *dim*."""
    n = tensor.shape[dim]
    if n >= target_len:
        return tensor.narrow(dim, 0, target_len)
    last = tensor.narrow(dim, n - 1, 1)
    repeats = [1] * tensor.ndim
    repeats[dim] = target_len - n
    return torch.cat([tensor, last.expand(*repeats)], dim=dim)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
class SVORSampler:
    """Run SVOR video object removal inference."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("SVOR_PIPELINE",),
                "frames": ("IMAGE",),
                "masks": ("MASK",),
                "seed": ("INT", {
                    "default": 43,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF,
                }),
            },
            "optional": {
                "prompt": ("STRING", {
                    "default": "Remove the target and fill the content appropriately",
                    "multiline": True,
                    "tooltip": "Text prompt guiding the removal",
                }),
                "negative_prompt": ("STRING", {
                    "default": _DEFAULT_NEGATIVE,
                    "multiline": True,
                }),
                "num_inference_steps": ("INT", {
                    "default": 20, "min": 1, "max": 100, "step": 1,
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 6.0, "min": 1.0, "max": 20.0, "step": 0.5,
                }),
                "context_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1,
                }),
                "dilation": ("INT", {
                    "default": 6, "min": 0, "max": 50, "step": 1,
                    "tooltip": "Mask dilation iterations (expand mask boundary)",
                }),
                "max_frames": ("INT", {
                    "default": 81, "min": 1, "max": 200, "step": 4,
                    "tooltip": "Max frames to process (auto-aligned to 1+4n)",
                }),
                "resolution": (list(_RESOLUTION_PRESETS.keys()),),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "sample"
    CATEGORY = "SVOR"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always re-execute: generation nodes should not be cached.
        return float("NaN")

    def sample(
        self,
        pipeline,
        frames: torch.Tensor,
        masks: torch.Tensor,
        seed: int,
        prompt: str = "Remove the target and fill the content appropriately",
        negative_prompt: str = _DEFAULT_NEGATIVE,
        num_inference_steps: int = 20,
        guidance_scale: float = 6.0,
        context_scale: float = 1.0,
        dilation: int = 6,
        max_frames: int = 81,
        resolution: str = "720p (720x1280)",
    ) -> tuple[torch.Tensor]:

        # ---- 1. Determine target frame count ----
        temporal_ratio = pipeline.vae.config.temporal_compression_ratio
        n_input = frames.shape[0]
        video_length = _align_video_length(min(n_input, max_frames), temporal_ratio)

        # ---- 2. Determine target spatial resolution ----
        orig_h, orig_w = frames.shape[1], frames.shape[2]
        preset = _RESOLUTION_PRESETS.get(resolution)
        if preset is not None:
            target_h, target_w = _compute_resolution(orig_h, orig_w, *preset)
        else:
            target_h = ((orig_h + 15) // 16) * 16
            target_w = ((orig_w + 15) // 16) * 16

        # ---- 3. Preprocess video frames ----
        # ComfyUI IMAGE: [T, H, W, C] float32 [0,1]
        # -> [T, C, H, W] -> resize -> pad/trim -> scale to [0, 255]
        video = frames.permute(0, 3, 1, 2)                        # [T, C, H, W]
        video = _resize_frames(video, target_h, target_w)          # [T, C, h, w]
        video = _pad_or_trim(video, video_length, dim=0)           # [L, C, h, w]
        video = video.unsqueeze(0).permute(0, 2, 1, 3, 4)         # [1, C, L, h, w]
        video = video * 255.0                                      # [0, 255]

        # ---- 4. Preprocess masks ----
        # ComfyUI MASK: [T, H, W] float32 [0,1]  (1 = masked area)
        mask = masks.unsqueeze(1)                                  # [T, 1, H, W]
        mask = _resize_masks(mask, target_h, target_w)             # [T, 1, h, w]
        mask = _pad_or_trim(mask, video_length, dim=0)             # [L, 1, h, w]

        # Binarize + dilate
        mask = (mask > 0.5).float()
        mask = _dilate_masks(mask, dilation)

        mask = mask.permute(1, 0, 2, 3).unsqueeze(0)              # [1, 1, L, h, w]

        # ---- 5. Apply mask to video (gray fill in masked area) ----
        mask_expanded = mask.expand_as(video)
        video = video * (mask_expanded < 0.5) + 128.0 * (mask_expanded >= 0.5)
        video = video / 127.5 - 1.0                               # [-1, 1]

        # ---- 6. Run pipeline ----
        # Defragment GPU memory before the heavy denoising loop
        gc.collect()
        torch.cuda.empty_cache()

        device = pipeline.device if hasattr(pipeline, "device") else torch.device("cuda")
        generator = torch.Generator(device=device).manual_seed(seed)

        with torch.no_grad():
            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=target_h,
                width=target_w,
                video=video,
                mask_video=mask,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                context_scale=context_scale,
                generator=generator,
                comfyui_progressbar=True,
            )

        # ---- 7. Postprocess to ComfyUI IMAGE [T, H, W, C] float32 [0,1] ----
        # result.videos: torch Tensor [1, C, T, H, W] in [0, 1]
        output = result.videos[0]              # [C, T, H, W]
        output = output.permute(1, 2, 3, 0)   # [T, H, W, C]
        output = output.cpu().float().clamp(0, 1)

        return (output,)
