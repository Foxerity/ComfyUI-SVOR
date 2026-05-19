"""SVOR Model Loader node for ComfyUI.

Loads the Wan2.1-VACE-1.3B base model, text encoder, VAE, scheduler,
and merges the two-stage SVOR LoRA weights into a ready-to-use pipeline.
"""

import gc
import inspect
import os

import torch

from comfyui_svor import SVOR_ROOT
from diffusers import FlowMatchEulerDiscreteScheduler
from transformers import AutoTokenizer

from videox_fun.models import AutoencoderKLWan, VaceWanModel, WanT5EncoderModel
from videox_fun.pipeline import SVORPipeline
from videox_fun.utils.fp8_optimization import (
    convert_model_weight_to_float8,
    convert_weight_dtype_wrapper,
    replace_parameters_by_name,
)
from videox_fun.utils.lora_utils import merge_lora

# ---------------------------------------------------------------------------
# Embedded config for Wan2.1-VACE-1.3B (mirrors config/wan2.1/wan_civitai.yaml)
# ---------------------------------------------------------------------------
_TRANSFORMER_KWARGS = {
    "transformer_subpath": "./",
    "dict_mapping": {"in_dim": "in_channels", "dim": "hidden_size"},
}

_VAE_KWARGS = {
    "vae_subpath": "Wan2.1_VAE.pth",
    "temporal_compression_ratio": 4,
    "spatial_compression_ratio": 8,
}

_TEXT_ENCODER_KWARGS = {
    "text_encoder_subpath": "models_t5_umt5-xxl-enc-bf16.pth",
    "tokenizer_subpath": "google/umt5-xxl",
    "text_length": 512,
    "vocab": 256384,
    "dim": 4096,
    "dim_attn": 4096,
    "dim_ffn": 10240,
    "num_heads": 64,
    "num_layers": 24,
    "num_buckets": 32,
    "shared_pos": False,
    "dropout": 0.0,
}

_SCHEDULER_KWARGS = {
    "num_train_timesteps": 1000,
    "shift": 5.0,
    "use_dynamic_shifting": False,
    "base_shift": 0.5,
    "max_shift": 1.15,
    "base_image_seq_len": 256,
    "max_image_seq_len": 4096,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _filter_kwargs(cls, kwargs: dict) -> dict:
    """Keep only the kwargs accepted by *cls.__init__*."""
    sig = inspect.signature(cls.__init__)
    valid = set(sig.parameters.keys()) - {"self", "cls"}
    return {k: v for k, v in kwargs.items() if k in valid}


def _resolve(path: str) -> str:
    """Return *path* as-is if absolute, otherwise resolve against SVOR_ROOT."""
    return path if os.path.isabs(path) else os.path.join(SVOR_ROOT, path)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
class SVORModelLoader:
    """Load the full SVOR pipeline (base model + LoRA) in one step."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "models_dir": ("STRING", {
                    "default": "models",
                    "tooltip": (
                        "Directory that contains Wan2.1-VACE-1.3B/ and "
                        "remove_model_stage{1,2}.safetensors"
                    ),
                }),
                "weight_dtype": (["bfloat16", "float16"],),
                "gpu_memory_mode": (
                    [
                        "model_cpu_offload_and_qfloat8",
                        "model_cpu_offload",
                        "sequential_cpu_offload",
                        "model_full_load",
                    ],
                    {"default": "model_cpu_offload_and_qfloat8"},
                ),
            },
            "optional": {
                "lora_weight_stage1": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Stage-1 LoRA merge strength",
                }),
                "lora_weight_stage2": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Stage-2 LoRA merge strength",
                }),
            },
        }

    RETURN_TYPES = ("SVOR_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load"
    CATEGORY = "SVOR"

    # ---- main entry ----
    def load(
        self,
        models_dir: str,
        weight_dtype: str,
        gpu_memory_mode: str,
        lora_weight_stage1: float = 1.0,
        lora_weight_stage2: float = 1.0,
    ):
        models_dir = _resolve(models_dir)

        dtype = torch.bfloat16 if weight_dtype == "bfloat16" else torch.float16
        device = torch.device("cuda")

        # --- paths ---
        base_model_dir = os.path.join(models_dir, "Wan2.1-VACE-1.3B")
        lora_stage1 = os.path.join(models_dir, "remove_model_stage1.safetensors")
        lora_stage2 = os.path.join(models_dir, "remove_model_stage2.safetensors")

        for p in [base_model_dir, lora_stage1, lora_stage2]:
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"Expected model file not found: {p}\n"
                    f"Please place model files according to the README."
                )

        # --- Transformer ---
        transformer_subpath = _TRANSFORMER_KWARGS.get("transformer_subpath", "transformer")
        transformer = VaceWanModel.from_pretrained(
            os.path.join(base_model_dir, transformer_subpath),
            transformer_additional_kwargs=dict(_TRANSFORMER_KWARGS),
            low_cpu_mem_usage=True,
            torch_dtype=dtype,
        )

        # --- VAE ---
        vae = AutoencoderKLWan.from_pretrained(
            os.path.join(base_model_dir, _VAE_KWARGS.get("vae_subpath", "vae")),
            additional_kwargs=dict(_VAE_KWARGS),
        ).to(dtype)

        # --- Tokenizer ---
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(
                base_model_dir,
                _TEXT_ENCODER_KWARGS.get("tokenizer_subpath", "tokenizer"),
            ),
        )

        # --- Text Encoder ---
        text_encoder = WanT5EncoderModel.from_pretrained(
            os.path.join(
                base_model_dir,
                _TEXT_ENCODER_KWARGS.get("text_encoder_subpath", "text_encoder"),
            ),
            additional_kwargs=dict(_TEXT_ENCODER_KWARGS),
        ).to(dtype).eval()

        # --- Scheduler ---
        scheduler = FlowMatchEulerDiscreteScheduler(
            **_filter_kwargs(FlowMatchEulerDiscreteScheduler, _SCHEDULER_KWARGS)
        )

        # --- Assemble Pipeline ---
        pipeline = SVORPipeline(
            transformer=transformer,
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            scheduler=scheduler,
        )

        # --- GPU memory mode ---
        if gpu_memory_mode == "sequential_cpu_offload":
            replace_parameters_by_name(transformer, ["modulation"], device=device)
            transformer.freqs = transformer.freqs.to(device=device)
            pipeline.enable_sequential_cpu_offload(device=device)
        elif gpu_memory_mode == "model_cpu_offload_and_qfloat8":
            convert_model_weight_to_float8(
                transformer, exclude_module_name=["modulation"]
            )
            convert_weight_dtype_wrapper(transformer, dtype)
            pipeline.enable_model_cpu_offload(device=device)
        elif gpu_memory_mode == "model_cpu_offload":
            pipeline.enable_model_cpu_offload(device=device)
        else:  # model_full_load
            pipeline.to(device=device)

        # --- Merge LoRAs ---
        lora_specs = [
            (lora_stage1, lora_weight_stage1),
            (lora_stage2, lora_weight_stage2),
        ]
        for path, weight in lora_specs:
            if weight > 0:
                print(f"[SVOR] Merging LoRA: {os.path.basename(path)}, weight={weight}")
                pipeline = merge_lora(pipeline, path, weight)

        # Aggressively reclaim memory after LoRA merge
        gc.collect()
        torch.cuda.empty_cache()

        print("[SVOR] Pipeline ready.")
        return (pipeline,)
