import os
import shutil
from pathlib import Path

import onnxruntime as ort
from huggingface_hub import snapshot_download
from vieneu import Vieneu

CODEC_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX"

# Select GPU using environment variable:
# Linux: VIE_NVIDIA_GPU=0 python app.py
GPU_ID = int(os.getenv("VIE_NVIDIA_GPU", "0"))


def get_onnx_providers():
    available = ort.get_available_providers()

    print("ONNX Runtime version:", ort.__version__)
    print("Available providers:", available)

    if "CUDAExecutionProvider" in available:
        providers = [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": GPU_ID,
                },
            ),
            "CPUExecutionProvider",
        ]

        print(f"Using NVIDIA GPU {GPU_ID}")
        return providers

    if "CoreMLExecutionProvider" in available:
        print("Using Apple CoreML")
        return [
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ]

    print("WARNING: CUDA/CoreML unavailable. Using CPU.")
    return ["CPUExecutionProvider"]


ONNX_PROVIDERS = get_onnx_providers()


def patch_onnx_sessions():
    """
    Patch ONNX Runtime session creation so VieNeu sessions
    use the selected execution providers.

    This only works if VieNeu uses ort.InferenceSession
    from the onnxruntime module patched below.
    """
    original_session = ort.InferenceSession

    def cuda_session(path_or_bytes, *args, **kwargs):
        # Preserve explicit providers if the caller supplied them.
        if kwargs.get("providers") is None:
            kwargs["providers"] = ONNX_PROVIDERS

        if kwargs.get("provider_options") is None:
            kwargs.pop("provider_options", None)

        print("Creating ONNX session with:", kwargs.get("providers"))

        return original_session(path_or_bytes, *args, **kwargs)

    ort.InferenceSession = cuda_session


def patch_vieneu_codec(root: Path):
    from vieneu._v3_turbo_engine.onnx_runtime_lite import (
        OnnxV3LiteEngine,
    )

    original_fetch = OnnxV3LiteEngine._fetch

    def fixed_fetch(repo, files, subfolder):
        if repo != CODEC_REPO:
            return original_fetch(repo, files, subfolder)

        required = [filename for filename in files if not filename.endswith(".json")]

        codec_dir = root / "vieneu-models" / "moss_codec"

        if all((codec_dir / filename).is_file() for filename in required):
            print(f"Using local MOSS codec: {codec_dir}")
            return codec_dir

        print("Downloading MOSS Audio Tokenizer...")

        snapshot = snapshot_download(
            repo_id=CODEC_REPO,
            repo_type="model",
            allow_patterns=files,
        )

        source = Path(snapshot)

        codec_dir.parent.mkdir(parents=True, exist_ok=True)

        if codec_dir.exists():
            shutil.rmtree(codec_dir)

        shutil.copytree(
            source,
            codec_dir,
            symlinks=False,
        )

        for filename in required:
            path = codec_dir / filename

            if not path.is_file():
                raise FileNotFoundError(f"MOSS model missing: {path}")

        print(f"MOSS codec ready: {codec_dir}")
        return codec_dir

    OnnxV3LiteEngine._fetch = staticmethod(fixed_fetch)

    print("VieNeu MOSS codec patch enabled!")


def load_model(root: Path, backbone_repo: str):
    root = Path(root)

    model_dir = root / "vieneu-models" / "onnx_update"

    if not model_dir.exists():
        raise FileNotFoundError(f"VieNeu ONNX model directory not found: {model_dir}")

    # Patch before creating the VieNeu engine.
    patch_onnx_sessions()
    patch_vieneu_codec(root)

    print("--------------------------------")
    print("Loading VieNeu V3 Turbo")
    print("Backend: ONNX")
    print("GPU ID:", GPU_ID)
    print("ONNX providers:", ONNX_PROVIDERS)
    print("--------------------------------")

    model = Vieneu(
        mode="v3turbo",
        backend="onnx",
        onnx_dir=str(model_dir),
        precision="fp32",
        backbone_repo=backbone_repo,
    )

    print("VieNeu model loaded!")
    return model
