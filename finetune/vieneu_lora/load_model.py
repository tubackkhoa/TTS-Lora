import os
import shutil
import logging
from pathlib import Path

import onnxruntime as ort
from huggingface_hub import snapshot_download
from vieneu import Vieneu

# =====================================================
# CONFIG
# =====================================================

CODEC_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX"

GPU_ID = int(os.getenv("VIE_NVIDIA_GPU", "0"))

# auto / cuda / coreml / cpu
DEVICE_BACKEND = os.getenv("VIE_ONNX_DEVICE", "auto").lower()

# ONNX precision
PRECISION = os.getenv("VIE_ONNX_PRECISION", "fp32")

logger = logging.getLogger("VieNeu")


# =====================================================
# ONNX PROVIDERS
# =====================================================


def get_onnx_providers():

    available = ort.get_available_providers()

    print("=" * 50)
    print("ONNX Runtime:", ort.__version__)
    print("Available providers:", available)
    print("Requested backend:", DEVICE_BACKEND)
    print("=" * 50)

    if DEVICE_BACKEND == "cpu":
        return ["CPUExecutionProvider"]

    if DEVICE_BACKEND in ("auto", "cuda"):

        if "CUDAExecutionProvider" in available:

            print(f"Using NVIDIA CUDA GPU {GPU_ID}")

            return [
                (
                    "CUDAExecutionProvider",
                    {"device_id": GPU_ID},
                ),
                "CPUExecutionProvider",
            ]

        if DEVICE_BACKEND == "cuda":
            raise RuntimeError(
                "CUDAExecutionProvider unavailable. " "Install onnxruntime-gpu."
            )

    if DEVICE_BACKEND in ("auto", "coreml"):

        if "CoreMLExecutionProvider" in available:

            print("Using Apple CoreML")

            return [
                "CoreMLExecutionProvider",
                "CPUExecutionProvider",
            ]

        if DEVICE_BACKEND == "coreml":
            raise RuntimeError("CoreMLExecutionProvider unavailable.")

    print("Using CPUExecutionProvider")

    return ["CPUExecutionProvider"]


ONNX_PROVIDERS = get_onnx_providers()


# =====================================================
# PATCH ONNX SESSION
# =====================================================


def patch_onnx_sessions():

    # Prevent multiple patches
    if getattr(ort, "_vieneu_stream_patched", False):
        return

    original_session = ort.InferenceSession

    def patched_session(path_or_bytes, *args, **kwargs):

        # Do not override explicitly supplied providers.
        # InferenceSession positional argument 3 is providers.
        providers_supplied = kwargs.get("providers") is not None or len(args) >= 2

        if not providers_supplied:
            kwargs["providers"] = ONNX_PROVIDERS

        print(
            "Creating ONNX session:",
            kwargs.get("providers", "explicit positional providers"),
        )

        return original_session(
            path_or_bytes,
            *args,
            **kwargs,
        )

    ort.InferenceSession = patched_session

    ort._vieneu_stream_patched = True

    print("ONNX Runtime provider patch enabled!")


# =====================================================
# PATCH MOSS CODEC
# =====================================================


def patch_vieneu_codec(root: Path):

    from vieneu._v3_turbo_engine.onnx_runtime_lite import (
        OnnxV3LiteEngine,
    )

    if getattr(OnnxV3LiteEngine, "_vieneu_codec_patched", False):
        return

    original_fetch = OnnxV3LiteEngine._fetch

    def fixed_fetch(repo, files, subfolder):

        if repo != CODEC_REPO:
            return original_fetch(repo, files, subfolder)

        codec_dir = root / "vieneu-models" / "moss_codec"

        required = [filename for filename in files if not filename.endswith(".json")]

        # Reuse local codec
        if all((codec_dir / filename).is_file() for filename in required):

            print("Using local MOSS codec:", codec_dir)

            return codec_dir

        print("Downloading MOSS Audio Tokenizer...")

        snapshot = snapshot_download(
            repo_id=CODEC_REPO,
            repo_type="model",
            allow_patterns=files,
        )

        source = Path(snapshot)

        codec_dir.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if codec_dir.exists():
            shutil.rmtree(codec_dir)

        shutil.copytree(
            source,
            codec_dir,
            symlinks=False,
        )

        # Validate required files
        missing = [
            filename for filename in required if not (codec_dir / filename).is_file()
        ]

        if missing:
            raise FileNotFoundError("MOSS codec files missing: " + ", ".join(missing))

        print("MOSS codec ready:", codec_dir)

        return codec_dir

    OnnxV3LiteEngine._fetch = staticmethod(fixed_fetch)

    OnnxV3LiteEngine._vieneu_codec_patched = True

    print("VieNeu MOSS codec patch enabled!")


# =====================================================
# STREAMING VALIDATION
# =====================================================


def validate_streaming_api(model):

    infer_stream = getattr(model, "infer_stream", None)

    if not callable(infer_stream):

        raise RuntimeError(
            "This VieNeu installation does not expose infer_stream(). "
            "Please update VieNeu-TTS to a version supporting v3 Turbo "
            "native streaming."
        )

    print("Native infer_stream() API detected!")

    return True


# =====================================================
# LOAD MODEL
# =====================================================


def load_model(root: Path, backbone_repo: str):

    root = Path(root)

    model_dir = root / "vieneu-models" / "onnx_update"

    if not model_dir.exists():

        raise FileNotFoundError(f"VieNeu ONNX model directory not found: {model_dir}")

    # Patches must run before constructing the model.
    patch_onnx_sessions()

    patch_vieneu_codec(root)

    print()
    print("=" * 55)
    print("Loading VieNeu V3 Turbo")
    print("Backend: ONNX")
    print("Streaming: Enabled (API validation)")
    print("GPU ID:", GPU_ID)
    print("Precision:", PRECISION)
    print("ONNX providers:", ONNX_PROVIDERS)
    print("Backbone:", backbone_repo)
    print("ONNX model:", model_dir)
    print("=" * 55)

    model = Vieneu(
        mode="v3turbo",
        backend="onnx",
        onnx_dir=str(model_dir),
        precision=PRECISION,
        backbone_repo=backbone_repo,
    )

    # Check streaming API
    validate_streaming_api(model)

    print()
    print("VieNeu model loaded successfully!")
    print("Streaming API ready!")
    print("=" * 55)

    return model
