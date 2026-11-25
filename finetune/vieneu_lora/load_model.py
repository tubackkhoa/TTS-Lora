from pathlib import Path

from huggingface_hub import snapshot_download
from vieneu import Vieneu

CODEC_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX"


def patch_vieneu_codec(root: Path):
    from vieneu._v3_turbo_engine.onnx_runtime_lite import OnnxV3LiteEngine
    import shutil

    original_fetch = OnnxV3LiteEngine._fetch

    def fixed_fetch(repo, files, subfolder):

        # Chỉ can thiệp vào MOSS Audio Tokenizer
        if repo != CODEC_REPO:
            return original_fetch(repo, files, subfolder)

        # Nếu model local đã đầy đủ thì sử dụng lại
        required = [f for f in files if not f.endswith(".json")]

        codec_dir = root / "vieneu-models" / "moss_codec"

        if all((codec_dir / f).is_file() for f in required):
            print(f"Using local MOSS codec: {codec_dir}")
            return codec_dir

        print("Downloading MOSS Audio Tokenizer to local storage...")

        snapshot = snapshot_download(
            repo_id=CODEC_REPO, repo_type="model", allow_patterns=files
        )

        source = Path(snapshot)

        codec_dir.parent.mkdir(parents=True, exist_ok=True)

        if codec_dir.exists():
            shutil.rmtree(codec_dir)

        shutil.copytree(source, codec_dir, symlinks=False)

        # Kiểm tra external data
        for filename in required:
            path = codec_dir / filename

            if not path.is_file():
                raise FileNotFoundError(f"MOSS model missing: {path}")

        print(f"MOSS codec ready: {codec_dir}")

        return codec_dir

    OnnxV3LiteEngine._fetch = staticmethod(fixed_fetch)

    print("VieNeu MOSS codec patch enabled!")


import onnxruntime as ort
from pathlib import Path


def get_onnx_providers():
    available = ort.get_available_providers()

    if "CUDAExecutionProvider" in available:
        return [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

    if "CoreMLExecutionProvider" in available:
        return [
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ]

    return ["CPUExecutionProvider"]


def load_model(root: Path, backbone_repo: str):
    model_dir = root / "vieneu-models" / "onnx_update"

    patch_vieneu_codec(root)

    providers = get_onnx_providers()

    print("ONNX available providers:", ort.get_available_providers())
    print("Selected providers:", providers)

    return Vieneu(
        mode="v3turbo",
        backend="onnx",
        onnx_dir=str(model_dir),
        precision="fp32",
        backbone_repo=backbone_repo,
    )
