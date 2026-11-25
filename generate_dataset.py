import csv
import time
import re
import shutil
from pathlib import Path

import soundfile as sf
from huggingface_hub import snapshot_download
from vieneu import Vieneu

# Import ONNX engine để sửa cơ chế tải codec
from vieneu._v3_turbo_engine.onnx_runtime_lite import OnnxV3LiteEngine

# =========================
# LOCAL MODEL CONFIGURATION
# =========================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "vieneu-models" / "onnx_update"
CODEC_DIR = BASE_DIR / "vieneu-models" / "moss_codec"

CODEC_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX"


# =========================
# FIX HUGGINGFACE SYMLINK
# =========================

_original_fetch = OnnxV3LiteEngine._fetch


def fixed_fetch(repo, files, subfolder):

    # Backbone model đã được lưu local
    # Chỉ xử lý MOSS Audio Tokenizer
    if repo != CODEC_REPO:
        return _original_fetch(repo, files, subfolder)

    required_files = [
        "moss_audio_tokenizer_decode_full.onnx",
        "moss_audio_tokenizer_decode_shared.data",
        "moss_audio_tokenizer_encode.onnx",
        "moss_audio_tokenizer_encode.data",
    ]

    # Nếu codec đã copy đầy đủ thì sử dụng luôn
    if all((CODEC_DIR / f).is_file() for f in required_files):
        print(f"Using local MOSS codec: {CODEC_DIR}")
        return CODEC_DIR

    print("Downloading MOSS Audio Tokenizer...")

    snapshot = snapshot_download(repo_id=CODEC_REPO, allow_patterns=files)

    source = Path(snapshot)

    # Xóa bản cũ nếu chưa đầy đủ
    if CODEC_DIR.exists():
        shutil.rmtree(CODEC_DIR)

    CODEC_DIR.parent.mkdir(parents=True, exist_ok=True)

    # Copy dữ liệu thật, không giữ symlink Hugging Face
    shutil.copytree(source, CODEC_DIR, symlinks=False)

    # Kiểm tra external data
    for filename in required_files:
        path = CODEC_DIR / filename

        if not path.is_file():
            raise FileNotFoundError(f"MOSS codec missing: {path}")

    print(f"MOSS codec ready: {CODEC_DIR}")

    return CODEC_DIR


# Override static method
OnnxV3LiteEngine._fetch = staticmethod(fixed_fetch)


# =========================
# CONFIGURATION
# =========================

REFERENCE = BASE_DIR / "reference/my_voice.wav"
TEXT_FILE = BASE_DIR / "texts/vietnamese_sentences.txt"

OUTPUT_DIR = BASE_DIR / "dataset/raw_audio"
METADATA = BASE_DIR / "dataset/metadata.csv"

MAX_SAMPLES = None
SLEEP_SECONDS = 0.2


# =========================
# INITIALIZATION
# =========================

if not REFERENCE.exists():
    raise FileNotFoundError(f"Không tìm thấy {REFERENCE}")

if not TEXT_FILE.exists():
    raise FileNotFoundError(f"Không tìm thấy {TEXT_FILE}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
METADATA.parent.mkdir(parents=True, exist_ok=True)

with open(TEXT_FILE, "r", encoding="utf-8") as f:
    texts = [line.strip() for line in f if line.strip()]

if MAX_SAMPLES:
    texts = texts[:MAX_SAMPLES]

print(f"Tổng số câu: {len(texts)}")


# =========================
# LOAD BACKBONE MODEL
# =========================

print("Đang kiểm tra VieNeu backbone...")

if not MODEL_DIR.exists():
    snapshot = snapshot_download(
        repo_id="pnnbao-ump/VieNeu-TTS-v3-Turbo", allow_patterns=["onnx_update/*"]
    )

    source = Path(snapshot) / "onnx_update"

    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)

    shutil.copytree(source, MODEL_DIR, symlinks=False)

print(f"VieNeu model: {MODEL_DIR}")

# Kiểm tra external data backbone
if not (MODEL_DIR / "vieneu_backbone_shared.data").is_file():
    raise FileNotFoundError("Missing vieneu_backbone_shared.data")


# =========================
# INITIALIZE TTS
# =========================

print("Đang tải VieNeu-TTS trên Mac M1...")

tts = Vieneu(backend="onnx", onnx_dir=str(MODEL_DIR), threads=4)

print("VieNeu-TTS đã sẵn sàng!")


# =========================
# GENERATE DATASET
# =========================

records = []

for index, text in enumerate(texts, start=1):

    filename = f"{index:04d}.wav"
    output_path = OUTPUT_DIR / filename

    print(f"\n[{index}/{len(texts)}] {text}")

    try:
        audio = tts.infer(text=text, ref_audio=str(REFERENCE), denoise=True)

        tts.save(audio, str(output_path))

        waveform, sample_rate = sf.read(str(output_path))

        duration = len(waveform) / sample_rate

        if duration < 3 or duration > 15:
            print(f"SKIP: Thời lượng không phù hợp ({duration:.2f}s)")
            output_path.unlink(missing_ok=True)
            continue

        records.append({"file_name": filename, "text": text})

        print(f"OK: {filename} | " f"{duration:.2f}s | " f"{sample_rate}Hz")

    except Exception as e:
        print(f"ERROR: {filename}: {e}")

    time.sleep(SLEEP_SECONDS)


# =========================
# EXPORT METADATA
# =========================

with open(METADATA, "w", encoding="utf-8", newline="") as f:

    writer = csv.DictWriter(f, fieldnames=["file_name", "text"], delimiter="|")

    writer.writeheader()
    writer.writerows(records)


print("\n=========================")
print("HOÀN THÀNH")
print("=========================")
print(f"Tổng câu đầu vào: {len(texts)}")
print(f"Audio thành công: {len(records)}")
print(f"Metadata: {METADATA}")
print(f"Audio folder: {OUTPUT_DIR}")
