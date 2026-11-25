import streamlit as st
import numpy as np
import torch
import tempfile
import os
import time

from pathlib import Path
from finetune.vieneu_lora.load_model import load_model

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(page_title="Vietnamese TTS", page_icon="🎙️", layout="centered")

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "finetune/output/my_voice/merged"


# =====================================================
# DEVICE DETECTION: CUDA -> MPS -> CPU
# =====================================================


def get_device():

    # NVIDIA GPU
    if torch.cuda.is_available():
        return torch.device("cuda")

    # Apple Silicon GPU
    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    # CPU fallback
    return torch.device("cpu")


DEVICE = get_device()


def get_device_name():

    if DEVICE.type == "cuda":
        return torch.cuda.get_device_name(0)

    elif DEVICE.type == "mps":
        return "Apple Silicon GPU (MPS)"

    return "CPU"


DEVICE_NAME = get_device_name()


# =====================================================
# GPU OPTIMIZATION
# =====================================================

if DEVICE.type == "cuda":

    torch.backends.cudnn.benchmark = True

    torch.backends.cuda.matmul.allow_tf32 = True

    torch.backends.cudnn.allow_tf32 = True


# =====================================================
# DEVICE SYNCHRONIZATION
# =====================================================


def synchronize_device():

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    elif DEVICE.type == "mps":
        torch.mps.synchronize()


# =====================================================
# LOAD MODEL ONLY ONCE
# =====================================================


@st.cache_resource(show_spinner="Đang tải model TTS lần đầu...")
def get_tts_model():

    print("=" * 50)
    print("Loading Vietnamese TTS Model")
    print("Device:", DEVICE)
    print("Device name:", DEVICE_NAME)
    print("=" * 50)

    model = load_model(ROOT, str(MODEL_PATH))

    print("Model loaded successfully!")

    return model


# =====================================================
# GENERATE AUDIO
# =====================================================


def generate_audio(tts, text, voice):

    synchronize_device()

    start_time = time.perf_counter()

    # Inference
    with torch.inference_mode():

        audio = tts.infer(text, voice=voice)

    synchronize_device()

    inference_time = time.perf_counter() - start_time

    # Convert audio
    audio = np.asarray(audio).squeeze()

    # Save WAV
    save_start = time.perf_counter()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:

        output_path = temp_file.name

    try:

        tts.save(audio, output_path)

        with open(output_path, "rb") as f:

            audio_bytes = f.read()

    finally:

        if os.path.exists(output_path):
            os.remove(output_path)

    save_time = time.perf_counter() - save_start

    total_time = time.perf_counter() - start_time

    return audio_bytes, {
        "inference": inference_time,
        "save": save_time,
        "total": total_time,
    }


# =====================================================
# LOAD MODEL
# =====================================================

tts = get_tts_model()


# =====================================================
# USER INTERFACE
# =====================================================

st.title("🎙️ Vietnamese Text To Speech")

st.markdown("### Công cụ chuyển văn bản thành giọng nói AI")


# =====================================================
# DEVICE INFORMATION
# =====================================================

st.markdown("### ⚡ Thông tin thiết bị")

if DEVICE.type == "cuda":

    st.success(f"🚀 NVIDIA GPU: {DEVICE_NAME}")

    gpu_memory = torch.cuda.memory_allocated() / 1024**3

    st.caption(f"GPU Memory: {gpu_memory:.2f} GB")

elif DEVICE.type == "mps":

    st.success(f"🍎 Apple Silicon GPU: {DEVICE_NAME}")

else:

    st.warning("⚠️ Không phát hiện GPU tương thích. " "Đang sử dụng CPU.")


# =====================================================
# INPUT
# =====================================================

text_input = st.text_area(
    "📝 Nhập văn bản",
    value="Xin chào, đây là giọng nói đã được fine-tune.",
    height=150,
    placeholder="Nhập nội dung bạn muốn chuyển thành giọng nói...",
)


voice = st.selectbox("Chọn giọng nói", options=["tupt"], index=0)


# =====================================================
# GENERATE BUTTON
# =====================================================

if st.button("🔊 Tạo giọng nói", use_container_width=True):

    if not text_input.strip():

        st.warning("Vui lòng nhập văn bản!")

    else:

        try:

            with st.spinner("Đang tạo giọng nói..."):

                audio_bytes, timing = generate_audio(tts, text_input.strip(), voice)

            st.success("Tạo giọng nói thành công!")

            # Audio player
            st.markdown("### 🎧 Nghe thử")

            st.audio(audio_bytes, format="audio/wav")

            # Download
            st.download_button(
                label="⬇️ Tải file WAV",
                data=audio_bytes,
                file_name="generated_voice.wav",
                mime="audio/wav",
                use_container_width=True,
            )

            # Performance
            st.markdown("### 📊 Thời gian xử lý")

            col1, col2, col3 = st.columns(3)

            col1.metric("Inference", f"{timing['inference']:.2f}s")

            col2.metric("Save WAV", f"{timing['save']:.2f}s")

            col3.metric("Tổng thời gian", f"{timing['total']:.2f}s")

            # GPU memory
            if DEVICE.type == "cuda":

                st.caption(
                    "VRAM sử dụng: " f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
                )

            elif DEVICE.type == "mps":

                st.caption(
                    "MPS memory: "
                    f"{torch.mps.current_allocated_memory() / 1024**3:.2f} GB"
                )

        except Exception as e:

            st.error(f"Lỗi khi tạo giọng nói: {str(e)}")

            st.exception(e)


# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption("Vietnamese TTS | CUDA / MPS / CPU | Fine-tuned Voice Model")
