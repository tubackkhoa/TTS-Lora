import streamlit as st
import numpy as np
from pathlib import Path
from finetune.vieneu_lora.load_model import load_model
import tempfile
import os

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Vietnamese TTS", page_icon="🎙️", layout="centered")

ROOT = Path(__file__).resolve().parents[0]
MODEL_PATH = ROOT / "finetune/output/my_voice/merged"


# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def get_tts_model():
    model = load_model(ROOT, str(MODEL_PATH))
    return model


# =========================
# UI
# =========================
st.title("🎙️ Vietnamese Text To Speech")
st.markdown("### Công cụ chuyển văn bản thành giọng nói")

st.info("Nhập nội dung bên dưới để tạo giọng nói AI đã fine-tune.")

text_input = st.text_area(
    "📝 Nhập văn bản",
    value="Xin chào, đây là giọng nói đã được fine-tune.",
    height=150,
    placeholder="Nhập nội dung bạn muốn chuyển thành giọng nói...",
)

voice = st.selectbox("Chọn giọng nói", options=["tupt"], index=0)

# =========================
# GENERATE AUDIO
# =========================
if st.button("🔊 Tạo giọng nói", use_container_width=True):

    if not text_input.strip():
        st.warning("Vui lòng nhập văn bản!")
    else:
        try:
            with st.spinner("Đang tải model và tạo giọng nói..."):

                # Load model (chỉ load một lần)
                tts = get_tts_model()

                # Generate audio
                audio = tts.infer(text_input, voice=voice)

                audio = np.asarray(audio).squeeze()

                # Save audio
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                ) as temp_file:
                    output_path = temp_file.name

                tts.save(audio, output_path)

            st.success("Tạo giọng nói thành công!")

            # Play audio
            st.markdown("### 🎧 Nghe thử")
            st.audio(output_path, format="audio/wav")

            # Download audio
            with open(output_path, "rb") as f:
                audio_bytes = f.read()

            st.download_button(
                label="⬇️ Tải file WAV",
                data=audio_bytes,
                file_name="generated_voice.wav",
                mime="audio/wav",
                use_container_width=True,
            )

            # Cleanup
            os.remove(output_path)

        except Exception as e:
            st.error(f"Lỗi khi tạo giọng nói: {str(e)}")

st.markdown("---")
st.caption("Vietnamese TTS | Fine-tuned Voice Model")
