import os
import json
import time
import socket
import threading
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import streamlit as st
import torch

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

import streamlit.components.v1 as components

from finetune.vieneu_lora.load_model import load_model

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(
    page_title="Vietnamese TTS Streaming",
    page_icon="🎙️",
    layout="centered",
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "finetune/output/my_voice/merged"

SAMPLE_RATE = 48000
STREAM_HOST = "0.0.0.0"
STREAM_PORT = int(os.getenv("VIE_STREAM_PORT", "8765"))

# Nếu chạy qua proxy/domain, có thể đặt biến môi trường:
# VIE_STREAM_URL=https://your-domain.com
CUSTOM_STREAM_URL = os.getenv("VIE_STREAM_URL", "").strip()


# =====================================================
# DEVICE INFORMATION
# =====================================================


def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


DEVICE = get_device()


def get_device_name():

    if DEVICE.type == "cuda":
        return torch.cuda.get_device_name(0)

    if DEVICE.type == "mps":
        return "Apple Silicon GPU (MPS)"

    return "CPU"


DEVICE_NAME = get_device_name()


# =====================================================
# LOAD MODEL
# =====================================================


@st.cache_resource(show_spinner="Đang tải VieNeu-TTS...")
def get_tts_model():

    print("=" * 55)
    print("Loading VieNeu-TTS V3 Turbo")
    print("Device:", DEVICE)
    print("Device name:", DEVICE_NAME)
    print("=" * 55)

    model = load_model(
        ROOT,
        str(MODEL_PATH),
    )

    if not callable(getattr(model, "infer_stream", None)):
        raise RuntimeError("Model không hỗ trợ infer_stream().")

    print("Native streaming ready!")

    return model


tts = get_tts_model()


# =====================================================
# AUDIO CONVERSION
# =====================================================


def float_to_pcm16(audio):

    audio = np.asarray(
        audio,
        dtype=np.float32,
    ).reshape(-1)

    audio = np.nan_to_num(
        audio,
        nan=0.0,
        posinf=1.0,
        neginf=-1.0,
    )

    audio = np.clip(audio, -1.0, 1.0)

    return (audio * 32767).astype("<i2").tobytes()


# =====================================================
# FASTAPI STREAMING SERVER
# =====================================================

api = FastAPI(title="VieNeu TTS Streaming API")

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class StreamRequest(BaseModel):

    text: str
    voice: str = "tupt"


@api.get("/health")
def health():

    return {
        "status": "ok",
        "model": "VieNeu V3 Turbo",
        "sample_rate": SAMPLE_RATE,
        "streaming": True,
    }


@api.post("/stream")
def stream_audio(request: StreamRequest):

    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty",
        )

    def generate():

        start_time = time.perf_counter()

        first_chunk = True
        chunk_count = 0
        total_samples = 0

        try:

            with torch.inference_mode():

                for chunk in tts.infer_stream(
                    text,
                    voice=request.voice,
                ):

                    if chunk is None:
                        continue

                    chunk = np.asarray(
                        chunk,
                        dtype=np.float32,
                    ).reshape(-1)

                    if len(chunk) == 0:
                        continue

                    pcm = float_to_pcm16(chunk)

                    if first_chunk:

                        first_chunk = False

                        ttfa = time.perf_counter() - start_time

                        print(f"First audio: {ttfa * 1000:.0f} ms")

                    chunk_count += 1
                    total_samples += len(chunk)

                    yield pcm

            audio_duration = total_samples / SAMPLE_RATE

            generation_time = time.perf_counter() - start_time

            rtf = generation_time / audio_duration if audio_duration > 0 else 0

            print(
                f"Stream completed | "
                f"Chunks: {chunk_count} | "
                f"Audio: {audio_duration:.2f}s | "
                f"Generation: {generation_time:.2f}s | "
                f"RTF: {rtf:.3f}"
            )

        except Exception as e:

            print("Streaming error:", repr(e))
            raise

    return StreamingResponse(
        generate(),
        media_type="application/octet-stream",
        headers={
            "X-Sample-Rate": str(SAMPLE_RATE),
            "X-Audio-Format": "pcm_s16le",
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


# =====================================================
# START STREAMING SERVER
# =====================================================


# =====================================================
# START STREAMING SERVER - START ONLY ONCE
# =====================================================


@st.cache_resource(show_spinner=False)
def start_streaming_server():

    config = uvicorn.Config(
        api,
        host=STREAM_HOST,
        port=STREAM_PORT,
        log_level="error",
        access_log=False,
    )

    server = uvicorn.Server(config)

    thread = threading.Thread(
        target=server.run,
        daemon=True,
        name="vieneu-stream-server",
    )

    thread.start()

    deadline = time.time() + 20

    while time.time() < deadline:

        if not thread.is_alive():
            raise RuntimeError(
                "Không thể khởi động Streaming Server. "
                "Có thể port đang được sử dụng."
            )

        try:
            with socket.create_connection(
                ("127.0.0.1", STREAM_PORT),
                timeout=0.3,
            ):
                print("VieNeu Streaming Server ready!")
                return server, thread

        except OSError:
            time.sleep(0.2)

    raise RuntimeError("Streaming Server startup timeout")


stream_server, stream_thread = start_streaming_server()


# =====================================================
# STREAM URL
# =====================================================


def get_stream_url():

    # Ưu tiên URL do người dùng cấu hình.
    if CUSTOM_STREAM_URL:

        base_url = CUSTOM_STREAM_URL.rstrip("/")

        if base_url.endswith("/stream"):
            return base_url

        return base_url + "/stream"

    # Streamlit có thể chạy phía sau reverse proxy.
    # Dùng hostname của trang hiện tại làm hostname API.
    # Nếu hostname rỗng, fallback localhost.
    host = (
        os.getenv("VIE_STREAM_HOST")
        or os.getenv("STREAMLIT_SERVER_ADDRESS")
        or "127.0.0.1"
    )

    # Khi truy cập trong iframe của Streamlit,
    # window.location.hostname sẽ được xác định ở JavaScript.
    # Đây chỉ là URL dự phòng.
    return f"http://{host}:{STREAM_PORT}/stream"


# =====================================================
# STREAMING PLAYER HTML
# =====================================================


# =====================================================
# STREAMING PLAYER HTML - TOGGLE NÓI / DỪNG
# =====================================================


def create_streaming_player(text, voice):

    text_json = json.dumps(text, ensure_ascii=False)
    voice_json = json.dumps(voice, ensure_ascii=False)

    api_url_json = json.dumps(
        get_stream_url(),
        ensure_ascii=False,
    )

    html = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">

<style>
    body {
        margin: 0;
        padding: 4px 0;
        font-family: Arial, sans-serif;
    }

    button {
        width: 100%;
        padding: 13px 20px;
        border: none;
        border-radius: 9px;
        background: #16a34a;
        color: white;
        cursor: pointer;
        font-size: 16px;
        font-weight: bold;
    }

    button:disabled {
        opacity: 0.6;
        cursor: not-allowed;
    }
</style>
</head>

<body>

<button id="toggle">🔊 Nói</button>

<script>
const TEXT = __TEXT__;
const VOICE = __VOICE__;
const API_URL = __API_URL__;

const button = document.getElementById("toggle");

const SAMPLE_RATE = 48000;

let audioContext = null;
let masterGain = null;

let audioChunks = [];
let scheduledCount = 0;
let nextPlayTime = 0;
let activeSources = 0;

let isGenerating = false;
let isPaused = false;
let isPlaying = false;
let isFinished = false;

let reader = null;
let carry = null;


// =====================================================
// AUDIO CONTEXT
// =====================================================

async function initAudio() {

    if (!audioContext) {
        audioContext = new AudioContext({
            sampleRate: SAMPLE_RATE
        });

        masterGain = audioContext.createGain();
        masterGain.gain.value = 1;
        masterGain.connect(audioContext.destination);
    }

    if (audioContext.state === "suspended") {
        await audioContext.resume();
    }
}


// =====================================================
// PCM16 -> FLOAT32
// =====================================================

function pcmToFloat32(bytes) {

    const length = bytes.length - (bytes.length % 2);

    const view = new DataView(
        bytes.buffer,
        bytes.byteOffset,
        length
    );

    const samples = new Float32Array(length / 2);

    for (let i = 0; i < samples.length; i++) {
        samples[i] = view.getInt16(i * 2, true) / 32768;
    }

    return samples;
}


// =====================================================
// SCHEDULE AUDIO CHUNKS
// =====================================================

function schedulePendingChunks() {

    if (!audioContext || isPaused) return;

    while (scheduledCount < audioChunks.length) {

        const samples = audioChunks[scheduledCount];

        if (!samples || samples.length === 0) {
            scheduledCount++;
            continue;
        }

        const buffer = audioContext.createBuffer(
            1,
            samples.length,
            SAMPLE_RATE
        );

        buffer.copyToChannel(samples, 0);

        const source = audioContext.createBufferSource();

        source.buffer = buffer;
        source.connect(masterGain);

        // Nối các chunk liên tục, tránh khoảng trống giữa chúng.
        const startAt = Math.max(
            nextPlayTime,
            audioContext.currentTime + 0.03
        );

        nextPlayTime = startAt + buffer.duration;

        activeSources++;

        source.onended = () => {
            activeSources--;
            checkPlaybackFinished();
        };

        source.start(startAt);

        scheduledCount++;
    }
}


// =====================================================
// CHECK PLAYBACK FINISHED
// =====================================================

function checkPlaybackFinished() {

    if (
        isFinished &&
        !isPaused &&
        activeSources === 0 &&
        scheduledCount >= audioChunks.length
    ) {
        isPlaying = false;
        button.textContent = "🔊 Nói lại";
    }
}


// =====================================================
// PAUSE / RESUME
// =====================================================

async function pauseAudio() {

    if (isPaused) return;

    isPaused = true;

    if (audioContext) {
        await audioContext.suspend();
    }

    button.textContent = "▶ Nói";
}


async function resumeAudio() {

    await initAudio();

    isPaused = false;
    isPlaying = true;

    button.textContent = "⏸ Dừng";

    // Các chunk đã lên lịch sẽ tiếp tục từ đúng vị trí
    // khi AudioContext được resume.
    // Các chunk mới nhận trong lúc pause sẽ được lên lịch ở đây.
    schedulePendingChunks();

    checkPlaybackFinished();
}


// =====================================================
// START STREAM
// =====================================================

async function startStreaming() {

    if (!TEXT.trim()) {
        button.textContent = "Vui lòng nhập văn bản";
        return;
    }

    isGenerating = true;
    isPaused = false;
    isPlaying = true;
    isFinished = false;

    audioChunks = [];
    scheduledCount = 0;
    nextPlayTime = 0;
    activeSources = 0;
    carry = null;

    button.textContent = "⏸ Dừng";

    try {

        // Khởi tạo AudioContext ngay trong thao tác click.
        await initAudio();

        const response = await fetch(API_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                text: TEXT,
                voice: VOICE
            })
        });

        if (!response.ok) {
            throw new Error(
                "HTTP " + response.status + ": " +
                await response.text()
            );
        }

        if (!response.body) {
            throw new Error("Trình duyệt không hỗ trợ streaming response.");
        }

        reader = response.body.getReader();

        while (true) {

            const result = await reader.read();

            if (result.done) break;

            let bytes = new Uint8Array(result.value);

            // Ghép byte lẻ từ lần đọc trước nếu có.
            if (carry !== null) {

                const merged = new Uint8Array(
                    carry.length + bytes.length
                );

                merged.set(carry, 0);
                merged.set(bytes, carry.length);

                bytes = merged;
                carry = null;
            }

            // PCM16 cần số byte chẵn.
            if (bytes.length % 2 !== 0) {
                carry = bytes.slice(-1);
                bytes = bytes.slice(0, -1);
            }

            if (bytes.length === 0) continue;

            const samples = pcmToFloat32(bytes);

            if (samples.length === 0) continue;

            audioChunks.push(samples);

            // Nếu đang dừng thì chỉ lưu chunk,
            // khi nhấn Nói sẽ phát tiếp.
            if (!isPaused) {
                schedulePendingChunks();
            }
        }

        if (carry !== null) {
            throw new Error("PCM stream kết thúc với byte chưa hoàn chỉnh.");
        }

        isFinished = true;
        isGenerating = false;
        reader = null;

        checkPlaybackFinished();

    } catch (error) {

        console.error("Streaming error:", error);

        isGenerating = false;
        isPlaying = false;
        isFinished = false;
        reader = null;

        button.textContent = "🔊 Thử lại";

    }
}


// =====================================================
// TOGGLE BUTTON
// =====================================================

async function toggleSpeech() {

    // Đang pause -> tiếp tục phát.
    if (isPaused) {
        await resumeAudio();
        return;
    }

    // Đang phát -> pause.
    if (isPlaying) {
        await pauseAudio();
        return;
    }

    // Chưa phát hoặc đã phát xong -> tạo lại audio.
    await startStreaming();
}


// =====================================================
// CLICK
// =====================================================

button.addEventListener("click", toggleSpeech);

</script>

</body>
</html>
"""

    return (
        html.replace("__TEXT__", text_json)
        .replace("__VOICE__", voice_json)
        .replace("__API_URL__", api_url_json)
    )


# =====================================================
# USER INTERFACE
# =====================================================

st.title("🎙️ Vietnamese Text To Speech")

st.markdown("### Công cụ chuyển văn bản thành giọng nói AI")

st.caption("VieNeu V3 Turbo | Native Streaming | 48 kHz")


# =====================================================
# DEVICE INFORMATION
# =====================================================

st.markdown("### ⚡ Thông tin thiết bị")

if DEVICE.type == "cuda":

    st.success(f"🚀 NVIDIA GPU: {DEVICE_NAME}")

elif DEVICE.type == "mps":

    st.success(f"🍎 Apple Silicon: {DEVICE_NAME}")

else:

    st.warning("Đang sử dụng CPU")


# =====================================================
# MODEL INFORMATION
# =====================================================

st.markdown("### 🎛️ Model")

col1, col2 = st.columns(2)

col1.metric("Backend", "ONNX")
col2.metric("Sample Rate", "48 kHz")


# =====================================================
# INPUT
# =====================================================

text_input = st.text_area(
    "📝 Nhập văn bản",
    value="Xin chào, đây là giọng nói đã được fine-tune.",
    height=150,
    placeholder="Nhập nội dung bạn muốn chuyển thành giọng nói...",
)

voice = st.selectbox(
    "Chọn giọng nói",
    options=["tupt"],
    index=0,
)


# =====================================================
# NÚT NÓI / DỪNG
# =====================================================

if text_input.strip():

    components.html(
        create_streaming_player(
            text_input.strip(),
            voice,
        ),
        height=65,
        scrolling=False,
    )

else:

    st.warning("Vui lòng nhập văn bản trước khi nói.")

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption(
    "Vietnamese TTS | VieNeu V3 Turbo | "
    "ONNX | CUDA / CoreML / CPU | Native Streaming"
)
