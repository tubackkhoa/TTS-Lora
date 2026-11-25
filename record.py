import sounddevice as sd
from scipy.io.wavfile import write

fs = 44100

print("🎙️ Bắt đầu ghi âm...")
print("Nhấn ENTER để kết thúc ghi âm.")

input("Nhấn ENTER để bắt đầu...")
recording = sd.rec(int(600 * fs), samplerate=fs, channels=1, dtype="int16")

input("Đang ghi âm... Nhấn ENTER để dừng.")
sd.stop()

write("reference/my_voice.wav", fs, recording)
print("Đã lưu file reference/my_voice.wav!")

# python3 -c 'import subprocess; subprocess.run(["afplay", "reference/my_voice.wav"])'
