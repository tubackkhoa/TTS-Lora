import csv
import soundfile as sf
import numpy as np

from pathlib import Path

AUDIO_DIR = Path("dataset/raw_audio")
METADATA = Path("dataset/metadata.csv")

valid = []
rejected = []

with open(METADATA, encoding="utf-8") as f:

    reader = csv.DictReader(f, delimiter="|")

    for row in reader:

        filename = row["file_name"]
        text = row["text"]

        path = AUDIO_DIR / filename

        try:
            audio, sr = sf.read(path)

            duration = len(audio) / sr

            # Chuyển stereo thành mono để đo
            if audio.ndim == 2:
                mono = audio.mean(axis=1)
            else:
                mono = audio

            # RMS
            rms = np.sqrt(np.mean(mono**2))

            # Peak
            peak = np.max(np.abs(mono))

            reasons = []

            if duration < 3 or duration > 15:
                reasons.append("duration")

            if rms < 0.005:
                reasons.append("too_quiet")

            if peak >= 0.999:
                reasons.append("clipping")

            if not text.strip():
                reasons.append("empty_text")

            if reasons:
                rejected.append({"file_name": filename, "reason": ",".join(reasons)})
            else:
                valid.append(row)

        except Exception as e:
            rejected.append({"file_name": filename, "reason": str(e)})

# Xuất metadata sạch
with open(METADATA, "w", encoding="utf-8", newline="") as f:

    writer = csv.DictWriter(f, fieldnames=["file_name", "text"], delimiter="|")

    writer.writeheader()
    writer.writerows(valid)

# Xuất danh sách lỗi
with open("dataset/rejected.csv", "w", encoding="utf-8", newline="") as f:

    writer = csv.DictWriter(f, fieldnames=["file_name", "reason"])

    writer.writeheader()
    writer.writerows(rejected)

print("Hợp lệ:", len(valid))
print("Bị loại:", len(rejected))
