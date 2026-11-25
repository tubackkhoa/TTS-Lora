# VieNeu Project --- Tạo giọng nói tiếng Việt và Fine-tune bằng LoRA

Dự án này xây dựng một pipeline **ghi âm giọng tham chiếu → tạo dữ liệu
giọng nói từ văn bản → kiểm tra dữ liệu → chuẩn bị dataset → LoRA
fine-tuning → đóng gói giọng → chuyển văn bản thành giọng nói (TTS)**
dựa trên VieNeu-TTS v3 Turbo.

> **Lưu ý quan trọng:** `generate_dataset.py` hiện dùng `ref_audio` để
> tạo các câu nói tổng hợp từ giọng tham chiếu. Vì vậy, dữ liệu đưa vào
> LoRA mặc định là **audio do model sinh ra**, không phải bản ghi âm gốc
> của từng câu. Cách này giúp tạo dataset nhanh, nhưng có thể học lại cả
> lỗi phát âm, nhiễu và đặc trưng của model nền. Nếu có thể, hãy kết hợp
> hoặc ưu tiên bản ghi âm thật của người nói, sau đó nghe kiểm tra và
> sửa transcript trước khi train. Chỉ sử dụng giọng nói mà bạn có quyền
> thu âm và sử dụng.

## 1. Pipeline tổng quan

```text
record.py
   │
   └── reference/my_voice.wav
             │
             ▼
generate_dataset.py + texts/vietnamese_sentences.txt
             │
             ├── dataset/raw_audio/*.wav
             └── dataset/metadata.csv
                         │
                         ▼
                  quality_check.py
                         │
                         ▼
             finetune/prepare_dataset.py
                         │
                         └── dataset/train.parquet
                                      │
                                      ▼
                           finetune/train_lora.py
                                      │
                         ┌────────────┴────────────┐
                         ▼                         ▼
                 output/my_voice/adapter    output/my_voice/merged
                                                   │
                                                   ▼
                                        finetune/make_voice.py
                                                   │
                                                   ▼
                                      merged/voices_v3_turbo.json
                                                   │
                                                   ▼
                                  app.py hoặc finetune/chat_tts.py
```

## 2. Flow chạy

Chạy tuần tự từ thư mục gốc:

```bash
# 1. Tạo thư mục đầu vào
mkdir -p reference texts dataset/raw_audio vieneu-models

# 2. Thu âm giọng tham chiếu
python record.py

# 3. Tạo texts/vietnamese_sentences.txt trước khi chạy bước này
python generate_dataset.py

# 4. Kiểm tra chất lượng (lưu ý metadata.csv sẽ bị ghi đè)
python quality_check.py

# 5. Chuẩn bị dataset
uv run python finetune/prepare_dataset.py --dataset-dir dataset

# 6. Train LoRA và merge model
uv run python finetune/train_lora.py \
  --data dataset/train.parquet \
  --run my_voice \
  --merge

# 7. Đăng ký voice preset
uv run python finetune/make_voice.py \
  --audio reference/my_voice.wav \
  --name "tupt" \
  --description "Giọng tiếng Việt cá nhân" \
  --gender female \
  --out finetune/output/my_voice/merged \
  --default

# 8. Mở ứng dụng TTS
uv run streamlit run app.py
```

---

**Kết quả cuối cùng:** một model VieNeu-TTS v3 Turbo đã merge LoRA và có
voice preset riêng. Người dùng nhập văn bản vào `app.py`, chọn voice đã
đăng ký và nhận audio WAV được tổng hợp tự động.
