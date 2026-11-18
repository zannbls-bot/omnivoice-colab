# omnivoice-colab

Run a zero-shot TTS model on a free Google Colab GPU and call it from your own PC over HTTP:

- **VoxCPM2** ([OpenBMB](https://github.com/OpenBMB/VoxCPM), 30 languages incl. Thai, 48 kHz) — default. Weights Apache-2.0: commercial use OK.
- **OmniVoice** ([k2-fsa](https://github.com/k2-fsa/OmniVoice), 600+ languages) — weights **CC-BY-NC** (non-commercial only); audio tokenizer under the Boson Higgs Audio 2 community license.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zannbls-bot/omnivoice-colab/blob/main/OmniVoice_API.ipynb)

## วิธีใช้

1. กดปุ่ม Open In Colab ด้านบน → Runtime → Change runtime type → **T4 GPU** → เลือก `ENGINE` ในช่อง 1 → Runtime → **Run all**
2. รอช่อง 3 พิมพ์ `URL = https://….trycloudflare.com` และ `KEY = …`
3. ที่เครื่องเรา (Python 3.10+ ไม่ต้องลงไลบรารีเพิ่ม):

```bat
python client.py --url URL --key KEY --language th --text "สวัสดีครับ" --out test.wav

:: โคลนเสียง (ไฟล์อ้างอิง 3-10 วิ, ไม่ใส่ --ref-text = ให้ Whisper ถอดเอง)
python client.py --url URL --key KEY --language th --ref voice.wav --ref-text "คำพูดในไฟล์" --text "ข้อความใหม่" --out clone.wav

:: ทีละหลายบรรทัด (1 บรรทัด = 1 ไฟล์ out\0001.wav …)
python client.py --url URL --key KEY --language th --ref voice.wav --text-file lines.txt --out-dir out

:: ออกแบบเสียงแทนการโคลน
python client.py --url URL --key KEY --instruct "female, low pitch" --text "Hello" --out design.wav
```

เรียกจากโค้ด Python: `from client import tts; wav = tts(URL, KEY, "ข้อความ", ref="voice.wav", language="th")`

## API

ทุก endpoint ต้องมี header `X-API-Key` · `GET /health`

**VoxCPM2** (`server_voxcpm.py`)
- `POST /tts` JSON `text` (required), `ref_audio_b64` (เสียงต้นแบบ: เอาแค่เนื้อเสียง), `prompt_audio_b64` + `prompt_text` (พูดต่อจากเสียงนี้ — บทต้องตรงกับเสียงเป๊ะ), `cfg` (2.0), `steps` (10), `seed`, `retry_badcase` → `audio/wav` 48 kHz 16-bit
- `POST /asr` JSON `audio_b64`, `language` → `{"text"}` (Whisper large-v3-turbo — ฟังย้อนตรวจเสียงที่เจน)
- `POST /embed` JSON `audios_b64: [...]` → `{"emb": [[...]]}` (ลายเสียง WavLM, dot = cosine — คัดต้นแบบคนเดียวกัน/วัดความเหมือน)
- `POST /judge` JSON `audios_b64: [...]` → `{"scores": [{"mos", "lang": {"th", "zh", "en"}}]}` (หูกลเลือกเทค: UTMOS ความเป็นธรรมชาติ 1–5 + Whisper ตรวจภาษา — เสียงไทยที่ติดสำเนียงจีน ค่า th ต่ำ)
- งานหนักของ Dub Studio V4 (ส่งไฟล์เสียงดิบใน body ไม่ห่อ JSON · ใช้ช่อง 1b ติดตั้งให้):
  `POST /separate` → FLAC เสียงพูด (BS-Roformer) · `POST /dnr` → FLAC เสียงเอฟเฟกต์ 44.1 kHz (Bandit-Plus DnR ทิ้งเพลง/เสียงพูด) ·
  `POST /whisper?language=zh&vad=1&beam=5&words=1` → `{"duration", "segments"}` (faster-whisper large-v3)

**OmniVoice** (`server.py`)
- `POST /tts` JSON `text` (required), `ref_audio_b64`, `ref_text`, `instruct`, `language`, `speed`, `duration`, `num_step` (default 32) → `audio/wav` 24 kHz 16-bit. Same reference audio is encoded once per session (cached).

## ข้อจำกัด

- URL/KEY เปลี่ยนทุกครั้งที่ Run all · ปิดแท็บหรือ Colab ตัด runtime (ฟรี: ว่างนาน ~90 นาที / สูงสุด ~12 ชม.) = ต้องรันใหม่
- Colab ฟรีห้ามใช้ทำ deepfake / บริการเว็บที่ไม่เกี่ยวกับการใช้งานแบบโต้ตอบ — ใช้กับเสียงที่ได้รับอนุญาตเท่านั้น
- ใครได้ URL + KEY ใช้ GPU เราได้ อย่าแชร์

License: this wrapper Apache-2.0. Model weights carry their own licenses (above).
