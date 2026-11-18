# omnivoice-colab

Run [OmniVoice](https://github.com/k2-fsa/OmniVoice) (zero-shot TTS, 600+ languages incl. Thai) on a free Google Colab GPU and call it from your own PC over HTTP.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zannbls-bot/omnivoice-colab/blob/main/OmniVoice_API.ipynb)

## วิธีใช้

1. กดปุ่ม Open In Colab ด้านบน → Runtime → Change runtime type → **T4 GPU** → Runtime → **Run all**
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

`POST /tts` header `X-API-Key` · JSON `text` (required), `ref_audio_b64`, `ref_text`, `instruct`, `language`, `speed`, `duration`, `num_step` (default 32) → `audio/wav` 24 kHz 16-bit. `GET /health`.
Same reference audio is encoded once per session (cached).

## ข้อจำกัด

- URL/KEY เปลี่ยนทุกครั้งที่ Run all · ปิดแท็บหรือ Colab ตัด runtime (ฟรี: ว่างนาน ~90 นาที / สูงสุด ~12 ชม.) = ต้องรันใหม่
- Colab ฟรีห้ามใช้ทำ deepfake / บริการเว็บที่ไม่เกี่ยวกับการใช้งานแบบโต้ตอบ — ใช้กับเสียงที่ได้รับอนุญาตเท่านั้น
- ใครได้ URL + KEY ใช้ GPU เราได้ อย่าแชร์

License: this wrapper Apache-2.0, same as OmniVoice.
