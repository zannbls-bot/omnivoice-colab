# -*- coding: utf-8 -*-
"""OmniVoice TTS over HTTP — run on Google Colab (GPU), call from your PC with client.py.

POST /tts   header X-API-Key: <OV_KEY>
  JSON {text, ref_audio_b64?, ref_text?, instruct?, language?, speed?, duration?, num_step?}
  -> audio/wav (24 kHz, 16-bit mono)
GET /health -> {"ok": true, "gpu": "..."}

Env: OV_KEY (required), OV_PORT (default 8000), OV_MODEL (default k2-fsa/OmniVoice).
"""
import base64
import hashlib
import io
import os
import tempfile
import threading

import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Response
from pydantic import BaseModel
from omnivoice import OmniVoice

KEY = os.environ["OV_KEY"]
model = OmniVoice.from_pretrained(os.environ.get("OV_MODEL", "k2-fsa/OmniVoice"),
                                  device_map="cuda:0", dtype=torch.float16,
                                  load_asr=True)  # Whisper: ref_text may be omitted
gpu_lock = threading.Lock()   # one generation at a time on the single GPU
voices = {}                   # sha1(ref audio + ref text) -> VoiceClonePrompt
# ponytail: unbounded voice cache; fine for a few dozen voices per Colab session.

app = FastAPI()


class TTS(BaseModel):
    text: str
    ref_audio_b64: str | None = None
    ref_text: str | None = None
    instruct: str | None = None
    language: str | None = None
    speed: float | None = None
    duration: float | None = None
    num_step: int = 32


def _voice(b64, ref_text):
    raw = base64.b64decode(b64)
    k = hashlib.sha1(raw + (ref_text or "").encode()).hexdigest()
    if k not in voices:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(raw)
        try:
            voices[k] = model.create_voice_clone_prompt(ref_audio=f.name, ref_text=ref_text)
        finally:
            os.remove(f.name)
    return voices[k]


@app.get("/health")
def health():
    return {"ok": True, "gpu": torch.cuda.get_device_name(0), "voices": len(voices)}


@app.post("/tts")
def tts(req: TTS, x_api_key: str = Header(None)):
    if x_api_key != KEY:
        raise HTTPException(401, "bad X-API-Key")
    if not req.text.strip():
        raise HTTPException(400, "empty text")
    kw = {"text": req.text.strip(), "num_step": req.num_step}
    for k in ("instruct", "language", "speed", "duration"):
        if getattr(req, k) is not None:
            kw[k] = getattr(req, k)
    with gpu_lock:
        if req.ref_audio_b64:
            kw["voice_clone_prompt"] = _voice(req.ref_audio_b64, req.ref_text)
        audio = model.generate(**kw)[0]
    buf = io.BytesIO()
    sf.write(buf, audio, model.sampling_rate, format="WAV", subtype="PCM_16")
    return Response(buf.getvalue(), media_type="audio/wav")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("OV_PORT", 8000)))
