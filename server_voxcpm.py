# -*- coding: utf-8 -*-
"""VoxCPM2 TTS over HTTP — run on Google Colab (GPU), call from your PC (client.py / Dub Studio V4).

VoxCPM2 weights are Apache-2.0 (commercial use OK). OmniVoice weights are CC-BY-NC.

POST /tts   header X-API-Key: <OV_KEY>
  JSON {text,
        ref_audio_b64?            reference_wav: timbre only ("controllable cloning")
        prompt_audio_b64?, prompt_text?   continuation prompt ("ultimate cloning"), text must match audio exactly
        cfg=2.0, steps=10, seed?, retry_badcase=true}
  -> audio/wav 16-bit mono at the model rate (48 kHz)
POST /asr   JSON {audio_b64, language?} -> {"text": ...}   (Whisper large-v3-turbo, for back-checking takes)
POST /embed JSON {audios_b64: [...]} -> {"emb": [[...], ...]}   (WavLM speaker vectors, L2-normalised:
            dot product = cosine similarity; for picking same-speaker references and scoring clones)
GET /health -> {"ok": true, "gpu": ..., "sr": ...}

Env: OV_KEY (required), OV_PORT (8000), VOX_MODEL (openbmb/VoxCPM2), VOX_OPTIMIZE (1 = torch.compile),
     VOX_DTYPE (override the checkpoint's bfloat16, e.g. float16/float32 on GPUs without bf16).
"""
import base64
import io
import os
import tempfile
import threading

import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Response
from pydantic import BaseModel
import voxcpm.model.voxcpm2 as _v2
from voxcpm import VoxCPM

if os.environ.get("VOX_DTYPE"):
    _v2.pick_runtime_dtype = lambda device, dtype: os.environ["VOX_DTYPE"]

KEY = os.environ["OV_KEY"]
model = VoxCPM.from_pretrained(os.environ.get("VOX_MODEL", "openbmb/VoxCPM2"), load_denoiser=False,
                               optimize=os.environ.get("VOX_OPTIMIZE", "1") == "1")
SR = model.tts_model.sample_rate
gpu_lock = threading.Lock()   # one job at a time on the single GPU
_asr = []                     # lazily loaded Whisper pipeline
_sv = []                      # lazily loaded speaker-vector model

app = FastAPI()


class TTS(BaseModel):
    text: str
    ref_audio_b64: str | None = None
    prompt_audio_b64: str | None = None
    prompt_text: str | None = None
    cfg: float = 2.0
    steps: int = 10
    seed: int | None = None
    retry_badcase: bool = True


class ASR(BaseModel):
    audio_b64: str
    language: str | None = None


class EMB(BaseModel):
    audios_b64: list[str]


def _auth(k):
    if k != KEY:
        raise HTTPException(401, "bad X-API-Key")


def _tmp(b64, files):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(base64.b64decode(b64))
    files.append(f.name)
    return f.name


@app.get("/health")
def health():
    return {"ok": True, "gpu": torch.cuda.get_device_name(0), "sr": SR, "engine": "voxcpm2"}


@app.post("/tts")
def tts(req: TTS, x_api_key: str = Header(None)):
    _auth(x_api_key)
    if not req.text.strip():
        raise HTTPException(400, "empty text")
    if bool(req.prompt_audio_b64) != bool(req.prompt_text):
        raise HTTPException(400, "prompt_audio_b64 and prompt_text go together")
    files = []
    try:
        kw = dict(text=req.text.strip(), cfg_value=req.cfg, inference_timesteps=req.steps, seed=req.seed,
                  retry_badcase=req.retry_badcase)
        if req.ref_audio_b64:
            kw["reference_wav_path"] = _tmp(req.ref_audio_b64, files)
        if req.prompt_audio_b64:
            kw["prompt_wav_path"], kw["prompt_text"] = _tmp(req.prompt_audio_b64, files), req.prompt_text
        with gpu_lock:
            wav = model.generate(**kw)
    finally:
        for p in files:
            os.remove(p)
    buf = io.BytesIO()
    sf.write(buf, wav, SR, format="WAV", subtype="PCM_16")
    return Response(buf.getvalue(), media_type="audio/wav")


@app.post("/asr")
def asr(req: ASR, x_api_key: str = Header(None)):
    _auth(x_api_key)
    files = []
    try:
        p = _tmp(req.audio_b64, files)
        with gpu_lock:
            if not _asr:
                from transformers import pipeline
                _asr.append(pipeline("automatic-speech-recognition", "openai/whisper-large-v3-turbo",
                                     torch_dtype=torch.float16, device="cuda:0"))
            gk = {"language": req.language, "task": "transcribe"} if req.language else {"task": "transcribe"}
            out = _asr[0](p, generate_kwargs=gk)
    finally:
        for f in files:
            os.remove(f)
    return {"text": out["text"].strip()}


@app.post("/embed")
def embed(req: EMB, x_api_key: str = Header(None)):
    _auth(x_api_key)
    import librosa
    out = []
    with gpu_lock:
        if not _sv:
            from transformers import AutoFeatureExtractor, WavLMForXVector
            _sv.append(AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv"))
            _sv.append(WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to("cuda:0").eval())
        fe, m = _sv
        for b in req.audios_b64:
            y, _ = librosa.load(io.BytesIO(base64.b64decode(b)), sr=16000, mono=True)
            x = fe(y, sampling_rate=16000, return_tensors="pt").to("cuda:0")
            with torch.no_grad():
                e = m(**x).embeddings[0]
            out.append(torch.nn.functional.normalize(e, dim=-1).tolist())
    return {"emb": out}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("OV_PORT", 8000)))
