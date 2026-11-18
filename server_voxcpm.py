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
POST /judge JSON {audios_b64: [...], langs?} -> {"scores": [{"mos": 1-5, "lang": {"th": p, "zh": p, ...}}]}
            (UTMOS naturalness + Whisper language-ID: machine ears for picking the best take)
Heavy dub-pipeline jobs for Dub Studio (raw audio bytes in the request body, not JSON):
POST /separate -> FLAC vocals (BS-Roformer model_bs_roformer_ep_317_sdr_12.9755, via audio-separator)
POST /dnr      -> FLAC effects stem, 44.1 kHz stereo (Bandit-Plus DnR via MSST; music and speech dropped)
POST /whisper?language=zh&vad=1&beam=5&words=1 -> {"duration", "segments": [{start, end, text, words: [[s, e, w, p]]}]}
               (faster-whisper large-v3, same call Dub Studio makes locally)
GET /health -> {"ok": true, "gpu": ..., "sr": ...}

Env: OV_KEY (required), OV_PORT (8000), VOX_MODEL (openbmb/VoxCPM2), VOX_OPTIMIZE (1 = torch.compile),
     VOX_DTYPE (override the checkpoint's bfloat16; default float16 on GPUs older than Ampere, e.g. T4).
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

if not os.environ.get("VOX_DTYPE") and torch.cuda.get_device_capability(0)[0] < 8:
    # T4/V100 have no native bfloat16: the checkpoint's bf16 runs emulated (~4x slower) and torch.compile skips it
    os.environ["VOX_DTYPE"] = "float16"
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
    return {"ok": True, "gpu": torch.cuda.get_device_name(0), "sr": SR, "engine": "voxcpm2",
            "dtype": model.tts_model.config.dtype}


@app.post("/tts")
def tts(req: TTS, x_api_key: str = Header(None)):
    _auth(x_api_key)
    if not req.text.strip():
        raise HTTPException(400, "empty text")
    if bool(req.prompt_audio_b64) != bool(req.prompt_text):
        raise HTTPException(400, "prompt_audio_b64 and prompt_text go together")
    files = []
    try:
        kw = dict(text=req.text.strip(), cfg_value=req.cfg, inference_timesteps=req.steps,
                  retry_badcase=req.retry_badcase)
        if req.ref_audio_b64:
            kw["reference_wav_path"] = _tmp(req.ref_audio_b64, files)
        if req.prompt_audio_b64:
            kw["prompt_wav_path"], kw["prompt_text"] = _tmp(req.prompt_audio_b64, files), req.prompt_text
        with gpu_lock:
            if req.seed is not None:      # PyPI voxcpm 2.0.3 has no seed= argument: seed the global RNGs instead
                torch.manual_seed(req.seed)
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


class JUDGE(BaseModel):
    audios_b64: list[str]
    langs: list[str] = ["th", "zh", "en"]


_mos = []                     # lazily loaded UTMOS predictor


@app.post("/judge")
def judge(req: JUDGE, x_api_key: str = Header(None)):
    """Machine ears for picking takes: UTMOS naturalness (1-5) + Whisper language-ID probabilities
    (low "th" / high "zh" on a Thai line = foreign accent)."""
    _auth(x_api_key)
    import librosa
    out = []
    with gpu_lock:
        if not _mos:
            _mos.append(torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to("cuda:0").eval())
        if not _asr:
            from transformers import pipeline
            _asr.append(pipeline("automatic-speech-recognition", "openai/whisper-large-v3-turbo",
                                 torch_dtype=torch.float16, device="cuda:0"))
        asr = _asr[0]
        tok = asr.tokenizer
        lang_ids = {code: tok.convert_tokens_to_ids(f"<|{code}|>")
                    for code in ("en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar",
                                 "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu",
                                 "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa",
                                 "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn",
                                 "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc",
                                 "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn",
                                 "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw",
                                 "su", "yue")}
        ids = torch.tensor(list(lang_ids.values()), device="cuda:0")
        sot = torch.tensor([[tok.convert_tokens_to_ids("<|startoftranscript|>")]], device="cuda:0")
        for b in req.audios_b64:
            y, _ = librosa.load(io.BytesIO(base64.b64decode(b)), sr=16000, mono=True)
            with torch.no_grad():
                mos = float(_mos[0](torch.from_numpy(y).unsqueeze(0).to("cuda:0"), 16000)[0])
                feats = asr.feature_extractor(y, sampling_rate=16000, return_tensors="pt").input_features
                logits = asr.model(input_features=feats.to("cuda:0", torch.float16), decoder_input_ids=sot).logits[0, -1]
                p = torch.softmax(logits[ids].float(), dim=-1)
            probs = dict(zip(lang_ids, p.tolist()))
            out.append({"mos": round(mos, 3), "lang": {k: round(probs.get(k, 0.0), 4) for k in req.langs}})
    return {"scores": out}


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


# ---------------------------------------------------------------- dub-pipeline jobs (raw bytes in, raw bytes out)
from fastapi import Request
from starlette.concurrency import run_in_threadpool

MSST_DIR = os.environ.get("MSST_DIR", "/content/MSST")
DNR_CKPT = os.path.join(MSST_DIR, "model_bandit_plus_dnr_sdr_11.47.chpt")
DNR_CFG = os.path.join(MSST_DIR, "configs", "config_dnr_bandit_bsrnn_multi_mus64.yaml")
SEP_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
SEP_OUT = "/content/sep_out"
_sep, _dnr, _fw = [], [], []


def _flac(y, sr):
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="FLAC", subtype="PCM_24")
    return buf.getvalue()


def _read(data, sr=None, mono=False):
    """audio bytes -> float32 (samples, channels) at sr (None = keep)"""
    import librosa
    y, r = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    if mono:
        y = y.mean(axis=1, keepdims=True)
    if sr and r != sr:
        y = librosa.resample(y.T, orig_sr=r, target_sr=sr).T
        r = sr
    return y, r


def _separate(data):
    import glob
    import shutil
    with gpu_lock:
        if not _sep:
            from audio_separator.separator import Separator
            s = Separator(output_dir=SEP_OUT, output_format="FLAC", output_single_stem="Vocals",
                          model_file_dir="/content/sep_models")
            s.load_model(SEP_MODEL)
            _sep.append(s)
        shutil.rmtree(SEP_OUT, ignore_errors=True)
        os.makedirs(SEP_OUT, exist_ok=True)
        src = os.path.join(SEP_OUT, "in.flac")
        with open(src, "wb") as f:
            f.write(data)
        _sep[0].separate(src)
        torch.cuda.empty_cache()
    voc = [p for p in glob.glob(os.path.join(SEP_OUT, "*")) if "vocals" in os.path.basename(p).lower()]
    if not voc:
        raise HTTPException(500, f"no vocals stem: {os.listdir(SEP_OUT)}")
    with open(voc[0], "rb") as f:
        return f.read()


def _dnr_effects(data):
    import numpy as np
    y, _ = _read(data, 44100)
    mix = y.T                                   # (channels, samples)
    if mix.shape[0] == 1:
        mix = np.concatenate([mix, mix])
    with gpu_lock:
        if not _dnr:
            import sys
            sys.path.insert(0, MSST_DIR)
            from utils.settings import get_model_from_config
            m, cfg = get_model_from_config("bandit", DNR_CFG)
            sd = torch.load(DNR_CKPT, map_location="cpu", weights_only=False)
            m.load_state_dict(sd["state_dict"] if "state_dict" in sd else sd)
            _dnr.extend([m.to("cuda:0").eval(), cfg])
        from utils.model_utils import demix
        with torch.no_grad():
            out = demix(_dnr[1], _dnr[0], mix, torch.device("cuda:0"), model_type="bandit")
        torch.cuda.empty_cache()
    return _flac(out["effects"].T, 44100)


def _whisper(data, language, vad, beam, words):
    y, _ = _read(data, 16000, mono=True)
    with gpu_lock:
        if not _fw:
            from faster_whisper import WhisperModel
            _fw.append(WhisperModel("large-v3", device="cuda", compute_type="float16"))
        segs, info = _fw[0].transcribe(y[:, 0], language=language or None, vad_filter=bool(vad),
                                       beam_size=beam, word_timestamps=bool(words))
        out = [{"start": s.start, "end": s.end, "text": s.text,
                "words": [[w.start, w.end, w.word, w.probability] for w in (s.words or [])]} for s in segs]
    return {"duration": info.duration, "segments": out}


@app.post("/separate")
async def separate(request: Request, x_api_key: str = Header(None)):
    _auth(x_api_key)
    return Response(await run_in_threadpool(_separate, await request.body()), media_type="audio/flac")


@app.post("/dnr")
async def dnr(request: Request, x_api_key: str = Header(None)):
    _auth(x_api_key)
    return Response(await run_in_threadpool(_dnr_effects, await request.body()), media_type="audio/flac")


@app.post("/whisper")
async def whisper(request: Request, language: str = "zh", vad: int = 1, beam: int = 5, words: int = 1,
                  x_api_key: str = Header(None)):
    _auth(x_api_key)
    return await run_in_threadpool(_whisper, await request.body(), language, vad, beam, words)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("OV_PORT", 8000)))
