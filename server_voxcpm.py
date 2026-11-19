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
  add ?wait=0 to get {"job": id} at once, then GET /job/<id>: 202 = still running, 200 = the result, 404 = server
  restarted (submit again). Needed behind the Cloudflare tunnel, which cuts requests that take over 100 s (HTTP 524).
POST /separate -> FLAC vocals (BS-Roformer model_bs_roformer_ep_317_sdr_12.9755, via audio-separator)
POST /dnr      -> FLAC effects stem, 44.1 kHz stereo (Bandit-Plus DnR via MSST; music and speech dropped)
POST /whisper?language=zh&vad=1&beam=5&words=1 -> {"duration", "segments": [{start, end, text, words: [[s, e, w, p]]}]}
               (faster-whisper large-v3, same call Dub Studio makes locally)
GET /health -> {"ok": true, "gpu": ..., "sr": ..., "idle": seconds since the last job, "up": uptime}
            (?keep=1 counts as a job: the caller still needs the server; the notebook watchdog frees the runtime after a long idle)

Env: OV_KEY (required), OV_PORT (8000), VOX_MODEL (openbmb/VoxCPM2), VOX_OPTIMIZE (1 = torch.compile),
     VOX_DTYPE (override the checkpoint's bfloat16; default float16 on GPUs older than Ampere, e.g. T4).
"""
import sys

if "--whisper-worker" in sys.argv:
    # Separate process: CTranslate2 (faster-whisper) is built for CUDA 12, Colab's torch is CUDA 13 — the parent
    # starts this with LD_LIBRARY_PATH on the cu12 pip libs. One JSON request per stdin line, one "@@"+JSON answer.
    import json
    import soundfile as sf
    from faster_whisper import WhisperModel
    _m = WhisperModel("large-v3", device="cuda", compute_type="float16")
    print("@@" + json.dumps({"ready": True}), flush=True)
    for _line in sys.stdin:
        _q = json.loads(_line)
        try:
            # array, not path: faster-whisper's own decoder (PyAV) breaks on Colab's newer PyAV
            _segs, _info = _m.transcribe(sf.read(_q["path"], dtype="float32")[0], language=_q["language"] or None,
                                         vad_filter=_q["vad"],
                                         beam_size=_q["beam"], word_timestamps=_q["words"])
            _out = {"duration": _info.duration, "segments": [
                {"start": s.start, "end": s.end, "text": s.text,
                 "words": [[w.start, w.end, w.word, w.probability] for w in (s.words or [])]} for s in _segs]}
        except Exception as e:
            _out = {"error": repr(e)}
        print("@@" + json.dumps(_out), flush=True)
    raise SystemExit

import base64
import io
import json
import os
import tempfile
import threading
import time

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

# Voice-clone requests (tts / asr / embed / judge) are short and come one after another, several per line. Queued
# behind separation or effects jobs (60-90 s each) every one of them waited a whole long job: 0 lines in 14 minutes
# on a real 2-hour clip. So short requests go first, and a long job only starts once no short request has been seen
# for SHORT_GRACE seconds - the long jobs fill the gaps while the clone stage is idle. Nothing is lost: the GPU never
# sits idle while work is waiting for longer than the grace period.
SHORT_GRACE = 8.0
_short = {"n": 0, "last": 0.0}
_short_guard = threading.Lock()


class _short_gpu:
    def __enter__(self):
        with _short_guard:
            _short["n"] += 1
        gpu_lock.acquire()

    def __exit__(self, *exc):
        gpu_lock.release()
        with _short_guard:
            _short["n"] -= 1
            _short["last"] = time.time()


def _long_gate():
    """long jobs call this before taking gpu_lock"""
    while _short["n"] or time.time() - _short["last"] < SHORT_GRACE:
        time.sleep(0.25)


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


_t0 = time.time()
_last_job = [time.time()]      # last real request (or /health?keep=1): the notebook releases the runtime after a long idle


_now = {"info": "", "t": 0.0}  # what the caller says it is doing (header X-Job-Info), shown by the notebook watchdog


@app.middleware("http")
async def _touch(request, call_next):
    if request.url.path != "/health" or request.query_params.get("keep"):
        _last_job[0] = time.time()
    info = request.headers.get("x-job-info")
    if info:
        import urllib.parse
        _now.update(info=urllib.parse.unquote(info)[:200], t=time.time())
    return await call_next(request)


@app.get("/health")
def health():
    return {"ok": True, "gpu": torch.cuda.get_device_name(0), "sr": SR, "engine": "voxcpm2",
            "dtype": model.tts_model.config.dtype, "idle": int(time.time() - _last_job[0]), "up": int(time.time() - _t0),
            "now": _now["info"], "now_age": int(time.time() - _now["t"])}


# ?wait=0 on these four as well (returns {"job": id}; result from GET /job/<id>): a short request can sit behind a long
# separation or effects job on the single GPU for more than the tunnel's 100 s and be cut with HTTP 524.
@app.post("/tts")
def tts(req: TTS, wait: int = 1, x_api_key: str = Header(None)):
    _auth(x_api_key)
    if not wait:
        return _start(tts, req, 1, x_api_key)
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
        with _short_gpu():
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
def asr(req: ASR, wait: int = 1, x_api_key: str = Header(None)):
    _auth(x_api_key)
    if not wait:
        return _start(asr, req, 1, x_api_key)
    files = []
    try:
        p = _tmp(req.audio_b64, files)
        with _short_gpu():
            if not _asr:
                from transformers import pipeline
                _asr.append(pipeline("automatic-speech-recognition", "openai/whisper-large-v3-turbo",
                                     torch_dtype=torch.float16, device="cuda:0"))
            gk = {"language": req.language, "task": "transcribe"} if req.language else {"task": "transcribe"}
            out = _asr[0](p, generate_kwargs=gk, return_timestamps=True)    # needed for audio over 30 s (else ValueError -> 500)
    finally:
        for f in files:
            os.remove(f)
    return {"text": out["text"].strip()}


class JUDGE(BaseModel):
    audios_b64: list[str]
    langs: list[str] = ["th", "zh", "en"]


_mos = []                     # lazily loaded UTMOS predictor


@app.post("/judge")
def judge(req: JUDGE, wait: int = 1, x_api_key: str = Header(None)):
    """Machine ears for picking takes: UTMOS naturalness (1-5) + Whisper language-ID probabilities
    (low "th" / high "zh" on a Thai line = foreign accent)."""
    _auth(x_api_key)
    if not wait:
        return _start(judge, req, 1, x_api_key)
    import librosa
    out = []
    with _short_gpu():
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
def embed(req: EMB, wait: int = 1, x_api_key: str = Header(None)):
    _auth(x_api_key)
    if not wait:
        return _start(embed, req, 1, x_api_key)
    import librosa
    out = []
    with _short_gpu():
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
    sf.write(buf, y, sr, format="FLAC", subtype="PCM_16")
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


def _separate(data, sr=0, mono=0):
    import glob
    import shutil
    _long_gate()
    with gpu_lock:
        if not _sep:
            from audio_separator.separator import Separator
            # overlap 4 (default 8) + batch 4 + autocast: about 3x faster on a T4, still clean enough for ASR and clone references
            s = Separator(output_dir=SEP_OUT, output_format="FLAC", output_single_stem="Vocals",
                          model_file_dir="/content/sep_models", use_autocast=True,
                          mdxc_params={"segment_size": 256, "override_model_segment_size": False, "batch_size": 4,
                                       "overlap": 4, "pitch_shift": 0})
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
        out = f.read()
    if sr or mono:
        out = _flac(*_read(out, sr or None, bool(mono)))
    return out


def _dnr_effects(data):
    import numpy as np
    y, _ = _read(data, 44100)
    mix = y.T                                   # (channels, samples)
    if mix.shape[0] == 1:
        mix = np.concatenate([mix, mix])
    _long_gate()
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


def _worker_answer(p):
    while True:
        line = p.stdout.readline()
        if not line:
            raise HTTPException(500, "whisper worker died - see /content/whisper_worker.log")
        if line.startswith("@@"):
            return json.loads(line[2:])


def _whisper(data, language, vad, beam, words):
    import glob
    import site
    import subprocess
    y, _ = _read(data, 16000, mono=True)
    path = "/content/_whisper_in.wav"
    _long_gate()
    with gpu_lock:
        if not _fw or _fw[0].poll() is not None:
            cu12 = [d for sp in site.getsitepackages() for d in glob.glob(os.path.join(sp, "nvidia", "*", "lib"))
                    if os.sep + "cu13" + os.sep not in d]
            env = dict(os.environ, LD_LIBRARY_PATH=":".join(cu12 + [os.environ.get("LD_LIBRARY_PATH", "")]))
            _fw[:] = [subprocess.Popen([sys.executable, "-u", os.path.abspath(__file__), "--whisper-worker"],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env,
                                       stderr=open("/content/whisper_worker.log", "a"))]
            _worker_answer(_fw[0])                       # "ready" (first time: downloads large-v3)
        sf.write(path, y[:, 0], 16000, subtype="PCM_16")
        _fw[0].stdin.write(json.dumps({"path": path, "language": language, "vad": bool(vad), "beam": beam,
                                       "words": bool(words)}) + chr(10))
        _fw[0].stdin.flush()
        out = _worker_answer(_fw[0])
    if "error" in out:
        raise HTTPException(500, out["error"])
    return out


# Cloudflare quick tunnels cut any request that takes longer than 100 s to answer (HTTP 524), so heavy jobs are
# submitted with ?wait=0 (returns {"job": id} at once) and fetched from GET /job/<id> (202 while running).
_jobs = {}


def _start(fn, *a, ctype="application/json"):
    jid = base64.urlsafe_b64encode(os.urandom(9)).decode()
    _jobs[jid] = j = {"t": time.time(), "done": 0.0, "out": None, "err": None, "ctype": ctype}

    def run():
        try:
            j["out"] = fn(*a)
        except HTTPException as e:
            j["err"] = str(e.detail)
        except Exception as e:
            j["err"] = repr(e)
        j["done"] = time.time()
    threading.Thread(target=run, daemon=True).start()
    for k in [k for k, v in _jobs.items() if v["done"] and time.time() - v["done"] > 600]:   # fetched or forgotten results
        _jobs.pop(k, None)
    return {"job": jid}


@app.get("/job/{jid}")
def job(jid: str, x_api_key: str = Header(None)):
    _auth(x_api_key)
    j = _jobs.get(jid)
    if j is None:
        raise HTTPException(404, "no such job (server restarted?) - submit it again")
    if not j["done"]:
        return Response(status_code=202)
    if j["err"]:
        raise HTTPException(500, j["err"])
    return Response(j["out"], media_type=j["ctype"]) if isinstance(j["out"], bytes) else j["out"]


# The tunnel's 100 s limit includes the time the caller needs to upload the request body: 25 MB of audio over a
# 1 Mbit/s uplink never makes it (HTTP 524 every time). So a big body is sent first in small parts with
# POST /put/<id>/<i>, and the job endpoint is then called with ?up=<id>&n=<parts> and an empty body.
UP_DIR = "/content/_up"
_UP_ID = __import__("re").compile(r"[0-9a-f]{32}")


@app.post("/put/{up}/{i}")
async def put(up: str, i: int, request: Request, x_api_key: str = Header(None)):
    _auth(x_api_key)
    if not _UP_ID.fullmatch(up) or not 0 <= i < 4096:
        raise HTTPException(400, "bad upload id")
    import shutil
    data = await request.body()
    d = os.path.join(UP_DIR, up)
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f"{i:05d}.tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, os.path.join(d, f"{i:05d}"))
    for name in os.listdir(UP_DIR):                      # uploads nobody came back for (kept 1 h so a retried job call still finds them)
        old = os.path.join(UP_DIR, name)
        if name != up and time.time() - os.path.getmtime(old) > 3600:
            shutil.rmtree(old, ignore_errors=True)
    return {"ok": True, "size": len(data)}


def _body(data, up, n):
    """the request body, or the n parts uploaded earlier under id `up`"""
    if not up:
        return data
    if not _UP_ID.fullmatch(up) or not 0 < n <= 4096:
        raise HTTPException(400, "bad upload id")
    parts = [os.path.join(UP_DIR, up, f"{i:05d}") for i in range(n)]
    if not all(os.path.exists(p) for p in parts):
        raise HTTPException(409, "upload incomplete (server restarted?) - send the parts again")
    out = bytearray()
    for p in parts:
        with open(p, "rb") as f:
            out += f.read()
    return bytes(out)


@app.post("/separate")
async def separate(request: Request, wait: int = 1, up: str = "", n: int = 0, sr: int = 0, mono: int = 0,
                   x_api_key: str = Header(None)):
    _auth(x_api_key)
    data = _body(await request.body(), up, n)
    if not wait:
        return _start(_separate, data, sr, mono, ctype="audio/flac")
    return Response(await run_in_threadpool(_separate, data, sr, mono), media_type="audio/flac")


@app.post("/dnr")
async def dnr(request: Request, wait: int = 1, up: str = "", n: int = 0, x_api_key: str = Header(None)):
    _auth(x_api_key)
    data = _body(await request.body(), up, n)
    if not wait:
        return _start(_dnr_effects, data, ctype="audio/flac")
    return Response(await run_in_threadpool(_dnr_effects, data), media_type="audio/flac")


@app.post("/whisper")
async def whisper(request: Request, language: str = "zh", vad: int = 1, beam: int = 5, words: int = 1, wait: int = 1,
                  up: str = "", n: int = 0, x_api_key: str = Header(None)):
    _auth(x_api_key)
    data = _body(await request.body(), up, n)
    if not wait:
        return _start(_whisper, data, language, vad, beam, words)
    return await run_in_threadpool(_whisper, data, language, vad, beam, words)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("OV_PORT", 8000)))
