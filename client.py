# -*- coding: utf-8 -*-
"""Send text (+ optional reference voice) to the OmniVoice server on Colab, save the WAV.

  python client.py --url https://xxxx.trycloudflare.com --key KEY --text "สวัสดีครับ" --out out.wav
  python client.py --url ... --key ... --text-file lines.txt --ref voice.wav --ref-text "..." --out-dir out\

--text-file = one line per clip (out_dir\0001.wav, 0002.wav, ...).
Import use: from client import tts; wav_bytes = tts(url, key, "text", ref="voice.wav")
Stdlib only.
"""
import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request


def tts(url, key, text, ref=None, ref_text=None, instruct=None, language=None,
        speed=None, duration=None, num_step=32, timeout=600):
    body = {"text": text, "ref_text": ref_text, "instruct": instruct, "language": language,
            "speed": speed, "duration": duration, "num_step": num_step}
    if ref:
        with open(ref, "rb") as f:
            body["ref_audio_b64"] = base64.b64encode(f.read()).decode()
    req = urllib.request.Request(url.rstrip("/") + "/tts", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "X-API-Key": key})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}") from None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--key", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--text")
    g.add_argument("--text-file")
    p.add_argument("--ref", help="reference voice wav/mp3, 3-10 s")
    p.add_argument("--ref-text", help="what the reference says (omit = server Whisper transcribes)")
    p.add_argument("--instruct", help='voice design, e.g. "female, low pitch"')
    p.add_argument("--language", help="e.g. th")
    p.add_argument("--speed", type=float)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--out", default="out.wav")
    p.add_argument("--out-dir", default="out")
    a = p.parse_args()
    kw = dict(ref=a.ref, ref_text=a.ref_text, instruct=a.instruct, language=a.language,
              speed=a.speed, num_step=a.steps)
    if a.text:
        jobs = [(a.text, a.out)]
    else:
        os.makedirs(a.out_dir, exist_ok=True)
        with open(a.text_file, encoding="utf-8-sig") as f:
            lines = [l.strip() for l in f if l.strip()]
        jobs = [(t, os.path.join(a.out_dir, f"{i:04d}.wav")) for i, t in enumerate(lines, 1)]
    for text, out in jobs:
        with open(out, "wb") as f:
            f.write(tts(a.url, a.key, text, **kw))
        print("ok", out, flush=True)


if __name__ == "__main__":
    sys.exit(main())
