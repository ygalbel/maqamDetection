"""Resolve an audio argument (local path or URL) to a local file path."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _cache_dir() -> Path:
    cache = Path(os.environ.get("MAQAM_CACHE", Path.home() / ".cache" / "maqam-detect"))
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _is_url(s: str) -> bool:
    return s.startswith(("http://", "https://"))


def _download(url: str) -> Path:
    try:
        import yt_dlp
    except ImportError as e:
        raise RuntimeError("yt-dlp not installed. pip install yt-dlp") from e

    cache = _cache_dir()
    # YouTube's JS challenge requires a runtime; auto-fetch the solver from yt-dlp's GitHub
    # unless the user opts out. Override with MAQAM_NO_REMOTE_COMPONENTS=1.
    base_opts: dict = {"quiet": True, "noplaylist": True}
    if not os.environ.get("MAQAM_NO_REMOTE_COMPONENTS"):
        base_opts["remote_components"] = ["ejs:github"]

    probe_opts = {**base_opts, "skip_download": True}
    with yt_dlp.YoutubeDL(probe_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    video_id = info.get("id", "unknown")
    existing = list(cache.glob(f"{video_id}.*"))
    if existing:
        return existing[0]

    download_opts = {
        **base_opts,
        "format": "bestaudio",
        "outtmpl": str(cache / "%(id)s.%(ext)s"),
        # Transcode to wav so soundfile can read it natively (avoids audioread fallback).
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }],
    }
    title = info.get("title", url).encode("ascii", "replace").decode("ascii")
    print(f"[download] {title}", file=sys.stderr)
    with yt_dlp.YoutubeDL(download_opts) as ydl:
        info = ydl.extract_info(url, download=True)
    out = Path(ydl.prepare_filename(info)).with_suffix(".wav")
    return out


def resolve_audio(arg: str) -> Path:
    if _is_url(arg):
        return _download(arg)
    p = Path(arg)
    if not p.exists():
        raise FileNotFoundError(p)
    return p
