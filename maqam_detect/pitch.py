from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np


@dataclass
class PitchTrack:
    times: np.ndarray
    f0_hz: np.ndarray
    voiced: np.ndarray
    confidence: np.ndarray


def extract_pitch(
    path: Path | str,
    fmin: float = 65.0,
    fmax: float = 1200.0,
    frame_length: int = 2048,
    hop_length: int = 256,
    sr: int = 22050,
) -> PitchTrack:
    y, sr = librosa.load(str(path), sr=sr, mono=True)
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    return PitchTrack(
        times=times,
        f0_hz=f0,
        voiced=voiced_flag.astype(bool),
        confidence=voiced_prob,
    )
