from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from .classify import classify
from .histogram import BINS, REFERENCE_HZ, pitch_class_distribution
from .pitch import extract_pitch
from .tonic import last_note_tonic_seed

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def hz_to_note_name(hz: float) -> str:
    midi = 69 + 12 * np.log2(hz / 440.0)
    midi_round = int(round(midi))
    cents_off = (midi - midi_round) * 100.0
    name = NOTE_NAMES[midi_round % 12]
    octave = midi_round // 12 - 1
    sign = "+" if cents_off >= 0 else ""
    return f"{name}{octave} ({sign}{cents_off:.0f}c)"


def parse_tonic(spec: str) -> float:
    """Accept '294hz', '294', 'D4', 'd#3'."""
    s = spec.strip().lower()
    m = re.fullmatch(r"([0-9]*\.?[0-9]+)\s*(hz)?", s)
    if m:
        return float(m.group(1))
    m = re.fullmatch(r"([a-g])([#b]?)(-?\d+)", s)
    if not m:
        raise ValueError(f"unrecognized tonic spec: {spec!r}")
    letter, accidental, octave = m.group(1).upper(), m.group(2), int(m.group(3))
    pc = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[letter]
    if accidental == "#":
        pc += 1
    elif accidental == "b":
        pc -= 1
    midi = (octave + 1) * 12 + pc
    return float(440.0 * 2 ** ((midi - 69) / 12))


def cents_to_hz_near(cents_mod: float, near_hz: float) -> float:
    """Resolve a cents-mod-1200 value to actual Hz in the octave of `near_hz`."""
    near_cents = 1200.0 * np.log2(near_hz / REFERENCE_HZ)
    octave = round((near_cents - cents_mod) / 1200.0)
    cents_total = cents_mod + 1200.0 * octave
    return float(REFERENCE_HZ * (2.0 ** (cents_total / 1200.0)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="maqam-detect", description="Detect Arabic maqam from audio.")
    parser.add_argument("audio", type=Path, help="Path to audio file (wav, mp3, flac, ...)")
    parser.add_argument("--tonic", type=str, default=None, help="Override tonic (e.g. D4, 294hz)")
    parser.add_argument("--top", type=int, default=3, help="Show top N matches")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    args = parser.parse_args(argv)

    if not args.audio.exists():
        print(f"error: {args.audio} not found", file=sys.stderr)
        return 1

    track = extract_pitch(args.audio)
    pcd = pitch_class_distribution(track)

    if args.tonic:
        tonic_hz = parse_tonic(args.tonic)
        seed_cents = (1200.0 * np.log2(tonic_hz / REFERENCE_HZ)) % BINS
    else:
        seed = last_note_tonic_seed(track)
        if seed is None:
            print("error: no voiced pitch detected", file=sys.stderr)
            return 2
        seed_cents, tonic_hz = seed

    matches = classify(pcd, tonic_seed_cents=seed_cents)
    top = matches[: args.top]
    winner_hz = cents_to_hz_near(top[0].tonic_shift_cents, tonic_hz)

    if args.json:
        total = sum(m.score for m in matches) or 1.0
        print(json.dumps({
            "file": str(args.audio),
            "tonic_hz": winner_hz,
            "tonic_note": hz_to_note_name(winner_hz),
            "matches": [
                {
                    "maqam": m.maqam,
                    "score": m.score,
                    "confidence": m.score / total,
                    "tonic_shift_cents": m.tonic_shift_cents,
                }
                for m in top
            ],
        }, indent=2))
    else:
        print(f"Tonic: {hz_to_note_name(winner_hz)} ({winner_hz:.1f} Hz)")
        total = sum(m.score for m in matches) or 1.0
        print("Top matches:")
        for i, m in enumerate(top, 1):
            print(f"  {i}. {m.maqam:<10}  score {m.score:.3f}  conf {m.score / total:.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
