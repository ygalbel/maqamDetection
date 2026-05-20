from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .classify import classify
from .explain import align_to_template, detected_peaks, interval_label
from .histogram import BINS, REFERENCE_HZ, pitch_class_distribution
from .pitch import extract_pitch
from .sources import resolve_audio
from .templates import MAQAMAT
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
    near_cents = 1200.0 * np.log2(near_hz / REFERENCE_HZ)
    octave = round((near_cents - cents_mod) / 1200.0)
    cents_total = cents_mod + 1200.0 * octave
    return float(REFERENCE_HZ * (2.0 ** (cents_total / 1200.0)))


class Logger:
    """Buffered logger: mirror to stderr (if verbose) AND collect for the log file."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.buffer: list[str] = []

    def write(self, line: str = "") -> None:
        self.buffer.append(line)
        if self.verbose:
            try:
                print(line, file=sys.stderr)
            except UnicodeEncodeError:
                print(line.encode("ascii", "replace").decode("ascii"), file=sys.stderr)

    def flush_to(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.buffer) + "\n", encoding="utf-8")


def _default_log_path(audio_path: Path, source_arg: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Use the resolved audio's stem; for URLs that often gives the video id.
    stem = audio_path.stem or "maqam-detect"
    return Path.cwd() / f"{stem}_{stamp}.log"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="maqam-detect",
        description="Detect Arabic maqam from audio.",
    )
    parser.add_argument("audio", type=str,
                        help="Path to audio file (wav, mp3, flac, ...) OR URL (YouTube etc.)")
    parser.add_argument("--tonic", type=str, default=None,
                        help="Override tonic (e.g. D4, 294hz)")
    parser.add_argument("--top", type=int, default=3,
                        help="Show top N matches")
    parser.add_argument("--json", action="store_true",
                        help="Emit JSON to stdout instead of text")
    parser.add_argument("--log", type=Path, default=None,
                        help="Write detailed analysis log to this path "
                             "(default: ./<stem>_<timestamp>.log in current dir)")
    parser.add_argument("--no-log", action="store_true",
                        help="Skip writing a log file")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress stage progress on stderr")
    args = parser.parse_args(argv)

    log = Logger(verbose=not args.quiet)
    t_start = time.time()
    log.write(f"=== maqam-detect ===")
    log.write(f"input:     {args.audio}")
    log.write(f"started:   {datetime.now().isoformat(timespec='seconds')}")

    bar = tqdm(total=5, desc="starting", ascii=True, leave=False,
               disable=args.quiet or args.json,
               bar_format="{desc:<32} [{bar:20}] {n_fmt}/{total_fmt} {elapsed}",
               file=sys.stderr)

    # 1. Resolve audio
    bar.set_description("resolving audio")
    t0 = time.time()
    try:
        audio_path = resolve_audio(args.audio)
    except FileNotFoundError as e:
        bar.close()
        print(f"error: {e} not found", file=sys.stderr)
        return 1
    except Exception as e:
        bar.close()
        print(f"error resolving audio: {e}", file=sys.stderr)
        return 1
    log.write(f"resolved:  {audio_path}  ({time.time() - t0:.1f}s)")
    bar.update(1)

    # 2. Extract pitch
    bar.set_description("extracting pitch (pYIN)")
    t0 = time.time()
    track = extract_pitch(audio_path)
    voiced_n = int(track.voiced.sum())
    total_n = int(track.voiced.size)
    log.write(f"audio:     {total_n} frames, {voiced_n} voiced "
              f"({voiced_n/total_n:.0%}), mean conf {track.confidence.mean():.2f}")
    log.write(f"pYIN:      {time.time() - t0:.1f}s")
    bar.update(1)

    # 3. Build PCD
    bar.set_description("building pitch histogram")
    t0 = time.time()
    pcd = pitch_class_distribution(track)
    log.write(f"PCD:       1200-bin smoothed histogram ({time.time() - t0:.2f}s)")
    bar.update(1)

    # 4. Tonic
    bar.set_description("detecting tonic")
    t0 = time.time()
    if args.tonic:
        tonic_hz = parse_tonic(args.tonic)
        seed_cents = (1200.0 * np.log2(tonic_hz / REFERENCE_HZ)) % BINS
        log.write(f"tonic:     OVERRIDE {hz_to_note_name(tonic_hz)} ({tonic_hz:.1f} Hz)")
    else:
        seed = last_note_tonic_seed(track)
        if seed is None:
            bar.close()
            print("error: no voiced pitch detected", file=sys.stderr)
            return 2
        seed_cents, tonic_hz = seed
        log.write(f"tonic:     {hz_to_note_name(tonic_hz)} ({tonic_hz:.1f} Hz) "
                  f"[qarar heuristic, {time.time() - t0:.2f}s]")
    bar.update(1)

    # 5. Classify
    bar.set_description("classifying")
    t0 = time.time()
    matches = classify(pcd, tonic_seed_cents=seed_cents)
    top = matches[: args.top]
    winner = top[0]
    winner_hz = cents_to_hz_near(winner.tonic_shift_cents, tonic_hz)
    log.write(f"classify:  {time.time() - t0:.2f}s")
    bar.update(1)
    bar.close()

    # === Detailed analysis section ===
    log.write("")
    log.write("--- Detected notes (peaks in pitch histogram) ---")
    peaks = detected_peaks(pcd)
    if not peaks:
        log.write("  (no peaks detected)")
    else:
        log.write(f"{'note':<14}{'abs cents':>10}{'from tonic':>12}{'interval':>26}{'height':>10}")
        for bin_pos, height in peaks:
            hz_at_peak = cents_to_hz_near(bin_pos, winner_hz)
            note = hz_to_note_name(hz_at_peak)
            from_tonic = (bin_pos - winner.tonic_shift_cents) % BINS
            log.write(f"{note:<14}{bin_pos:>10}{from_tonic:>12}{interval_label(from_tonic):>26}"
                      f"{height*1000:>9.2f}")

    log.write("")
    log.write("--- Maqam scores (Bhattacharyya) ---")
    for i, m in enumerate(matches, 1):
        mark = "  <-- chosen" if i == 1 else ""
        log.write(f"  {i}. {m.maqam:<10}  {m.score:.4f}{mark}")

    log.write("")
    log.write(f"--- Why {winner.maqam}? ---")
    aligned, missing = align_to_template(peaks, winner.maqam, winner.tonic_shift_cents)
    log.write(f"Tonic alignment:  shift {winner.tonic_shift_cents}c -> tonic at "
              f"{hz_to_note_name(winner_hz)}")
    log.write(f"Aligned degrees ({len(aligned)}/{len(MAQAMAT[winner.maqam])}):")
    for cents_rel, weight, peak_bin, height, dist in aligned:
        log.write(f"  +{cents_rel:>4}c  ({interval_label(cents_rel):<24}) "
                  f"w={weight:.1f}  matched peak @ {peak_bin}c (off by {dist}c, h={height*1000:.2f})")
    if missing:
        log.write(f"Missing degrees ({len(missing)}):")
        for cents_rel, weight in missing:
            log.write(f"  +{cents_rel:>4}c  ({interval_label(cents_rel):<24}) "
                      f"w={weight:.1f}  -- no detected peak within 35c")
    log.write("")
    log.write(f"finished:  {datetime.now().isoformat(timespec='seconds')} "
              f"(total {time.time() - t_start:.1f}s)")

    # Write log file
    if not args.no_log:
        log_path = args.log or _default_log_path(audio_path, args.audio)
        log.flush_to(log_path)
        log.write(f"log:       {log_path}")
        if not args.quiet:
            print(f"[log] {log_path}", file=sys.stderr)

    # === Stdout output ===
    if args.json:
        total = sum(m.score for m in matches) or 1.0
        print(json.dumps({
            "file": str(audio_path),
            "source": args.audio,
            "tonic_hz": winner_hz,
            "tonic_note": hz_to_note_name(winner_hz),
            "results": winner.maqam,
            "possibilities": [
                {
                    "maqam": m.maqam,
                    "score": m.score,
                    "confidence": m.score / total,
                    "tonic_shift_cents": m.tonic_shift_cents,
                }
                for m in top[1:]
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
