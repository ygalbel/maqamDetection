from .pitch import PitchTrack, extract_pitch
from .histogram import pitch_class_distribution
from .tonic import last_note_tonic_seed
from .templates import MAQAMAT, all_templates, build_template
from .classify import Match, classify

__all__ = [
    "PitchTrack",
    "extract_pitch",
    "pitch_class_distribution",
    "last_note_tonic_seed",
    "MAQAMAT",
    "all_templates",
    "build_template",
    "Match",
    "classify",
]
