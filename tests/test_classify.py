import numpy as np

from maqam_detect.classify import classify
from maqam_detect.histogram import BINS
from maqam_detect.templates import MAQAMAT, build_template


def test_each_template_classifies_as_itself():
    for name, degrees in MAQAMAT.items():
        tpl = build_template(degrees)
        matches = classify(tpl)
        assert matches[0].maqam == name, f"{name} → {matches[0].maqam}"
        assert matches[0].tonic_shift_cents == 0


def test_shifted_template_recovers_tonic():
    rast = build_template(MAQAMAT["Rast"])
    for shift in (100, 350, 500, 700, 950):
        shifted = np.roll(rast, shift)
        matches = classify(shifted)
        assert matches[0].maqam == "Rast"
        assert abs(matches[0].tonic_shift_cents - shift) <= 2


def test_seed_constraint_picks_nearby_shift():
    rast = build_template(MAQAMAT["Rast"])
    shifted = np.roll(rast, 500)
    matches = classify(shifted, tonic_seed_cents=510, seed_tolerance_cents=30)
    assert matches[0].maqam == "Rast"
    assert abs(matches[0].tonic_shift_cents - 500) <= 5


def test_pcd_is_normalized():
    rast = build_template(MAQAMAT["Rast"])
    assert abs(rast.sum() - 1.0) < 1e-9
    assert rast.shape == (BINS,)
