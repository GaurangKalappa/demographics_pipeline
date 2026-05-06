# ============================================================
#  utils/decision.py  –  Ensemble logic + age-range mapping
# ============================================================

from typing import Optional, Tuple
from config import (
    AGE_BINS,
    FACE_DOMINANCE_THRESH,
    BODY_DOMINANCE_THRESH,
)


# ── Age range mapper ────────────────────────────────────────────────────────────

def map_age_to_range(age_raw: float) -> str:
    """
    Map a continuous age prediction to a static age-range label.

    Bins (from spec):
        0-12   → Child
        13-18  → Teen
        19-30  → Young Adult
        31-45  → Adult
        46-60  → Middle-Aged
        60+    → Senior
    """
    age = max(0.0, float(age_raw))
    for lo, hi, label in AGE_BINS:
        if lo <= age <= hi:
            return label
    return "Senior"   # fallback for age > 999


# ── Decision ensemble ───────────────────────────────────────────────────────────

class DecisionResult:
    """Holds the resolved demographic prediction for one Track ID."""
    __slots__ = ("gender", "gender_score", "age_raw", "age_range", "source")

    def __init__(self, gender: str, gender_score: float,
                 age_raw: float, age_range: str, source: str):
        self.gender       = gender        # "Male" | "Female"
        self.gender_score = gender_score  # probability of Female [0,1]
        self.age_raw      = age_raw
        self.age_range    = age_range
        self.source       = source        # "face" | "body" | "ensemble"


def resolve(
    c_face:            Optional[float],   # Highest face detection confidence
    gender_face_score: Optional[float],   # P(Female) from MTL model  [0,1]
    age_raw_face:      Optional[float],   # Continuous age from MTL model

    c_body:            float,             # Body PAR confidence [0,1]
    gender_body_score: float,             # P(Female) from PAR model [0,1]
    age_coarse:        str,               # Coarse age label from PAR
) -> DecisionResult:
    """
    Apply the three-case metric decision logic from the spec.

    Case A  (C_face > 0.85)       → Use Face MTL exclusively
    Case B  (C_face < 0.60)       → Use Body PAR exclusively
    Case C  (0.60 ≤ C_face ≤ 0.85) → Weighted ensemble
                w = C_face / (C_face + C_body)
                G_final = w·G_face + (1-w)·G_body
    """

    # No face detected at all → Case B
    if c_face is None or gender_face_score is None or age_raw_face is None:
        return _body_result(gender_body_score, age_coarse)

    # ── Case A: Face dominance ──────────────────────────────────────────
    if c_face > FACE_DOMINANCE_THRESH:
        gender_score = gender_face_score
        age_range    = map_age_to_range(age_raw_face)
        return DecisionResult(
            gender       = _score_to_gender(gender_score),
            gender_score = gender_score,
            age_raw      = age_raw_face,
            age_range    = age_range,
            source       = "face",
        )

    # ── Case B: Body dominance ──────────────────────────────────────────
    if c_face < BODY_DOMINANCE_THRESH:
        return _body_result(gender_body_score, age_coarse)

    # ── Case C: Ambiguous ensemble ──────────────────────────────────────
    # w = C_face / (C_face + C_body)
    denom = c_face + c_body if (c_face + c_body) > 0 else 1.0
    w     = c_face / denom

    g_final   = w * gender_face_score + (1 - w) * gender_body_score

    # For age, blend the continuous face prediction with the midpoint of
    # the coarse PAR bucket, weighted by the same w.
    age_body_mid = _coarse_age_to_midpoint(age_coarse)
    age_final    = w * age_raw_face + (1 - w) * age_body_mid

    return DecisionResult(
        gender       = _score_to_gender(g_final),
        gender_score = g_final,
        age_raw      = age_final,
        age_range    = map_age_to_range(age_final),
        source       = "ensemble",
    )


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _score_to_gender(p_female: float) -> str:
    return "Female" if p_female >= 0.5 else "Male"


def _body_result(gender_body_score: float, age_coarse: str) -> DecisionResult:
    age_mid = _coarse_age_to_midpoint(age_coarse)
    return DecisionResult(
        gender       = _score_to_gender(gender_body_score),
        gender_score = gender_body_score,
        age_raw      = float(age_mid),
        age_range    = age_coarse,
        source       = "body",
    )


def _coarse_age_to_midpoint(label: str) -> float:
    """Map PAR coarse age label → numeric midpoint for blending."""
    mapping = {
        "Child":       6.0,
        "Teen":       15.5,
        "Young Adult": 24.5,
        "Adult":       38.0,
        "Middle-Aged": 53.0,
        "Senior":      70.0,
    }
    return mapping.get(label, 38.0)   # default to Adult midpoint