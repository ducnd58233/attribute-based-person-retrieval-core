"""Attribute prediction, ranking distances, and Track-2 retrieval evaluation.

Layout by concern:

- ``predict``: test-time augmentation views and gallery prediction.
- ``calibration``: per-attribute temperature fitting and application.
- ``distances``: the scorer registry and every ranking distance.
- ``rerank``: dual-softmax query normalisation.
- ``pipeline``: the one predict -> calibrate -> rank -> score sequence.
"""

from __future__ import annotations

from src.wacv.evaluation.calibration import (
    calibrate_probabilities,
    fit_per_attribute_temperatures,
)
from src.wacv.evaluation.distances import (
    clear_hard_distance,
    get_scorer,
    group_categorical_nll_distance,
    hybrid_distance,
    l1_distance,
    nll_distance,
    ranking_distance,
    register_scorer,
    specker_distance,
)
from src.wacv.evaluation.pipeline import RankingEvalData, compute_ranking_metrics
from src.wacv.evaluation.predict import (
    apply_tta_view,
    predict_gallery_embeddings,
    predict_probabilities,
)
from src.wacv.evaluation.rerank import dual_softmax_rerank

__all__ = [
    "RankingEvalData",
    "apply_tta_view",
    "calibrate_probabilities",
    "clear_hard_distance",
    "compute_ranking_metrics",
    "dual_softmax_rerank",
    "fit_per_attribute_temperatures",
    "get_scorer",
    "group_categorical_nll_distance",
    "hybrid_distance",
    "l1_distance",
    "nll_distance",
    "predict_gallery_embeddings",
    "predict_probabilities",
    "ranking_distance",
    "register_scorer",
    "specker_distance",
]
