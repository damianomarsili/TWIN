#!/usr/bin/env python3
"""
Simple binary outcome reward built around <answer>...</answer> tags.

The model is expected to produce a free-form explanation followed by the final
decision wrapped inside <answer></answer> tags (case-insensitive). We normalize
both the predicted answer and the provided ground truth and return 1.0 if they
match, 0.0 otherwise.
"""

from __future__ import annotations

import re
import string
from typing import Iterable, Sequence

ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
PUNCT_TRANSLATOR = str.maketrans("", "", string.punctuation)


def _extract_answer(text: str) -> str | None:
    """Grab the last <answer>...</answer> span (case-insensitive)."""
    if not text:
        return None

    matches = ANSWER_PATTERN.findall(text)
    if not matches:
        return None

    # Use the last match as the final answer signal.
    return matches[-1].strip()


def _normalize_label(label: str | None) -> str | None:
    """Lower-case, strip whitespace, and remove surrounding punctuation."""
    if label is None:
        return None

    normalized = label.strip().lower()
    normalized = normalized.translate(PUNCT_TRANSLATOR)
    normalized = normalized.strip()
    return normalized or None


def compute_score(solution_str: str, ground_truth: str) -> float:
    """
    Return 1.0 when the predicted answer (inside <answer></answer>) matches the
    provided ground truth after simple normalization, else 0.0.
    """

    predicted_raw = _extract_answer(solution_str)
    predicted = _normalize_label(predicted_raw)
    target = _normalize_label(ground_truth)

    if predicted is None or target is None:
        return 0.0

    return 1.0 if predicted == target else 0.0


def compute_score_batched(
    data_sources: Sequence[str],
    solution_strs: Sequence[str],
    ground_truths: Sequence[str],
    extra_infos: Sequence[dict] | Iterable[dict],
) -> list[float]:

    scores: list[float] = []
    for solution_str, ground_truth in zip(solution_strs, ground_truths, strict=True):
        scores.append(compute_score(solution_str, ground_truth))
    return scores


__all__ = ["compute_score", "compute_score_batched"]

