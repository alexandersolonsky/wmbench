"""Extraction-rate, confidence-interval and false-positive metrics."""

from __future__ import annotations

import math

from wmbench.core.interfaces import ExtractResult


def bit_accuracy(embedded: list[int], recovered: list[int]) -> float:
    """Fraction of bits recovered correctly.

    Compares over the overlapping prefix (adapters with different payload widths
    still produce a meaningful number). Empty input ⇒ 0.0.
    """
    n = min(len(embedded), len(recovered))
    if n == 0:
        return 0.0
    matches = sum(1 for a, b in zip(embedded[:n], recovered[:n]) if a == b)
    return matches / n


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion.

    Used to attach a confidence band to an aggregate success rate (default 95%).
    Returns (low, high), both clamped to [0, 1]. ``n == 0`` ⇒ (0.0, 1.0).
    """
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def false_positive(
    result: ExtractResult,
    registered_message: list[int] | None = None,
    threshold: float = 0.99,
) -> bool:
    """Did the detector wrongly flag a *non-watermarked* image?

    A false positive is either a hard ``present`` decision, or — when a set of
    registered payloads exists — recovering bits that match one of them at or
    above ``threshold`` (a spurious "successful" extraction).
    """
    if result.present:
        return True
    if registered_message is not None:
        if bit_accuracy(registered_message, result.message) >= threshold:
            return True
    return False
