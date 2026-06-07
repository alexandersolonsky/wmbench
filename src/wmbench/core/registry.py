"""Plugin registries for algorithms and distortions.

Both are keyed by name and store *factories* (zero-arg callables returning an
instance), so heavy backends are only constructed when a benchmark actually
selects them. Registration is via decorator; lookup raises a clear error listing
what is available.
"""

from __future__ import annotations

from typing import Callable

from wmbench.core.interfaces import Distortion, Watermarker

_ALGORITHMS: dict[str, Callable[[], Watermarker]] = {}
_DISTORTIONS: dict[str, Callable[[], Distortion]] = {}


def register_algorithm(name: str) -> Callable[[Callable[[], Watermarker]], Callable[[], Watermarker]]:
    """Decorator registering a zero-arg factory under ``name``."""

    def deco(factory: Callable[[], Watermarker]) -> Callable[[], Watermarker]:
        _ALGORITHMS[name] = factory
        return factory

    return deco


def register_distortion(name: str) -> Callable[[Callable[[], Distortion]], Callable[[], Distortion]]:
    def deco(factory: Callable[[], Distortion]) -> Callable[[], Distortion]:
        _DISTORTIONS[name] = factory
        return factory

    return deco


def get_algorithm(name: str) -> Watermarker:
    if name not in _ALGORITHMS:
        raise KeyError(f"Unknown algorithm {name!r}. Available: {sorted(_ALGORITHMS)}")
    return _ALGORITHMS[name]()


def get_distortion(name: str) -> Distortion:
    if name not in _DISTORTIONS:
        raise KeyError(f"Unknown distortion {name!r}. Available: {sorted(_DISTORTIONS)}")
    return _DISTORTIONS[name]()


def list_algorithms() -> list[str]:
    return sorted(_ALGORITHMS)


def list_distortions() -> list[str]:
    return sorted(_DISTORTIONS)
