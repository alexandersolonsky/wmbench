"""wmbench — a benchmark framework for image watermarking algorithms.

Importing this package is deliberately lightweight: it pulls in no heavy
dependencies (torch, ffmpeg, …). Algorithm adapters import their backends lazily
so the framework runs with whatever subset is installed.
"""

from wmbench.core.interfaces import (
    Distortion,
    ExtractResult,
    ResultRow,
    Watermarker,
)
from wmbench.core.registry import (
    get_algorithm,
    get_distortion,
    list_algorithms,
    list_distortions,
    register_algorithm,
    register_distortion,
)

# Register built-in distortions on import. This is intentionally lightweight
# (Pillow only) — algorithm adapters are NOT imported here so that `import
# wmbench` never pulls in torch; the runner imports them on demand.
from wmbench import distortions as _distortions  # noqa: E402,F401

__version__ = "0.1.0"

__all__ = [
    "Distortion",
    "ExtractResult",
    "ResultRow",
    "Watermarker",
    "get_algorithm",
    "get_distortion",
    "list_algorithms",
    "list_distortions",
    "register_algorithm",
    "register_distortion",
]
