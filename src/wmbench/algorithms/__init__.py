"""Watermarking algorithm adapters.

Importing this package registers the real adapters (TrustMark, PixelSeal, WAM).
Each adapter module is import-safe without its backend installed — the heavy
``import torch`` happens inside the adapter's ``__init__``, so registration here
never pulls in torch. Constructing an adapter whose backend is missing raises a
clear ``ImportError`` that the runner turns into a skip.
"""

from wmbench.algorithms import (  # noqa: F401  (registers adapters)
    pixelseal_adapter,
    trustmark_adapter,
    wam_adapter,
)
