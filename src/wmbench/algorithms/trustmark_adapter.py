"""Adapter for Adobe's TrustMark (https://github.com/adobe/trustmark).

CPU-only. The heavy ``import trustmark`` (which pulls in torch) happens inside
``__init__`` so this module is import-safe without the backend installed.

TrustMark works with a string secret + BCH error-correction. We drive it in
binary mode so our bit-list payload maps directly to the secret, and read the
payload width from the model's schema capacity.
"""

from __future__ import annotations

import os

from PIL import Image

from wmbench.core.interfaces import ExtractResult
from wmbench.core.registry import register_algorithm


@register_algorithm("trustmark")
class TrustMarkAdapter:
    name = "trustmark"

    def __init__(self, model_type: str = "Q", name: str = "trustmark",
                 ecc: str = "BCH_5") -> None:
        self.name = name
        self._ecc = ecc
        # TrustMark's default watermark strength (1.0) can be too weak to embed
        # recoverably in some high-resolution photos; override via env without a
        # code change (e.g. WMBENCH_TRUSTMARK_STRENGTH=2.0).
        self._strength = float(os.environ.get("WMBENCH_TRUSTMARK_STRENGTH", "1.0"))
        try:
            from trustmark import TrustMark
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ImportError(
                "trustmark is not installed. Install with `pip install trustmark` "
                "on the machine that runs the benchmark (CPU is fine)."
            ) from exc

        # ECC schema trades payload capacity for error-correction strength:
        # BCH_SUPER=40, BCH_5=61 (default), BCH_4=68, BCH_3=75 usable data bits;
        # ecc="NONE" disables ECC for the full 100 raw bits (least error-tolerant).
        # device='cpu' forces CPU; verbose off to keep benchmark logs clean.
        kwargs = dict(verbose=False, model_type=model_type, device="cpu")
        if ecc == "NONE":
            kwargs["use_ECC"] = False
        else:
            kwargs["encoding_type"] = getattr(TrustMark.Encoding, ecc)
        self._tm = TrustMark(**kwargs)
        # Usable data-bit capacity for the active ECC schema.
        try:
            self.payload_bits = int(self._tm.schemaCapacity())
        except Exception:  # pragma: no cover - older API
            self.payload_bits = 61

        self.settings = {
            "model_type": model_type,
            "device": "cpu",
            "ecc": ecc,
            "payload_bits": self.payload_bits,
            "wm_strength": self._strength,
        }

    def embed(self, image: Image.Image, message: list[int]) -> Image.Image:
        secret = "".join(str(int(b) & 1) for b in message[: self.payload_bits])
        return self._tm.encode(image.convert("RGB"), secret, MODE="binary",
                               WM_STRENGTH=self._strength)

    def extract(self, image: Image.Image) -> ExtractResult:
        secret, present, _schema = self._tm.decode(image.convert("RGB"), MODE="binary")
        bits = [int(c) for c in secret] if secret else []
        bits = bits[: self.payload_bits]
        bits += [0] * (self.payload_bits - len(bits))
        return ExtractResult(message=bits, present=bool(present),
                             confidence=1.0 if present else 0.0)


# Error-correction sweep across ALL four model types (Q/P/B/C). Each model gets
# the same BCH ladder (stronger ECC corrects more bit errors but carries fewer
# payload bits). Variant name = trustmark_<model>_<usable data bits>.
_ECC_BITS = (("40", "BCH_SUPER"), ("61", "BCH_5"), ("68", "BCH_4"),
             ("75", "BCH_3"), ("100", "NONE"))
for _model in ("Q", "P", "B", "C"):
    for _bits, _ecc in _ECC_BITS:
        _name = f"trustmark_{_model}_{_bits}"
        register_algorithm(_name)(
            lambda n=_name, m=_model, e=_ecc: TrustMarkAdapter(model_type=m, name=n, ecc=e))
