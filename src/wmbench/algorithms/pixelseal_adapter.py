"""Adapter for Meta's PixelSeal, loaded via the ``videoseal`` library
(https://github.com/facebookresearch/videoseal). CPU-only.

The heavy ``import videoseal`` happens inside ``__init__`` so this module is
import-safe without the backend. We pass an explicit message to ``embed`` so the
recovered bits can be scored against a known payload.

Optionally wraps the 256-bit raw payload in a BCH error-correcting code (the same
scheme TrustMark uses, via ``bchlib``): ``bch_data_bits`` data bits are encoded
into the 256-bit codeword, and extraction BCH-decodes them back, silently fixing
up to ``t`` bit-errors. Fewer data bits => more parity => more robust exact-ID.
"""

from __future__ import annotations

from PIL import Image

from wmbench.algorithms._torch_io import pil_to_tensor, tensor_to_pil
from wmbench.core.interfaces import ExtractResult
from wmbench.core.registry import register_algorithm


def _bits_to_bytes(bits: list[int]) -> bytes:
    out = bytearray((len(bits) + 7) // 8)
    for i, v in enumerate(bits):
        if int(v) & 1:
            out[i // 8] |= 1 << (7 - i % 8)
    return bytes(out)


def _bytes_to_bits(data: bytes, n: int) -> list[int]:
    return [(data[i // 8] >> (7 - i % 8)) & 1 for i in range(n)]


@register_algorithm("pixelseal_sw0.2")
class PixelSealAdapter:
    name = "pixelseal_sw0.2"

    def __init__(self, model_name: str = "pixelseal", scaling_mult: float = 1.0,
                 name: str = "pixelseal_sw0.2", bch_data_bits: int | None = None) -> None:
        self.name = name
        self._scaling_mult = scaling_mult
        try:
            import torch
            import videoseal
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ImportError(
                "videoseal is not installed. Install it on the benchmark machine "
                "(see github.com/facebookresearch/videoseal); CPU torch is fine."
            ) from exc

        self._torch = torch
        self._model = videoseal.load(model_name)
        self._model.eval()
        try:
            self._model.to("cpu")
        except Exception:  # pragma: no cover - some builds pin device internally
            pass
        # The model's raw payload width (256); BCH data bits are carried inside it.
        self._raw_bits = int(getattr(self._model, "nbits", 256))

        # Optional BCH ECC layer. Pick the largest t (most correction) whose
        # codeword (data + ecc bytes) still fits the 256-bit raw payload.
        self._bch = None
        self._bch_t = None
        if bch_data_bits:
            import bchlib
            dbytes = (bch_data_bits + 7) // 8
            for t in range(1, 40):
                try:
                    cand = bchlib.BCH(t, m=8)
                except Exception:  # pragma: no cover - invalid t/m
                    continue
                if (dbytes + cand.ecc_bytes) * 8 <= self._raw_bits:
                    self._bch, self._bch_t = cand, t
            if self._bch is None:
                raise ValueError(
                    f"no BCH(m=8) fits {bch_data_bits} data bits in {self._raw_bits}")
            self._bch_dbytes = dbytes
            self.payload_bits = int(bch_data_bits)
        else:
            self.payload_bits = self._raw_bits

        base = None
        try:
            base = float(self._model.blender.scaling_w)
        except Exception:  # pragma: no cover - attribute path may vary
            pass
        scaling_w = base
        if base is not None and scaling_mult != 1.0:
            scaling_w = base * scaling_mult
            try:
                self._model.blender.scaling_w = scaling_w
            except Exception:  # pragma: no cover - attribute path may vary
                pass
        self.settings = {
            "model": model_name,
            "device": "cpu",
            "payload_bits": self.payload_bits,
            "raw_bits": self._raw_bits,
            "ecc": f"BCH(t={self._bch_t})" if self._bch else "none",
            "scaling_w": scaling_w,
            "scaling_mult": scaling_mult,
        }

    def _payload_to_raw(self, message: list[int]) -> list[int]:
        """K-bit message -> the 256-bit codeword the model actually embeds."""
        if self._bch is None:
            bits = [int(b) & 1 for b in message[: self._raw_bits]]
            return bits + [0] * (self._raw_bits - len(bits))
        data = _bits_to_bytes([int(b) & 1 for b in message[: self.payload_bits]])
        data = (data + bytes(self._bch_dbytes))[: self._bch_dbytes]  # pad to dbytes
        ecc = bytes(self._bch.encode(data))
        code = (_bytes_to_bits(data, self._bch_dbytes * 8)
                + _bytes_to_bits(ecc, self._bch.ecc_bytes * 8))
        return code + [0] * (self._raw_bits - len(code))

    def _raw_to_payload(self, raw_bits: list[int]) -> tuple[list[int], bool]:
        """Recovered 256 raw bits -> (K data bits, decoded_ok)."""
        if self._bch is None:
            return raw_bits[: self.payload_bits], True
        db, eb = self._bch_dbytes, self._bch.ecc_bytes
        data = bytearray(_bits_to_bytes(raw_bits[: db * 8])[:db])
        ecc = bytearray(_bits_to_bytes(raw_bits[db * 8: (db + eb) * 8])[:eb])
        try:
            nerr = self._bch.decode(data, ecc)
            self._bch.correct(data, ecc)
            ok = nerr is not None and nerr >= 0
        except Exception:  # pragma: no cover - uncorrectable
            ok = False
        return _bytes_to_bits(bytes(data), self.payload_bits), ok

    def embed(self, image: Image.Image, message: list[int]) -> Image.Image:
        torch = self._torch
        raw = self._payload_to_raw(message)
        x = pil_to_tensor(image)
        msg = torch.tensor([[float(b) for b in raw]], dtype=torch.float32)
        with torch.no_grad():
            out = self._model.embed(x, msgs=msg, is_video=False)
        return tensor_to_pil(out["imgs_w"])

    def extract(self, image: Image.Image) -> ExtractResult:
        torch = self._torch
        x = pil_to_tensor(image)
        with torch.no_grad():
            det = self._model.detect(x, is_video=False)
        preds = det["preds"][0]  # [1 + nbits]
        bit_logits = preds[1: 1 + self._raw_bits]
        raw_bits = (bit_logits > 0).long().tolist()
        # PixelSeal's detection bit (preds[0]) is not a reliable presence flag, so
        # confidence comes from the bit-decoding margin.
        margin = float(torch.sigmoid(bit_logits.abs()).mean())  # [0.5, 1]
        confidence = max(0.0, (margin - 0.5) * 2.0)             # -> [0, 1]
        bits, ok = self._raw_to_payload(raw_bits)
        return ExtractResult(message=bits, present=ok and confidence > 0.5,
                             confidence=confidence)


# Stronger-embed variant: 2x the default blend strength (scaling_w 0.2 -> 0.4) —
# more robust to distortion at the cost of perceptual quality.
register_algorithm("pixelseal_sw0.4")(
    lambda: PixelSealAdapter(scaling_mult=2.0, name="pixelseal_sw0.4"))

# BCH error-correction profiles on the default-strength model: 100 / 64 / 40 data
# bits carried inside the 256-bit payload (the rest is parity that fixes errors).
for _bits in (100, 64, 40):
    _nm = f"pixelseal_sw0.2_bch{_bits}"
    register_algorithm(_nm)(
        lambda b=_bits, n=_nm: PixelSealAdapter(name=n, bch_data_bits=b))
