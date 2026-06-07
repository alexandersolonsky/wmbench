"""Adapter for Meta's PixelSeal, loaded via the ``videoseal`` library
(https://github.com/facebookresearch/videoseal). CPU-only.

The heavy ``import videoseal`` happens inside ``__init__`` so this module is
import-safe without the backend. We pass an explicit message to ``embed`` so the
recovered bits can be scored against a known payload.
"""

from __future__ import annotations

from PIL import Image

from wmbench.algorithms._torch_io import pil_to_tensor, tensor_to_pil
from wmbench.core.interfaces import ExtractResult
from wmbench.core.registry import register_algorithm


@register_algorithm("pixelseal_sw0.2")
class PixelSealAdapter:
    name = "pixelseal_sw0.2"

    def __init__(self, model_name: str = "pixelseal", scaling_mult: float = 1.0,
                 name: str = "pixelseal_sw0.2") -> None:
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
        self.payload_bits = int(getattr(self._model, "nbits", 256))
        base = None
        try:
            base = float(self._model.blender.scaling_w)
        except Exception:  # pragma: no cover - attribute path may vary
            pass
        # Stronger embed = more robust, lower quality. scaling_mult scales the
        # model's default blend strength (e.g. 2.0 => 2x the watermark).
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
            "scaling_w": scaling_w,
            "scaling_mult": scaling_mult,
        }

    def embed(self, image: Image.Image, message: list[int]) -> Image.Image:
        torch = self._torch
        x = pil_to_tensor(image)
        msg = torch.tensor([[int(b) & 1 for b in message[: self.payload_bits]]],
                           dtype=torch.float32)
        with torch.no_grad():
            out = self._model.embed(x, msgs=msg, is_video=False)
        return tensor_to_pil(out["imgs_w"])

    def extract(self, image: Image.Image) -> ExtractResult:
        torch = self._torch
        x = pil_to_tensor(image)
        with torch.no_grad():
            det = self._model.detect(x, is_video=False)
        preds = det["preds"][0]  # [1 + nbits]
        bit_logits = preds[1 : 1 + self.payload_bits]
        bits = (bit_logits > 0).long().tolist()
        # PixelSeal's detection bit (preds[0]) is not a reliable presence flag, so
        # we derive confidence from the bit-decoding margin: confident logits ->
        # ~1, near-zero logits (no watermark) -> ~0.
        margin = float(torch.sigmoid(bit_logits.abs()).mean())  # [0.5, 1]
        confidence = max(0.0, (margin - 0.5) * 2.0)             # -> [0, 1]
        return ExtractResult(message=bits, present=confidence > 0.5,
                             confidence=confidence)


# Stronger-embed variant: 2x the default blend strength (scaling_w 0.2 -> 0.4) —
# more robust to distortion at the cost of perceptual quality.
register_algorithm("pixelseal_sw0.4")(
    lambda: PixelSealAdapter(scaling_mult=2.0, name="pixelseal_sw0.4"))
