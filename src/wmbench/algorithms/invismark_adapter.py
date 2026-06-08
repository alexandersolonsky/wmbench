"""Adapter for Microsoft InvisMark (https://github.com/microsoft/InvisMark).

CPU-only. The repo's modules are imported from ``INVISMARK_REPO`` (default
``vendor/InvisMark``); the pretrained 100-bit checkpoint path comes from
``INVISMARK_CKPT``. Heavy imports happen inside ``__init__`` so importing this
module never requires the backend.

We build the ``Encoder`` + ``Extractor`` directly and replicate the repo's
``_encode``/``_decode`` (resize to the model resolution, embed, upscale the
residual back to full size); the ``Watermark`` training wrapper is skipped because
its ``__init__`` hardcodes ``.cuda()``.

The released model is **100-bit, no ECC**, which loses a few bits even on clean
high-res images (so it's not exact-ID-viable). Passing ``bch_data_bits`` wraps the
payload in BCH error-correction (see ``_ecc.BchCodec``) so K data bits survive
exactly.
"""

from __future__ import annotations

import os
import sys

from PIL import Image

from wmbench.core.interfaces import ExtractResult
from wmbench.core.registry import register_algorithm


@register_algorithm("invismark")
class InvisMarkAdapter:
    name = "invismark"

    def __init__(self, name: str = "invismark", bch_data_bits: int | None = None) -> None:
        self.name = name
        repo = os.environ.get("INVISMARK_REPO", os.path.abspath("vendor/InvisMark"))
        ckpt = os.environ.get(
            "INVISMARK_CKPT", os.path.abspath("checkpoints/invismark/paper.ckpt"))
        if repo not in sys.path:
            sys.path.insert(0, repo)  # so `import model, configs, ...` resolve

        try:
            import torch
            import torchvision.transforms as T
            import model  # InvisMark repo module
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ImportError(
                "InvisMark backend not available. Clone github.com/microsoft/InvisMark "
                "to INVISMARK_REPO and install its CPU deps (torch, torchvision, kornia, "
                "lpips, bchlib, focal-frequency-loss, torchmetrics, tensorboard)."
            ) from exc
        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                f"InvisMark checkpoint not found at {ckpt}. Download the pretrained "
                "weights from the OneDrive link in the InvisMark README and point "
                "INVISMARK_CKPT at the .ckpt file."
            )

        self._torch = torch
        self._T = T
        # weights_only=False: the ckpt pickles a ModelConfig object alongside the
        # tensors (trusted official InvisMark release).
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg = state["config"]
        self._encoder = model.Encoder(cfg)
        self._encoder.load_state_dict(state["encoder_state_dict"])
        self._encoder.eval()
        self._decoder = model.Extractor(cfg)
        self._decoder.load_state_dict(state["decoder_state_dict"])
        self._decoder.eval()

        self._resize = T.Resize(tuple(cfg.image_shape))  # to model resolution (256)
        self._norm = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # -> [-1, 1]
        ])
        self._raw_bits = int(getattr(cfg, "num_encoded_bits", 100))
        self._codec = None
        if bch_data_bits:
            from wmbench.algorithms._ecc import BchCodec
            self._codec = BchCodec(bch_data_bits, self._raw_bits)
            self.payload_bits = int(bch_data_bits)
        else:
            self.payload_bits = self._raw_bits
        self.settings = {
            "checkpoint": os.path.basename(ckpt),
            "device": "cpu",
            "payload_bits": self.payload_bits,
            "raw_bits": self._raw_bits,
            "ecc": f"BCH(t={self._codec.t})" if self._codec else "none",
            "resolution": int(cfg.image_shape[0]),
        }

    def embed(self, image: Image.Image, message: list[int]) -> Image.Image:
        torch = self._torch
        x = self._norm(image.convert("RGB")).unsqueeze(0)   # [1,3,H,W] in [-1,1]
        rx = self._resize(x)                                # [1,3,256,256]
        if self._codec:
            raw = self._codec.encode(message)
        else:
            raw = [int(b) & 1 for b in message[: self._raw_bits]]
            raw = raw + [0] * (self._raw_bits - len(raw))
        secret = torch.tensor([[float(b) for b in raw]], dtype=torch.float32)
        with torch.no_grad():
            enc = self._encoder(rx, secret)
            diff = self._T.Resize(x.shape[-2:])(enc - rx)   # residual upscaled to full res
            out = torch.clamp(x + diff, -1.0, 1.0)
        arr = ((out[0] * 0.5 + 0.5) * 255.0).round()
        return Image.fromarray(arr.permute(1, 2, 0).to(torch.uint8).cpu().numpy(), "RGB")

    def extract(self, image: Image.Image) -> ExtractResult:
        torch = self._torch
        x = self._norm(image.convert("RGB")).unsqueeze(0)
        rx = self._resize(x)
        with torch.no_grad():
            pred = self._decoder(rx)[0][: self._raw_bits].float()
        probs = pred if (float(pred.min()) >= 0.0 and float(pred.max()) <= 1.0) \
            else torch.sigmoid(pred)
        raw_bits = (probs > 0.5).long().tolist()
        if self._codec:
            bits, ok = self._codec.decode(raw_bits)
        else:
            bits, ok = raw_bits[: self.payload_bits], True
        conf = float((probs - 0.5).abs().mean() * 2.0)
        return ExtractResult(message=bits, present=ok and conf > 0.5,
                             confidence=max(0.0, min(1.0, conf)))


# BCH error-correction profiles on the 100-bit payload — 64 / 40 data bits with
# parity that corrects the extraction errors the raw no-ECC model leaves.
for _bits in (64, 40):
    _nm = f"invismark_bch{_bits}"
    register_algorithm(_nm)(
        lambda b=_bits, n=_nm: InvisMarkAdapter(name=n, bch_data_bits=b))
