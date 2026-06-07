"""Adapter for Meta's Watermark Anything / WAM ("WMA")
(https://github.com/facebookresearch/watermark-anything). CPU-only.

WAM is distributed as a git repo + a downloaded checkpoint rather than a pip
package, so this adapter is configured via two environment variables, read inside
``__init__`` (keeping the module import-safe):

  * ``WAM_REPO``           — path to the cloned watermark-anything repo (so that
                             ``notebooks.inference_utils`` is importable).
  * ``WAM_CHECKPOINT_DIR`` — dir containing ``params.json`` and ``checkpoint.pth``
                             (default: ``checkpoints``). Get weights with:
        wget https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth \\
             -O checkpoints/checkpoint.pth
"""

from __future__ import annotations

import os
import sys

from PIL import Image

from wmbench.algorithms._torch_io import pil_to_tensor, tensor_to_pil
from wmbench.core.interfaces import ExtractResult
from wmbench.core.registry import register_algorithm


@register_algorithm("wam_sw2.0")
class WamAdapter:
    name = "wam_sw2.0"
    payload_bits = 32  # WAM embeds a 32-bit message

    def __init__(self, scaling_w: float | None = None, name: str = "wam_sw2.0") -> None:
        self.name = name
        self._scaling_w_override = scaling_w
        repo = os.environ.get("WAM_REPO")
        repo = os.path.abspath(repo) if repo else None
        if repo and repo not in sys.path:
            sys.path.insert(0, repo)
        try:
            import torch
            from notebooks.inference_utils import load_model_from_checkpoint
            # WAM operates on ImageNet-normalized tensors; its embed output is
            # likewise normalized and must be un-normalized before saving.
            from watermark_anything.data.transforms import (
                normalize_img,
                unnormalize_img,
            )
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ImportError(
                "Watermark Anything (WAM) backend not importable. Clone "
                "github.com/facebookresearch/watermark-anything and set WAM_REPO "
                "to its path (CPU torch is fine)."
            ) from exc
        self._normalize = normalize_img
        self._unnormalize = unnormalize_img

        ckpt_dir = os.environ.get("WAM_CHECKPOINT_DIR", "checkpoints")
        json_path = os.path.abspath(os.path.join(ckpt_dir, "params.json"))
        ckpt_path = os.path.abspath(os.path.join(ckpt_dir, "checkpoint.pth"))
        if not (os.path.exists(json_path) and os.path.exists(ckpt_path)):
            raise ImportError(
                f"WAM checkpoint not found in {ckpt_dir!r} (need params.json + "
                "checkpoint.pth). Set WAM_CHECKPOINT_DIR and download the weights."
            )

        self._torch = torch
        # load_model_from_checkpoint reads params.json's configs/*.yaml relative to
        # cwd, so load from the repo root — then restore cwd. This lets the runner
        # be launched from anywhere (no need to cd into the WAM repo).
        prev_cwd = os.getcwd()
        try:
            if repo:
                os.chdir(repo)
            self._wam = load_model_from_checkpoint(json_path, ckpt_path).to("cpu").eval()
        finally:
            os.chdir(prev_cwd)

        # Optionally override the watermark strength (imgs_w = scaling_i*imgs +
        # scaling_w*pred); the checkpoint default is used otherwise.
        if self._scaling_w_override is not None:
            self._wam.scaling_w = self._scaling_w_override

        import json as _json
        try:
            params = _json.loads(open(json_path, encoding="utf-8").read())
        except Exception:  # pragma: no cover
            params = {}
        self.settings = {
            "checkpoint": os.path.basename(ckpt_path),
            "device": "cpu",
            "payload_bits": self.payload_bits,
            "img_size": params.get("img_size"),
            "scaling_w": float(self._wam.scaling_w),
        }

    def embed(self, image: Image.Image, message: list[int]) -> Image.Image:
        torch = self._torch
        x = self._normalize(pil_to_tensor(image))  # ImageNet-normalized
        # WAM expects a batched message of shape [B, K].
        msg = torch.tensor([[int(b) & 1 for b in message[: self.payload_bits]]],
                           dtype=torch.float32)
        with torch.no_grad():
            out = self._wam.embed(x, msg)
        # imgs_w is normalized; bring it back to [0, 1] before saving.
        imgs_w = self._unnormalize(out["imgs_w"]).clamp(0, 1)
        return tensor_to_pil(imgs_w)

    def extract(self, image: Image.Image) -> ExtractResult:
        torch = self._torch
        x = self._normalize(pil_to_tensor(image))
        with torch.no_grad():
            preds = self._wam.detect(x)["preds"]  # [1, 1 + nbits, H, W]
            mask = torch.sigmoid(preds[:, 0, :, :])  # per-pixel presence prob
            bit_logits = preds[:, 1:, :, :]          # [1, nbits, H, W]
            # Recover one global message: average each bit's logit over the image
            # weighted by the detected mask, then threshold at 0.
            w = mask.unsqueeze(1)                    # [1, 1, H, W]
            agg = (bit_logits * w).sum(dim=(2, 3)) / (w.sum(dim=(2, 3)) + 1e-8)
            pred_message = (agg[0] > 0).long()       # [nbits]
        bits = pred_message.tolist()[: self.payload_bits]
        confidence = float(mask.mean())
        return ExtractResult(message=bits, present=confidence > 0.5,
                             confidence=confidence)


# A second WAM variant at a lower watermark strength (the checkpoint default is
# 2.0). Same model, weaker embedding — trades robustness for image quality.
register_algorithm("wam_sw1.0")(lambda: WamAdapter(scaling_w=1.0, name="wam_sw1.0"))
