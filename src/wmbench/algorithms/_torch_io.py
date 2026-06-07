"""Shared PIL <-> torch tensor helpers for the CPU adapters.

Imported lazily from inside adapter methods so that importing the adapter modules
never requires torch.
"""

from __future__ import annotations

from PIL import Image


def pil_to_tensor(image: "Image.Image"):
    """RGB PIL -> float tensor [1, 3, H, W] in [0, 1] on CPU."""
    import numpy as np
    import torch

    arr = np.asarray(image.convert("RGB"), dtype="float32") / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()


def tensor_to_pil(tensor) -> "Image.Image":
    """float tensor [1, 3, H, W] (or [3, H, W]) in [0, 1] -> RGB PIL."""
    import numpy as np
    import torch

    t = tensor.detach().to("cpu")
    if t.dim() == 4:
        t = t[0]
    t = t.clamp(0, 1).mul(255).round().byte()
    arr = t.permute(1, 2, 0).numpy().astype(np.uint8)
    return Image.fromarray(arr, "RGB")
