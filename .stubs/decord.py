"""Stub for `decord` (no Apple-Silicon wheel exists). videoseal hard-imports it
for VIDEO; PixelSeal image watermarking never touches it. Imports succeed;
actually using a video reader raises clearly."""

class _Unavailable:
    def __init__(self, *a, **k):
        raise RuntimeError("decord is unavailable on this platform; video features disabled")

VideoReader = _Unavailable
VideoLoader = _Unavailable
AudioReader = _Unavailable

def cpu(*a, **k):
    return None

def gpu(*a, **k):
    return None

class _Bridge:
    @staticmethod
    def set_bridge(*a, **k):
        return None

bridge = _Bridge()
