"""Jersey crop super-resolution — 4x upscale before OCR."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

LOGGER = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


if _TORCH_AVAILABLE:
    class SRBlock(nn.Module):
        def __init__(self, c=64):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(c, c, 3, 1, 1), nn.BatchNorm2d(c), nn.PReLU(),
                nn.Conv2d(c, c, 3, 1, 1), nn.BatchNorm2d(c),
            )

        def forward(self, x):
            return x + self.block(x)

    class JerseySR(nn.Module):
        def __init__(self):
            super().__init__()
            self.entry = nn.Sequential(nn.Conv2d(3, 64, 9, 1, 4), nn.PReLU())
            self.res = nn.Sequential(*[SRBlock(64) for _ in range(8)])
            self.mid = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64))
            self.up = nn.Sequential(
                nn.Conv2d(64, 256, 3, 1, 1), nn.PixelShuffle(2), nn.PReLU(),
                nn.Conv2d(64, 256, 3, 1, 1), nn.PixelShuffle(2), nn.PReLU(),
            )
            self.out = nn.Conv2d(64, 3, 9, 1, 4)

        def forward(self, x):
            e = self.entry(x)
            r = self.mid(self.res(e))
            return torch.clamp(self.out(self.up(e + r)), 0, 1)


class JerseyUpscaler:
    def __init__(self, model_path: Path):
        self._model = None
        self._path = model_path

    def load(self) -> bool:
        if not _TORCH_AVAILABLE:
            LOGGER.info("PyTorch not available — SR disabled")
            return False
        if not self._path.exists():
            LOGGER.info("jersey_upscaler_v5.pth not found — skipping SR")
            return False
        try:
            self._model = JerseySR()
            state = torch.load(str(self._path), map_location="cpu")
            # Handle both full checkpoint and state_dict-only saves
            if "model_state" in state:
                self._model.load_state_dict(state["model_state"])
            else:
                self._model.load_state_dict(state)
            self._model.eval()
            if torch.cuda.is_available():
                self._model = self._model.cuda()
            LOGGER.info("Jersey SR model loaded from %s", self._path)
            return True
        except Exception as exc:
            LOGGER.warning("Jersey SR load failed: %s", exc)
            self._model = None
            return False

    def upscale(self, crop: np.ndarray) -> np.ndarray:
        """Upscale a BGR jersey crop 4x. Returns upscaled BGR numpy array.
        If model not loaded, returns bicubic 4x resize as fallback."""
        import cv2
        if self._model is None:
            h, w = crop.shape[:2]
            return cv2.resize(crop, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
        try:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
            if torch.cuda.is_available():
                tensor = tensor.cuda()
            with torch.no_grad():
                sr = self._model(tensor)
            sr_np = sr.squeeze(0).permute(1, 2, 0).cpu().numpy()
            sr_bgr = cv2.cvtColor(
                (sr_np * 255).clip(0, 255).astype(np.uint8),
                cv2.COLOR_RGB2BGR,
            )
            return sr_bgr
        except Exception as exc:
            LOGGER.warning("SR inference failed: %s — using bicubic fallback", exc)
            h, w = crop.shape[:2]
            return cv2.resize(crop, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)

    def unload(self):
        self._model = None
