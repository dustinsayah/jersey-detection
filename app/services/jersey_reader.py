from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.services.detection_runtime import PipelineSettings

LOGGER = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_VALID_NUMBER_RE = re.compile(r"^\d{1,2}$")

_GRAD_VITB_CHECKPOINT_URL = (
    "https://drive.google.com/file/d/16npJY-gyboRE_HNTQI1dC_fIxh3oxa0S/view?usp=drive_link"
)
_KOSHKINA_LEGIBILITY_URL = (
    "https://drive.google.com/uc?id=18HAuZbge3z8TSfRiX_FzsnKgiBs-RRNw"
)
_KOSHKINA_PARSEQ_URL = (
    "https://drive.google.com/uc?id=1uRln22tlhneVt3P6MePmVxBWSLMsL3bm"
)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _resolve_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return (_PROJECT_ROOT / path).resolve()


def _ensure_sys_path(path: Path) -> None:
    raw = str(path.resolve())
    if raw not in sys.path:
        sys.path.insert(0, raw)


def _load_module_from_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _to_float(value) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    if isinstance(value, (list, tuple)) and value:
        return float(value[0])
    return float(value)


def _sequence_confidence(probabilities) -> float:
    if probabilities is None:
        return 0.0
    if hasattr(probabilities, "numel"):
        if probabilities.numel() == 0:
            return 0.0
        return float(probabilities.prod().item())
    if isinstance(probabilities, (list, tuple)):
        if not probabilities:
            return 0.0
        score = 1.0
        for item in probabilities:
            score *= _to_float(item)
        return float(score)
    return _to_float(probabilities)


def _download_if_missing(
    path: Path,
    *,
    url: str,
    enabled: bool,
) -> Path | None:
    if path.exists():
        return path
    if not enabled:
        return None
    try:
        import gdown
    except Exception as error:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Auto-download requested, but gdown is not installed."
        ) from error

    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_suffix(f"{path.suffix}.part")
    LOGGER.info("Downloading public checkpoint to %s", path)
    downloaded = gdown.download(
        url,
        str(partial_path),
        quiet=False,
        fuzzy=True,
        resume=partial_path.exists(),
    )
    if not downloaded or not partial_path.exists():
        raise RuntimeError(f"Failed to download public checkpoint from {url}")
    partial_path.replace(path)
    return path


def _redownload_public_asset(
    path: Path,
    *,
    url: str,
    enabled: bool,
) -> Path | None:
    path.unlink(missing_ok=True)
    partial_path = path.with_suffix(f"{path.suffix}.part")
    partial_path.unlink(missing_ok=True)
    return _download_if_missing(path, url=url, enabled=enabled)


def _load_trusted_lightning_checkpoint(
    load_from_checkpoint,
    checkpoint_spec: str,
    *,
    torch_module,
    device: str,
    **model_kwargs,
):
    original_torch_load = torch_module.load

    def _trusted_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch_module.load = _trusted_torch_load
    try:
        return load_from_checkpoint(
            checkpoint_spec,
            **model_kwargs,
        ).eval().to(device)
    finally:
        torch_module.load = original_torch_load


def _device_from_preference(preference: str) -> str:
    normalized = str(preference or "auto").strip().lower()
    if normalized and normalized != "auto":
        return normalized
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


def _torso_crop(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr.size == 0:
        return image_bgr
    height, width = image_bgr.shape[:2]
    top = int(height * 0.15)
    bottom = int(height * 0.70)
    left = int(width * 0.10)
    right = int(width * 0.90)
    if bottom <= top or right <= left:
        return image_bgr
    return image_bgr[top:bottom, left:right]


def _center_crop(image_bgr: np.ndarray, *, width_ratio: float, height_ratio: float) -> np.ndarray:
    if image_bgr.size == 0:
        return image_bgr
    height, width = image_bgr.shape[:2]
    crop_h = max(1, int(height * height_ratio))
    crop_w = max(1, int(width * width_ratio))
    y1 = max(0, (height - crop_h) // 2)
    x1 = max(0, (width - crop_w) // 2)
    y2 = min(height, y1 + crop_h)
    x2 = min(width, x1 + crop_w)
    return image_bgr[y1:y2, x1:x2]


def _clahe_enhance(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr.size == 0:
        return image_bgr
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(l_channel)
    merged = cv2.merge((enhanced, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def _fallback_variants(player_crop_bgr: np.ndarray) -> list[tuple[str, np.ndarray]]:
    torso = _torso_crop(player_crop_bgr)
    centered = _center_crop(torso if torso.size else player_crop_bgr, width_ratio=0.80, height_ratio=0.90)
    variants: list[tuple[str, np.ndarray]] = []
    for name, image in (
        ("torso", torso),
        ("torso_center", centered),
        ("torso_clahe", _clahe_enhance(torso if torso.size else player_crop_bgr)),
        ("full_crop", player_crop_bgr),
    ):
        if image.size == 0:
            continue
        variants.append((name, image))
    return variants


@dataclass(frozen=True)
class CropReadResult:
    number: int | None
    confidence: float
    uncertainty: float
    readable: bool
    backend: str


class PublicCheckpointGradReader:
    def __init__(self, settings: PipelineSettings, *, device: str) -> None:
        self.settings = settings
        self.device = device
        self.available = False
        self.error: str | None = None
        self._torch = None
        self._model = None
        self._transform = None
        self._load()

    def _load(self) -> None:
        repo_root = _resolve_path(self.settings.public_reader_repo_dir)
        config_path = _resolve_path(self.settings.grad_config_path)
        checkpoint_path = _resolve_path(self.settings.grad_checkpoint_path)

        try:
            if not repo_root.exists():
                raise FileNotFoundError(
                    f"uncertainty-jnr repo not found at {repo_root}"
                )
            if not config_path.exists():
                raise FileNotFoundError(f"Grad config not found at {config_path}")
            if not checkpoint_path.exists():
                checkpoint = _download_if_missing(
                    checkpoint_path,
                    url=_GRAD_VITB_CHECKPOINT_URL,
                    enabled=self.settings.public_reader_auto_download,
                )
                if checkpoint is None:
                    raise FileNotFoundError(
                        f"Grad checkpoint not found at {checkpoint_path}"
                    )

            _ensure_sys_path(repo_root)
            _ensure_sys_path(repo_root / "src")
            config_module = _load_module_from_file(
                "_uncertainty_jnr_external_config",
                repo_root / "config.py",
            )
            torch = importlib.import_module("torch")
            augmentation_module = importlib.import_module("uncertainty_jnr.augmentation")
            model_module = importlib.import_module("uncertainty_jnr.model")
            utils_module = importlib.import_module("uncertainty_jnr.utils")

            os.environ.setdefault("OCR_DIR", str(repo_root))
            config = config_module.Config.from_yaml(config_path)
            model = model_module.TimmOCRModel(
                model_name=config.model.model_name,
                pretrained=False,
                classifier_type=config.model.classifier_type,
                embedding_type=config.model.embedding_type,
                per_digit_bias=config.model.per_digit_bias,
                uncertainty_head=config.model.uncertainty_head,
            )
            try:
                utils_module.load_checkpoint(
                    model,
                    checkpoint_path,
                    torch.device(self.device),
                    strict=False,
                )
            except Exception as load_error:
                if not self.settings.public_reader_auto_download:
                    raise
                LOGGER.warning(
                    "Grad checkpoint at %s appears invalid; retrying download: %s",
                    checkpoint_path,
                    load_error,
                )
                checkpoint = _redownload_public_asset(
                    checkpoint_path,
                    url=_GRAD_VITB_CHECKPOINT_URL,
                    enabled=True,
                )
                if checkpoint is None:
                    raise
                utils_module.load_checkpoint(
                    model,
                    checkpoint,
                    torch.device(self.device),
                    strict=False,
                )

            self._torch = torch
            self._model = model.to(self.device).eval()
            self._transform = augmentation_module.get_val_transforms(
                target_size=config.data.target_size
            )
            self.available = True
            LOGGER.info("Loaded Grad public checkpoint reader from %s", checkpoint_path)
        except Exception as error:  # pragma: no cover - depends on optional assets
            self.error = str(error)
            LOGGER.warning("Grad public checkpoint reader unavailable: %s", error)

    def read_crop(self, crop_bgr: np.ndarray) -> CropReadResult:
        if not self.available or crop_bgr.size == 0:
            return CropReadResult(
                number=None,
                confidence=0.0,
                uncertainty=1.0,
                readable=False,
                backend="grad_unavailable",
            )
        assert self._torch is not None
        assert self._model is not None
        assert self._transform is not None

        image_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        transformed = self._transform(image=image_rgb)["image"]
        tensor = self._torch.from_numpy(transformed.transpose(2, 0, 1))
        tensor = ((tensor.float() / 127.5) - 1.0).unsqueeze(0).to(self.device)

        with self._torch.inference_mode():
            if str(self.device).startswith("cuda"):
                with self._torch.autocast(device_type="cuda", dtype=self._torch.float16):
                    output = self._model(tensor)
            else:
                output = self._model(tensor)

        pred_scores, pred_numbers = self._torch.max(output.number_probs, dim=1)
        confidence = float(pred_scores[0].item())
        uncertainty = float(output.uncertainty.reshape(-1)[0].item())
        number = int(pred_numbers[0].item())
        readable = (
            confidence >= self.settings.grad_min_confidence
            and uncertainty <= self.settings.grad_max_uncertainty
        )
        return CropReadResult(
            number=number,
            confidence=confidence,
            uncertainty=uncertainty,
            readable=readable,
            backend="grad_vitb",
        )


class KoshkinaLegibilityGate:
    def __init__(self, settings: PipelineSettings, *, device: str) -> None:
        self.settings = settings
        self.device = device
        self.available = False
        self.error: str | None = None
        self._torch = None
        self._Image = None
        self._model = None
        self._transform = None
        self._load()

    def _load(self) -> None:
        weight_path = _resolve_path(self.settings.legibility_model_path)
        try:
            if not weight_path.exists():
                checkpoint = _download_if_missing(
                    weight_path,
                    url=_KOSHKINA_LEGIBILITY_URL,
                    enabled=self.settings.public_reader_auto_download,
                )
                if checkpoint is None:
                    raise FileNotFoundError(
                        f"Legibility checkpoint not found at {weight_path}"
                    )

            torch = importlib.import_module("torch")
            torchvision = importlib.import_module("torchvision")
            Image = importlib.import_module("PIL.Image")

            class _LegibilityClassifier34(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.model_ft = torchvision.models.resnet34(pretrained=False)
                    num_ftrs = self.model_ft.fc.in_features
                    self.model_ft.fc = torch.nn.Linear(num_ftrs, 1)

                def forward(self, x):
                    return torch.sigmoid(self.model_ft(x))

            try:
                state_dict = torch.load(weight_path, map_location=torch.device(self.device))
            except Exception as load_error:
                if not self.settings.public_reader_auto_download:
                    raise
                LOGGER.warning(
                    "Legibility checkpoint at %s appears invalid; retrying download: %s",
                    weight_path,
                    load_error,
                )
                checkpoint = _redownload_public_asset(
                    weight_path,
                    url=_KOSHKINA_LEGIBILITY_URL,
                    enabled=True,
                )
                if checkpoint is None:
                    raise
                state_dict = torch.load(checkpoint, map_location=torch.device(self.device))
            model = _LegibilityClassifier34()
            if hasattr(state_dict, "_metadata"):
                del state_dict._metadata
            model.load_state_dict(state_dict)
            transforms = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize((256, 256)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )

            self._torch = torch
            self._Image = Image
            self._model = model.to(self.device).eval()
            self._transform = transforms
            self.available = True
            LOGGER.info("Loaded Koshkina legibility weights from %s", weight_path)
        except Exception as error:  # pragma: no cover - depends on optional assets
            self.error = str(error)
            LOGGER.warning("Koshkina legibility gate unavailable: %s", error)

    def predict(self, crop_bgr: np.ndarray) -> float:
        if not self.available or crop_bgr.size == 0:
            return 1.0
        assert self._torch is not None
        assert self._Image is not None
        assert self._model is not None
        assert self._transform is not None

        image_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        image = self._Image.fromarray(image_rgb)
        tensor = self._transform(image).unsqueeze(0).to(self.device)
        with self._torch.inference_mode():
            score = self._model(tensor).reshape(-1)[0]
        return _clamp01(float(score.item()))


class ParseqFallbackReader:
    def __init__(self, settings: PipelineSettings, *, device: str) -> None:
        self.settings = settings
        self.device = device
        self.available = False
        self.error: str | None = None
        self._torch = None
        self._Image = None
        self._model = None
        self._transform = None
        self._load()

    def _load(self) -> None:
        parseq_repo_root = _resolve_path(self.settings.parseq_repo_dir)
        checkpoint_path = _resolve_path(self.settings.parseq_checkpoint_path)
        try:
            if not parseq_repo_root.exists():
                raise FileNotFoundError(
                    f"PARSeq repo not found at {parseq_repo_root}"
                )

            checkpoint_spec = str(checkpoint_path)
            if not checkpoint_path.exists():
                checkpoint = _download_if_missing(
                    checkpoint_path,
                    url=_KOSHKINA_PARSEQ_URL,
                    enabled=self.settings.public_reader_auto_download,
                )
                if checkpoint is None:
                    if not self.settings.parseq_use_pretrained_fallback:
                        raise FileNotFoundError(
                            f"PARSeq checkpoint not found at {checkpoint_path}"
                        )
                    checkpoint_spec = "pretrained=parseq"

            _ensure_sys_path(parseq_repo_root)
            torch = importlib.import_module("torch")
            SceneTextDataModule = importlib.import_module(
                "strhub.data.module"
            ).SceneTextDataModule
            load_from_checkpoint = importlib.import_module(
                "strhub.models.utils"
            ).load_from_checkpoint
            Image = importlib.import_module("PIL.Image")

            try:
                model = _load_trusted_lightning_checkpoint(
                    load_from_checkpoint,
                    checkpoint_spec,
                    torch_module=torch,
                    device=self.device,
                )
            except Exception as load_error:
                if (
                    checkpoint_spec.startswith("pretrained=")
                    or not self.settings.public_reader_auto_download
                    or not checkpoint_path.exists()
                ):
                    raise
                LOGGER.warning(
                    "PARSeq checkpoint at %s appears invalid; retrying download: %s",
                    checkpoint_path,
                    load_error,
                )
                checkpoint = _redownload_public_asset(
                    checkpoint_path,
                    url=_KOSHKINA_PARSEQ_URL,
                    enabled=True,
                )
                if checkpoint is None:
                    raise
                model = _load_trusted_lightning_checkpoint(
                    load_from_checkpoint,
                    str(checkpoint),
                    torch_module=torch,
                    device=self.device,
                )

            self._torch = torch
            self._Image = Image
            self._model = model
            self._transform = SceneTextDataModule.get_transform(model.hparams.img_size)
            self.available = True
            LOGGER.info("Loaded PARSeq fallback reader from %s", checkpoint_spec)
        except Exception as error:  # pragma: no cover - depends on optional assets
            self.error = str(error)
            LOGGER.warning("PARSeq fallback reader unavailable: %s", error)

    def read_crop(self, crop_bgr: np.ndarray) -> CropReadResult:
        if not self.available or crop_bgr.size == 0:
            return CropReadResult(
                number=None,
                confidence=0.0,
                uncertainty=1.0,
                readable=False,
                backend="parseq_unavailable",
            )
        assert self._torch is not None
        assert self._Image is not None
        assert self._model is not None
        assert self._transform is not None

        best = CropReadResult(
            number=None,
            confidence=0.0,
            uncertainty=1.0,
            readable=False,
            backend="parseq",
        )
        for variant_name, variant in _fallback_variants(crop_bgr):
            image_rgb = cv2.cvtColor(variant, cv2.COLOR_BGR2RGB)
            image = self._Image.fromarray(image_rgb)
            tensor = self._transform(image).unsqueeze(0).to(self.device)
            with self._torch.inference_mode():
                logits = self._model(tensor, max_length=2)
            pred = logits.softmax(-1)
            labels, confidences = self._model.tokenizer.decode(pred)
            label = str(labels[0]).strip()
            if not _VALID_NUMBER_RE.fullmatch(label):
                continue
            confidence = _sequence_confidence(confidences[0])
            result = CropReadResult(
                number=int(label),
                confidence=confidence,
                uncertainty=1.0 - confidence,
                readable=confidence >= self.settings.parseq_min_confidence,
                backend=f"parseq:{variant_name}",
            )
            if result.confidence > best.confidence:
                best = result
        return best


class PublicCheckpointReaderEnsemble:
    def __init__(self, settings: PipelineSettings, *, device: str | None = None) -> None:
        self.settings = settings
        self.device = _device_from_preference(device or settings.yolo_device)
        self.primary = PublicCheckpointGradReader(settings, device=self.device)
        self.legibility = KoshkinaLegibilityGate(settings, device=self.device)
        self.fallback = ParseqFallbackReader(settings, device=self.device)

    def is_available(self) -> bool:
        return self.primary.available or self.fallback.available

    def read_crop(self, crop_bgr: np.ndarray) -> CropReadResult:
        primary = self.primary.read_crop(crop_bgr)
        if (
            primary.number is not None
            and primary.readable
            and primary.confidence >= self.settings.grad_min_confidence
            and primary.uncertainty <= self.settings.grad_max_uncertainty
        ):
            return primary

        legibility_score = self.legibility.predict(crop_bgr)
        if legibility_score < self.settings.legibility_threshold:
            return CropReadResult(
                number=None,
                confidence=0.0,
                uncertainty=max(primary.uncertainty, 1.0 - legibility_score),
                readable=False,
                backend="legibility_gate",
            )

        fallback = self.fallback.read_crop(crop_bgr)
        return fuse_crop_read_results(
            primary=primary,
            fallback=fallback,
            settings=self.settings,
        )


def fuse_crop_read_results(
    *,
    primary: CropReadResult,
    fallback: CropReadResult,
    settings: PipelineSettings,
) -> CropReadResult:
    if primary.readable and not fallback.readable:
        return primary
    if fallback.readable and not primary.readable:
        return fallback
    if not primary.readable and not fallback.readable:
        return CropReadResult(
            number=None,
            confidence=max(primary.confidence, fallback.confidence) * 0.25,
            uncertainty=max(primary.uncertainty, fallback.uncertainty),
            readable=False,
            backend="ensemble_none",
        )
    if primary.number == fallback.number and primary.number is not None:
        return CropReadResult(
            number=primary.number,
            confidence=_clamp01(max(primary.confidence, fallback.confidence) + 0.10),
            uncertainty=min(primary.uncertainty, fallback.uncertainty),
            readable=True,
            backend="ensemble_agree",
        )
    if (
        primary.number is not None
        and primary.uncertainty <= (settings.grad_max_uncertainty * 0.5)
        and primary.confidence >= fallback.confidence
    ):
        return CropReadResult(
            number=primary.number,
            confidence=primary.confidence,
            uncertainty=primary.uncertainty,
            readable=True,
            backend="ensemble_primary",
        )
    if (
        fallback.number is not None
        and fallback.confidence >= settings.parseq_min_confidence
        and primary.uncertainty > settings.grad_max_uncertainty
    ):
        return CropReadResult(
            number=fallback.number,
            confidence=fallback.confidence,
            uncertainty=fallback.uncertainty,
            readable=True,
            backend="ensemble_parseq",
        )
    return CropReadResult(
        number=None,
        confidence=max(primary.confidence, fallback.confidence) * 0.50,
        uncertainty=max(primary.uncertainty, fallback.uncertainty),
        readable=False,
        backend="ensemble_disagree",
    )
