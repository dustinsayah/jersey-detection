from __future__ import annotations

from dataclasses import replace

from app.services.detection_runtime import PipelineSettings
from app.services.jersey_reader import CropReadResult, fuse_crop_read_results


def test_fuse_prefers_readable_primary_when_fallback_is_unreadable() -> None:
    settings = replace(PipelineSettings(), grad_max_uncertainty=0.25)

    result = fuse_crop_read_results(
        primary=CropReadResult(
            number=12,
            confidence=0.82,
            uncertainty=0.11,
            readable=True,
            backend="grad_vitb",
        ),
        fallback=CropReadResult(
            number=None,
            confidence=0.21,
            uncertainty=0.88,
            readable=False,
            backend="parseq",
        ),
        settings=settings,
    )

    assert result.number == 12
    assert result.backend == "grad_vitb"
    assert result.readable is True


def test_fuse_boosts_confidence_when_models_agree() -> None:
    settings = replace(PipelineSettings(), grad_max_uncertainty=0.25)

    result = fuse_crop_read_results(
        primary=CropReadResult(
            number=23,
            confidence=0.73,
            uncertainty=0.09,
            readable=True,
            backend="grad_vitb",
        ),
        fallback=CropReadResult(
            number=23,
            confidence=0.69,
            uncertainty=0.18,
            readable=True,
            backend="parseq:torso",
        ),
        settings=settings,
    )

    assert result.number == 23
    assert result.backend == "ensemble_agree"
    assert result.confidence > 0.73
    assert result.readable is True


def test_fuse_returns_unreadable_when_models_disagree_weakly() -> None:
    settings = replace(PipelineSettings(), grad_max_uncertainty=0.25)

    result = fuse_crop_read_results(
        primary=CropReadResult(
            number=11,
            confidence=0.54,
            uncertainty=0.31,
            readable=False,
            backend="grad_vitb",
        ),
        fallback=CropReadResult(
            number=17,
            confidence=0.52,
            uncertainty=0.48,
            readable=False,
            backend="parseq:torso",
        ),
        settings=settings,
    )

    assert result.number is None
    assert result.readable is False
    assert result.backend == "ensemble_none"
