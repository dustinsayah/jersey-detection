# Audio analysis dataclasses

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AudioEvent:
    timestamp: float
    event_type: str  # whistle, cheering, buzzer, crowd_noise, speech
    confidence: float = 0.0
    duration: float = 0.0


@dataclass
class EnergyPoint:
    timestamp: float
    energy: float  # 0.0 - 1.0 normalized RMS


@dataclass
class PlayBoundary:
    start_time: float
    end_time: float
    confidence: float = 0.0
    source: str = ""  # whistle_pair, energy_spike, yamnet_transition


@dataclass
class AudioAnalysisResult:
    events: list[AudioEvent] = field(default_factory=list)
    energy_curve: list[EnergyPoint] = field(default_factory=list)
    play_boundaries: list[PlayBoundary] = field(default_factory=list)
    duration: float = 0.0
    has_audio: bool = False
