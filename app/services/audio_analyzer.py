# Audio analysis pipeline — whistle DSP + YAMNet TFLite + crowd energy

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from app.services.audio_types import (
    AudioAnalysisResult,
    AudioEvent,
    EnergyPoint,
    PlayBoundary,
)

if TYPE_CHECKING:
    pass

LOGGER = logging.getLogger(__name__)

# YAMNet TFLite model path — downloaded at Docker build time
YAMNET_MODEL_PATH = os.getenv("YAMNET_MODEL_PATH", "/app/app/model/yamnet.tflite")

# Whistle detection DSP params
WHISTLE_LOW_HZ = 3600
WHISTLE_HIGH_HZ = 4100
WHISTLE_MIN_DURATION = 0.3
WHISTLE_MAX_DURATION = 2.0
WHISTLE_THRESHOLD = 0.3

# Audio chunk size for YAMNet (seconds)
YAMNET_CHUNK_SECONDS = 30

# YAMNet class indices for sports-relevant sounds
YAMNET_WHISTLE_CLASSES = {399, 400, 401}  # Whistle, Steam whistle
YAMNET_CHEERING_CLASSES = {13, 14, 15, 16}  # Cheering, Crowd, Applause
YAMNET_BUZZER_CLASSES = {389, 390}  # Buzzer, Alarm
YAMNET_SPEECH_CLASSES = {0, 1, 2, 3}  # Speech, Narration

# Sliding window for crowd energy (seconds)
ENERGY_WINDOW_SECONDS = 2.0
ENERGY_HOP_SECONDS = 0.5


def _load_audio(audio_path: Path) -> tuple[np.ndarray, int]:
    """Load audio file as numpy array. Returns (samples, sample_rate)."""
    try:
        import soundfile as sf
        data, sr = sf.read(str(audio_path), dtype="float32")
        if len(data.shape) > 1:
            data = data.mean(axis=1)  # mono
        return data, sr
    except ImportError:
        LOGGER.warning("soundfile not installed, trying scipy")
        from scipy.io import wavfile
        sr, data = wavfile.read(str(audio_path))
        if data.dtype != np.float32:
            data = data.astype(np.float32) / np.iinfo(data.dtype).max
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        return data, sr


def detect_whistles(samples: np.ndarray, sample_rate: int) -> list[AudioEvent]:
    """Pure DSP whistle detection using bandpass filter + peak detection."""
    events: list[AudioEvent] = []
    try:
        from scipy.signal import butter, sosfilt

        # Bandpass filter for whistle frequency range
        nyquist = sample_rate / 2
        low = WHISTLE_LOW_HZ / nyquist
        high = min(WHISTLE_HIGH_HZ / nyquist, 0.99)
        if low >= high:
            return events

        sos = butter(4, [low, high], btype="band", output="sos")
        filtered = sosfilt(sos, samples)

        # Compute energy in short frames
        frame_length = int(sample_rate * 0.05)  # 50ms frames
        hop_length = int(sample_rate * 0.025)    # 25ms hop

        num_frames = (len(filtered) - frame_length) // hop_length + 1
        if num_frames <= 0:
            return events

        energy = np.zeros(num_frames)
        for i in range(num_frames):
            start = i * hop_length
            frame = filtered[start:start + frame_length]
            energy[i] = np.sqrt(np.mean(frame ** 2))

        # Normalize energy
        max_energy = energy.max()
        if max_energy > 0:
            energy = energy / max_energy

        # Find whistle regions (energy above threshold)
        is_whistle = energy > WHISTLE_THRESHOLD
        in_whistle = False
        whistle_start = 0

        for i in range(len(is_whistle)):
            t = i * (hop_length / sample_rate)
            if is_whistle[i] and not in_whistle:
                in_whistle = True
                whistle_start = t
            elif not is_whistle[i] and in_whistle:
                in_whistle = False
                duration = t - whistle_start
                if WHISTLE_MIN_DURATION <= duration <= WHISTLE_MAX_DURATION:
                    confidence = float(energy[max(0, i - 5):i].mean()) if i >= 5 else float(energy[:i + 1].mean())
                    events.append(AudioEvent(
                        timestamp=whistle_start,
                        event_type="whistle",
                        confidence=min(1.0, confidence * 1.5),
                        duration=duration,
                    ))

    except Exception as exc:
        LOGGER.warning("Whistle detection failed: %s", exc)

    return events


def compute_crowd_energy(samples: np.ndarray, sample_rate: int) -> list[EnergyPoint]:
    """Compute RMS energy over sliding window, normalized 0-1."""
    points: list[EnergyPoint] = []
    window_samples = int(ENERGY_WINDOW_SECONDS * sample_rate)
    hop_samples = int(ENERGY_HOP_SECONDS * sample_rate)

    if len(samples) < window_samples:
        return points

    energies = []
    timestamps = []
    for start in range(0, len(samples) - window_samples, hop_samples):
        chunk = samples[start:start + window_samples]
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        t = start / sample_rate
        energies.append(rms)
        timestamps.append(t)

    if not energies:
        return points

    # Normalize to 0-1
    max_e = max(energies)
    if max_e > 0:
        energies = [e / max_e for e in energies]

    for t, e in zip(timestamps, energies):
        points.append(EnergyPoint(timestamp=t, energy=e))

    return points


def classify_with_yamnet(samples: np.ndarray, sample_rate: int) -> list[AudioEvent]:
    """Classify audio using YAMNet TFLite model in 30s chunks."""
    events: list[AudioEvent] = []

    if not Path(YAMNET_MODEL_PATH).exists():
        LOGGER.info("YAMNet model not found at %s, skipping", YAMNET_MODEL_PATH)
        return events

    try:
        import tflite_runtime.interpreter as tflite

        interpreter = tflite.Interpreter(model_path=YAMNET_MODEL_PATH)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # YAMNet expects 16kHz mono audio
        if sample_rate != 16000:
            from scipy.signal import resample
            num_samples = int(len(samples) * 16000 / sample_rate)
            samples = resample(samples, num_samples)
            sample_rate = 16000

        chunk_samples = YAMNET_CHUNK_SECONDS * sample_rate

        for chunk_start in range(0, len(samples), chunk_samples):
            chunk = samples[chunk_start:chunk_start + chunk_samples]
            if len(chunk) < sample_rate:  # Skip very short chunks
                continue

            chunk_time = chunk_start / sample_rate

            # Pad to expected input size if needed
            expected_len = input_details[0]["shape"][-1] if len(input_details[0]["shape"]) > 1 else len(chunk)
            if len(chunk) < expected_len:
                chunk = np.pad(chunk, (0, expected_len - len(chunk)))
            elif len(chunk) > expected_len:
                chunk = chunk[:expected_len]

            chunk = chunk.astype(np.float32)
            if len(input_details[0]["shape"]) == 1:
                interpreter.resize_tensor_input(input_details[0]["index"], chunk.shape)
                interpreter.allocate_tensors()

            interpreter.set_tensor(input_details[0]["index"], chunk)
            interpreter.invoke()

            scores = interpreter.get_tensor(output_details[0]["index"])
            if len(scores.shape) > 1:
                scores = scores.mean(axis=0)  # Average over time frames

            # Check for sports-relevant sounds
            for class_set, event_type in [
                (YAMNET_WHISTLE_CLASSES, "whistle"),
                (YAMNET_CHEERING_CLASSES, "cheering"),
                (YAMNET_BUZZER_CLASSES, "buzzer"),
                (YAMNET_SPEECH_CLASSES, "commentary"),
            ]:
                for idx in class_set:
                    if idx < len(scores) and scores[idx] > 0.3:
                        events.append(AudioEvent(
                            timestamp=chunk_time,
                            event_type=event_type,
                            confidence=float(scores[idx]),
                        ))
                        break  # One event per type per chunk

    except ImportError:
        LOGGER.info("tflite_runtime not installed, skipping YAMNet classification")
    except Exception as exc:
        LOGGER.warning("YAMNet classification failed: %s", exc)

    return events


def find_play_boundaries(
    whistle_events: list[AudioEvent],
    energy_points: list[EnergyPoint],
    yamnet_events: list[AudioEvent],
    duration: float,
) -> list[PlayBoundary]:
    """Combine signals to find play start/end boundaries."""
    boundaries: list[PlayBoundary] = []

    # 1. Whistle pairs → play boundaries
    whistles = sorted([e for e in whistle_events if e.event_type == "whistle"], key=lambda e: e.timestamp)
    for i in range(len(whistles) - 1):
        gap = whistles[i + 1].timestamp - whistles[i].timestamp
        if 3.0 <= gap <= 45.0:  # Reasonable play duration
            boundaries.append(PlayBoundary(
                start_time=whistles[i].timestamp,
                end_time=whistles[i + 1].timestamp,
                confidence=(whistles[i].confidence + whistles[i + 1].confidence) / 2,
                source="whistle_pair",
            ))

    # 2. Energy spikes → active play regions
    if energy_points:
        in_active = False
        active_start = 0.0
        spike_threshold = 0.6

        for pt in energy_points:
            if pt.energy > spike_threshold and not in_active:
                in_active = True
                active_start = pt.timestamp
            elif pt.energy < spike_threshold * 0.5 and in_active:
                in_active = False
                if pt.timestamp - active_start >= 2.0:
                    boundaries.append(PlayBoundary(
                        start_time=active_start,
                        end_time=pt.timestamp,
                        confidence=0.5,
                        source="energy_spike",
                    ))

        if in_active:
            last_t = energy_points[-1].timestamp
            if last_t - active_start >= 2.0:
                boundaries.append(PlayBoundary(
                    start_time=active_start,
                    end_time=last_t,
                    confidence=0.4,
                    source="energy_spike",
                ))

    # 3. Cheering events from YAMNet → boost nearby boundaries
    cheering_times = [e.timestamp for e in yamnet_events if e.event_type == "cheering"]
    for boundary in boundaries:
        for ct in cheering_times:
            if boundary.start_time - 5 <= ct <= boundary.end_time + 5:
                boundary.confidence = min(1.0, boundary.confidence + 0.15)

    # Sort by start time and merge overlapping
    boundaries.sort(key=lambda b: b.start_time)
    merged: list[PlayBoundary] = []
    for b in boundaries:
        if merged and b.start_time <= merged[-1].end_time + 2.0:
            last = merged[-1]
            merged[-1] = PlayBoundary(
                start_time=last.start_time,
                end_time=max(last.end_time, b.end_time),
                confidence=max(last.confidence, b.confidence),
                source=f"{last.source}+{b.source}" if last.source != b.source else last.source,
            )
        else:
            merged.append(b)

    return merged


def analyze_audio(audio_path: Path) -> AudioAnalysisResult:
    """Full audio analysis pipeline."""
    if not audio_path or not audio_path.exists():
        return AudioAnalysisResult(has_audio=False)

    try:
        samples, sr = _load_audio(audio_path)
        duration = len(samples) / sr
        LOGGER.info("Audio loaded: %.1fs at %dHz, %d samples", duration, sr, len(samples))

        # Run all analyses
        whistle_events = detect_whistles(samples, sr)
        LOGGER.info("Whistle detection: %d events", len(whistle_events))

        energy_curve = compute_crowd_energy(samples, sr)
        LOGGER.info("Crowd energy: %d points", len(energy_curve))

        yamnet_events = classify_with_yamnet(samples, sr)
        LOGGER.info("YAMNet: %d events", len(yamnet_events))

        # Combine all audio events
        all_events = whistle_events + yamnet_events

        # Find play boundaries
        play_boundaries = find_play_boundaries(whistle_events, energy_curve, yamnet_events, duration)
        LOGGER.info("Play boundaries: %d found", len(play_boundaries))

        return AudioAnalysisResult(
            events=all_events,
            energy_curve=energy_curve,
            play_boundaries=play_boundaries,
            duration=duration,
            has_audio=True,
        )

    except Exception as exc:
        LOGGER.warning("Audio analysis failed: %s", exc)
        return AudioAnalysisResult(has_audio=False)
