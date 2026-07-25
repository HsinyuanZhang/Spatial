"""Lightweight spike detection for Spatial experiments.

Provides NEO-based detection and simple window extraction.
"""

import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(data: np.ndarray, fs: int, low: float = 300.0, high: float = 6000.0, order: int = 3) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter.

    Args:
        data: (n_channels, n_samples)
    """
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, data, axis=1)


def compute_neo(signal: np.ndarray) -> np.ndarray:
    """Nonlinear Energy Operator: NEO[x](n) = x(n)^2 - x(n-1)*x(n+1).

    Args:
        signal: (n_channels, n_samples)
    Returns:
        neo: (n_channels, n_samples) with first/last samples zeroed.
    """
    neo = np.zeros_like(signal)
    neo[:, 1:-1] = signal[:, 1:-1] ** 2 - signal[:, :-2] * signal[:, 2:]
    return neo


def detect_spikes_neo(
    raw_data: np.ndarray,
    fs: int,
    c_neo: float = 40.0,
    filter_cutoff: tuple = (300.0, 6000.0),
    refractory_samples: int = 30,
) -> tuple:
    """Detect spikes using NEO threshold on multi-channel data.

    Args:
        raw_data: (n_channels, n_samples) raw recording.
        fs: Sampling rate.
        c_neo: Threshold multiplier. threshold = c_neo * median(|NEO|) / 0.6745.
        filter_cutoff: Bandpass filter frequencies.
        refractory_samples: Minimum samples between detections on same channel.

    Returns:
        spike_times: (n_spikes,) sample indices of detected spikes.
        central_channels: (n_spikes,) channel with max NEO at each detection.
    """
    filtered = bandpass_filter(raw_data, fs, filter_cutoff[0], filter_cutoff[1])
    neo = compute_neo(filtered)

    # Per-channel adaptive threshold
    n_channels, n_samples = neo.shape
    thresholds = np.zeros(n_channels)
    for ch in range(n_channels):
        sigma = np.median(np.abs(neo[ch])) / 0.6745
        thresholds[ch] = c_neo * sigma

    # Find peaks above threshold on any channel
    # Use max across channels for detection, then find central channel
    neo_max = np.max(neo, axis=0)  # (n_samples,)
    global_threshold = np.min(thresholds)  # conservative

    # Simple peak detection with refractory period
    spike_times = []
    central_channels = []
    last_spike = -refractory_samples

    above = neo_max > global_threshold
    candidates = np.where(above)[0]

    i = 0
    while i < len(candidates):
        t = candidates[i]
        if t - last_spike < refractory_samples:
            i += 1
            continue

        # Find local peak in a small window
        window_end = min(t + refractory_samples, n_samples)
        local_region = neo_max[t:window_end]
        peak_offset = np.argmax(local_region)
        peak_t = t + peak_offset

        # Central channel: channel with max NEO at peak time
        central_ch = np.argmax(neo[:, peak_t])

        # Verify this channel exceeds its own threshold
        if neo[central_ch, peak_t] > thresholds[central_ch]:
            spike_times.append(peak_t)
            central_channels.append(central_ch)
            last_spike = peak_t

        # Skip past refractory period
        i = np.searchsorted(candidates, peak_t + refractory_samples)

    return np.array(spike_times, dtype=np.int64), np.array(central_channels, dtype=np.int64)


def extract_spike_windows(
    raw_data: np.ndarray,
    spike_times: np.ndarray,
    window_pre: int = 15,
    window_post: int = 15,
) -> np.ndarray:
    """Extract multi-channel spike waveform snippets.

    Args:
        raw_data: (n_channels, n_samples) filtered or raw data.
        spike_times: (n_spikes,) center sample indices.
        window_pre: Samples before peak.
        window_post: Samples after peak.

    Returns:
        snippets: (n_valid_spikes, n_channels, window_pre + window_post)
    """
    n_channels, n_samples = raw_data.shape
    win_len = window_pre + window_post

    valid_mask = (spike_times >= window_pre) & (spike_times < n_samples - window_post)
    valid_times = spike_times[valid_mask]

    snippets = np.zeros((len(valid_times), n_channels, win_len), dtype=np.float64)
    for i, t in enumerate(valid_times):
        snippets[i] = raw_data[:, t - window_pre:t + window_post]

    return snippets


def get_peak_amplitudes(
    raw_data: np.ndarray,
    spike_times: np.ndarray,
    window: int = 15,
) -> tuple:
    """Get peak amplitude per channel for each spike.

    Returns:
        amplitudes: (n_valid_spikes, n_channels) signed peak amplitudes.
        valid_times: (n_valid_spikes,) corresponding spike times.
        central_channels: (n_valid_spikes,) channel with max |amplitude|.
    """
    n_channels, n_samples = raw_data.shape
    valid_mask = (spike_times >= window) & (spike_times < n_samples - window)
    valid_times = spike_times[valid_mask]

    amplitudes = np.zeros((len(valid_times), n_channels), dtype=np.float64)
    for i, t in enumerate(valid_times):
        snippet = raw_data[:, t - window:t + window]
        # Signed peak: value at max |sample|
        peak_idx = np.argmax(np.abs(snippet), axis=1)
        for ch in range(n_channels):
            amplitudes[i, ch] = snippet[ch, peak_idx[ch]]

    central_channels = np.argmax(np.abs(amplitudes), axis=1)
    return amplitudes, valid_times, central_channels
