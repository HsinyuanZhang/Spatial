"""Generate 32x32 2D staggered grid MEArec-style recordings.

Matches the TU Delft WIRED-OR reference configuration:
  - 32x32 electrodes, 36um pitch, 18um stagger (honeycomb)
  - Synthetic EAP templates with 2D Gaussian spatial decay
  - Poisson spike trains + convolution + colored noise
  - Saves as HDF5 compatible with MEArec format

Usage:
    python -m Spatial.data.generate_2d_grid [--output-dir Spatial/output/mearec_2d_grid]
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import h5py
import yaml
from scipy.signal import butter, sosfiltfilt

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG = {
    "grid_rows": 32,
    "grid_cols": 32,
    "pitch_um": 36.0,
    "stagger_um": 18.0,
    "sampling_frequency": 20000,
    "duration_s": 60,
    "num_neurons": 15,
    "mean_firing_rate": 8.0,
    "std_firing_rate": 2.0,
    "min_firing_rate": 1.0,
    "refractory_period_ms": 2.0,
    "neuron_min_distance_um": 250.0,
    "snr_levels": [3, 5, 8, 12],
    "reference_amplitude_uv": 400.0,
    "min_amplitude_uv": 300.0,
    "max_amplitude_uv": 800.0,
    "spatial_decay_um": 30.0,
    "template_duration_ms": 3.0,
    "seeds": [42],
    "filter_cutoff": [300, 6000],
    "filter_order": 3,
    "noise_color": True,
}


def build_staggered_grid(n_rows: int, n_cols: int, pitch: float, stagger: float):
    """Build 2D staggered (honeycomb) electrode positions.

    Odd rows are offset by stagger in x-direction.

    Returns:
        positions: (n_rows * n_cols, 2) array of (x, y) in um.
        row_col_map: (n_rows * n_cols, 2) array of (row, col) indices.
    """
    positions = []
    row_col_map = []
    for row in range(n_rows):
        x_offset = stagger if (row % 2 == 1) else 0.0
        for col in range(n_cols):
            x = col * pitch + x_offset
            y = row * pitch
            positions.append([x, y])
            row_col_map.append([row, col])
    return np.array(positions), np.array(row_col_map)


def generate_template_waveform(fs: int, duration_ms: float, rng: np.random.Generator):
    """Generate a single EAP-like waveform (negative-positive biphasic).

    Returns:
        waveform: (n_samples,) template waveform normalized to peak=-1.
    """
    n_samples = int(fs * duration_ms / 1000.0)
    t = np.arange(n_samples) / fs * 1000.0  # ms

    # Biphasic: fast negative trough + slower positive rebound
    t_peak = duration_ms * 0.35
    sigma_neg = duration_ms * 0.12
    sigma_pos = duration_ms * 0.25

    neg_phase = -np.exp(-0.5 * ((t - t_peak) / sigma_neg) ** 2)
    pos_phase = 0.4 * np.exp(-0.5 * ((t - t_peak - sigma_neg * 2) / sigma_pos) ** 2)

    waveform = neg_phase + pos_phase

    # Add slight randomness to shape
    jitter = rng.normal(0, 0.02, n_samples)
    waveform += jitter

    # Normalize to peak = -1
    waveform /= np.abs(waveform.min())
    return waveform


def generate_2d_templates(
    positions: np.ndarray,
    neuron_positions: np.ndarray,
    amplitudes: np.ndarray,
    fs: int,
    duration_ms: float,
    spatial_decay: float,
    rng: np.random.Generator,
):
    """Generate templates for all neurons on the 2D grid.

    Each neuron has a base waveform that decays spatially with distance.

    Args:
        positions: (n_channels, 2) electrode positions.
        neuron_positions: (n_neurons, 2) neuron (x, y) positions.
        amplitudes: (n_neurons,) peak amplitudes in uV.
        fs: Sampling frequency.
        duration_ms: Template duration in ms.
        spatial_decay: Spatial decay constant in um.
        rng: Random number generator.

    Returns:
        templates: (n_neurons, n_channels, n_samples) template array.
    """
    n_channels = len(positions)
    n_neurons = len(neuron_positions)
    n_samples = int(fs * duration_ms / 1000.0)

    templates = np.zeros((n_neurons, n_channels, n_samples), dtype=np.float64)

    for n in range(n_neurons):
        # Base waveform for this neuron (slightly different shape per neuron)
        base_wf = generate_template_waveform(fs, duration_ms, rng)

        # Spatial amplitude decay: A_i = A_peak * exp(-d^2 / (2*sigma^2))
        dists = np.sqrt(np.sum((positions - neuron_positions[n]) ** 2, axis=1))
        spatial_weights = np.exp(-dists ** 2 / (2 * spatial_decay ** 2))

        # Scale by neuron's peak amplitude
        for ch in range(n_channels):
            templates[n, ch, :] = base_wf * amplitudes[n] * spatial_weights[ch]

    return templates


def generate_poisson_spiketrains(
    n_neurons: int,
    duration_s: float,
    mean_rate: float,
    std_rate: float,
    min_rate: float,
    refractory_ms: float,
    fs: int,
    rng: np.random.Generator,
):
    """Generate Poisson spike trains with refractory period.

    Returns:
        spike_trains: list of (n_spikes_i,) arrays of sample indices.
        rates: (n_neurons,) actual firing rates.
    """
    duration_samples = int(duration_s * fs)
    refractory_samples = int(refractory_ms / 1000.0 * fs)

    spike_trains = []
    rates = np.zeros(n_neurons)

    for n in range(n_neurons):
        rate = max(min_rate, rng.normal(mean_rate, std_rate))
        rates[n] = rate

        # Generate Poisson process
        n_expected = int(rate * duration_s * 1.5)
        isis = rng.exponential(1.0 / rate, n_expected)
        times_s = np.cumsum(isis)
        times_s = times_s[times_s < duration_s]

        # Convert to samples
        spike_samples = (times_s * fs).astype(np.int64)

        # Enforce refractory period
        if len(spike_samples) > 1:
            keep = [0]
            for i in range(1, len(spike_samples)):
                if spike_samples[i] - spike_samples[keep[-1]] >= refractory_samples:
                    keep.append(i)
            spike_samples = spike_samples[keep]

        spike_trains.append(spike_samples)

    return spike_trains, rates


def convolve_spiketrains(
    templates: np.ndarray,
    spike_trains: list,
    n_channels: int,
    n_samples: int,
    rng: np.random.Generator,
):
    """Convolve spike trains with templates to produce recording.

    Args:
        templates: (n_neurons, n_channels, n_template_samples).
        spike_trains: list of spike sample arrays per neuron.
        n_channels: Number of channels.
        n_samples: Total recording samples.
        rng: Random generator for jitter.

    Returns:
        recording: (n_channels, n_samples) float64 recording.
    """
    recording = np.zeros((n_channels, n_samples), dtype=np.float64)
    n_template = templates.shape[2]
    half_t = n_template // 2

    for n, spikes in enumerate(spike_trains):
        for s in spikes:
            # Add temporal jitter (±1 sample)
            jitter = rng.integers(-1, 2)
            s_jit = s + jitter
            start = s_jit - half_t
            end = start + n_template

            # Boundary handling
            t_start = max(0, -start)
            t_end = min(n_template, n_samples - start)
            r_start = max(0, start)
            r_end = min(n_samples, end)

            if r_end > r_start:
                recording[:, r_start:r_end] += templates[n, :, t_start:t_end]

    return recording


def generate_colored_noise(n_channels: int, n_samples: int, fs: int, rng: np.random.Generator):
    """Generate colored noise resembling extracellular recording noise.

    Uses a resonating filter at ~300Hz to shape white noise.
    """
    white = rng.standard_normal((n_channels, n_samples))

    # Shape with bandpass-like coloring (emphasize 300-3000Hz)
    sos = butter(2, [200, 5000], btype='band', fs=fs, output='sos')
    colored = sosfiltfilt(sos, white, axis=1)

    # Normalize to unit variance per channel
    std = colored.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    colored /= std

    return colored


def bandpass_filter(data: np.ndarray, fs: int, cutoff: list, order: int):
    """Apply bandpass filter to recording."""
    sos = butter(order, cutoff, btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, data, axis=1)


def generate_one_recording(config: dict, snr: float, seed: int, output_dir: Path):
    """Generate a single 32x32 recording.

    Returns:
        Path to saved .h5 file, or None if skipped.
    """
    rec_id = f"grid2d_snr{snr}_seed{seed}"
    rec_dir = output_dir / rec_id
    h5_path = rec_dir / "recording.h5"

    if h5_path.exists():
        print(f"  [skip] {rec_id} already exists")
        return h5_path

    rng = np.random.default_rng(seed)
    fs = config["sampling_frequency"]
    duration_s = config["duration_s"]
    n_samples = int(duration_s * fs)
    n_rows = config["grid_rows"]
    n_cols = config["grid_cols"]
    n_channels = n_rows * n_cols
    n_neurons = config["num_neurons"]

    print(f"  Generating {rec_id}: {n_channels}ch, {duration_s}s, "
          f"{n_neurons} neurons, SNR={snr}...")

    # 1. Build geometry
    positions, row_col_map = build_staggered_grid(
        n_rows, n_cols, config["pitch_um"], config["stagger_um"]
    )

    # 2. Place neurons randomly on the grid
    x_range = positions[:, 0].max()
    y_range = positions[:, 1].max()
    margin = config["spatial_decay_um"]
    neuron_x = rng.uniform(margin, x_range - margin, n_neurons)
    neuron_y = rng.uniform(margin, y_range - margin, n_neurons)
    neuron_positions = np.column_stack([neuron_x, neuron_y])

    # Ensure minimum distance between neurons (rejection sampling)
    min_dist = config.get("neuron_min_distance_um", config["pitch_um"] * 5)
    for i in range(1, n_neurons):
        for attempt in range(500):
            dists = np.sqrt(np.sum((neuron_positions[:i] - neuron_positions[i]) ** 2, axis=1))
            if dists.min() >= min_dist:
                break
            neuron_positions[i] = [
                rng.uniform(margin, x_range - margin),
                rng.uniform(margin, y_range - margin),
            ]
        if dists.min() < min_dist:
            print(f"    WARNING: neuron {i} only {dists.min():.0f}um from nearest "
                  f"(target {min_dist:.0f}um) after 500 attempts")

    # 3. Assign amplitudes
    amplitudes = rng.uniform(
        config["min_amplitude_uv"], config["max_amplitude_uv"], n_neurons
    )

    # 4. Generate templates
    templates = generate_2d_templates(
        positions, neuron_positions, amplitudes,
        fs, config["template_duration_ms"],
        config["spatial_decay_um"], rng,
    )

    # 5. Generate spike trains
    spike_trains, rates = generate_poisson_spiketrains(
        n_neurons, duration_s,
        config["mean_firing_rate"], config["std_firing_rate"],
        config["min_firing_rate"], config["refractory_period_ms"],
        fs, rng,
    )

    total_spikes = sum(len(st) for st in spike_trains)
    print(f"    Spikes: {total_spikes}, rates: {rates.min():.1f}-{rates.max():.1f} Hz")

    # 6. Convolve
    recording = convolve_spiketrains(templates, spike_trains, n_channels, n_samples, rng)

    # 7. Add noise
    noise_level = config["reference_amplitude_uv"] / snr
    if config["noise_color"]:
        noise = generate_colored_noise(n_channels, n_samples, fs, rng)
    else:
        noise = rng.standard_normal((n_channels, n_samples))
    recording += noise * noise_level

    # 8. Bandpass filter
    recording = bandpass_filter(recording, fs, config["filter_cutoff"], config["filter_order"])

    # 9. Convert to int16 (with gain)
    max_val = np.abs(recording).max()
    gain = max_val / 32000.0 if max_val > 0 else 1.0
    recording_int16 = np.clip(recording / gain, -32768, 32767).astype(np.int16)

    # 10. Save
    rec_dir.mkdir(parents=True, exist_ok=True)

    # Build ground truth
    gt_spike_times = []
    gt_unit_ids = []
    for n, st in enumerate(spike_trains):
        gt_spike_times.append(st)
        gt_unit_ids.append(np.full(len(st), n, dtype=np.int32))
    gt_spike_times = np.concatenate(gt_spike_times) if gt_spike_times else np.array([], dtype=np.int64)
    gt_unit_ids = np.concatenate(gt_unit_ids) if gt_unit_ids else np.array([], dtype=np.int32)

    # Sort by time
    sort_idx = np.argsort(gt_spike_times)
    gt_spike_times = gt_spike_times[sort_idx]
    gt_unit_ids = gt_unit_ids[sort_idx]

    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('recordings', data=recording_int16, chunks=(n_channels, min(10000, n_samples)))
        f.create_dataset('spiketrains', data=gt_spike_times)
        f.create_dataset('unit_ids', data=gt_unit_ids)

        info = f.create_group('info')
        info.create_dataset('fs', data=fs)
        info.create_dataset('duration', data=duration_s)
        info.create_dataset('n_channels', data=n_channels)
        info.create_dataset('n_neurons', data=n_neurons)
        info.create_dataset('gain', data=gain)

        elec = info.create_group('electrodes')
        elec.create_dataset('positions', data=positions)
        elec.create_dataset('row_col', data=row_col_map)

        params_grp = f.create_group('params')
        params_grp.create_dataset('noise_level', data=noise_level)
        params_grp.create_dataset('snr', data=snr)
        params_grp.create_dataset('seed', data=seed)
        params_grp.create_dataset('grid_rows', data=n_rows)
        params_grp.create_dataset('grid_cols', data=n_cols)
        params_grp.create_dataset('pitch_um', data=config["pitch_um"])
        params_grp.create_dataset('stagger_um', data=config["stagger_um"])
        params_grp.create_dataset('spatial_decay_um', data=config["spatial_decay_um"])
        params_grp.create_dataset('filter_cutoff', data=config["filter_cutoff"])

    file_size_mb = h5_path.stat().st_size / 1e6
    print(f"    [done] {h5_path.name}: {file_size_mb:.0f} MB, "
          f"{total_spikes} spikes, {n_neurons} units")

    return h5_path


def check_disk_space(path: Path, min_free_gb: float = 5.0) -> float:
    """Check available disk space. Returns free GB."""
    import shutil
    usage = shutil.disk_usage(str(path))
    free_gb = usage.free / 1e9
    if free_gb < min_free_gb:
        print(f"  WARNING: Only {free_gb:.1f} GB free! Stopping generation.")
        return -1.0
    return free_gb


def main():
    parser = argparse.ArgumentParser(description="Generate 32x32 2D grid MEArec recordings")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "output" / "mearec_2d_grid")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--snr", type=float, nargs="+", default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--num-neurons", type=int, default=None)
    parser.add_argument("--min-free-gb", type=float, default=5.0,
                        help="Stop if free space drops below this")
    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    if args.config:
        with open(args.config) as f:
            config.update(yaml.safe_load(f))
    if args.snr:
        config["snr_levels"] = args.snr
    if args.duration:
        config["duration_s"] = args.duration
    if args.num_neurons:
        config["num_neurons"] = args.num_neurons

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Size estimate
    n_channels = config["grid_rows"] * config["grid_cols"]
    n_samples = int(config["duration_s"] * config["sampling_frequency"])
    bytes_per_rec = n_channels * n_samples * 2  # int16
    n_recordings = len(config["snr_levels"]) * len(config["seeds"])
    total_estimate_gb = bytes_per_rec * n_recordings / 1e9

    print(f"=== 32x32 2D Grid MEArec Generation ===")
    print(f"Grid: {config['grid_rows']}x{config['grid_cols']} = {n_channels} channels")
    print(f"Pitch: {config['pitch_um']}um, stagger: {config['stagger_um']}um")
    print(f"Duration: {config['duration_s']}s @ {config['sampling_frequency']}Hz")
    print(f"Neurons: {config['num_neurons']}, SNR levels: {config['snr_levels']}")
    print(f"Estimated size: {total_estimate_gb:.1f} GB ({n_recordings} recordings)")
    print(f"Output: {output_dir}")
    print()

    free_gb = check_disk_space(output_dir, args.min_free_gb)
    if free_gb < 0:
        sys.exit(1)
    print(f"Free space: {free_gb:.1f} GB")

    if total_estimate_gb > 50:
        print(f"ERROR: Estimated size {total_estimate_gb:.1f} GB exceeds 50 GB limit!")
        sys.exit(1)

    if total_estimate_gb > free_gb - args.min_free_gb:
        print(f"ERROR: Not enough space. Need {total_estimate_gb:.1f} GB, "
              f"have {free_gb:.1f} GB (keeping {args.min_free_gb} GB reserve)")
        sys.exit(1)

    print()
    results = []
    for snr in config["snr_levels"]:
        for seed in config["seeds"]:
            # Check space before each recording
            free_gb = check_disk_space(output_dir, args.min_free_gb)
            if free_gb < 0:
                print("Stopping early due to low disk space.")
                break

            h5_path = generate_one_recording(config, snr, seed, output_dir)
            if h5_path:
                results.append({
                    "id": f"grid2d_snr{snr}_seed{seed}",
                    "path": str(h5_path),
                    "snr": snr,
                    "seed": seed,
                    "size_mb": h5_path.stat().st_size / 1e6,
                })

    # Write manifest
    manifest = {
        "config": config,
        "recordings": results,
        "generation_time": datetime.now(timezone.utc).isoformat(),
        "total_size_mb": sum(r["size_mb"] for r in results),
    }
    manifest_path = output_dir / "manifest.yaml"
    with open(manifest_path, "w") as f:
        yaml.dump(manifest, f, default_flow_style=False)

    print(f"\n=== Done: {len(results)} recordings, "
          f"{sum(r['size_mb'] for r in results) / 1000:.1f} GB total ===")
    print(f"Manifest: {manifest_path}")

    free_gb = check_disk_space(output_dir, 0)
    print(f"Remaining free space: {free_gb:.1f} GB")


if __name__ == "__main__":
    main()
