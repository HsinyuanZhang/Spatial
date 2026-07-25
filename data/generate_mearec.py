"""Generate MEArec-compatible 32-channel recordings for Spatial experiments.

Bypasses MEArec's gen_recordings API (which has strict template format requirements)
and generates data directly: Poisson spike trains + template convolution + noise.
Output format is compatible with our load_mearec_h5() loader.

Usage:
    python -m Spatial.data.generate_mearec [--snr 3 5 12] [--duration 60]
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml

SPATIAL_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = SPATIAL_ROOT / "output" / "templates_Neuronexus_32_n15.h5"

DEFAULT_CONFIG = {
    "sampling_frequency": 30000,
    "duration_s": 60,
    "num_channels": 32,
    "snr_levels": [3, 5, 12],
    "seeds": [202600],
    "num_exc_neurons": 8,
    "num_inh_neurons": 2,
    "reference_amplitude_uv": 50,
    "filter_cutoff": [300, 6000],
    "filter_order": 3,
}


def load_templates(template_path: Path):
    """Load template waveforms and geometry from HDF5."""
    with h5py.File(str(template_path), "r") as f:
        templates = f["templates"][:]  # (n_templates, n_channels, n_samples)
        locations = f["locations"][:]  # (n_templates, 3)
    return templates, locations


def generate_poisson_spiketrains(
    n_units: int,
    duration_s: float,
    fs: int,
    min_rate: float = 1.0,
    max_rate: float = 15.0,
    refractory_ms: float = 2.0,
    rng: np.random.Generator = None,
):
    """Generate Poisson spike trains with refractory period.

    Returns:
        spike_trains: list of arrays, each containing spike times in samples.
    """
    if rng is None:
        rng = np.random.default_rng()

    refractory_samples = int(refractory_ms / 1000 * fs)
    total_samples = int(duration_s * fs)
    spike_trains = []

    for _ in range(n_units):
        rate = rng.uniform(min_rate, max_rate)  # Hz
        n_expected = int(rate * duration_s * 1.5)
        # Generate inter-spike intervals
        isis = rng.exponential(1.0 / rate, size=n_expected)
        times_s = np.cumsum(isis)
        times_s = times_s[times_s < duration_s]
        times_samples = (times_s * fs).astype(np.int64)

        # Enforce refractory period
        if len(times_samples) > 1:
            keep = [times_samples[0]]
            for t in times_samples[1:]:
                if t - keep[-1] >= refractory_samples:
                    keep.append(t)
            times_samples = np.array(keep, dtype=np.int64)

        spike_trains.append(times_samples)

    return spike_trains


def convolve_templates(
    spike_trains: list,
    templates: np.ndarray,
    n_channels: int,
    n_samples: int,
    rng: np.random.Generator = None,
):
    """Convolve spike trains with templates to produce multi-channel recording.

    Args:
        spike_trains: list of spike time arrays (in samples).
        templates: (n_templates, n_channels, template_len).
        n_channels: number of output channels.
        n_samples: total recording length in samples.

    Returns:
        recording: (n_samples, n_channels) float64.
    """
    if rng is None:
        rng = np.random.default_rng()

    recording = np.zeros((n_samples, n_channels), dtype=np.float64)
    n_templates = templates.shape[0]
    template_len = templates.shape[2]
    pre_peak = template_len // 3  # align peak at 1/3 into template

    for unit_idx, st in enumerate(spike_trains):
        tmpl_idx = unit_idx % n_templates
        tmpl = templates[tmpl_idx]  # (n_channels, template_len)

        for spike_t in st:
            start = spike_t - pre_peak
            end = start + template_len
            # Clip to recording bounds
            t_start = max(0, start)
            t_end = min(n_samples, end)
            tmpl_start = t_start - start
            tmpl_end = tmpl_start + (t_end - t_start)

            if t_end > t_start:
                recording[t_start:t_end] += tmpl[:, tmpl_start:tmpl_end].T

    return recording


def add_noise(recording: np.ndarray, noise_level: float, rng: np.random.Generator = None):
    """Add Gaussian noise to recording."""
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0, noise_level, recording.shape)
    return recording + noise


def bandpass_filter_recording(recording: np.ndarray, fs: int, low: float, high: float, order: int):
    """Apply zero-phase bandpass filter."""
    from scipy.signal import butter, filtfilt
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, recording, axis=0)


def generate_one_recording(
    config: dict,
    snr: float,
    seed: int,
    templates: np.ndarray,
    output_dir: Path,
):
    """Generate a single recording and save as HDF5."""
    fs = config["sampling_frequency"]
    duration = config["duration_s"]
    n_channels = config["num_channels"]
    n_exc = config["num_exc_neurons"]
    n_inh = config["num_inh_neurons"]
    n_units = n_exc + n_inh
    ref_amp = config["reference_amplitude_uv"]
    noise_level = ref_amp / snr

    rng = np.random.default_rng(seed)
    n_samples = int(duration * fs)

    rec_id = f"spatial_snr{snr}_seed{seed}"
    rec_dir = output_dir / rec_id
    rec_dir.mkdir(parents=True, exist_ok=True)
    h5_path = rec_dir / "recording.h5"

    if h5_path.exists():
        print(f"  [skip] {rec_id} already exists")
        return {"id": rec_id, "path": str(h5_path), "snr": snr}

    print(f"  Generating {rec_id} (SNR={snr}, noise={noise_level:.2f} uV)...")

    # Generate spike trains
    spike_trains = generate_poisson_spiketrains(
        n_units=n_units,
        duration_s=duration,
        fs=fs,
        min_rate=1.0,
        max_rate=15.0,
        rng=rng,
    )

    # Convolve with templates
    recording = convolve_templates(spike_trains, templates, n_channels, n_samples, rng)

    # Add noise
    recording = add_noise(recording, noise_level, rng)

    # Bandpass filter
    recording = bandpass_filter_recording(
        recording, fs,
        config["filter_cutoff"][0],
        config["filter_cutoff"][1],
        config["filter_order"],
    )

    # Save as HDF5 in MEArec-compatible format
    with h5py.File(str(h5_path), "w") as f:
        f.create_dataset("recordings", data=recording.astype(np.float32))
        f["recordings"].attrs["fs"] = fs

        # Channel positions (linear probe, 25um pitch)
        ch_pos = np.zeros((n_channels, 3), dtype=np.float64)
        ch_pos[:, 1] = np.arange(n_channels) * 25.0
        f.create_dataset("channel_positions", data=ch_pos)

        # Spike trains group
        st_grp = f.create_group("spiketrains")
        for unit_idx, st in enumerate(spike_trains):
            times_s = st.astype(np.float64) / fs
            st_grp.create_dataset(f"unit_{unit_idx}", data=times_s)

        # Info
        info = f.create_group("info")
        info.attrs["fs"] = fs
        info.attrs["n_units"] = n_units
        info.attrs["duration"] = duration
        info.attrs["noise_level"] = noise_level
        info.attrs["snr"] = snr

    total_spikes = sum(len(st) for st in spike_trains)
    print(f"  [done] {h5_path.name}: {n_units} units, {total_spikes} spikes, "
          f"{h5_path.stat().st_size / 1e6:.1f} MB")

    return {
        "id": rec_id,
        "path": str(h5_path),
        "snr": snr,
        "n_units": n_units,
        "total_spikes": total_spikes,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate MEArec 32ch recordings for Spatial experiments")
    parser.add_argument("--output-dir", type=Path, default=SPATIAL_ROOT / "output" / "mearec_generated")
    parser.add_argument("--template", type=Path, default=TEMPLATE_PATH)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--snr", type=float, nargs="+", default=None)
    parser.add_argument("--duration", type=float, default=None)
    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    if args.config:
        with open(args.config) as f:
            config.update(yaml.safe_load(f))
    if args.snr:
        config["snr_levels"] = args.snr
    if args.duration:
        config["duration_s"] = args.duration

    if not args.template.exists():
        print(f"ERROR: Template not found: {args.template}")
        sys.exit(1)

    templates, locations = load_templates(args.template)
    print(f"Template: {args.template} ({templates.shape[0]} templates, {templates.shape[1]} channels)")
    print(f"Output:   {args.output_dir}")
    print(f"Config:   SNR={config['snr_levels']}, duration={config['duration_s']}s, "
          f"units={config['num_exc_neurons']}E+{config['num_inh_neurons']}I")
    print()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for snr in config["snr_levels"]:
        for seed in config["seeds"]:
            row = generate_one_recording(config, snr, seed, templates, args.output_dir)
            results.append(row)

    manifest_path = args.output_dir / "manifest.yaml"
    with open(manifest_path, "w") as f:
        yaml.dump({"config": config, "recordings": results}, f, default_flow_style=False)
    print(f"\nManifest: {manifest_path}")
    print(f"Total: {len(results)} recordings")


if __name__ == "__main__":
    main()
