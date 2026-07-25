"""Validate the 32x32 2D grid dataset using the reference pipeline.

Loads a generated recording, runs WiredOR → Detection → Features → SOM,
and evaluates clustering accuracy against ground truth.

Usage:
    python -m Spatial.experiments.validate_2d_grid [--snr 5] [--duration 10]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from algorithms.reference_port import (
    wired_or_fast,
    extract_features_reference,
    detect_spikes_reference,
    ReferenceSOM,
    find_neighbor_elec,
)


def load_2d_grid_recording(h5_path: Path, duration_s: float = None):
    """Load a 2D grid recording from HDF5.

    Returns:
        raw_data: (n_rows, n_cols, n_samples) float64 in uV.
        positions: (n_channels, 2) electrode positions.
        row_col: (n_channels, 2) row/col indices.
        gt_times: (n_spikes,) ground truth spike sample indices.
        gt_units: (n_spikes,) ground truth unit IDs.
        fs: sampling frequency.
        gain: int16→uV gain.
    """
    with h5py.File(h5_path, 'r') as f:
        recordings = f['recordings'][:]  # (n_channels, n_samples) int16
        gt_times = f['spiketrains'][:]
        gt_units = f['unit_ids'][:]
        fs = int(f['info/fs'][()])
        gain = float(f['info/gain'][()])
        positions = f['info/electrodes/positions'][:]
        row_col = f['info/electrodes/row_col'][:]
        n_rows = int(f['params/grid_rows'][()])
        n_cols = int(f['params/grid_cols'][()])

    # Truncate to duration
    if duration_s is not None:
        n_samples = int(duration_s * fs)
        recordings = recordings[:, :n_samples]
        mask = gt_times < n_samples
        gt_times = gt_times[mask]
        gt_units = gt_units[mask]

    # Convert to uV and reshape to (n_rows, n_cols, n_samples)
    raw_uv = recordings.astype(np.float64) * gain
    n_samples = raw_uv.shape[1]
    raw_3d = raw_uv.reshape(n_rows, n_cols, n_samples)

    return raw_3d, positions, row_col, gt_times, gt_units, fs, gain


def evaluate_clustering(pred_labels, pred_times, gt_units, gt_times, tolerance_samples=10):
    """Evaluate clustering accuracy by matching predicted spikes to GT.

    Uses greedy time-based matching + Hungarian algorithm for label mapping.

    Returns:
        accuracy: fraction of GT spikes correctly assigned.
        n_matched: number of matched spikes.
        n_clusters: number of predicted clusters.
    """
    from scipy.optimize import linear_sum_assignment

    if len(pred_times) == 0 or len(gt_times) == 0:
        return 0.0, 0, 0

    # Greedy matching: for each GT spike, find closest pred spike within tolerance
    matched_pred = []
    matched_gt = []
    pred_used = set()

    # Sort both by time
    gt_sort = np.argsort(gt_times)
    pred_sort = np.argsort(pred_times)

    pred_times_sorted = pred_times[pred_sort]
    pred_labels_sorted = pred_labels[pred_sort]

    for gi in gt_sort:
        gt_t = gt_times[gi]
        # Binary search for closest pred spike
        idx = np.searchsorted(pred_times_sorted, gt_t)
        best_pi = -1
        best_dt = tolerance_samples + 1

        for candidate in [idx - 1, idx, idx + 1]:
            if 0 <= candidate < len(pred_times_sorted) and candidate not in pred_used:
                dt = abs(pred_times_sorted[candidate] - gt_t)
                if dt < best_dt:
                    best_dt = dt
                    best_pi = candidate

        if best_pi >= 0 and best_dt <= tolerance_samples:
            matched_pred.append(pred_labels_sorted[best_pi])
            matched_gt.append(gt_units[gi])
            pred_used.add(best_pi)

    n_matched = len(matched_gt)
    if n_matched == 0:
        return 0.0, 0, len(np.unique(pred_labels[pred_labels >= 0]))

    matched_pred = np.array(matched_pred)
    matched_gt = np.array(matched_gt)

    # Hungarian algorithm for optimal label mapping
    pred_unique = np.unique(matched_pred[matched_pred >= 0])
    gt_unique = np.unique(matched_gt)

    if len(pred_unique) == 0:
        return 0.0, n_matched, 0

    # Build cost matrix
    cost = np.zeros((len(pred_unique), len(gt_unique)))
    for i, pl in enumerate(pred_unique):
        for j, gl in enumerate(gt_unique):
            cost[i, j] = -np.sum((matched_pred == pl) & (matched_gt == gl))

    row_ind, col_ind = linear_sum_assignment(cost)
    n_correct = -cost[row_ind, col_ind].sum()

    accuracy = n_correct / len(gt_times)
    n_clusters = len(np.unique(pred_labels[pred_labels >= 0]))

    return accuracy, n_matched, n_clusters


def run_validation(h5_path: Path, duration_s: float, wire_num: int = 4,
                   lsb_step: int = 4, remove_scale: int = 8,
                   neo_threshold: float = 40.0, verbose: bool = True):
    """Run full reference pipeline validation on a 2D grid recording."""

    if verbose:
        print(f"\n{'='*60}")
        print(f"Validating: {h5_path.name}")
        print(f"Duration: {duration_s}s, wire_num={wire_num}, "
              f"lsb_step={lsb_step}, beta={remove_scale}")
        print(f"{'='*60}")

    # Load data
    raw_3d, positions, row_col, gt_times, gt_units, fs, gain = \
        load_2d_grid_recording(h5_path, duration_s)

    n_rows, n_cols, n_samples = raw_3d.shape
    n_gt_spikes = len(gt_times)
    n_gt_units = len(np.unique(gt_units))

    if verbose:
        print(f"  Data: {n_rows}x{n_cols} = {n_rows*n_cols} ch, "
              f"{n_samples} samples @ {fs}Hz")
        print(f"  GT: {n_gt_spikes} spikes, {n_gt_units} units")

    # Quantize: simulate ADC with noise-floor thresholding
    # The reference hardware only sees signals above the comparator threshold.
    # Zero out values below noise floor before Wired-OR.
    noise_std = np.median(np.abs(raw_3d)) / 0.6745  # robust std estimate
    threshold = noise_std * 3.0
    data_threshed = raw_3d.copy()
    data_threshed[np.abs(data_threshed) < threshold] = 0.0
    data_q = np.round(data_threshed / lsb_step).astype(np.int16)

    # Wired-OR
    if verbose:
        active_frac = (data_q != 0).mean() * 100
        print(f"  [1/4] Wired-OR (wire_num={wire_num}, "
              f"threshold={threshold:.1f}uV, active={active_frac:.1f}%)...")
    wor_data = wired_or_fast(data_q.astype(np.float64), wire_num)

    # Detection
    if verbose:
        print(f"  [2/4] Spike detection (NEO thr={neo_threshold})...")
    detected = detect_spikes_reference(
        wor_data, n_rows, n_cols, fs=fs,
        amp_threshold=0, neo_threshold=neo_threshold,
        central_channel_check=True,
    )
    n_detected = int(detected.sum())
    if verbose:
        print(f"        Detected: {n_detected} spikes "
              f"(GT: {n_gt_spikes}, recall: {n_detected/max(n_gt_spikes,1)*100:.1f}%)")

    # Feature extraction
    if verbose:
        print(f"  [3/4] Feature extraction...")
    feat_x, feat_y, feat_t = extract_features_reference(
        wor_data, detected, n_rows, n_cols
    )
    n_features = sum(len(feat_t[r][c]) for r in range(n_rows) for c in range(n_cols))
    if verbose:
        print(f"        Features extracted: {n_features} spikes")

    # SOM classification
    if verbose:
        print(f"  [4/4] SOM training (beta={remove_scale})...")
    som = ReferenceSOM(n_rows, n_cols, remove_scale=remove_scale)
    history = som.train(feat_x, feat_y, feat_t, n_rows, n_cols,
                        fs=fs, batch_duration_s=6.0)

    n_clusters = int(som.valid_mem.sum())
    if verbose:
        print(f"        Converged: {som.converged}, "
              f"clusters: {n_clusters}, batches: {len(history)}")

    # Assignment
    cluster_idx, times, rows, cols = som.assign_all(
        feat_x, feat_y, feat_t, n_rows, n_cols
    )

    # Evaluate
    valid_mask = cluster_idx >= 0
    if valid_mask.sum() > 0:
        accuracy, n_matched, n_pred_clusters = evaluate_clustering(
            cluster_idx[valid_mask], times[valid_mask],
            gt_units, gt_times, tolerance_samples=int(fs * 0.001)  # 1ms
        )
    else:
        accuracy, n_matched, n_pred_clusters = 0.0, 0, 0

    if verbose:
        print(f"\n  === Results ===")
        print(f"  Clustering accuracy: {accuracy*100:.1f}%")
        print(f"  Matched spikes: {n_matched}/{n_gt_spikes}")
        print(f"  Predicted clusters: {n_pred_clusters} (GT: {n_gt_units})")

    return {
        "accuracy": accuracy,
        "n_matched": n_matched,
        "n_gt_spikes": n_gt_spikes,
        "n_gt_units": n_gt_units,
        "n_pred_clusters": n_pred_clusters,
        "n_detected": n_detected,
        "n_features": n_features,
        "converged": som.converged,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate 2D grid dataset with reference pipeline")
    parser.add_argument("--data-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "output" / "mearec_2d_grid")
    parser.add_argument("--snr", type=float, default=None, help="Validate specific SNR only")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Duration to process in seconds (default 10 for speed)")
    parser.add_argument("--wire-num", type=int, default=4)
    parser.add_argument("--lsb-step", type=int, default=4)
    parser.add_argument("--beta", type=int, default=8)
    parser.add_argument("--neo-threshold", type=float, default=40.0)
    args = parser.parse_args()

    data_dir = args.data_dir
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    # Find recordings
    recordings = sorted(data_dir.glob("grid2d_*/recording.h5"))
    if args.snr is not None:
        snr_int = int(args.snr) if args.snr == int(args.snr) else args.snr
        recordings = [r for r in recordings if f"snr{snr_int}" in str(r)]

    if not recordings:
        print(f"ERROR: No recordings found in {data_dir}")
        sys.exit(1)

    print(f"Found {len(recordings)} recordings in {data_dir}")
    print(f"Processing {args.duration}s of each recording")

    results = []
    for h5_path in recordings:
        result = run_validation(
            h5_path, args.duration,
            wire_num=args.wire_num,
            lsb_step=args.lsb_step,
            remove_scale=args.beta,
            neo_threshold=args.neo_threshold,
        )
        result["file"] = str(h5_path)
        results.append(result)

    # Summary table
    print(f"\n{'='*60}")
    print(f"{'SNR':>6} | {'Accuracy':>10} | {'Matched':>10} | {'Clusters':>10} | {'Detected':>10}")
    print(f"{'-'*60}")
    for r in results:
        snr_str = Path(r["file"]).parent.name.split("snr")[1].split("_")[0]
        print(f"{snr_str:>6} | {r['accuracy']*100:>9.1f}% | "
              f"{r['n_matched']:>5}/{r['n_gt_spikes']:<5} | "
              f"{r['n_pred_clusters']:>5}/{r['n_gt_units']:<5} | "
              f"{r['n_detected']:>10}")


if __name__ == "__main__":
    main()
