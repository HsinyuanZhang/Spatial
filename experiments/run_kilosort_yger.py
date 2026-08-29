"""Prepare a Yger juxta-cellular MEA recording and run Kilosort4.

Ground truth is the single patched ganglion cell. Scoring reports the KS
cluster that best coincides with juxta spike times.

Must be run in the CUDA-capable env (currently snn_cuda), on a free GPU.
GPU 1 may already be occupied; default device is cuda:0.

    python -m Spatial.data.prepare_yger_zenodo --recording 20160415_patch2
    python -m Spatial.experiments.run_kilosort_yger --recording 20160415_patch2 --device cuda:0 --nblocks 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.data.yger_zenodo import (
    DEFAULT_ARCHIVE_DIR,
    DEFAULT_WORK_DIR,
    FS,
    assert_on_hdd,
    coincidence_counts,
    kilosort_probe_dict,
    parse_mea256_prb,
    pin_tmpdir,
    prepare_recording,
)


def _load_ks_spikes(results_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    times_path = results_dir / "spike_times.npy"
    clu_path = results_dir / "spike_clusters.npy"
    if not times_path.exists() or not clu_path.exists():
        raise FileNotFoundError(f"Kilosort output missing under {results_dir}")
    times = np.load(times_path).astype(np.int64, copy=False).reshape(-1)
    clusters = np.load(clu_path).astype(np.int64, copy=False).reshape(-1)
    if times.shape != clusters.shape:
        raise ValueError(f"spike_times {times.shape} vs spike_clusters {clusters.shape}")
    return times, clusters


def score_against_juxta(
    spike_times: np.ndarray,
    spike_clusters: np.ndarray,
    gt_times: np.ndarray,
    tol_ms: float = 1.0,
    fs: int = FS,
) -> dict:
    tol = max(1, int(round(tol_ms * 1e-3 * fs)))
    rows = []
    for cluster_id in np.unique(spike_clusters):
        pred = spike_times[spike_clusters == cluster_id]
        stats = coincidence_counts(gt_times, pred, tol)
        stats["cluster_id"] = int(cluster_id)
        rows.append(stats)
    rows.sort(key=lambda r: (r["accuracy"], r["recall"], r["precision"]), reverse=True)
    best = rows[0] if rows else None
    return {
        "tol_ms": float(tol_ms),
        "tol_samples": int(tol),
        "n_ks_clusters": int(len(rows)),
        "n_ks_spikes": int(spike_times.size),
        "best": best,
        "top5": rows[:5],
    }


def run_kilosort4(
    bin_path: Path,
    probe: dict,
    results_dir: Path,
    device: str,
    nblocks: int,
    nt: int | None,
    extra_settings: dict | None = None,
) -> None:
    import torch
    from kilosort import run_kilosort

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "fs": FS,
        "n_chan_bin": int(probe["n_chan"]),
        "nblocks": int(nblocks),
    }
    if nt is not None:
        settings["nt"] = int(nt)
    if extra_settings:
        settings.update(extra_settings)
    run_kilosort(
        settings=settings,
        probe=probe,
        filename=str(bin_path),
        data_dir=str(bin_path.parent),
        results_dir=str(results_dir),
        data_dtype="int16",
        device=torch.device(device),
        save_preprocessed_copy=False,
        verbose_console=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", default="20160415_patch2")
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--nblocks",
        type=int,
        default=0,
        help="KS4 drift blocks. 0 disables correction (default for in-vitro MEA).",
    )
    parser.add_argument(
        "--nt",
        type=int,
        default=61,
        help="Template length in samples. Default 61 (~3 ms at 20 kHz).",
    )
    parser.add_argument("--tol-ms", type=float, default=1.0)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--tag", default="", help="Suffix for the results directory.")
    parser.add_argument("--th-learned", type=float, default=None)
    parser.add_argument("--th-universal", type=float, default=None)
    parser.add_argument("--ccg-threshold", type=float, default=None)
    parser.add_argument("--dminx", type=float, default=None)
    parser.add_argument("--max-channel-distance", type=float, default=None)
    parser.add_argument(
        "--x-centers",
        type=int,
        default=None,
        help="2D-grid clustering bins. KS4 docs: place a center every 200-300 um.",
    )
    parser.add_argument(
        "--bin-path",
        type=Path,
        default=None,
        help="Override binary (e.g. extra_int16_k64.bin). Probe must match.",
    )
    parser.add_argument(
        "--probe-json",
        type=Path,
        default=None,
        help="Override KS probe JSON when using --bin-path.",
    )
    parser.add_argument("--nearest-templates", type=int, default=None)
    args = parser.parse_args()

    assert_on_hdd(args.archive_dir, "archive_dir")
    pin_tmpdir(args.work_dir)
    meta = prepare_recording(
        args.recording, archive_dir=args.archive_dir, work_dir=args.work_dir
    )
    extra_settings = {}
    if args.th_learned is not None:
        extra_settings["Th_learned"] = float(args.th_learned)
    if args.th_universal is not None:
        extra_settings["Th_universal"] = float(args.th_universal)
    if args.ccg_threshold is not None:
        extra_settings["ccg_threshold"] = float(args.ccg_threshold)
    if args.dminx is not None:
        extra_settings["dminx"] = float(args.dminx)
    if args.max_channel_distance is not None:
        extra_settings["max_channel_distance"] = float(args.max_channel_distance)
    if args.x_centers is not None:
        extra_settings["x_centers"] = int(args.x_centers)
    if args.nearest_templates is not None:
        extra_settings["nearest_templates"] = int(args.nearest_templates)
    out_name = args.recording if not args.tag else f"{args.recording}_{args.tag}"
    results_dir = Path(args.work_dir) / "ks_out" / out_name
    summary_path = results_dir / "juxta_score.json"
    print(json.dumps({k: meta[k] for k in (
        "recording_id", "duration_s", "n_extra_channels", "n_juxta_spikes",
        "peak_channel_original", "peak_channel_ks", "bin_path",
    )}, indent=2), flush=True)

    if args.prepare_only:
        return 0

    if args.probe_json is not None:
        probe_obj = json.loads(Path(args.probe_json).read_text())
        probe = {
            "chanMap": np.asarray(probe_obj["chanMap"], dtype=np.int32),
            "xc": np.asarray(probe_obj["xc"], dtype=np.float64),
            "yc": np.asarray(probe_obj["yc"], dtype=np.float64),
            "kcoords": np.asarray(probe_obj["kcoords"], dtype=np.float64),
            "n_chan": int(probe_obj["n_chan"]),
        }
    else:
        probe = kilosort_probe_dict(parse_mea256_prb(args.archive_dir / "mea_256.prb"))
    bin_path = Path(args.bin_path) if args.bin_path is not None else Path(meta["bin_path"])
    if not args.score_only:
        print(
            f"running kilosort4 on {bin_path} device={args.device} "
            f"nblocks={args.nblocks} extra={extra_settings}",
            flush=True,
        )
        run_kilosort4(
            bin_path,
            probe,
            results_dir,
            device=args.device,
            nblocks=args.nblocks,
            nt=args.nt,
            extra_settings=extra_settings,
        )

    gt = np.load(meta["gt_path"])
    times, clusters = _load_ks_spikes(results_dir)
    score = score_against_juxta(times, clusters, gt["spike_times"], tol_ms=args.tol_ms)
    payload = {
        "recording_id": args.recording,
        "device": args.device,
        "nblocks": int(args.nblocks),
        "nt": args.nt,
        "tag": args.tag,
        "extra_settings": extra_settings,
        "bin_path": str(bin_path),
        "results_dir": str(results_dir),
        "gt_semantics": meta["gt_semantics"],
        "n_juxta_spikes": meta["n_juxta_spikes"],
        "peak_channel_original": meta["peak_channel_original"],
        "score": score,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2))
    best = score["best"]
    if best is None:
        print("no Kilosort clusters found", flush=True)
        return 1
    print(
        f"best cluster {best['cluster_id']}: "
        f"acc={best['accuracy']:.3f} rec={best['recall']:.3f} "
        f"prec={best['precision']:.3f} tp={best['tp']} fp={best['fp']} fn={best['fn']} "
        f"n_pred={best['n_pred']} / {score['n_ks_clusters']} clusters",
        flush=True,
    )
    print(f"wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
