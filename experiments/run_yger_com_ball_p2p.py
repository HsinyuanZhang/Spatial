"""P2P / POSNEG WTA among the Yger 3-pitch COM radius ball.

Frozen ball: global R=70 µm on analog COM (candidate recall 0.990, mean
|C|~29, 12.4x vs 358 train units). COM WTA inside that ball is the 55%
unique-ID already reported. This diagnostic asks whether P2P/POSNEG can
unique-ID the remaining packed neighbours. Labels are KS4 Th=13 clusters.
Raw stays memmapped on /mnt/data.

    python -m Spatial.experiments.run_yger_com_ball_p2p --duration 60
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from Spatial.algorithms.spatial_footprint import (
    build_radius_neighbor_table,
    extract_local_extrema,
    posneg_codes,
)
from Spatial.algorithms.two_stage_assign import fit_unit_centroids
from Spatial.data.loader import YGER_KS4_NPZ_PATH, load_yger_ks4
from Spatial.data.yger_zenodo import DEFAULT_KS4_TH13_DIR
from Spatial.experiments.candidate_list_refine import (
    integer_mean_templates,
    scatter_to_channels,
    summarize_refine,
    unit_mean_templates,
    wta_among_lists,
)
from Spatial.experiments.run_com_mainchannel_lowbit import (
    WINDOW,
    align_ks4_spike_positions,
    fill_missing_seed_xy,
    prepare_com_mainchannel_events,
)
from Spatial.experiments.spatial_cim_common import write_json


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "Spatial" / "output" / "yger_com_ball_p2p"
BALL_RADIUS_UM = 70.0
TRAIN_FRAC = 0.5
POSNEG_BITS = 4
P2P_BITS = 5


def _com_ball_lists(
    com_um: np.ndarray,
    labels: np.ndarray,
    times: np.ndarray,
    radius_um: float,
    train_frac: float,
) -> dict[str, Any]:
    t = np.asarray(times, dtype=np.int64).ravel()
    order = np.argsort(t, kind="stable")
    xy = np.asarray(com_um, dtype=np.float64)[order]
    y = np.asarray(labels).ravel()[order]
    n = int(y.size)
    n_train = max(1, int(n * train_frac))
    if n_train >= n:
        n_train = max(1, n - 1)
    tr = np.arange(n_train)
    te = np.arange(n_train, n)
    cents = fit_unit_centroids(xy[tr], y[tr])
    units = np.array(sorted(cents.keys()), dtype=np.int64)
    mat = np.stack([cents[int(u)] for u in units], axis=0)
    dist = np.linalg.norm(xy[te][:, None, :] - mat[None, :, :], axis=2)
    keep = dist <= float(radius_um)
    lists = [units[row] for row in keep]
    orig_train = order[tr]
    orig_test = order[te]
    return {
        "lists": lists,
        "units": units,
        "train_index": orig_train,
        "test_index": orig_test,
        "y_test": y[te],
        "centroids": mat,
    }


def _per_event_codes(values: np.ndarray, neighbor_ids: np.ndarray, n_bits: int) -> np.ndarray:
    amp = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    valid = np.asarray(neighbor_ids, dtype=np.int64) >= 0
    peak = np.where(valid, amp, 0.0).max(axis=1, keepdims=True)
    peak = np.maximum(peak, 1e-12)
    max_level = float((1 << int(n_bits)) - 1)
    codes = np.clip(np.rint(max_level * amp / peak), 0.0, max_level)
    return np.where(valid, codes, 0.0)


def _eval_arm(
    name: str,
    query: np.ndarray,
    train_feat: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    units: np.ndarray,
    lists: list[np.ndarray],
    *,
    integer: bool = False,
    metric: str = "l1",
) -> dict[str, Any]:
    tmpl = (
        integer_mean_templates(train_feat, y_train, units)
        if integer
        else unit_mean_templates(train_feat, y_train, units)
    )
    wta = wta_among_lists(query, tmpl, units, lists, metric=metric)
    summary = summarize_refine(wta["pred"], y_test, lists, n_units_train=int(units.size))
    summary["method"] = name
    summary["metric"] = metric
    summary["integer_templates"] = bool(integer)
    summary["feature_dim"] = int(query.shape[1])
    return summary


def run(
    duration_s: float,
    *,
    npz_path: Path,
    ks4_dir: Path,
    out_dir: Path,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    ds = load_yger_ks4(path=npz_path, duration_s=duration_s, good_only=False)
    seed_xy = fill_missing_seed_xy(
        align_ks4_spike_positions(ds.spike_times, ds.spike_units, ks4_dir),
        ds.spike_units,
    )
    prep = prepare_com_mainchannel_events(
        ds,
        same_spike_max_pitches=3.0,
        seed_xy=seed_xy,
        return_filtered=True,
    )
    filtered = prep["filtered"]
    table_width = int(prep["neighbor_ids"].shape[1])
    n_ch = int(ds.n_channels)
    radius = float(prep["meta"]["radius_um"])
    table = build_radius_neighbor_table(ds.geom, radius)
    vmax, vmin, _imax, _imin, ext_times, ext_nbr = extract_local_extrema(
        filtered, prep["times"], prep["central"], table, window=WINDOW
    )
    if not np.array_equal(ext_times, prep["times"]):
        raise AssertionError("extrema dropped events")
    geom_min = np.min(ds.geom, axis=0)
    geom_span = np.maximum(np.max(ds.geom, axis=0) - geom_min, 1e-8)
    com_um = prep["com"] * geom_span + geom_min
    source = _com_ball_lists(
        com_um, prep["labels"], prep["times"], BALL_RADIUS_UM, TRAIN_FRAC
    )
    lists = source["lists"]
    tr = source["train_index"]
    te = source["test_index"]
    y_tr = prep["labels"][tr]
    y_te = prep["labels"][te]
    units = source["units"]
    p2p_ch = scatter_to_channels(prep["p2p"], prep["neighbor_ids"], n_ch)
    posneg = posneg_codes(vmax, vmin, ext_nbr, n_bits=POSNEG_BITS)
    k = table_width
    posneg_ch = np.concatenate(
        [
            scatter_to_channels(posneg[:, :k], ext_nbr, n_ch),
            scatter_to_channels(posneg[:, k:], ext_nbr, n_ch),
        ],
        axis=1,
    )
    p2p_s5 = scatter_to_channels(
        _per_event_codes(prep["p2p"], prep["neighbor_ids"], P2P_BITS),
        prep["neighbor_ids"],
        n_ch,
    )
    rows = [
        _eval_arm("com_xy_l2", com_um[te], com_um[tr], y_tr, y_te, units, lists, metric="l2"),
        _eval_arm("p2p_float_l1", p2p_ch[te], p2p_ch[tr], y_tr, y_te, units, lists),
        _eval_arm("p2p_s5_l1", p2p_s5[te], p2p_s5[tr], y_tr, y_te, units, lists, integer=True),
        _eval_arm(
            "posneg_s4_l1", posneg_ch[te], posneg_ch[tr], y_tr, y_te, units, lists, integer=True
        ),
    ]
    com_mu = com_um[tr].mean(axis=0)
    com_sd = np.maximum(com_um[tr].std(axis=0), 1e-8)
    p2p_n = p2p_ch / np.maximum(p2p_ch.max(axis=1, keepdims=True), 1e-8)
    combo = np.concatenate([(com_um - com_mu) / com_sd, p2p_n], axis=1)
    rows.append(_eval_arm("com_p2p_l1", combo[te], combo[tr], y_tr, y_te, units, lists))
    if verbose:
        for row in rows:
            print(
                f"  {row['method']:<16} acc={row['argmin_accuracy']:.4f} "
                f"inC={row['argmin_in_candidates']:.4f} "
                f"R={row['candidate_recall']:.4f} C={row['mean_candidates']:.2f} "
                f"x={row['traffic_reduction_x']:.1f}"
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "yger_com_ball_p2p.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        out_dir / "yger_com_ball_p2p_summary.json",
        {
            "dataset": ds.name,
            "duration_s": float(duration_s),
            "ball_radius_um": BALL_RADIUS_UM,
            "n_test": int(y_te.size),
            "n_units_train": int(units.size),
            "rows": rows,
        },
    )
    if verbose:
        print(f"Saved {csv_path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--npz", type=Path, default=YGER_KS4_NPZ_PATH)
    parser.add_argument("--ks4-dir", type=Path, default=DEFAULT_KS4_TH13_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if float(args.duration) != 60.0:
        parser.error("duration locked to 60 s to match the COM-ball source")
    run(
        float(args.duration),
        npz_path=Path(args.npz),
        ks4_dir=Path(args.ks4_dir),
        out_dir=Path(args.output_dir),
        verbose=not bool(args.quiet),
    )


if __name__ == "__main__":
    main()
