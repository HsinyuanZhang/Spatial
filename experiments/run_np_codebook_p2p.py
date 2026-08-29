"""P2P / POSNEG WTA among the frozen NP ``(home, 3-bit residual)`` codebook.

COM unique-ID is occupancy-limited on sim_no_drift. The selected Level-1
object is already a candidate list (recall 0.992, mean |C|~26, 46.5x).
This diagnostic asks whether a cheap spatial descriptor can unique-ID
inside that list. Not a STAR-Mem joint-gate result. Streams the Figshare
zip prefix; nothing is copied off /mnt/data.

    python -m Spatial.experiments.run_np_codebook_p2p --duration 60
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from Spatial.algorithms.detection import bandpass_filter
from Spatial.algorithms.spatial_footprint import (
    build_radius_neighbor_table,
    extract_local_extrema,
    extract_local_p2p,
    posneg_codes,
)
from Spatial.data.ks4_sim import load_ks4_sim_no_drift
from Spatial.experiments.analyze_np_com_lowbit import quantize_span
from Spatial.experiments.candidate_list_refine import (
    codebook_candidate_lists,
    integer_mean_templates,
    scatter_to_channels,
    summarize_refine,
    unit_mean_templates,
    wta_among_lists,
)
from Spatial.experiments.run_com_ks4sim_nodrift import (
    DEFAULT_OUT as COM_OUT,
    FILTER_HIGH_HZ,
    FILTER_LOW_HZ,
    WINDOW,
    prepare_ks4sim_com_events,
)
from Spatial.experiments.spatial_cim_common import write_json


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "Spatial" / "output" / "np_codebook_p2p"
CODE_BITS = 3
CODE_HALF_UM = 90.0
CODE_DELTA = 1
P2P_BITS = 5
POSNEG_BITS = 4


def _extract_footprint(
    filtered: np.ndarray,
    times: np.ndarray,
    homes: np.ndarray,
    table: np.ndarray,
    window: int,
) -> dict[str, np.ndarray]:
    p2p, p2p_times, nbr = extract_local_p2p(filtered, times, homes, table, window=window)
    if not np.array_equal(p2p_times, times):
        raise AssertionError("P2P row alignment changed")
    vmax, vmin, _imax, _imin, ext_times, ext_nbr = extract_local_extrema(
        filtered, times, homes, table, window=window
    )
    if not np.array_equal(ext_times, times) or not np.array_equal(ext_nbr, nbr):
        raise AssertionError("extrema row alignment changed")
    return {"p2p": p2p, "vmax": vmax, "vmin": vmin, "neighbor_ids": nbr}


def prepare_np_footprint(
    duration_s: float,
    *,
    cache_dir: Path,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    cache_dir = Path(cache_dir)
    com_cache = cache_dir / f"events_{int(duration_s)}s.npz"
    foot_cache = cache_dir / f"events_{int(duration_s)}s_footprint.npz"
    com = prepare_ks4sim_com_events(
        duration_s, cache_path=com_cache, verbose=verbose
    )
    if foot_cache.exists():
        payload = np.load(foot_cache, allow_pickle=False)
        if verbose:
            print(f"Loaded footprint cache: {foot_cache}")
        out = {k: com[k] for k in com}
        out.update({k: payload[k] for k in payload.files})
        return out
    ds = load_ks4_sim_no_drift(duration_s=duration_s)
    if verbose:
        print(f"Bandpass {ds.n_channels}ch {ds.duration_s:.1f}s for P2P/POSNEG")
    filtered = bandpass_filter(ds.raw_data, ds.fs, FILTER_LOW_HZ, FILTER_HIGH_HZ)
    radius = float(np.asarray(com["radius_um"]))
    table = build_radius_neighbor_table(ds.geom, radius)
    foot = _extract_footprint(
        filtered, com["times"], com["homes"], table, window=WINDOW
    )
    np.savez_compressed(foot_cache, **foot)
    if verbose:
        print(f"Saved footprint cache: {foot_cache}")
    out = {k: com[k] for k in com}
    out.update(foot)
    return out


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
    if integer:
        tmpl = integer_mean_templates(train_feat, y_train, units)
    else:
        tmpl = unit_mean_templates(train_feat, y_train, units)
    wta = wta_among_lists(query, tmpl, units, lists, metric=metric)
    summary = summarize_refine(wta["pred"], y_test, lists, n_units_train=int(units.size))
    summary["method"] = name
    summary["metric"] = metric
    summary["integer_templates"] = bool(integer)
    summary["feature_dim"] = int(query.shape[1])
    return summary


def run(duration_s: float, out_dir: Path, verbose: bool = True) -> list[dict[str, Any]]:
    prep = prepare_np_footprint(duration_s, cache_dir=COM_OUT, verbose=verbose)
    times = prep["times"]
    labels = prep["labels"]
    homes = prep["homes"]
    geom = prep["geom"]
    n_ch = int(np.asarray(prep["n_channels"]))
    rel_y = prep["com_um"][:, 1] - geom[homes, 1]
    codes = quantize_span(rel_y, CODE_BITS, -CODE_HALF_UM, CODE_HALF_UM).ravel()
    source = codebook_candidate_lists(
        homes, codes, labels, times, train_frac=0.5, code_delta=CODE_DELTA
    )
    lists = source["lists"]
    tr = source["train_index"]
    te = source["test_index"]
    y_tr = labels[tr]
    y_te = labels[te]
    units = source["units"]
    p2p_ch = scatter_to_channels(prep["p2p"], prep["neighbor_ids"], n_ch)
    posneg = posneg_codes(prep["vmax"], prep["vmin"], prep["neighbor_ids"], n_bits=POSNEG_BITS)
    k = int(prep["neighbor_ids"].shape[1])
    pos_ch = scatter_to_channels(posneg[:, :k], prep["neighbor_ids"], n_ch)
    neg_ch = scatter_to_channels(posneg[:, k:], prep["neighbor_ids"], n_ch)
    posneg_ch = np.concatenate([pos_ch, neg_ch], axis=1)
    p2p_s5 = scatter_to_channels(
        _per_event_codes(prep["p2p"], prep["neighbor_ids"], P2P_BITS),
        prep["neighbor_ids"],
        n_ch,
    )
    com_um = prep["com_um"]
    rows: list[dict[str, Any]] = []
    majority = summarize_refine(
        source["majority_pred"], y_te, lists, n_units_train=int(units.size)
    )
    majority.update(
        {
            "method": "codebook_majority",
            "metric": "majority",
            "integer_templates": False,
            "feature_dim": 0,
        }
    )
    rows.append(majority)
    rows.append(_eval_arm("com_xy_l2", com_um[te], com_um[tr], y_tr, y_te, units, lists, metric="l2"))
    rows.append(_eval_arm("p2p_float_l1", p2p_ch[te], p2p_ch[tr], y_tr, y_te, units, lists))
    rows.append(
        _eval_arm(
            "p2p_s5_l1",
            p2p_s5[te],
            p2p_s5[tr],
            y_tr,
            y_te,
            units,
            lists,
            integer=True,
        )
    )
    rows.append(
        _eval_arm(
            "posneg_s4_l1",
            posneg_ch[te],
            posneg_ch[tr],
            y_tr,
            y_te,
            units,
            lists,
            integer=True,
        )
    )
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
    csv_path = out_dir / "np_codebook_p2p.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        out_dir / "np_codebook_p2p_summary.json",
        {
            "dataset": "ks4sim_no_drift",
            "duration_s": float(duration_s),
            "codebook": {
                "bits": CODE_BITS,
                "half_um": CODE_HALF_UM,
                "delta": CODE_DELTA,
            },
            "n_test": int(y_te.size),
            "n_units_train": int(units.size),
            "n_bins_train": int(source["n_bins_train"]),
            "rows": rows,
        },
    )
    if verbose:
        print(f"Saved {csv_path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if float(args.duration) != 60.0:
        parser.error("duration locked to 60 s to match the COM codebook")
    run(float(args.duration), Path(args.output_dir), verbose=not bool(args.quiet))


if __name__ == "__main__":
    main()
