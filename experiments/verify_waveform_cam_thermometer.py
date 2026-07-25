"""Large-scale oracle: waveform stage-2 CAM encodings (thermometer Hamming).

Stage-1 fixed: COM (float L2) ∧ P2P (4-bit maxnorm L1, τ≥1).
Stage-2 variants:
  - float L2 waveform (baseline)
  - |DC-removed| 4-bit L1
  - flat signed 2-bit Hamming (cliff control)
  - bipolar thermometer 4-cell × {uniform, quantile} Hamming
  - bipolar thermometer 4-cell quantile L1 (binary L1 ≡ Hamming)

Usage:
  python -m Spatial.experiments.verify_waveform_cam_thermometer --family all --duration 60
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.data.loader import list_hybrid_janelia_scenes, load_hybrid_janelia, load_mearec_npz
from Spatial.algorithms.detection import bandpass_filter, get_peak_amplitudes
from Spatial.algorithms.spatial_footprint import (
    build_knn_table_with_self,
    extract_local_p2p,
    com_features,
    quantize_p2p,
)
from Spatial.algorithms.two_stage_assign import (
    fit_unit_centroids,
    assign_nearest,
    calibrate_thresholds_codes,
    stage1_candidates,
    stage1_normalized_score,
    _intra_unit_distances,
)
from Spatial.algorithms.cim_thermometer import (
    remove_dc,
    fit_bipolar_thermometer,
    bipolar_thermometer_encode,
    abs_uniform_quantize,
    signed_uniform_quantize,
    hamming_distance_rows,
    l1_distance_rows,
)
from Spatial.experiments.run_two_stage_upper_bound import (
    _chronological_split,
    extract_main_channel_waveforms,
    normalize_waveforms,
)
from Spatial.experiments.run_hj16_method_sweep import select_16ch_scenes
from Spatial.experiments.run_mearec_method_sweep import DEFAULT_NPZ_DIR, discover_npz, parse_rec_name, mean_std


def oracle_acc(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(pred == y)) if len(y) else 0.0


def two_stage_wave_distance(
    com_tr,
    p2p_tr,
    wave_feat_tr,
    y_tr,
    com_te,
    p2p_te,
    wave_feat_te,
    *,
    wave_raw_tr=None,
    encode_centroid_fn=None,
    wave_metric: str = "l2",
    percentile: float = 95.0,
    tau_com: float | None = None,
    tau_p2p: float | None = None,
):
    """COM∧P2P gate then nearest wave feature (L2 / L1 / Hamming).

    If ``encode_centroid_fn`` is set, centroids are ``encode(mean(raw_wave))``
    (hardware-like: store analog mean, encode once). Else mean of ``wave_feat``.
    """
    com_cents = fit_unit_centroids(com_tr, y_tr)
    p2p_cents = fit_unit_centroids(p2p_tr, y_tr)

    if encode_centroid_fn is not None:
        if wave_raw_tr is None:
            raise ValueError("wave_raw_tr required when encode_centroid_fn is set")
        wave_cents = {}
        for u in np.unique(y_tr):
            mu = np.mean(wave_raw_tr[y_tr == u], axis=0)
            wave_cents[int(u)] = encode_centroid_fn(mu[None, :])[0]
    else:
        wave_cents = fit_unit_centroids(wave_feat_tr, y_tr)

    if tau_com is None:
        d_com = _intra_unit_distances(com_tr, y_tr, metric="l2")
        d_com = d_com[np.isfinite(d_com)]
        tau_com = float(np.percentile(d_com, percentile)) if d_com.size else 1.0
        tau_com = max(tau_com, 1e-8)
    if tau_p2p is None:
        # Assume p2p_tr already in code or float space; use L1 percentile floored
        if np.issubdtype(np.asarray(p2p_tr).dtype, np.integer) or np.all(
            np.asarray(p2p_tr) == np.rint(p2p_tr)
        ):
            tau_p2p = float(calibrate_thresholds_codes(p2p_tr, y_tr, percentile=percentile))
        else:
            d_p2p = _intra_unit_distances(p2p_tr, y_tr, metric="l1")
            d_p2p = d_p2p[np.isfinite(d_p2p)]
            tau_p2p = float(np.percentile(d_p2p, percentile)) if d_p2p.size else 1.0
            tau_p2p = max(tau_p2p, 1e-8)

    units = sorted(wave_cents.keys())
    cent_mat = np.stack([wave_cents[u] for u in units], axis=0)

    n = len(com_te)
    pred = np.full(n, -1, dtype=np.int64)
    n_fallback = 0
    n_cands = []

    for i in range(n):
        cands = stage1_candidates(
            com_te[i],
            p2p_te[i],
            com_cents,
            p2p_cents,
            float(tau_com),
            float(tau_p2p),
            com_metric="l2",
            p2p_metric="l1",
        )
        n_cands.append(len(cands))
        if not cands:
            n_fallback += 1
            # stage-1 fallback
            best_u, best_s = -1, np.inf
            for u in com_cents:
                if u not in p2p_cents:
                    continue
                s = stage1_normalized_score(
                    com_te[i], p2p_te[i], com_cents[u], p2p_cents[u],
                    float(tau_com), float(tau_p2p), p2p_metric="l1",
                )
                if s < best_s:
                    best_s, best_u = s, int(u)
            pred[i] = best_u
            continue

        q = wave_feat_te[i]
        sub_units = [u for u in cands if u in wave_cents]
        if not sub_units:
            n_fallback += 1
            pred[i] = cands[0]
            continue
        sub_mat = np.stack([wave_cents[u] for u in sub_units], axis=0)
        if wave_metric == "l2":
            dists = np.linalg.norm(sub_mat - q[None, :], axis=1)
        elif wave_metric == "l1":
            dists = np.sum(np.abs(sub_mat.astype(np.float64) - q.astype(np.float64)), axis=1)
        elif wave_metric == "hamming":
            dists = hamming_distance_rows(sub_mat, q)
        else:
            raise ValueError(wave_metric)
        pred[i] = int(sub_units[int(np.argmin(dists))])

    meta = {
        "tau_com": float(tau_com),
        "tau_p2p": float(tau_p2p),
        "fallback_rate": float(n_fallback / max(n, 1)),
        "mean_n_candidates": float(np.mean(n_cands)) if n_cands else 0.0,
        "wave_metric": wave_metric,
    }
    return pred, meta


def prepare_dataset(dataset, k_neighbors: int = 7, window: int = 15):
    filtered = bandpass_filter(dataset.raw_data, dataset.fs)
    _, times, central = get_peak_amplitudes(filtered, dataset.spike_times, window=window)
    t2u = dict(zip(dataset.spike_times.tolist(), dataset.spike_units.tolist()))
    labels = np.array([t2u.get(int(t), -1) for t in times], dtype=np.int64)
    keep = labels >= 0
    times, central, labels = times[keep], central[keep], labels[keep]

    k = min(k_neighbors, dataset.geom.shape[0])
    nbr = build_knn_table_with_self(dataset.geom, k)
    p2p_raw, _, neighbor_ids = extract_local_p2p(
        filtered, times, central, nbr, window=window
    )
    com = com_features(p2p_raw, neighbor_ids, dataset.geom)
    p2p_codes = quantize_p2p(p2p_raw, n_bits=4, mode="per_spike_max")
    wfs = extract_main_channel_waveforms(filtered, times, central)
    # Peak-normalize for float baseline comparability with prior studies
    wf_norm = normalize_waveforms(wfs)

    order = np.argsort(times)
    return {
        "com": com[order],
        "p2p_codes": p2p_codes[order],
        "wf_raw": wfs[order],
        "wf_norm": wf_norm[order],
        "labels": labels[order],
        "name": dataset.name,
        "n_units": int(len(np.unique(labels))),
    }


def run_one(dataset, percentile: float = 95.0, verbose: bool = True) -> list[dict]:
    data = prepare_dataset(dataset)
    com = data["com"]
    p2p = data["p2p_codes"]
    wf_raw = data["wf_raw"]
    wf_norm = data["wf_norm"]
    labels = data["labels"]
    tr, te = _chronological_split(len(labels), 0.5)
    y_tr, y_te = labels[tr], labels[te]

    # Shared stage-1 τ from train
    d_com = _intra_unit_distances(com[tr], y_tr, metric="l2")
    d_com = d_com[np.isfinite(d_com)]
    tau_com = max(float(np.percentile(d_com, percentile)), 1e-8)
    tau_p2p = float(calibrate_thresholds_codes(p2p[tr], y_tr, percentile=percentile))

    if verbose:
        print(
            f"  {data['name']}: n={len(labels)} units={data['n_units']} "
            f"τ_com={tau_com:.4g} τ_p2p={tau_p2p}"
        )

    rows = []

    def add(method: str, pred, meta, extra=None):
        acc = oracle_acc(pred, y_te)
        extra = extra or {}
        row = {
            "dataset": data["name"],
            "method": method,
            "oracle_acc": acc,
            "fallback_rate": meta.get("fallback_rate", ""),
            "mean_n_candidates": meta.get("mean_n_candidates", ""),
            "tau_com": tau_com,
            "tau_p2p": tau_p2p,
            "n_test": int(len(y_te)),
            "n_units": data["n_units"],
            "n_cells": extra.get("n_cells", ""),
            "thresh_mode": extra.get("thresh_mode", ""),
        }
        rows.append(row)
        if verbose:
            print(
                f"    {method:<32} acc={acc:.3f}  fb={meta.get('fallback_rate', float('nan')):.3f}"
            )

    # 1) float L2 waveform (normalized)
    pred, meta = two_stage_wave_distance(
        com[tr], p2p[tr], wf_norm[tr], y_tr,
        com[te], p2p[te], wf_norm[te],
        wave_metric="l2", tau_com=tau_com, tau_p2p=tau_p2p,
    )
    add("wave_float_l2", pred, meta)

    # 2) |DC| 4-bit L1
    scale_abs = float(np.percentile(np.abs(remove_dc(wf_raw[tr])), 99.0))
    abs4_tr = abs_uniform_quantize(wf_raw[tr], n_bits=4, scale=scale_abs).astype(np.float64)
    abs4_te = abs_uniform_quantize(wf_raw[te], n_bits=4, scale=scale_abs).astype(np.float64)
    pred, meta = two_stage_wave_distance(
        com[tr], p2p[tr], abs4_tr, y_tr,
        com[te], p2p[te], abs4_te,
        wave_metric="l1", tau_com=tau_com, tau_p2p=tau_p2p,
    )
    add("wave_abs4bit_l1", pred, meta)

    # 3) flat signed 2-bit Hamming (cliff control)
    scale_s = float(np.percentile(np.abs(remove_dc(wf_raw[tr])), 99.0))
    s2_tr = signed_uniform_quantize(wf_raw[tr], n_bits=2, scale=scale_s)
    s2_te = signed_uniform_quantize(wf_raw[te], n_bits=2, scale=scale_s)
    pred, meta = two_stage_wave_distance(
        com[tr], p2p[tr], s2_tr, y_tr,
        com[te], p2p[te], s2_te,
        wave_metric="hamming", tau_com=tau_com, tau_p2p=tau_p2p,
    )
    add("wave_signed2bit_hamming", pred, meta)

    # 4–5) bipolar thermometer 4-cell
    for mode in ("uniform", "quantile"):
        thr = fit_bipolar_thermometer(wf_raw[tr], n_cells=4, mode=mode)
        codes_tr = bipolar_thermometer_encode(wf_raw[tr], thr)
        codes_te = bipolar_thermometer_encode(wf_raw[te], thr)

        def _enc(batch, _thr=thr):
            return bipolar_thermometer_encode(batch, _thr)

        pred, meta = two_stage_wave_distance(
            com[tr], p2p[tr], codes_tr, y_tr,
            com[te], p2p[te], codes_te,
            wave_raw_tr=wf_raw[tr],
            encode_centroid_fn=_enc,
            wave_metric="hamming",
            tau_com=tau_com,
            tau_p2p=tau_p2p,
        )
        add(f"wave_therm4_{mode}_hamming", pred, meta, {"n_cells": 4, "thresh_mode": mode})

        # L1 on bits ≡ Hamming for {0,1}
        pred, meta = two_stage_wave_distance(
            com[tr], p2p[tr], codes_tr.astype(np.float64), y_tr,
            com[te], p2p[te], codes_te.astype(np.float64),
            wave_raw_tr=wf_raw[tr],
            encode_centroid_fn=lambda b, _thr=thr: bipolar_thermometer_encode(b, _thr).astype(np.float64),
            wave_metric="l1",
            tau_com=tau_com,
            tau_p2p=tau_p2p,
        )
        add(f"wave_therm4_{mode}_l1bits", pred, meta, {"n_cells": 4, "thresh_mode": mode})

    # 6) thermometer 2-cell quantile (fewer CAM cells)
    thr2 = fit_bipolar_thermometer(wf_raw[tr], n_cells=2, mode="quantile")
    c2_tr = bipolar_thermometer_encode(wf_raw[tr], thr2)
    c2_te = bipolar_thermometer_encode(wf_raw[te], thr2)
    pred, meta = two_stage_wave_distance(
        com[tr], p2p[tr], c2_tr, y_tr,
        com[te], p2p[te], c2_te,
        wave_raw_tr=wf_raw[tr],
        encode_centroid_fn=lambda b, _thr=thr2: bipolar_thermometer_encode(b, _thr),
        wave_metric="hamming",
        tau_com=tau_com,
        tau_p2p=tau_p2p,
    )
    add("wave_therm2_quantile_hamming", pred, meta, {"n_cells": 2, "thresh_mode": "quantile"})

    return rows


def print_summary(rows: list[dict]):
    methods = []
    for r in rows:
        if r["method"] not in methods:
            methods.append(r["method"])
    base = {
        r["dataset"]: r["oracle_acc"]
        for r in rows
        if r["method"] == "wave_float_l2"
    }
    print("\n=== Mean ± std oracle (vs float L2 Δpp) ===")
    for m in methods:
        vals = [r["oracle_acc"] for r in rows if r["method"] == m]
        deltas = [
            (r["oracle_acc"] - base[r["dataset"]]) * 100.0
            for r in rows
            if r["method"] == m and r["dataset"] in base
        ]
        mu, sd = mean_std(vals)
        dmu, _ = mean_std(deltas) if deltas else (float("nan"), float("nan"))
        print(f"  {m:<32} {mu:.3f} ± {sd:.3f}   Δ={dmu:+.2f}pp")


def main():
    parser = argparse.ArgumentParser(description="Waveform CAM thermometer large-scale oracle")
    parser.add_argument("--family", choices=["all", "hj", "mearec"], default="all")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--percentile", type=float, default=95.0)
    parser.add_argument("--npz-dir", type=Path, default=DEFAULT_NPZ_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Spatial/output/waveform_cam_thermometer"),
    )
    args = parser.parse_args()
    verbose = not args.quiet
    args.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    all_rows: list[dict] = []

    if args.family in ("all", "hj"):
        scenes = select_16ch_scenes(list_hybrid_janelia_scenes(ready_only=True))
        print(f"HJ scenes: {len(scenes)}")
        hj_rows = []
        for e in scenes:
            short = e.get("short_name") or e["scene_key"]
            print(f"\n########## HJ {short} ##########")
            ds = load_hybrid_janelia(scene=short, duration_s=args.duration)
            rows = run_one(ds, percentile=args.percentile, verbose=verbose)
            for r in rows:
                r["family"] = "hj"
                r["scene"] = short
            hj_rows.extend(rows)
        out = args.output_dir / f"wave_cam_hj16_{int(args.duration)}s.csv"
        fields = list(hj_rows[0].keys())
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(hj_rows)
        print(f"\nSaved: {out}")
        print_summary(hj_rows)
        all_rows.extend(hj_rows)

    if args.family in ("all", "mearec"):
        paths = discover_npz(args.npz_dir, seeds=args.seeds)
        print(f"\nMEArec NPZs: {len(paths)}")
        me_rows = []
        for path in paths:
            info = parse_rec_name(path.stem)
            print(f"\n########## MEArec {info['rec']} ##########")
            ds = load_mearec_npz(path, duration_s=args.duration)
            rows = run_one(ds, percentile=args.percentile, verbose=verbose)
            for r in rows:
                r["family"] = "mearec"
                r["scene"] = info["rec"]
                r["snr"] = info["snr"]
                r["n_units_design"] = info["n_units_design"]
            me_rows.extend(rows)
        out = args.output_dir / f"wave_cam_mearec_{int(args.duration)}s.csv"
        # Ensure optional MEArec keys present on all rows
        for r in me_rows:
            r.setdefault("snr", "")
            r.setdefault("n_units_design", "")
        fields = list(me_rows[0].keys())
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(me_rows)
        print(f"\nSaved: {out}")
        print_summary(me_rows)
        all_rows.extend(me_rows)

    print(f"\nTotal elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
