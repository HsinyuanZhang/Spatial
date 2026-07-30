"""Shared helpers for the geometric / CiM prefilter multi-method pilot."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from Spatial.algorithms.adaptive_range_search import (
    AdaptiveRangeSearch,
    evaluate_candidate_search,
)
from Spatial.algorithms.detection import bandpass_filter, get_peak_amplitudes
from Spatial.algorithms.geometric_prefilters import (
    causal_first_channel,
    cycles_estimate,
    energy_estimate,
    extract_event_snippets,
    fit_channel_mad_thresholds,
    fit_cosine_radii,
    fit_unit_home_tiles,
    index_neighborhood_unit_mask,
    jaccard_filter_mask,
    match_stream_first_detect,
    median_channel_spacing,
    proxy_max_channel,
    prune_candidates_topk,
    query_cosine_radius,
    rank_fusion_topk,
    rising_edge_first_channel,
    stream_rising_edge_detections,
    tile_activation_mask,
    tile_ids_from_positions,
    trigger_channel_masks,
    cosine_distance,
)
from Spatial.algorithms.spatial_footprint import com_features, extract_local_p2p, build_knn_table_with_self
from Spatial.data.loader import Dataset
from Spatial.experiments.run_adaptive_5bit_range_search import (
    WAVEFORM_LENGTH,
    WAVEFORM_PRE,
    WAVEFORM_POST,
    build_directory_mask,
    candidate_waveform_assign,
    fit_home_channels,
    fit_unit_channel_support,
    summarize_configuration,
)
from Spatial.experiments.run_causal_temporal_sketch import (
    FILTER_HIGH_HZ,
    FILTER_LOW_HZ,
    FILTER_ORDER,
    prepare_causal_events,
    stable_three_way_split,
)


BITS = 5
DESCRIPTOR_DIM = 9
WAVEFORM_TEMPLATE_BITS = WAVEFORM_LENGTH * BITS
RADIUS_PERCENTILES = (95.0, 99.0, 99.9)
NEIGHBOR_RADII = (1, 2, 3, 4)
TILE_RADII = (1, 2, 3)
TOP_KS = (3, 5, 7, 10)
JACCARD_THRESHOLDS = (0.2, 0.3, 0.4, 0.5)
SRAM_PJ = 100.0
CIM_PJ = 2.0


def prepare_pilot_bundle(dataset: Dataset, *, k_neighbors: int = 7) -> dict[str, Any]:
    """Causal D9 events plus multi-channel snippets and peak amplitudes for A2/B/C."""
    prepared = prepare_causal_events(dataset, k_neighbors=k_neighbors)
    filtered = bandpass_filter(
        dataset.raw_data, dataset.fs, FILTER_LOW_HZ, FILTER_HIGH_HZ, FILTER_ORDER
    )
    amps, peak_times, central = get_peak_amplitudes(
        filtered, dataset.spike_times, window=WAVEFORM_PRE
    )
    # Align amplitudes to prepare_causal_events rows via times.
    time_to_amp = {int(t): amps[i] for i, t in enumerate(peak_times)}
    times = np.asarray(prepared["times"], dtype=np.int64)
    peak_amps = np.stack([time_to_amp[int(t)] for t in times])
    # Sanity: central from prepare should match proxy_max on these amps.
    proxy = proxy_max_channel(peak_amps)
    snippets = extract_event_snippets(
        filtered, times, pre=WAVEFORM_PRE, post=WAVEFORM_POST
    )
    table = build_knn_table_with_self(dataset.geom, min(k_neighbors, dataset.n_channels))
    p2p, _, neighbors = extract_local_p2p(
        filtered, times, prepared["central"], table, window=WAVEFORM_PRE
    )
    com = com_features(p2p, neighbors, dataset.geom)
    fit_idx, cal_idx, test_idx = stable_three_way_split(times)
    return {
        **prepared,
        "filtered": filtered,
        "peak_amplitudes": peak_amps,
        "snippets": snippets,
        "com_xy": com,
        "geom": np.asarray(dataset.geom, dtype=np.float64),
        "n_channels": int(dataset.n_channels),
        "fit_idx": fit_idx,
        "cal_idx": cal_idx,
        "test_idx": test_idx,
        "proxy_vs_central_agreement": float(np.mean(proxy == np.asarray(prepared["central"]))),
        "waveforms": prepared["raw_waveforms"],
    }


def _fit_search(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_cal: np.ndarray,
    y_cal: np.ndarray,
    percentile: float,
) -> AdaptiveRangeSearch:
    model = AdaptiveRangeSearch(
        n_bits=BITS, fixed_unit_interval=True, radius_percentile=float(percentile)
    ).fit(x_fit, y_fit)
    # Recalibrate radii on calibration only (same pattern as temporal pilot).
    codes_cal = model.transform(x_cal)
    index = {
        (u.item() if isinstance(u, np.generic) else u): i for i, u in enumerate(model.units_)
    }
    radii = model.radii_.copy()
    for unit, row in index.items():
        local = codes_cal[np.asarray(y_cal) == unit]
        if local.size:
            radii[row] = int(
                np.ceil(
                    np.percentile(
                        np.abs(local - model.centroids_[row]).sum(axis=1),
                        float(percentile),
                        method="linear",
                    )
                )
            )
    model.radii_ = radii
    return model


def _waveform_centroids(
    waveforms: np.ndarray, labels: np.ndarray, units: np.ndarray
) -> dict[Any, np.ndarray]:
    out: dict[Any, np.ndarray] = {}
    for unit in units:
        rows = waveforms[labels == unit]
        if rows.size:
            out[unit.item() if isinstance(unit, np.generic) else unit] = rows.mean(axis=0)
    return out


def _traffic_bits(mean_candidates: float, n_units: int) -> dict[str, float]:
    full = float(n_units) * WAVEFORM_TEMPLATE_BITS
    actual = float(mean_candidates) * WAVEFORM_TEMPLATE_BITS
    return {
        "waveform_template_bits_per_unit": float(WAVEFORM_TEMPLATE_BITS),
        "full_scan_bits_per_event": full,
        "candidate_bits_per_event": actual,
        "waveform_read_reduction_factor": full / max(actual, 1e-12),
    }


def evaluate_config(
    *,
    method: str,
    config: dict[str, Any],
    model: AdaptiveRangeSearch,
    x_test: np.ndarray,
    y_test: np.ndarray,
    waveforms_test: np.ndarray,
    wave_centroids: dict[Any, np.ndarray],
    active_mask: np.ndarray,
    score: str = "l1",
    cosine_radii: np.ndarray | None = None,
    top_k: int | None = None,
    n_tiles_total: int | None = None,
    n_tiles_active_mean: float | None = None,
    memory_bits_extra_per_unit: int = 0,
    fallback_to_max_rate: float | None = None,
    effective_radius_mean: float | None = None,
) -> dict[str, Any]:
    """Run Level-1 + candidate waveform assign; emit one CSV-ready row."""
    codes = model.transform(x_test)
    units = model.units_
    n_units = int(units.size)
    if score == "l1":
        candidates, diag = model.query(
            x_test, mode="l1", active_mask=active_mask, ensure_nonempty=True
        )
        pool_recall_report = evaluate_candidate_search(candidates, y_test, diag)
        if top_k is not None:
            candidates = prune_candidates_topk(
                candidates, codes, model.centroids_, units, int(top_k), metric="l1"
            )
            diag = {
                **diag,
                "active_row_count": np.asarray(
                    [len(c) for c in candidates], dtype=np.int64
                ),
            }
        cycle_score = "l1"
        macs_per_row = DESCRIPTOR_DIM
    elif score == "cosine":
        if cosine_radii is None:
            raise ValueError("cosine_radii required for score=cosine")
        candidates, diag = query_cosine_radius(
            codes,
            model.centroids_,
            units,
            cosine_radii,
            active_mask=active_mask,
            ensure_nonempty=True,
        )
        pool_recall_report = evaluate_candidate_search(candidates, y_test, diag)
        if top_k is not None:
            candidates = prune_candidates_topk(
                candidates, codes, model.centroids_, units, int(top_k), metric="cosine"
            )
        cycle_score = "cosine"
        macs_per_row = DESCRIPTOR_DIM
    elif score == "fusion":
        l1_dist = np.abs(codes[:, None, :] - model.centroids_[None, :, :]).sum(axis=2)
        cos_dist = cosine_distance(codes, model.centroids_)
        k = int(top_k) if top_k is not None else 10
        candidates = rank_fusion_topk(
            l1_dist, cos_dist, active_mask, units, k, cosine_weight=0.7, l1_weight=0.3
        )
        diag = {
            "widen_level": np.zeros(len(candidates), dtype=np.int64),
            "fallback": np.zeros(len(candidates), dtype=bool),
            "active_row_count": active_mask.sum(axis=1).astype(np.int64),
            "row_comparisons": active_mask.sum(axis=1).astype(np.int64),
            "fallback_row_comparisons": np.zeros(len(candidates), dtype=np.int64),
            "total_row_comparisons_including_fallback": active_mask.sum(axis=1).astype(
                np.int64
            ),
        }
        pool_recall_report = evaluate_candidate_search(candidates, y_test, diag)
        cycle_score = "cosine"
        macs_per_row = DESCRIPTOR_DIM * 2
    else:
        raise ValueError(f"unknown score {score}")

    pred, wave_meta = candidate_waveform_assign(
        candidates,
        waveforms_test,
        wave_centroids,
        codes,
        model.centroids_,
        units,
        metric="l1",
    )
    summary = summarize_configuration(
        candidates, y_test, diag, active_mask, units, pred, wave_meta
    )
    metrics = evaluate_candidate_search(candidates, y_test, diag)
    mean_c = float(metrics["mean_candidates"])
    mean_active = float(metrics.get("mean_active_rows", mean_c))
    traffic = _traffic_bits(mean_c, n_units)
    cycles = cycles_estimate(
        n_active_rows_mean=mean_active, n_dims=DESCRIPTOR_DIM, score=cycle_score
    )
    energy = energy_estimate(
        n_events=int(y_test.size),
        n_sram_lookups_per_event=1.0,
        n_cim_macs_per_event=mean_active * macs_per_row,
    )
    row_bits = int(model.l1_row_bits_full_scale) + int(memory_bits_extra_per_unit)
    out = {
        "method": method,
        **{f"cfg_{k}": v for k, v in config.items()},
        "score": score,
        "n_test": int(y_test.size),
        "n_units": n_units,
        "true_unit_candidate_recall": float(metrics["true_unit_candidate_recall"]),
        "pool_recall_before_topk": float(pool_recall_report["true_unit_candidate_recall"]),
        "mean_candidates": mean_c,
        "p95_candidates": float(metrics["p95_candidates"]),
        "mean_active_rows": mean_active,
        "zero_fraction": float(metrics["zero_fraction"]),
        "final_waveform_accuracy": float(summary["final_waveform_accuracy"]),
        "waveform_read_reduction_factor": float(traffic["waveform_read_reduction_factor"]),
        "candidate_bits_per_event": float(traffic["candidate_bits_per_event"]),
        "search_payload_bits_per_unit": row_bits,
        "memory_bits_total": row_bits * n_units,
        "ops_per_event": float(cycles["ops_per_event"]),
        "energy_per_event_pj": float(energy["energy_per_event_pj"]),
        "fraction_tiles_active": (
            None
            if n_tiles_total is None or n_tiles_active_mean is None
            else float(n_tiles_active_mean) / max(float(n_tiles_total), 1.0)
        ),
        "fallback_to_max_rate": fallback_to_max_rate,
        "effective_radius_mean": effective_radius_mean,
    }
    return out


def run_dataset_sweep(
    dataset_id: str,
    family: str,
    condition: str,
    bundle: dict[str, Any],
    *,
    methods: Sequence[str],
) -> list[dict[str, Any]]:
    """Sweep requested methods on one prepared recording."""
    fit_idx = bundle["fit_idx"]
    cal_idx = bundle["cal_idx"]
    test_idx = bundle["test_idx"]
    desc = np.asarray(bundle["descriptor"], dtype=np.float64)
    labels = np.asarray(bundle["labels"])
    waves = np.asarray(bundle["waveforms"], dtype=np.float64)
    central = np.asarray(bundle["central"], dtype=np.int64)
    peak_amps = np.asarray(bundle["peak_amplitudes"], dtype=np.float64)
    snippets = np.asarray(bundle["snippets"], dtype=np.float64)
    com_xy = np.asarray(bundle["com_xy"], dtype=np.float64)
    geom = np.asarray(bundle["geom"], dtype=np.float64)
    n_channels = int(bundle["n_channels"])
    times = np.asarray(bundle["times"], dtype=np.int64)

    x_fit, y_fit = desc[fit_idx], labels[fit_idx]
    x_cal, y_cal = desc[cal_idx], labels[cal_idx]
    x_test, y_test = desc[test_idx], labels[test_idx]
    w_fit, w_test = waves[fit_idx], waves[test_idx]

    # Fit MAD thresholds on fit-time sample span.
    t_fit = times[fit_idx]
    thr = fit_channel_mad_thresholds(
        bundle["filtered"],
        sample_start=int(t_fit.min()) - WAVEFORM_PRE,
        sample_stop=int(t_fit.max()) + WAVEFORM_POST,
    )
    a1_all = proxy_max_channel(peak_amps)
    a2_legacy_all, fb_legacy_all = causal_first_channel(
        snippets, thr, fallback_channels=a1_all
    )
    a2_rise_all, fb_rise_all, _rel_rise = rising_edge_first_channel(
        snippets, thr, pre_samples=WAVEFORM_PRE, fallback_channels=a1_all
    )
    det_t, det_ch = stream_rising_edge_detections(
        bundle["filtered"], thr, refractory_samples=WAVEFORM_PRE
    )
    a2_stream_all, fb_stream_all, _rel_stream = match_stream_first_detect(
        times,
        det_t,
        det_ch,
        pre_samples=WAVEFORM_PRE,
        post_samples=5,
        fallback_channels=a1_all,
    )
    triggers_all = trigger_channel_masks(
        snippets, thr, peak_amplitudes=peak_amps, top_k_fallback=3
    )

    rows: list[dict[str, Any]] = []
    method_set = {m.upper() for m in methods}

    for percentile in RADIUS_PERCENTILES:
        model = _fit_search(x_fit, y_fit, x_cal, y_cal, percentile)
        units = model.units_
        homes = fit_home_channels(y_fit, central[fit_idx], units)
        wave_centroids = _waveform_centroids(w_fit, y_fit, units)
        codes_cal = model.transform(x_cal)
        cos_radii = fit_cosine_radii(
            codes_cal, y_cal, model.centroids_, units, percentile
        )

        def base_meta(method: str, cfg: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
            return {
                "dataset": dataset_id,
                "family": family,
                "condition": condition,
                "radius_percentile": float(percentile),
                **result,
            }

        # ---- Baseline ----
        if "BASELINE" in method_set or "ALL" in method_set:
            for dir_mode, mask in (
                (
                    "all",
                    np.ones((test_idx.size, units.size), dtype=bool),
                ),
                (
                    "mode_knn7",
                    build_directory_mask(
                        central[test_idx], units, homes, geom, directory_neighbors=7
                    ),
                ),
            ):
                result = evaluate_config(
                    method="baseline",
                    config={"directory": dir_mode},
                    model=model,
                    x_test=x_test,
                    y_test=y_test,
                    waveforms_test=w_test,
                    wave_centroids=wave_centroids,
                    active_mask=mask,
                    score="l1",
                )
                rows.append(base_meta("baseline", {}, result))

        # ---- Scheme A ----
        if "A" in method_set or "ALL" in method_set:
            for anchor_name, anchors, fb_rate in (
                ("A1_proxy_max", a1_all[test_idx], 0.0),
                (
                    "A2_legacy_window_hot",
                    a2_legacy_all[test_idx],
                    float(fb_legacy_all[test_idx].mean()),
                ),
                (
                    "A2_rising_edge",
                    a2_rise_all[test_idx],
                    float(fb_rise_all[test_idx].mean()),
                ),
                (
                    "A2_stream_detect",
                    a2_stream_all[test_idx],
                    float(fb_stream_all[test_idx].mean()),
                ),
            ):
                for n in NEIGHBOR_RADII:
                    mask, eff = index_neighborhood_unit_mask(
                        homes, anchors, n_channels, n, ensure_nonempty=True
                    )
                    result = evaluate_config(
                        method=anchor_name,
                        config={"neighbor_radius": int(n)},
                        model=model,
                        x_test=x_test,
                        y_test=y_test,
                        waveforms_test=w_test,
                        wave_centroids=wave_centroids,
                        active_mask=mask,
                        score="l1",
                        fallback_to_max_rate=fb_rate,
                        effective_radius_mean=float(eff.mean()),
                    )
                    rows.append(base_meta(anchor_name, {}, result))

        # ---- Scheme D ----
        if "D" in method_set or "ALL" in method_set:
            full = np.ones((test_idx.size, units.size), dtype=bool)
            result = evaluate_config(
                method="D_cosine",
                config={},
                model=model,
                x_test=x_test,
                y_test=y_test,
                waveforms_test=w_test,
                wave_centroids=wave_centroids,
                active_mask=full,
                score="cosine",
                cosine_radii=cos_radii,
                memory_bits_extra_per_unit=16,  # precomputed ||c||^2
            )
            rows.append(base_meta("D_cosine", {}, result))
            result = evaluate_config(
                method="D_fusion",
                config={"top_k": 10},
                model=model,
                x_test=x_test,
                y_test=y_test,
                waveforms_test=w_test,
                wave_centroids=wave_centroids,
                active_mask=full,
                score="fusion",
                top_k=10,
                memory_bits_extra_per_unit=16,
            )
            rows.append(base_meta("D_fusion", {}, result))

        # ---- Scheme B / E ----
        pitch = 2.0 * median_channel_spacing(geom)
        unit_tiles = fit_unit_home_tiles(y_fit, com_xy[fit_idx], units, pitch)
        # Query tile from trigger-centroid geom.
        trig = triggers_all[test_idx]
        query_tiles = np.empty(test_idx.size, dtype=np.int64)
        for i in range(test_idx.size):
            chs = np.flatnonzero(trig[i])
            pos = geom[chs].mean(axis=0)
            query_tiles[i] = int(tile_ids_from_positions(pos[None, :], pitch)[0])
        all_tiles = np.unique(
            np.concatenate(
                [unit_tiles, query_tiles, tile_ids_from_positions(geom, pitch)]
            )
        )
        n_tiles_total = int(all_tiles.size)

        if "B" in method_set or "ALL" in method_set:
            for tr in TILE_RADII:
                mask = tile_activation_mask(query_tiles, unit_tiles, tr)
                active_tiles = []
                for i in range(test_idx.size):
                    active_tiles.append(
                        len(
                            {
                                int(unit_tiles[j])
                                for j in range(units.size)
                                if mask[i, j]
                            }
                        )
                    )
                mean_tiles = float(np.mean(active_tiles)) if active_tiles else 0.0
                result = evaluate_config(
                    method="B_tile",
                    config={"tile_radius": int(tr), "top_k": None},
                    model=model,
                    x_test=x_test,
                    y_test=y_test,
                    waveforms_test=w_test,
                    wave_centroids=wave_centroids,
                    active_mask=mask,
                    score="l1",
                    n_tiles_total=n_tiles_total,
                    n_tiles_active_mean=mean_tiles,
                    memory_bits_extra_per_unit=8,
                )
                rows.append(base_meta("B_tile", {}, result))
                for k in TOP_KS:
                    result = evaluate_config(
                        method="B_tile_topk",
                        config={"tile_radius": int(tr), "top_k": int(k)},
                        model=model,
                        x_test=x_test,
                        y_test=y_test,
                        waveforms_test=w_test,
                        wave_centroids=wave_centroids,
                        active_mask=mask,
                        score="l1",
                        top_k=int(k),
                        n_tiles_total=n_tiles_total,
                        n_tiles_active_mean=mean_tiles,
                        memory_bits_extra_per_unit=8,
                    )
                    rows.append(base_meta("B_tile_topk", {}, result))

        if "E" in method_set or "ALL" in method_set:
            for tr in TILE_RADII:
                mask = tile_activation_mask(query_tiles, unit_tiles, tr)
                active_tiles = []
                for i in range(test_idx.size):
                    active_tiles.append(
                        len(
                            {
                                int(unit_tiles[j])
                                for j in range(units.size)
                                if mask[i, j]
                            }
                        )
                    )
                mean_tiles = float(np.mean(active_tiles)) if active_tiles else 0.0
                result = evaluate_config(
                    method="E_power_gate",
                    config={"tile_radius": int(tr)},
                    model=model,
                    x_test=x_test,
                    y_test=y_test,
                    waveforms_test=w_test,
                    wave_centroids=wave_centroids,
                    active_mask=mask,
                    score="l1",
                    n_tiles_total=n_tiles_total,
                    n_tiles_active_mean=mean_tiles,
                    memory_bits_extra_per_unit=8,
                )
                rows.append(base_meta("E_power_gate", {}, result))

        # ---- Scheme C ----
        if "C" in method_set or "ALL" in method_set:
            support = fit_unit_channel_support(
                y_fit, central[fit_idx], units, n_channels, coverage=0.95
            )
            event_masks = triggers_all[test_idx]
            for tau in JACCARD_THRESHOLDS:
                mask = jaccard_filter_mask(event_masks, support, tau)
                # Ensure nonempty: if empty, activate all (widen).
                for i in range(mask.shape[0]):
                    if not mask[i].any():
                        mask[i] = True
                result = evaluate_config(
                    method="C_jaccard",
                    config={"jaccard_tau": float(tau)},
                    model=model,
                    x_test=x_test,
                    y_test=y_test,
                    waveforms_test=w_test,
                    wave_centroids=wave_centroids,
                    active_mask=mask,
                    score="l1",
                    memory_bits_extra_per_unit=n_channels,  # support bitmask
                )
                rows.append(base_meta("C_jaccard", {}, result))

        # ---- Hybrid A+D (true first-detect pool → cosine) ----
        if "HYBRID" in method_set or "ALL" in method_set:
            for n in NEIGHBOR_RADII:
                mask, eff = index_neighborhood_unit_mask(
                    homes, a2_stream_all[test_idx], n_channels, n, ensure_nonempty=True
                )
                result = evaluate_config(
                    method="hybrid_A2stream_D",
                    config={"neighbor_radius": int(n)},
                    model=model,
                    x_test=x_test,
                    y_test=y_test,
                    waveforms_test=w_test,
                    wave_centroids=wave_centroids,
                    active_mask=mask,
                    score="cosine",
                    cosine_radii=cos_radii,
                    fallback_to_max_rate=float(fb_stream_all[test_idx].mean()),
                    effective_radius_mean=float(eff.mean()),
                    memory_bits_extra_per_unit=16,
                )
                rows.append(base_meta("hybrid_A2stream_D", {}, result))

    return rows


def _event_weighted_mean(rows: list[dict[str, Any]], key: str) -> float:
    n = sum(int(r["n_test"]) for r in rows)
    if n == 0:
        return 0.0
    return sum(float(r[key]) * int(r["n_test"]) for r in rows) / n


def evaluate_gates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pre-registered G-A / G-D / G-drift / G-E from development rows."""
    by_family = defaultdict(list)
    for r in rows:
        by_family[r["family"]].append(r)

    def family_method_rows(family: str, method_prefix: str) -> list[dict[str, Any]]:
        return [
            r
            for r in by_family.get(family, [])
            if str(r["method"]).startswith(method_prefix)
        ]

    # G-A: some A config recall>=0.99 and mean_C<10 on both families
    def best_a_pass(family: str) -> dict[str, Any]:
        cands = [
            r
            for r in by_family.get(family, [])
            if str(r["method"]).startswith("A1_") or str(r["method"]).startswith("A2_")
        ]
        ok = [
            r
            for r in cands
            if float(r["true_unit_candidate_recall"]) >= 0.99
            and float(r["mean_candidates"]) < 10.0
        ]
        # Pareto vs mode_knn7 at p99.9
        mode_refs = [
            r
            for r in by_family.get(family, [])
            if r["method"] == "baseline"
            and r.get("cfg_directory") == "mode_knn7"
            and float(r["radius_percentile"]) == 99.9
        ]
        dominate = []
        if mode_refs:
            ref = mode_refs[0]
            for r in cands:
                if float(r["radius_percentile"]) != 99.9:
                    continue
                better_recall = float(r["true_unit_candidate_recall"]) >= float(
                    ref["true_unit_candidate_recall"]
                )
                better_c = float(r["mean_candidates"]) <= float(ref["mean_candidates"])
                strict = (
                    float(r["true_unit_candidate_recall"])
                    > float(ref["true_unit_candidate_recall"])
                    or float(r["mean_candidates"]) < float(ref["mean_candidates"])
                )
                if better_recall and better_c and strict:
                    dominate.append(r)
        return {
            "strict_pass_rows": len(ok),
            "pareto_dominate_rows": len(dominate),
            "pass": bool(ok) or bool(dominate),
            "best_recall": max(
                (float(r["true_unit_candidate_recall"]) for r in cands), default=0.0
            ),
            "best_mean_c_at_high_recall": min(
                (
                    float(r["mean_candidates"])
                    for r in cands
                    if float(r["true_unit_candidate_recall"]) >= 0.99
                ),
                default=None,
            ),
        }

    ga_hj = best_a_pass("hj")
    ga_me = best_a_pass("mearec")
    g_a = {
        "hj": ga_hj,
        "mearec": ga_me,
        "pass": bool(ga_hj["pass"] and ga_me["pass"]),
    }

    # G-D: cosine within 1pp of L1 at matched mean |C|
    def gd_family(family: str) -> dict[str, Any]:
        l1_rows = [
            r
            for r in by_family.get(family, [])
            if r["method"] == "baseline" and r.get("cfg_directory") == "all"
        ]
        cos_rows = [
            r for r in by_family.get(family, []) if r["method"] == "D_cosine"
        ]
        matches = []
        for cos in cos_rows:
            target_c = float(cos["mean_candidates"])
            # Find L1 row with closest mean_c within 10% relative.
            best = None
            best_gap = None
            for l1 in l1_rows:
                lc = float(l1["mean_candidates"])
                if lc <= 0:
                    continue
                rel = abs(lc - target_c) / lc
                if rel <= 0.10 and (best_gap is None or rel < best_gap):
                    best_gap = rel
                    best = l1
            if best is None:
                continue
            delta_pp = (
                float(cos["true_unit_candidate_recall"])
                - float(best["true_unit_candidate_recall"])
            ) * 100.0
            matches.append({"delta_pp": delta_pp, "rel_c": best_gap})
        if not matches:
            return {"pass": False, "reason": "no matched mean_|C| pair", "matches": []}
        worst = min(m["delta_pp"] for m in matches)
        return {
            "pass": worst >= -1.0,
            "worst_delta_pp": worst,
            "n_matched": len(matches),
        }

    gd_hj = gd_family("hj")
    gd_me = gd_family("mearec")
    g_d = {"hj": gd_hj, "mearec": gd_me, "pass": bool(gd_hj["pass"] and gd_me["pass"])}

    # G-drift: true first-detect (stream / rising) vs A1 on drift vs static
    def drift_gap_for(method: str) -> dict[str, Any]:
        drift_gaps = []
        static_gaps = []
        for percentile in RADIUS_PERCENTILES:
            for n in NEIGHBOR_RADII:
                def pick(condition: str, meth: str) -> dict[str, Any] | None:
                    for r in rows:
                        if (
                            r["family"] == "hj"
                            and r["condition"] == condition
                            and r["method"] == meth
                            and float(r["radius_percentile"]) == float(percentile)
                            and int(r.get("cfg_neighbor_radius", -1)) == int(n)
                        ):
                            return r
                    return None

                a1_d = pick("drift", "A1_proxy_max")
                a2_d = pick("drift", method)
                a1_s = pick("static", "A1_proxy_max")
                a2_s = pick("static", method)
                if a1_d and a2_d:
                    drift_gaps.append(
                        (
                            float(a2_d["true_unit_candidate_recall"])
                            - float(a1_d["true_unit_candidate_recall"])
                        )
                        * 100.0
                    )
                if a1_s and a2_s:
                    static_gaps.append(
                        (
                            float(a2_s["true_unit_candidate_recall"])
                            - float(a1_s["true_unit_candidate_recall"])
                        )
                        * 100.0
                    )
        return {
            "pass": bool(drift_gaps)
            and max(drift_gaps) >= 2.0
            and bool(static_gaps)
            and min(static_gaps) >= -1.0,
            "max_drift_A2_minus_A1_pp": max(drift_gaps) if drift_gaps else None,
            "min_static_A2_minus_A1_pp": min(static_gaps) if static_gaps else None,
        }

    g_drift = {
        "A2_stream_detect": drift_gap_for("A2_stream_detect"),
        "A2_rising_edge": drift_gap_for("A2_rising_edge"),
        "A2_legacy_window_hot": drift_gap_for("A2_legacy_window_hot"),
    }
    g_drift["pass"] = bool(
        g_drift["A2_stream_detect"]["pass"] or g_drift["A2_rising_edge"]["pass"]
    )

    # G-E: tile fraction <= 25% and recall >= 0.99 on HJ
    e_ok = [
        r
        for r in by_family.get("hj", [])
        if r["method"] == "E_power_gate"
        and r.get("fraction_tiles_active") is not None
        and float(r["fraction_tiles_active"]) <= 0.25
        and float(r["true_unit_candidate_recall"]) >= 0.99
    ]
    g_e = {
        "pass": bool(e_ok),
        "n_pass_rows": len(e_ok),
        "best_fraction": min(
            (
                float(r["fraction_tiles_active"])
                for r in by_family.get("hj", [])
                if r["method"] == "E_power_gate" and r.get("fraction_tiles_active") is not None
            ),
            default=None,
        ),
    }

    return {
        "G_A": g_a,
        "G_D": g_d,
        "G_drift": g_drift,
        "G_E": g_e,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys: list[str] = sorted({k for r in rows for k in r})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in keys})
