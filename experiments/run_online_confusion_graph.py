"""Causal online ceiling for a static 5-bit indexed confusion graph.

Centroids, radii, and quantization are fit-only and remain static.  Only the
anchor→label observation history and its pointer graph can update after a test
block has been predicted.  ``oracle_delayed_label`` is explicitly an
un-deployable delayed-label ceiling; ``self_predicted_label`` never reads GT.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.algorithms.adaptive_range_search import AdaptiveRangeSearch
from Spatial.data.loader import Dataset
from Spatial.experiments.run_adaptive_5bit_range_search import (
    REPO_ROOT, _chronological_split, _iter_hj_datasets, _iter_mearec_datasets,
    _write_csv, candidate_waveform_assign, prepare_dataset_events,
)
from Spatial.experiments.run_certified_dual_range_pipeline import chronological_fraction_split
from Spatial.experiments.run_indexed_confusion_candidate_search import (
    RankingMode, build_anchor_graph, graph_storage_bits, rank_unit_rows,
    stable_anchor_union_candidates, summarize_indexed_candidates,
)
from Spatial.experiments.run_mearec_method_sweep import DEFAULT_NPZ_DIR

UpdateSource = Literal["oracle_delayed_label", "self_predicted_label"]
HistoryMode = Literal["accumulate", "rolling"]


def block_event_slices(times: np.ndarray, block_samples: int) -> list[np.ndarray]:
    """Group sorted events by sample-time block without splitting equal timestamps."""
    event_times = np.asarray(times, dtype=np.int64).ravel()
    if int(block_samples) < 1:
        raise ValueError("block_samples must be positive")
    if event_times.size == 0:
        return []
    if np.any(event_times[1:] < event_times[:-1]):
        raise ValueError("times must already be stably sorted")
    ids = event_times // int(block_samples)
    starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
    ends = np.r_[starts[1:], event_times.size]
    return [np.arange(start, end) for start, end in zip(starts, ends)]


def _rankings_with_anchor_first(anchors: np.ndarray, n_units: int) -> np.ndarray:
    """Make valid rank rows whose first entry is the supplied anchor index."""
    base = np.tile(np.arange(n_units, dtype=np.int64), (len(anchors), 1))
    for row, anchor in enumerate(np.asarray(anchors, dtype=np.int64)):
        position = int(np.flatnonzero(base[row] == anchor)[0])
        base[row, 0], base[row, position] = base[row, position], base[row, 0]
    return base


def graph_from_observations(
    observations: Sequence[tuple[int, Any, int]], units: np.ndarray, coverage: float
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Rebuild the minimal graph from (anchor, label, timestamp) history."""
    if observations:
        anchors = np.asarray([row[0] for row in observations], dtype=np.int64)
        labels = np.asarray([row[1] for row in observations])
        graph, diagnostics = build_anchor_graph(
            _rankings_with_anchor_first(anchors, len(units)), labels, units, coverage=coverage, include_anchor=True
        )
    else:
        # ``build_anchor_graph`` itself provides [anchor] for every no-event row.
        graph, diagnostics = build_anchor_graph(
            np.empty((0, len(units)), dtype=np.int64), np.empty(0, dtype=units.dtype), units,
            coverage=coverage, include_anchor=True,
        )
    return graph, {**diagnostics, **graph_storage_bits(graph, len(units))}


def pointer_delta(
    previous: Sequence[np.ndarray] | None,
    current: Sequence[np.ndarray],
    pointer_bits: int,
    count_field_bits: int = 0,
) -> dict[str, int]:
    """Account membership lower bound and physical ordered-list pointer rewrites.

    The legacy additions/removals fields intentionally remain set-membership
    lower bounds.  The new position fields instead treat each anchor list as a
    contiguous ordered pointer array: any changed/inserted/removed slot writes
    one pointer field.  A changed list length additionally rewrites exactly
    one explicit count-field metadata word.
    """
    keys = (
        "pointer_additions", "pointer_removals", "pointer_bit_rewrites",
        "pointer_position_content_rewrites", "pointer_position_content_bit_rewrites",
        "pointer_count_field_metadata_rewrites", "pointer_count_field_metadata_bit_rewrites",
        "pointer_ordered_list_total_bit_rewrites_including_count_metadata",
    )
    if previous is None:
        return {key: 0 for key in keys}
    if len(previous) != len(current):
        raise ValueError("previous and current graph must have the same anchor-row count")
    additions = 0
    removals = 0
    position_rewrites = 0
    count_rewrites = 0
    for old, new in zip(previous, current):
        additions += sum(not any(value == before for before in old) for value in new)
        removals += sum(not any(value == after for after in new) for value in old)
        common = min(len(old), len(new))
        position_rewrites += sum(old[index] != new[index] for index in range(common))
        position_rewrites += abs(len(old) - len(new))
        count_rewrites += int(len(old) != len(new))
    membership_bits = int((additions + removals) * int(pointer_bits))
    position_bits = int(position_rewrites * int(pointer_bits))
    count_bits = int(count_rewrites * int(count_field_bits))
    return {
        "pointer_additions": int(additions), "pointer_removals": int(removals),
        # Compatibility field: membership-only lower-bound rewrite payload.
        "pointer_bit_rewrites": membership_bits,
        "pointer_position_content_rewrites": int(position_rewrites),
        "pointer_position_content_bit_rewrites": position_bits,
        "pointer_count_field_metadata_rewrites": int(count_rewrites),
        "pointer_count_field_metadata_bit_rewrites": count_bits,
        "pointer_ordered_list_total_bit_rewrites_including_count_metadata": int(position_bits + count_bits),
    }


def post_update_graph_transition(
    previous_graph: Sequence[np.ndarray] | None,
    history: Sequence[tuple[int, Any, int]],
    units: np.ndarray,
    coverage: float,
) -> tuple[list[np.ndarray], dict[str, Any], dict[str, int]]:
    """Materialize the graph after an update, including a final non-query state.

    Online queries use the graph reconstructed before their own block.  This
    helper accounts the transition caused by the block's delayed update even
    when there is no following query block (the final state is explicitly
    reported as post-update/non-query rather than silently omitted).
    """
    graph, info = graph_from_observations(history, units, coverage)
    delta = pointer_delta(
        previous_graph, graph, int(info["graph_pointer_id_bits"]), int(info["graph_count_field_bits"])
    )
    return graph, info, delta


def expire_observations(history: deque[tuple[int, Any, int]], reference_time: int, horizon_samples: int) -> int:
    """Expire observations strictly older than the rolling time horizon."""
    threshold = int(reference_time) - int(horizon_samples)
    removed = 0
    while history and history[0][2] < threshold:
        history.popleft()
        removed += 1
    return removed


def update_observations(
    history: deque[tuple[int, Any, int]], anchors: np.ndarray, labels: Sequence[Any] | np.ndarray,
    times: np.ndarray, units: np.ndarray, *, mode: HistoryMode, block_end_time: int, horizon_samples: int,
) -> tuple[int, int]:
    """Append post-prediction observations, then expire rolling history by time."""
    lookup = {value.item() if isinstance(value, np.generic) else value for value in units}
    unknown = 0
    for anchor, label, event_time in zip(anchors, labels, times):
        key = label.item() if isinstance(label, np.generic) else label
        if key in lookup:
            history.append((int(anchor), key, int(event_time)))
        else:
            unknown += 1
    removed = 0
    if mode == "rolling":
        removed = expire_observations(history, block_end_time, horizon_samples)
    return unknown, removed


def _waveform_centroids(waveforms: np.ndarray, labels: np.ndarray, units: np.ndarray) -> dict[Any, np.ndarray]:
    return {unit.item() if isinstance(unit, np.generic) else unit: waveforms[labels == unit].mean(axis=0) for unit in units}


def _block_subset_metrics(candidates: list[np.ndarray], labels: np.ndarray, pred: np.ndarray, reads: np.ndarray, fallback: np.ndarray, n_units: int) -> dict[str, float]:
    if not labels.size:
        return {"recall": 0.0, "waveform_reduction_factor": float("inf")}
    diag = {"row_comparisons": np.full(labels.size, n_units, dtype=np.int64), "active_row_count": np.full(labels.size, n_units, dtype=np.int64)}
    report = summarize_indexed_candidates(candidates, labels, diag, pred, {"candidate_waveform_reads": reads, "waveform_fallback": fallback}, n_units=n_units)
    return {"recall": float(report["true_unit_candidate_recall"]), "waveform_reduction_factor": float(report["global_waveform_reduction_factor"])}


def run_dataset_online_confusion_graph(
    dataset: Dataset, *, bits: int = 5, k_neighbors: int = 7,
    ranking_modes: Sequence[RankingMode] = ("raw_l1", "radius_normalized_l1"),
    graph_coverages: Sequence[float] = (0.95, 0.99, 1.0), top_anchors: Sequence[int] = (1, 2),
    block_seconds: Sequence[float] = (1.0, 5.0), update_sources: Sequence[UpdateSource] = ("oracle_delayed_label", "self_predicted_label"),
    histories: Sequence[HistoryMode] = ("accumulate", "rolling"), rolling_horizon_seconds: float = 10.0,
    fit_fraction: float = 2.0 / 3.0, waveform_metric: str = "l1", verbose: bool = True,
) -> list[dict[str, Any]]:
    """Run static graph and causal post-block graph-only updates for one recording."""
    prepared = prepare_dataset_events(dataset, k_neighbors=k_neighbors)
    train_idx, test_idx = _chronological_split(prepared["times"])
    fit_rel, cal_rel = chronological_fraction_split(prepared["times"][train_idx], fit_fraction)
    fit_idx, cal_idx = train_idx[fit_rel], train_idx[cal_rel]
    model = AdaptiveRangeSearch(n_bits=bits, fixed_unit_interval=True, radius_percentile=95.0, box_coverage=1.0).fit(prepared["descriptor"][fit_idx], prepared["labels"][fit_idx])
    units = model.units_
    u = len(units)
    cal_codes, test_codes = model.transform(prepared["descriptor"][cal_idx]), model.transform(prepared["descriptor"][test_idx])
    wave_centroids = _waveform_centroids(prepared["waveforms"][train_idx], prepared["labels"][train_idx], units)
    test_times, test_labels = prepared["times"][test_idx], prepared["labels"][test_idx]
    if verbose:
        print(f"{dataset.name}: train/fit/cal/test={len(train_idx)}/{len(fit_idx)}/{len(cal_idx)}/{len(test_idx)}, U={u}; static 5-bit centroids/radii, graph-only online ceiling")
    rows: list[dict[str, Any]] = []

    def base_row(method: str, mode: str, coverage: float, anchors: int, block_s: float | None, source: str | None, history: str | None, metrics: dict[str, Any], extra: dict[str, Any]) -> None:
        rows.append({
            "dataset": dataset.name, "n_events": int(len(prepared["labels"])), "n_channels": int(dataset.n_channels),
            "n_train": int(len(train_idx)), "n_fit": int(len(fit_idx)), "n_calibration": int(len(cal_idx)), "n_test": int(len(test_idx)), "fit_fraction_of_train": float(fit_fraction),
            "bits": int(bits), "descriptor_dim": int(prepared["descriptor"].shape[1]), "k_neighbors": int(min(k_neighbors, dataset.n_channels)), "n_units_fit": int(u),
            "method": method, "ranking_mode": mode, "graph_coverage": float(coverage), "top_anchors": int(anchors), "block_seconds": "" if block_s is None else float(block_s),
            "update_label_source": "" if source is None else source, "oracle_update_source": bool(source == "oracle_delayed_label"), "history_mode": "" if history is None else history,
            "rolling_horizon_seconds": float(rolling_horizon_seconds), "centroid_radius_policy": "static_fit_only_graph_updates_only", "no_ppa_claim": True,
            "waveform_centroid_training_rows": int(len(train_idx)), "waveform_centroid_source": "raw_train_fit_plus_calibration", **metrics, **extra,
        })

    for mode in dict.fromkeys(ranking_modes):
        cal_rank = rank_unit_rows(cal_codes, model.centroids_, model.radii_, mode=mode)
        test_rank = rank_unit_rows(test_codes, model.centroids_, model.radii_, mode=mode)
        initial_observations = deque((int(anchor), label.item() if isinstance(label, np.generic) else label, int(time)) for anchor, label, time in zip(cal_rank[:, 0], prepared["labels"][cal_idx], prepared["times"][cal_idx]))
        for coverage in dict.fromkeys(float(v) for v in graph_coverages):
            initial_graph, initial_graph_info = graph_from_observations(initial_observations, units, coverage)
            for anchors in dict.fromkeys(int(v) for v in top_anchors):
                static_candidates = stable_anchor_union_candidates(test_rank, initial_graph, anchors)
                diag = {"row_comparisons": np.full(len(test_idx), u, dtype=np.int64), "active_row_count": np.full(len(test_idx), u, dtype=np.int64), "widen_level": np.zeros(len(test_idx), dtype=np.int64), "fallback": np.zeros(len(test_idx), dtype=bool)}
                static_pred, static_wave = candidate_waveform_assign(static_candidates, prepared["waveforms"][test_idx], wave_centroids, test_codes, model.centroids_, units, metric=waveform_metric)
                static_metrics = summarize_indexed_candidates(static_candidates, test_labels, diag, static_pred, static_wave, n_units=u)
                base_row("static_graph", mode, coverage, anchors, None, None, None, static_metrics, {
                    **initial_graph_info, "block_count": 0,
                    "pointer_additions": 0, "pointer_removals": 0, "pointer_bit_rewrites": 0,
                    "pointer_position_content_rewrites": 0, "pointer_position_content_bit_rewrites": 0,
                    "pointer_count_field_metadata_rewrites": 0, "pointer_count_field_metadata_bit_rewrites": 0,
                    "pointer_ordered_list_total_bit_rewrites_including_count_metadata": 0,
                    "pointer_rewrite_accounting_scope": "ordered_contiguous_pointer_slots_plus_count_metadata; membership_fields_are_lower_bound; static_no_updates",
                    "pointer_rewrite_includes_final_post_update_state": False,
                    "pointer_graph_final_post_update_total_bits": int(initial_graph_info["graph_total_bits"]),
                    "pointer_graph_final_post_update_total_pointers": int(initial_graph_info["graph_total_pointers"]),
                    "label_counter_observation_writes": 0, "online_unknown_fit_label_count": 0,
                    "first_block_recall": None, "post_first_block_recall": None,
                    "first_block_waveform_reduction_factor": None, "post_first_block_waveform_reduction_factor": None,
                })
                for block_s in dict.fromkeys(float(v) for v in block_seconds):
                    slices = block_event_slices(test_times, max(1, int(round(block_s * dataset.fs))))
                    for source in dict.fromkeys(update_sources):
                        for history_mode in dict.fromkeys(histories):
                            history = deque(initial_observations)
                            previous_graph: list[np.ndarray] | None = None
                            all_candidates: list[np.ndarray] = []
                            all_pred: list[Any] = []
                            all_reads: list[int] = []
                            all_fallback: list[bool] = []
                            graph_bits: list[int] = []
                            graph_pointers: list[int] = []
                            adds = removes = rewrites = position_rewrites = position_bits = 0
                            count_rewrites = count_bits = ordered_total_bits = unknown = writes = 0
                            first_size = len(slices[0]) if slices else 0
                            for block in slices:
                                # Causality: graph is built before this block's labels/predictions update history.
                                if history_mode == "rolling":
                                    expire_observations(history, int(test_times[block[0]]), max(1, int(round(rolling_horizon_seconds * dataset.fs))))
                                graph, graph_info = graph_from_observations(history, units, coverage)
                                delta = pointer_delta(previous_graph, graph, int(graph_info["graph_pointer_id_bits"]), int(graph_info["graph_count_field_bits"]))
                                adds += delta["pointer_additions"]; removes += delta["pointer_removals"]; rewrites += delta["pointer_bit_rewrites"]
                                position_rewrites += delta["pointer_position_content_rewrites"]; position_bits += delta["pointer_position_content_bit_rewrites"]
                                count_rewrites += delta["pointer_count_field_metadata_rewrites"]; count_bits += delta["pointer_count_field_metadata_bit_rewrites"]
                                ordered_total_bits += delta["pointer_ordered_list_total_bit_rewrites_including_count_metadata"]
                                previous_graph = graph
                                graph_bits.append(int(graph_info["graph_total_bits"])); graph_pointers.append(int(graph_info["graph_total_pointers"]))
                                block_candidates = stable_anchor_union_candidates(test_rank[block], graph, anchors)
                                block_pred, block_wave = candidate_waveform_assign(block_candidates, prepared["waveforms"][test_idx][block], wave_centroids, test_codes[block], model.centroids_, units, metric=waveform_metric)
                                all_candidates.extend(block_candidates); all_pred.extend(block_pred.tolist()); all_reads.extend(block_wave["candidate_waveform_reads"].tolist()); all_fallback.extend(block_wave["waveform_fallback"].tolist())
                                update_labels = test_labels[block] if source == "oracle_delayed_label" else block_pred
                                block_unknown, _ = update_observations(history, test_rank[block, 0], update_labels, test_times[block], units, mode=history_mode, block_end_time=int(test_times[block[-1]]), horizon_samples=max(1, int(round(rolling_horizon_seconds * dataset.fs))) if history_mode == "rolling" else 0)
                                unknown += block_unknown; writes += len(block)
                            # The last block's delayed labels still cause a
                            # physical graph-state update.  No query consumes
                            # this state, but its rewrite is included rather
                            # than leaving the accounting one block incomplete.
                            final_graph, final_graph_info, final_delta = post_update_graph_transition(previous_graph, history, units, coverage)
                            adds += final_delta["pointer_additions"]; removes += final_delta["pointer_removals"]; rewrites += final_delta["pointer_bit_rewrites"]
                            position_rewrites += final_delta["pointer_position_content_rewrites"]; position_bits += final_delta["pointer_position_content_bit_rewrites"]
                            count_rewrites += final_delta["pointer_count_field_metadata_rewrites"]; count_bits += final_delta["pointer_count_field_metadata_bit_rewrites"]
                            ordered_total_bits += final_delta["pointer_ordered_list_total_bit_rewrites_including_count_metadata"]
                            online_candidates = [np.asarray(row) for row in all_candidates]
                            online_pred = np.asarray(all_pred, dtype=units.dtype)
                            online_diag = {"row_comparisons": np.full(len(test_idx), u, dtype=np.int64), "active_row_count": np.full(len(test_idx), u, dtype=np.int64), "widen_level": np.zeros(len(test_idx), dtype=np.int64), "fallback": np.zeros(len(test_idx), dtype=bool)}
                            online_metrics = summarize_indexed_candidates(online_candidates, test_labels, online_diag, online_pred, {"candidate_waveform_reads": np.asarray(all_reads), "waveform_fallback": np.asarray(all_fallback)}, n_units=u)
                            first = _block_subset_metrics(online_candidates[:first_size], test_labels[:first_size], online_pred[:first_size], np.asarray(all_reads[:first_size]), np.asarray(all_fallback[:first_size]), u)
                            post = _block_subset_metrics(online_candidates[first_size:], test_labels[first_size:], online_pred[first_size:], np.asarray(all_reads[first_size:]), np.asarray(all_fallback[first_size:]), u)
                            base_row("online_graph", mode, coverage, anchors, block_s, source, history_mode, online_metrics, {
                                "initial_graph_total_bits": int(initial_graph_info["graph_total_bits"]), "pointer_graph_mean_bits_over_blocks": float(np.mean(graph_bits)) if graph_bits else 0.0, "pointer_graph_max_bits_over_blocks": int(max(graph_bits)) if graph_bits else 0,
                                "pointer_graph_mean_pointers_over_blocks": float(np.mean(graph_pointers)) if graph_pointers else 0.0, "pointer_graph_max_pointers_over_blocks": int(max(graph_pointers)) if graph_pointers else 0,
                                "block_count": int(len(slices)), "pointer_additions": adds, "pointer_removals": removes, "pointer_bit_rewrites": rewrites,
                                "pointer_position_content_rewrites": position_rewrites, "pointer_position_content_bit_rewrites": position_bits,
                                "pointer_count_field_metadata_rewrites": count_rewrites, "pointer_count_field_metadata_bit_rewrites": count_bits,
                                "pointer_ordered_list_total_bit_rewrites_including_count_metadata": ordered_total_bits,
                                "pointer_rewrite_accounting_scope": "ordered_contiguous_pointer_slots_plus_count_metadata; pointer_bit_rewrites_is_membership_lower_bound; final_post_update_state_included",
                                "pointer_rewrite_includes_final_post_update_state": True,
                                "pointer_graph_final_post_update_total_bits": int(final_graph_info["graph_total_bits"]),
                                "pointer_graph_final_post_update_total_pointers": int(final_graph_info["graph_total_pointers"]),
                                "label_counter_observation_writes": writes, "online_unknown_fit_label_count": unknown,
                                "first_block_recall": first["recall"], "post_first_block_recall": post["recall"], "first_block_waveform_reduction_factor": first["waveform_reduction_factor"], "post_first_block_waveform_reduction_factor": post["waveform_reduction_factor"],
                                "unseen_calibration_label_count": int(initial_graph_info["unseen_calibration_label_count"]), "unseen_calibration_label_rate": float(initial_graph_info["unseen_calibration_label_rate"]),
                            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Causal graph-only online confusion retrieval ceiling")
    parser.add_argument("--family", choices=("hj", "mearec", "all"), default="all"); parser.add_argument("--duration", type=float, default=60.0); parser.add_argument("--bits", type=int, default=5); parser.add_argument("--k", type=int, default=7)
    parser.add_argument("--ranking-modes", nargs="+", choices=("raw_l1", "radius_normalized_l1"), default=["raw_l1", "radius_normalized_l1"]); parser.add_argument("--graph-coverages", nargs="+", type=float, default=[0.95, 0.99, 1.0]); parser.add_argument("--top-anchors", nargs="+", type=int, default=[1, 2]); parser.add_argument("--block-seconds", nargs="+", type=float, default=[1.0, 5.0]); parser.add_argument("--update-sources", nargs="+", choices=("oracle_delayed_label", "self_predicted_label"), default=["oracle_delayed_label", "self_predicted_label"]); parser.add_argument("--histories", nargs="+", choices=("accumulate", "rolling"), default=["accumulate", "rolling"]); parser.add_argument("--rolling-horizon-seconds", type=float, default=10.0); parser.add_argument("--fit-fraction", type=float, default=2.0 / 3.0); parser.add_argument("--seeds", nargs="+", type=int, default=None); parser.add_argument("--waveform-metric", choices=("l1", "l2"), default="l1"); parser.add_argument("--quiet", action="store_true"); parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "output" / "online_confusion_graph"); parser.add_argument("--mearec-npz-dir", type=Path, default=DEFAULT_NPZ_DIR, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not (1 <= args.bits <= 62 and args.k > 0 and 0 < args.fit_fraction < 1 and args.rolling_horizon_seconds > 0 and all(v > 0 for v in args.block_seconds) and all(v > 0 for v in args.top_anchors) and all(0 <= v <= 1 for v in args.graph_coverages)): parser.error("invalid numeric arguments")
    start = time.time(); families: list[tuple[str, Any]] = []
    if args.family in {"hj", "all"}: families.append(("hj", _iter_hj_datasets(args.duration)))
    if args.family in {"mearec", "all"}: families.append(("mearec", _iter_mearec_datasets(args.duration, args.seeds, args.mearec_npz_dir)))
    for family, datasets in families:
        rows: list[dict[str, Any]] = []
        for dataset in datasets:
            rows.extend(run_dataset_online_confusion_graph(dataset, bits=args.bits, k_neighbors=args.k, ranking_modes=args.ranking_modes, graph_coverages=args.graph_coverages, top_anchors=args.top_anchors, block_seconds=args.block_seconds, update_sources=args.update_sources, histories=args.histories, rolling_horizon_seconds=args.rolling_horizon_seconds, fit_fraction=args.fit_fraction, waveform_metric=args.waveform_metric, verbose=not args.quiet)); del dataset
        if rows:
            output = args.output_dir / f"online_confusion_{args.bits}bit_{family}_{int(args.duration)}s.csv"; _write_csv(output, rows); print(f"Saved {len(rows)} rows: {output}")
    print(f"Elapsed: {time.time() - start:.1f}s")


if __name__ == "__main__": main()
