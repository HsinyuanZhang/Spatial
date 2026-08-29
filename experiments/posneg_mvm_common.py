"""Shared I/O, event prep, and row packing for the POSNEG MVM / shift / tracking knife."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from Spatial.algorithms.deformable_spatial_template import (
    event_peak_channel,
    extract_all_channel_extrema,
    geometry_k_table,
)
from Spatial.algorithms.detection import bandpass_filter
from Spatial.algorithms.posneg_mvm_matching import (
    WINDOW,
    _scalar,
    fit_candidate_home_means,
    gather_unit_posneg,
    probe_meta,
    rec_key,
    score_candidate_home,
    assignment_metrics,
    operation_count,
    build_features,
)
from Spatial.experiments.run_adaptive_5bit_range_search import prepare_dataset_events
from Spatial.experiments.run_adaptive_weighted_5bit import stable_chronological_three_way_split
from Spatial.experiments.run_mearec_method_sweep import DEFAULT_NPZ_DIR
from Spatial.experiments.run_shape_descriptor_pilot import (
    LOCKED_DEVELOPMENT_RECORDINGS,
    _load_locked_recording,
)
from Spatial.experiments.spatial_cim_common import sha256_file, write_json


REPO_ROOT = Path(__file__).resolve().parents[2]


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return json_ready(_scalar(value))
        if value.size <= 64 and value.ndim <= 2:
            return json_ready(value.tolist())
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return json_ready(_scalar(value))
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def flatten_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    skip = {
        "pred",
        "correct",
        "per_unit_recall",
        "m_home",
        "m_all",
        "have_home_rival",
        "distances",
        "delta_trace",
        "mu_move_trace",
        "mu_move_ratio_end",
        "delta_end",
        "scores",
    }
    out: dict[str, Any] = {}
    for key, val in metrics.items():
        if key in skip:
            continue
        if isinstance(val, np.ndarray):
            continue
        if isinstance(val, (float, np.floating)):
            out[key] = float(val) if np.isfinite(val) else float("nan")
        elif isinstance(val, (int, np.integer, bool, np.bool_)):
            out[key] = _scalar(val)
        elif isinstance(val, str):
            out[key] = val
        elif val is None:
            out[key] = ""
    return out


def prepare_pack(dataset, family: str) -> dict[str, Any]:
    knn = prepare_dataset_events(dataset, k_neighbors=7, footprint_layout="knn")
    filtered = bandpass_filter(dataset.raw_data, dataset.fs)
    vmax, vmin = extract_all_channel_extrema(filtered, knn["times"], window=WINDOW)
    if vmax.shape[0] != knn["times"].shape[0]:
        raise AssertionError("extrema rows drifted from prepared events")
    homes = event_peak_channel(vmax, vmin)
    fit, cal, test = stable_chronological_three_way_split(knn["times"])
    meta = probe_meta(dataset.geom)
    return {
        "dataset": dataset.name,
        "family": family,
        "rec_key": rec_key(family, dataset.name),
        "geom": np.asarray(dataset.geom, dtype=np.float64),
        "fs": float(dataset.fs),
        "vmax": vmax,
        "vmin": vmin,
        "labels": knn["labels"],
        "times": knn["times"],
        "homes": homes,
        "fit": fit,
        "cal": cal,
        "test": test,
        "pitch": meta["pitch"],
        "axis": meta["axis"],
        "n_ch": meta["n_ch"],
        "isolation": "GT events, GT units, zero-phase 300-6000 Hz",
    }


def load_pilot_packs(duration_s: float, mearec_dir: Path, quiet: bool) -> list[dict[str, Any]]:
    if float(duration_s) != 60.0:
        raise ValueError("duration locked to 60 s")
    packs = []
    for family, name in LOCKED_DEVELOPMENT_RECORDINGS:
        dataset = _load_locked_recording(family, name, 60.0, mearec_dir)
        pack = prepare_pack(dataset, family)
        packs.append(pack)
        if not quiet:
            print(
                f"prepared {pack['rec_key']} n={pack['times'].size} C={pack['n_ch']}",
                flush=True,
            )
        del dataset
    keys = {p["rec_key"] for p in packs}
    required = {"hj_drift", "hj_static", "mearec_10u", "mearec_20u"}
    if keys != required:
        raise AssertionError(f"expected {required}, got {keys}")
    return packs


def fit_k_model(pack: dict[str, Any], k: int, frame: str = "candidate_home") -> dict[str, Any]:
    table, rel = geometry_k_table(pack["geom"], int(k))
    idx = pack["fit"]
    model = fit_candidate_home_means(
        pack["vmax"][idx],
        pack["vmin"][idx],
        pack["labels"][idx],
        pack["homes"][idx],
        table,
        frame=frame,
    )
    pos_eu, neg_eu = gather_unit_posneg(pack["vmax"], pack["vmin"], table, model["homes"])
    return {
        "table": table,
        "rel_xy": rel,
        "model": model,
        "pos_eu": pos_eu,
        "neg_eu": neg_eu,
        "k": int(k),
        "frame": frame,
    }


def eval_arm(
    pack: dict[str, Any],
    fitted: dict[str, Any],
    split: str,
    representation: str,
    kind: str,
    *,
    lam_amp: float = 1.0,
    lam_pos: float = 1.0,
    lam_neg: float = 1.0,
) -> dict[str, Any]:
    idx = pack[split]
    model = fitted["model"]
    scores = score_candidate_home(
        fitted["pos_eu"][idx],
        fitted["neg_eu"][idx],
        model["mu_pos"],
        model["mu_neg"],
        fitted["rel_xy"],
        model["homes"],
        representation,
        kind,
        lam_amp=lam_amp,
        lam_pos=lam_pos,
        lam_neg=lam_neg,
    )
    metrics = assignment_metrics(
        scores, pack["labels"][idx], model["units"], model["homes"], kind
    )
    dim = int(
        build_features(
            model["mu_pos"][0],
            model["mu_neg"][0],
            "posneg" if representation == "posneg_weighted" else representation,
            fitted["rel_xy"][int(model["homes"][0])],
        ).shape[-1]
    )
    ops = operation_count(kind, dim)
    metrics.update(ops)
    metrics["kind"] = kind
    metrics["representation"] = representation
    metrics["split"] = split
    return metrics


def base_row(pack: dict[str, Any], k: int, part: str, arm: str) -> dict[str, Any]:
    return {
        "rec_key": pack["rec_key"],
        "dataset": pack["dataset"],
        "family": pack["family"],
        "k": int(k),
        "part": part,
        "arm": arm,
        "isolation": pack["isolation"],
        "gt_used": True,
    }


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", restval="")
        writer.writeheader()
        for row in rows:
            clean = {}
            for key, val in row.items():
                if isinstance(val, float) and not np.isfinite(val):
                    clean[key] = ""
                elif isinstance(val, (np.floating, np.integer, np.bool_)):
                    clean[key] = _scalar(val)
                else:
                    clean[key] = val
            writer.writerow(clean)


def source_hashes(paths: Iterable[str]) -> dict[str, str]:
    out = {}
    for rel in paths:
        p = REPO_ROOT / rel
        if p.exists():
            out[rel] = sha256_file(p)
    return out


def write_sha256_manifest(path: Path, mapping: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{digest}  {name}\n" for name, digest in sorted(mapping.items())]
    path.write_text("".join(lines), encoding="utf-8")


def output_hashes(paths: Iterable[Path]) -> dict[str, str]:
    out = {}
    for p in paths:
        if p.exists():
            rel = str(p.relative_to(REPO_ROOT)) if p.is_relative_to(REPO_ROOT) else str(p)
            out[rel] = sha256_file(p)
    return out


def try_savefig(path: Path, fig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")


def rel_or_str(path: Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def hash_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
