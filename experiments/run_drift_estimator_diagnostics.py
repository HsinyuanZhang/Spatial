"""Zero-cost diagnostics for the shared-shift estimator and the EMA rate.

Two questions, both answered on the locked 60 s development recordings with
GT events and GT unit rows. Nothing here is a tracker or a selection.

A. Does the deployed shift estimator convert amplitude error into false
   motion? ``estimate_best_shift`` minimizes ||x - S(d) mu||^2 without
   profiling out a gain, so an event whose amplitude differs from its
   template can lower that objective by moving d away from zero. If true,
   |d_hat| depends on ||x||/||mu||, the static-recording noise floor is a
   bias rather than noise, and the drift-vs-static |d_hat| gap is partly an
   amplitude artefact.

B. Where is the EMA collapse rate? An EMA with rate eta has effective
   sample size N_eff = (2-eta)/eta against a template of dimension D = 2K.
   If the observed MEArec 20u collapse is a variance effect, the collapse
   rate should move with 1/D, i.e. K=7 should tolerate a larger eta than
   K=12 and K=16 a smaller one.

    python -m Spatial.experiments.run_drift_estimator_diagnostics --pilot
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from Spatial.algorithms.drift_tracking import CausalTracker, run_causal_stream
from Spatial.algorithms.posneg_mvm_matching import _scalar
from Spatial.algorithms.spatial_shift_matching import (
    SHIFT_FRACTIONAL,
    precompute_shift_mats,
)
from Spatial.experiments.posneg_mvm_common import (
    REPO_ROOT,
    eval_arm,
    fit_k_model,
    json_ready,
    load_pilot_packs,
    rel_or_str,
    source_hashes,
    write_csv_rows,
    write_sha256_manifest,
)
from Spatial.experiments.run_mearec_method_sweep import DEFAULT_NPZ_DIR
from Spatial.experiments.spatial_cim_common import sha256_file, write_json


DEFAULT_OUT = REPO_ROOT / "Spatial" / "output" / "drift_estimator_diag"
SOURCE_PATHS = (
    "Spatial/algorithms/drift_tracking.py",
    "Spatial/algorithms/spatial_shift_matching.py",
    "Spatial/experiments/run_drift_estimator_diagnostics.py",
)

K_ID = 12
K_GRID_B = (7, 12, 16)
ETA_GRID_B = (0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1)
REC_B = ("mearec_20u", "hj_drift")
EPS = 1e-12


# --------------------------------------------------------------------------
# Diagnostic A
# --------------------------------------------------------------------------


def _true_unit_index(labels: np.ndarray, units: np.ndarray) -> np.ndarray:
    row = {_scalar(u): i for i, u in enumerate(units)}
    return np.array([row.get(_scalar(v), -1) for v in labels], dtype=np.int64)


def _delta_variants(
    x: np.ndarray,
    shifted_mu: np.ndarray,
    deltas: np.ndarray,
) -> dict[str, np.ndarray]:
    """Per-event best delta under three objectives.

    ``x`` is (n, 2K); ``shifted_mu`` is (n_delta, 2K) for one unit.
    ``plain`` reproduces the deployed estimator. ``gain`` profiles out a
    non-negative scalar gain at each delta. ``gain_sub`` adds a three-point
    parabolic refinement of the gain-profiled objective.
    """
    n = int(x.shape[0])
    n_d = int(shifted_mu.shape[0])
    x_sq = np.sum(x * x, axis=1)
    err_plain = np.empty((n, n_d), dtype=np.float64)
    err_gain = np.empty((n, n_d), dtype=np.float64)
    for d in range(n_d):
        m = shifted_mu[d]
        m_sq = float(np.dot(m, m))
        xm = x @ m
        err_plain[:, d] = x_sq - 2.0 * xm + m_sq
        a = np.maximum(xm / max(m_sq, EPS), 0.0)
        err_gain[:, d] = x_sq - 2.0 * a * xm + (a * a) * m_sq
    i_plain = np.argmin(err_plain, axis=1)
    i_gain = np.argmin(err_gain, axis=1)
    d_plain = deltas[i_plain]
    d_gain = deltas[i_gain]

    # Parabolic sub-grid refinement of the gain-profiled objective.
    d_sub = d_gain.copy()
    interior = (i_gain > 0) & (i_gain < n_d - 1)
    if np.any(interior):
        idx = np.flatnonzero(interior)
        j = i_gain[idx]
        y0 = err_gain[idx, j - 1]
        y1 = err_gain[idx, j]
        y2 = err_gain[idx, j + 1]
        den = y0 - 2.0 * y1 + y2
        ok = np.abs(den) > EPS
        step = np.zeros(idx.size, dtype=np.float64)
        step[ok] = 0.5 * (y0[ok] - y2[ok]) / den[ok]
        step = np.clip(step, -0.5, 0.5)
        grid_step = float(deltas[1] - deltas[0]) if n_d > 1 else 0.0
        d_sub[idx] = deltas[j] + step * grid_step
    return {"plain": d_plain, "gain": d_gain, "gain_sub": d_sub}


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if int(ok.sum()) < 10:
        return float("nan")
    from scipy.stats import rankdata

    ra = rankdata(a[ok])
    rb = rankdata(b[ok])
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = float(np.linalg.norm(ra) * np.linalg.norm(rb))
    if den <= EPS:
        return float("nan")
    return float(np.dot(ra, rb) / den)


def _decile_table(ratio: np.ndarray, absd: np.ndarray, n_bins: int = 5) -> list[dict[str, Any]]:
    edges = np.quantile(ratio, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    out = []
    for i in range(n_bins):
        m = (ratio >= edges[i]) & (ratio < edges[i + 1])
        if not np.any(m):
            continue
        out.append(
            {
                "bin": i,
                "ratio_lo": float(edges[i]),
                "ratio_hi": float(edges[i + 1]),
                "ratio_mean": float(np.mean(ratio[m])),
                "abs_delta_mean": float(np.mean(absd[m])),
                "frac_nonzero": float(np.mean(absd[m] > 1e-9)),
                "n": int(m.sum()),
            }
        )
    return out


def diagnostic_a(packs: list[dict[str, Any]], quiet: bool) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    bins: list[dict[str, Any]] = []
    per_event: dict[str, dict[str, np.ndarray]] = {}
    deltas = np.asarray(SHIFT_FRACTIONAL, dtype=np.float64)

    for pack in packs:
        fitted = fit_k_model(pack, K_ID)
        model = fitted["model"]
        units = model["units"]
        homes = np.asarray(model["homes"], dtype=np.int64)
        bank = precompute_shift_mats(
            fitted["rel_xy"], fitted["table"], SHIFT_FRACTIONAL, pack["pitch"], pack["axis"]
        )
        # Held-out events only: the fit split defines mu, so its amplitude
        # ratio is centered by construction.
        idx = np.sort(np.concatenate([pack["cal"], pack["test"]]))
        ti = _true_unit_index(pack["labels"][idx], units)
        ratio_all: list[np.ndarray] = []
        got: dict[str, list[np.ndarray]] = {"plain": [], "gain": [], "gain_sub": []}
        for ui in range(int(units.size)):
            sel = np.flatnonzero(ti == ui)
            if sel.size == 0:
                continue
            h = int(homes[ui])
            if h < 0:
                continue
            mu = np.concatenate([model["mu_pos"][ui], model["mu_neg"][ui]])
            mu_norm = float(np.linalg.norm(mu))
            if mu_norm <= EPS:
                continue
            mats = bank[h]
            shifted = np.stack(
                [
                    np.concatenate(
                        [mats[d] @ model["mu_pos"][ui], mats[d] @ model["mu_neg"][ui]]
                    )
                    for d in range(mats.shape[0])
                ],
                axis=0,
            )
            e_idx = idx[sel]
            x = np.concatenate(
                [fitted["pos_eu"][e_idx, ui], fitted["neg_eu"][e_idx, ui]], axis=1
            )
            var = _delta_variants(x, shifted, deltas)
            ratio_all.append(np.linalg.norm(x, axis=1) / mu_norm)
            for key in got:
                got[key].append(var[key])
        if not ratio_all:
            continue
        ratio = np.concatenate(ratio_all)
        store = {"ratio": ratio}
        for key in got:
            d = np.concatenate(got[key])
            store[key] = d
            absd = np.abs(d)
            rho = _spearman(ratio, absd)
            rows.append(
                {
                    "rec_key": pack["rec_key"],
                    "k": K_ID,
                    "estimator": key,
                    "n_events": int(d.size),
                    "mean_abs_delta": float(np.mean(absd)),
                    "median_abs_delta": float(np.median(absd)),
                    "frac_nonzero_delta": float(np.mean(absd > 1e-9)),
                    "mean_signed_delta": float(np.mean(d)),
                    "spearman_absdelta_vs_ratio": rho,
                    "ratio_p10": float(np.percentile(ratio, 10)),
                    "ratio_p50": float(np.percentile(ratio, 50)),
                    "ratio_p90": float(np.percentile(ratio, 90)),
                }
            )
            for b in _decile_table(ratio, absd):
                b.update({"rec_key": pack["rec_key"], "estimator": key})
                bins.append(b)
            if not quiet:
                print(
                    f"A {pack['rec_key']:<12} {key:<9} mean|d|={np.mean(absd):.4f} "
                    f"nonzero={np.mean(absd > 1e-9):.3f} rho={rho:+.3f}",
                    flush=True,
                )
        per_event[pack["rec_key"]] = store
        del fitted
    return {"rows": rows, "bins": bins, "per_event": per_event}


def amplitude_adjusted_gap(per_event: dict[str, dict[str, np.ndarray]]) -> list[dict[str, Any]]:
    """How much of the drift-vs-static |d_hat| gap survives amplitude matching?

    Reweight HJ drift events so their ||x||/||mu|| histogram matches HJ
    static, then recompute mean |d_hat|. A gap that vanishes under matching
    was an amplitude artefact, not motion.
    """
    if "hj_drift" not in per_event or "hj_static" not in per_event:
        return []
    out: list[dict[str, Any]] = []
    drift = per_event["hj_drift"]
    static = per_event["hj_static"]
    edges = np.quantile(static["ratio"], np.linspace(0.0, 1.0, 11))
    edges[0] = -np.inf
    edges[-1] = np.inf
    for key in ("plain", "gain", "gain_sub"):
        a_d = np.abs(drift[key])
        a_s = np.abs(static[key])
        bd = np.digitize(drift["ratio"], edges[1:-1])
        bs = np.digitize(static["ratio"], edges[1:-1])
        num = 0.0
        den = 0.0
        for b in range(len(edges) - 1):
            md = bd == b
            ms = bs == b
            if not np.any(md) or not np.any(ms):
                continue
            w = float(ms.sum())
            num += w * float(np.mean(a_d[md]))
            den += w
        matched = num / den if den > 0 else float("nan")
        out.append(
            {
                "estimator": key,
                "hj_drift_mean_abs_delta": float(np.mean(a_d)),
                "hj_static_mean_abs_delta": float(np.mean(a_s)),
                "raw_gap": float(np.mean(a_d) - np.mean(a_s)),
                "hj_drift_amplitude_matched": matched,
                "matched_gap": float(matched - np.mean(a_s)),
            }
        )
    return out


# --------------------------------------------------------------------------
# Diagnostic B
# --------------------------------------------------------------------------


def diagnostic_b(packs: list[dict[str, Any]], quiet: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pack in packs:
        if pack["rec_key"] not in REC_B:
            continue
        for k in K_GRID_B:
            if k > int(pack["n_ch"]):
                continue
            fitted = fit_k_model(pack, k)
            model = fitted["model"]
            cal = eval_arm(pack, fitted, "cal", "posneg", "l2")
            p10 = cal.get("margin_all_p10", 0.0)
            thresh = float(max(0.0, p10 if np.isfinite(p10) else 0.0))
            idx = pack["cal"]
            frozen_acc = float(cal["accuracy"])
            for eta in ETA_GRID_B:
                tr = CausalTracker(
                    "ema",
                    model["mu_pos"],
                    model["mu_neg"],
                    model["homes"],
                    fitted["rel_xy"],
                    fitted["table"],
                    pack["geom"],
                    pack["pitch"],
                    pack["axis"],
                    np.zeros(model["homes"].size, dtype=np.int64),
                    eta_ema=float(eta),
                    kind="l2",
                    representation="posneg",
                    confidence_threshold=thresh,
                )
                met = run_causal_stream(
                    tr,
                    fitted["pos_eu"][idx],
                    fitted["neg_eu"][idx],
                    pack["labels"][idx],
                    model["units"],
                )
                ratio = np.asarray(met["mu_move_ratio_end"], dtype=np.float64)
                ratio = ratio[np.isfinite(ratio)]
                dim = 2 * int(k)
                n_eff = (2.0 - float(eta)) / float(eta)
                rows.append(
                    {
                        "rec_key": pack["rec_key"],
                        "k": int(k),
                        "dim": dim,
                        "eta": float(eta),
                        "n_eff": n_eff,
                        "n_eff_per_dim": n_eff / dim,
                        "split": "cal",
                        "accuracy": float(met["accuracy"]),
                        "frozen_accuracy": frozen_acc,
                        "delta_acc_vs_frozen": float(met["accuracy"] - frozen_acc),
                        "same_home_accuracy": float(met["same_home_accuracy"]),
                        "worst_unit_recall": float(met["worst_unit_recall"]),
                        "mu_move_ratio_mean": (
                            float(np.mean(ratio)) if ratio.size else float("nan")
                        ),
                        "update_rate": float(met["update_rate"]),
                        "n_test": int(met["n_test"]),
                    }
                )
                if not quiet:
                    print(
                        f"B {pack['rec_key']:<12} K={k:<3} eta={eta:<6} "
                        f"Neff/D={n_eff / dim:6.2f} acc={met['accuracy']:.4f} "
                        f"(frozen {frozen_acc:.4f})",
                        flush=True,
                    )
            del fitted
    return rows


def collapse_summary(rows: list[dict[str, Any]], drop_pp: float = 5.0) -> list[dict[str, Any]]:
    """Smallest eta whose accuracy falls more than ``drop_pp`` below Frozen."""
    out = []
    keys = sorted({(r["rec_key"], r["k"]) for r in rows})
    for rec, k in keys:
        sub = sorted(
            [r for r in rows if r["rec_key"] == rec and r["k"] == k],
            key=lambda r: r["eta"],
        )
        hit = next(
            (r for r in sub if r["delta_acc_vs_frozen"] * 100.0 < -float(drop_pp)), None
        )
        out.append(
            {
                "rec_key": rec,
                "k": int(k),
                "dim": 2 * int(k),
                "eta_collapse": float(hit["eta"]) if hit else float("nan"),
                "n_eff_per_dim_at_collapse": (
                    float(hit["n_eff_per_dim"]) if hit else float("nan")
                ),
                "predicted_eta_collapse_1_over_dim": 1.0 / (2.0 * int(k)),
                "drop_pp_threshold": float(drop_pp),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--mearec-dir", type=Path, default=DEFAULT_NPZ_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--only", choices=("a", "b", "both"), default="both")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if not args.pilot:
        parser.error("pass --pilot")
    if float(args.duration) != 60.0:
        parser.error("duration locked to 60 s")

    packs = load_pilot_packs(60.0, Path(args.mearec_dir), bool(args.quiet))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    summary: dict[str, Any] = {
        "isolation": "GT events, GT units, zero-phase 300-6000 Hz, 60 s, K=12 unless noted",
        "source_sha256": source_hashes(SOURCE_PATHS),
        "gt_used": True,
    }

    if args.only in ("a", "both"):
        res_a = diagnostic_a(packs, bool(args.quiet))
        p = out / "estimator_amplitude_bias.csv"
        write_csv_rows(p, res_a["rows"])
        written.append(p)
        pb = out / "estimator_amplitude_bins.csv"
        write_csv_rows(pb, res_a["bins"])
        written.append(pb)
        adj = amplitude_adjusted_gap(res_a["per_event"])
        pa = out / "estimator_amplitude_matched_gap.csv"
        write_csv_rows(pa, adj)
        written.append(pa)
        summary["diagnostic_a"] = json_ready(res_a["rows"])
        summary["diagnostic_a_amplitude_matched"] = json_ready(adj)
        if not args.quiet and adj:
            print("\namplitude-matched HJ drift vs static gap:")
            for r in adj:
                print(
                    f"  {r['estimator']:<9} raw gap {r['raw_gap']:+.4f} -> "
                    f"matched gap {r['matched_gap']:+.4f}",
                    flush=True,
                )

    if args.only in ("b", "both"):
        rows_b = diagnostic_b(packs, bool(args.quiet))
        p = out / "ema_rate_vs_k.csv"
        write_csv_rows(p, rows_b)
        written.append(p)
        coll = collapse_summary(rows_b)
        pc = out / "ema_collapse_threshold.csv"
        write_csv_rows(pc, coll)
        written.append(pc)
        summary["diagnostic_b"] = json_ready(rows_b)
        summary["diagnostic_b_collapse"] = json_ready(coll)
        if not args.quiet:
            print("\nEMA collapse threshold (>5 pp below Frozen):")
            for r in coll:
                print(
                    f"  {r['rec_key']:<12} K={r['k']:<3} D={r['dim']:<3} "
                    f"eta_collapse={r['eta_collapse']} "
                    f"(1/D = {r['predicted_eta_collapse_1_over_dim']:.4f})",
                    flush=True,
                )

    json_path = out / "drift_estimator_diag_summary.json"
    write_json(json_path, json_ready(summary))
    written.append(json_path)
    write_sha256_manifest(
        out / "outputs.sha256",
        {rel_or_str(p): sha256_file(p) for p in written if p.exists()},
    )
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
