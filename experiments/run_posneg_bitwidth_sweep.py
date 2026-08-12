"""POSNEG code-width sweep and accuracy-ceiling curve.

The registered Phase A grid tested only 4 and 5 bits and 4 won on every
development recording, which is the edge of the grid.  This sweeps wider on the
full corpus to locate the optimum, and renders the curve.

Evidence level: exploratory.  GT event times, GT unit rows, chronological 50/50,
full-recording zero-phase filtering.  It cannot select a configuration, because
it is outside the registered Phase A grid; it can only show whether that grid
was placed correctly.

  python -m Spatial.experiments.run_posneg_bitwidth_sweep --quiet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from Spatial.algorithms.spatial_footprint import posneg_codes
from Spatial.data.loader import (
    list_hybrid_janelia_scenes,
    load_hybrid_janelia,
    load_mearec_npz,
)
from Spatial.experiments.run_adaptive_5bit_range_search import _chronological_split
from Spatial.experiments.run_mearec_method_sweep import DEFAULT_NPZ_DIR, discover_npz
from Spatial.experiments.shape_descriptor_common import (
    COM_BITS,
    build_descriptor_blocks,
)
from Spatial.experiments.spatial_cim_common import write_csv, write_json

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "Spatial" / "output" / "posneg_bitwidth_sweep"
DEFAULT_BITS = (2, 3, 4, 5, 6)


def _row_bits(posneg_bits: int, n_posneg_cols: int) -> int:
    return 2 * COM_BITS + n_posneg_cols * int(posneg_bits)


def evaluate_recording(
    dataset, *, family: str, bits: Sequence[int]
) -> dict[str, Any]:
    """Nearest integer-template L1 accuracy for COM+POSNEG at each code width."""
    blocks = build_descriptor_blocks(dataset)
    train_idx, test_idx = _chronological_split(blocks["times"])
    labels = blocks["labels"]
    y_train, y_test = labels[train_idx], labels[test_idx]

    # build_descriptor_blocks already emits COM as COM_BITS codes; quantizing it
    # again would saturate every column to the maximum code.
    com = np.asarray(blocks["com"], dtype=np.float64)
    if float(com.max()) > float((1 << COM_BITS) - 1):
        raise AssertionError("blocks['com'] is not already quantized as expected")

    def accuracy_of(codes: np.ndarray) -> float:
        units = np.unique(y_train)
        templates = np.stack(
            [np.rint(codes[train_idx][y_train == u].mean(axis=0)) for u in units]
        ).astype(np.int64)
        query = np.rint(codes[test_idx]).astype(np.int64)
        distances = np.abs(query[:, None, :] - templates[None, :, :]).sum(axis=2)
        # Stable argmin keeps the first template on an exact integer tie.
        predicted = units[distances.argmin(axis=1)]
        keep = np.isin(y_test, units)
        return float((predicted[keep] == y_test[keep]).mean())

    row: dict[str, Any] = {
        "dataset": dataset.name,
        "family": family,
        "n_train": int(train_idx.size),
        "n_test": int(test_idx.size),
        "n_units": int(np.unique(y_train).size),
        # The current D9/B5 descriptor, evaluated on identical rows as reference.
        "accuracy_d9": accuracy_of(
            np.concatenate([com, np.asarray(blocks["p2p"], dtype=np.float64)], axis=1)
        ),
    }
    for n_bits in bits:
        codes = np.concatenate(
            [
                com,
                posneg_codes(
                    blocks["vmax"],
                    blocks["vmin"],
                    blocks["neighbor_ids"],
                    n_bits=int(n_bits),
                ),
            ],
            axis=1,
        )
        row[f"accuracy_{n_bits}bit"] = accuracy_of(codes)
        row[f"row_bits_{n_bits}bit"] = _row_bits(n_bits, codes.shape[1] - 2)
    return row


def summarize(rows: Sequence[dict[str, Any]], bits: Sequence[int]) -> dict[str, Any]:
    summary: dict[str, Any] = {"bits": list(bits), "by_family": {}}
    for family in sorted({r["family"] for r in rows}):
        subset = [r for r in rows if r["family"] == family]
        total = sum(int(r["n_test"]) for r in subset)
        if total == 0:
            raise ValueError(f"no test events for family {family!r}")
        means = {
            int(b): sum(
                r[f"accuracy_{b}bit"] * int(r["n_test"]) for r in subset
            )
            / total
            for b in bits
        }
        peak = max(means, key=lambda b: means[b])
        summary["by_family"][family] = {
            "n_recordings": len(subset),
            "accuracy": {str(b): round(means[b], 4) for b in bits},
            "row_bits": {str(b): subset[0][f"row_bits_{b}bit"] for b in bits},
            "peak_bits": peak,
            "peak_accuracy": round(means[peak], 4),
            "d9_reference_accuracy": round(
                sum(r["accuracy_d9"] * int(r["n_test"]) for r in subset) / total, 4
            ),
            "d9_row_bits": 2 * COM_BITS + 7 * 5,
        }
    return summary


def render_figure(
    rows: Sequence[dict[str, Any]],
    summary: dict[str, Any],
    bits: Sequence[int],
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hj_color, me_color, grey = "#1f4e9c", "#c2570a", "#7a7a7a"
    by_family = summary["by_family"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.6, 5.4))

    d9_values = [by_family[f]["d9_reference_accuracy"] for f in by_family]
    ax.axhspan(min(d9_values), max(d9_values), color=grey, alpha=0.22, zorder=0)
    ax.text(
        min(bits) + 0.02,
        max(d9_values) + 0.0018,
        "current D9/B5 descriptor · "
        + " / ".join(
            f"{by_family[f]['d9_reference_accuracy']:.3f} "
            f"{'HJ' if f == 'hj' else 'MEArec'}"
            for f in ("hj", "mearec")
            if f in by_family
        )
        + f" · {by_family['hj']['d9_row_bits']} bit row",
        fontsize=8.8,
        color="#4a4a4a",
        va="bottom",
    )

    for family, color, label in (
        ("hj", hj_color, f"Hybrid Janelia  (n={by_family['hj']['n_recordings']})"),
        ("mearec", me_color, f"MEArec SNR5  (n={by_family['mearec']['n_recordings']})"),
    ):
        y = [by_family[family]["accuracy"][str(b)] for b in bits]
        ax.plot(bits, y, "-o", color=color, lw=2.6, ms=8, label=label, zorder=4)
        peak = by_family[family]["peak_bits"]
        ax.plot(
            peak,
            by_family[family]["peak_accuracy"],
            "o",
            ms=16,
            mfc="none",
            mec=color,
            mew=2.4,
            zorder=5,
        )

    ax.set_xlabel("POSNEG code width (bits per coordinate)", fontsize=10.5)
    ax.set_ylabel("Assignment accuracy ceiling", fontsize=10.5)
    ax.set_xticks(list(bits))
    ax.grid(alpha=0.28, ls="--", lw=0.7)
    ax.legend(loc="lower right", fontsize=9.2, framealpha=0.95)
    ax.text(
        0.972,
        0.30,
        "\n".join(
            f"peak  {'HJ' if f == 'hj' else 'MEArec'} {by_family[f]['peak_bits']} bit"
            f" · {by_family[f]['peak_accuracy']:.3f}"
            for f in ("hj", "mearec")
        ),
        transform=ax.transAxes,
        fontsize=9.3,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#999999", lw=0.9),
    )
    ax.set_title(
        "More precision makes it worse",
        fontsize=12.5,
        loc="left",
        fontweight="bold",
        pad=38,
    )
    top = ax.secondary_xaxis("top")
    top.set_xticks(list(bits))
    top.set_xticklabels(
        [by_family["hj"]["row_bits"][str(b)] for b in bits], fontsize=9
    )
    top.set_xlabel("stored bits per cluster row", fontsize=9.3, labelpad=5)

    for r in rows:
        bx.plot(
            bits,
            [r[f"accuracy_{b}bit"] for b in bits],
            "-",
            color=hj_color if r["family"] == "hj" else me_color,
            lw=1.0,
            alpha=0.38,
            zorder=2,
        )
    for family, color, name in (("hj", hj_color, "HJ"), ("mearec", me_color, "MEArec")):
        bx.plot(
            bits,
            [by_family[family]["accuracy"][str(b)] for b in bits],
            "-o",
            color=color,
            lw=3.0,
            ms=7.5,
            zorder=4,
            label=f"{name} event-weighted mean",
        )

    values = [r[f"accuracy_{b}bit"] for r in rows for b in bits]
    bx.set_xlabel("POSNEG code width (bits per coordinate)", fontsize=10.5)
    bx.set_ylabel("Assignment accuracy", fontsize=10.5)
    bx.set_xticks(list(bits))
    bx.set_ylim(min(values) - 0.012, max(values) + 0.055)
    bx.grid(alpha=0.28, ls="--", lw=0.7)
    bx.legend(loc="lower left", fontsize=9.2, framealpha=0.95)

    lo, hi = min(bits), max(bits)
    worst = min(rows, key=lambda r: r[f"accuracy_{hi}bit"] - r[f"accuracy_{lo}bit"])
    bx.annotate(
        f"steepest fall · {worst['dataset'].replace('hybrid_janelia_', '')}\n"
        f"{worst['accuracy_4bit']:.3f} at 4 bit  →  {worst[f'accuracy_{hi}bit']:.3f}"
        f" at {hi} bit",
        xy=(hi, worst[f"accuracy_{hi}bit"]),
        xytext=((lo + hi) / 2 + 0.45, max(values) + 0.035),
        fontsize=9.2,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#555555", lw=0.9),
        arrowprops=dict(
            arrowstyle="->", color="#555555", lw=1.2, connectionstyle="arc3,rad=-0.18"
        ),
        zorder=6,
    )
    bx.set_title(
        "Every recording, same shape",
        fontsize=12.5,
        loc="left",
        fontweight="bold",
        pad=38,
    )
    bx.text(
        0.0,
        1.028,
        f"Thin lines are the {len(rows)} individual recordings; "
        "the drift scenes fall hardest",
        transform=bx.transAxes,
        fontsize=9.6,
        color="#333333",
    )

    fig.text(
        0.008,
        -0.035,
        "Exploratory diagnostic · GT event times and GT unit rows · chronological 50/50"
        " · full-recording zero-phase filtering · logical row bits, not area or energy",
        fontsize=8.4,
        color=grey,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="POSNEG code-width sweep")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--bits", type=int, nargs="+", default=list(DEFAULT_BITS))
    parser.add_argument("--mearec-npz-dir", type=Path, default=DEFAULT_NPZ_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    bits = []
    for value in args.bits:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"bits must be integers, got {value!r}")
        if not 1 <= int(value) <= 16:
            raise ValueError(f"bits must lie in [1, 16], got {value}")
        if int(value) not in bits:
            bits.append(int(value))
    bits.sort()
    if not bits:
        raise ValueError("at least one code width is required")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for entry in list_hybrid_janelia_scenes(ready_only=True):
        scene = entry.get("scene_key") or entry.get("short_name")
        dataset = load_hybrid_janelia(scene=scene, duration_s=args.duration)
        rows.append(evaluate_recording(dataset, family="hj", bits=bits))
        if not args.quiet:
            print(rows[-1]["dataset"], flush=True)
    for path in discover_npz(Path(args.mearec_npz_dir)):
        if "snr5" not in path.name:
            continue
        dataset = load_mearec_npz(path, duration_s=args.duration)
        rows.append(evaluate_recording(dataset, family="mearec", bits=bits))
        if not args.quiet:
            print(rows[-1]["dataset"], flush=True)
    if not rows:
        raise ValueError("no recordings were evaluated")

    summary = summarize(rows, bits)
    write_csv(out_dir / "posneg_bitwidth_sweep.csv", rows)
    write_json(out_dir / "posneg_bitwidth_sweep_summary.json", summary)
    if not args.no_figure:
        render_figure(rows, summary, bits, out_dir / "posneg_bitwidth_curve.png")

    if not args.quiet:
        print(json.dumps(summary["by_family"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
