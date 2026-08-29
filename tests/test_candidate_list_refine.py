"""WTA among frozen candidate lists (no recording I/O)."""

import numpy as np

from Spatial.experiments.candidate_list_refine import (
    codebook_candidate_lists,
    integer_mean_templates,
    scatter_to_channels,
    summarize_refine,
    unit_mean_templates,
    wta_among_lists,
)
from Spatial.experiments.run_bulk_tail_escalation import bulk_tail_from_codes


def test_scatter_writes_only_valid_slots():
    values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    nbr = np.array([[0, 2, -1], [1, 2, 0]])
    out = scatter_to_channels(values, nbr, n_channels=3)
    np.testing.assert_allclose(out[0], [1.0, 0.0, 2.0])
    np.testing.assert_allclose(out[1], [6.0, 4.0, 5.0])


def test_wta_skips_reads_on_empty_and_unique():
    query = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    tmpl = np.array([[0.0, 1.0], [1.0, 0.0]])
    units = np.array([7, 9])
    lists = [np.array([], dtype=np.int64), np.array([9]), np.array([7, 9])]
    out = wta_among_lists(query, tmpl, units, lists)
    assert out["pred"][0] == -1
    assert out["pred"][1] == 9
    assert out["pred"][2] == 7
    np.testing.assert_array_equal(out["reads"], [0, 0, 2])


def test_wta_l1_picks_closer_template_inside_list():
    query = np.array([[0.0, 10.0]])
    tmpl = np.array([[0.0, 0.0], [0.0, 9.0], [8.0, 8.0]])
    units = np.array([1, 2, 3])
    lists = [np.array([1, 2, 3])]
    out = wta_among_lists(query, tmpl, units, lists, metric="l1")
    assert out["pred"][0] == 2


def test_codebook_lists_union_neighbour_codes():
    homes = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)
    codes = np.array([1, 2, 1, 2, 1, 2, 1, 2], dtype=np.float64)
    labels = np.array([1, 2, 1, 2, 1, 2, 1, 2], dtype=np.int64)
    times = np.arange(8, dtype=np.int64)
    rec = codebook_candidate_lists(homes, codes, labels, times, code_delta=1)
    assert rec["y_test"].size == 4
    # Train saw unit 1 at code 1 and unit 2 at code 2; delta=1 unions both.
    for cand in rec["lists"]:
        assert set(cand.tolist()) == {1, 2}


def test_templates_and_summary_conditional_on_true_in_list():
    feat = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.1], [1.0, 0.1]])
    labels = np.array([3, 8, 3, 8])
    units = np.array([3, 8])
    tmpl = unit_mean_templates(feat[:2], labels[:2], units)
    lists = [np.array([3, 8]), np.array([3])]
    out = wta_among_lists(feat[2:], tmpl, units, lists)
    summary = summarize_refine(out["pred"], labels[2:], lists, n_units_train=2)
    assert summary["n_eval"] == 2
    assert summary["candidate_recall"] == 0.5
    assert 0.0 <= summary["argmin_accuracy"] <= 1.0


def test_integer_templates_rint_mean():
    codes = np.array([[0, 10], [1, 11], [10, 0]])
    labels = np.array([1, 1, 2])
    tmpl = integer_mean_templates(codes, labels, np.array([1, 2]))
    np.testing.assert_array_equal(tmpl[0], [0, 10])
    np.testing.assert_array_equal(tmpl[1], [10, 0])


def test_bulk_tail_empty_inner_escalates_without_unique_early_exit():
    # Unit 0 centroid 0, unit 1 centroid 20. Most events sit at the centroid;
    # two unit-0 tail events sit at 15, outside a tight inner ball.
    codes = np.zeros((40, 1), dtype=np.int64)
    labels = np.zeros(40, dtype=np.int64)
    codes[20:] = 20
    labels[20:] = 1
    times = np.arange(40, dtype=np.int64)
    codes[9] = 15  # in the first 50% (fit)
    codes[19] = 15  # calibration/test boundary: index 19 is still first half
    # 50/25/25 on 40 rows: fit 0:20, cal 20:30, test 30:40.
    # Put a tail in test for unit 0... but test is all unit 1 with this layout.
    # Rebuild: interleave so both units appear in every third.
    rng_codes = np.zeros((60, 1), dtype=np.int64)
    rng_y = np.zeros(60, dtype=np.int64)
    rng_codes[:30, 0] = 0
    rng_y[:30] = 0
    rng_codes[30:, 0] = 20
    rng_y[30:] = 1
    # Chronological split will put unit 1 entirely in cal+test. Use mixed times
    # already sorted but mixed labels by construction of repeating blocks.
    codes = np.array([[0], [20]] * 30, dtype=np.int64)
    labels = np.array([0, 1] * 30, dtype=np.int64)
    times = np.arange(60, dtype=np.int64)
    codes[50] = 8  # unit 0 tail in test (index 50: label 0)
    rec = bulk_tail_from_codes(codes, labels, times, inner_pctl=50.0, outer_pctl=100.0)
    assert rec["n_test"] == 15
    assert rec["frac_C1"] + rec["frac_Cgt1"] + rec["frac_C0"] == 1.0
    # Unique inner hits remain candidates, not labels: traffic uses |C|>1 only.
    assert rec["traffic_x"] >= 1.0
    assert rec["true_unit_candidate_recall"] >= rec["inner_only_recall"] - 1e-12
