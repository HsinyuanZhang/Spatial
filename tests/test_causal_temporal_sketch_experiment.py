import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.algorithms.causal_temporal_sketch import CausalTemporalSketch
from Spatial.data.loader import Dataset
from Spatial.experiments import run_causal_temporal_sketch as runner
from Spatial.experiments import causal_temporal_selection as selection
from Spatial.experiments.causal_temporal_selection import PILOT_DATASET_IDS, validate_development_rows


def test_prepare_uses_one_filter_pass_and_preserves_joint_rows_ties_and_digest(monkeypatch):
    calls = []

    def identity_filter(data, fs, low, high, order):
        calls.append((fs, low, high, order))
        return np.asarray(data, dtype=float)

    monkeypatch.setattr(runner, "bandpass_filter", identity_filter)
    samples = np.arange(100, dtype=float)
    tied_channel_waveform = np.sin(samples / 3.0)
    dataset = Dataset(
        raw_data=np.tile(tied_channel_waveform, (7, 1)),
        geom=np.column_stack([np.arange(7), np.zeros(7)]),
        fs=30_000,
        spike_times=np.array([14, 20, 20, 52]),
        spike_units=np.array([10, 11, 12, 13]),
        name="synthetic",
    )

    prepared = runner.prepare_causal_events(dataset)

    assert calls == [(30_000, 300.0, 6000.0, 3)]
    # 14 is outside the left 15-sample peak window; 52 fails 52 + 48 < 100.
    np.testing.assert_array_equal(prepared["times"], [20, 20])
    np.testing.assert_array_equal(prepared["event_row_id"], [1, 2])
    np.testing.assert_array_equal(prepared["labels"], [11, 12])
    # Identical per-channel peaks use np.argmax's first (smallest) channel.
    np.testing.assert_array_equal(prepared["central"], [0, 0])
    assert prepared["descriptor"].shape == (2, 9)
    assert prepared["raw_waveforms"].shape == (2, 64)
    assert prepared["legacy_normalized_waveforms"].shape == (2, 64)
    assert prepared["provenance"]["row_digest"] == runner._digest_rows(
        prepared["event_row_id"], prepared["times"], prepared["labels"], prepared["central"]
    )
    assert prepared["provenance"]["filter"]["zero_phase"] is True
    assert prepared["provenance"]["recording_input"] == "dataset.raw_data (channel x sample)"
    assert prepared["provenance"]["sampling_frequency_hz"] == 30_000
    assert prepared["provenance"]["temporal_input"] == "filtered_pre_event_normalization_main_channel"
    assert prepared["provenance"]["window"]["interval"] == "[t-15, t+49)"


def test_prepare_fails_fast_when_joint_boundary_filter_removes_every_row(monkeypatch):
    monkeypatch.setattr(runner, "bandpass_filter", lambda data, *args: np.asarray(data, float))
    dataset = Dataset(
        raw_data=np.ones((7, 100)), geom=np.column_stack([np.arange(7), np.zeros(7)]), fs=30_000,
        spike_times=np.array([14, 52]), spike_units=np.array([1, 2]), name="empty",
    )
    try:
        runner.prepare_causal_events(dataset)
    except ValueError as error:
        assert "no events remain" in str(error)
    else:
        raise AssertionError("empty causal row set must fail fast")


def test_split_keeps_equal_times_and_pair_support_is_distinct():
    a, b, c = runner.stable_three_way_split(np.array([0, 0, 1, 1, 2, 2, 3, 3]))
    np.testing.assert_array_equal(a, np.arange(4))
    np.testing.assert_array_equal(b, np.arange(4, 6))
    np.testing.assert_array_equal(c, np.arange(6, 8))
    rows = [np.array([9, 2, 9]), np.array([2, 5])]
    assert runner.distinct_unordered_pairs(rows) == {(2, 9), (2, 5)}
    summary = runner.candidate_support(rows, np.array([2, 5]))
    assert summary["ambiguous_events"] == 2 and summary["distinct_unordered_pairs"] == 2


def test_uniform_source_is_d9_finite_calibration_only_and_includes_radius_boundary():
    fit = np.vstack([np.zeros((10, 9)), np.full((10, 9), 0.5)])
    labels = np.array([1] * 10 + [9] * 10)
    # Unit 1's calibration row is exactly L1 distance one code from its centroid.
    calibration = np.vstack([np.r_[1.0 / 31.0, np.zeros(8)], np.full(9, 0.5)])
    calibration_labels = np.array([1, 9])
    source = runner.frozen_uniform_candidate_source(
        fit, labels, calibration, calibration_labels, calibration[:1]
    )
    assert source["calibration_radius_percentile"] == 99.9
    np.testing.assert_array_equal(source["calibration_counts"], [1, 1])
    assert not source["calibration_fallback_mask"].any()
    assert source["radius_source"].tolist() == ["calibration_p99.9_l1"] * 2
    assert source["radii"][0] == 1
    assert 1 in source["candidates"][0]  # query distance == calibrated radius is inclusive
    assert source["all_rows"].shape == (1, 2) and source["all_rows"].all()
    assert source["payload54"]["bits_per_row"] == 54
    # A unit absent from calibration retains its frozen fit radius and is auditable.
    fallback = runner.frozen_uniform_candidate_source(fit, labels, calibration[:1], calibration_labels[:1], calibration[:1])
    np.testing.assert_array_equal(fallback["calibration_counts"], [1, 0])
    np.testing.assert_array_equal(fallback["calibration_fallback_mask"], [False, True])
    assert fallback["radius_source"].tolist() == ["calibration_p99.9_l1", "no_calibration_rows_fit_p99.9"]
    assert fallback["radii"][1] == fallback["model"].radii_[1]
    try:
        runner.frozen_uniform_candidate_source(np.zeros((1, 8)), [1], np.zeros((1, 9)), [1], np.zeros((1, 9)))
    except ValueError as error:
        assert "D9" in str(error)
    else:
        raise AssertionError("non-D9 candidate source input must be rejected")


def test_uniform_source_uses_one_distance_matrix_not_model_query_and_matches_query_codes(monkeypatch):
    fit = np.vstack([np.zeros((10, 9)), np.full((10, 9), 0.5)])
    labels = np.array([1] * 10 + [9] * 10)
    calibration = fit[[0, 10]]
    test = np.vstack([np.zeros(9), np.full(9, 0.5), np.full(9, 1.0)])
    monkeypatch.setattr(
        runner.AdaptiveRangeSearch, "query",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("query must not be called")),
    )
    source = runner.frozen_uniform_candidate_source(fit, labels, calibration, np.array([1, 9]), test)
    expected, expected_diag = source["model"].query_codes(
        source["codes"], mode="l1", active_mask=source["all_rows"]
    )
    assert [row.tolist() for row in source["candidates"]] == [row.tolist() for row in expected]
    assert source["diagnostics"]["row_comparisons"].tolist() == [2, 2, 2]
    assert source["level1_source_unit_count"] == 2
    assert source["level1_source_descriptor_quantizations"] == 3 * 9
    assert source["level1_source_radius_threshold_comparisons"] == 3 * 2
    assert source["level1_source_argmin_reduction_comparisons"] == 3
    np.testing.assert_array_equal(
        source["diagnostics"]["final_candidate_count"], expected_diag["final_candidate_count"]
    )
    assert len(source["candidates"][-1]) == 0


def test_true_tail_perturbation_checks_core_codes_features_predictions_and_distances():
    fit = np.zeros((20, 64)); fit[10:] = 0.2
    sketch = CausalTemporalSketch(delay_samples=8, n_features=8).fit(fit, [1] * 10 + [9] * 10)
    assert runner.assert_future_tail_invariance(
        sketch, np.zeros((2, 64)), [np.array([1, 9]), np.array([], dtype=int)]
    )


def _prepared_for_integration(n_events=160):
    times = np.arange(n_events, dtype=np.int64) * 100
    labels = np.arange(n_events, dtype=np.int64) % 2 + 1
    raw = np.zeros((n_events, 64), dtype=float)
    raw[labels == 2, 20:42] = 0.4
    raw[labels == 1, 8:30] = -0.4
    # Deliberately identical D9 rows keep the frozen source ambiguous, so the
    # selection artifact has finite ambiguous metrics and temporal traffic.
    descriptor = np.full((n_events, 9), 0.5, dtype=float)
    event_ids = np.arange(n_events, dtype=np.int64)
    central = np.zeros(n_events, dtype=np.int64)
    return {
        "event_row_id": event_ids, "times": times, "labels": labels, "central": central,
        "descriptor": descriptor, "raw_waveforms": raw,
        "legacy_normalized_waveforms": raw / np.maximum(np.abs(raw).max(axis=1, keepdims=True), 1e-8),
        "provenance": {
            "row_digest": runner._digest_rows(event_ids, times, labels, central),
            "n_events": n_events,
            "recording_input": "dataset.raw_data (channel x sample)",
            "sampling_frequency_hz": 30_000,
            "temporal_input": "filtered_pre_event_normalization_main_channel",
            "filter": {
                "kind": "Butterworth bandpass", "low_cutoff_hz": 300.0,
                "high_cutoff_hz": 6000.0, "order": 3, "zero_phase": True,
                "implementation": "scipy.signal.filtfilt(axis=1)",
            },
            "alignment_source": "get_peak_amplitudes(filtered, dataset.spike_times, window=PRE_ALIGNMENT_SAMPLES) preserves boundary-filtered input order",
            "central_channel_source": "get_peak_amplitudes: np.argmax(abs(per-channel signed peak amplitude)); exact ties select the smallest channel index",
            "central_post_alignment_lookahead_samples": 14,
            "window": {"pre_alignment_samples": 15, "post_alignment_samples": 48, "interval": "[t-15, t+49)", "samples": 64},
            "raw_waveform_source": "filtered central channel",
            "legacy_waveform_source": "per-row max-absolute normalized raw_waveforms",
        },
    }


def _record(dataset_id, family):
    return runner.PilotRecordSpec(
        dataset_id=dataset_id,
        family=family,
        source_input_path=f"/synthetic/{dataset_id}.npz",
        source_file_sha256=runner.LOCKED_PILOT_FILE_SHA256[dataset_id],
        source_input_subset_digest=runner.LOCKED_INPUT_SUBSET_SHA256,
        load_dataset=lambda: object(),
    )


def test_no_io_prepared_integration_emits_frozen_15_grid_baselines_and_selection_schema():
    prepared = _prepared_for_integration()
    rows_by_family = {family: [] for family in PILOT_DATASET_IDS}
    for family, dataset_ids in PILOT_DATASET_IDS.items():
        for dataset_id in dataset_ids:
            rows = runner.run_prepared_recording(prepared, record=_record(dataset_id, family))
            assert len(rows) == 15
            assert len({row["candidate_source_digest"] for row in rows}) == 1
            assert len({repr(row["candidate_source_summary"]) for row in rows}) == 1
            assert "candidate_lists" not in rows[0] and "baseline_predictions" not in rows[0]
            for row in rows:
                assert row["full_teacher_overall_correct"] >= 0
                assert row["legacy_full64_overall_correct"] >= 0
                assert row["signed_full64x5_overall_correct"] >= 0
                assert row["fisher48_signed5_overall_correct"] >= 0
                assert row["morphology12_unsigned5_overall_correct"] >= 0
                assert row["candidate_template_reads"] == row["logical_candidate_template_reads"]
                assert row["future_sample_isolation_ok"]
                assert row["source_file_sha256"] == runner.LOCKED_PILOT_FILE_SHA256[dataset_id]
                assert row["temporal_input_contract_sha256"] == runner.TEMPORAL_INPUT_CONTRACT_SHA256
                assert row["level1_c0_reuse_events"] == row["c0_events"]
                assert row["repository_base_revision"] == runner.BASE_REPOSITORY_REVISION
                assert row["implementation_source_digest"] == runner.sha256_json(row["implementation_source_sha256"])
            rows_by_family[family].extend(rows)
    normalized = validate_development_rows(rows_by_family)
    assert {family: len(rows) for family, rows in normalized.items()} == {"hj": 30, "mearec": 30}


def test_pilot_main_writes_only_pilot_artifacts_and_never_discovers_confirmation(monkeypatch, tmp_path):
    prepared = _prepared_for_integration()
    discovered = {
        family: [_record(dataset_id, family) for dataset_id in dataset_ids]
        for family, dataset_ids in PILOT_DATASET_IDS.items()
    }
    monkeypatch.setattr(runner, "discover_pilot_datasets", lambda **_: discovered)
    monkeypatch.setattr(runner, "prepare_causal_events", lambda _: prepared)
    runner.main(["--pilot", "--output-dir", str(tmp_path)])
    assert (tmp_path / "causal_temporal_sketch_hj_pilot.csv").exists()
    assert (tmp_path / "causal_temporal_sketch_mearec_pilot.csv").exists()
    assert (tmp_path / "causal_temporal_sketch_selection.json").exists()
    assert not list(tmp_path.glob("*confirmation*"))
    try:
        runner.main(["--pilot", "--duration", "59"])
    except SystemExit:
        pass
    else:
        raise AssertionError("pilot duration must be locked to 60 seconds")


def test_pilot_discovery_sha_gate_returns_lazy_loaders_without_loading(monkeypatch, tmp_path):
    drift_target, drift = tmp_path / "drift_target.npz", tmp_path / "drift.npz"
    drift_target.write_bytes(b"placeholder")
    drift.symlink_to(drift_target)
    static = tmp_path / "static.npz"
    ten, twenty = tmp_path / "rec_v1_units10_snr5_seed202601.npz", tmp_path / "rec_v1_units20_snr5_seed202601.npz"
    for path in (static, ten, twenty):
        path.write_bytes(b"placeholder")
    locked_hashes = {
        "hybrid_janelia_drift16c_600s_11": runner._sha256_file(drift),
        "hybrid_janelia_static16c_600s_11": runner._sha256_file(static),
        "mearec_rec_v1_units10_snr5_seed202601": runner._sha256_file(ten),
        "mearec_rec_v1_units20_snr5_seed202601": runner._sha256_file(twenty),
    }
    monkeypatch.setattr(runner, "LOCKED_PILOT_FILE_SHA256", locked_hashes)
    monkeypatch.setattr(runner, "list_hybrid_janelia_scenes", lambda **_: [
        {"short_name": "drift16c_600s_11", "npz_path": str(drift)},
        {"short_name": "static16c_600s_11", "npz_path": str(static)},
    ])
    monkeypatch.setattr(runner, "discover_npz", lambda _: [ten, twenty])
    calls = []
    def checked_hj_loader(*, path=None, scene=None, duration_s=None):
        calls.append((path, scene, duration_s))
        return object()
    monkeypatch.setattr(runner, "load_hybrid_janelia", checked_hj_loader)
    monkeypatch.setattr(runner, "load_mearec_npz", lambda *_ , **__: (_ for _ in ()).throw(AssertionError("eager MEArec load")))
    discovered = runner.discover_pilot_datasets(mearec_npz_dir=tmp_path)
    assert {family: [item.dataset_id for item in rows] for family, rows in discovered.items()} == {
        family: list(ids) for family, ids in PILOT_DATASET_IDS.items()
    }
    assert all(callable(record.load_dataset) for rows in discovered.values() for record in rows)
    assert not calls  # discovery verifies bytes and returns factories; it does not load recordings.
    alternate = tmp_path / "drift_alternate.npz"
    alternate.write_bytes(b"placeholder")
    drift.unlink()
    drift.symlink_to(alternate)
    discovered["hj"][0].load_dataset()
    assert calls == [(drift_target.resolve(), None, 60.0)]
    assert discovered["hj"][0].source_input_path == str(drift_target.resolve())
    assert discovered["hj"][0].source_file_sha256 == locked_hashes["hybrid_janelia_drift16c_600s_11"]
    drift_target.write_bytes(b"changed after discovery")
    try:
        discovered["hj"][0].load_dataset()
    except ValueError as error:
        assert "changed before load" in str(error)
    else:
        raise AssertionError("lazy load must re-check the verified file to close TOCTOU")
    drift_target.write_bytes(b"placeholder")
    twenty.write_bytes(b"changed before second discovery")
    try:
        runner.discover_pilot_datasets(mearec_npz_dir=tmp_path)
    except ValueError as error:
        assert "exactly the two locked mearec" in str(error)
    else:
        raise AssertionError("a mismatched locked pilot hash must be rejected")


def test_runner_fails_closed_for_nonfinite_json_and_wrong_legacy_or_temporal_input_route():
    for value in (float("nan"), float("inf"), -float("inf"), np.float64("nan")):
        try:
            runner.sha256_json({"bad": value})
        except ValueError:
            pass
        else:
            raise AssertionError("non-finite JSON input must fail closed")
    prepared = _prepared_for_integration()
    prepared["legacy_normalized_waveforms"] = prepared["legacy_normalized_waveforms"].copy()
    prepared["legacy_normalized_waveforms"][0, 0] += 0.125
    try:
        runner.run_prepared_recording(
            prepared, record=_record(PILOT_DATASET_IDS["hj"][0], "hj")
        )
    except ValueError as error:
        assert "legacy_normalized_waveforms" in str(error)
    else:
        raise AssertionError("a normalized waveform must never silently route into primary sketch input")
    drifted = _prepared_for_integration()
    drifted["provenance"] = dict(drifted["provenance"])
    drifted["provenance"]["filter"] = dict(drifted["provenance"]["filter"])
    drifted["provenance"]["filter"]["high_cutoff_hz"] = 5999.0
    try:
        runner.run_prepared_recording(
            drifted, record=_record(PILOT_DATASET_IDS["hj"][0], "hj")
        )
    except ValueError as error:
        assert "temporal provenance" in str(error)
    else:
        raise AssertionError("filter/provenance drift must fail closed")
    for key, bad_value in (("n_events", 159), ("recording_input", "other"), ("sampling_frequency_hz", 20_000)):
        drifted = _prepared_for_integration()
        drifted["provenance"] = dict(drifted["provenance"])
        drifted["provenance"][key] = bad_value
        try:
            runner.run_prepared_recording(
                drifted, record=_record(PILOT_DATASET_IDS["hj"][0], "hj")
            )
        except ValueError as error:
            assert "temporal provenance" in str(error)
        else:
            raise AssertionError(f"{key} drift must fail closed")
    record = _record(PILOT_DATASET_IDS["hj"][0], "hj")
    wrong_record = runner.PilotRecordSpec(
        dataset_id=record.dataset_id, family=record.family, source_input_path=record.source_input_path,
        source_file_sha256="0" * 64, source_input_subset_digest=record.source_input_subset_digest,
        load_dataset=record.load_dataset,
    )
    try:
        runner.run_prepared_recording(_prepared_for_integration(), record=wrong_record)
    except ValueError as error:
        assert "source metadata" in str(error)
    else:
        raise AssertionError("callers must not decouple a loader from locked source metadata")

    low_amplitude = _prepared_for_integration()
    low_amplitude["raw_waveforms"] = low_amplitude["raw_waveforms"] * 5e-9
    low_amplitude["legacy_normalized_waveforms"] = runner.legacy_peak_normalize(low_amplitude["raw_waveforms"])
    raw_rows = runner.run_prepared_recording(low_amplitude, record=record)
    # Raw 2e-9-scale rows produce a very negative power-of-two exponent;
    # routing per-event-normalized rows into sketch would instead be near -2.
    assert raw_rows[0]["scale_exponent"] < -10


def test_c0_uses_frozen_level1_argmin_and_csv_roundtrip_is_deterministic(tmp_path):
    fit = np.vstack([np.zeros((10, 9)), np.full((10, 9), 0.5)])
    labels = np.array([1] * 10 + [9] * 10)
    source = runner.frozen_uniform_candidate_source(
        fit, labels, fit[:2], labels[:2], np.full((1, 9), 1.0)
    )
    assert len(source["candidates"][0]) == 0
    reused = runner._with_spatial_fallback(
        np.array([None], dtype=object), source["candidates"], source
    )
    assert reused[0] == source["level1_argmin_predictions"][0]
    assert source["level1_source_centroid_reads"] == 2
    rows = [{"z": {"b": 2, "a": 1}, "a": 3}, {"z": {"a": 4}, "a": 5}]
    csv_path = tmp_path / "rows.csv"
    runner.write_rows_csv(csv_path, rows)
    first = csv_path.read_bytes()
    runner.write_rows_csv(csv_path, rows)
    assert csv_path.read_bytes() == first


def test_csv_roundtrip_detects_tampered_readback(monkeypatch, tmp_path):
    monkeypatch.setattr(runner.csv, "DictReader", lambda _: iter([{"a": "tampered"}]))
    try:
        runner.write_rows_csv(tmp_path / "bad.csv", [{"a": 1}])
    except AssertionError as error:
        assert "round-trip" in str(error)
    else:
        raise AssertionError("read-back cell mismatches must fail")


def test_verified_loader_rejects_source_mutation_during_load(tmp_path):
    source = tmp_path / "source.npz"
    source.write_bytes(b"before")
    expected = runner._sha256_file(source.resolve())

    def mutating_loader(path):
        assert path == source.resolve()
        path.write_bytes(b"after")
        return object()

    try:
        runner._load_verified_path(source.resolve(), expected, mutating_loader)
    except ValueError as error:
        assert "during load" in str(error)
    else:
        raise AssertionError("post-load SHA validation must reject a changed source")


def test_implementation_snapshot_uses_selection_source_set_and_verified_git_base(monkeypatch):
    assert runner.IMPLEMENTATION_SOURCE_PATHS is selection.IMPLEMENTATION_SOURCE_PATHS
    snapshot = runner.implementation_source_snapshot()
    assert set(snapshot["implementation_source_sha256"]) == set(selection.IMPLEMENTATION_SOURCE_PATHS)
    class Result:
        stdout = "different-base\n"
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: Result())
    try:
        runner.implementation_source_snapshot()
    except ValueError as error:
        assert "HEAD" in str(error)
    else:
        raise AssertionError("an unverified repository base revision must fail closed")
