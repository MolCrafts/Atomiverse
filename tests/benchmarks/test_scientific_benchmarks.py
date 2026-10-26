"""Framework unit tests: synthetic observables only, no HF/MD solves."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.framework.comparison import compare_values, evaluate
from benchmarks.framework.execution import run_sample, worker, write_json
from benchmarks.framework.reporting import report, roadmap_status
from benchmarks.framework.specification import Bundle, Cell, Observation, Participant, Row, System


def row() -> Row:
    return Row(id="example.energy", label="Example energy", capability="example.energy", task="energy",
        model={}, systems=[System(id="h2", label="H2", atomic_numbers=[1, 1], positions_angstrom=[[0, 0, 0], [0, 0, 0.74]])],
        participants=[Participant(id="atv.cpu", engine="atv", backend="cpu", role="atv", required=True,
                                  entrypoint="benchmark_probe:prepare"),
                      Participant(id="oracle.cpu", engine="independent", backend="cpu", role="oracle", required=True,
                                  entrypoint="benchmark_probe:prepare")], reference="oracle.cpu",
        checks=[{"observable": "energy", "unit": "Ha", "atol": 1e-8,
                 "justification": "synthetic framework boundary test"}])


def bundle() -> Bundle:
    declaration = row()
    cells = []
    for participant in declaration.participants:
        cells.append(Cell(row_id=declaration.id, system_id="h2", participant_id=participant.id,
                          required=True, observations=[Observation(phase="cold", repeat=0, status="completed", seconds=1,
                              observables={"energy": -1.0}, metadata={"binary_sha256": "a" * 64})] + [Observation(phase="warm", repeat=i + 1, status="completed",
                              seconds=float(i + 1) * (1 if participant.role == "atv" else 2),
                              observables={"energy": -1.0}, metadata={"binary_sha256": "a" * 64}) for i in range(5)]))
    return Bundle(campaign_id="unit-observation", started="2026-10-26T00:00:00Z", tier="performance",
                  repeats=5, warmup=1, threads=1, timeout_seconds=5, profile="synthetic",
                  source={"commit": "a" * 40, "dirty": True, "build_verified": False},
                  environment={"cpu_model": "synthetic", "physical_cores": 1, "logical_cpus": 1,
                               "host_memory_bytes": 4096, "gpus": [], "os": "test", "python": "test"},
                  rows=[declaration], cells=cells, protocol={"selection_partial": False})


@pytest.mark.parametrize("a,b", [([1, 2], [1]), (float("nan"), 1), (float("inf"), 1), ([], [])])
def test_invalid_observables_cannot_pass(a, b):
    assert compare_values(a, b, 1e-8, 0)["pass"] is False


def test_componentwise_tolerance_and_shapes():
    assert compare_values([1, 2], [1, 2], 0, 0)["pass"]
    assert compare_values([[1, 2]], [1, 2], 100, 0)["pass"] is False
    assert compare_values([1, 2.01], [1, 2], 1e-3, 0)["pass"] is False
    assert compare_values([1, 2.01], [1, 2], 0, 1e-2)["pass"]


def test_failed_sample_cannot_be_hidden_by_passing_median():
    data = bundle()
    data.cells[0].observations[-1].observables["energy"] = -0.9
    evaluate(data)
    assert data.cells[0].numerical == "fail"
    assert data.cells[0].speedup is None
    assert data.summary["complete"] is False


def test_median_ratio_and_full_samples():
    data = evaluate(bundle())
    assert data.cells[0].timing["median_seconds"] == 3
    assert len(data.cells[0].timing["samples_seconds"]) == 5
    assert data.cells[0].speedup == 2
    assert data.summary["complete"]


def test_missing_reference_is_not_a_zero_error_or_pass():
    data = bundle()
    data.cells[1].observations = [Observation(phase="cold", repeat=0, status="dependency_missing", reason="not installed")]
    evaluate(data)
    assert all(c.numerical == "unverified" and c.speedup is None for c in data.cells)
    assert data.summary["incomplete"] == 2


def test_empty_evidence_cannot_inherit_previously_derived_pass():
    data = evaluate(bundle())
    assert data.summary["complete"]
    for cell in data.cells:
        cell.observations = []
    restored = Bundle.model_validate(data.model_dump())
    evaluate(restored)
    assert restored.summary["complete"] is False
    assert restored.summary["passed"] == 0
    assert all(c.numerical == "unverified" and c.status == "not_run" for c in restored.cells)


def test_actual_resource_mismatch_prevents_ratio():
    data = bundle()
    data.cells[1].observations[0].metadata["resource_mismatch"] = True
    evaluate(data)
    assert data.cells[0].numerical == "pass"
    assert data.cells[0].speedup is None
    assert "resources" in data.cells[0].speedup_reason


@pytest.mark.parametrize("participant", [0, 1])
@pytest.mark.parametrize("fingerprint", [None, "invalid", ["a" * 64], "b" * 64])
def test_unidentified_or_changed_binary_withholds_ratio_without_erasing_accuracy(participant, fingerprint):
    data = bundle()
    data.cells[participant].observations[-1].metadata["binary_sha256"] = fingerprint
    evaluate(data)
    assert data.summary["complete"]
    assert data.cells[0].numerical == "pass"
    assert data.cells[0].speedup is None
    assert "fingerprint" in data.cells[0].speedup_reason


@pytest.mark.parametrize("mode", ["missing", "duplicate"])
def test_missing_or_duplicated_samples_cannot_complete_campaign(mode):
    data = bundle()
    if mode == "missing":
        data.cells[0].observations.pop()
    else:
        data.cells[0].observations[-1] = data.cells[0].observations[-2].model_copy(deep=True)
    evaluate(data)
    assert not data.summary["complete"]
    assert data.cells[0].numerical == "unverified"
    assert data.cells[0].speedup is None


@pytest.mark.parametrize("bad", ["bad", "1.0", {"value": 1}, [[1], [1, 2]], True])
def test_malformed_observable_is_failure_not_comparison_exception(bad):
    assert compare_values(bad, -1, 1e-8, 0)["pass"] is False


def test_derived_error_overflow_is_reportable_failure():
    assert compare_values(1e308, -1e308, 1e-8, 0)["pass"] is False
    assert compare_values(1e200, 0.0, 1e201, 0)["rms_error"] == 1e200


def test_strict_coverage_cannot_drop_or_demote_required_cells():
    raw = bundle().model_dump()
    raw["cells"] = raw["cells"][:1]
    with pytest.raises(ValueError, match="cover"):
        Bundle.model_validate(raw)
    raw = bundle().model_dump()
    raw["cells"][0]["required"] = False
    with pytest.raises(ValueError, match="required policy"):
        Bundle.model_validate(raw)


def test_nonfinite_nested_observable_is_rejected():
    raw = bundle().model_dump()
    raw["cells"][0]["observations"][0]["observables"]["energy"] = float("nan")
    with pytest.raises(ValueError):
        Bundle.model_validate(raw)


def test_roadmap_never_promotes_dirty_unverified_build():
    data = evaluate(bundle())
    result = roadmap_status(data)
    cpu = next(s for s in result if s["backend"] == "cpu")
    assert cpu["mark"] == "◐"
    assert cpu["numerical_pass"] is True


def test_missing_catalog_never_promotes_even_if_build_is_attested():
    data = evaluate(bundle())
    data.source.update(dirty=False, build_verified=True)
    cpu = next(s for s in roadmap_status(data) if s["backend"] == "cpu")
    assert cpu["mark"] == "◐"
    assert cpu["coverage_complete"] is False


def test_backend_scope_cannot_inherit_an_unmeasured_required_row():
    data = evaluate(bundle())
    data.source.update(dirty=False, build_verified=True)
    data.protocol["catalog"] = {"capabilities": [{"id": "example.energy", "rows": ["example.energy", "other.energy"],
        "cpu": "implemented", "cuda": "not_implemented", "chapter": "hartree-fock.md"}]}
    cpu = next(s for s in roadmap_status(data) if s["backend"] == "cpu")
    assert cpu["mark"] == "◐"
    assert cpu["coverage_complete"] is False


def test_roadmap_command_updates_only_the_evidence_block(tmp_path, monkeypatch):
    from benchmarks import __main__ as cli

    data = evaluate(bundle())
    data.protocol["catalog"] = {"capabilities": [{"id": "example.energy", "rows": ["example.energy"],
        "cpu": "implemented", "cuda": "not_implemented", "chapter": "hartree-fock.md"}]}
    input_path = tmp_path / "result.json"
    write_json(input_path, data.model_dump())
    chapter = tmp_path / "docs/roadmap/hartree-fock.md"
    chapter.parent.mkdir(parents=True)
    chapter.write_text("Before\n<!-- benchmark-evidence:start -->\nold\n<!-- benchmark-evidence:end -->\nAfter\n")
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr("sys.argv", ["benchmarks", "roadmap", "--input", str(input_path), "--write"])
    assert cli.main() == 0
    updated = chapter.read_text()
    assert updated.startswith("Before\n") and updated.endswith("\nAfter\n")
    assert "CPU numerical pass" in updated
    assert "example.energy" in updated
    monkeypatch.setattr("sys.argv", ["benchmarks", "roadmap", "--input", str(input_path), "--check"])
    assert cli.main() == 0


def test_numerical_failure_remains_visible_in_roadmap():
    data = bundle()
    data.cells[0].observations[0].observables["energy"] = 2
    evaluate(data)
    assert next(s for s in roadmap_status(data) if s["backend"] == "cpu")["mark"] == "✗"


def test_acceptance_samples_cannot_substitute_for_performance_tier():
    data = evaluate(bundle())
    data.tier = "acceptance"
    data.source.update(dirty=False, build_verified=True, tree_sha256="tree")
    data.protocol.update(build_receipt={"source_tree_sha256": "tree", "binary_sha256": "a" * 64},
        catalog={"capabilities": [{"id": "example.energy", "rows": ["example.energy"],
            "cpu": "implemented", "cuda": "not_implemented", "chapter": "hartree-fock.md"}]})
    cpu = next(s for s in roadmap_status(data) if s["backend"] == "cpu")
    assert cpu["coverage_complete"] and cpu["official_source"]
    assert cpu["mark"] == "✓"
    data.tier = "performance"
    assert next(s for s in roadmap_status(data) if s["backend"] == "cpu")["mark"] == "⚡"


def test_performance_subset_cannot_replace_required_acceptance_coverage():
    data = evaluate(bundle())
    data.source.update(dirty=False, build_verified=True, tree_sha256="tree")
    data.protocol.update(build_receipt={"source_tree_sha256": "tree", "binary_sha256": "a" * 64},
        coverage_policy={"example.energy": {"systems": [{"id": "h2", "tiers": ["performance"]}]},
                         "other.energy": {"systems": [{"id": "water", "tiers": ["acceptance"]}]}},
        catalog={"capabilities": [{"id": "example.energy", "rows": ["example.energy", "other.energy"],
            "cpu": "implemented", "cuda": "not_implemented", "chapter": "hartree-fock.md"}]})
    cpu = next(s for s in roadmap_status(data) if s["backend"] == "cpu")
    assert cpu["numerical_pass"] and cpu["official_source"]
    assert not cpu["coverage_complete"]
    assert cpu["mark"] == "◐"


def test_report_escapes_labels_preserves_missing_cells_and_hardware(tmp_path):
    data = evaluate(bundle())
    data.rows[0].label = "<script>alert(1)</script>"
    report(data, tmp_path, snapshot=False)
    text = (tmp_path / "matrix.html").read_text()
    assert "<script>" not in text
    assert "&lt;script&gt;" in text
    assert "CPU=synthetic" in text
    assert "Notes: hardware / environment" in text
    assert "2×" in text
    assert (tmp_path / "junit.xml").is_file()
    assert Bundle.model_validate_json((tmp_path / "results.json").read_text()).campaign_id == data.campaign_id


@pytest.fixture
def probe(tmp_path, monkeypatch):
    path = tmp_path / "benchmark_probe.py"
    path.write_text("def prepare(row, system, participant, threads):\n"
                    "    state = []\n"
                    "    def run():\n"
                    "        state.append(1)\n"
                    "        return {'observables': {'energy': -1.0}, 'metadata': {'calls': len(state)}}\n"
                    "    return run\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    return tmp_path


def test_worker_warmup_is_outside_measurement_and_result_is_structured(probe):
    spec = row()
    request, result = probe / "request.json", probe / "result.json"
    write_json(request, {"row": spec.model_dump(), "system": spec.systems[0].model_dump(),
                         "participant": spec.participants[0].model_dump(), "threads": 1, "warmup": 2})
    assert worker(request, result) == 0
    actual = json.loads(result.read_text())
    assert actual["metadata"]["calls"] == 3
    assert actual["seconds"] > 0
    assert actual["status"] == "completed"


def test_worker_missing_dependency_is_a_terminal_result(tmp_path):
    spec = row()
    spec.participants[0].entrypoint = "absent_benchmark_engine_xyz:prepare"
    request, result = tmp_path / "request.json", tmp_path / "result.json"
    write_json(request, {"row": spec.model_dump(), "system": spec.systems[0].model_dump(),
                         "participant": spec.participants[0].model_dump(), "threads": 1, "warmup": 1})
    assert worker(request, result) == 1
    assert json.loads(result.read_text())["status"] == "dependency_missing"


def test_adapter_cannot_override_framework_timing(probe, monkeypatch):
    import benchmark_probe

    monkeypatch.setattr(benchmark_probe, "prepare", lambda *args: lambda: {
        "observables": {"energy": -1}, "metadata": {}, "seconds": 1e-30})
    spec = row()
    request, result = probe / "request.json", probe / "result.json"
    write_json(request, {"row": spec.model_dump(), "system": spec.systems[0].model_dump(),
                         "participant": spec.participants[0].model_dump(), "threads": 1, "warmup": 0})
    assert worker(request, result) == 1
    actual = json.loads(result.read_text())
    assert actual["status"] == "error"
    assert "timing fields" in actual["reason"]


def test_worker_rejects_a_different_native_build(probe):
    spec = row()
    request, result = probe / "request.json", probe / "result.json"
    write_json(request, {"row": spec.model_dump(), "system": spec.systems[0].model_dump(),
                         "participant": spec.participants[0].model_dump(), "threads": 1, "warmup": 0,
                         "expected_binary_sha256": "a" * 64})
    assert worker(request, result) == 1
    assert "build receipt" in json.loads(result.read_text())["reason"]


def test_build_receipt_records_the_successful_cmake_binary(tmp_path, monkeypatch):
    import benchmarks.framework.build as build
    from benchmarks.framework.provenance import file_hash

    binary = tmp_path / "native/_atomiverse.so"
    binary.parent.mkdir()
    binary.write_bytes(b"synthetic native build output")
    (binary.parent / "CMakeCache.txt").write_text(f"CMAKE_HOME_DIRECTORY:INTERNAL={tmp_path}\n")
    source = {"tree_sha256": "source", "commit": "a" * 40, "dirty": False}
    monkeypatch.setattr(build, "source_info", lambda *args: dict(source))
    monkeypatch.setattr(build, "command", lambda *args: None)
    calls = []
    monkeypatch.setattr(build.subprocess, "run", lambda args, **kwargs: calls.append(args))
    receipt = build.attest_build(tmp_path, binary.parent, tmp_path / "receipt.json")
    assert calls[0][:2] == ["cmake", "--build"]
    assert receipt["binary_sha256"] == file_hash(binary)
    assert build.verify_receipt(tmp_path / "receipt.json", source) == receipt
    with pytest.raises(ValueError, match="current source"):
        build.verify_receipt(tmp_path / "receipt.json", {**source, "tree_sha256": "changed"})


def test_build_receipt_rejects_source_edits_during_build(tmp_path, monkeypatch):
    import benchmarks.framework.build as build

    (tmp_path / "CMakeCache.txt").write_text(f"CMAKE_HOME_DIRECTORY:INTERNAL={tmp_path}\n")
    sources = iter([{"tree_sha256": "before"}, {"tree_sha256": "after"}])
    monkeypatch.setattr(build, "source_info", lambda *args: next(sources))
    monkeypatch.setattr(build.subprocess, "run", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="changed during"):
        build.attest_build(tmp_path, tmp_path, tmp_path / "receipt.json")
    assert not (tmp_path / "receipt.json").exists()


def test_output_files_do_not_masquerade_as_source_mutations(tmp_path, monkeypatch):
    import benchmarks.framework.provenance as provenance

    (tmp_path / "input.py").write_text("source")
    out = tmp_path / "out"
    out.mkdir()
    result = out / "results.json"
    result.write_text("first observation")
    def git(args, cwd=None):
        return {"status": "?? out/", "diff": "", "ls-files": "input.py\nout/results.json",
                "rev-parse": "a" * 40}[args[1]]
    monkeypatch.setattr(provenance, "command", git)
    before = provenance.source_info(tmp_path, exclude=(out,))
    result.write_text("second observation")
    after = provenance.source_info(tmp_path, exclude=(out,))
    assert before["tree_sha256"] == after["tree_sha256"]
    assert list(after["source_files"]) == ["input.py"]


def test_controller_timeout_is_terminal_and_logged(tmp_path):
    spec = row()
    result = run_sample(spec, spec.systems[0], spec.participants[0], phase="cold", repeat=0,
                        threads=1, warmup=0, timeout=0.001, out=tmp_path, root=Path.cwd())
    assert result.status == "timeout"
    assert list(tmp_path.glob("*.log"))
    assert result.process_seconds > 0
    terminal = tmp_path.parent / result.artifact
    assert terminal.is_file()
    assert json.loads(terminal.read_text())["status"] == "timeout"
    assert (tmp_path.parent / result.metadata["log_artifact"]).is_file()
