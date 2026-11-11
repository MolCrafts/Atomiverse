"""HF declaration/input contracts. No SCF, subprocesses, network or GPU runtime."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from types import SimpleNamespace
import tomllib

import numpy as np
import pytest

from benchmarks.scf import hf
from benchmarks.scf.bench_ghf import SPEC as GHF
from benchmarks.scf.bench_rhf import SPEC as RHF
from benchmarks.scf.bench_rohf import SPEC as ROHF
from benchmarks.scf.bench_uhf import SPEC as UHF

METHODS = (RHF, UHF, ROHF, GHF)
ROWS = [row for method in METHODS for row in method.rows]
ROOT = Path(__file__).resolve().parents[2]


def participant(row, name):
    return next(p for p in row.participants if p.id == name)


def checks(row):
    return {c.observable: c for c in row.checks}


def test_shared_library_discovery_covers_linux_and_macos_without_source_files(tmp_path):
    native = ["libcgto.so", "libcint.so.6", "libcgto.dylib"]
    for name in native + ["libcgto.c", "libcgto.so.a"]:
        (tmp_path / name).write_bytes(name.encode())
    (tmp_path / "directory.so").mkdir()
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "libgvhf.dylib").write_bytes(b"gpu shared library")

    assert hf._shared_libraries(tmp_path) == sorted(tmp_path / name for name in native)
    assert hf._shared_libraries(tmp_path, recursive=True) == sorted(
        [tmp_path / name for name in native] + [nested / "libgvhf.dylib"])


def test_hf_matrix_has_four_methods_and_twenty_distinct_scientific_rows():
    assert {m.id: len(m.rows) for m in METHODS} == {"rhf": 7, "uhf": 6, "rohf": 4, "ghf": 3}
    assert len({r.id for r in ROWS}) == 20
    for method in METHODS:
        assert all(r.id.startswith(method.id + ".") for r in method.rows)
        assert {r.model["method"] for r in method.rows} == {method.id.upper()}
        assert all(r.model["comparison_class"] == "matched_accuracy" for r in method.rows)


@pytest.mark.parametrize("row", ROWS, ids=lambda r: r.id)
def test_each_row_names_required_independent_live_reference(row):
    reference = participant(row, row.reference)
    assert (reference.id, reference.engine, reference.backend, reference.role) == (
        "pyscf.cpu", "pyscf", "cpu", "oracle")
    assert reference.required and reference.available
    assert reference.entrypoint == "benchmarks.scf.hf:pyscf_adapter"
    assert all(p.engine != reference.engine for p in row.participants if p.role == "atv")
    assert row.checks and all(c.justification for c in row.checks)
    assert row.cuda_only


def test_cpu_hf_has_one_automatic_exact_strategy():
    for method in (RHF, UHF, ROHF, GHF):
        for row in method.rows:
            cpu = [p for p in row.participants if p.engine == "atomiverse" and p.backend == "cpu"]
            assert len(cpu) == 1
            assert cpu[0].id == "atv.cpu" and cpu[0].required and cpu[0].available
            assert cpu[0].options == {"max_memory_mb": 4000.0}
    assert all(not participant(row, "atv.cuda").available for row in GHF.rows)
    assert not any(r.task == "forces" for r in GHF.rows)


def test_triplet_o2_minao_and_hcore_remain_distinct_rows():
    branches = [r for r in UHF.rows if r.model["basis"] == "STO-3G"]
    assert {(r.task, r.model["init_guess"]) for r in branches} == {
        ("energy", "minao"), ("forces", "minao"), ("energy", "hcore"), ("forces", "hcore")}
    for row in branches:
        oxygen = next(s for s in row.systems if s.id == "o2_triplet")
        assert oxygen.atomic_numbers == [8, 8] and (oxygen.charge, oxygen.spin) == (0, 2)
        assert oxygen.positions_angstrom == [[0, 0, 0], [0, 0, 1.2075]]
        assert participant(row, "atv.cuda").available == (row.model["init_guess"] == "minao")
    assert all(not participant(r, "atv.cuda").available for r in UHF.rows if r.model["basis"] == "def2-SVP")


def test_device_force_envelope_does_not_inherit_rhf_energy_support():
    for row in RHF.rows:
        cuda = participant(row, "atv.cuda")
        assert cuda.available == (row.task == "energy" or row.model["basis"] == "STO-3G")
        if not cuda.available:
            assert cuda.unavailable_reason
    assert all(not participant(row, "atv.cuda").available for row in ROHF.rows)


def test_force_workloads_have_separate_rows_nonzero_inputs_and_physical_units():
    force_rows = [r for r in ROWS if r.task == "forces"]
    assert {r.model["method"] for r in force_rows} == {"RHF", "UHF", "ROHF"}
    for row in force_rows:
        assert row.model["nonzero_force"] is True
        force = checks(row)["forces_hartree_per_bohr"]
        assert force.unit == "Ha/Bohr" and 0 < force.atol <= 1e-6 and force.rtol == 0
        assert all(len(s.atomic_numbers) >= 2 for s in row.systems)
        assert all(len({tuple(p) for p in s.positions_angstrom}) > 1 for s in row.systems)
    stretched = next(s for r in force_rows for s in r.systems if s.id == "h2_stretched")
    assert stretched.positions_angstrom == [[0, 0, 0], [0, 0, 0.90]]
    assert all("forces_hartree_per_bohr" not in checks(r) for r in ROWS if r.task == "energy")


def test_basis_conventions_and_accuracy_gates_are_declared():
    for row in ROWS:
        basis = row.model["basis"]
        assert row.model["cartesian"] is (basis in {"STO-3G", "6-31G*"})
        energy = checks(row)["energy_hartree"]
        assert energy.unit == "Ha" and energy.rtol == 0
        assert energy.atol <= (1e-6 if any(s.id == "water_chain16" for s in row.systems) else 1e-8)
        if basis != "STO-3G":
            assert all("smoke" not in s.tiers for s in row.systems)
    assert {r.model["basis"] for r in RHF.rows} == {"STO-3G", "6-31G*", "def2-SVP"}
    assert all("def2-SVP" in {r.model["basis"] for r in m.rows} for m in METHODS)


def test_water_chain_matches_canonical_atoms_coordinates_and_112_ao_count():
    row = next(r for r in RHF.rows if r.id == "rhf.sto3g.water_chain16.energy")
    system = row.systems[0]
    assert system.tiers == ["performance"] and system.atomic_numbers == [8, 1, 1] * 16
    expected = [[4.0*w, y, z] for w in range(16) for y, z in (
        (0.0, 0.1173), (0.7572, -0.4692), (-0.7572, -0.4692))]
    np.testing.assert_array_equal(np.asarray(system.positions_angstrom), expected)
    assert len(expected) == 48
    assert sum({1: 1, 8: 5}[z] for z in system.atomic_numbers) == 112
    assert row.model["conv_tol"] == 1e-10
    assert checks(row)["energy_hartree"].atol == 1e-6


def test_noncollinear_h3_is_a_separate_seed_and_density_gate():
    row = next(r for r in GHF.rows if r.model["neel"])
    system = row.systems[0]
    assert system.id == "h3_triangle" and system.atomic_numbers == [1, 1, 1]
    assert system.spin == 1 and "smoke" not in system.tiers
    assert row.model["spin_rotation"] == pytest.approx(2 * math.pi / 3)
    assert system.positions_angstrom == [[0, 0, 0], [1, 0, 0], [.5, .8660254037844386, 0]]
    assert checks(row)["spin_mixing_norm"].atol <= 1e-4


def test_published_molpy_distribution_version_is_used(monkeypatch):
    requested = []
    def version(name):
        requested.append(name)
        return "0.20.0"
    monkeypatch.setattr(hf.importlib.metadata, "version", version)
    assert hf._version("molpy") == "0.20.0"
    assert requested == ["molcrafts-molpy"]


@pytest.fixture
def lightweight_public_engines(monkeypatch, tmp_path):
    """Contract doubles only: no molecule, density, integrals or SCF algorithm."""
    binary = tmp_path / "fake_hf_extension.so"
    binary.write_bytes(b"unit-test binary identity")
    events = []
    serial = iter(range(100))

    class Block(dict):
        def insert(self, key, value):
            self[key] = value

    class Frame(dict):
        def __init__(self):
            self.meta = {}
            events.append("frame")

    class PublicSolver:
        __module__ = "fake_hf_extension"
        def __init__(self, **settings):
            self.token = next(serial)
            self.__dict__.update(settings)
            self.level_shift = 0.0
            events.append(("solver", self.token))
        def run(self, frame):
            assert frame["atoms"]["atomic_number"].dtype == np.uint32
            assert all(frame["atoms"][key].dtype == np.float64 for key in ["x", "y", "z", "mass"])
            frame.meta["energy"] = -hf.HARTREE_IN_EV
            events.append(("run", self.token))
        def energy(self):
            return -1.0
        def converged(self):
            return True
        def iterations(self):
            return 3
        def last_scf_on_device(self):
            return True
        def last_force_on_device(self):
            return False

    native = SimpleNamespace(RHF=PublicSolver, CudaRHF=PublicSolver,
        backend_info=lambda: {"openmp_threads": 1},
        cuda_backend_info=lambda: {"synchronized": True},
        cuda_synchronize=lambda: events.append("sync"))
    fake_molpy = SimpleNamespace(Frame=Frame, Block=Block,
                                Box=SimpleNamespace(cube=lambda length: length))
    pools = SimpleNamespace(threadpool_limits=lambda **kwargs: object(),
                            threadpool_info=lambda: [{"num_threads": 1}])
    modules = {"numpy": np, "molpy": fake_molpy, "atomiverse.scf": native,
               "threadpoolctl": pools, "fake_hf_extension": SimpleNamespace(__file__=str(binary))}
    real_import = hf.importlib.import_module
    monkeypatch.setattr(hf.importlib, "import_module", lambda name: modules[name] if name in modules else real_import(name))
    monkeypatch.setattr(hf, "_version", lambda name: "0.20.0" if name == "molpy" else "unit")
    return events, binary


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_public_adapter_fresh_state_and_actual_settings_metadata(lightweight_public_engines, backend):
    events, binary = lightweight_public_engines
    row = RHF.rows[0]
    p = participant(row, "atv.cpu" if backend == "cpu" else "atv.cuda")
    run = hf.atv_adapter(row, row.systems[0], p, threads=1)
    assert events == []  # Factory may prepare runtime/metadata, never scientific state.
    outputs = [run(), run()]
    constructors = [e for e in events if isinstance(e, tuple) and e[0] == "solver"]
    executions = [e for e in events if isinstance(e, tuple) and e[0] == "run"]
    assert constructors == [("solver", 0), ("solver", 1)]
    assert executions == [("run", 0), ("run", 1)]
    assert events.count("frame") == 2
    meta = outputs[0]["metadata"]
    assert meta["binary_path"] == str(binary)
    assert meta["binary_sha256"] == hashlib.sha256(binary.read_bytes()).hexdigest()
    assert meta["molpy_version"] == "0.20.0" and isinstance(meta["package_version"], str)
    assert meta["comparison_class"] == "matched_accuracy" and not meta["resource_mismatch"]
    assert "declared" not in meta["actual_engine_config"]
    actual = meta["resolved_settings"]
    assert actual["converged"] is True and isinstance(actual["iterations"], int)
    assert isinstance(actual["tol"], float) and isinstance(actual["max_iter"], int)
    if backend == "cpu":
        assert actual["max_memory_mb"] == 4000.0
        assert isinstance(actual["conv_tol_grad"], float)
        assert isinstance(actual["diis_space"], int)
    else:
        assert isinstance(actual["density_residual_tol"], float)
        assert isinstance(actual["orbital_residual_tol"], float)
        assert actual["orbital_residual_tol"] != row.model["conv_tol_grad"]
        assert isinstance(actual["conv_tol_grad"], str) and "not_exposed" in actual["conv_tol_grad"]
        assert actual["diis_space"] == "not_exposed"
        assert events.count("sync") == 2


def test_cuda_adapter_does_not_accept_reported_cpu_fallback(lightweight_public_engines, monkeypatch):
    native = hf.importlib.import_module("atomiverse.scf")
    monkeypatch.setattr(native.CudaRHF, "last_scf_on_device", lambda self: False)
    row = RHF.rows[0]
    run = hf.atv_adapter(row, row.systems[0], participant(row, "atv.cuda"), threads=1)
    with pytest.raises(hf.ScientificFailure, match="CPU SCF fallback"):
        run()


def test_declared_unavailable_ghf_cuda_has_explicit_status_before_engine_import():
    row = GHF.rows[0]
    with pytest.raises(hf.CapabilityUnavailable) as failure:
        hf.atv_adapter(row, row.systems[0], participant(row, "atv.cuda"), threads=1)
    assert failure.value.benchmark_status == "not_implemented"
