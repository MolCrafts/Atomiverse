"""Public HF binding contracts; tiny frames exercise bridges, not new oracles."""
from __future__ import annotations

import numpy as np
import pytest

from _domains import skip_module_unless

molpy = pytest.importorskip("molpy")
# atomiverse.scf raises MissingDomainError (an ImportError, not a
# ModuleNotFoundError) without the hf module; pytest >= 9 importorskip no
# longer skips on it, so the domain is checked first.
skip_module_unless("ATV_BUILD_HF")
scf = pytest.importorskip("atomiverse.scf")


def h2_frame():
    frame = molpy.Frame()
    atoms = molpy.Block()
    atoms.insert("x", np.array([0.0, 0.0], dtype=np.float64))
    atoms.insert("y", np.array([0.0, 0.0], dtype=np.float64))
    atoms.insert("z", np.array([0.0, 0.74], dtype=np.float64))
    atoms.insert("mass", np.array([1.008, 1.008], dtype=np.float64))
    atoms.insert("atomic_number", np.array([1, 1], dtype=np.uint32))
    for name, value in (("fx", 0.125), ("fy", -0.25), ("fz", 0.5)):
        atoms.insert(name, np.full(2, value, dtype=np.float64))
    frame["atoms"] = atoms
    frame.box = molpy.Box.cube(20.0)
    frame.meta["energy"] = -314.15
    return frame


def test_cpu_backend_info_reports_compiler_and_observed_team():
    info = scf.backend_info()
    assert info["backend"] == "cpu"
    assert info["compiler"] and info["compiler_id"] and info["compiler_version"]
    assert info["compiler_executable"]
    assert 1 <= info["openmp_threads"] <= info["openmp_max_threads"]
    if not info["openmp"]:
        assert info["openmp_threads"] == info["openmp_max_threads"] == 1


@pytest.mark.parametrize("name,args", [("RHF", {}), ("UHF", {"n_alpha": 1, "n_beta": 1}),
                                      ("ROHF", {"n_alpha": 1, "n_beta": 1}),
                                      ("GHF", {"n_electrons": 2})])
def test_cpu_hf_public_methods_and_energy_only_frame_roundtrip(name, args):
    import atomiverse as av

    cls = getattr(scf, name)
    assert getattr(av, name) is cls
    assert name in scf.__all__ and name in av.__all__
    method = cls(basis="STO-3G", tol=1e-10, max_iter=200, **args)
    method.init_guess = "hcore"
    assert not hasattr(method, "integral_mode")
    assert method.max_memory_mb == 4000.0 and method.mpi is False
    method.max_memory_mb = 4000.0
    method.set_threads(1)
    method.damping = 0.0
    method.conv_tol_grad = 1e-6
    method.diis_space = 8
    method.diis_start_cycle = 1
    assert method.basis == "STO-3G" and method.tol == 1e-10
    assert method.max_iter == 200 and method.charge == 0
    assert method.damping == 0.0 and method.diis_space == 8
    assert method.diis_start_cycle == 1 and method.conv_tol_grad == 1e-6
    with pytest.raises(ValueError, match="max_memory_mb"):
        method.max_memory_mb = -1
    with pytest.raises(ValueError, match="init_guess"):
        method.init_guess = "guess"
    frame = h2_frame()
    method.run(frame)
    assert method.converged() and method.iterations() > 0
    assert frame.meta["energy"] == pytest.approx(method.energy() * 27.211386245988, abs=1e-8)
    for key, value in (("fx", 0.125), ("fy", -0.25), ("fz", 0.5)):
        np.testing.assert_array_equal(np.asarray(frame["atoms"][key]), np.full(2, value))
    if name == "ROHF":
        assert method.n_alpha == method.n_beta == 1
    if name == "GHF":
        assert method.n_electrons == 2 and method.spin_rotation == 0.2
        ab = np.asarray(method.spin_density_block("alpha", "beta"))
        ba = np.asarray(method.spin_density_block("beta", "alpha"))
        assert ab.shape == ba.shape == (2, 2)
        np.testing.assert_allclose(ab, ba.T, atol=1e-14)
        assert not np.shares_memory(ab, ba)
        with pytest.raises(ValueError, match="spin"):
            method.spin_density_block("up", "beta")


def test_ghf_density_requires_converged_result():
    method = scf.GHF("STO-3G", n_electrons=2, spin_rotation=0.0)
    assert method.spin_rotation == 0.0
    with pytest.raises(RuntimeError, match="converged"):
        method.spin_density_block()


def test_cpu_mp2_publishes_correlated_total_and_retains_rhf_energy():
    frame = h2_frame()
    rhf = scf.RHF("STO-3G", tol=1e-12)
    rhf.set_threads(1)
    result = scf.MP2(rhf).run(frame)
    assert isinstance(result, scf.MP2Result)
    assert result.frozen == 0 and not result.ri
    assert rhf.converged() and result.e_corr < 0
    assert result.e_corr == pytest.approx(result.e_corr_os + result.e_corr_ss, abs=1e-12)
    assert result.e_tot == pytest.approx(rhf.energy() + result.e_corr, abs=1e-12)
    assert frame.meta["energy"] == pytest.approx(result.e_tot * 27.211386245988, abs=1e-8)
    for key, value in (("fx", 0.125), ("fy", -0.25), ("fz", 0.5)):
        np.testing.assert_array_equal(np.asarray(frame["atoms"][key]), np.full(2, value))


def test_cpu_mp3_publishes_correlated_total_and_retains_rhf_energy():
    frame = h2_frame()
    rhf = scf.RHF("STO-3G", tol=1e-12)
    rhf.set_threads(1)
    result = scf.MP3(rhf).run(frame)
    assert isinstance(result, scf.MP3Result)
    assert result.frozen == 0 and rhf.converged()
    # H2/STO-3G pins from regressions/scf/mp2/mp3-h2-sto3g.cpp.
    assert result.e_corr_mp2 == pytest.approx(-0.013138073590, abs=1e-6)
    assert result.e_corr_mp3 == pytest.approx(-0.004836072637, abs=1e-6)
    assert result.e_corr == pytest.approx(result.e_corr_mp2 + result.e_corr_mp3, abs=1e-12)
    assert result.e_tot == pytest.approx(rhf.energy() + result.e_corr, abs=1e-12)
    assert frame.meta["energy"] == pytest.approx(result.e_tot * 27.211386245988, abs=1e-8)
    for key, value in (("fx", 0.125), ("fy", -0.25), ("fz", 0.5)):
        np.testing.assert_array_equal(np.asarray(frame["atoms"][key]), np.full(2, value))


@pytest.mark.parametrize(
    ("name", "components"),
    [
        ("MP2", ("e_corr_os", "e_corr_ss")),
        ("MP3", ("e_corr_mp2", "e_corr_mp3")),
    ],
)
def test_cpu_correlation_keeps_reference_alive_and_checks_frozen_occupation(name, components):
    cls = getattr(scf, name)
    rhf = scf.RHF("STO-3G")
    rhf.set_threads(1)
    method = cls(rhf, frozen=1)
    del rhf
    result = method.run(h2_frame())
    assert method.frozen == result.frozen == 1
    assert result.e_corr == 0.0
    assert all(getattr(result, field) == 0.0 for field in components)

    rhf = scf.RHF("STO-3G")
    rhf.set_threads(1)
    with pytest.raises(ValueError, match="frozen"):
        cls(rhf, frozen=-1)
    frame = h2_frame()
    with pytest.raises(ValueError, match="frozen"):
        cls(rhf, frozen=2).run(frame)
    assert frame.meta["energy"] == -314.15


_HAS_CUDA = "CudaRHF" in scf.__all__


@pytest.mark.skipif(not _HAS_CUDA, reason="extension built without CUDA HF")
def test_cuda_backend_metadata_completes_synchronization():
    info = scf.cuda_backend_info()
    assert info["backend"] == "cuda" and info["device_count"] > 0
    assert info["device_name"] and len(info["devices"]) == info["device_count"]
    assert info["total_memory_bytes"] >= info["free_memory_bytes"] > 0
    assert info["driver_version"] > 0 and info["runtime_version"] > 0
    assert info["synchronized"] and info["run_forces_synchronize"]
    scf.cuda_synchronize()


@pytest.mark.skipif(not _HAS_CUDA, reason="extension built without CUDA HF")
@pytest.mark.parametrize("name,args", [("CudaRHF", {}), ("CudaUHF", {"n_alpha": 1, "n_beta": 1})])
def test_cuda_hf_run_preserves_forces_and_explicit_force_array_is_owned(name, args):
    import atomiverse as av

    cls = getattr(scf, name)
    assert getattr(av, name) is cls and name in av.__all__
    method = cls("STO-3G", tol=1e-10, max_iter=100, **args)
    with pytest.raises(RuntimeError, match="standalone"):
        method.energy()
    frame = h2_frame()
    method.run(frame)
    assert method.converged() and method.last_scf_on_device()
    assert not method.last_force_on_device()
    for key, value in (("fx", 0.125), ("fy", -0.25), ("fz", 0.5)):
        np.testing.assert_array_equal(np.asarray(frame["atoms"][key]), np.full(2, value))
    forces = np.asarray(method.forces(frame))
    assert forces.shape == (2, 3) and np.isfinite(forces).all()
    assert method.last_force_on_device()
    for axis, key in enumerate(("fx", "fy", "fz")):
        np.testing.assert_allclose(np.asarray(frame["atoms"][key]), forces[:, axis], atol=1e-12)
    saved = forces.copy()
    method.run(frame)
    np.testing.assert_array_equal(forces, saved)
    assert not method.last_force_on_device()
    assert frame.meta["energy"] == pytest.approx(method.energy() * 27.211386245988, abs=1e-8)


@pytest.mark.skipif(not _HAS_CUDA, reason="extension built without CUDA HF")
def test_cuda_nonconvergence_does_not_publish_failed_frame():
    frame = h2_frame()
    method = scf.CudaRHF("STO-3G", max_iter=1)
    with pytest.raises(RuntimeError, match="converge"):
        method.run(frame)
    assert frame.meta["energy"] == -314.15
    np.testing.assert_array_equal(np.asarray(frame["atoms"]["fx"]), [0.125, 0.125])
    with pytest.raises(RuntimeError, match="standalone"):
        method.energy()
