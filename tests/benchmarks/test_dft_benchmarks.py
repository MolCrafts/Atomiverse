"""KS benchmark contracts; no electronic-structure solves in unit tests."""

import importlib

import pytest

from benchmarks.scf.bench_rks import SPEC as RKS
from benchmarks.scf.bench_uks import SPEC as UKS
from benchmarks.scf.dft import PYSCF_XC, atv_adapter, row, systems
from benchmarks.scf.hf import CapabilityUnavailable


def test_dft_declarations_require_live_oracle_and_full_force_vectors():
    rows = RKS.rows + UKS.rows
    assert len({item.id for item in rows}) == len(rows)
    assert {item.model["xc"] for item in rows} == {
        "LSDA",
        "PBE",
        "PBEsol",
        "RevPBE",
        "RPBE",
        "PBE0",
        "BLYP",
        "B3LYP",
        "CAMB3LYP",
    }
    for item in rows:
        reference = next(p for p in item.participants if p.id == item.reference)
        assert reference.required and reference.role == "oracle"
        assert reference.entrypoint == "benchmarks.scf.dft:pyscf_adapter"
        assert item.model["grid"] == "MatchPySCF"
        assert item.model["pyscf_grid_level"] == 3
        assert item.model["pyscf_xc"] == PYSCF_XC[item.model["xc"]]
        assert item.checks[0].atol == 1e-8
        if item.task == "forces":
            assert item.model["grid_response"]
            assert any(
                c.observable == "forces_hartree_per_bohr" and c.atol == 1e-6
                for c in item.checks
            )


def test_gaussian_cuda_ks_cannot_fall_back_before_importing_native_runtime(monkeypatch):
    item = RKS.rows[0]
    cuda = next(p for p in item.participants if p.backend == "cuda")
    assert not cuda.available and not cuda.required
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: pytest.fail(f"Unexpected import {name}"),
    )
    with pytest.raises(CapabilityUnavailable, match="not implemented"):
        atv_adapter(item, item.systems[0], cuda, 1)


def test_functionals_distinguish_vwn5_and_vwn_rpa_without_double_hybrid_aliases():
    assert PYSCF_XC["LSDA"] == "LDA_X,LDA_C_VWN"
    assert "LDA_C_VWN_RPA" in PYSCF_XC["B3LYP"]
    assert "0.25*HF" in PYSCF_XC["PBE0"]
    with pytest.raises(ValueError, match="unsupported"):
        row("RKS", "B2PLYP", "energy", ["water"])


def test_spherical_gga_hybrid_forces_cover_asymmetric_water():
    for xc in ["PBE", "PBE0"]:
        item = next(
            row
            for row in RKS.rows
            if row.model["xc"] == xc
            and row.model["basis"] == "def2-SVP"
            and row.task == "forces"
        )
        assert not item.model["cartesian"]
        assert {system.id for system in item.systems} == {"water", "water_asymmetric"}
    asymmetric, shared = systems("water_asymmetric", "h2_plus")
    assert asymmetric.atomic_numbers == [8, 1, 1]
    assert all(
        len({point[axis] for point in asymmetric.positions_angstrom}) == 3
        for axis in range(3)
    )
    assert shared.charge == 1 and shared.spin == 1
    # Removing smoke eligibility for one row must not mutate another load.
    asymmetric.tiers.clear()
    assert systems("water_asymmetric")[0].tiers


def test_standard_cam_uses_the_independent_full_functional():
    assert PYSCF_XC["CAMB3LYP"] == "HYB_GGA_XC_CAM_B3LYP"
    for method in [RKS, UKS]:
        cam = [item for item in method.rows if item.model["xc"] == "CAMB3LYP"]
        assert {item.model["basis"] for item in cam} == {"STO-3G", "def2-SVP"}
        assert {item.task for item in cam} == {"energy", "forces"}
        assert all(item.model["pyscf_xc"] == "HYB_GGA_XC_CAM_B3LYP" for item in cam)
