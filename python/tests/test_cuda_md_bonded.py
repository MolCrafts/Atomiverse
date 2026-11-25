"""Molecular force field on ``MD(device="cuda")``: harmonic bonds and angles,
multi-harmonic and CHARMM dihedrals, fused LJ + Ewald real space with Amber
special bonds, and PPPM — one step from the same state on CPU and CUDA.

The bonded tables follow the CPU convention: they are indexed by the raw
``type_id`` of the bonds / angles / dihedrals blocks, so a 1-based topology
(LAMMPS data, molpy) leaves entry 0 unused. Entry 0 carries deliberately
wild parameters here: a backend that shifted the index would blow up.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import molpy

from atomiverse._binding import _cpp, extension_available
from _domains import skip_module_unless

skip_module_unless("ATV_BUILD_MD")

import atomiverse as av
from atomiverse.md import MD
from atomiverse.md.integrator import VelocityVerlet
from atomiverse.neighbor import NeighborList

_HAS_CUDA_MD: bool = extension_available() and hasattr(_cpp, "_CudaMD")
_HAS_KSPACE: bool = extension_available() and hasattr(_cpp, "KSpacePPPM")
needs_cuda = pytest.mark.skipif(not _HAS_CUDA_MD, reason="needs ATV_BUILD_CUDA")
needs_kspace = pytest.mark.skipif(not _HAS_KSPACE, reason="needs ATV_BUILD_KSPACE")

_BOX = 26.0  # [Å]
_CUTOFF = 8.0  # [Å]
_SKIN = 1.0  # [Å]
_CHARGES = (0.25, -0.4, 0.3, -0.4, 0.25)  # neutral 5-site chain [e]


def _chains() -> "molpy.Frame":
    """12 five-site chains on a 3×2×2 grid, perturbed zigzag geometry (so
    every dihedral has a nonzero torque), small random velocities."""
    rng = np.random.default_rng(2026)
    xyz, vel, charge, bonds, angles, dihedrals = [], [], [], [], [], []
    base = np.array(
        [[0.0, 0.0, 0.0], [1.25, 0.88, 0.0], [2.5, 0.0, 0.0],
         [3.75, 0.88, 0.0], [5.0, 0.0, 0.0]]
    )
    n = 0
    for gx in range(3):
        for gy in range(2):
            for gz in range(2):
                origin = np.array([1.5 + 8.0 * gx, 2.0 + 12.0 * gy, 3.0 + 12.0 * gz])
                pts = base + origin + rng.normal(scale=0.25, size=base.shape)
                xyz.extend(pts)
                vel.extend(rng.normal(scale=2.0, size=base.shape))  # [Å/ps]
                charge.extend(_CHARGES)
                bonds += [(n + k, n + k + 1) for k in range(4)]
                angles += [(n + k, n + k + 1, n + k + 2) for k in range(3)]
                dihedrals += [(n + k, n + k + 1, n + k + 2, n + k + 3) for k in range(2)]
                n += 5
    xyz = np.asarray(xyz)
    vel = np.asarray(vel)
    atoms = molpy.Block()
    for c, key in enumerate("xyz"):
        atoms.insert(key, np.ascontiguousarray(xyz[:, c]))
        atoms.insert("v" + key, np.ascontiguousarray(vel[:, c]))
    atoms.insert("mass", np.full(n, 14.027))
    atoms.insert("charge", np.asarray(charge, dtype=np.float64))
    atoms.insert("type_id", np.full(n, 1, dtype=np.uint32))
    atoms.insert("atomic_number", np.full(n, 6, dtype=np.uint32))
    frame = molpy.Frame()
    frame.meta["units_style"] = "metal"
    frame["atoms"] = atoms
    for name, rows in (("bonds", bonds), ("angles", angles), ("dihedrals", dihedrals)):
        rows = np.asarray(rows, dtype=np.uint32)
        block = molpy.Block()
        for c, end in enumerate(("atomi", "atomj", "atomk", "atoml")[: rows.shape[1]]):
            block.insert(end, np.ascontiguousarray(rows[:, c]))
        block.insert("type_id", np.full(len(rows), 1, dtype=np.uint32))  # 1-based
        frame[name] = block
    frame.box = molpy.Box.cube(_BOX)
    return frame


def _molecular_md(device: str, *, charmm: bool) -> MD:
    """metal units: eV, Å, ps. Entry 0 of every bonded table is a decoy."""
    alpha = 3.5 / _CUTOFF
    sim = MD(device=device, units="metal")
    sim.set_neighbor(NeighborList(_CUTOFF, _SKIN))
    sim.set_evaluator(
        av.PairLJCutCoulLong([av.LJTypeCoeff(1, 1, 0.0045, 3.5)], alpha, _CUTOFF)
    )
    sim.set_evaluator(av.KSpacePPPM(alpha, 1.0, 5))
    sim.set_evaluator(av.BondHarmonic([1e3, 13.0], [9.0, 1.53]))
    sim.set_evaluator(av.AngleHarmonic([1e3, 2.6], [0.1, math.radians(112.0)]))
    sim.set_evaluator(
        av.DihedralMultiHarmonic(
            [[5.0, 5.0, 5.0, 5.0, 5.0], [0.06, -0.12, 0.03, 0.15, -0.02]]
        )
    )
    if charmm:
        sim.set_evaluator(av.DihedralCharmm([3.0, 0.02], [2, 3], [1.0, 0.0]))
    sim.set_integrator(VelocityVerlet(0.001))
    return sim


def _by_tag(frame: "molpy.Frame") -> np.ndarray:
    atoms = frame["atoms"]
    return np.argsort(np.asarray(atoms["tag"])) if "tag" in atoms.keys() else slice(None)


def _cols(frame: "molpy.Frame", keys) -> np.ndarray:
    a = frame["atoms"]
    return np.stack([np.asarray(a[k], dtype=np.float64) for k in keys], axis=1)[
        _by_tag(frame)
    ]


@needs_cuda
@needs_kspace
class TestCudaMolecularParity:
    @pytest.mark.parametrize("charmm", [False, True])
    def test_one_step_matches_cpu(self, charmm: bool) -> None:
        frames = {}
        for device in ("cpu", "cuda"):
            frame = _chains()
            _molecular_md(device, charmm=charmm).run(frame, 1)
            frames[device] = frame
        start = _chains()
        cpu, gpu = frames["cpu"], frames["cuda"]
        # The bonded forces did act (the step is not a free flight).
        dv = _cols(cpu, ("vx", "vy", "vz")) - _cols(start, ("vx", "vy", "vz"))
        assert np.max(np.abs(dv)) > 1e-3
        np.testing.assert_allclose(
            _cols(gpu, ("vx", "vy", "vz")), _cols(cpu, ("vx", "vy", "vz")),
            rtol=0, atol=1e-8,
        )
        np.testing.assert_allclose(
            _cols(gpu, ("x", "y", "z")), _cols(cpu, ("x", "y", "z")),
            rtol=0, atol=1e-10,
        )
        for key in ("energy", "energy_bond", "energy_angle", "energy_dihedral"):
            np.testing.assert_allclose(
                float(gpu.meta[key]), float(cpu.meta[key]), rtol=1e-9, atol=1e-9,
                err_msg=key,
            )

    def test_energy_terms_use_the_raw_type_id(self) -> None:
        """Every bond/angle/dihedral is type 1: the bond energy equals
        Σ K₁(r − r₀₁)² with entry 1, on both devices (entry 0 would give
        ~10⁴ eV)."""
        frame = _chains()
        xyz = _cols(frame, ("x", "y", "z"))
        bonds = frame["bonds"]
        i = np.asarray(bonds["atomi"], dtype=int)
        j = np.asarray(bonds["atomj"], dtype=int)
        r = np.linalg.norm(xyz[j] - xyz[i], axis=1)
        expect = float(np.sum(13.0 * (r - 1.53) ** 2))
        for device in ("cpu", "cuda"):
            f = _chains()
            _molecular_md(device, charmm=False).run(f, 0)
            assert float(f.meta["energy_bond"]) == pytest.approx(expect, rel=1e-10)

    def test_multi_harmonic_beyond_cos4_is_refused_on_cuda(self) -> None:
        sim = MD(device="cuda", units="metal")
        with pytest.raises(ValueError, match="cos\\^0..cos\\^4"):
            sim.set_evaluator(av.DihedralMultiHarmonic([[0.0] * 6]))
