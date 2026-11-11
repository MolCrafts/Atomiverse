"""Long native calls release the GIL, so other Python threads keep running.

molab runs independent workflow tasks on worker threads. A native call that
holds the GIL for its whole run serialises them: a minimisation beside an AM1
single point would make both wait. Each case runs the call on a worker thread
while the main thread spins a Python loop; the loop must keep at least a
quarter of its free rate (the molpy ``test_gil_release`` gate).
"""

from __future__ import annotations

import threading
import time

import molpy
import numpy as np
import pytest

from _domains import needs

#: Fraction of the free-running loop rate another thread must keep.
_MIN_RATE_FRACTION = 0.25
#: The measured call must outlast this [s], or the rate is meaningless.
_MIN_ELAPSED = 0.05


def _loop_rate(seconds: float) -> float:
    t0 = time.perf_counter()
    n = 0
    while time.perf_counter() - t0 < seconds:
        n += 1
    return n / seconds


def _assert_releases_gil(work) -> None:
    free = _loop_rate(0.3)
    done = threading.Event()
    failure: list[BaseException] = []

    def run() -> None:
        try:
            work()
        except BaseException as error:  # surfaced on the main thread
            failure.append(error)
        finally:
            done.set()

    worker = threading.Thread(target=run)
    t0 = time.perf_counter()
    worker.start()
    ticks = 0
    while not done.is_set():
        ticks += 1
    elapsed = time.perf_counter() - t0
    worker.join()
    if failure:
        raise failure[0]
    assert elapsed > _MIN_ELAPSED, f"the call finished too fast to measure ({elapsed:.3f} s)"
    assert ticks / elapsed > _MIN_RATE_FRACTION * free, (ticks / elapsed, free)


def _polyacene_frame(n_rings: int) -> "molpy.Frame":
    """Planar linear acene C(4n+2)H(2n+4): 1.40 Å C–C, 1.08 Å C–H."""
    cc, ch = 1.40, 1.08
    a = cc * np.sqrt(3.0)
    carbons: list[np.ndarray] = []
    for ring in range(n_rings):
        centre = np.array([a * ring, 0.0])
        for k in range(6):
            angle = np.pi / 6 + k * np.pi / 3
            p = centre + cc * np.array([np.cos(angle), np.sin(angle)])
            if not any(np.linalg.norm(p - q) < 0.1 for q in carbons):
                carbons.append(p)
    c = np.array(carbons)
    d = np.linalg.norm(c[:, None] - c[None], axis=-1)
    hydrogens = []
    for i in range(len(c)):
        nb = np.where((d[i] > 0.1) & (d[i] < cc + 0.1))[0]
        if len(nb) == 2:
            v = 2 * c[i] - c[nb[0]] - c[nb[1]]
            hydrogens.append(c[i] + ch * v / np.linalg.norm(v))
    xy = np.vstack([c, np.array(hydrogens)])
    frame = molpy.Frame()
    atoms = molpy.Block()
    atoms.insert("x", xy[:, 0].copy())
    atoms.insert("y", xy[:, 1].copy())
    atoms.insert("z", np.zeros(len(xy)))
    atoms.insert(
        "atomic_number",
        np.array([6] * len(c) + [1] * len(hydrogens), dtype=np.uint32),
    )
    frame["atoms"] = atoms
    return frame


def _argon_frame(side: int, jitter: float = 0.25, spacing: float = 3.6) -> "molpy.Frame":
    """Perturbed simple-cubic argon, side³ atoms."""
    rng = np.random.default_rng(7)
    grid = np.arange(side) * spacing
    xyz = np.array(np.meshgrid(grid, grid, grid, indexing="ij")).reshape(3, -1).T
    xyz = xyz + rng.normal(0.0, jitter, xyz.shape)
    frame = molpy.Frame()
    atoms = molpy.Block()
    for axis, column in zip("xyz", xyz.T):
        atoms.insert(axis, column.copy())
    atoms.insert("mass", np.full(len(xyz), 39.948))
    atoms.insert("atomic_number", np.full(len(xyz), 18, dtype=np.uint32))
    atoms.insert("charge", np.zeros(len(xyz)))
    frame["atoms"] = atoms
    frame.box = molpy.Box.cube(side * spacing)
    return frame


@needs("ATV_BUILD_SEMIEMPIRICAL")
def test_am1_single_point_lets_other_python_threads_run() -> None:
    import atomiverse as av

    _assert_releases_gil(
        lambda: [av.single_point(_polyacene_frame(9), av.AM1(charge=0)) for _ in range(3)]
    )


@needs("ATV_BUILD_SEMIEMPIRICAL")
def test_am1bcc_assign_lets_other_python_threads_run() -> None:
    import atomiverse as av

    frame = _polyacene_frame(9)
    atoms = frame["atoms"]
    mol = molpy.Atomistic()
    added = [
        mol.add_atom(atomic_number=int(z), x=float(x), y=float(y), z=float(w))
        for z, x, y, w in zip(atoms["atomic_number"], atoms["x"], atoms["y"], atoms["z"])
    ]
    xyz = np.stack([np.asarray(atoms[k], dtype=float) for k in "xyz"], axis=1)
    d = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    for i in range(len(added)):
        for j in range(i):
            if d[i, j] < 1.45:
                order = 2 if (atoms["atomic_number"][i] == 6 and atoms["atomic_number"][j] == 6 and (i + j) % 2) else 1
                mol.add_bond(added[i], added[j], bond_type=order, bond_number=order)
    model = av.AM1BCCCharges(method=av.AM1(charge=0), correction="bcc")

    def assign() -> None:
        for _ in range(3):
            try:
                model.assign(mol)
            except RuntimeError as error:  # unsupported BCC typing is not the point
                if "SCF" in str(error):
                    raise
                pytest.skip(f"BCC typing of the test acene: {error}")

    _assert_releases_gil(assign)


@needs("ATV_BUILD_MINIMIZER")
def test_minimize_lets_other_python_threads_run() -> None:
    from atomiverse.minimizer import LBFGS, ForceConverged, Minimize
    from atomiverse.neighbor import NeighborList
    from atomiverse.potentials.pair import LJCut

    def minimize() -> None:
        # test_minimizer.py's recipe on a jittered 6x6x6 cell; repeated so
        # the native work outlasts the measurement floor.
        for _ in range(4):
            opt = Minimize()
            opt.add_criterion(ForceConverged(ftol=0.05, rms_ftol=0.03, max_dtol=0.002,
                                             rms_dtol=0.001, etol=1e-5))
            opt.set_optimizer(LBFGS(memory=5, alpha0=0.01, max_ls=10))
            opt.set_neighbor(NeighborList(8.5, 0.3))
            opt.set_evaluator(LJCut(0.0104, 3.4, 8.5))
            opt.run(_argon_frame(6, jitter=0.1, spacing=3.4), 5000)

    _assert_releases_gil(minimize)


@needs("ATV_BUILD_MD")
def test_md_run_lets_other_python_threads_run() -> None:
    from atomiverse.md import MD
    from atomiverse.md.integrator import VelocityVerlet
    from atomiverse.neighbor import NeighborList
    from atomiverse.potentials.pair import LJCut

    def run() -> None:
        md = MD(dt=1.0)
        md.set_integrator(VelocityVerlet(dt=1.0))
        md.set_neighbor(NeighborList(8.5, 0.3))
        md.set_evaluator(LJCut(0.0104, 3.4, 8.5))
        md.run(_argon_frame(10, jitter=0.05), 2000)

    _assert_releases_gil(run)
