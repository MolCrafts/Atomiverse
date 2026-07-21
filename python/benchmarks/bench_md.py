"""
MD step throughput benchmark: Atomiverse CPU backend.

Measures wall-clock time per step as a function of system size (N atoms).
Compares configurations:
  - bare_vv:   VelocityVerlet only (no forces — integration overhead)
  - lj_md:     VelocityVerlet + NeighborList + LJ (realistic workload)
  - nvt_md:    lj_md + NoseHoover thermostat

Usage:
    python bench_md.py
    python bench_md.py --sizes 108 256 500 1372   # custom sizes
    python bench_md.py --steps 200                # warmup + measurement steps
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
import molrs

import atomiverse as av


# ── FCC lattice builder ───────────────────────────────────────────────────────

_FCC_BASIS = [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]


def make_fcc_frame(n_cells: int, a0: float = 5.26) -> "molrs.Frame":
    """Return an FCC argon molrs.Frame with n_cells^3 * 4 atoms."""
    atoms = [
        ((ix + bx) * a0, (iy + by) * a0, (iz + bz) * a0)
        for ix in range(n_cells)
        for iy in range(n_cells)
        for iz in range(n_cells)
        for bx, by, bz in _FCC_BASIS
    ]
    n = len(atoms)
    xs, ys, zs = (np.array(c, dtype=np.float64) for c in zip(*atoms))

    frame = molrs.Frame()
    block = molrs.Block()
    block.insert("x", xs)
    block.insert("y", ys)
    block.insert("z", zs)
    block.insert("mass", np.full(n, 39.948, dtype=np.float64))
    block.insert("type", np.zeros(n, dtype=np.int32))
    frame["atoms"] = block
    frame.simbox = molrs.Box.cube(float(n_cells * a0))
    return frame


# ── MD configurations ─────────────────────────────────────────────────────────


def build_bare_vv(dt: float = 0.001) -> av.MD:
    sim = av.MD(dt=dt)
    sim.add(av.InitVelocity(10.0, 42))
    return sim


def build_lj_md(dt: float = 0.001, cutoff: float = 8.5) -> av.MD:
    sim = av.MD(dt=dt)
    sim.add(av.InitVelocity(10.0, 42))
    sim.on("force").add(av.NeighborList(cutoff), order=-10).add(
        av.PairLJCut(0.0104, 3.4, cutoff)
    )
    return sim


def build_nvt_md(dt: float = 0.001, cutoff: float = 8.5, T: float = 100.0) -> av.MD:
    sim = av.MD(dt=dt)
    sim.add(av.InitVelocity(T, 42))
    sim.add(av.NoseHoover(T, 100.0, dt))
    sim.on("force").add(av.NeighborList(cutoff), order=-10).add(
        av.PairLJCut(0.0104, 3.4, cutoff)
    )
    return sim


# ── Timing helper ─────────────────────────────────────────────────────────────


@dataclass
class BenchResult:
    label: str
    n_atoms: int
    n_steps: int
    warmup_steps: int
    time_per_step_ms: float
    throughput_ns_per_step: float  # atom*steps/s in Mstep·atoms/s

    def __str__(self) -> str:
        return (
            f"{self.label:<20} N={self.n_atoms:>6}  "
            f"{self.time_per_step_ms:8.3f} ms/step  "
            f"{self.throughput_ns_per_step:8.3f} Matom·step/s"
        )


def run_bench(
    label: str,
    frame: "molrs.Frame",
    sim: av.MD,
    n_steps: int = 100,
    warmup: int = 20,
) -> BenchResult:
    n = int(frame["atoms"].nrows)
    # Warmup
    sim.run(frame, warmup)
    # Measure
    t0 = time.perf_counter()
    sim.run(frame, n_steps)
    elapsed = time.perf_counter() - t0
    ms_per_step = elapsed / n_steps * 1e3
    throughput = n * n_steps / elapsed / 1e6
    return BenchResult(label, n, n_steps, warmup, ms_per_step, throughput)


# ── Main ──────────────────────────────────────────────────────────────────────


def n_cells_for(target: int) -> int:
    """Return smallest n_cells such that n_cells^3 * 4 >= target."""
    n = 1
    while n**3 * 4 < target:
        n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="Atomiverse MD throughput benchmark")
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[108, 256, 500, 1372, 2048],
        help="Target atom counts (nearest FCC will be used)",
    )
    parser.add_argument("--steps", type=int, default=100, help="Measurement steps")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup steps")
    args = parser.parse_args()

    print(f"{'Label':<20} {'N':>6}  {'ms/step':>10}  {'Matom·step/s':>14}")
    print("-" * 60)

    configs = [
        ("bare_vv", build_bare_vv),
        ("lj_md", build_lj_md),
        ("nvt_md", build_nvt_md),
    ]

    for target in args.sizes:
        nc = n_cells_for(target)
        for label, builder in configs:
            # A fresh frame per config — each builder runs an independent
            # trajectory, and run() materializes a transient frame each call.
            frame = make_fcc_frame(nc)
            sim = builder()
            result = run_bench(
                label, frame, sim, n_steps=args.steps, warmup=args.warmup
            )
            print(result)
        print()


if __name__ == "__main__":
    main()
