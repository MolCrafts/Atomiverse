"""molpy is the only user-facing Frame type.

A molpy-built Atomistic system exports via ``to_frame()`` to the
``molpy.Frame`` an atomiverse driver borrows and writes back into.
"""

from __future__ import annotations

import numpy as np
import molpy

from atomiverse.md import MD, InitVelocity
from atomiverse.neighbor import NeighborList
from atomiverse.potentials.pair import LJCut


def _molpy_argon_frame() -> "molpy.Frame":
    scale = 3.4
    system = molpy.Atomistic()
    for ix in range(2):
        for iy in range(2):
            for iz in range(2):
                system.def_atom(
                    element="Ar",
                    x=ix * scale,
                    y=iy * scale,
                    z=iz * scale,
                    mass=39.948,
                )
    frame = system.to_frame()
    frame["atoms"].insert("atomic_number", np.full(8, 18, dtype=np.uint32))
    frame.box = molpy.Box.cube(2 * scale)
    return frame


class TestMolpyInterop:
    def test_atomistic_exports_molpy_frame(self) -> None:
        frame = _molpy_argon_frame()
        assert isinstance(frame, molpy.Frame)
        assert np.asarray(frame["atoms"]["x"]).shape == (8,)

    def test_molpy_frame_runs_atv_lj_md(self) -> None:
        from atomiverse.md.integrator import VelocityVerlet

        frame = _molpy_argon_frame()
        cutoff = 8.5
        sim = MD()
        sim.set_integrator(VelocityVerlet(0.001))
        sim.set_observer(InitVelocity(10.0, 42))
        sim.set_neighbor(NeighborList(cutoff))
        sim.set_evaluator(LJCut(0.0104, 3.4, cutoff))
        sim.run(frame, 10)
        energy = frame.meta["energy"]
        assert isinstance(energy, float)
        assert np.isfinite(energy)
