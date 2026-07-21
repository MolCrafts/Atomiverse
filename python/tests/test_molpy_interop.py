"""molpy <-> atomiverse interop smoke (spec: python-api-namespaces-am1).

molpy and atomiverse share ``molrs.Frame`` as the exchange currency without
either project re-exporting it. A molpy-built ``Atomistic`` system exports via
``to_frame()`` to the
very frame type an atomiverse driver borrows and writes back into.

This is a seconds-fast smoke check, not a physics validation: a molpy
system runs a 10-step LJ MD and the written-back energy is finite (the
LJ parameters reuse the verified conftest values — no new reference
values). Skips cleanly when molpy is absent.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest
import molrs

molpy = pytest.importorskip("molpy")

from atomiverse.md import MD, InitVelocity  # noqa: E402
from atomiverse.neighbor import NeighborList  # noqa: E402
from atomiverse.potentials.pair import LJCut  # noqa: E402


def _molpy_argon_frame() -> "molrs.Frame":
    """2x2x2 simple-cubic Ar built through molpy's Atomistic API and
    exported to the shared molrs.Frame (atv needs an int32 ``type``
    column and a box, which Atomistic atom attrs do not carry)."""
    scale = 3.4  # ~sigma_Ar [A]
    system = molpy.Atomistic()
    for ix in range(2):
        for iy in range(2):
            for iz in range(2):
                system.def_atom(
                    symbol="Ar",
                    xyz=[ix * scale, iy * scale, iz * scale],
                    mass=39.948,
                )
    frame = system.to_frame()
    frame["atoms"].insert("type", np.zeros(8, dtype=np.int32))
    frame.simbox = molrs.Box.cube(2 * scale)
    return frame


class TestMolpyInterop:
    def test_molpy_does_not_reexport_frame(self) -> None:
        """The exchange type has one owner: callers import it from molrs."""
        assert not hasattr(molpy, "Frame")
        assert not hasattr(molpy, "Block")

    def test_molpy_core_frame_compatibility_module_is_deleted(self) -> None:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("molpy.core.frame")

    def test_molpy_system_exports_molrs_frame(self) -> None:
        """Atomistic.to_frame() yields a molrs.Frame with coordinates."""
        frame = _molpy_argon_frame()
        assert isinstance(frame, molrs.Frame)
        assert np.asarray(frame["atoms"]["x"]).shape == (8,)

    def test_molpy_frame_runs_atv_lj_md(self) -> None:
        """A molpy-built system runs a 10-step atv LJ MD; the energy
        written back into the same frame is finite."""
        frame = _molpy_argon_frame()
        cutoff = 8.5
        sim = MD(dt=0.001)
        sim.add(InitVelocity(10.0, 42))
        sim.on("force").add(NeighborList(cutoff), order=-10).add(
            LJCut(0.0104, 3.4, cutoff)
        )
        sim.run(frame, 10)
        assert "scalars" not in frame.keys()
        energy_meta = frame.meta["energy"]
        assert energy_meta.dtype == "f64"
        energy = float(energy_meta.value)
        assert np.isfinite(energy)
