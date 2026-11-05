"""molpy is the only user-facing Frame type.

A molpy-built Atomistic system exports via ``to_frame()`` to the
``molpy.Frame`` an atomiverse driver borrows and writes back into.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np

from _domains import skip_module_unless

skip_module_unless("ATV_BUILD_MD")


def _molpy_argon_frame() -> "molpy.Frame":
    import molpy

    scale = 3.4
    system = molpy.Atomistic()
    for ix in range(2):
        for iy in range(2):
            for iz in range(2):
                system.add_atom(
                    atomic_number=18,
                    x=ix * scale,
                    y=iy * scale,
                    z=iz * scale,
                    mass=39.948,
                )
    frame = system.to_frame()
    frame.box = molpy.Box.cube(2 * scale)
    return frame


class TestMolpyInterop:
    def test_atomistic_exports_molpy_frame(self) -> None:
        import molpy

        frame = _molpy_argon_frame()
        assert isinstance(frame, molpy.Frame)
        assert np.asarray(frame["atoms"]["x"]).shape == (8,)

    def test_molpy_frame_runs_atv_lj_md(self) -> None:
        from atomiverse.md import MD, InitVelocity
        from atomiverse.md.integrator import VelocityVerlet
        from atomiverse.neighbor import NeighborList
        from atomiverse.potentials.pair import LJCut

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


def test_molpy_abi_mismatch_rejected_before_capsule_access() -> None:
    # A fresh interpreter exercises the first, cached ABI handshake even if
    # earlier tests already borrowed a frame. No simulation is run.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""\
            import molpy._core
            from atomiverse._binding import _cpp

            # The bridge reads the token from molpy._core (the native module
            # that owns the Frame capsule), so that is the one to fake.
            token = molpy._core._ffi_abi_token()
            molpy._core._ffi_abi_token = lambda: ("999.0", *token[1:])

            class UnreadableFrame:
                def _ffi_frameref_capsule(self):
                    raise AssertionError("capsule accessed before ABI check")

            try:
                _cpp.MD().run(UnreadableFrame(), 0)
            except RuntimeError as exc:
                assert "molpy ABI-line mismatch" in str(exc), str(exc)
                assert "999.0" in str(exc), str(exc)
                assert "linked" in str(exc), str(exc)
            else:
                raise AssertionError("mismatched molpy ABI was accepted")
            """),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
