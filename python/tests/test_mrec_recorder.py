"""``MrecRecorder`` — molrec (``*.mrec``) trajectories from ``MD.set_record``.

The C++ unit tests own the recorder mechanics (cadence, tag order, the CUDA
shared capture buffer and writer: tests/cpu/record/test_mrec_recorder.cpp,
tests/cuda/record/test_mrec_recorder.cu, tests/cuda/asynchronous/). These tests pin the Python
surface: the record type, its argument checks, the columns and the
``step`` / ``time`` axes a ``molpy.io.mrec.MrecReader`` reads back, that a
CPU run and a CUDA run of the same short trajectory read back as the same
frames, and that CUDA recording is lossless under a tiny driver capture
buffer (``MD(capture_buffer_bytes=...)``) and stops the run on a write
failure.
"""

from __future__ import annotations

import faulthandler
import gc
import os
import stat
from pathlib import Path

import numpy as np
import pytest
import molpy

from atomiverse._binding import _cpp, extension_available

# The recorders are bound with the MD driver (ATV_BUILD_MD).
_HAS_MREC: bool = extension_available() and hasattr(_cpp, "MD")
_HAS_CUDA_MD: bool = extension_available() and hasattr(_cpp, "_CudaMD")

pytestmark = pytest.mark.skipif(not _HAS_MREC, reason="needs ATV_BUILD_MD")

if _HAS_MREC:
    import atomiverse as av
    from atomiverse.md import MD
    from atomiverse.md.integrator import VelocityVerlet
    from atomiverse.neighbor import NeighborList
    from atomiverse.potentials.pair import LJCut
    from atomiverse.recorder import MrecRecorder
    from molpy.io.mrec import MrecFile, MrecReader

_DT = 1.0  # [fs]
_CUTOFF = 8.5  # [Å]
_SKIN = 0.5  # [Å]

#: Every column both backends write, by dtype class.
_FLOAT_COLUMNS = ("x", "y", "z", "vx", "vy", "vz", "fx", "fy", "fz", "mass", "charge")
_INT_COLUMNS = ("ix", "iy", "iz", "tag", "atomic_number")


def _argon_frame() -> "molpy.Frame":
    """64 Ar on a slightly perturbed simple-cubic lattice in a 20 Å cell,
    with a deterministic velocity field (no RNG on either device)."""
    side, a = 4, 20.0 / 4
    grid = np.arange(side) * a + 0.5 * a
    x, y, z = (c.ravel() for c in np.meshgrid(grid, grid, grid, indexing="ij"))
    n = x.size
    i = np.arange(n, dtype=np.float64)
    atoms = molpy.Block()
    atoms.insert("x", (x + 0.05 * np.sin(i)).astype(np.float64))
    atoms.insert("y", (y + 0.05 * np.cos(i)).astype(np.float64))
    atoms.insert("z", z.astype(np.float64))
    atoms.insert("vx", 0.002 * np.sin(3.0 * i))
    atoms.insert("vy", 0.002 * np.cos(5.0 * i))
    atoms.insert("vz", np.full(n, 0.001))
    atoms.insert("mass", np.full(n, 39.948, dtype=np.float64))
    atoms.insert("atomic_number", np.full(n, 18, dtype=np.uint32))
    frame = molpy.Frame()
    frame["atoms"] = atoms
    frame.box = molpy.Box.cube(20.0)
    return frame


def _lj_md(
    device: str, recorder: object, capture_buffer_bytes: int | None = None
) -> "MD":
    sim = MD(device=device, capture_buffer_bytes=capture_buffer_bytes)
    sim.set_integrator(VelocityVerlet(_DT))
    sim.set_neighbor(NeighborList(_CUTOFF, _SKIN))
    sim.set_evaluator(LJCut(0.0104, 3.4, _CUTOFF))
    sim.set_record(recorder)
    return sim


def _read_all(path: Path) -> tuple[list[int], list[float] | None, list]:
    with MrecReader(Path(path)) as reader:
        frames = [reader[i] for i in range(len(reader))]
        return list(reader.step), reader.time, frames


class TestMrecRecorderSurface:
    def test_identity_reexport(self) -> None:
        assert av.MrecRecorder is MrecRecorder

    def test_arguments_are_recorded(self) -> None:
        rec = MrecRecorder("traj.mrec", 10, flush_every=5, durable=False)
        assert (rec.path, rec.every, rec.flush_every, rec.durable) == (
            "traj.mrec",
            10,
            5,
            False,
        )
        assert MrecRecorder("traj.mrec").every == 1
        # The frame buffer is the driver's, not the recorder's.
        assert not hasattr(rec, "buffer_bytes")
        with pytest.raises(TypeError):
            MrecRecorder("traj.mrec", 1, buffer_bytes=4096)  # type: ignore[call-arg]

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_md_rejects_a_zero_capture_buffer(self, device: str) -> None:
        if device == "cuda" and not _HAS_CUDA_MD:
            pytest.skip("needs ATV_BUILD_CUDA")
        with pytest.raises(ValueError, match="capture_buffer_bytes"):
            MD(device=device, capture_buffer_bytes=0)

    @pytest.mark.parametrize("path,every", [("traj.zarr", 1), ("", 1), ("t.mrec", 0)])
    def test_set_record_rejects_bad_arguments(self, path: str, every: int) -> None:
        sim = MD()
        with pytest.raises(ValueError):
            sim.set_record(MrecRecorder(path, every))


def _identity_frame(which: str) -> "molpy.Frame":
    """The argon fixture with only the identity columns ``which`` names:
    ``"type"`` (coarse-grained: ``type_id`` only), ``"species"`` (QM:
    ``atomic_number`` only) or ``"both"``. Single-type LJ reads neither."""
    frame = _argon_frame()
    atoms = frame["atoms"]
    n = len(atoms["x"])
    rebuilt = molpy.Block()
    for col in ("x", "y", "z", "vx", "vy", "vz", "mass"):
        rebuilt.insert(col, np.asarray(atoms[col]))
    if which in ("type", "both"):
        rebuilt.insert("type_id", (np.arange(n) % 3).astype(np.uint32))
    if which in ("species", "both"):
        rebuilt.insert("atomic_number", (6 + np.arange(n) % 2).astype(np.uint32))
    frame["atoms"] = rebuilt
    return frame


def _check_identity_store(path: Path, which: str) -> None:
    """Every frame carries exactly the identity columns of the input."""
    _, _, frames = _read_all(path)
    assert frames
    n = 64
    for fr in frames:
        atoms = fr["atoms"]
        assert ("type_id" in atoms) == (which in ("type", "both"))
        assert ("atomic_number" in atoms) == (which in ("species", "both"))
        if "type_id" in atoms:
            np.testing.assert_array_equal(atoms["type_id"], np.arange(n) % 3)
        if "atomic_number" in atoms:
            np.testing.assert_array_equal(atoms["atomic_number"], 6 + np.arange(n) % 2)


class TestMrecRecorderCpu:
    def test_frames_steps_time_and_columns(self, tmp_path: Path) -> None:
        """6 steps every 2 → steps 0, 2, 4, 6; time = step · dt; every
        column."""
        path = tmp_path / "cpu.mrec"
        frame = _argon_frame()
        _lj_md("cpu", MrecRecorder(str(path), 2)).run(frame, 6)

        steps, time, frames = _read_all(path)
        assert steps == [0, 2, 4, 6]
        assert time == pytest.approx([0.0, 2.0 * _DT, 4.0 * _DT, 6.0 * _DT])
        atoms = frames[0]["atoms"]
        for col in _FLOAT_COLUMNS + _INT_COLUMNS:
            assert col in atoms, col
            assert len(atoms[col]) == 64
        assert atoms["ix"].dtype == np.int32
        np.testing.assert_array_equal(atoms["tag"], np.sort(atoms["tag"]))
        np.testing.assert_allclose(frames[0].box.matrix, 20.0 * np.eye(3))
        # Step 6 is the state after the sixth (last) step: the state run()
        # wrote back into the frame (identity tags, so the same row order).
        np.testing.assert_allclose(frames[-1]["atoms"]["x"], frame["atoms"]["x"],
                                   rtol=0, atol=1e-12)

    @pytest.mark.parametrize("which", ["type", "species", "both"])
    def test_identity_columns_follow_the_frame(self, tmp_path: Path, which: str) -> None:
        path = tmp_path / f"cpu-{which}.mrec"
        _lj_md("cpu", MrecRecorder(str(path), 1)).run(_identity_frame(which), 2)
        _check_identity_store(path, which)

    def test_status_section_records_a_clean_run(self, tmp_path: Path) -> None:
        path = tmp_path / "cpu.mrec"
        _lj_md("cpu", MrecRecorder(str(path), 2)).run(_argon_frame(), 5)
        status = MrecFile(Path(path)).status()
        assert status is not None
        assert status["state"] == "succeeded"
        assert status["frames_written"] == 3
        assert "buffer" not in status  # the CPU recorder writes synchronously
        assert "error" not in status


@pytest.mark.skipif(not _HAS_CUDA_MD, reason="needs ATV_BUILD_CUDA")
class TestMrecRecorderCuda:
    def test_cuda_frames_match_cpu(self, tmp_path: Path) -> None:
        """The same two-step trajectory read back from a CPU and a CUDA
        store gives the same frames, axes and columns."""
        cpu_path, gpu_path = tmp_path / "cpu.mrec", tmp_path / "gpu.mrec"
        _lj_md("cpu", MrecRecorder(str(cpu_path), 1)).run(_argon_frame(), 2)
        _lj_md("cuda", MrecRecorder(str(gpu_path), 1)).run(_argon_frame(), 2)

        cpu_steps, cpu_time, cpu_frames = _read_all(cpu_path)
        gpu_steps, gpu_time, gpu_frames = _read_all(gpu_path)
        assert cpu_steps == gpu_steps == [0, 1, 2]
        assert cpu_time == pytest.approx(gpu_time)
        # FP64 device tier: only integrator/reduction rounding differs.
        atol = 1e-9
        for a, b in zip(cpu_frames, gpu_frames):
            for col in _INT_COLUMNS:
                np.testing.assert_array_equal(a["atoms"][col], b["atoms"][col], col)
            for col in _FLOAT_COLUMNS:
                np.testing.assert_allclose(
                    a["atoms"][col], b["atoms"][col], rtol=0, atol=atol, err_msg=col
                )
            np.testing.assert_allclose(a.box.matrix, b.box.matrix, rtol=0, atol=0)
            # The per-term breakdown matches too: LJ is all pair energy.
            for key in ("energy", "energy_pair", "kinetic", "stress"):
                np.testing.assert_allclose(
                    np.asarray(a.meta[key], dtype=np.float64),
                    np.asarray(b.meta[key], dtype=np.float64),
                    rtol=0,
                    atol=atol,
                    err_msg=key,
                )

    def test_cuda_continuation_extends_one_trajectory(self, tmp_path: Path) -> None:
        """On CUDA two run() calls on one frame continue the step axis, the
        boundary step is written once, and each run() returns with its
        frames in the store."""
        path = tmp_path / "gpu.mrec"
        frame = _argon_frame()
        sim = _lj_md("cuda", MrecRecorder(str(path), 2))
        sim.run(frame, 4)  # steps 0, 2, 4
        assert _read_all(path)[0] == [0, 2, 4]
        sim.run(frame, 4)  # (4 already written), 6, 8
        assert _read_all(path)[0] == [0, 2, 4, 6, 8]

    @pytest.mark.parametrize("which", ["type", "species", "both"])
    def test_cuda_identity_columns_follow_the_frame(
        self, tmp_path: Path, which: str
    ) -> None:
        """Never a type written as Z, never a zero-filled species."""
        path = tmp_path / f"gpu-{which}.mrec"
        _lj_md("cuda", MrecRecorder(str(path), 1)).run(_identity_frame(which), 2)
        _check_identity_store(path, which)

    def test_cuda_status_section_records_a_clean_run(self, tmp_path: Path) -> None:
        path = tmp_path / "gpu.mrec"
        sim = _lj_md("cuda", MrecRecorder(str(path), 2))
        sim.run(_argon_frame(), 6)  # steps 0, 2, 4, 6
        stats = sim.capture_stats()
        assert stats["captures"] == 4
        assert stats["budget_bytes"] > 0
        del sim  # teardown closes the store and writes its status
        gc.collect()
        status = MrecFile(Path(path)).status()
        assert status is not None
        assert status["state"] == "succeeded"
        assert status["frames_written"] == 4
        assert "error" not in status
        buffer = status["buffer"]
        assert buffer["budget_bytes"] == stats["budget_bytes"]
        assert buffer["frame_bytes"] == stats["frame_bytes"]
        assert buffer["blocked_episodes"] == 0
        assert buffer["blocked_seconds"] == 0
        assert 0 < buffer["peak_bytes"] <= 4 * buffer["frame_bytes"]

    def test_cuda_buffer_below_one_frame_is_refused(self, tmp_path: Path) -> None:
        """64 atoms need ~7 KiB per captured frame: a 4096-byte driver budget
        holds none and the run refuses it when it starts, naming both sizes."""
        path = tmp_path / "gpu.mrec"
        sim = _lj_md("cuda", MrecRecorder(str(path), 1), capture_buffer_bytes=4096)
        with pytest.raises(
            ValueError, match=r"capture_buffer_bytes = 4096 .* needs \d+ bytes"
        ):
            sim.run(_argon_frame(), 2)
        assert not path.exists()

    def test_cuda_rejects_cpu_only_recorders(self, tmp_path: Path) -> None:
        sim = MD(device="cuda")
        with pytest.raises(TypeError, match="MrecRecorder"):
            sim.set_record(av.XYZ(str(tmp_path / "t.xyz"), 1))


# ── LAMMPS step convention ───────────────────────────────────────────────
#
# LAMMPS increments ``update->ntimestep`` at the top of every step
# (src/verlet.cpp ``Verlet::run``), dumps at the end of a step whose
# timestep is a multiple of ``every`` (src/output.cpp ``Output::write``),
# dumps the current state at the start of every run (``Output::setup``) and
# never writes a timestep twice (``last_dump``). The oracle is not the
# recorder: each frame is compared with an independent run of exactly that
# many steps on a fresh copy of the fixture.


def _state_after(device: str, steps: int) -> "molpy.Frame":
    """The fixture after exactly ``steps`` steps of a fresh MD with no
    recorder (``steps == 0``: the initial state)."""
    frame = _argon_frame()
    if steps > 0:
        sim = MD(device=device)
        sim.set_integrator(VelocityVerlet(_DT))
        sim.set_neighbor(NeighborList(_CUTOFF, _SKIN))
        sim.set_evaluator(LJCut(0.0104, 3.4, _CUTOFF))
        sim.run(frame, steps)
    return frame


def _assert_frame_is(recorded: "molpy.Frame", oracle: "molpy.Frame", atol: float) -> None:
    for col in ("x", "y", "z", "vx", "vy", "vz"):
        np.testing.assert_allclose(
            np.asarray(recorded["atoms"][col]),
            np.asarray(oracle["atoms"][col]),
            rtol=0,
            atol=atol,
            err_msg=col,
        )


def _assert_oracle_separates_steps(device: str, atol: float) -> None:
    a = np.asarray(_state_after(device, 2)["atoms"]["x"])
    b = np.asarray(_state_after(device, 3)["atoms"]["x"])
    assert np.max(np.abs(a - b)) > 100 * atol


class TestMrecRecorderLammpsSteps:
    def test_cpu_fresh_run_records_zero_through_n(self, tmp_path: Path) -> None:
        """6 steps every 3 → steps 0, 3, 6, each the state after that many
        steps; time = step · dt."""
        path = tmp_path / "cpu.mrec"
        _lj_md("cpu", MrecRecorder(str(path), 3)).run(_argon_frame(), 6)
        steps, time, frames = _read_all(path)
        assert steps == [0, 3, 6]
        assert time == pytest.approx([s * _DT for s in steps])
        _assert_oracle_separates_steps("cpu", 1e-10)
        for step, fr in zip(steps, frames):
            _assert_frame_is(fr, _state_after("cpu", step), 1e-10)

    @pytest.mark.skipif(not _HAS_CUDA_MD, reason="needs ATV_BUILD_CUDA")
    def test_cuda_continued_run_matches_independent_runs(self, tmp_path: Path) -> None:
        """Chunked CUDA runs (4 + 3 + 1 steps every 2) record steps 0, 2, 4,
        6, 8 — the boundary step once — each the state after that many
        steps of an independent CUDA run."""
        path = tmp_path / "gpu.mrec"
        frame = _argon_frame()
        sim = _lj_md("cuda", MrecRecorder(str(path), 2))
        for n in (4, 3, 1):
            sim.run(frame, n)
        del sim
        gc.collect()
        steps, time, frames = _read_all(path)
        assert steps == [0, 2, 4, 6, 8]
        assert time == pytest.approx([s * _DT for s in steps])
        _assert_oracle_separates_steps("cuda", 1e-9)
        for step, fr in zip(steps, frames):
            _assert_frame_is(fr, _state_after("cuda", step), 1e-9)


# ── Lossless backpressure (CUDA) ─────────────────────────────────────────
#
# Trajectory recording is asynchronous but lossless: when the driver's
# capture buffer is full the run waits for the writer, and a writer failure
# stops the run. The
# oracles are independent of the recorder (independent runs of exactly that
# many steps; the store's own status section).

#: Holds one frame of the 64-atom fixture (~7 KiB) and not two.
_ONE_FRAME_BUDGET = 10_000


def _set_writable(root: Path, writable: bool) -> None:
    """Add or remove write permission on ``root`` and everything below it."""
    paths = [root]
    for base, dirs, files in os.walk(root):
        paths += [Path(base) / d for d in dirs] + [Path(base) / f for f in files]
    for p in paths:
        mode = p.stat().st_mode
        if writable:
            p.chmod(mode | stat.S_IWUSR)
        else:
            p.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


@pytest.mark.skipif(not _HAS_CUDA_MD, reason="needs ATV_BUILD_CUDA")
class TestMrecRecorderCudaLossless:
    def test_one_frame_buffer_blocks_but_loses_nothing(self, tmp_path: Path) -> None:
        """A buffer of one frame, a frame every step: the run must wait for
        the writer, yet every step 0…12 is in the store, each the state after
        exactly that many steps of an independent run."""
        path = tmp_path / "gpu.mrec"
        n_steps = 12
        sim = _lj_md(
            "cuda", MrecRecorder(str(path), 1), capture_buffer_bytes=_ONE_FRAME_BUDGET
        )
        sim.run(_argon_frame(), n_steps)
        stats = sim.capture_stats()
        assert stats["capacity"] == 1
        assert stats["captures"] == n_steps + 1
        assert stats["blocked_episodes"] > 0
        del sim
        gc.collect()
        steps, time, frames = _read_all(path)
        assert steps == list(range(n_steps + 1))
        assert time == pytest.approx([s * _DT for s in steps])
        _assert_oracle_separates_steps("cuda", 1e-9)
        for step, fr in zip(steps, frames):
            _assert_frame_is(fr, _state_after("cuda", step), 1e-9)
        status = MrecFile(Path(path)).status()
        assert status["state"] == "succeeded"
        assert status["frames_written"] == n_steps + 1
        buffer = status["buffer"]
        assert buffer["budget_bytes"] == _ONE_FRAME_BUDGET
        assert buffer["frame_bytes"] <= _ONE_FRAME_BUDGET < 2 * buffer["frame_bytes"]
        assert buffer["peak_bytes"] == buffer["frame_bytes"]
        assert buffer["blocked_episodes"] > 0
        assert buffer["blocked_seconds"] > 0

    def test_write_failure_stops_the_run(self, tmp_path: Path) -> None:
        """The store becomes read-only between two runs: the second run's
        writer fails, and the run stops with RuntimeError naming the store
        (bounded by a watchdog, so a hang fails); the store's status records
        the failure."""
        path = tmp_path / "gpu.mrec"
        frame = _argon_frame()
        sim = _lj_md(
            "cuda",
            MrecRecorder(str(path), 1, flush_every=1, durable=False),
            capture_buffer_bytes=_ONE_FRAME_BUDGET,
        )
        sim.run(frame, 4)  # steps 0…4 land
        assert _read_all(path)[0] == [0, 1, 2, 3, 4]
        _set_writable(path, False)
        # A molpy.Frame cannot leave its thread, so the run stays here and a
        # hang is bounded by a watchdog that dumps the stacks and exits.
        faulthandler.dump_traceback_later(120, exit=True)
        try:
            with pytest.raises(RuntimeError) as err:
                sim.run(frame, 2000)
        finally:
            faulthandler.cancel_dump_traceback_later()
            _set_writable(path, True)
        assert str(path) in str(err.value)
        del err  # its traceback keeps `sim` alive
        del sim  # teardown commits what it can and writes the status
        gc.collect()
        status = MrecFile(Path(path)).status()
        assert status is not None
        assert status["state"] == "failed"
        assert str(path) in status["error"]["message"]
        assert status["frames_written"] >= 5

    def test_thermo_and_recorder_share_one_capture_per_step(
        self, tmp_path: Path
    ) -> None:
        """Thermo and the recorder read the same captured frame: with a
        one-frame budget the run neither deadlocks nor loses a step, and the
        buffer accounts one capture per step, not one per consumer."""
        path = tmp_path / "gpu.mrec"
        n_steps = 6
        sim = _lj_md(
            "cuda", MrecRecorder(str(path), 1), capture_buffer_bytes=_ONE_FRAME_BUDGET
        )
        sim.set_thermo(1)
        sim.run(_argon_frame(), n_steps)
        stats = sim.capture_stats()
        assert stats["capacity"] == 1
        assert stats["captures"] == n_steps + 1
        assert stats["peak_bytes"] == stats["frame_bytes"]
        del sim
        gc.collect()
        steps, _, frames = _read_all(path)
        assert steps == list(range(n_steps + 1))
        for step, fr in zip(steps, frames):
            _assert_frame_is(fr, _state_after("cuda", step), 1e-9)
