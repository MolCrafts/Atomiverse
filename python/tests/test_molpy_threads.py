"""A molpy.Frame borrowed through the capsule bridge may cross Python threads.

Workflow runners build a frame in a worker thread (``asyncio.to_thread``,
thread pools) and compute on it, or drop it, on another. The frame's store
is thread-safe, so the borrowed handle is too.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor

import molpy
import numpy as np
import pytest
from _domains import skip_module_unless

skip_module_unless("ATV_BUILD_SEMIEMPIRICAL")

import atomiverse as av

#: CH4 H offset per axis [Å], the geometry test_semiempirical.py pins.
_H = 0.629118


def _ch4_frame() -> molpy.Frame:
    frame = molpy.Frame()
    atoms = molpy.Block()
    atoms.insert("x", np.array([0.0, _H, -_H, -_H, _H]))
    atoms.insert("y", np.array([0.0, _H, -_H, _H, -_H]))
    atoms.insert("z", np.array([0.0, _H, _H, -_H, -_H]))
    atoms.insert("mass", np.array([12.011, 1.008, 1.008, 1.008, 1.008]))
    atoms.insert("atomic_number", np.array([6, 1, 1, 1, 1], dtype=np.uint32))
    frame["atoms"] = atoms
    frame.box = molpy.Box.cube(20.0)
    return frame


def _on_worker(fn, *args):
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(fn, *args).result()


@pytest.fixture
def unraisable():
    seen = []
    previous = sys.unraisablehook
    sys.unraisablehook = lambda record: seen.append(record.exc_value)
    yield seen
    sys.unraisablehook = previous


def _single_point(frame: molpy.Frame) -> tuple[float, list[float]]:
    av.single_point(frame, av.AM1(charge=0))
    return frame.meta["energy"], frame["atoms"]["charge"].tolist()


def test_single_point_on_a_frame_built_in_another_thread(unraisable) -> None:
    reference = _single_point(_ch4_frame())

    frame = _on_worker(_ch4_frame)
    energy, charges = _single_point(frame)

    assert np.isfinite(energy)
    assert energy == reference[0]
    assert charges == reference[1]
    del frame
    assert unraisable == []


def test_single_point_in_a_worker_writes_back_to_this_threads_frame(
    unraisable,
) -> None:
    reference = _single_point(_ch4_frame())

    frame = _ch4_frame()
    assert _on_worker(_single_point, frame) == reference
    # the worker's write-back reached the frame this thread holds
    assert frame.meta["energy"] == reference[0]
    assert frame["atoms"]["charge"].tolist() == reference[1]

    held = [frame]
    del frame
    _on_worker(held.clear)
    assert unraisable == []
