"""Shared native NDDO Python end-to-end checks, called by method cases.

Oracle data and method selection belong to each case. This module exercises
real SCF, public result channels and composition; it implements no solver.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import molpy
import atomiverse as av
from atomiverse._binding import _cpp

Z = {"H": 1, "C": 6, "N": 7, "O": 8}
MASS = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999}


def frame_for(pin):
    frame, atoms = molpy.Frame(), molpy.Block()
    xyz = np.asarray(pin["coordinates_angstrom"], dtype=np.float64)
    for axis, key in enumerate(("x", "y", "z")):
        atoms.insert(key, np.ascontiguousarray(xyz[:, axis]))
    for key in ("vx", "vy", "vz", "fx", "fy", "fz"):
        atoms.insert(key, np.zeros(len(xyz), dtype=np.float64))
    atoms.insert("atomic_number", np.array([Z[e] for e in pin["elements"]], dtype=np.uint32))
    atoms.insert("mass", np.array([MASS[e] for e in pin["elements"]], dtype=np.float64))
    frame["atoms"] = atoms
    return frame


def columns(frame, keys):
    return np.column_stack([np.asarray(frame["atoms"][key]) for key in keys])


def near(actual, expected, tolerance):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=tolerance)


def method_for(pin, device):
    return getattr(av, pin["method"])(max_iter=1000, density_tol=1e-10,
                                     energy_tol=1e-12, device=device)


def check_pin(method, frame, pin, force=True):
    result = method.last_result()
    assert result.converged
    near(frame.meta["energy"], pin["total_energy_ev"], 1e-6)
    near(result.total_energy_ev, pin["total_energy_ev"], 1e-6)
    near(result.mulliken_charges, pin["mulliken_charges_e"], 1e-7)
    near(result.heat_of_formation_kcal_mol, pin["heat_of_formation_kcal_mol"], 1e-6)
    near(np.asarray(frame["atoms"]["charge"]), result.mulliken_charges, 1e-12)
    if force:
        near(columns(frame, ("fx", "fy", "fz")), pin["forces_ev_angstrom"], 1e-5)
        near(np.asarray(result.forces_ev_a).reshape(-1, 3), pin["forces_ev_angstrom"], 1e-5)


def boundaries(pin, device):
    method, frame = method_for(pin, device), frame_for(pin)
    force = method.forces(frame)
    check_pin(method, frame, pin)
    preserved = np.array(force, copy=True)
    assert method.run(frame) is None
    check_pin(method, frame, pin, force=False)
    near(columns(frame, ("fx", "fy", "fz")), preserved, 0)
    near(force, preserved, 0)  # the returned array owns its memory
    assert len(method.last_result().forces_ev_a) == 0
    evaluation = av.Evaluate(device=device).set_evaluator(method)
    evaluation.run(frame)
    check_pin(method, frame, pin)
    # The method keeps its public result and remains valid after owner cleanup.
    del evaluation
    gc.collect()
    method.forces(frame)
    check_pin(method, frame, pin)
    # Explicit finite-budget failure leaves the public object retryable.
    method.max_iter = 0
    try:
        method.forces(frame)
    except (RuntimeError, ValueError) as error:
        assert "budget" in str(error).lower() or "iteration" in str(error).lower()
    else:
        raise AssertionError("zero iteration budget unexpectedly succeeded")
    method.max_iter = 1000
    method.forces(frame)
    check_pin(method, frame, pin)
    av.single_point(frame, method, device=device)
    check_pin(method, frame, pin)


def copy_state(frame):
    """Independent Frame storage, including the old force passed to a restart."""
    copied, atoms = molpy.Frame(), molpy.Block()
    for key in ("x", "y", "z", "vx", "vy", "vz", "fx", "fy", "fz", "mass", "atomic_number"):
        atoms.insert(key, np.array(frame["atoms"][key], copy=True))
    copied["atoms"] = atoms
    copied.box = frame.box
    return copied


def same_observables(frame, reference):
    near(frame.meta["energy"], reference.meta["energy"], 1e-8)
    near(columns(frame, ("fx", "fy", "fz", "charge")),
         columns(reference, ("fx", "fy", "fz", "charge")), 1e-6)


def md_edits(pin, device, driver, method, frame, pins):
    # Keep the same Python Frame and native MD owner across every edit.
    x = np.array(frame["atoms"]["x"], copy=True)
    x[1] += .013
    frame["atoms"].insert("x", x)
    vz = np.array(frame["atoms"]["vz"], copy=True)
    vz[0] -= .0003
    frame["atoms"].insert("vz", vz)
    reference = copy_state(frame)
    restart = av.MD(dt=.01, device=device)
    restart.set_evaluator(method_for(pin, device))
    driver.run(frame, 1)
    restart.run(reference, 1)
    near(columns(frame, ("x", "y", "z", "vx", "vy", "vz")),
         columns(reference, ("x", "y", "z", "vx", "vy", "vz")), 1e-8)
    same_observables(frame, reference)
    del restart
    gc.collect()
    # CPU molecular methods accept a box; this CUDA slice explicitly rejects it.
    # Restoring the original isolated boundary must permit the same owner to retry.
    frame.box = molpy.Box.cube(20.)
    if device == "cuda":
        unchanged = columns(frame, ("x", "y", "z", "vx", "vy", "vz"))
        try:
            driver.run(frame, 0)
        except (ValueError, RuntimeError) as error:
            assert any(word in str(error).lower() for word in ("box", "periodic", "cell"))
        else:
            raise AssertionError("CUDA periodic box was silently accepted")
        near(columns(frame, ("x", "y", "z", "vx", "vy", "vz")), unchanged, 0)
    else:
        reference = copy_state(frame)
        method_for(pin, device).forces(reference)
        driver.run(frame, 0)
        same_observables(frame, reference)
    frame.box = None
    reference = copy_state(frame)
    method_for(pin, device).forces(reference)
    driver.run(frame, 0)
    same_observables(frame, reference)

    name = pin["method"].lower()
    co = pins[name + "-co"]
    frame["atoms"] = frame_for(co)["atoms"]
    driver.run(frame, 0)  # changed atom count and orbital identity
    check_pin(method, frame, co)
    for key in ("atomic_number", "mass"):
        frame["atoms"].insert(key, np.array(frame["atoms"][key], copy=True)[::-1].copy())
    reference = copy_state(frame)
    method_for(pin, device).forces(reference)
    driver.run(frame, 0)  # same count, changed elements
    same_observables(frame, reference)
    ammonia = pins[name + "-ammonia"]
    frame["atoms"] = frame_for(ammonia)["atoms"]
    driver.run(frame, 0)
    check_pin(method, frame, ammonia)
    print(f"PASS {device} {pin['method']} Python same-owner MD x/v, box/retry, count/element edits")


def composition(pin, device, pins):
    if hasattr(_cpp, "MD") and (device == "cpu" or hasattr(_cpp, "_CudaMD")):
        split_frame, whole_frame = frame_for(pin), frame_for(pin)
        split_method, whole_method = method_for(pin, device), method_for(pin, device)
        split, whole = av.MD(dt=.01, device=device), av.MD(dt=.01, device=device)
        split.set_evaluator(split_method)
        whole.set_evaluator(whole_method)
        split.run(split_frame, 0)
        check_pin(split_method, split_frame, pin)
        split.run(split_frame, 2)
        split.run(split_frame, 2)
        whole.run(whole_frame, 4)
        near(columns(split_frame, ("x", "y", "z", "vx", "vy", "vz", "fx", "fy", "fz")),
             columns(whole_frame, ("x", "y", "z", "vx", "vy", "vz", "fx", "fy", "fz")), 1e-8)
        near(split_frame.meta["energy"], whole_frame.meta["energy"], 1e-8)
        assert split_method.last_result().converged
        md_edits(pin, device, split, split_method, split_frame, pins)
        del split, whole
        gc.collect()
        # Parent teardown releases the original method for a standalone solve.
        split_method.forces(split_frame)
        check_pin(split_method, split_frame, pins[pin["method"].lower() + "-ammonia"])
        print(f"PASS {device} {pin['method']} Python MD initial + 2+2 vs 4")
    else:
        print(f"SKIP {device} MD binding not built")
    if hasattr(_cpp, "Minimize") and (device == "cpu" or hasattr(_cpp, "_CudaMinimize")):
        frame, method = frame_for(pin), method_for(pin, device)
        optimizer = (av.LBFGS(alpha0=.01) if device == "cpu" else
                     av.SteepestDescent(alpha=.01, device=device))
        minimize = av.Minimize(device=device)
        minimize.set_optimizer(optimizer)
        minimize.set_evaluator(method)
        minimize.add_criterion(av.ForceConverged(1e-10))
        minimize.run(frame, 0)
        check_pin(method, frame, pin)
        initial_energy = float(frame.meta["energy"])
        outcome = minimize.run(frame, 3)
        assert 0 < outcome.iterations <= 3
        assert np.isfinite(frame.meta["energy"]) and frame.meta["energy"] < initial_energy
        final_energy = float(frame.meta["energy"])
        final_force = columns(frame, ("fx", "fy", "fz"))
        del minimize
        gc.collect()
        method.forces(frame)
        near(frame.meta["energy"], final_energy, 1e-8)
        near(columns(frame, ("fx", "fy", "fz")), final_force, 1e-6)
        print(f"PASS {device} {pin['method']} Python Minimize initial + 3 finite descent steps")
    else:
        print(f"SKIP {device} Minimize binding not built")


def main(pins_path: Path, methods: tuple[str, ...]):
    all_pins = json.loads(pins_path.read_text())
    pins = {name: pin for name, pin in all_pins.items() if pin["method"] in methods}
    for method in methods:
        assert all(method.lower() + "-" + molecule in pins
                   for molecule in ("water", "co", "ammonia")), method
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda", "both"), default="cpu")
    args = parser.parse_args()
    devices = ("cpu", "cuda") if args.device == "both" else (args.device,)
    for device in devices:
        for name, pin in pins.items():
            boundaries(pin, device)
            print(f"PASS {device} {name} Python run/forces/Evaluate/retry external E/q/F")
        for method in methods:
            composition(pins[method.lower() + "-water"], device, pins)
