"""Complete CPU AM1/RM1/MNDO/PM3 forces, including Python/Frame boundaries.

Run serially before and after a change, with OMP_NUM_THREADS=1 and
OPENBLAS_NUM_THREADS=1. Usage: run.py OUTPUT.json [AM1|RM1|MNDO|PM3].
Not a CTest test. Native CPU/CUDA comparisons use the separate run.cu harness.
"""
import hashlib
import json
import math
import os
import platform
import resource
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import molpy
import atomiverse as av

METHODS = ("AM1", "RM1", "MNDO", "PM3")
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")
CONTROLS = dict(max_iter=1000, density_tol=1.e-10, energy_tol=1.e-12)


def make_frame(nwater):
    monomer = np.array([[0., 0., 0.], [.96, 0., 0.], [-.24, .93, .1]])
    xyz = np.concatenate([
        monomer + [4 * (i % 4), 4 * ((i // 4) % 4), 4 * (i // 16)]
        for i in range(nwater)
    ])
    frame = molpy.Frame()  # Zero cell, matching the native isolated CUDA path.
    block = molpy.Block()
    for axis, key in enumerate(('x', 'y', 'z')):
        block.insert(key, xyz[:, axis].copy())
    block.insert('atomic_number', np.tile(np.array([8, 1, 1], dtype=np.uint32), nwater))
    block.insert('mass', np.tile(np.array([15.999, 1.008, 1.008]), nwater))
    frame['atoms'] = block
    return frame, xyz


def set_geometry(frame, original, step=0):
    """Prescribed geometry edits, not a dynamics trajectory; outside timing."""
    xyz = original.copy()
    xyz[1::3, 0] += .002 * math.sin(.2 * step)
    xyz[2::3, 1] += .002 * math.sin(.17 * step)
    xyz[2::3, 2] += .001 * math.sin(.11 * step)
    for axis, key in enumerate(('x', 'y', 'z')):
        frame['atoms'].insert(key, xyz[:, axis].copy())


def scientific_sample(method, forces, kind, repeat, seconds, atoms):
    """Extract after timing so diagnostics do not contaminate the call interval."""
    result = method.last_result()
    forces = np.asarray(forces)
    charges = np.asarray(result.mulliken_charges)
    if (not result.converged or forces.shape != (atoms, 3)
            or charges.shape != (atoms,) or not np.isfinite(forces).all()
            or not np.isfinite(charges).all()
            or not math.isfinite(result.total_energy_ev)
            or not math.isfinite(result.heat_of_formation_kcal_mol)):
        raise RuntimeError(f'Invalid {type(method).__name__} {atoms}-atom {kind} result {repeat}')
    return dict(
        kind=kind, repeat=repeat, seconds=seconds, iterations=result.iterations,
        energy_ev=result.total_energy_ev,
        heat_of_formation_kcal_mol=result.heat_of_formation_kcal_mol,
        charges_e=charges.tolist(), forces_ev_a=forces.reshape(-1).tolist(),
        force_squared_sum=float(np.sum(forces * forces)),
    )


def measure(name, nwater):
    frame, original = make_frame(nwater)
    method_type = getattr(av, name)
    samples = []
    for repeat in range(5):
        start = time.perf_counter()
        method = method_type(**CONTROLS)
        forces = method.forces(frame)
        elapsed = time.perf_counter() - start
        samples.append(scientific_sample(method, forces, 'cold', repeat, elapsed, len(original)))
        del method  # Destruction is outside the measured cold interval.
    for kind in ('warm_fixed', 'warm_moving'):
        set_geometry(frame, original)
        method = method_type(**CONTROLS)
        for _ in range(5):
            method.forces(frame)
        for repeat in range(20):
            if kind == 'warm_moving':
                set_geometry(frame, original, repeat + 1)
            start = time.perf_counter()
            forces = method.forces(frame)
            elapsed = time.perf_counter() - start
            samples.append(scientific_sample(method, forces, kind, repeat, elapsed, len(original)))
        del method
    return dict(
        method=name, atoms=len(original), samples=samples,
        median_seconds={kind: statistics.median(s['seconds'] for s in samples if s['kind'] == kind)
                        for kind in ('cold', 'warm_fixed', 'warm_moving')},
    )


def main():
    if len(sys.argv) not in (2, 3) or (len(sys.argv) == 3 and sys.argv[2] not in METHODS):
        raise SystemExit('usage: run.py OUTPUT.json [AM1|RM1|MNDO|PM3]')
    if any(os.environ.get(key) != '1' for key in THREAD_KEYS):
        raise SystemExit('set OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1')
    names = (sys.argv[2],) if len(sys.argv) == 3 else METHODS
    rows = [measure(name, nwater) for name in names for nwater in (1, 4, 8, 16)]
    module = Path(av.require_cpp().__file__)
    Path(sys.argv[1]).write_text(json.dumps(dict(
        schema_version=2,
        boundary='CPU public forces including Python/Frame conversion; zero cell; cold includes constructor; '
                 'Frame construction, geometry writes, destruction and serialization excluded',
        cold_calls=5, warmup_calls=5, warm_calls_per_mode=20,
        moving_geometry='step=repeat+1; from original coordinates: H1.x+=.002*sin(.2*step); '
                        'H2.y+=.002*sin(.17*step); H2.z+=.001*sin(.11*step) Angstrom',
        platform=platform.platform(), machine=platform.machine(), python=sys.version,
        threads={key: os.environ[key] for key in THREAD_KEYS},
        module=str(module), module_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),
        controls=CONTROLS,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                               * (1 if sys.platform == 'darwin' else 1024),
        rss_scope='whole Python process high-water mark; includes imports/all cases; not per-call memory',
        rows=rows,
    ), indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
