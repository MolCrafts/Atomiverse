"""Complete AM1 energy+force calls, including the Python/Frame boundary.

Run serially, with OMP_NUM_THREADS=1 and OPENBLAS_NUM_THREADS=1, before and
 after a change. The first argument is a JSON output path. Not a ctest test.
"""
import hashlib
import json
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
import _atomiverse

output = Path(sys.argv[1])
rows = []
for nwater in (1, 4, 8, 16):
    monomer = np.array([[0., 0., 0.], [.96, 0., 0.], [-.24, .93, .1]])
    xyz = np.concatenate([
        monomer + [4 * (i % 4), 4 * ((i // 4) % 4), 4 * (i // 16)]
        for i in range(nwater)
    ])
    frame = molpy.Frame()
    block = molpy.Block()
    for axis, key in enumerate(('x', 'y', 'z')):
        block.insert(key, xyz[:, axis].copy())
    block.insert('atomic_number', np.tile(np.array([8, 1, 1], dtype=np.uint32), nwater))
    block.insert('mass', np.tile(np.array([15.999, 1.008, 1.008]), nwater))
    frame['atoms'] = block
    frame.box = molpy.Box.cube(100.)
    samples = []
    for repeat in range(5):
        start = time.perf_counter()
        method = av.AM1(max_iter=1000, density_tol=1.e-10, energy_tol=1.e-12)
        forces = method.forces(frame)
        elapsed = time.perf_counter() - start
        samples.append(dict(kind='cold', seconds=elapsed, iterations=method.last_result().iterations))
    for repeat in range(20):
        start = time.perf_counter()
        forces = method.forces(frame)
        elapsed = time.perf_counter() - start
        samples.append(dict(kind='warm', seconds=elapsed, iterations=method.last_result().iterations))
    result = method.last_result()
    if not result.converged or not np.isfinite(forces).all():
        raise RuntimeError('Non-converged or non-finite benchmark result')
    rows.append(dict(
        atoms=len(xyz), samples=samples,
        cold_median_seconds=statistics.median(s['seconds'] for s in samples if s['kind'] == 'cold'),
        warm_median_seconds=statistics.median(s['seconds'] for s in samples if s['kind'] == 'warm'),
        energy_ev=result.total_energy_ev, forces_ev_a=forces.tolist(),
        charges_e=list(result.mulliken_charges),
    ))
module = Path(_atomiverse.__file__)
output.write_text(json.dumps(dict(
    platform=platform.platform(), machine=platform.machine(), python=sys.version,
    threads={k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
    module=str(module), module_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),
    density_tol=1.e-10, energy_tol_ev=1.e-12, max_iter=1000,
    process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
    rows=rows,
), indent=2) + '\n')
