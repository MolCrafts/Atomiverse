# HF benchmark: Apple M4, 2026-10-26

This is a separate hardware profile from the GH200 campaigns. Do not construct
a controlled cross-machine speedup from these times.

RHF/STO-3G, canonical 16-water chain, 48 atoms / 112 AOs, complete public Python
call, fresh state and one CPU thread. Five warm samples and every cold result
passed the existing 1e-6 Ha gate.

| ATV CPU direct | ATV CPU stored | PySCF CPU | Hardware / environment |
| ---: | ---: | ---: | --- |
| 7.757518 s | 21.827065 s | 1.941593 s | Apple M4; one requested/observed thread; complete details in the matrix |

The original recorder omitted PySCF's macOS `.dylib` fingerprint. Accuracy and
raw timings remain valid observations, but controlled speed ratios are withheld
because the original binary identity is incomplete. The adapter now records
macOS shared libraries for future campaigns; this report does not retroactively
insert a fingerprint into an existing measurement.
The original per-worker files were in a temporary directory and are no longer
available; the compressed bundle retains every measured sample and its metadata.

[Full matrix](water16-performance-cpu1/matrix.md) ·
[HTML](water16-performance-cpu1/matrix.html) ·
[Original evidence](water16-performance-cpu1/measurements.json.gz).

CUDA and gpu4pyscf were unavailable in this CPU-only environment and remain
explicit missing cells. The environment uses the published molpy 0.20.0 wheel.
The source snapshot is dirty and the installed build is not an official clean
commit qualification; no roadmap promotion or global publication is claimed.
