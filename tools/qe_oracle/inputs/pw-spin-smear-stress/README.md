# H3 spin/smearing physical stress: actual QE 7.6 reference

`reference.json`, `pins.h` and `captures/` contain actual independent QE
energies, all atomic forces and Cartesian stress tensors for the same H3
geometry and GTH-PBE-q1 potential as the CPU/CUDA PW stress diagnostics.
The reference was run using the Arrhenius `QuantumESPRESSO/7.6` module;
the complete executable SHA256, exact inputs and output hashes are retained.
`capture.py` runs QE and parses printed quantities. `export_pins.py` exports
those numbers as data; neither implements Atomiverse's algorithms.

The physical model is three H atoms in the skew cell whose columns are
(4.2,0,0), (.3,4,0), (.2,.1,3.8) Å, at (1.2,1.1,1.0), (2.1,1.6,1.3),
(1.6,2.25,1.8) Å. It uses Γ, 6/24 Ry wave/density cutoffs, five bands per
spin, free collinear magnetization seeded at one electron, and a 72³ FFT
grid. PBE and explicit `SLA+VWN+NOGX+NOGC` (Slater/VWN5 LSDA) are separately
recorded. The GTH-PBE parameters are held fixed for both XC choices.

| Smearing | Width [Ry] | Qualification input |
| --- | ---: | --- |
| Fermi–Dirac | .02 | Nonzero magnetization and entropy contribution |
| Gaussian | .04 | Nonzero magnetization and generalized smearing correction |
| Methfessel–Paxton, order 1 | .04 | Nonzero magnetization and generalized smearing correction |

The original .12/.04/.02 FD probes and .02 MP runs remain in the capture
history. High-temperature FD can lose magnetization, and .02 MP can have a
negligible correction. They do not qualify the nontrivial joint-spin stress
envelope. MP quasioccupations may be negative or exceed one; they are not
clipped into probability bounds.

The native comparison bounds were declared before comparison: **1e-5 Ha
total/free energy and 2e-3 eV/Å³ for every stress component**. Stress is
compressive-positive, in physical Voigt order xx,yy,zz,xy,xz,yz. Its entries
come directly from QE's printed Ry/Bohr³ tensor, with the documented unit
conversion; off-diagonal components have no extra factor of two. QE's
[stress routine](https://github.com/QEF/q-e/blob/qe-7.5/PW/src/stress.f90)
documents the printed Cartesian tensor. The independent native diagnostics
also test simple shear, avoiding a shared symmetric-strain normalization.

The H potential is transcribed from the installed CP2K 2026.2 published
GTH_POTENTIALS block (path/hash in the record). QE's
[native GTH reader](https://github.com/QEF/q-e/blob/qe-7.5/upflib/gth.f90)
handles the analytic local operator and Hartree/Rydberg normalization.
The final empty s-channel record is a reader-format declaration with zero
projectors; this local H potential has no nonlocal contribution.

Reproduce on a compute allocation:

```sh
module load buildenv-gcc/2026.03-mpich QuantumESPRESSO/7.6-gcc-2026.03-mpich-eb
export LD_LIBRARY_PATH="$LIBRARY_PATH:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python capture.py /path/to/pw.x --matrix --out /path/to/fresh/qe-matrix
python capture.py /path/to/pw.x --mp-probe --out /path/to/fresh/qe-mp-probes
python export_pins.py --matrix /path/to/qe-matrix/reference.json \
  --mp-probes /path/to/qe-mp-probes/reference.json --out /path/to/export
```

The archived raw outputs are compressed. Decompress each `qe.out.gz` to
inspect all energy, force, stress and convergence lines. QE prints final
magnetization to two decimal places, so it certifies a nonzero moment here,
not an exact high-precision moment. This directory supplies an external
reference; native acceptance, build/source qualification and performance
require their own evidence.
