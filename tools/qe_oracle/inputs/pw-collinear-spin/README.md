# O₂ collinear PBE: Quantum ESPRESSO energy oracle

The energy pin in `reference.json` is a number printed by an actual Quantum
ESPRESSO 7.5 `pw.x` run on 2026-10-25. The source revision is
[`770a0b2d12928a67048e2f3da8d10d057e52179e`](https://github.com/QEF/q-e/tree/770a0b2d12928a67048e2f3da8d10d057e52179e).
The archived inputs and complete QE outputs are the evidence; native
validation files are kept separately and do not generate oracle pins.

## Physical inputs and acceptance

Both QE inputs use the repository's SG15 NC ONCV PBE oxygen UPF,
`tests/data/pw/O_ONCV_PBE-1.2.upf`, SHA-256
`212d0bcacf75a66f5e086100069a3ddf5d754afd335f054ddfb4ca4b9e730400`.
The periodic cell is cubic 8 Å, with oxygen at (0, 0, 0) and (0, 0, 1.2075) Å.
PBE, Γ sampling, fixed occupations, 12 valence electrons, total magnetization
2, and wavefunction/density cutoffs 12/48 Ry match the native regression.
QE uses `conv_thr=1e-10`, `electron_maxstep=200`, `mixing_beta=0.3`, Davidson
diagonalization and `diago_thr_init=1e-10`. The native calculation uses its
own unmixed two-spin residual criterion `tol=1e-7`, `max_iter=200` and
`pulay_depth=5`; these are different numerical convergence criteria.

The acceptance bounds were retained: energy error ≤1e-3 Ha, fixed-occupation
magnetization error ≤1e-9, GGA activation relative to the native LSDA solve
>1e-4 Ha, converged finite results, and ≤60 s for each whole native section.
The LSDA calculation supplies the activation comparison, not an external
absolute-energy pin.

| Archived input | Output | Purpose and actual result |
| --- | --- | --- |
| `o2_pbe.in` | `o2_pbe.qe.txt` | Energy-only oracle; default grid 36³, 24 iterations, **−57.46535365 Ry = −28.732676825 Ha** |
| `o2_pbe_grid72.in` | `o2_pbe_grid72.qe.txt` | Grid diagnostic; explicit 72³, forces enabled, 25 iterations, −57.46536001 Ry |

The 72³ energy differs by 3.18e-6 Ha. Its input enables forces, so its wall
cannot be used as a speed comparison with the 36³ energy-only calculation.
The canonical energy line is `o2_pbe.qe.txt:671`; convergence is reported at
line 683 and QE's internal wall at line 736.

## Reproduce the external calculation

Build QE at the stated revision. The exact original CMake command and build
target are in `reference.json`; `compiler-version.txt` records the compiler.
The original build disabled MPI, OpenMP, ScaLAPACK, Libxc and HDF5 and used
Homebrew OpenBLAS/FFTW. The actual binary path and SHA-256 are recorded in
`reference.json` rather than assumed to identify a future rebuild.

From the Atomiverse repository root, prepare a clean working directory and
run the archived input with the real QE executable:

```sh
repo_root="$PWD"
mkdir -p /tmp/qe-o2-reproduce
cp "$repo_root/tests/data/pw/O_ONCV_PBE-1.2.upf" /tmp/qe-o2-reproduce/
cp "$repo_root/tools/qe_oracle/inputs/pw-collinear-spin/o2_pbe.in" /tmp/qe-o2-reproduce/
cd /tmp/qe-o2-reproduce
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /tmp/atv-qe-7.5-build/bin/pw.x -in o2_pbe.in > o2_pbe.qe.txt
```

The actual original command was the same `pw.x -in o2_pbe.in`, run in
`/tmp/atv-gpw-repair-evidence/qe-spin`, with OpenMP/OpenBLAS 1/1. It exited
zero. For the grid diagnostic, copy and run `o2_pbe_grid72.in` similarly.
Keep the two outputs distinct. Input/output hashes and the measured process
walls are archived in `reference.json`.

## Native validation and timing boundaries

`native-pw-spin-polarized-o2.txt` and `native-pw-spin-gga.txt` are exact
section extracts from `/tmp/atv-gpw-repair-evidence/pw-spin-qe-final.log`.
`native-validation.json` records the source hash, extract hashes, thread
width and boundaries. These are native validation evidence, not oracle data.
The captured log does not record its exact command line; the case is
`pw/collinear-spin`.

The native PBE result is −28.7326751302 Ha, 35 iterations and M=2. Its error
against the external pin is 1.6948e-6 Ha. The one-PBE section takes
16.865870 s. The GGA section additionally runs LSDA (−28.6078529734 Ha,
34 iterations), giving a printed activation gap 1.248222e-1 Ha and whole
section time 26.585936 s. Both sections pass the strict 60 s gate.

QE's energy-only subprocess takes 1.882814625 s; its printed internal
`PWSCF` wall is 1.85 s. The quotient 16.865870 / 1.882814625 ≈8.96 is
**observational**: the native value is a complete section timer while the
QE value is a process timer, and native debug iteration logging is enabled.
No independent final performance claim is made before a final batch without
debug logging. The two-SCF GGA section is not compared to the one-SCF QE
process. The force-enabled 72³ diagnostic process takes 4.822693042 s and
supplies only the grid-precision observation above.

This archive validates the changed CPU NC-UPF radial/projector path.
Historical CUDA job records in the regression reference remain historical;
the affected CUDA slices need a fresh same-settings GH200 validation.
