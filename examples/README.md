# Examples

A flat, numbered tutorial sequence. Each file teaches exactly one concept and
is meant to be read top-to-bottom; numbers define the recommended reading
order.

| # | File | Concept | Requires |
|---|------|---------|----------|
| 01 | `01_scf.cpp` | The bare SCF loop on raw xtensor arrays (no atomiverse link) | — |
| 02 | `02_rhf.cpp` / `02_rhf.cu` | First real chemistry: RHF single-point (CPU via `cpu::scf::RHF`; CUDA twin via `single_point + cuda::scf::SCF`) | `ATV_BUILD_HF` (+`ATV_BUILD_CUDA` for the twin) |
| 03 | `03_uhf.cpp` | Open-shell UHF (`cpu::scf::UHF`) + spin contamination | `ATV_BUILD_HF` |
| 04 | `04_uks.cpp` | DFT (UKS-LSDA) with the native XC stack (`cpu::scf::UKS`) | `ATV_BUILD_HF` |
| 05 | `05_geom_opt.cpp` | Geometry optimization: `Minimize + LBFGS + cpu::scf::RHF` (RHF/STO-3G forces) | `ATV_BUILD_HF` |
| 06 | `06_aimd.cpp` | Ab-initio MD: `MD + VelocityVerlet + NoseHoover + cpu::scf::RHF` | `ATV_BUILD_HF` |
| 07 | `07_md_lj.cpp` / `07_md_lj.cu` | Classical MD (NVE LJ argon): `md::MD + NeighborList + PairLJCut` (CPU canonical; CUDA twin = `07_md_lj_cuda`) | — (`.cu` needs `ATV_BUILD_CUDA`) |
| — | `in.lj` | Same LJ argon through the native `atomiverse -in` CLI | `ATV_BUILD_CLI` (default ON with MD) |
| 08 | `08_pw.cpp` | Plane-wave DFT for a periodic crystal: `Evaluate + pw::PWEvaluator` (silicon) | `ATV_BUILD_PW` |
| 09 | `09_gpw.cpp` | Gaussian-and-plane-waves DFT single point: `cpu::gpw::GPWEvaluator` | `ATV_BUILD_GPW` |
| 11 | `11_koopmans.cpp` | Koopmans' theorem: HOMO energy → vertical IP via `scf::OrbitalInfoObserver` (5×2 basis sweep) | `ATV_BUILD_HF` |
| 12 | `12_delta_scf.cpp` | ΔSCF ionization energies beyond Koopmans: neutral RHF vs `SCFInitialGuess`-seeded cation UHF | `ATV_BUILD_HF` |
| 13 | `13_mp2.cpp` | Closed-shell MP2 correlation energy (`scf::mp2_energy`, AO→MO ovov transform) | `ATV_BUILD_HF` |
| 14 | `14_fci.cpp` | Exact correlation via tiny full CI (`scf::fci_energy`) on H2/STO-3G | `ATV_BUILD_HF` |
| 15 | `15_am1.cpp` | AM1 semiempirical: energy, ΔHf, forces, Minimize (`AM1Evaluator`) | — |
| 16 | `16_density_cube.cpp` | Write H₂ density + HOMO Gaussian cubes for MolVis | `ATV_BUILD_HF` |

## Literature-facing results

After a successful run, the printed numbers used by the documentation live in
[`results/`](results/). Tutorials embed those values in tables and MolPlot
charts. When you change physics that an example exercises:

1. Re-run the binary.
2. Update the matching `results/*.json`.
3. Update any hard-coded tables/charts in `docs/tutorial/`.

Do **not** invent numbers for the docs. References cite textbooks / experiment;
Atomiverse columns always come from a real binary on this tree.

## Rules for adding an example

1. **One concept per file.** If a file needs two headlines, split it.
2. **Numbered, flat, ordered.** Pick the next free number in the pedagogical
   order; no subdirectories.
3. **CPU `.cpp` is the canonical lesson.** A CUDA `.cu` twin reuses the same
   number and gets a `_cuda`-suffixed target (`02_rhf.cu` → `02_rhf_cuda`,
   `07_md_lj.cu` → `07_md_lj_cuda`).
4. **Examples are not end-to-end checks.** Anything that measures speed or
   compares against LAMMPS / PySCF / CP2K / QE belongs in
   `regressions/<domain>/<case>/` — an out-of-library consumer checked against
   a cited value. Never in `tests/`.
5. Register the target in `CMakeLists.txt` behind the narrowest option gate
   (`ATV_BUILD_HF`, `ATV_BUILD_CUDA`, …).
6. **Ship a `results/<name>.json`** when the example produces a number the
   docs should quote (energy, IP, drift, ΔH_f, …).

Build: configure with `-DATV_BUILD_EXAMPLES=ON`, then
`cmake --build build --target 02_rhf` (any number works the same way).
