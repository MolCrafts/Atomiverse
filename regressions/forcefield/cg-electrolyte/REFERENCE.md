# forcefield/cg-electrolyte

Archived evidence, no runner. One single point of a coarse-grained polymer
electrolyte, term by term, on Atomiverse CPU, Atomiverse CUDA and LAMMPS.

## Oracle

```text
oracle: lammps
oracle_version: LAMMPS (2 Sep 2026), git develop e8c4c278a8-modified
binary: lab/projects/pzpe/engines/lmp-cu132 (aarch64, packages KOKKOS KSPACE MOLECULE)
run: GH200 node, plain (non-suffixed) double-precision styles, 1 MPI rank
```

Inputs: `tools/lammps_oracle/inputs/cg_electrolyte/` — `make_case.py`
(deterministic generator, seed 20261029) writes `system.txt` (Atomiverse),
`data.lmp`, `in.lmp`, `lj_rows*.lmp` (archived next to it) and `r4.table`
(14 MB, regenerate). The LAMMPS output lines (`ATV_RESULT` / `ATV_VIRIAL`
of every term, the PPPM accuracy estimate; orders 5, 6, 7) are in
`lammps_output.txt`; `comparison.order{5,6,7}.txt` are the output of
`compare.py` for `CG_ORDER` = 5, 6 (default), 7.

## Model

Units: σ0 = 1 Å, m = 1 amu, ε0 = k_B · 300 K = 0.025851999786 eV
(k_B = 8.617333262e-5 eV/K); LAMMPS `units metal`.

| type | B | E | C | A | T | L |
|---|---|---|---|---|---|---|
| σ [Å] | 1.00 | 1.00 | 1.10 | 1.15 | 1.20 | 0.55 |
| ε_ii [ε0] | 1 | 1 | 1.2 | 1.2 | 1 | 1 |
| z | 0 | 0 | +1 | −1 | −1 | +1 |

- LJ: Lorentz–Berthelot with ε_LT = 1.0 ε0 and ε_LA = 1.2 ε0 overrides,
  each pair cut at 2.5 σ_ij and shifted (`pair_modify shift yes`). LAMMPS
  gets all 21 rows explicitly; Atomiverse gets the six diagonals (own
  cutoff) plus the two override rows and mixes the rest (LAMMPS `init_one`
  in `core/lj_type_table.h`).
- Charges q = z · sqrt(ε0 ℓ_B* σ0 / 14.399645) e = ±0.073389141347849 e
  (ℓ_B* = 3). Real-space Coulomb cutoff 3.5 Å; PPPM ik, α = 1.1 Å⁻¹,
  mesh 32³ (h = 0.28125 Å), order 6 (LAMMPS estimate: RMS force accuracy
  5e-8 eV/Å).
- FENE K = 30 ε0/σ0², R0 = 1.5 σ0, no WCA (LAMMPS `bond_style fene` with
  ε = 0). Cosine angle K = 2 ε0 on B–B–B only.
- L–E r⁻⁴: U = −S[(σ/r)⁴ − (σ/r_c)⁴ + 4(σ/r_c)⁴(r/r_c − 1)], S = 4 ε0,
  σ = 0.775 Å, r_c = 2.5σ. Atomiverse: `set_r4_attraction`. LAMMPS:
  `pair_style table linear 200000` (RSQ spacing from 0.3 Å, the closed form
  tabulated by `make_case.py`) — in `hybrid/overlay` with
  `lj/cut/coul/long` for the total, so the term is counted once.
- `special_bonds lj/coul 1 1 1`.
- 360 beads in a 9 Å cube: 6 chains of 12 B (8 PEGM B–E–E–E, 2 METAC B–C,
  2 MTFSI B–A per chain, shuffled), 60 free L and 60 free T; 234 bonds,
  60 angles; every nonbonded distance ≥ 0.85 σ_ij.

Terms run separately on all three engines (`-var term …` in LAMMPS,
`cg_electrolyte.cpp` / `cuda-cg-electrolyte.cu` here): `fene`, `angle`,
`lj` (`PairLJCut`, typed), `r4` (`PairLJCut` with ε = 0 rows), `coul`
(`PairLJCutCoulLong` with ε = 0 rows + `KSpacePPPM`), `kspace` (the same
with a 0.05 Å real-space cutoff; LAMMPS `coul/long 0.05`), `total`
(`PairLJCutCoulLong` + r⁻⁴ + FENE + cosine + PPPM). `real` = `coul` −
`kspace` (energy, forces, virial).

## Metrics

rel dE = |E_a − E_b| / |E_b|; F max = max_i |ΔF_i| / max_i |F_i|;
F rms = ‖ΔF‖₂ / ‖F‖₂; W = max_k |ΔW_k| / max_k |W_k| over the six virial
components (LAMMPS pressure × V / nktv2p).

## Result (order 6, `comparison.order6.txt`)

| term | E [eV] (LAMMPS) | CPU−LAMMPS dE / F max / W | CUDA−LAMMPS dE / F max / W | CUDA−CPU dE / F max / W |
|---|---|---|---|---|
| fene | 110.57442410567712 | 3.7e-15 / 1.2e-15 / 8.4e-16 | 2.8e-15 / 2.4e-13 / 8.4e-16 | 9.0e-16 / 2.4e-13 / 1.4e-16 |
| angle | 1.2714959031187130 | 0 / 3.7e-16 / 5.1e-16 | 0 / 4.8e-12 / 6.8e-16 | 0 / 4.8e-12 / 3.4e-16 |
| lj | 4.4414604965008335 | 1.6e-15 / 1.1e-15 / 9.2e-16 | 2.4e-15 / 2.8e-14 / 1.2e-15 | 8.0e-16 / 2.8e-14 / 7.7e-16 |
| r4 | −2.7730480875579722 | 3.3e-10 / 1.2e-9 / 5.7e-10 | 3.3e-10 / 1.2e-9 / 5.7e-10 | 0 / 3.4e-13 / 1.7e-16 |
| kspace | −1.0808960556019092 | 6.2e-9 / 1.1e-9 / 1.1e-9 | 6.2e-9 / 1.1e-9 / 1.1e-9 | 7.9e-14 / 1.6e-12 / 1.9e-14 |
| real | 0.097167545578892733 | 3.4e-7 / 1.3e-7 / 4.1e-7 | 3.4e-7 / 1.3e-7 / 4.1e-7 | 3.7e-14 / 4.5e-12 / 4.3e-15 |
| total | 112.53060390771569 | 2.2e-10 / 2.9e-9 / 4.8e-10 | 2.2e-10 / 2.9e-9 / 4.8e-10 | 1.4e-15 / 1.2e-13 / 1.8e-15 |

Every engine's terms sum to its total to ≤ 2.5e-16. Orders 5 and 7
(`comparison.order5.txt`, `comparison.order7.txt`) give the same rows for
every term: k-space 6.2e-9 / 1.1e-9 / 1.1e-9 on both backends against LAMMPS,
CUDA − CPU k-space ≤ 1.1e-13 / 1.6e-12 / 3.4e-14 (table below).

## Attribution of the differences above 1e-12

A direct pair sum (`exact_terms.py`, pure Python, libm `erfc`) is an extra
consistency check that attributes each difference to one engine's function
approximation; it is not the oracle.

| engine vs direct sum (order 6) | real: dE / F max / W | r4: dE / F max / W |
|---|---|---|
| CPU vs exact | 5.7e-16 / 3.4e-16 / 1.8e-15 | 3.2e-16 / 3.2e-16 / 1.0e-15 |
| CUDA vs exact | 3.6e-14 / 4.5e-12 / 3.0e-15 | 3.2e-16 / 3.4e-13 / 8.7e-16 |
| LAMMPS vs exact | 3.4e-7 / 1.3e-7 / 4.1e-7 | 3.3e-10 / 1.2e-9 / 5.7e-10 |
| LAMMPS vs A&S emulation | 2.4e-15 / 4.2e-16 / 5.9e-15 | — |

- **Real-space Coulomb.** Both Atomiverse backends evaluate erfc(αr)
  analytically and match the exact sum to rounding: CPU
  `PairLJCutCoulLong` through the vectorized fdlibm erfc of
  `include/atomiverse/cpu/potentials/ewald_erfc.h` (xsimd), CUDA through
  erfcx(αr)·e^{−α²r²}. LAMMPS `pair_modify table 0` uses the
  Abramowitz–Stegun 7.1.26 polynomial (|Δerfc| ≤ 1.5e-7) and the truncated
  EWALD_F = 1.12837917 (`src/ewald_const.h`); the emulation reproduces
  LAMMPS to 1e-15, so CPU − LAMMPS and CUDA − LAMMPS real space (3.4e-7 /
  1.3e-7 / 4.1e-7) are exactly LAMMPS's polynomial floor. All three apply the
  same cutoff and pair set. Before this change the CPU interpolated a linear
  r² table (`cpu::md::ErfcTable`, 10000 bins): CPU − exact 3.7e-6 / 7.6e-6 /
  6.8e-6, the whole CPU − CUDA difference of `coul` and `total`; the table is
  removed.
- **r⁻⁴.** LAMMPS interpolates the tabulated closed form linearly in r².
  With 2·10⁶ table points (`comparison.r4-table-2e6.txt`) the LAMMPS − exact
  difference falls to 3.7e-12 / 2.2e-11 / 5.7e-12, the Δ² scaling of linear
  interpolation; both Atomiverse backends equal the closed form.
- **k-space.** LAMMPS `compute_gf_ik` uses 12.5663706 for 4π (relative
  1.143e-9 below 4π): forces and virial differ by exactly that factor; the
  energy difference 6.2e-9 is the same factor on the reciprocal sum
  (≈ 5.85 eV) relative to the self-energy-reduced total (−1.08 eV).
  Atomiverse CPU and CUDA PPPM agree to 1e-13 (energy) and 1.6e-12
  (forces, CUDA fixed point).
- **CUDA forces** of the bonded terms carry the fixed-point force columns'
  resolution (≤ 5e-12 relative); energies and virials are in FP64.

## PPPM order (odd and even)

Both backends spread charge and gather the ik field on LAMMPS's centred
stencil (`src/KSPACE/pppm.cpp` `set_grid_local` / `particle_map` /
`make_rho` / `fieldforce_ik`, ported in `include/atomiverse/core/pppm_ik.h`):
grid points nx + nlower … nx + nupper, nlower = −(p−1)/2, nupper = p/2, nx the
grid point below the charge for even p and the nearest one for odd p. Before
this change both used the one-sided smooth-PME stencil [⌊u⌋ − p + 1, ⌊u⌋] —
for even p the LAMMPS mesh shifted by whole cells (same discretization), for
odd p half a cell away. k-space term, CPU − LAMMPS (CUDA − CPU after the
change in the last column):

| order | LAMMPS E_kspace [eV] | before: dE / F max / W | after: dE / F max / W | after, CUDA − CPU |
|---|---|---|---|---|
| 5 | −1.0809020049967888 | 1.3e-5 / 2.8e-5 / 4.7e-5 | 6.2e-9 / 1.1e-9 / 1.1e-9 | 1.1e-13 / 1.6e-12 / 2.5e-14 |
| 6 | −1.0808960556019092 | 6.2e-9 / 1.1e-9 / 1.1e-9 | 6.2e-9 / 1.1e-9 / 1.1e-9 | 7.9e-14 / 1.6e-12 / 1.9e-14 |
| 7 | −1.0808954194771965 | 3.0e-7 / 1.5e-6 / 1.8e-6 | 6.2e-9 / 1.1e-9 / 1.1e-9 | 9.5e-14 / 1.5e-12 / 3.3e-14 |

The "before" CPU numbers come from the base tree (dev e10b6effb + the merge
patch), rebuilt and rerun against the same LAMMPS logs; order 7 reproduces
the previously archived 3.0e-7 / 1.5e-6 / 1.8e-6 (CPU − CUDA then 9e-14).
After the port every order differs from LAMMPS by its truncated 4π only. The
unit tests `PppmStencil.*` (tests/core, tests/cpu/longrange) and
`CudaPppmStencil.*` (tests/cuda/longrange) pin the placement against a
literal LAMMPS `make_rho` / `fieldforce_ik` reference
(`tests/core/pppm_lammps_reference.h`) for orders 3–7, orthorhombic and
triclinic; the MPI test `PppmSubcommParity.BrickArmsMatchReplicatedForOddAndEvenOrders`
pins the brick ghost extents (LAMMPS `Grid3d::ghost_grid` for this stencil).

## Found while building this case

`cuda::PairLJCutCoulLong` on its per-atom Lorentz–Berthelot tile (taken when
every interacting pair sits at the global cutoff) used the reported cutoff
— the largest of the Coulomb and every LJ row cutoff — as the pair cutoff,
so an ε = 0 row with a cutoff beyond the Coulomb cutoff extended real-space
Coulomb (and any LJ pair) past `cut_coul`. Fixed (the LB tile now uses the
Coulomb cutoff, `src/cuda/potentials/pair_lj_cut_coul_long.cu`); unit test
`*PairLJCutCoulLongTypedCutoff.ZeroEpsilonRowCutoffDoesNotExtendCoulombOrLJ`
on both backends. The `kspace` term above is the case that exposed it.

With the stencil port: `cpu::KSpacePPPM` chose its mesh from {2, 3, 5, 7}
sizes with an even K2 and defaulted to order 5, `cuda::KSpacePPPM` from
LAMMPS `factorable` {2, 3, 5} sizes with order 6; both now use the LAMMPS
rule (`longrange::pppm_fft_size`) and order 6 (the Python default). The
triclinic alias-sum bound now follows `compute_gf_ik_triclinic` (H^T applied
to the per-axis bound) instead of the column lengths. The CPU move
constructor dropped the serial FFT buffer the FFTW plans point at and the MPI
request array; both are now moved. None of these changes this case (32³ mesh,
cubic cell, order given explicitly).

## After the merge (PPPM leftovers)

Rerun on the merged tree (CPU x86_64, CUDA GH200 13.2.1, LAMMPS oracle
rerun on the GH200 for every term and order): every CPU − LAMMPS and
CUDA − LAMMPS figure above is unchanged; CUDA − CPU moved in the last digit
(k-space F max 1.5e-12 → 1.6e-12, the fixed-point force columns; a repeat
run moves CUDA − CPU by ≤ 1e-14, the atomic summation order). The
changes in this step do not touch this case's numbers (32³ mesh, Nyquist
G ≈ 1e-11 of its peak, fixed α and mesh): the half-spectrum fold of each mode
with its conjugate partner (`core/pppm_ik.h` `pppm_half_mode`), which makes
even meshes equal LAMMPS's full complex `poisson_ik` (unit tests
`PppmStencil.EvenMesh*` / `CudaPppmStencil.EvenMesh*`, energy, forces and
virial); the triclinic alias bound with LAMMPS `lamda2xT`'s |tilt|; the CPU
brick ghosts sized by `0.5·skin`; and `KSpacePPPM::from_accuracy`
(LAMMPS `PPPM::init`, `core/pppm_accuracy.h`), whose final estimate for this
system equals the oracle's printed `estimated absolute RMS force accuracy`
for orders 5, 6, 7 and the 0.05 Å k-space-only cutoff
(`PppmAccuracy.FinalEstimateMatchesTheCgElectrolyteOracle`).

## Reproduce

```sh
# any directory outside the tree
python3 tools/lammps_oracle/inputs/cg_electrolyte/make_case.py RUN   # CG_ORDER=5 / 7, CG_R4_N=2000000 for the variants
cg_electrolyte RUN/system.txt RUN          # CPU harness (this directory)
cuda-cg-electrolyte RUN/system.txt RUN     # CUDA harness, GH200
for t in fene angle lj r4 coul kspace total; do
  (cd RUN && lmp -in in.lmp -var term $t -log log.$t -screen none); done
python3 tools/lammps_oracle/inputs/cg_electrolyte/exact_terms.py RUN
python3 tools/lammps_oracle/inputs/cg_electrolyte/compare.py RUN
```

The harnesses are standalone `main()` programs compiled against a build of
the library (no CMake registration); they were built with the unit-test
targets' compile flags and link line (`atv_tests` on the x86_64 login node,
`atv_tests_cuda` on the GH200, CUDA 13.2.1, `nvcc -rdc=true`).
