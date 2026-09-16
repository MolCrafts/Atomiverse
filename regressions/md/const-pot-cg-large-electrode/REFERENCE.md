# Reference — constant-potential CG solve (large multi-site electrode)

oracle: metalwalls
oracle_version: **MetalWalls release 20.05, git `c8ad175d` (release branch, 2026-06-18)** — real binary at
`/nobackup/proj/disk/teoroo/personal/jicli594/work/opt/metalwalls/mw`
verdict: **confirmed** (2026-07-29) — the pinned \(Q_{\mathrm{top}}\) reproduces MetalWalls to
\(3.1\times10^{-9}\) relative, 320× tighter than the stated \(10^{-6}\) tolerance.

Supersedes the pre-2026-07-29 entry, which pinned this case from the in-tree dense
`ConstPotBO` twin plus the MetalWalls *formula* (Scalfi/Siepmann–Sprik) because no
MetalWalls binary existed. The pin is unchanged; it is now backed by the real code.

## System

Larger planar capacitor with many electrode sites (queue geom: larger wall /
many electrode sites):

| Species | Type | Geometry | initial q [e] |
| --- | --- | --- | --- |
| Bottom electrode | 1 | \(4\times 4\) grid, \(z=8\) Å | 0 (solved) |
| Top electrode | 2 | \(4\times 4\) grid, \(z=12\) Å | 0 (solved) |
| Electrolyte + | 3 | box center \(xy\), \(z=9\) Å | +0.5 |
| Electrolyte − | 3 | box center \(xy\), \(z=11\) Å | −0.5 |

- \(N_{\mathrm{elec}} = 32\) (16 + 16)
- Box: \(L_x=L_y=4\) Å, \(L_z=20\) Å (orthorhombic, fully periodic ATV)
- Applied potentials: \(\Psi_{\mathrm{bot}}=0\) V, \(\Psi_{\mathrm{top}}=1\) V
- Gaussian width \(\eta=2.0\) Å⁻¹
- Electroneutrality on (`symm`)
- Lattice images: primary / min-image (`xy_image_range = 0`) for multi-site wall

The pinned number comes from the **primary-cell** sanity system (`madelung_primary()`):
4 particles sharing one \((x,y)\), electrode sites at \(z=8,12\) Å held at 0 and 1 V,
point ions \(\pm0.5\,e\) at \(z=9,11\) Å, cell \(1\times1\times20\) Å.

## Unit conversion (ATV eV/Å/e ⇄ MetalWalls Hartree/Bohr/e)

| Quantity | ATV | factor | MetalWalls |
| --- | --- | --- | --- |
| energy | eV | \(1\,\mathrm{Ha}=27.211386245988\) eV | Hartree |
| length | Å | \(1\,\mathrm{Bohr}=0.529177210903\) Å | Bohr |
| potential \(\Psi\) | 1 V | \(\div\,27.211386245988\) | \(0.03674932217565499\) Ha/e |
| Gaussian width \(\eta\) | 2.0 Å⁻¹ | \(\times\,0.529177210903\) | \(1.058354421806\) Bohr⁻¹ |
| site \(z=8\) Å | 8 Å | \(\times\,1.8897261254578\) | \(15.117808997\) Bohr |
| force | eV/Å | \(1\,\mathrm{Ha/Bohr}=51.42206747633\) eV/Å | Ha/Bohr |
| charge \(Q\) | e | **1** (MetalWalls a.u. charge = 1 e) | e |

Dimensional check on \(\Psi\): the solve is \(A Q=\Psi-b\) with \(A\sim k_e\eta\).
ATV \(k_e=14.399645\) eV·Å/e², MW \(k_e=1\) Ha·Bohr/e². The exact identity
\(k_e = \mathrm{Ha}\!\times\!\mathrm{Bohr} = 27.211386245988\times0.529177210903
= 14.399645478456\) eV·Å is what makes the two elastances the same operator; the
ATV literal is truncated at 8 digits (rel. \(3.3\times10^{-8}\)) and that truncation
is the *entire* residual disagreement below. Because \(Q\) is dimensionless in e,
no conversion is applied to the compared observable — the agreement therefore
cannot be an artefact of a conversion factor.

## MetalWalls twin — how it was run

Inputs (generator + `runtime.inpt`/`data.inpt`/raw `charges.out`) are committed at
`tools/metalwalls_oracle/inputs/metalwalls/const-pot-cg-large-electrode/`.

```bash
module load buildenv-gcc/2026.03-mpich
export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
cd <case-dir>; mpirun -n 1 .../opt/metalwalls/mw > run.stdout 2>&1
```

MetalWalls settings: `num_pbc 2`, `num_steps 0`, `electrode_charges cg 1.0e-14 2000`,
`charge_neutrality true electrodes`, `coulomb_rtol/ktol 1.0e-12` — i.e. the same
Jacobi-free CG constant-potential solve with the same \(\sum_i Q_i=0\) projector
Atomiverse's `ConstPotCG` implements. Syntax learned from upstream
`tests/charge_neutrality/Psi_average_0.5V` (which is where `potential 0.0367493225`
is confirmed to mean 1 V) and `tests/matrix_inversion/electrodes/LiCl-Al`.

**Why the twin needs a large cell.** MetalWalls has no non-periodic mode
(`num_pbc` ∈ {2,3}), while ATV builds this case with `xy_image_range = 0`, i.e.
min-image distances and *no* lattice sum. Every separation in the primary system is
along \(z\) and short compared with \(L_z=20\) Å, so ATV's min-image matrix is
**exactly the isolated 4-particle cluster**. The MW twin therefore takes
\(L_x=L_y=L\to\infty\) *and* \(L_z\to\infty\) (MetalWalls' 2D scheme discretizes the
\(k_z\) integral at spacing \(2\pi/L_z\), so \(L_z\) is a convergence parameter, not
just a container). Both limits were measured, not assumed:

Raw MetalWalls \(Q_{\mathrm{top}}\) [e] (`charges.out`, second entry; first entry is
\(Q_{\mathrm{bot}}=-Q_{\mathrm{top}}\) to 16 digits):

| \(L\) [Å] \ \(L_z\) [Å] | 200 | 400 | 800 | 1600 |
| --- | --- | --- | --- | --- |
| 200 | 0.2717534197407 | 0.2717534090720 | 0.2717534090539 | 0.2717534090539 |
| 400 | 0.2717535820153 | 0.2717535380244 | 0.2717535366985 | 0.2717535366963 |
| 800 | 0.2717536201144 | 0.2717535583154 | 0.2717535528293 | 0.2717535526638 |
| **1600** | 0.2717536296360 | 0.2717535630778 | 0.2717535553659 | **0.2717535546806** |

The \(L_z=1600\) Å column converges laterally as \(1/L^3\) (successive deltas
\(1.2764\times10^{-7}\), \(1.5968\times10^{-8}\), \(2.0167\times10^{-9}\); ratios
7.994, 7.918 vs. the ideal 8), so Richardson extrapolation is unambiguous:

\[
  Q_{\mathrm{top}}^{\mathrm{MW}}(\text{isolated}) = 0.2717535549687\ \mathrm{e}
  \quad(\pm\sim10^{-10})
\]

Provenance of the largest cell (`primary/L1600_Z1600/`): `alpha 3.32083E-03`,
`rcut 1.36060E+03` Bohr, `kmax 16 16 16`, 33 z-integral points, CG converged in
1 iteration with residual norm `3.43458E-17`; `energies.out` Hamiltonian
`-1.187394509001282E-001` Ha = \(-3.231065061080\) eV (contextual only — ATV's
`ConstPotCG` writes no energy, so no energy is pinned here).

## Result — pin CONFIRMED

| Quantity | Value [e] | vs MW |
| --- | --- | --- |
| MetalWalls `c8ad175d`, isolated limit | 0.2717535549687 | — |
| ATV `ConstPotCG` / `ConstPotBO` (printed) | 0.2717535558024 | \(+8.34\times10^{-10}\) abs, \(3.07\times10^{-9}\) rel |
| **hard-coded pin `kQTopPrimary`** | **0.27175356** | \(+5.03\times10^{-9}\) abs (tol \(10^{-6}\)) |
| same ATV formula with exact \(k_e=14.399645478456\) | 0.271753554945 | \(-2.4\times10^{-11}\) abs |

The \(8.3\times10^{-10}\) ATV−MW gap is fully accounted for by the truncated
`units::coulomb = 14.399645` literal (`core/units.h`): substituting the exact
\(\mathrm{Ha}\times\mathrm{Bohr}\) product reduces the gap to \(2.4\times10^{-11}\),
i.e. MetalWalls' round-off. **`old_pin = new_pin = 0.27175356`; nothing was
changed, no tolerance was touched.**

What this confirms is the whole model, not one number: the Gaussian self term
\(A_{ii}=k_e\eta\sqrt{2/\pi}\), the site–site elastance
\(A_{ij}=k_e\,\mathrm{erf}(\eta_{ij}r)/r\) with \(\eta_{ij}=\eta/\sqrt2\), the point-charge
RHS \(b_i=\sum_j k_e q_j\,\mathrm{erf}(\eta r_{ij})/r_{ij}\), the \(\Psi\) sign and the
V ⇄ Ha/e conversion, and the \(\sum_i Q_i=0\) projector — all agree with
MetalWalls' `MW_coulomb_self` / `MW_cg` to \(10^{-9}\).

## Known gap — `xy_image_range` is a truncation, not the periodic answer

Measured, not inferred, on the regression's own large plate (4×4+4×4 sites,
\(L_x=L_y=4\) Å, \(L_z=20\) Å; MW twin `plate/Z20`, identical for
\(L_z=20\ldots320\) Å so it is z-converged):

| Model | \(Q_{\mathrm{top}}\) [e] |
| --- | --- |
| MetalWalls full 2D Ewald | 0.301963702968 |
| ATV `xy_image_range = 0` (what the regression runs) | 0.365146378429 |
| | **+20.92 %** |

ATV's real-space image series *does* converge to MetalWalls — increasing
`xy_image_range` \(R\) gives a gap of exactly \(0.0260/R\)
(\(R=16\): \(1.623\times10^{-3}\); \(R=24\): \(1.084\times10^{-3}\);
\(R=32\): \(8.134\times10^{-4}\)) — but only as \(1/R\), so \(10^{-5}\) relative
accuracy would need \(R\approx 8.6\times10^3\), i.e. \(\sim3\times10^8\) image terms
per site pair. **`ConstPot{BO,CG}` has no Ewald-accurate electrode elastance.**
That is a capability gap for the MetalWalls loop queue, not a defect in this
regression: the regression pins only the isolated-limit \(Q_{\mathrm{top}}\) (where
min-image is exact) and the CG-vs-dense-BO equivalence on a *shared* \(A\), both of
which are correct as asserted. Do **not** compare the plate \(Q_{\mathrm{top}}\)
printed by the binary against a MetalWalls number until an Ewald elastance exists.

## What Atomiverse asserts

Public API: `cpu::md::ConstPotCG` on `md::Phase::pre_force`, writes
`kAtom_charge`.

| Check | Tol |
| --- | --- |
| \(N_{\mathrm{elec}} \ge 18\) | exact |
| \(\lvert\sum Q_i\rvert\) | \(\le 10^{-10}\) e |
| \(\max_i \lvert Q_i^{\mathrm{CG}} - Q_i^{\mathrm{BO}}\rvert\) | \(\le 10^{-8}\) e |
| \(\lvert Q_{\mathrm{group}}^{\mathrm{CG}} - Q_{\mathrm{group}}^{\mathrm{BO}}\rvert\) | \(\le 10^{-8}\) e |
| Primary \(Q_{\mathrm{top}}\) vs 0.27175356 | abs \(\le 10^{-6}\) e |

Observed 2026-07-29: primary \(Q_{\mathrm{top}} = 2.717535558024\times10^{-1}\),
\(\max_i|\Delta Q_i| = 3.25\times10^{-16}\), \(|\sum Q_i| = 2.34\times10^{-17}\) — PASS.

## Perf

```
perf_exempt: false
oracle: MetalWalls c8ad175d, mpirun -n 1, tools/.../const-pot-cg-large-electrode/plate/Z20
atv: ./build/regressions/reg_const-pot-cg-large-electrode, single rank, OMP_NUM_THREADS=1
wall (best of 5): ATV 0.01 s   MetalWalls 0.58 s
ratio_atv_over_mw: 0.026
within_10x: yes
```

Same box, same 32 electrode sites, same zero MD steps, whole-process wall on both
sides. **Read it with the caveat:** MetalWalls' 0.58 s includes MPI start-up and
Ewald k-table construction and it computes the *converged 2D-Ewald* elastance,
while ATV's 0.01 s (of which the 32-site CG solve is 0.032 ms and the dense BO
0.203 ms) computes only the min-image truncation. At equal accuracy the ordering
reverses — see the gap section above. The ratio is reported because the systems
match; it is not a claim of an algorithmic win.

## Repro

```bash
# ATV (in ctest / regressions target; no external binary involved)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DATV_BUILD_MD=ON -DATV_BUILD_KSPACE=ON \
      -DATV_BUILD_TESTS=ON -DATV_BUILD_REGRESSIONS=ON -DBLA_VENDOR=OpenBLAS
cmake --build build --target reg_const-pot-cg-large-electrode -j4
./build/regressions/reg_const-pot-cg-large-electrode

# MetalWalls oracle (OFFLINE ONLY — never at ctest time)
cd tools/metalwalls_oracle/inputs/metalwalls/const-pot-cg-large-electrode
python3 gen_inputs.py                     # regenerates every case directory
module load buildenv-gcc/2026.03-mpich
export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
cd primary/L1600_Z1600 && mpirun -n 1 .../opt/metalwalls/mw   # -> charges.out
```
