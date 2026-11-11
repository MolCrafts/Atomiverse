# md/constant-potential

One case; every section below is run by `atv_regressions --case md/constant-potential` and keeps its own oracle record.


## Reference — constant-potential BO solve (parallel plate / madelung)

oracle: lammps  
oracle_version: 3 Nov 2022 (ELECTRODE package example `madelung`, log `log.1Dec2022.madelung-ewald-ew3dc.g++.1`)

### System

LAMMPS ELECTRODE **madelung** twin (Ahrens-Iwers et al., *J. Chem. Phys.* **157**,
084801 (2022), DOI [10.1063/5.0099239](https://doi.org/10.1063/5.0099239)):

| Species | Type | z [Å] | initial q [e] |
| --- | --- | --- | --- |
| Bottom electrode | 1 | −2 | 0 (solved) |
| Top electrode | 2 | +2 | 0 (solved) |
| Electrolyte + | 3 | −1 | +0.5 |
| Electrolyte − | 3 | +1 | −0.5 |

Box: \(L_x=L_y=1\) Å, \(L_z=20\) Å (slab-like; electrode solve uses **2D \(xy\)
Gaussian lattice images**, v1 — not full `ewald/electrode`). Atom \(z\) values are
the madelung relative coordinates shifted into \((0,L_z)\) so fully periodic ATV
boxes do not wrap the slab under VelocityVerlet (LAMMPS example uses
`boundary p p f`).

Applied potentials: \(\Psi_{\mathrm{bot}}=0\) V, \(\Psi_{\mathrm{top}}=1\) V,
\(\Delta\Psi=1\) V. Gaussian width \(\eta=2.0\) Å⁻¹. Electroneutrality on
(`symm`).

### Offline pin (LAMMPS ELECTRODE log)

From `~/work/lammps/examples/PACKAGES/electrode/madelung/log.1Dec2022.madelung-ewald-ew3dc.g++.1`
(`run 0` after `fix electrode/conp 0 2 couple top 1 symm on`,
`kspace_style ewald/electrode 1e-8`, `kspace_modify slab 8.0`):

| Observable | Value |
| --- | --- |
| \(Q_{\mathrm{bot}}\) [e] | **−0.279214485147238** |
| \(Q_{\mathrm{top}}\) [e] | **+0.279214485147238** |
| PE (real units, kcal/mol) | 139.943964815502 (not compared — unit system differs) |

### Literature cross-check (Madelung / plate_cap)

`plate_cap.py` in the same example directory: pure real-space 2D Gaussian
lattice sum of the BO matrix (Siepmann–Sprik Gaussian charges). With metal
units (`units::coulomb = 14.399645`) and image range ≳ 200:

\[
Q_{\mathrm{top}}^{\mathrm{Madelung}} \approx +0.27941\,e
\]

(relative difference vs LAMMPS Ewald-electrode pin ≈ **0.07%**).

### What Atomiverse asserts

Public API: `cpu::md::ConstPotBO` on `md::Phase::pre_force`, writes
`kAtom_charge`.

| Check | Tol |
| --- | --- |
| \(\lvert\sum Q_i\rvert\) | \(\le 10^{-10}\) e |
| \(\lvert Q_{\mathrm{top}} - 0.279214485147238\rvert / 0.279214485147238\) | **≤ 1%** (v1 2D-image Gaussian \(A\) vs full Ewald electrode) |
| \(\mathrm{sign}(Q_{\mathrm{top}})=+1\) for \(\Delta\Psi=+1\) V | exact |

### Approximation note (map E1)

v1 builds \(A\) from Gaussian real-space + 2D \(xy\) images only. Full
`ewald/electrode` / MetalWalls reciprocal electrode matrix is **E2**
(`slab-2d-ewald-foothold`). The 1% gate is deliberate for this foothold.

### Perf

```
perf_exempt: true
perf_exempt_reason: local LAMMPS binary lacks ELECTRODE package; no fair twin binary for fix electrode/conp
ratio_atv_over_oracle: n/a
within_10x: exempt
```

### Repro of pin (offline only — never ctest)

```bash
## Requires LAMMPS built with -D PKG_ELECTRODE=ON
cd $LAMMPS_DIR/examples/PACKAGES/electrode/madelung
lmp -in in.ewald-ew3dc
## thermo: pe c_qbot c_qtop at step 0
```

Or literature Madelung:

```bash
python3 plate_cap.py   # writes plate_cap.csv (real units)
```


## Reference — constant-potential CG solve (large multi-site electrode)

oracle: metalwalls
oracle_version: **MetalWalls release 20.05, git `c8ad175d` (release branch, 2026-06-18)** — real binary at
`/nobackup/proj/disk/teoroo/personal/jicli594/work/opt/metalwalls/mw`
verdict: **confirmed** (2026-07-29) — the pinned \(Q_{\mathrm{top}}\) reproduces MetalWalls to
\(3.1\times10^{-9}\) relative, 320× tighter than the stated \(10^{-6}\) tolerance.

Supersedes the pre-2026-07-29 entry, which pinned this case from the in-tree dense
`ConstPotBO` twin plus the MetalWalls *formula* (Scalfi/Siepmann–Sprik) because no
MetalWalls binary existed. The pin is unchanged; it is now backed by the real code.

### System

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

### Unit conversion (ATV eV/Å/e ⇄ MetalWalls Hartree/Bohr/e)

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

### MetalWalls twin — how it was run

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

### Result — pin CONFIRMED

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

### Known gap — `xy_image_range` is a truncation, not the periodic answer

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

### What Atomiverse asserts

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

### Perf

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

### Repro

```bash
## ATV (in ctest / regressions target; no external binary involved)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DATV_BUILD_MD=ON -DATV_BUILD_KSPACE=ON \
      -DATV_BUILD_TESTS=ON -DATV_BUILD_REGRESSIONS=ON -DBLA_VENDOR=OpenBLAS
cmake --build build --target reg_const-pot-cg-large-electrode -j4
./build/regressions/reg_const-pot-cg-large-electrode

## MetalWalls oracle (OFFLINE ONLY — never at ctest time)
cd tools/metalwalls_oracle/inputs/metalwalls/const-pot-cg-large-electrode
python3 gen_inputs.py                     # regenerates every case directory
module load buildenv-gcc/2026.03-mpich
export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
cd primary/L1600_Z1600 && mpirun -n 1 .../opt/metalwalls/mw   # -> charges.out
```


## Reference — ghost-charge correctness after the const-pot Q solve (replicated ranks)

oracle: literature  
oracle_version: n/a (self-consistency invariant; no oracle binary is run —
see citations below; perf_exempt)

### Invariant under test

Constant-potential MD rewrites electrode charges **mid-step** (the Q solve at
`pre_force`), i.e. after the step's forward halo has populated ghost atoms.
The updated owner charges must be **forward-communicated to every ghost copy
before the force stages read them** — otherwise pair/k-space Coulomb forces
across a periodic face use charges that are one solve stale (and identically
zero on the first step, since electrode charges start at 0). CommBrick creates
explicit periodic image rows at every rank count, so the ghosts under test are
the self-images at the ±x / ±y faces of the cell; a remote ghost rides the
same `ChargePacker` forward pass.

This is the documented contract of the LAMMPS ELECTRODE implementation: after
every conp charge solve, `fix electrode` issues a forward communication of the
per-atom charge so ghosts match owners before force evaluation.

- A. P. Thompson et al., *LAMMPS — a flexible simulation tool for
  particle-based materials modeling at the atomic, meso, and continuum
  scales*, Comput. Phys. Commun. **271**, 108171 (2022),
  DOI 10.1016/j.cpc.2021.108171 — forward/reverse ghost communication
  semantics (owner → ghost refresh of per-atom state).
- L. J. V. Ahrens-Iwers, M. Janssen, S. R. Tee, R. H. Meißner,
  *ELECTRODE: An electrochemistry package for atomistic simulations*,
  J. Chem. Phys. **157**, 084801 (2022), DOI 10.1063/5.0099239 —
  constant-potential charge solve per step; solved charges are
  forward-communicated (`comm->forward_comm`) before force evaluation.
- A. Marin-Laflèche et al., *MetalWalls: A classical molecular dynamics
  software dedicated to the simulation of electrochemical systems*,
  J. Open Source Softw. **5**, 2373 (2020), DOI 10.21105/joss.02373 —
  the MetalWalls constant-potential loop this capability maps.

### System (child, np = 2)

38-atom parallel-plate capacitor, box 20 × 10 × 16 Å (orthorhombic). Both
ranks run the whole system as a replica (`MD::set_serial_on_replica`): the
electrode solve requires a complete Frame, so a forced 2×1×1 spatial
decomposition is attempted first and must be **rejected** at frame binding
(gate d) rather than run as a rank-local partial solve.

| Species | Type | Count | z [Å] | q₀ [e] |
| --- | --- | --- | --- | --- |
| Bottom plate (Ψ = 0 V) | 1 | 15 (5×3 grid) | 5 | 0 (solved) |
| Top plate (Ψ = 1 V) | 2 | 15 (5×3 grid) | 11 | 0 (solved) |
| Electrolyte ions | 3 | 8 | 7–9.5 | ±0.25 (neutral) |

Pipeline (all defaults — the packer refresh is the feature under
test): `VelocityVerlet{0.1 fs}` + `NeighborList{3.0, 0.3}` +
`PairCoulCut{3.0}` + `ConstPotBO{η = 2 Å⁻¹}` (electroneutrality on, matrix
rebuilt per step). Halo / exchange / reverse live on `ctx.md.comm`;
`ConstPotBO` posts `ChargePacker` after writing owned charges. 5 MD steps.

An out-of-library `GhostChargeChecker` Component (defined in the regression
file — the user-extension path) runs at the force phase and compares every
ghost row's charge against its owner's current value (tag-matched via
allgather).

### Pinned numbers (hard-coded in the regression)

| Gate | Value | Why |
| --- | --- | --- |
| max \|q_ghost − q_owner\| over all ghosts, ranks, steps | **0** (tol 1e-15 e) | halo transport is a memcpy of the owner value — bitwise |
| electrode ghosts with \|q\| > 1e-12 e | **> 0** | non-vacuous: electrode charges start at 0; only the refreshed solve makes ghosts nonzero |
| unresolved ghost tags | **0** | every ghost must map to an owner |
| checker launches | ≥ 5 (one per step) | invariant enforced on every step |
| decomposed `ConstPotBO` run | **rejected** (`requires a complete Frame`) | no silent rank-local partial solve |

Without the auto-injected charge refresh, gate (a) fails with
max |dq| ≈ the solved electrode charge magnitude (~1e-2 e in this geometry)
at the first step — the stale-ghost failure mode this regression pins.

### Scope note

A DD-aware global electrode solve is a separate queued capability; until it
exists the decomposed combination is refused explicitly. This regression pins
the **transport** invariant: whatever charge the owner solved, every ghost
copy carries it before forces read it. Runtime ≈ seconds (≤ 60 s).


## Reference — electrode total charge observer (parallel-plate foothold)

### oracle

```text
oracle: lammps
oracle_version: LAMMPS (30 Mar 2026 - Development - patch_30Mar2026-1074-g7f680de296)
```

Binary used offline: `/home/jicli594/.local/x86_64/bin/lmp` (no ELECTRODE package
required; pin uses core `compute reduce sum` on group charge).

### Physical system

Parallel-plate **constant-charge control** toy for E1 observe (map §1 / §8):

| Quantity | Value |
| --- | --- |
| Box | orthorhombic \(20 \times 10 \times 10\) Å |
| Left wall | type 1, 4 atoms, \(q_i = +0.125\) e |
| Right wall | type 2, 4 atoms, \(q_i = -0.125\) e |
| Electrolyte placeholders | type 3, 2 atoms, \(q_i = +1\) e (excluded from electrode sums) |
| Ensemble | none (`run 0`) — pure charge accounting |

Analytic totals:

\[
Q_L = 4 \times 0.125 = +0.5\,\mathrm{e},\qquad
Q_R = 4 \times (-0.125) = -0.5\,\mathrm{e}.
\]

### Offline pin procedure

Input twin: `tools/metalwalls_oracle/inputs/lammps/const_pot_charge_observer.in`

```bash
lmp -in tools/metalwalls_oracle/inputs/lammps/const_pot_charge_observer.in
```

Recorded screen lines:

```text
PIN_QL 0.5
PIN_QR -0.5
```

Thermo step 0: `c_ql = 0.5`, `c_qr = -0.5`.

### Assertion

Atomiverse `cpu::ElectrodeChargeObserver({1, 2})` on the same geometry/charges:

| Observable | Pin | Tol |
| --- | ---: | ---: |
| \(Q_L\) [e] | \(+0.5\) | \(10^{-12}\) |
| \(Q_R\) [e] | \(-0.5\) | \(10^{-12}\) |
| \(Q_L + Q_R\) [e] | \(0\) | \(10^{-12}\) |

Tolerance is machine-eps scale: the pin is an exact finite sum of fixed charges
(no force loop, no k-space). Catches wrong type selection, ghost double-counting,
or unit mistakes; not a golden of Atomiverse-only output.

### Why LAMMPS (not MetalWalls)

Summing group charge is standard LAMMPS (`variable atom q` + `compute reduce sum`).
The ELECTRODE package is not installed on this oracle binary; for **this**
observer the reduce-sum path is the correct LAMMPS twin. MetalWalls remains the
oracle for MW-only BO/CG/MaZe charge *solves* (other E1 slugs).

### Perf

```text
perf_exempt: true
perf_exempt_reason: >
  Micro-observer is a single O(N) sum over 10 atoms. Fair wall vs LAMMPS is
  dominated by binary startup (~0.3 s for `run 0` with no pair/kspace), not
  the reduce itself. Not a force-loop twin; ratio would not measure product
  MD throughput.
```

When a longer MD+electrode fair bench exists (const-pot-md-nvt-short), re-record
\(T_{\mathrm{atv}}/T_{\mathrm{oracle}}\) there.


## const-pot-il-edl-box

oracle: metalwalls
oracle_version: MetalWalls release 20.05, git **c8ad175d** (2026-06-18)
oracle_kind: real binary empty capacitor (re-pinned 2026-07-29); with-IL
absolute Q not expressible under ATV truncated image-sum elastance

### Geometry

Honeycomb C walls (112 sites each) at z = 12 / 28 Å, Ψ = ∓1 V,
η = 1.805132 Å⁻¹; 12+12 CG IL ions; box ≈ 17.04 × 17.22 × 40 Å.

Twin: `tools/metalwalls_oracle/inputs/metalwalls/const-pot-il-edl-box/mw_twin/`.

### Pins (MetalWalls printed)

| case | Q_left [e] | Q_right [e] |
|------|------------|-------------|
| empty | **−2.076457628129008×10⁻¹** | **+2.076457628129015×10⁻¹** |
| with-IL | −4.818386149307364 | +4.818386149307384 |

ATV empty at `xy_image_range=80` matches MW to ~0.5 %. Absolute with-IL Q
under image-sum (~1.36 e) does **not** approach MW Ewald (~4.82 e) —
gate screening order `|Q|_IL > |Q|_empty` + polarity + neutrality instead.

### Superseded

Wave-5 `pin_il_edl.py` numpy BO at `xy_image_range=0` was a self-golden of
ATV's own min-image elastance (forbidden by pin-provenance rule).

### Perf

`perf_exempt` — product box; wall ratio deferred.


## Reference — constant-potential MD NVT short trajectory

oracle: metalwalls
oracle_version: MetalWalls release 20.05, git **c8ad175d** (2026-06-18),
binary at `/nobackup/proj/disk/teoroo/personal/jicli594/work/opt/metalwalls/mw`
oracle_kind: real binary (re-pinned 2026-07-29)

### Map slice

E1 product: fluctuating electrode charges at fixed potential, frozen walls,
classical Coulomb MD on `cpu::md::MD`.

### Geometry

- Box 14×14×20 Å; walls 2×2 at z = 2 / 18 Å (types 2/3); 4 Na⁺ + 4 Cl⁻
- Ψ = ∓0.5 V, Gaussian w = 0.5 Å (η = 2 Å⁻¹)
- Soft LJ + CoulLong/PME (α = 0.3 Å⁻¹); NVT 400 × 0.5 fs at 300 K

### Pin (MetalWalls c8ad175d)

Twin: `tools/metalwalls_oracle/inputs/metalwalls/const-pot-md-nvt-short/mw_twin/`
(`runtime_pbc3.inpt`, `num_pbc 3` matches ATV's 3D PME cell).

From `mw_output/total_charges.out`:

| step | Q_C1 [e] | Q_C2 [e] |
|------|----------|----------|
| 0 | **−9.711155787094686×10⁻²** | **+9.711155787094686×10⁻²** |
| 1 | −9.742254974821957×10⁻² | +9.742254974821957×10⁻² |

### Assertions

| Check | Bound |
| ----- | ----- |
| Polarity after step 1 | Q_L < 0 < Q_R |
| Neutrality | \|Q_L+Q_R\| ≤ 1e-8 e |
| Soft \|Q\| vs MW step-0 | 5 % rel (PME vs Ewald residual ~3 %) |
| Finite E/K/charges over NVT | no NaN |
| Freeze walls | Δx ≤ 1e-10 Å |
| End polarity | Q_L < 0 < Q_R |

### Perf

`perf_exempt` — short product trajectory; fair wall ratio deferred.


## const-pot-tip4p-edl

oracle: metalwalls
oracle_version: MetalWalls git **c8ad175d** (re-checked 2026-07-29)
oracle_kind: real binary for polarity/neutrality; absolute |Q| not expressible
(ATV truncated image sum ~0.022 e vs MW 2D-Ewald ~0.256 e at matched geometry)

Twin: `tools/metalwalls_oracle/inputs/metalwalls/const-pot-tip4p-edl/mw_twin/`
— MW total_charges: Q = ±2.556468605877299e-1 e.

Regression foothold: neutrality residual < 1e-8; polarity.


## nanoporous-carbon-const-pot

oracle: metalwalls
oracle_version: MetalWalls git **c8ad175d** (product composition gate; empty-slit
soft pin optional when twin archived)
oracle_kind: real binary available; product NVT gates are self-consistency of
the const-pot + freeze + occupancy composition

### Physics

Map §9 nanoporous product: **slit-pore carbon** electrodes (honeycomb sheets)
held at fixed Ψ by `ConstPotBO`, frozen walls, mobile electrolyte inside the
pore. Occupancy proxy = ion count with \(z \in (z_{\mathrm{bot}}+\delta,
z_{\mathrm{top}}-\delta)\).

### Geometry

| | value |
|---|---|
| walls | honeycomb C, 4×7 rect cells → 112 sites/wall |
| planes | z = 8 / 32 Å in Lz = 40 Å (pore gap 24 Å) |
| Ψ | ∓1 V, η = 1.805132 Å⁻¹ (MW graphite) |
| ions | 4 Na⁺ + 4 Cl⁻ in pore mid-band |
| image sum | `xy_image_range = 40` |
| MD | 200 fs NVT, freeze walls, LJ + CoulLong/PME |

### Assertions

| check | bound |
|-------|-------|
| empty polarity / neutrality | Q_L<0<Q_R, \|sum\|≤1e-8 |
| filled polarity / neutrality | same after step 1 and chunks |
| freeze | max \|Δz_wall\| ≤ 1e-10 Å |
| occupancy | ≥ N_ions/2 after short NVT |
| finite thermo | E,K finite |

### Perf

`perf_exempt` — product box composition, not a kernel microbench.


## Reference — per-electrode Gaussian width η_e (ConstPotBO multi-wall)

oracle: metalwalls
oracle_version: MetalWalls git `c8ad175d` (release branch, 2026-06-18), built
with GCC 14.3 + MPICH + OpenBLAS at
`/nobackup/proj/disk/teoroo/personal/jicli594/work/opt/metalwalls/mw`
(upstream self-check `tests/maze/electrodes/LiCl-Al` reproduces its shipped
`energies.ref` Hamiltonian to 1.2e-12 relative)

**Re-pinned 2026-07-29 against the real MetalWalls binary.** Waves 1–5 pinned
this slug from the *literature closed form* because no MW binary existed on
the host. The binary now exists; the pinned numbers were re-derived from MW
directly and are **confirmed** (see the discrepancy budget below). The old
literature citations remain valid as the origin of the closed form:
Reed–Lanning–Madden JCP 126, 084704 (2007) DOI 10.1063/1.2464084;
Scalfi et al. JCP 153, 174704 (2020) DOI 10.1063/5.0028232;
MetalWalls JOSS 5(53), 2373 (2020) DOI 10.21105/joss.02373.

MW twin inputs: `tools/metalwalls_oracle/inputs/metalwalls/per-electrode-eta/`
(`case1_L320/`, `case2_super3x3/`, generators `gen_case1.py` / `gen_case2.py`,
comparison driver `analyze.py`). Full derivation:
`tools/metalwalls_oracle/pin_notes_per-electrode-eta.md`.
**ctest never launches MetalWalls** — the numbers below are hard-coded.

### Physics pinned

MetalWalls-style per-electrode Gaussian charge widths on the
constant-potential BO solve (public surface:
`ConstPotBO::add_electrode(type, Psi, eta)`), with the published elastance
elements (units: eV, Å, e; \(k = 14.399645\) eV·Å/e² = `units::coulomb`):

- \(A_{ii} = k\sqrt{2/\pi}\,\eta_i\)
- \(A_{ij} = k\,\mathrm{erf}(\eta_{ij} r_{ij})/r_{ij}\),
  \(\eta_{ij}=\eta_i\eta_j/\sqrt{\eta_i^2+\eta_j^2}\)

MetalWalls uses **exactly** this convention. Verified in the MW sources:
`src/coulomb_sr_2DPBC.inc:56-58` builds
`etaij = (etai*etaj)/sqrt(etai*etai + etaj*etaj)` from the per-electrode
`electrodes(itype)%eta`, and `src/coulomb_self.f90:63` subtracts
`selffactor = (2α − √2·η_i)/√π` from the Ewald diagonal, whose net effect is
\(A_{ii} = \sqrt{2/\pi}\,\eta_i\) in atomic units. `src/electrode_charge.F90:91`
forms the same RHS `b_i = Ψ_i − V_i^{elyt}` that `ConstPotBO::build_rhs_` does.

### Units: MW (atomic) → ATV (eV, Å, e, V)

| Quantity | ATV value | MW input value (a.u.) | Conversion |
| --- | --- | --- | --- |
| \(\eta_{\mathrm{bot}}\) | 1.979 Å⁻¹ | 1.047241700377037 bohr⁻¹ | ×0.529177210903 Å/bohr |
| \(\eta_{\mathrm{top}}\) | 0.955 Å⁻¹ | 0.505364236412365 bohr⁻¹ | ×0.529177210903 |
| \(d\) (case 1) | 4 Å | 7.558904498503081 bohr | ÷0.529177210903 |
| \(\Delta\Psi\) | 1 V | 3.674932217565499e-02 Ha/e | ÷27.211386245988 V per Ha/e |
| \(Q\) | e | e | MW `charges.out` header: 1 a.u. = 1 e — **no conversion** |
| \(k=e^2/4\pi\varepsilon_0\) | 14.399645 eV·Å/e² | 1 Ha·bohr/e² | 27.211386245988 × 0.529177210903 = **14.399645478456184** |

Dimensional cross-check (not a number match): MW returns a charge, whose atomic
unit *is* the elementary charge, so the only conversion that can hide an error
is the elastance scale \(k\). \(Q=\Delta\Psi/S\) with \(S\propto k\), so
\(Q\propto \Delta\Psi/k\): converting ΔΨ from V to Ha/e (÷27.2114) and \(k\)
from Ha·bohr to eV·Å (×27.2114×0.52918) leaves \(Q\) invariant up to
0.529177210903, i.e. the Bohr radius must appear exactly once — it does.

### Case 1 — two-site vacuum capacitor, dissimilar η

\(\eta_1 = 1.979\) Å⁻¹ (LAMMPS ELECTRODE default, Ahrens-Iwers et al. JCP 157,
084801 (2022); identical to MW `tests/charge_neutrality` `charge gaussian
1.04724169991593`), \(\eta_2 = 0.955\) Å⁻¹ (a deliberately dissimilar second
width — see the provenance correction below), \(d = 4\) Å,
\(\Delta\Psi = 1\) V, electroneutral:

\[
Q = \frac{\Delta\Psi}{A_{11}+A_{22}-2A_{12}} = \mathbf{0.037722100829787}\ e
\]

(\(A_{11}=22.737234510126967\), \(A_{22}=10.972237977347778\),
\(A_{12}=3.599907138217925\), \(\eta_{12}=0.860091344088193\) Å⁻¹).

#### Why the MW twin needs an extrapolation

ATV case 1 is the **image-free** two-site capacitor: `ConstPotBO` with the
default `xy_image_range = 0` sums a single minimum image, i.e. exactly one pair
at \(r=4\) Å. MetalWalls is an Ewald code and is **unconditionally periodic** —
there is no aperiodic mode. The MW twin therefore carries the 2D image lattice,
and the pin is its \(L\to\infty\) limit. For the electroneutral two-site solve
the elastance combination \(S = A_{11}+A_{22}-2A_{12} = \Delta\Psi/Q\) is
background-free, and its periodic value exceeds the isolated one by an
*absolutely convergent* dipole-lattice sum (every erf factor is 1 to <1e-300 at
these separations):

\[
\Delta S(L) = 2k\!\!\sum_{\mathbf n\neq 0}\!\Big[\frac{1}{R_{\mathbf n}}
 - \frac{1}{\sqrt{R_{\mathbf n}^2+d^2}}\Big],\qquad R_{\mathbf n}=L|\mathbf n| .
\]

A second MW artefact must be removed first: `num_pbc 2` approximates the
\(k_z\) integral by a Riemann sum of step \(2\pi/L_z\)
(`src/ewald.f90:203-217`), which is numerically identical to keeping
**z-periodic images at spacing \(L_z\)**. At \(L_z=L_{xy}\) those contribute
≈8 % of the total image term (measured: \(L_z\) 40 Å → 160 Å moves \(Q\) by
−9.2e-5 relative). All production runs below use \(L_z = 20\,L_{xy}\).

#### Raw MetalWalls output — `charges.out` (atomic units = e)

`mpirun -n 1 mw` on `inputs/metalwalls/per-electrode-eta/case1_L320` and its
box-size siblings (\(L_z = 20L_{xy}\)). MW prints \(Q_{\mathrm{bot}}\) first;
\(Q_{\mathrm{top}} = -Q_{\mathrm{bot}}\) exactly.

| \(L_{xy}\) [Å] | MW \(Q_{\mathrm{top}}\) [e] (raw) | \(S_{\rm MW}=\Delta\Psi/Q\) | \(\Delta S(L)\) | \(S_{\rm MW}-\Delta S\) | \(Q_{\rm iso}\) [e] |
| --- | --- | --- | --- | --- | --- |
| 40 | 0.037676075064003 | 26.542042882684 | 3.2384e-02 | 26.509658979107 | 0.037722099736859 |
| 60 | 0.037708419130699 | 26.519276677549 | 9.6176e-03 | 26.509659058459 | 0.037722099623945 |
| 80 | 0.037716322200046 | 26.513719834507 | 4.0608e-03 | 26.509659077777 | 0.037722099596457 |
| 120 | 0.037720386573408 | 26.510862979994 | 1.2039e-03 | 26.509659087696 | 0.037722099582342 |
| 160 | 0.037721376735941 | 26.510167086431 | 5.0800e-04 | 26.509659090111 | 0.037722099578905 |
| 240 | 0.037721885367143 | 26.509809630858 | 1.5054e-04 | 26.509659091352 | 0.037722099577140 |
| **320** | **0.037722009201931** | 26.509722603769 | 6.3512e-05 | 26.509659091653 | **0.037722099576710** |

\(S_{\rm MW}-\Delta S\) is **flat to 3e-11 relative over a factor-8 box sweep**
while \(\Delta S\) itself shrinks by 510×; that flatness is the evidence that
the image model is right and that MW's isolated-limit elastance is
26.5096590917 eV/e².

#### Verdict: confirmed — residual is the truncated Coulomb constant

| Quantity | Value |
| --- | --- |
| Pinned \(Q\) (ATV, \(k=14.399645\)) | 0.037722100829787 e |
| MW \(Q_{\rm iso}\) (\(L=320\) Å, image-corrected) | 0.037722099576710 e |
| Closed form with **exact** \(k=14.399645478456184\) | 0.037722099576397 e |
| MW vs exact-\(k\) closed form | **+8.3e-12** |
| MW vs pin | **−3.322e-08** |
| \((k_{\rm exact}-k_{\rm ATV})/k_{\rm exact}\) | **3.3227e-08** |

\(Q\propto 1/k\), so the −3.322e-08 offset is *entirely* the 8-digit truncation
of `units::coulomb = 14.399645` (`include/atomiverse/core/units.h:140`) against
27.211386245988 × 0.529177210903 = 14.399645478456184 eV·Å/e². Cancel that one
constant and MetalWalls reproduces the pinned physics to **8e-12** — below the
`matrix_inversion 1.0e-14` solver tolerance's own noise floor. The pin is kept
unchanged: it is what ATV's own \(k\) produces, and the regression's 1e-9
tolerance is a code-vs-its-own-closed-form check, not an oracle tolerance.

The same constant explains the offset for both shared-η twins, so the
discrepancy budget is closed with no free parameter (MW at \(L=320\) Å,
\(L_z=6400\) Å, image-corrected):

| \((\eta_1,\eta_2)\) [Å⁻¹] | MW \(Q_{\rm iso}\) [e] | closed form, exact \(k\) | MW vs exact \(k\) | closed form, ATV \(k\) | MW vs ATV \(k\) |
| --- | --- | --- | --- | --- | --- |
| (1.979, 0.955) | 0.037722099576710 | 0.037722099576397 | +8.3e-12 | 0.037722100829787 | −3.322e-08 |
| (1.979, 1.979) | 0.026126954987021 | 0.026126954986871 | +5.8e-12 | 0.026126955854989 | −3.322e-08 |
| (0.955, 0.955) | 0.067816772335944 | 0.067816772334931 | +1.5e-11 | 0.067816774588275 | −3.321e-08 |

The two shared-η rows pin \(A_{11}(\eta)\) and \(A_{22}(\eta)\) **separately**
(there \(A_{11}=A_{22}\)), so the mixed row is over-determined: MW confirms the
per-electrode \(\eta_{ij}\) mixing rule, not just one lumped combination.

| Check | Tol |
| --- | --- |
| \(Q_{\mathrm{top}}\) vs 0.037722100829787 e | rel ≤ 1e-9 |
| \(\lvert\sum Q\rvert\) | ≤ 1e-12 e |

#### Provenance correction (unit-convention slip in the pre-2026-07-29 note)

The superseded REFERENCE.md called \(\eta_2 = 0.955\) Å⁻¹ "the Merlet/Salanne
graphite width used in MW examples". That is **wrong**: MW inputs are atomic
units, and MW's graphite width is `charge gaussian 0.955234657` **bohr⁻¹**
(MW's own comment in `example/nanoporous-graphene/runtime.inpt:55` reads
`# gaussian_width (bohr^-1) charge`), i.e. **1.805131886 Å⁻¹** — the numeral
0.955 was lifted from an atomic-unit input and relabelled Å⁻¹. The pinned
*number* is unaffected (the case is defined by whichever two widths are chosen,
and \(\eta_2=0.955\) Å⁻¹ is a legitimately dissimilar second width, now
verified against MW at exactly that value), but the citation was not evidence
for it. \(\eta_1 = 1.979\) Å⁻¹ *is* genuine: MW's own
`tests/charge_neutrality` uses `charge gaussian 1.04724169991593` bohr⁻¹
= 1.979 Å⁻¹.

### Case 2 — multi-atom two-wall, mixed η, 2D xy Gaussian images

2×2-atom square walls (a = 2 Å, \(L_x=L_y=4\) Å), wall separation 5 Å,
\(\eta_{\mathrm{bot}}=1.979\), \(\eta_{\mathrm{top}}=0.955\) Å⁻¹,
\(\Delta\Psi=1\) V, `xy_image_range = 100`.

MW twin `inputs/metalwalls/per-electrode-eta/case2_super3x3` (same 2 Å site lattice,
same 5 Å gap, `num_pbc 2`, `matrix_inversion 1.0e-14`). Two MW-side pitfalls,
both real, both caught here:

1. **The primitive 4 Å cell is unusable in MW.** MW applies the minimum-image
   convention, so `coulomb_rcut ≤ L/2 = 2` Å, while the electrode–electrode
   short-range term carries \(\mathrm{erfc}(\eta_{ij}r)\) with
   \(\eta_{ij}=0.860\) Å⁻¹ — still 1.5e-2 at \(r=2\) Å. That truncation is
   *silent*: MW runs and prints a clean, converged, wrong answer
   (\(Q_{\rm top}=1.9633237859942\times10^{-2}\) e, 0.87 % low). The fix is a
   physically identical \(n\times n\) **supercell**, which lets `rcut` grow.
2. z-images are harmless here (unlike case 1): dense walls act as uniform
   dipole layers, so `charges.out` is invariant to 1.5e-12 over
   \(L_z = 24\ldots960\) Å.

| MW supercell | sites/wall | `rcut` [Å] | \(Q_{\mathrm{top}}\) per 4×4 Å cell [e] |
| --- | --- | --- | --- |
| 1×1 (4 Å) | 4 | 2.00 | 1.963323785994192e-02 ← **cutoff-limited, do not use** |
| 2×2 (8 Å) | 16 | 3.99 | 1.980427008706078e-02 |
| 3×3 (12 Å) | 36 | 5.99 | **1.980450775193917e-02** |
| 4×4 (16 Å) | 64 | 7.98 | 1.980450776415409e-02 |
| 6×6 (24 Å) | 144 | 11.98 | 1.980450776415245e-02 |

Converged MW value: \(Q_{\mathrm{top}} = 1.980450776415\times10^{-2}\) e,
uniform over sites to 5e-15, exactly electroneutral.

**ATV agrees with it.** `ConstPotBO`'s truncated real-space image sum is
absolutely convergent (square and circular truncation Richardson-extrapolate
to the same limit, 1.98045081e-02 / 1.98045065e-02 e), and that limit matches
MW to **+1.7e-8 / −6.5e-8** — i.e. to the extrapolation's own uncertainty and
the same order as the `units::coulomb` truncation. The periodic-image path is
therefore verified, not merely the isolated one.

The case-2 charge is nevertheless **not** hard-coded, because the regression
runs `xy_image_range = 100` and that sum converges only as \(O(1/R)\):

| `xy_image_range` | \(Q_{\mathrm{top}}\) [e] | vs converged MW |
| --- | --- | --- |
| 100 (what the regression runs) | 1.992946777240788e-02 | +0.631 % |
| 200 | 1.986694868399804e-02 | +0.315 % |
| 800 | 1.982011096692778e-02 | +0.0788 % |
| 3200 | 1.980840858393783e-02 | +0.0197 % |
| →∞ (Richardson) | 1.98045081e-02 | +1.7e-8 |

Pinning \(Q\) at \(R=100\) would pin the truncation error of a numerical knob,
not the physics, so the asserted invariants below stay truncation-independent.
MW independently reproduces the ordering this case asserts —
\(Q(\eta{=}1.979\ \text{shared}) < Q(\text{mixed}) < Q(\eta{=}0.955\
\text{shared})\) — from the shared-η table above.

| Check | Tol |
| --- | --- |
| \(\lvert\sum Q\rvert\) | ≤ 1e-10 e |
| \(Q_{\mathrm{top}} > 0\) (with \(\Psi_{\mathrm{top}}>\Psi_{\mathrm{bot}}\)) | sign |
| \(Q^{\mathrm{shared}\ 1.979} < Q^{\mathrm{mixed}} < Q^{\mathrm{shared}\ 0.955}\) | strict |
| 3-arg `add_electrode` at equal η ≡ 2-arg shared path | ≤ 1e-14 e/site |

### Wall clock (matched twin)

Same box (4×4×24 Å), same 2×2 walls, same η, same ΔΨ, one BO solve, single
rank, `OMP_NUM_THREADS=1`, shared 48-core node:

| | wall clock | work done |
| --- | --- | --- |
| MetalWalls `mpirun -n 1` | 0.32–0.76 s (median 0.44) | 1 BO solve + 2D Ewald setup + file I/O |
| `reg_per-electrode-eta`, single rank | 0.04 s (7/7 runs) | **5** BO solves (case 1 + 4× case 2 at R = 100) |

\(\text{ratio}_{\rm ATV/MW} \approx 0.09\) — and that is conservative: ATV is
charged for five solves against MetalWalls' one. The two are not doing the
same arithmetic (real-space image lattice vs Ewald), and at this size both are
dominated by process start-up, so read the ratio as "same order, ATV not
slower", not as a kernel benchmark.

### Reproducing

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
  cd tools/metalwalls_oracle/inputs/metalwalls/per-electrode-eta/case1_L320
  mpirun -n 1 /nobackup/.../work/opt/metalwalls/mw > run.stdout 2>&1
  sed -n 5,6p charges.out
'
```


## Reference — Gaussian electrode charge kernels (E2)

oracle: literature  
oracle_version: Reed JCP 126 084704 (2007) Gaussian-charge Coulomb; MetalWalls JOSS 5 2373 (2020) / Scalfi constant-potential electrode form  
perf_exempt: true  
perf_exempt_reason: No MetalWalls binary and no LAMMPS ELECTRODE package on agent host; pin is analytic closed-form kernel (not a wall-time twin).

### Physics

Electrode atoms carry normalized 3D Gaussian charges of width \(\eta\) [Å⁻¹]:

\[
\rho_i(\mathbf{r}) = q_i \frac{\eta_i^3}{\pi^{3/2}}
  e^{-\eta_i^2|\mathbf{r}-\mathbf{R}_i|^2},
\quad
\eta_{ij}=\frac{\eta_i\eta_j}{\sqrt{\eta_i^2+\eta_j^2}},
\quad
G_{ij}(r)=\frac{\mathrm{erf}(\eta_{ij} r)}{r},\quad
G_{ii}=\eta_i\sqrt{\frac{2}{\pi}}.
\]

Energy (ATV metal units, `units::coulomb = 14.399645` eV·Å/e²):

\[
U=\texttt{coulomb}\Big[
  \sum_{i<j} q_i q_j G_{ij}(r_{ij})
  +\tfrac12\sum_i q_i^2 G_{ii}\Big].
\]

Gaussian–point: \(\eta_j\to 0\) flag (point) ⇒ \(\eta_{ij}=\eta_i\).  
Point–point: \(G=1/r\), no self.

These are the **same kernels MetalWalls uses** for electrode Gaussians. Offline
pin = direct evaluation of the published formula (no live MW run).

### Point-charge \(A\) is an approximation

Constant-potential BO solves \(A Q = \Psi\) (plus neutrality). Using
**point-charge** \(A_{ij}=\texttt{coulomb}/r_{ij}\) (no self diagonal) is
**not** the MetalWalls electrode model: it drops \(G_{ii}\) and replaces
\(\mathrm{erf}(\eta r)/r\) by \(1/r\). The toy two-site solve below shows
\(Q\) changes by \(O(1)\) (and can flip sign) under that replacement.
Production electrode MD must use the Gaussian kernel for \(A\).

### Geometry — fixed-\(Q\) energy pin

Cubic box \(L=100\) Å (min-image irrelevant at this spacing), cutoff 40 Å.

| Atom | type | \(\eta\) [Å⁻¹] | \(q\) [e] | \((x,y,z)\) [Å] |
| ---- | ---- | -------------- | --------- | --------------- |
| L0 | 1 | 1.0 | +0.25 | (0, 0, 0) |
| L1 | 1 | 1.0 | +0.25 | (2, 0, 0) |
| R0 | 1 | 1.0 | −0.25 | (0, 0, 10) |
| R1 | 1 | 1.0 | −0.25 | (2, 0, 10) |
| ion | 0 | point | +1 | (1, 0, 5) |

**Pin:** \(E = 1.938689655050537\) eV  
**Tol:** \(|E - E_{\mathrm{pin}}| < 10^{-9}\) eV  
**API:** `cpu::Evaluate` + `cpu::PairGaussianCoul{40, {0, 1.0}, /*self=*/true}`

### Geometry — toy \(Q\) vs point-charge \(A\)

Two Gaussian sites at \(z=0\) and \(z=L=10\) Å, \(\eta=1.0\) Å⁻¹,
\(\Psi=(+0.5,-0.5)\) eV/e, neutrality \(Q_1+Q_2=0\):

\[
Q_1=\frac{\Psi_1-\Psi_2}{2(A_{11}-A_{12})},\quad
A_{11}=\texttt{coulomb}\,G_{ii},\quad
A_{12}=\texttt{coulomb}\,G(L).
\]

| Model | \(A_{11}\) | \(A_{12}\) | \(Q_1\) [e] |
| ----- | ---------- | ---------- | ----------- |
| Gaussian (this feature) | \(\texttt{coulomb}\,\eta\sqrt{2/\pi}\) | \(\texttt{coulomb}\,\mathrm{erf}(\eta_{\mathrm{eff}} L)/L\) | **+0.0497547591576** |
| Point-charge \(A\) (approx) | 0 (no self) | \(\texttt{coulomb}/L\) | −0.347230782426 |

**Tol:** \(|Q_1^{\mathrm{gauss}} - Q_{\mathrm{pin}}| < 10^{-9}\) e  
**Requirement:** \(|Q_1^{\mathrm{gauss}} - Q_1^{\mathrm{point}}| > 10^{-6}\) e (documents the approximation).

### Public API (no Config bags)

```cpp
cpu::PairGaussianCoul pot(/*cutoff [Å]=*/40.0,
                          /*eta_by_type=*/{0.0, 1.0}, // type0 point, type1 Gaussian
                          /*include_self=*/true);
cpu::Evaluate eval;
eval.set_evaluator(std::move(pot));
eval.set_neighbor(cpu::md::NeighborList<>{40.0, /*skin [Å]=*/1.0});
eval.build();
eval.run(frame); // E on frame.meta[kEnergy] [eV]
```

`PairGaussianCoul` is a `NeighborConsumer` (it gained a neighbour-listed bulk
path in the wave-4 electrochem drop), so `Evaluate::build()` requires the list
instead of letting the style fall back to its O(N²) sweep unannounced. At
L = 100 Å with r_c = 40 Å every pair of this 5-atom cell is inside the cutoff,
so the list selects which loop runs and never which pairs are summed: the
energy reproduces the analytic pin to |ΔE| = 0.

Math-only: `cpu::GaussianChargeKernel::{eta_eff,greens,self_greens,dgreens_dr}`.

### Build (MD-only)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DATV_BUILD_MD=ON -DATV_BUILD_KSPACE=ON \
  -DATV_BUILD_TESTS=ON -DATV_BUILD_REGRESSIONS=ON \
  -DATV_BUILD_HF=OFF -DATV_BUILD_PW=OFF -DATV_BUILD_GPW=OFF \
  -DATV_BUILD_CUDA=OFF -DBLA_VENDOR=OpenBLAS
```

### Perf

**perf_exempt:** no oracle twin binary for wall timing. Kernel is \(O(N^2)\)
all-pairs for the foothold; neighbor-listed path is future work.


## electrode-atom-forces

oracle: metalwalls
oracle_version: MetalWalls git **c8ad175d** (re-checked 2026-07-29)
oracle_kind: real binary for Q; forces via MW `forces.out` on mobile ion
(not electrode_forces keyword — unsupported)

Twin: `tools/metalwalls_oracle/inputs/metalwalls/electrode-atom-forces/mw_twin/`
— Q_E1 = −5.048583047766454e-1 e, Q_E2 = −4.951416952233546e-1 e;
ion Fz = 1.769232616266544e-2 Ha/a0 ≈ 0.910 eV/Å.

ATV foothold: non-zero electrode Q and electrode-atom forces under
`write_electrode_forces` (truncated image sum; absolute F not tightly MW-matched).


## Reference — electrode charge predictor (ASPC)

oracle: metalwalls
oracle_version: MetalWalls release 20.05, git **c8ad175d** (release branch,
2026-06-18), binary at
`/nobackup/proj/disk/teoroo/personal/jicli594/work/opt/metalwalls/mw`
oracle_kind: real binary + source constants (re-pinned 2026-07-29; supersedes the
wave-2 literature-formula pin)

### Physics

Constant-potential electrode charges \(Q(t)\) evolve smoothly as the electrolyte
moves. MetalWalls warm-starts the iterative (CG) charge solve from a short
history of prior \(Q\) using **Kolafa's Always-Stable Predictor-Corrector**
(J. Comput. Chem. **25**, 335 (2004)). LAMMPS ELECTRODE has no ASPC twin, so
MetalWalls is the only oracle for this feature.

### The pin — MetalWalls' predictor coefficients

MetalWalls hard-codes the predictor in `src/electrode_charge.F90:572-577`
(verbatim from the tree the pinned binary was built from):

```fortran
    real(wp), parameter :: B1 =  22.0_wp /  7.0_wp
    real(wp), parameter :: B2 = -55.0_wp / 14.0_wp
    real(wp), parameter :: B3 =  55.0_wp / 21.0_wp
    real(wp), parameter :: B4 = -22.0_wp / 21.0_wp
    real(wp), parameter :: B5 =   5.0_wp / 21.0_wp
    real(wp), parameter :: B6 =  -1.0_wp / 42.0_wp
...
          q_atoms(i, q_now) = B1 * q_atoms(i, q_m1) + B2 * q_atoms(i, q_m2) + ...
```

Six stored charge vectors (`QATOMS_NUM_STEPS = 7`, `src/system.F90:107`);
the predictor switches on from step 7 (`src/system.F90:1430-1435`,
`src/electrode_charge.F90:499`). There is **no user switch** and no other
order — \(K=5\) is MetalWalls' only choice.

Those six constants are exactly the \(K=5\) row of Kolafa's closed form
(Eq. 10):

\[
B_j = (-1)^{j+1}\, j\, \frac{\binom{2K+2}{K+1-j}}{\binom{2K}{K}},
\qquad j = 1 \ldots K+1
\]

| \(j\) | closed form | MetalWalls | decimal |
| --- | --- | --- | --- |
| 1 | \(\binom{12}{5}/\binom{10}{5} = 792/252\) | `22/7` | 3.142857142857143 |
| 2 | \(-2\binom{12}{4}/252 = -990/252\) | `-55/14` | −3.9285714285714284 |
| 3 | \(+3\binom{12}{3}/252 = 660/252\) | `55/21` | 2.619047619047619 |
| 4 | \(-4\binom{12}{2}/252 = -264/252\) | `-22/21` | −1.0476190476190477 |
| 5 | \(+5\binom{12}{1}/252 = 60/252\) | `5/21` | 0.23809523809523808 |
| 6 | \(-6\binom{12}{0}/252 = -6/252\) | `-1/42` | −0.023809523809523808 |

Invariants: \(\sum_j B_j = 1\) (consistency) and \(\sum_j j B_j = 0\)
(exactness through degree 1). \(\sum_j j^2 B_j \ne 0\) — ASPC is **deliberately
not** exact for quadratics; that damping is what buys unconditional stability.

**No unit conversion is involved in this pin.** The \(B_j\) are dimensionless
linear-combination weights on a charge history; the only quantity in the
identity \(Q^{\rm pred} = \sum_j B_j Q_{n+1-j}\) is \(Q\) itself, in \(e\) on
both sides (MetalWalls' atomic-unit charge *is* \(e\), see the `charges.out`
header "1 au = 1.602176565e-19 C = 1 e"). A conversion slip cannot manufacture
agreement here.

### Verdict — atv_bug, found and fixed

| | old pin (waves 1–5) | real MetalWalls c8ad175d |
| --- | --- | --- |
| order | \(K=2\) (3 history vectors) | \(K=5\) (6 history vectors, fixed) |
| coefficients (newest → oldest) | \(\{3, -3, 1\}\) | \(\{22/7, -55/14, 55/21, -22/21, 5/21, -1/42\}\) |
| same family at \(K=2\)? | — | \(\{5/2, -2, 1/2\}\) |
| exactness | degree ≤ 2 | degree ≤ 1 (damped) |

The old pin \(\{3,-3,1\}\) is **Lagrange (Newton-forward) polynomial
extrapolation**, not ASPC at any order. Relative discrepancy against the same
\(K=2\) row of the real family: \(3/2.5 - 1 = +20\%\) on \(B_1\),
\(-3/-2 - 1 = +50\%\) on \(B_2\), \(1/0.5 - 1 = +100\%\) on \(B_3\) — i.e. the
pin was a different algorithm, not a mis-tolerance.

This was an Atomiverse implementation error, not merely a mislabelled pin:
`src/cpu/electrodes/charge_aspc.cpp` computed Lagrange coefficients while its
`correct()` used the genuine Kolafa corrector weights
\(\frac{K+2}{2K+3}, \frac{K+1}{2K+3}\). Those corrector weights are derived
*assuming* the ASPC predictor; pairing them with Lagrange extrapolation is the
one combination that has neither the accuracy of Lagrange nor the
time-reversibility of ASPC — and plain polynomial extrapolation in a
self-consistent loop is precisely the scheme Kolafa 2004 exists to replace.

**Fix (this branch):** `compute_coefficients_()` now evaluates Kolafa's
closed form with exact integer binomials and a single division, so the \(K=5\)
row is bit-identical to MetalWalls' constants (792/252 and 22/7 round to the
same double). No tolerance was touched; no assertion was weakened.

**Known identical defect, NOT fixed here:** the SCF/CPMD twin
`src/cpu/electronic/aspc_predictor.cpp:24-39` (`ASPCPredictor::compute_coefficients`,
and its `ASPCPredictorCplx` sibling) has the same Lagrange-labelled-as-Kolafa
predictor next to a correct Kolafa corrector. It is owned by the in-flight
`esmm-cpmd-*` work and is not touched from this branch; it needs its own fix.

### Geometry

#### CG vs dense pin (2 electrode sites)

Madelung-like 4-atom primary cell, `xy_image_range = 0`:

| Site | Type | role | \(z\) [Å] |
| --- | --- | --- | --- |
| 0 | 1 | bot electrode | 8 |
| 1 | 2 | top electrode | 12 |
| 2 | 3 | electrolyte + | 9 |
| 3 | 3 | electrolyte − | 11 |

Box \(L_x=L_y=1\), \(L_z=20\) Å. \(\eta=2\) Å⁻¹, \(\Psi=(0,1)\) V.

#### CG iteration comparison (multi-site)

3×3 atoms per wall (18 electrode sites) + 4 electrolyte charges in
\(L_{xy}=6\) Å, \(L_z=20\) Å. Electrolyte \(z\) drifts each launch so
\(b=\Phi_{\mathrm{elec}}\) changes. Primary cell only.

### The MetalWalls twin that was actually run

`tools/metalwalls_oracle/inputs/metalwalls/electrode-charge-predictor/`
(`runtime.inpt` + `data.inpt`), run as

```
module load buildenv-gcc/2026.03-mpich
export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
cd /tmp/mw_oracle_electrode-charge-predictor
mpirun -n 1 .../opt/metalwalls/mw > run.stdout 2>&1
```

Same 18 electrode sites + 4 ions, same \(\eta\), same box, same \(\Psi\), with
every quantity converted into MetalWalls' atomic units:

| ATV (public units) | factor | MetalWalls (a.u.) |
| --- | --- | --- |
| \(L_{xy} = 6\) Å | ÷ 0.529177210903 Å/Bohr | 11.33835674775462 Bohr |
| \(L_z = 20\) Å | ÷ 0.529177210903 | 37.79452249251541 Bohr |
| \(\eta = 2.0\) Å⁻¹ | × 0.529177210903 | 1.058354421806 Bohr⁻¹ |
| \(\Psi_{\rm top} = 1\) V | ÷ 27.211386245988 V/(Ha/e) | 0.036749322175655 Ha/e |
| \(\Delta t = 10\) fs | × 41.341 a.u./fs | 413.41 a.u. |

Dimensional check on \(\eta\): \(\eta\) is an inverse length, so Å⁻¹ → Bohr⁻¹
**multiplies** by 0.529177 (fewer inverse-Bohr than inverse-Å) — the opposite
sense from the box lengths, as it must be. The kernel convention matches:
MetalWalls' same-\(\eta\) pair interaction is \({\rm erf}(\eta r/\sqrt2)/r\)
(self factor \(\eta/\sqrt{2\pi}\), `src/coulomb_self.f90:63`) and Atomiverse
uses `erf((eta/√2) r)/r` (`src/cpu/dynamics/const_pot_bo.cpp:368`).

Raw MetalWalls output at step 0 (`charges.out`, atomic units = \(e\)):

```
## step        0
-3.716936940768154E-002
-3.862179629716499E-002
...
Q(E1, bottom, Psi=0 V) = -3.566190297414107E-01 e
Q(E2, top,    Psi=1 V) = +3.566190297414106E-01 e
sum                    = -6.9e-18 e
```

`run.out` step 0: `|step| Global charge on electrode 1: -3.56619E-01`.

**These absolute charges are deliberately not pinned by this regression**, and
they must not be compared numerically with the ATV multi-site case: the ATV
regression runs `xy_image_range = 0` — a **non-periodic primary-cell direct
sum** — whereas MetalWalls always applies full 2D-PBC Ewald (`num_pbc 2`), which
it has no option to disable. Those are different electrostatic models, so a
charge mismatch here would say nothing about the predictor. Absolute
const-potential charges are pinned by the sibling slugs
(`const-pot-bo-parallel-plate`, `gaussian-electrode-charge`).

### What Atomiverse asserts

Public API:

- `cpu::electrodes::ChargeASPC` (vector ASPC)
- `cpu::md::ConstPotBO` with `use_cg`, `predictor_order`, `last_iters()`

| Check | Tol / gate |
| --- | --- |
| ASPC \(K=5\) coeffs == MetalWalls `B1..B6` | rel ≤ \(10^{-15}\) each; \(\sum B_j = 1\) to \(10^{-14}\) |
| ASPC \(K=2\) coeffs \(\{1/2,-2,5/2\}\) | exact |
| linear history \(Q_k=k\) → \(Q_{n+1}=n+1\) | ≤ \(10^{-12}\) (degree-1 exactness) |
| quadratic history \(Q_k=k^2\) → 8, **not** the Lagrange 9 | ≤ \(10^{-12}\) (ASPC-vs-Lagrange discriminator) |
| CG \(Q\) vs dense Cholesky on fixed RHS | abs ≤ \(10^{-8}\) e |
| Mean CG iters with `predictor_order=2` **and** `=5` **<** cold (`=0`) after history fill | strict inequality (measured on this branch: cold 6.000, \(K=2\) 5.000, \(K=5\) 5.000 over 10 post-fill steps) |
| \(\lvert\sum Q_i\rvert\) | ≤ \(10^{-8}\) e |

Measured regression output after the re-pin:

```
ASPC K=5 == MetalWalls B1..B6 {22/7,-55/14,55/21,-22/21,5/21,-1/42}, sum=1.0000000000000002
ASPC K=2: omega={0.5,-2,2.5}, linear exact, quadratic damped to 8
CG vs dense: Q_top dense=2.71753556e-01 cg=2.71753556e-01 Δ=0.00e+00
Multi-site: n_atoms=22 n_electrode=18
Mean CG iters (post-fill n=10): cold=6.000 ASPC K=2 5.000  ASPC K=5 (MetalWalls) 5.000
PASS
```

Sibling regressions that consume `predictor_order` were re-run against the
fixed coefficients and are unchanged: `reg_const-pot-il-edl-box` reproduces its
pinned charges to its own 1e-6 tolerance, and `reg_maze-symplectic-charge-dynamics`
still shows the loose-CG ASPC twin an order of magnitude less time-reversible
than MaZe (`ret_pos` 7.5e-07 Å vs 1.4e-14 Å).

### What MetalWalls could NOT confirm

The "warm start reduces CG iteration count" claim is **not** A/B-testable
against this binary, for two specific reasons:

1. MetalWalls has no switch to disable the predictor — it turns on
   unconditionally at step 7 (`src/system.F90:1430`), and there is no runtime
   keyword for it. The only available contrast is steps 1–6 vs steps 7+ of the
   same trajectory, which are different configurations.
2. MetalWalls' non-predictor branch is not a cold start: it initialises from the
   previous step's charges (`src/electrode_charge.F90:503-508`). Atomiverse's
   `predictor_order = 0` starts CG from the **zero** vector
   (`src/cpu/dynamics/const_pot_bo.cpp:670`), so "cold" means different things
   in the two codes.

Measured anyway on the twin: at `cg 1.0e-10` MetalWalls needs **4** iterations
at every one of the 30 steps; tightening to `cg 1.0e-14` gives 5–6 iterations
with no visible step-7 discontinuity. With only 18 unknowns the CG count
saturates and the predictor's benefit is below resolution in either code. The
Atomiverse inequality assertion therefore remains an internal-consistency check
of its own warm-start path, not an oracle-confirmed number — and it is labelled
as such here rather than dressed up as a MetalWalls result.

### Perf

```
perf_exempt: true
perf_exempt_reason: >
  A real MetalWalls binary now exists and was run (30 steps, 0.9265 s wall,
  0.0309 s/step, mpirun -n 1), but the two codes do not solve the same linear
  system for this case. MetalWalls is matrix-free: every CG iteration re-runs a
  full 2D-PBC Ewald apply_A over 18 unknowns plus LJ and Verlet integration.
  The ATV regression builds the dense 18x18 elastance ONCE
  (rebuild_matrix_every_step = false) over a non-periodic primary-cell sum, with
  no Ewald, no pair potential and no integrator. A wall-clock ratio would
  measure the electrostatics model and the matrix-free-vs-dense choice, not the
  charge predictor this slug pins. Pair-plate charge timing belongs to
  const-pot-bo-parallel-plate, where both codes run the same 2D-Ewald model.
ratio_atv_over_oracle: n/a
within_10x: exempt
```

### Repro of pin

1. Coefficients: `sed -n '556,596p' src/electrode_charge.F90` in the MetalWalls
   tree at c8ad175d, or evaluate the closed form
   `B_j = (-1)^(j+1) j C(2K+2,K+1-j)/C(2K,K)` at `K=5`.
   The binary exposes no diagnostic that prints the guess (CG errors out at
   `max_iterations = 0` rather than reporting the predictor output,
   `src/cg.f90:457-463`), so the pin is a source-level read of the exact tree
   the pinned binary was built from — not a black-box fit.
2. Twin run: the inputs under
   `tools/metalwalls_oracle/inputs/metalwalls/electrode-charge-predictor/`
   reproduce `Q(E2) = +3.566190297414106E-01 e` at step 0.


## Reference — freeze electrode / wall atoms (LAMMPS `fix setforce 0`)

oracle: lammps
oracle_version: develop / patch_30Mar2026-1074-g7f680de296 (30 Mar 2026 Development; binary `lmp` at pin time)

### Physical setup

Immobile electrode (wall) atoms under classical LJ NVE with a mobile fluid.
This is the MetalWalls E1 wall-immobility foothold and the LAMMPS twin of
`fix setforce 0.0 0.0 0.0` on a wall group plus `velocity wall set 0 0 0`.

| Quantity | Value |
|---|---|
| Box | cubic \(L = 10\) Å, PBC |
| Wall (type 2) | 4 atoms at \((2,2,0)\), \((2,6,0)\), \((6,2,0)\), \((6,6,0)\) Å; mass 196.9665 amu |
| Fluid (type 1) | 4 atoms at \(z=3.5\) above each wall site; mass 39.948 amu |
| Pair | LJ 12-6, \(\varepsilon = 0.010341\) eV, \(\sigma = 3.4\) Å, \(r_c = 8.0\) Å (all type pairs) |
| Ensemble | NVE velocity-Verlet, \(dt = 1\) fs |
| Steps (correctness pin) | 1000 |
| Wall initial \(v\) | \(0\) |
| Fluid initial \(T\) | Maxwell–Boltzmann 200 K (seed 99; RNG not bit-matched to LAMMPS) |

Offline input: `tools/metalwalls_oracle/inputs/lammps/freeze-electrode-atoms/in.lammps`.

### What is asserted (not a self-golden)

From LAMMPS `fix setforce 0.0 0.0 0.0` on group `wall` (`fix_setforce.cpp`):
after forces are computed, wall forces are zeroed so the NVE half-kicks do not
change wall velocities. With \(v_{\mathrm{wall}}(0)=0\), wall positions remain
exactly at their create positions for the entire trajectory.

**Offline LAMMPS pin (1000 steps, metal units, `timestep 0.001` ps = 1 fs):**

| Observable | Pin value |
|---|---|
| Wall positions after 1000 steps | identical to create: \((2,2,0)\), \((2,6,0)\), \((6,2,0)\), \((6,6,0)\) Å |
| Wall velocities | all components exactly `0` |
| Wall \(z\) span | \(z \in [0, 0]\) throughout (thermo `c_min_dz`/`c_max_dz`) |

Dump excerpt (`wall_final.lammpstrj` after 1000 steps):

```
1 2 2 2 0 0 0 0
2 2 2 6 0 0 0 0
3 2 6 2 0 0 0 0
4 2 6 6 0 0 0 0
```

Atomiverse regression asserts the same **immobility invariants** (not fluid
coordinates — fluid RNG and unit-velocity conventions differ):

1. \(\max_i |x_i(t) - x_i(0)| \le 10^{-10}\) Å on the wall type.
2. \(\max |v|\) components on wall ≤ \(10^{-12}\) Å/fs.
3. At least one fluid atom displaces by more than \(10^{-3}\) Å (freeze is selective).

### Units note

LAMMPS `units metal` time is **ps**; Atomiverse public time is **fs**.
`timestep 0.001` (ps) ⇔ `dt = 1.0` fs. Velocities in metal dumps are Å/ps;
the freeze pin is zero in either convention.

### Perf gate (fair wall)

Same geometry, **20000** NVE steps, single OpenMP thread (`OMP_NUM_THREADS=1`),
no dump/thermo thrash:

| Engine | Wall time | Notes |
|---|---|---|
| LAMMPS `lmp` (1 rank, OMP=1) | Loop time **0.0186 s** | `in.bench` |
| Atomiverse `FixFreeze` + VV + LJ | **0.0740 s** (instrumented in reg binary) | same N, steps, dt |

\[
\frac{T_{\mathrm{atv}}}{T_{\mathrm{oracle}}} = \frac{0.0740}{0.0186} \approx \mathbf{3.98}
\]

Gate: ≤ 10.0 → **PASS** (`within_10x=true`). Tiny 8-atom system; ratio is
not a production throughput claim — only a fair twin ceiling for this slug.


### API

```cpp
cpu::md::FixFreeze freeze(/*type_id=*/2);  // freeze electrode type
sim.add(std::move(freeze));
// with VelocityVerlet + PairLJCut on cpu::md::MD
```


## global-electroneutrality-bo

oracle: metalwalls
oracle_version: MetalWalls release 20.05, git **c8ad175d** (release branch,
2026-06-18), binary at
`/nobackup/proj/disk/teoroo/personal/jicli594/work/opt/metalwalls/mw`
oracle_kind: real binary (re-pinned 2026-07-29; supersedes the wave-4
self-consistency / literature foothold)

### Physics

Under global electroneutrality the constant-potential BO solve constrains the
sum of all electrode charges to cancel the electrolyte charge:

\[
\sum_e Q_e = -q_{\mathrm{electrolyte}}.
\]

MetalWalls implements this as `charge_neutrality true global`
(`src/electrode_charge.F90`). ATV exposes the same constraint as
`ConstPotBO::electroneutrality = true` + `::global_electroneutrality = true`.

### Geometry (matched twin)

| | ATV | MetalWalls |
|---|---|---|
| box | 10×10×20 Å | same, converted to Bohr |
| E1 | type 1, z = 5 Å, Ψ = 0 V, η = 2 Å⁻¹ | species E1, gaussian charge |
| E2 | type 2, z = 15 Å, Ψ = 1 V, η = 2 Å⁻¹ | species E2 |
| ion | type 3, z = 10 Å, q = +1 e | species Ion, point charge |
| solver | dense elastance, `xy_image_range = 1280` | `num_pbc 2`, matrix_inversion, Ewald |

Twin inputs:
`tools/metalwalls_oracle/inputs/metalwalls/global-electroneutrality-bo/`
(`data.inpt`, `runtime.inpt`, generated by `gen_toy.py`).

### Pin (numbers MetalWalls printed)

Command (offline only — never from ctest):

```bash
module load buildenv-gcc/2026.03-mpich
export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
MW=…/work/opt/metalwalls/mw
cd <twin-dir> && mpirun -n 1 "$MW"
```

From `total_charges.out` step 0 (also archived under `mw_output/`):

| quantity | value [e] |
| -------- | --------- |
| \(Q_{\mathrm{E1}}\) (Ψ = 0) | **−5.189485789245821×10⁻¹** |
| \(Q_{\mathrm{E2}}\) (Ψ = 1 V) | **−4.810514210754179×10⁻¹** |
| \(\sum Q_e\) | **−1.0** (= −q_ion) |

### Assertions

| check | tolerance |
| ----- | --------- |
| `neutrality_target()` = −1 e | 1e-12 |
| `sum_e Q_e` = −1 e (primary MW pin) | 1e-8 |
| \(Q_0\), \(Q_1\) vs MW per-electrode | 2e-5 |

The per-electrode residual is the difference between ATV's truncated xy-image
elastance and MetalWalls' 2D-periodic Ewald elastance. Raising
`xy_image_range` from 50 → 1280 drops the residual from ~5×10⁻⁵ to ~6×10⁻⁶;
the 2e-5 gate leaves headroom without re-implementing Ewald inside this
regression. Neutrality (sum) is exact on both sides independent of the
image truncation.

### Perf

`perf_exempt` — single 2×2 elastance solve; wall ratio vs MW would measure
Ewald setup, not the electroneutrality constraint.

### Superseded

Wave-4 pin claimed `oracle: literature` / `metalwalls-map-wave4-manual` and
only checked `sumQ ≈ target` with no number from an external binary.
EOF


## Reference — piston electrode constant \(P_z\) (E6)

```text
oracle: metalwalls
oracle_version: MetalWalls git c8ad175d (release branch, 2026-06-18),
  binary /nobackup/proj/disk/teoroo/personal/jicli594/work/opt/metalwalls/mw,
  run as `mpirun -n 1 mw`. Upstream self-check: tests/maze/electrodes/LiCl-Al
  reproduces its shipped energies.ref Hamiltonian to 1.2e-12 relative.
oracle_input: tools/metalwalls_oracle/inputs/metalwalls/piston-electrode-pz/
  (runtime.inpt + data.inpt, regenerated by gen_inputs.py; MW's own output
   kept verbatim as electrode_forces.ref)
```

**Status: re-pinned 2026-07-29.** Waves 1–5 pinned this case from the
MetalWalls *paper* (Marin-Laflèche et al., JOSS 5, 2373 (2020),
DOI 10.21105/joss.02373) because no binary existed. That pin was purely
self-consistent — \(P_{\mathrm{target}}\) was whatever ATV itself measured —
and contained no external number at all. It has now been checked against the
real binary, one fixture defect was found and fixed, and an absolute MW
number is hard-coded.

### Map slice

**E6 — electrode mechanics:** piston electrode + constant normal pressure
\(P_z\). Rigid piston ≠ generic isotropic barostat. Orthogonal to E1 charge
solve (pair forces only in this foothold).

### What MetalWalls actually does (source-verified, c8ad175d)

Read out of the binary's own tree, not out of the paper:

| MW source | Statement |
| --- | --- |
| `src/system.F90:2466` `piston_positions_update` | \(v \mathrel{+}= \tfrac{1}{2}\Delta t\,M^{-1}(F_z^{\text{ions}} + P_{\text{sig}} A)\); then every electrode atom gets \(z_i \mathrel{+}= \Delta t\,v\) — one scalar velocity per electrode, \(z\) only |
| `src/system.F90:2615` `piston_velocities_update` | the matching second half-kick with the current-step force ⇒ **velocity Verlet on a rigid body** |
| `src/configuration.F90:512` | \(M = \text{count}\times\text{mass}\) — the **total** electrode mass |
| `src/system.F90:2485` | \(A = \texttt{box\%area(3)} = L_x L_y\) |
| `src/electrode_parameters.f90:150` `set_pressure` | `piston <P> <dir>` ⇒ \(P_{\text{sig}} = \text{dir}\times P\); upstream `tests/piston/LiCl-Al` uses `+1` on the lower plate, `-1` on the upper |
| `src/output.F90:1376` | the `pressure` column of `electrode_forces.out` is literally `force_ions(3,1) / area` |
| `src/system.F90:2186` | `force_ions(:,1)` = sum of terms 2…10 = total force on the electrode |
| `src/system.F90:3938` `energy_piston` | \(H_{\text{piston}} = -\sum_e P_{\text{sig}} A\,(z_{\text{com}} - z_{\text{com}}^0)\) |

Mapped onto `cpu::md::PistonElectrode`:

| Aspect | MetalWalls c8ad175d | `PistonElectrode` | Verdict |
| --- | --- | --- | --- |
| Rigid body | one scalar \(v_z\), \(z_i \mathrel{+}= \Delta t\,v\) | \(F_{z,i}=F_{\mathrm{net}} m_i/M\), then common \(v_z\) + rigid offsets | **equivalent** (equal accelerations ⇒ rigid translation) |
| Mass | \(\sum_i m_i\) over the electrode | \(\sum_i m_i\) over the type | **aligned** |
| Area | \(L_x L_y\) | \(L_x L_y\) | **aligned** |
| External force | \(+P_{\text{sig}} A\), sign from the input `dir` | \(-PA\) (Upper) / \(+PA\) (Lower) | **aligned** — MW's upper plate carries `dir = -1` |
| Free axes | \(z\) only (electrodes immobile in \(xy\)) | \(F_x=F_y=0\), \(xy\) offsets restored | **aligned** |
| Instantaneous \(P_z\) | `force_ions(3,1)/area`, signed, both plates | Upper \(+F_z/A\); Lower \(-F_z/A\) | **divergence (display only)** — ATV flips the lower plate's sign so both plates read positive under compression. Identical mechanics. |
| COM friction | none | optional `tau_damp` (default off) | ATV extension, off by default |
| Piston work in \(H\) | `Piston_work` energy term | not reported | gap, not a disagreement |

Force law, unchanged from waves 1–5 and now confirmed against the binary:

\[
F_z^{\mathrm{atoms}}=\sum_{i\in t} F_{z,i},\quad
A=L_x L_y,\quad
F_{\mathrm{ext}}=\begin{cases}
-P_{\mathrm{target}} A & \text{Upper}\\
+P_{\mathrm{target}} A & \text{Lower}
\end{cases}
\]

\[
F_{\mathrm{net}}=F_z^{\mathrm{atoms}}+F_{\mathrm{ext}}(+F_{\mathrm{damp}}),\quad
F_{z,i}\leftarrow F_{\mathrm{net}}\,m_i/M,\quad F_{x,i}=F_{y,i}=0.
\]

### The pinned number

MetalWalls twin of this exact 24-atom system (generator + inputs under
`tools/metalwalls_oracle/inputs/metalwalls/piston-electrode-pz/`), step 0,
electrode 2 = `Wup` = the piston. Raw `electrode_forces.out` line, atomic
units, columns as written by `src/output.F90:1358` — 1–3 total,
4–6 coulomb\_sr, 7–9 coulomb\_k0, 10–12 coulomb\_lr,
**13–15 lennard\_jones**, 16–18 fumi\_tosi, 19 pressure, 20 velocity\_z,
21 com\_z. (The file's *header* line lists 24 labels including `xft`; only
21 numbers are written — an upstream cosmetic bug, not a column shift.)

```text
## step        0
+2.275923962312989E-015 +2.416412523274872E-020 +1.596167239253399E+000 \
+0.000000000000000E+000 +0.000000000000000E+000 +0.000000000000000E+000 \
+0.000000000000000E+000 +0.000000000000000E+000 -1.335101285915558E-002 \
-3.323816858197838E-020 +2.416412523274872E-020 -2.965250959688260E-005 \
+2.275957200481571E-015 +0.000000000000000E+000 +1.609547904622151E+000 \
+0.000000000000000E+000 +0.000000000000000E+000 +0.000000000000000E+000 \
+4.469723501474648E-003 +0.000000000000000E+000 +1.511780900366263E+001
```

Read: `fz_lennard_jones = +1.609547904622151 Ha/Bohr`;
`com_z = 15.11780900366263 Bohr` \(= 8.000\) Å, i.e. the upper plate ✓.

Explicit conversion (MW works in Hartree/Bohr; ATV's public surface is eV/Å):

```text
F_z = 1.609547904622151 Ha/Bohr
    × 27.211386245988 eV/Ha        (ATV core/units.h)
    ÷ 0.52917721067    Å/Bohr      (MW src/constants.F90 bohr2angstrom)
    = 1.609547904622151 × 51.42206749896734
    = 82.7662809943017 eV/Å

A   = L_x L_y = 10 Å × 10 Å = 100 Å²

P_z = F_z / A = 8.27662809943017e-01 eV/Å³
```

Dimensional cross-check done the other way round, so a bad factor cannot
hide: MW's own \(P_z\) in atomic units is
\(F_z/A_{\mathrm{Bohr}} = 1.609547904622151/(100/0.52917721067^2)
= 4.507193180712053\times10^{-3}\) Ha/Bohr³, and
\(4.507193180712053\times10^{-3}\times 27.211386245988 / 0.52917721067^3
= 8.27662809943017\times10^{-1}\) eV/Å³ — the same answer via
energy/volume instead of force/area.

Independent third opinion: a from-scratch minimum-image LJ sum over the same
coordinates (`gen_inputs.py::analytic_fz_upper`) gives
\(F_z = 82.76628232744\) eV/Å, **1.6e-8 relative** from MW — exactly the
CODATA drift between MW's \(1\,E_h = 27.21138602\) eV and ATV's
\(27.211386245988\) eV. (The factor `51.42208619083232` that circulates in
some worker prompts is internally inconsistent with the constants quoted
beside it: \(27.211386245988/0.529177210903 = 51.422067476\). The
self-consistent value is used here.)

| Quantity | Old pin (waves 1–5) | New pin (MW c8ad175d) |
| --- | --- | --- |
| probe \(P_z\) | *(none — the target was whatever ATV itself measured)* | \(8.27662809943017\times10^{-1}\) eV/Å³, rel. tol \(10^{-6}\) |
| \(F_z\) on the piston | *(none)* | \(82.7662809943017\) eV/Å \(=1.609547904622151\) Ha/Bohr |

Observed ATV–MW agreement is at the \(10^{-8}\) level (the unit round-trip);
the case prints it on its `MW pin:` line.

### Fixture defect found and fixed: \(r_c > L/2\)

The wave-1–5 fixture ran LJ at \(r_c = 6\) Å in a 10 Å periodic box. ATV's
single-rank `CellList` folds every separation through minimum image
(`include/atomiverse/cpu/neighbor/cell_list.h`, `min_image_unextended_`), so
a cutoff above half the shortest edge silently drops the second image of
every pair with \(|d| = L/2\) — the fixture was computing a
min-image-truncated LJ, not "LJ with \(r_c = 6\) Å". Quantified: the true
lattice sum at \(r_c = 6\) gives \(F_z = 82.630417\) eV/Å against ATV's
min-image \(82.660592\) eV/Å, a **3.7e-4** systematic error.

MetalWalls refuses such an input outright (`src/system.F90`: *"Invalid value
for lj rcut > min(a, b, c)/2"*), so the old fixture was not expressible in
MW — **for a reason that indicts the fixture, not MW**.

Fix: \(r_c = 4.5\) Å, skin \(0.3\) Å (\(r_c+\text{skin}=4.8 < 5\)). At
\(r_c \le 5\) Å the min-image sum and the full lattice sum agree to machine
precision for this geometry, so the pinned number is unambiguous. The
physics barely moves — \(P_z\) goes from \(0.826606\) to \(0.827663\) eV/Å³
(0.13 %), because the piston force is dominated by the 2.53 Å fluid–wall
contacts.

Carried forward, not fixed here: `cpu::md::NeighborList` accepts
\(r_c > L/2\) without a word. That is a live footgun for every single-rank
user, but it is library-wide behaviour (LAMMPS-like, "user's
responsibility"), not specific to this case.

### Geometry (regression)

| Species | type | role |
| --- | ---: | --- |
| Ar-like fluid | 1 | mobile LJ fluid (16 atoms, two layers) |
| lower wall | 2 | frozen (`FixFreeze`) at \(z=2\) Å |
| upper wall | 3 | **piston** (`PistonElectrode::Upper`) at \(z=8\) Å |

- Box: orthorhombic \(L_x=L_y=10\) Å, \(L_z=20\) Å → \(A=100\) Å²
- Walls: \(2\times2\) atoms each, at \(x,y\in\{2.5,7.5\}\)
- Fluid: \(x=(i\%4+0.5)\cdot 2.5\), \(y=((i/4)\%2+0.5)\cdot 5\),
  \(z = 4.2\) for \(i<8\) else \(5.8\)
- Pair: LJ \(\varepsilon=0.10\) eV, \(\sigma=3.0\) Å, \(r_c=4.5\) Å, skin 0.3 Å
- NVT fluid: VelocityVerlet + NoseHoover \(T=300\) K, \(\tau_T=50\) fs, \(dt=0.5\) fs
- Piston: \(P_{\mathrm{target}}=\) the MW-pinned probe \(P_z\); \(\tau_{\mathrm{damp}}=20\) fs
- Warm 400 + prod 600 steps (chunk-sampled \(P_z\))

### How to reproduce the MW side

```bash
python3 tools/metalwalls_oracle/inputs/metalwalls/piston-electrode-pz/gen_inputs.py
mkdir -p /tmp/mw && cp <that dir>/{runtime.inpt,data.inpt} /tmp/mw
bash -lc 'module load buildenv-gcc/2026.03-mpich
          export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
          cd /tmp/mw && mpirun -n 1 <opt>/metalwalls/mw > run.stdout 2>&1'
## electrode_forces.out, "# step 0", SECOND data line (electrode Wup),
## column 15 = fz_lennard_jones
```

Two MW-side accommodations, both provably neutral for the pinned number:

1. **Applied potential \(\Delta\psi = 0.5\) a.u.** (\(\psi_{\mathrm{lo}} =
   -0.25\), \(\psi_{\mathrm{up}} = +0.25\)). The ATV system has no charges
   at all, but MW's piston is only available on a constant-potential
   electrode, and MW's CG computes `alpha = dp/dAd` (`src/cg.f90`), which is
   \(0/0 =\) NaN when the right-hand side vanishes: a chargeless electrode
   makes MW abort with *"CG failed to converge"*. (Seeding a non-zero charge
   guess in `data.inpt` does not help.) A non-zero \(\Delta\psi\) makes the
   solve well posed; the LJ force is a function of geometry alone, so column
   15 is untouched. This is also why the `pressure` column — which is
   `fz_total/area` and therefore carries the induced-charge Coulomb terms
   \(-1.335\times10^{-2}\) and \(-2.965\times10^{-5}\) Ha/Bohr — is **not**
   the number to read here.
2. **Electrode–electrode LJ is absent in MW**
   (`src/lennard_jones_2DPBC.inc` computes ion–ion and ion–electrode only).
   Irrelevant here: the upper plate's four atoms are coplanar (\(dz=0\), no
   \(z\) force) and the plate–plate separation is 6 Å \(> r_c\). Verified
   numerically — including or excluding those pairs changes \(F_z\) by
   exactly 0.

### Observables and tolerances

| Check | Bound | Rationale |
| --- | --- | --- |
| Probe \(P_z\) vs MetalWalls | **rel ≤ \(10^{-6}\)** | oracle pin, c8ad175d; observed \(\approx 10^{-8}\) (unit round-trip only) |
| Probe \(P_z\) | \(>10^{-8}\) eV/Å³ and \(>0\) | Compressed fluid pushes upper wall |
| \(\lvert\langle P_z\rangle-P_{\mathrm{target}}\rvert/P_{\mathrm{target}}\) | ≤ 0.10 | MW constant-\(P_z\) equilibration (10% textbook gate) |
| Upper wall coplanar | \(\Delta z\le 10^{-9}\) Å | Rigid piston |
| Upper \(\lvert v_x\rvert,\lvert v_y\rvert\) | ≤ \(10^{-12}\) Å/fs | \(z\)-only motion |
| Lower wall \(\Delta z\) | ≤ \(10^{-10}\) Å | `FixFreeze` |
| Energy finite | true | Smoke stability |

Unit tests pin the static identities \(P_z=F_z^{\mathrm{atoms}}/A\) and
\(F_{\mathrm{net}}=0\) when \(P_{\mathrm{target}}=P_z^{\mathrm{atoms}}\) to
machine precision (no MD statistics).

### Perf

```text
perf_exempt: true
perf_exempt_reason: no configuration exists in which MetalWalls and
  Atomiverse do the same work per step for this case.
  (1) MW's rigid piston is only available on a constant-potential
      electrode, so every MW step additionally runs a CG electrode-charge
      solve plus a 2D Ewald sum. The ATV regression is pure LJ with no
      charges, and MW cannot express that at all (its CG divides by zero,
      see above), so MW is doing strictly more physics per step.
  (2) MW cannot even propagate this fixture's deliberately compressed
      start (closest fluid-fluid contact 1.6 Å, r/sigma = 0.53): NVE hits
      "Temperature is NaN or meaningless" at step ~12
      (src/system.F90 print_step) and the Nose-Hoover chain aborts with
      "Velocity thermostating failed to converge" (src/thermostat.f90).
      For the record, and NOT to be quoted as a per-step cost: MW managed
      12 steps in 2.55 s wall before aborting.
```

The compressed start is a fixture-quality observation worth carrying
forward: the two fluid layers sit 1.6 Å apart at identical \(xy\), i.e.
\(r/\sigma = 0.53\). The gates still hold (the Nose-Hoover fluid plus the
damped, mass-heavy piston keep \(\langle P_z\rangle\) at its static value),
and the *pinned* quantity is a step-0 static force that does not depend on
the trajectory at all — but a future revision should stagger the layers so
the MD half is not riding on an overlap no other engine will integrate.

### Units

Public surface: eV, Å, fs, amu, eV/Å, eV/Å³ (pressure). 1 bar ≈ 6.2415×10⁻⁷ eV/Å³.
MetalWalls surface: Hartree, Bohr, e, and a timestep in \(\hbar/E_h\)
(1 fs = 41.341373336493 a.u.; the twin uses 20.6706866682465 a.u. = 0.5 fs).
LJ parameters enter MW's `lj_pair` as kJ/mol and Å; the twin emits
\(\varepsilon = 9.648533052529773\) kJ/mol, which is exactly 0.10 eV under
MW's own `kJpermol2hartree`.

### API

```cpp
cpu::md::PistonElectrode piston(/*type=*/3, /*P [eV/Å³]=*/P,
                                cpu::md::PistonElectrode::Side::Upper);
piston.tau_damp = 20.0; // optional [fs]
sim.add(std::move(piston)); // or add_ref for diagnostics
sim.add(cpu::md::FixFreeze{/*lower type=*/2});
```

No public `*Config` bag; no `make_*` factory.


## Reference — coupled PIM dipoles ⊕ electrode \(Q\) (E5)

oracle: metalwalls
oracle_version: MetalWalls git **c8ad175d** (release branch, 2026-06-18), binary
`/nobackup/proj/disk/teoroo/personal/jicli594/work/opt/metalwalls/mw`, run
`mpirun -n 1`. Twin inputs:
`tools/metalwalls_oracle/inputs/metalwalls/dipole-electrode-joint/{runtime,data}.inpt`
mw_expressible: **false** — see *Why MetalWalls cannot express this system*
perf_exempt: true
perf_exempt_reason: Measured, and the measurement is meaningless. MW twin
(`num_pbc 3`, `mpirun -n 1`): 0.52 / 0.73 / 0.67 s. Whole ATV regression
(`OMP_NUM_THREADS=1`, four joint solves + one MD step + thermo/timing report):
0.04 / 0.04 / 0.04 s. Both are ~100 % fixed overhead for a 4-atom system — MW's
is MPI startup plus Ewald table setup (a 16×16×33 k-grid and a 67-point
z-integral table built for 4 atoms), ATV's is process startup; the joint solve
itself is a 2×2 Cholesky costing microseconds. The two are also not solving the
same linear system (MW: preconditioned CG over
\(3N_{\text{ion}}+N_{\text{elec}}\) Ewald-coupled unknowns; ATV: direct
min-image 2×2). The resulting ~0.07 ratio measures startup, not the solver, and
must not be quoted as a speed result. A fair number needs an electrode-slab twin
(≥10³ atoms) that ATV cannot yet express (no Ewald in this component) — see
*Debt* below.

### Physics

Self-consistent joint of:

1. **Constant-potential BO** electrode charges \(Q\) with Gaussian elastance \(A\)
   (same kernels as `cpu::md::ConstPotBO`).
2. **PIM induced dipoles** \(\boldsymbol{\mu}\) on polarizable electrolyte sites
   with Thole damping (same kernels as `cpu::PimInducedDipole`).

Coupling:

\[
A Q = \Psi - \Phi^{\mathrm{perm}} - \Phi^{\mu},\quad
\Phi_i^{\mu} = k_e\sum_j\frac{\boldsymbol{\mu}_j\cdot\mathbf{r}_{ij}}{r_{ij}^3}
\]

\[
\boldsymbol{\mu}_i = \alpha_i\mathbf{E}_i^{\mathrm{tot}}(q_{\mathrm{all}},\boldsymbol{\mu})
\]

Outer alternating solve until \(\max|\Delta Q|\) and \(\max|\Delta\mu|\) below
`joint_tol`. **Oracle is MetalWalls** — LAMMPS has no SCF PIM ⊕ const-pot
joint twin (ELECTRODE is non-polarizable; Drude is a different model).

### Geometry

Orthorhombic \(L_x=L_y=10\) Å, \(L_z=20\) Å; \(\eta=2\) Å⁻¹; \(\Psi_{\mathrm{bot}}=0\),
\(\Psi_{\mathrm{top}}=1\) V; electroneutrality on; `xy_image_range=0` (min-image);
`thole_a=0.39`.

| Site | type | \(q\) [e] | \(\alpha\) [Å³] | \((x,y,z)\) [Å] |
| ---- | ---- | --------- | --------------- | --------------- |
| bot graphene wall | 1 | solved | 0 | (0, 0, 2) |
| top graphene wall | 2 | solved | 0 | (0, 0, 18) |
| IL cation | 3 | +1 | 1.0 | (0, 0, 8) |
| IL anion | 4 | −1 | 1.5 | (0, 0, 12) |

#### Pin values (hard-coded in regression)

| Observable | Value |
| ---------- | ----- |
| \(Q_{\mathrm{bot}}\) [e] | **−7.3285008958883610e−02** |
| \(Q_{\mathrm{top}}\) [e] | **+7.3285008958883610e−02** |
| \(\mu_{\mathrm{cation},z}\) [e·Å] | **+5.8951916148990660e−02** |
| \(\mu_{\mathrm{anion},z}\) [e·Å] | **+8.7432701391015569e−02** |
| \(\mu_{*,x}\), \(\mu_{*,y}\) | 0 |
| \(\lvert\sum Q\rvert\) | 0 (within \(10^{-12}\)) |
| \(\alpha=0\) \(Q_{\mathrm{top}}\) [e] | **+7.5339627935767609e−02** |

**Tol (dual gate):**
relative \(|Q-Q_{\mathrm{pin}}|/|Q_{\mathrm{pin}}| \le 10^{-6}\);
\(|\mu_z-\mu_{\mathrm{pin}}| \le 10^{-8}\) e·Å; transverse \(|\mu|<10^{-10}\);
neutrality \(|\sum Q|\le 10^{-10}\) e.

These are **ATV's own minimum-image algebra**, not MetalWalls output. See the
re-pin section below for exactly what that means and what MetalWalls says.

**API:** `cpu::md::DipoleElectrodeJoint{2.0, {0,0,0,1.0,1.5}}` +
`add_electrode(1,0).add_electrode(2,1)`.

---

### Real-MetalWalls re-pin (2026-07-29)

Waves 1–5 pinned this case from published MetalWalls *formulas* because no MW
binary was installed. A verified binary now exists, so the MW twin was actually
built and run. This section records what came back. **Verdict: not_expressible**
— the pinned numbers are kept, the tolerances are kept, and the provenance claim
is corrected.

#### The MW twin

`tools/metalwalls_oracle/inputs/metalwalls/dipole-electrode-joint/` reproduces the
regression's system exactly — same 4 sites, same \(z\), same permanent charges,
same masses, same \(\eta\), same box, same \(\Psi\) — converted Å/eV/V → Bohr/Ha/(Ha·e⁻¹)
with **MetalWalls' own** `bohr2angstrom = 0.52917721067` (`src/constants.F90`):

| quantity | regression (ATV units) | MW input (atomic units) |
| --- | --- | --- |
| \(L_x, L_y\) | 10 Å | 18.897261254578282 Bohr |
| \(L_z\) | 20 Å | 37.794522509156565 Bohr |
| \(z\) bot elec | 2 Å | 3.7794522509156563 Bohr |
| \(z\) cation | 8 Å | 15.117809003662625 Bohr |
| \(z\) anion | 12 Å | 22.676713505493936 Bohr |
| \(z\) top elec | 18 Å | 34.015070258240904 Bohr |
| \(\eta\) | 2.0 Å⁻¹ | 1.05835442134 Bohr⁻¹ |
| \(\alpha_{\mathrm{cation}}\) | 1.0 Å³ | 6.748334503514373 Bohr³ |
| \(\alpha_{\mathrm{anion}}\) | 1.5 Å³ | 10.12250175527156 Bohr³ |
| \(\Psi_{\mathrm{top}}\) | 1 V | 0.03674932217565499 Ha/e (= 1/27.211386245988) |
| \(\Psi_{\mathrm{bot}}\) | 0 V | 0.0 |

MW block: `dipoles_and_electrodes / algorithm cg 1.0e-13 5000 / preconditioner
jacobi / charge_neutrality true`, `coulomb_rtol 1e-12`, `coulomb_rcut 9.4` Bohr
(the \(L_{\min}/2 = 9.4486\) Bohr ceiling), `coulomb_ktol 1e-12`.

Two MW-side gotchas, both handled in the shipped input and worth remembering:

- A `damping` block is **mandatory** whenever dipoles are on. `system%tt` is
  allocated only by `read_tt_parameters` (`src/configuration.F90`), yet
  `coulomb_sr_pim` dereferences `tt%order/b/c` unconditionally → SIGSEGV in
  `qmelt2mumelt_electricfield` without it. A `tt_pair … c = 0` entry allocates
  it with every damping term identically zero.
- With `temperature 0` the `data.inpt` must carry a `# velocities` section for
  **all** atoms (electrode rows included, and they must be exactly 0), otherwise
  `print_step` aborts on `Temperature is NaN or meaningless`.

#### Raw MetalWalls output

`charges.out` / `dipoles.out`, atomic units, `num_pbc 3`:

```
## Electrode atom charges in atomic unit
## step        0
-6.707261457445883E-002
+6.707261457445883E-002

## Dipoles on mobile particles in atomic unit
##          mux                     muy                     muz
## step        0
+6.745304354600813E-018 -3.094316960449654E-018 +1.326729243171588E-001
+2.064507384972561E-018 -4.018619627801056E-018 +1.957837940515414E-001
```

and with the dipoles switched off (`electrode_charges cg`, the \(\alpha=0\) branch):

```
-6.913534940031944E-002
+6.913534940031944E-002
```

For reference, the same twin under `num_pbc 2` (slab Ewald) gives
\(Q_{\mathrm{top}} = +1.302179493406125\times10^{-1}\) (joint) and
\(+1.362412229597930\times10^{-1}\) (\(\alpha=0\)). `num_pbc 3` is the closer
analogue of ATV's min-image, which wraps all three axes.

**Unit conversion.** Charge is already in \(e\) (MW: "1 au = 1 e"), so no
conversion. Dipole is \(e\cdot a_0\) → \(e\cdot\)Å by \(\times 0.52917721067\):

- \(\mu_{\mathrm{cat},z}\): \(0.1326729243171588 \times 0.52917721067 = 7.0207488022\times10^{-2}\) e·Å
- \(\mu_{\mathrm{an},z}\): \(0.1957837940515414 \times 0.52917721067 = 1.0360432203\times10^{-1}\) e·Å

Dimensional check (not number-matching): MW's \(\mu = \alpha E\) is in atomic
units where \(k_e \equiv 1\); ATV's \(\mu[e\,\text{Å}] = \alpha[\text{Å}^3]\,q[e]/r^2[\text{Å}^2]\)
is the same \(k_e\)-absorbed convention with polarizability as a volume. So the
dipole conversion is length-only (\(a_0\to\)Å) and \(\alpha\) converts as a volume
(\(a_0^3\to\)Å³), which is what the twin table does.

#### Pin vs MetalWalls

| Observable | ATV pin | MW `num_pbc 3` | (pin − MW)/\|MW\| |
| --- | --- | --- | --- |
| \(Q_{\mathrm{top}}\) joint | +7.3285008958883610e−02 | +6.7072614574458830e−02 | **+9.26 %** |
| \(Q_{\mathrm{bot}}\) joint | −7.3285008958883610e−02 | −6.7072614574458830e−02 | **−9.26 %** |
| \(Q_{\mathrm{top}}\), \(\alpha=0\) | +7.5339627935767609e−02 | +6.9135349400319443e−02 | **+8.97 %** |
| \(\mu_{\mathrm{cat},z}\) | +5.8951916148990660e−02 | +7.0207488022e−02 | **−16.03 %** |
| \(\mu_{\mathrm{an},z}\) | +8.7432701391015569e−02 | +1.0360432203e−01 | **−15.61 %** |

Nine percent, against a \(10^{-6}\) gate. The pin is **not** confirmed by MW.

#### Where the gap comes from — measured, not asserted

`tools/metalwalls_oracle/inputs/metalwalls/dipole-electrode-joint/ewald_check.py`
re-implements **the ATV kernels verbatim** — Gaussian self term
\(k_e\eta\sqrt{2/\pi}\), Gaussian–Gaussian \(k_e\operatorname{erf}(\eta r/\sqrt2)/r\),
Gaussian–point \(k_e q\operatorname{erf}(\eta r)/r\), neutrality projection,
\(AQ=\Psi-b\) — with one switch: the real-space sum is either **minimum image**
(what ATV does) or a **full 3D Ewald lattice sum**. The \(\alpha=0\) branch has no
dipoles, so it isolates the electrode algebra:

| summation convention | \(Q_{\mathrm{top}}\) [e] |
| --- | --- |
| ATV kernels, minimum image (**= the pin**) | 7.5339627935767609e−02 |
| ATV kernels, 3D Ewald | 6.9135350235340731e−02 |
| **MetalWalls c8ad175d, `num_pbc 3`** | **6.9135349400319443e−02** |

ATV's own formulas under a matched lattice sum reproduce MetalWalls to
**1.21 × 10⁻⁸ relative**, stable to 1e-15 across five independent Ewald
splittings (\(\alpha_{\mathrm{Ew}} = 0.25\ldots0.45\) Å⁻¹, real cutoff 10–14 cells,
k-cutoff 18–30). That residual is not numerical noise — it is
`ATV::units::coulomb = 14.399645`, which is 3.28 × 10⁻⁸ below the exact
\(1\,\mathrm{Ha}\cdot a_0 = 14.399645472\) eV·Å; propagating that constant through
\(Q=[\Delta\Psi-\Delta b]/[2(A_{11}-A_{12})]\) predicts +1.1 × 10⁻⁸, which is what
is observed.

**Conclusion: the ATV kernels are MetalWalls' kernels, exactly.** The 9 % is the
minimum-image truncation of a conditionally convergent lattice sum, which is a
*deliberate, documented* property of `DipoleElectrodeJoint`
(`xy_image_range = 0`), not a coding error. The larger μ gap (16 %) is the same
truncation inside the PIM SCC field — `scc_mu_` always uses `box.min_image`
regardless of `xy_image_range` — plus the damping-model difference below.

#### Why MetalWalls cannot express this system (`mw_expressible: false`)

Two specific, checkable reasons — neither is "it was hard":

1. **MetalWalls has no minimum-image / non-Ewald electrostatics.** Every Coulomb
   entry point dispatches `select case (num_PBC)` with exactly `case(2)` and
   `case(3)` and no default (`src/coulomb.F90:136`,
   `src/coulomb_sr.f90:47,78,108,144,173,204,235`). `num_pbc 0` or `1` silently
   returns zero potential. The regression's system is defined by a *different
   boundary condition* — min-image real-space Coulomb — which is not a parameter
   MW exposes, not a tolerance that can be loosened, and not something MW can be
   configured into.
2. **MetalWalls has no Thole damping.** `grep -ri thole src/` returns nothing.
   MW's only induced-dipole damping is Tang–Toennies on the *charge–dipole* term
   (`src/damping.f90`, `tt_pair b order c`), and there is **no dipole–dipole
   damping at all**. The regression pins `thole_a = 0.39` applied to charge–dipole
   *and* dipole–dipole (λ₃/λ₅). The MW twin was therefore run undamped (`c = 0`),
   i.e. the ATV `thole_a = 0` limit — a different model from the one the pin
   describes. For scale, at matched min-image summation ATV gives
   \(Q_{\mathrm{top}} = 7.3285\times10^{-2}\) (Thole 0.39) vs
   \(7.3162\times10^{-2}\) (undamped): damping is a 0.17 % effect on \(Q\) but a
   ~6 % effect on \(\mu\).

The pinned values are therefore **kept** and the tolerances are **kept**. They
remain what they always were: ATV's own min-image algebra — now *independently
validated in form* against a real MetalWalls binary. What changed is that this
file no longer claims the numbers came out of MetalWalls.

#### Debt — named, not buried

`cpu::md::DipoleElectrodeJoint` computes a **min-image** electrode elastance and
a **min-image** PIM field. For a real electrode that is not a converged model:
the electrode lattice sum is conditionally convergent, and min-image truncation
costs ~9 % on \(Q\) and ~16 % on \(\mu\) for this 10×10×20 Å cell.
`xy_image_range > 0` only adds a bare real-space \(xy\) image sum to \(A\) and
\(b\); it adds no \(k\)-space term and does not touch `scc_mu_` at all, so it does
not converge to the Ewald answer either. Closing E5 against MetalWalls for real
requires wiring this component to the existing `longrange/` Ewald/PME machinery.
Until then this regression is an **algebra pin**, not an electrode-physics pin,
and the numbers above are the measured size of the gap.

Second, smaller smell found while building the twin: the cation–top-electrode and
anion–bottom-electrode separations are **exactly \(L_z/2 = 10\) Å**, i.e. on the
minimum-image tie. Magnitudes are unambiguous, so \(A\) and the permanent-charge
\(b\) are safe, but the sign of \(\Delta z\) in the \(\boldsymbol\mu\)-to-electrode
term \(k_e(\boldsymbol\mu\cdot\mathbf r)/r^3\) is decided purely by the rounding
mode. `cpu::Box::min_image` uses `std::nearbyint` (round-half-to-**even** in the
default FP environment) and `pin_joint.py` uses Python `round` (also
round-half-to-even), so the two agree today — but under round-half-away-from-zero
those two contributions would flip sign. A future geometry tweak should move a
site off the \(L/2\) tie rather than rely on that coincidence.

### Provenance

- MetalWalls c8ad175d — twin inputs + raw output, this file (2026-07-29).
- Coretti et al., *J. Chem. Phys.* (2022) — polarizable electrolyte + metallic
  electrodes (MetalWalls `dipoles_and_electrodes`).
- Marin-Laflèche et al., JOSS **5**, 2373 (2020).
- Thole, *Chem. Phys.* **59**, 341 (1981).
- Pin generator (ATV algebra, min-image):
  `tools/metalwalls_oracle/inputs/metalwalls/dipole-electrode-joint/pin_joint.py`.
- Boundary-condition decomposition:
  `tools/metalwalls_oracle/inputs/metalwalls/dipole-electrode-joint/ewald_check.py`.
- Map: MetalWalls capability **E5** (`dipole-electrode-joint`).
