# Pin notes — maze-symplectic-charge-dynamics

- Wave: 5 (metalwalls-method-loop), 2026-07-27 — literature pin
- **Re-pinned 2026-07-29 against the real MetalWalls binary
  (git `c8ad175d`, release branch 2026-06-18).** Verdict: **confirmed**.
- Oracle priority: **metalwalls** (MaZe constant-potential charge dynamics is
  MetalWalls-only physics — LAMMPS ELECTRODE has conp/conq matrix/CG solvers
  but no mass-zero constrained SHAKE charge dynamics).

## What changed at re-pin

The wave-5 note said "no MetalWalls binary is installed → literature pin,
`perf_exempt`". The binary now exists. Every invariant this regression asserts
was re-checked by running MetalWalls on twins of the same system, and one new
pin — a real MetalWalls number — was **added**. Nothing was loosened.

Inputs: `inputs/metalwalls/maze-symplectic-charge-dynamics/` (`gen_inputs.py`
generates every case; `atv_twin.cpp` is the matching out-of-library Atomiverse
consumer). Decisive raw MW outputs are kept under `.../mw_output/`.

## The modelling gap MetalWalls exposed

Atomiverse evaluates this cell in **minimum image** — `PairGaussianCoul` is a
hard-cutoff pair style and `ConstPotBO::xy_image_range = 0`. MetalWalls has no
minimum-image Coulomb mode at all: `num_pbc` is 2 or 3 and the electrostatics
is Ewald in every path. On the regression's own 6 × 6 × 20 Å cell:

| Q_top | [e] |
| --- | ---: |
| ATV, minimum image | +0.1358218636749 |
| MW c8ad175d, 3D Ewald (`twinA_ewald_6x6x20`) | +0.1529601174306 |
| relative | −11.20 % |

MW's value is converged to 1.5e-13 in its own Ewald parameters (r_cut
2.0/2.5/2.9 Å, rtol/ktol 1e-12 and 1e-14 all agree), so the 11.2 % is the
truncation of the ATV *model*, not a solver defect. It was deliberately **not**
substituted into the regression: doing so would silently replace the model the
regression exists to test.

## The kernel-matched check (this is the real pin)

Grow the cell and MW's Ewald converges to the isolated pairwise
Gaussian-Coulomb sum — precisely what the minimum-image path computes once the
cell exceeds twice the cluster extent. ATV's answer there is box-independent
(bit-identical at L = 40/60/200 Å):

| MW L [Å] | MW Q_top [e] | rel Δ vs ATV | MW E_grand [eV] | rel Δ |
| ---: | ---: | ---: | ---: | ---: |
| 40 | +0.1267873919608 | +6.301e-4 | −0.9840388431722 | +9.328e-4 |
| 60 | +0.1267203095694 | +1.006e-4 | −0.9833929872312 | +2.759e-4 |
| 100 | +0.1267091472474 | +1.255e-5 | −0.9831803436266 | +5.959e-5 |
| 200 | **+0.1267076937357** | **+1.076e-6** | −0.9831291093381 | +7.473e-6 |
| ATV | **+0.1267075573679** | — | −0.9831217623464 | — |

Monotone ~1/L³ convergence onto Atomiverse's value: the elastance matrix, the
electrolyte RHS, the neutrality projection and the MaZe solve are all correct.
`kMwQTopDilute = 0.1267076937357 e` with a 1e-4 relative tolerance is now
asserted in the regression (MW's own residual image error at L = 200 Å is
1.1e-6, so the tolerance is honest headroom, not slack).

## Invariants re-checked against the binary

| Pin | ATV measured | MetalWalls c8ad175d |
| --- | ---: | ---: |
| ‖σ‖∞ ≤ 1e-10 V | 9.990e-11 | 1.04083e-17 a.u. = **2.83e-16 V** |
| \|ΣQ\| ≤ 1e-9 e | 5.551e-17 | exact (`charge_neutrality true electrodes`) |
| palindrome Δr ≤ 1e-6 Å | 1.421e-14 | **2.288e-7** (restart-text round-trip floor) |
| palindrome ΔQ ≤ 1e-6 e | 9.015e-13 | **5.946e-10** |
| grand-energy drift ≤ 1e-6 eV, 800 steps | 7.571e-08 | **6.758e-7** (slope 5.07e-10 eV/step) |
| Q_top > 0 at ΔΨ = +1 V | +0.135822 | +0.152960 |

MetalWalls also confirms MaZe ≡ Born–Oppenheimer directly: `twinB_dilute_60`
(`maze inversion`) and `twinB_dilute_60_inv` (`matrix_inversion`) agree to
1e-17 e per site, and `|maze| number of iterations: 0` — the linear constraint
is met by one exact correction, which is the Coretti 2020 §II.C statement the
wave-5 pin took on faith.

MW's palindrome is done by running `twinA_pal_fwd` for 400 steps with
`restart 400`, negating the `# velocities` block of `restart.000000400.out`,
and rerunning it as `twinA_pal_back`. MW's 2.29e-7 Å is an **upper bound**:
the restart file is 15-significant-digit text, seeding ~1e-14 Bohr that this
locally unstable ion pair amplifies over 400 steps. ATV's in-memory flip has no
such floor.

## MetalWalls input gotchas found here

1. `electrode_charges maze <alg> <tol>` aborts unless `data.inpt` carries at
   least two `# charges` sections (its Verlet history). Zeros work — the
   constraint is linear.
2. MW's random velocity initializer divides by the instantaneous temperature,
   so a zero-temperature start must supply an explicit all-zero
   `# velocities` block or MW aborts "Temperature is NaN or meaningless".
3. In this small cell, r_cut ≤ 2.5 Å pushes α high enough that MaZe cannot
   reach a 1e-14 tolerance ("M-SHAKE failed to satisfy constraints"). Use
   `matrix_inversion` for parameter sweeps, or keep r_cut 2.9 Å.

## Performance

MW: 7.6–9.0 s for 800 MaZe NVE steps of twin A (`mpirun -n 1`, output every
800 steps). ATV: ≈ 0.015 s for its 800-step MaZe leg. Raw ratio ≈ 0.002 —
**not** an engine benchmark: the 6 Å lateral cell caps MW's real-space cutoff
below 3 Å, forcing an O(10⁵)-mode k-mesh, while ATV sums 45 minimum-image
pairs and no reciprocal space. `perf_exempt = true`, reason: different
electrostatic method per step, which MetalWalls cannot be configured out of.

## Primary citations (still the theory of record)

Coretti, Scalfi, Bacon, Rotenberg, Vuilleumier, Salanne, Ciccotti, Bonella,
*Mass-zero constrained molecular dynamics for electrode charges in simulations
of electrochemical systems*, **J. Chem. Phys. 152, 194701 (2020)**,
DOI 10.1063/5.0007192. Marin-Laflèche et al., **JOSS 5(53), 2373 (2020)**,
DOI 10.21105/joss.02373.
