# Polarizable and fluctuating-charge models

> The regression runner has been removed. Regression commands below are historical; case sources and oracle records are retained as archived evidence, not current executable checks.

Classical polarization and charge-response models for electrolytes, molten
salts and electrode cells. Each is a Component on `cpu::md::MD` (or
`cpu::Evaluate` for a single point) — never a new Driver category and never a
charge solve inside `KSpace*`. Parameters are constructor arguments and public
members; there is no `*Config` bag and no factory. Frame is the only I/O token.

- `cpu::PimInducedDipole{cutoff, alpha_by_type, thole_a, pol_tol, pol_max_iter}`
  — `Potential`: Polarizable Ion Model (PIM) self-consistent induced dipoles,
  polarization energy, Hellmann–Feynman forces and virial (into `kStress`).
- `cpu::PimXlDipole{cutoff, alpha_by_type, thole_a, xl_order, pol_tol,
  pol_max_iter}` — `Potential`: the same PIM with an extended-Lagrangian
  (iEL/SCF) warm start for the SCC.
- `cpu::md::QEq(cutoff)` + `set_type(type, chi, eta)` — charge equilibration,
  default `pre_force` order −50 (LAMMPS `fix qeq/point`).
- `cpu::md::ConstPotMaZe(eta)` — mass-zero constrained (MaZe) electrode-charge
  dynamics, default `pre_force` order −50.

All sections run on CPU only. The case also registers the section
`maze-molten-salt-al` (molten NaCl between Al walls with MaZe charges,
`REQUIRES KSPACE MD`), which has no section on this page.

Case: `ctest -R '^md/polarizable$'` — sources and oracle record in [`regressions/md/polarizable/`](../../../regressions/md/polarizable/). One section of the case per `##` heading below: `atv_regressions --case md/polarizable --section <name>`.

## Common setup

- **Build:** the sections below require the `MD` module
  (`extended-lagrangian-dipoles` also `KSPACE`); CPU login shell with
  `buildenv-gcc/2026.03-mpich`, as in the top-level build instructions.
- **Oracle policy.** LAMMPS is the binding oracle when it implements the same
  algorithm (`qeq-fluctuating-charge`, QEQ package). Bulk SCF PIM and MaZe
  charge dynamics are MetalWalls-only physics; LAMMPS's polarizable paths
  (Drude oscillators, permanent multipoles, QEq) are a different algorithm.
  MetalWalls sections pin against the MetalWalls binary, **git c8ad175d**
  (release branch, `mpirun -n 1 mw`), where MetalWalls can express the model,
  and against closed-form literature evaluations where it cannot. Offline pins
  only — never live MetalWalls or LAMMPS in ctest.
- **Unit bridge** (MetalWalls is atomic units, Atomiverse metal units):
  1 Bohr = 0.529177210903 Å; 1 Ha = 27.211386245988 eV;
  1 Ha/a₀ = 51.42208619083 eV/Å; 1 Eh/e = 27.211386245988 V;
  1 fs = 41.341373335182 a.u. (CODATA-2018, the constants MetalWalls prints in
  its own headers).
- **`units::coulomb` floor.** `core/units.h` has
  \(k_e = 14.399645\) eV·Å/e², an 8-digit truncation of the exact
  14.399645478456 eV·Å/e²; every Atomiverse energy and force against an
  atomic-unit MetalWalls number therefore carries a fixed −3.3227e-08 relative
  offset (dipoles do not — the SCC runs in Gaussian units).

### PIM physics (shared by the three PIM sections)

Applequist isotropic induced-dipole SCC. Each atom type carries a
polarizability \(\alpha\) [Å³]; for sites \(i\):

\[
\boldsymbol{\mu}_i=\alpha_i\mathbf{E}_i^{\mathrm{tot}},\quad
\mathbf{E}_i^{\mathrm{tot}}
  =\mathbf{E}_i^{\mathrm{perm}}
  +\sum_{j\neq i}\mathbf{T}_{ij}\cdot\boldsymbol{\mu}_j
\]

\[
E_{\mathrm{pol}}
  =-\tfrac12\sum_i\boldsymbol{\mu}_i\cdot\mathbf{E}_i^{\mathrm{perm}}
  \quad\text{[eV via `units::coulomb`]}
\]

The permanent field \(\mathbf{E}^{\mathrm{perm}}\) is the Coulomb field of Frame
point charges (min-image, real-space cutoff). Mutual induction uses the dipole
interaction tensor with optional **Thole** exponential screening
(\(u = a\,r/(\alpha_i\alpha_j)^{1/6}\), factors \(\lambda_3,\lambda_5,\lambda_7\);
Thole 1981, van Duijnen–Swart form). Optional **Tang–Toennies** damping
\(f_n(br)\) multiplies the same kernels when `tt_b > 0`. Forces are
Hellmann–Feynman at converged SCC (charge–dipole + dipole–dipole). Permanent
charge–charge Coulomb is **not** included — pair it with `PairCoulCut` / PME;
double-counting is a user composition error. The SCC throws on any unconverged
step. `E_pol` is MetalWalls' own bulk-PIM energy: MetalWalls minimises
\(U = U_{qq} + U_{q\mu} + U_{\mu\mu} + \sum|\mu|^2/2\alpha\), which at the
stationary point equals \(U_{qq} - \tfrac12\sum\mu\cdot E^{\mathrm{perm}}\), so
\(E_{\mathrm{pol}}^{\mathrm{MW}}\) = Coulomb(polarizable run) − Coulomb(same
geometry, \(\alpha = 0\)).

What MetalWalls cannot express: it has **no Thole damping** (`grep -ri thole`
over the c8ad175d tree returns zero hits; its only multipole screening is
Tang–Toennies on the charge→dipole field with a per-pair intensity, a different
form from `tt_b`), and it is always Ewald-periodic (no truncated-Coulomb mode).
MetalWalls-comparable configurations therefore use `thole_a = 0`, `tt_b = 0`
and an isolated-limit extrapolation.

**Shared PIM limitations.** Bulk only (no electrode charge solve or joint
dipole⊕electrode coupling, E5 `dipole-electrode-joint`); all-pairs \(O(N^2)\)
SCC (no NeighborConsumer path); isotropic \(\alpha\) by type (no Drude
oscillators, no permanent multipoles beyond Frame charges); real-space cutoff
only — there is no long-range path for the induced-dipole field (pairing with
`CoulLong`/PME extends only the permanent charge–charge term).

### References

- A. Marin-Laflèche et al., *J. Open Source Softw.* **5**(53), 2373 (2020) (MetalWalls), DOI [10.21105/joss.02373](https://doi.org/10.21105/joss.02373)
- B. T. Thole, *Chem. Phys.* **59**, 341 (1981); van Duijnen & Swart, *J. Phys. Chem. A* **102**, 2399 (1998) (exponential damping)
- J. Applequist, J. R. Carl, K.-K. Fung, *J. Am. Chem. Soc.* **94**, 2952 (1972) (isotropic damped point-dipole SCC)
- Capability map: `.claude/notes/metalwalls-capability-map.md` (E4, §1, §2, §9)

## pim-induced-dipole-bulk — Bulk PIM induced dipoles, two-site ion pair {#pim-induced-dipole-bulk}

### Summary

The base PIM pin: a two-site charged ion pair evaluated by
`cpu::PimInducedDipole` on `cpu::Evaluate` (`add_ref` at
`evaluate::Phase::evaluate`). Gate A compares the undamped model
(`thole_a = 0`) against the real MetalWalls binary extrapolated to the isolated
limit; Gate B pins the Thole-damped model (`thole_a = 0.39`) against the
closed-form Applequist/Thole solution (a literature pin, not a MetalWalls
number). Fair wall vs MetalWalls on a 512-ion melt: ratio 0.77.

### System

| Item | Setting |
| ---- | ------- |
| Geometry | 2 sites: A at origin, B at \((3,0,0)\) Å |
| Charges | \(q_A=+1\,e\), \(q_B=-1\,e\) |
| Types / \(\alpha\) | type 1 → \(\alpha=1.0\) Å³; type 2 → \(\alpha=1.5\) Å³ (`alpha_by_type={0,1.0,1.5}`) |
| Box | cubic \(L=50\) Å (min-image ≡ free space at this spacing) |
| Cutoff | 20 Å |
| Gate A | `thole_a = 0`, `tt_b = 0`, `pol_tol = 1e-14` e·Å, `pol_max_iter = 500` |
| Gate B | Thole \(a = 0.39\), Tang–Toennies \(b = 0\) (off), `pol_tol = 10^{-12}` e·Å, `pol_max_iter = 200`; converges in 8 iterations |

MetalWalls twin (atomic units): r = 5.669178373877311 a₀, L =
94.486306231288509 a₀, α(A) = 6.748334494600375 a₀³, α(B) =
10.122501741900562 a₀³, `coulomb_rcut 47.0` a₀ (Ewald, rtol = ktol = 1e-15),
`dipoles_minimization cg 1.0e-14 500`. MetalWalls' `damping` block is mandatory
whenever `polarizability > 0` (omitting it segfaults), so the twin declares
`tt_pair … c = 0.0`, which is exactly undamped. MetalWalls was run over
L = 50, 100, 200, 400, 800, 1600, 3200 Å and Richardson-extrapolated in
\(1/L^3\) (the neutralising-background term \(-\tfrac{2\pi}{3V}r^2\) is the only
difference from the isolated pair); the ladder is stationary to 8e-13 over the
last three doublings.

### Usage

```cpp
cpu::Frame frame(2);
frame.span(kAtom_x)[0] = 0.0;
frame.span(kAtom_y)[0] = 0.0;
frame.span(kAtom_z)[0] = 0.0;
frame.span(kAtom_x)[1] = 3.0;
frame.span(kAtom_y)[1] = 0.0;
frame.span(kAtom_z)[1] = 0.0;
frame.span(kAtom_charge)[0] = 1.0;
frame.span(kAtom_charge)[1] = -1.0;
frame.span(kAtom_type)[0] = 1;
frame.span(kAtom_type)[1] = 2;
frame.span(kAtom_mass)[0] = frame.span(kAtom_mass)[1] = 1.0;
frame.span(kAtom_tag)[0] = 1;
frame.span(kAtom_tag)[1] = 2;
frame.box = cpu::Box::cubic(50.0);
frame.meta.get(kEnergy) = 0;

cpu::PimInducedDipole pim(/*cutoff [Å]=*/20.0,
                          /*alpha_by_type=*/{0.0, 1.0, 1.5},
                          /*thole_a=*/0.39,
                          /*pol_tol [e·Å]=*/1e-12,
                          /*pol_max_iter=*/200);
// optional: pim.tt_b = 0.0;  // Tang–Toennies off by default

cpu::Evaluate eval;
eval.on(evaluate::Phase::evaluate).add_ref(pim);
eval.build();
eval.run(frame);

const double E = frame.meta.get(kEnergy);           // [eV]
const double e_pol = pim.last_pol_energy();         // [eV]
const int iters = pim.last_pol_iters();
const bool ok = pim.last_pol_converged();
// induced dipoles: pim.last_mu() packed xyz [e·Å]; forces: kAtom_fx/fy/fz
```

Files: `pim_induced_dipole.h`, `src/cpu/potentials/pim_induced_dipole.cpp`
(SCC, Thole/TT, \(E_{\mathrm{pol}}\), HF forces); unit tests in
`tests/cpu/potentials/test_pim_induced_dipole.cpp` (undamped two-site SCC
against the closed-form Applequist solution, FD of \(E_{\mathrm{pol}}\) against
analytic \(F\) within \(5\times 10^{-6}\) eV/Å, Thole).

### Validation

**Gate A — MetalWalls c8ad175d, `thole_a = 0`, isolated limit.** The section
converts Atomiverse output to atomic units and compares there, so the pinned
digits are MetalWalls' own output.

| Observable | MetalWalls pin (a.u.) | eV / Å units | Tol | Measured |
| ---------- | --------------------- | ------------ | --- | -------- |
| μ_A | **+2.3523561717334e-01** e·a₀ | +1.2448132780408e-01 e·Å | 1e-9 rel | 2.6e-14 |
| μ_B | **+3.4109164490134e-01** e·a₀ | +1.8049792531618e-01 e·Å | 1e-9 rel | 2.7e-14 |
| E_pol | **−8.9660039191731e-03** Ha | −2.4397739572766e-01 eV | 1e-7 rel | 3.32e-08 |
| F_A,x = −F_B,x | **+6.7922032485569e-03** Ha/a₀ | +3.4926913376021e-01 eV/Å | 1e-7 rel | 3.32e-08 |

The dipoles match to machine precision (14 SCC iterations); the energy and
force sit exactly on the `units::coulomb` floor (Common setup). Rescaling
MetalWalls by `14.399645 / 14.399645478456` reproduces the Atomiverse energy to
6e-14 and the force to 2e-14. The 1e-7 tolerance is one decade above that
floor.

**Gate B — Thole/Applequist closed form, `thole_a = 0.39` (literature pin).**
Two-site Applequist SCC with van Duijnen/Swart exponential λ:
\(\lambda_3 = 1-e^{-u^3}\), \(\lambda_5 = 1-(1+u^3)e^{-u^3}\);
\(E^{\mathrm{perm}} = \lambda_3/r^2\), \(K = 3\lambda_5/r^3 - \lambda_3/r^3\),
\(\mu_A = \alpha_A(E+K\alpha_B E)/(1-\alpha_A\alpha_B K^2)\),
\(E_{\mathrm{pol}} = -\tfrac12 k(\mu_A+\mu_B)E\), re-derived independently from
the paper formulae.

| Quantity | Atomiverse / pin | Tol | Notes |
| -------- | ---------------: | --- | ----- |
| \(E_{\mathrm{pol}}\) [eV] | **−0.12040946414707571** | \(\lvert\Delta E\rvert < 10^{-9}\) | `kEnergy` and `last_pol_energy()` |
| \(F_{A,x}\) [eV/Å] | **+0.040341072868572643** | \(\lvert\Delta F_x\rvert < 10^{-8}\) | action–reaction with \(B\) |
| \(F_{B,x}\) [eV/Å] | **−0.040341072868572643** | \(\lvert\Delta F_x\rvert < 10^{-8}\) | \(F_{A,x}+F_{B,x}=0\) |
| \(F_{*,y}\), \(F_{*,z}\) [eV/Å] | 0 | \(\lvert F\rvert < 10^{-12}\) | collinear geometry |
| \(\mu_{A,x}\) [e·Å] | 0.082882545215708481 | diagnostic | packed in `last_mu()` |
| \(\mu_{B,x}\) [e·Å] | 0.12342697961661356 | diagnostic | packed in `last_mu()` |
| SCC iterations | 8 | converged | `pol_tol = 10^{-12}` e·Å |
| \(E_{\mathrm{pol}} < 0\) | true | exact | polarization stabilizes for \(\alpha\ge 0\) |

### Performance

The two-site case is below timer resolution, so the wall twin is a 512-ion
disordered rocksalt PIM single point with the same species (q = ±1,
α = 1.0 / 1.5 Å³, L = 22.56 Å); single rank, single thread, best of 7 cold runs.

| | wall | iterations |
| --- | --- | --- |
| MetalWalls c8ad175d (`Melt dipoles computation` timer) | 8.0124e-02 s | 17 (preconditioned CG, Ewald) |
| Atomiverse `PimInducedDipole` launch | 6.17e-02 s | 35 (Jacobi, cutoff O(N²)) |

`ratio_atv_over_mw = 0.77` (gate ≤ 10). Not an equal-accuracy benchmark: the
two E_pol differ by 3.0 % because Atomiverse truncates at 11 Å with no
reciprocal-space dipole path while MetalWalls does the full Ewald sum, so 0.77
is a lower bound on the cost of real parity.

## pim-npt-polarizable-il — Bulk polarizable IL under NPT {#pim-npt-polarizable-il}

### Summary

A bulk polarizable ionic melt barostats on the **full** pressure including the
polarization contribution. `cpu::PimInducedDipole` accumulates the pair
decomposition of its Hellmann–Feynman forces into the Voigt virial

\[
W=\tfrac12\sum_{i\neq j}\mathbf r_{ij}\otimes\mathbf f_{ij}
\quad\text{(Voigt, eV, LAMMPS sign)}
\]

and sums it into `kStress`, the stress meta every barostat reads — so
`cpu::md::NoseHooverNPT` (MTK, LAMMPS `fix npt iso` convention) sees
\(P_{\mathrm{pol}}=\mathrm{tr}(W)/(3V)\) alongside the LJ / Coulomb pair virials
with no PIM-specific plumbing. No new class: the section composes existing
Components. Three legs:

1. **PIM pin** — \(E_{\mathrm{pol}}\), all six Voigt virial components and
   \(P_{\mathrm{pol}}\) against an offline independent numpy implementation of
   the published Applequist/Thole equations.
2. **Stress additivity** — the LJ+coul/cut+PIM stack's trace minus the
   LJ+coul/cut stack's trace equals the pinned PIM trace.
3. **NPT product run** — a stretched toy ionic melt must contract, boundedly,
   with per-step SCC convergence and a finite MTK conserved quantity.

### System

| Item | Setting |
| ---- | ------- |
| Geometry | 4×4×4 simple-cubic rock-salt toy IL, \(d=2.35\) Å (compressed start), cubic \(L=9.4\) Å, \(N=64\) |
| Charges / types | \((-1)^{i_x+i_y+i_z}\): +1 cation (type 0, \(\alpha=0\)), −1 anion (type 1, \(\alpha=3.0\) Å³) |
| Symmetry breaking | `disp[a] = 0.15·sin(1.7·(i+1) + 2.3·a)` Å (a perfect rock-salt lattice has \(E_{\mathrm{pol}}\equiv 0\)) |
| PIM | cutoff 4.0 Å (min-image), `thole_a = 0.39`, `tt_b = 0`, `pol_tol = 10^{-13}` e·Å, `pol_max_iter = 1000` (pin converges in 25 Jacobi iterations); \(k = 14.399645\) eV·Å/e² |
| NPT leg | lattice **stretched** to \(d=2.9\) Å (\(L = 11.6\) Å, start tension \(P_0\approx-0.12\) eV/Å³); `PairLJCut{0.01 eV, 3.0 Å}` + `PairCoulCut` + PIM (`pol_tol` 1e-10, 500 iterations); `NoseHooverNPT{T=800 K, τ_T=100 fs, P=0, τ_P=500 fs, dt=1 fs}` + `Virial` + `InitVelocity`, 250 steps |

The NPT leg starts stretched: a stretched ionic melt contracts onto the stiff
\(r^{-12}\) wall (self-limiting), while the compressed pin-leg density
(\(P_0\approx+0.88\) eV/Å³) would expand ballistically through the soft
attractive side. MTK equations of motion: Tuckerman 2006, Shinoda 2004.

### Usage

Pin leg — PIM alone, virial read from `kStress`:

```cpp
cpu::PimInducedDipole pim(/*cutoff [Å]=*/4.0,
                          /*alpha_by_type [Å³]=*/{0.0, 3.0},
                          /*thole_a=*/0.39,
                          /*pol_tol [e·Å]=*/1e-13,
                          /*pol_max_iter=*/1000);
cpu::Evaluate eval;
eval.on(evaluate::Phase::evaluate).add_ref(pim);
eval.build();
eval.run(f);

const double E_pol = f.meta.get(kEnergy);         // [eV]
const double* W = f.meta.get(kStress);            // Voigt virial [eV]
const double P_pol = (W[0] + W[1] + W[2]) / (3.0 * f.box.volume());
```

NPT product composition (the section adds an out-of-library `TrackVolume`
Criterion via `add_criterion_ref`):

```cpp
cpu::PimInducedDipole pim2(4.0, {0.0, 3.0}, 0.39,
                           /*pol_tol=*/1e-10, /*max_iter=*/500);
cpu::md::MD sim;
sim.set_evaluator(cpu::PairLJCut{0.01, 3.0, 4.0});
sim.set_evaluator(cpu::PairCoulCut{4.0});
sim.set_evaluator_ref(pim2);
sim.set_neighbor(cpu::md::NeighborList<>{4.0, 0.3});
sim.set_integrator(cpu::md::NoseHooverNPT{
    /*T_target [K]=*/800.0, /*tau_T [fs]=*/100.0,
    /*P_target [eV/Å³]=*/0.0, /*tau_P [fs]=*/500.0, /*dt [fs]=*/1.0});
sim.set_observer(cpu::md::InitVelocity{800.0, 20260727u});
sim.set_observer(cpu::md::Virial{});         // kinetic stress
sim.build();
sim.run(f2, 250);

const double H = f2.meta.get(kConserved);    // MTK conserved quantity [eV]
const double V1 = f2.box.volume();           // [Å³]
```

### Validation (oracle: literature, MetalWalls-form-validated)

Pins come from an independent numpy implementation
(`pim_npt_polarizable_il_pin.py`, numpy 2.5.1) of the truncated Hamiltonian
`PimInducedDipole` implements, evaluated on the deterministic start
configuration — not a golden dump of Atomiverse output.

| Quantity | Oracle pin | Tol | Notes |
| -------- | ---------: | --- | ----- |
| \(E_{\mathrm{pol}}\) [eV] | **−8.695749870555906** | \(\lvert\Delta E\rvert \le 10^{-8}\) | `kEnergy` and `last_pol_energy()` |
| \(W_{xx}\) [eV] | **−10.66184373345355** | \(\lvert\Delta W\rvert \le 10^{-7}\) | Voigt on `kStress` |
| \(W_{yy}\) [eV] | **−12.125914211727187** | \(\le 10^{-7}\) | |
| \(W_{zz}\) [eV] | **−9.657852793291474** | \(\le 10^{-7}\) | |
| \(W_{xy}\) [eV] | **−0.2325271422922529** | \(\le 10^{-7}\) | full off-diagonal, not just trace |
| \(W_{xz}\) [eV] | **−2.2469237837238** | \(\le 10^{-7}\) | |
| \(W_{yz}\) [eV] | **+1.375774440282161** | \(\le 10^{-7}\) | |
| \(P_{\mathrm{pol}}=\mathrm{tr}(W)/(3V)\) [eV/Å³] | **−0.013021203851134545** | \(\lvert\Delta P\rvert \le 10^{-10}\) | the pressure the barostat sees |
| Stress additivity [eV] | pinned PIM trace | \(\le 10^{-6}\) | PIM shares the barostat's `kStress` |
| SCC | converged (25 iters) | `pol_tol = 10^{-13}` e·Å | throws on non-convergence |

Tolerances cover `std::sin` vs `np.sin` position ulps, summation order, and the
\(10^{-13}\) SCC tail; Atomiverse reproduces the pins to < 1e-14 eV.

**NPT leg invariants (not pins):** per-step SCC convergence, finite conserved
quantity \(H\), final \(E_{\mathrm{pol}} < 0\), contraction
(\(V_{\min} < 0.97\,V_0\)) that is bounded (\(0.5\,V_0 < V_{\min}\),
\(V_{\max} < 1.2\,V_0\)), finite positive final volume.

**MetalWalls cross-check.** On a cation–anion dimer (only one polarizable site,
no damping, so both codes evaluate the same kernel), MetalWalls' Ewald,
Richardson-extrapolated in \(1/L^3\), gives
\(E_{\mathrm{pol}} = -0.708234375674\) eV against the exact isolated value
\(-\tfrac12 k\alpha/d^4 = -0.708225558187103\) eV (1.2e-05, the two-point
extrapolation residual) — confirming the convention (sign, factor ½,
prefactor). On the 64-ion start configuration MetalWalls (full Ewald,
undamped) gives \(E_{\mathrm{pol}} = -1.682169167852\) eV and
\(P_{\mathrm{pol}} = -3.2121326023\times10^{-3}\) eV/Å³: the pin is **not
MetalWalls-expressible** (no Thole damping, no truncated-Coulomb mode in
MetalWalls), and the gap is the real-space truncation — the numpy oracle at
rc = 4.7 Å = L/2 already moves \(E_{\mathrm{pol}}\) to −4.892485851954 eV
(undamped) or −2.668325687238 eV (Thole), while Thole damping alone moves the
rc = 4.0 Å number by only 0.9 %. The truncated \(E_{\mathrm{pol}}\) of a
rock-salt lattice is not a converged physical quantity.

### Performance

Same 64-ion box as the NPT leg (d = 2.9 Å, L = 11.6 Å), 250 steps, MetalWalls at
the production tolerances of its `example/NaCl_PIM_NPT` (rtol 1.63e-5,
ktol 1e-7, CG dipoles 1e-7), `mpirun -n 1` vs ATV single rank,
`OMP_NUM_THREADS=1`.

| | wall clock | MD loop only |
| --- | --- | --- |
| MetalWalls (best of 3) | 0.83 s | 0.330 s |
| ATV (best of 5, all 3 legs) | 0.06 s | ≈0.055 s |
| **ratio ATV/MW** | **0.072** | **0.17** |

The ratio flatters ATV: MetalWalls does a full Ewald sum over charges,
charge–dipole and dipole–dipole plus a CG dipole solve, ATV a 4.0 Å-truncated
\(O(N^2)\) Jacobi SCC plus LJ + coul/cut. A throughput datapoint, not a
like-for-like kernel benchmark. Runtime ≪ 60 s.

### Limitations

- **Toy IL, not a force field.** 64-ion rock-salt lattice with generic
  LJ + coul/cut; a production polarizable-IL parameterization (CL&Pol,
  Fumi–Tosi + PIM) is downstream usage. The 4.0 Å cutoff is a deterministic
  pin choice, not production electrostatics.
- **Isotropic NPT** (`NoseHooverNPT` iso convention); no anisotropic /
  triclinic cell dynamics in this pin.

### References

- M. E. Tuckerman et al., *J. Phys. A* **39**, 5629 (2006); W. Shinoda et al., *Phys. Rev. B* **69**, 134103 (2004) (MTK NPT, LAMMPS `fix npt iso` convention)

## extended-lagrangian-dipoles — Extended-Lagrangian (iEL/SCF) PIM dipoles {#extended-lagrangian-dipoles}

### Summary

Cold-starting the induced-dipole SCC from \(\boldsymbol{\mu}=0\) every MD step
wastes iterations, and a truncated SCC pumps energy into the trajectory
because the converged dipole lags the nuclei. `cpu::PimXlDipole` is the
iEL/SCF extended-Lagrangian cure (Albaugh, Demerdash & Head-Gordon 2015) — the
classical-polarization analogue of XL-BOMD: an auxiliary dipole set
\(\boldsymbol{\mu}^{aux}\) follows time-reversible dissipative Niklasson
dynamics and warm-starts the SCC each step. Energy and forces written to the
Frame are always the owned `PimInducedDipole`'s SCC-converged Hellmann–Feynman
values — the scheme changes the guess dynamics, never the physics, so the fixed
point at a stationary geometry **is** the plain SCF PIM answer. MetalWalls'
own mass-zero route for dipoles (MaZe, SHAKE-like constrained propagation) is
not implemented.

### Method

Auxiliary dipoles follow the dissipative Niklasson recursion (Table-I
coefficients, default order K = 5):

\[
\mu^{aux}_{n+1} = 2\mu^{aux}_n - \mu^{aux}_{n-1}
  + \kappa\,(\mu^{SCC}_n - \mu^{aux}_n)
  + \alpha \sum_{k=0}^{K} c_k\, \mu^{aux}_{n-k}
\]

Each MD step: (1) the SCC of the owned `PimInducedDipole` is warm-started from
\(\boldsymbol{\mu}^{aux}\) and converged to `pol_tol`; (2) \(E_{\mathrm{pol}}\)
[eV] and Hellmann–Feynman forces land on the Frame; (3) the auxiliary dynamics
advances with the converged \(\boldsymbol{\mu}^{SCC}\). The implementation
composes `cpu::PimInducedDipole` (SCC + forces) and `cpu::electronic::XlBomd`
(Niklasson recursion) element-wise on the nlocal×3 dipole array. The defining
iEL/SCF properties are the dynamics gates: the XL trajectory tracks the
Born–Oppenheimer (cold-SCF) dipole surface, warm starts are strictly cheaper
than cold SCC, and \(\mathrm{KE} + E_{\mathrm{pol}}\) is conserved as well as by
the SCF twin.

### Usage

`solver()` reaches the owned SCC solver (`thole_a`, `tt_b`, `pol_tol`, …);
`propagator()` the Niklasson propagator; `reset()` drops the auxiliary history
at a geometry discontinuity (atom-count changes reset automatically).

```cpp
cpu::PimXlDipole xl(/*cutoff [Å]=*/10.0,
                    /*alpha_by_type=*/{0.0, 1.0, 1.5},
                    /*thole_a=*/0.39,
                    /*xl_order=*/5,          // Niklasson K ∈ [3, 9]
                    /*pol_tol [e·Å]=*/1e-12,
                    /*pol_max_iter=*/200);

cpu::md::MD sim;
sim.set_integrator(cpu::md::VelocityVerlet{/*dt [fs]=*/0.5});
sim.set_evaluator_ref(xl);             // permanent q–q Coulomb added separately
sim.build();
sim.run(frame, 150);

xl.last_pol_energy();    // E_pol [eV] (also on frame.meta.get(kEnergy))
xl.last_pol_iters();     // SCC iterations — small once warm
xl.last_warm();          // false only on the first launch / after reset()
xl.last_aux_error();     // max|μ_aux − μ_scc| [e·Å] — distance to BO surface
xl.last_mu();            // converged dipoles, packed xyz [e·Å]
```

The section runs the XL twin and the cold-SCF Born–Oppenheimer twin
(`PimInducedDipole{10.0, {0,1.0,1.5}, 0.39, 1e-12, 200}`) on identical
trajectories through `cpu::md::MD` (`VelocityVerlet`, `set_evaluator_ref`),
seeding \(F(t=0)\) with a single-point `cpu::Evaluate` before the first
half-kick, and differences them step by step. Files: `pim_xl_dipole.h`,
`pim_induced_dipole.h`, `xl_bomd.h`; unit tests in
`tests/cpu/potentials/test_pim_xl_dipole.cpp`.

### Validation

**Gate A — stationary two-site fixed point, MetalWalls c8ad175d
(`thole_a = 0`).** Geometry of
[`pim-induced-dipole-bulk`](#pim-induced-dipole-bulk) (q = ±1 e,
α = 1.0/1.5 Å³, r = 3 Å, L = 50 Å), driven by `cpu::md::MD` with
`VelocityVerlet{0.0}` so the auxiliary state persists between launches:

| Quantity | MetalWalls pin | Tol | Notes |
| -------- | -------------: | --- | ----- |
| \(E_{\mathrm{pol}}\) [eV] | **−2.4397739572766e-01** (−8.9660039191731e-3 Ha) | 1e-7 rel | cold **and** warm launches |
| \(F_{A,x} = -F_{B,x}\) [eV/Å] | **+3.4926913376021e-01** | 1e-7 rel | cold and warm |
| warm SCC iterations | < cold, and ≤ 2 | exact | `last_warm()` flagged |
| warm aux deviation [e·Å] | < 1e-10 | — | `last_aux_error()`, 0 at a fixed point |

The Thole two-site values of `pim-induced-dipole-bulk` Gate B
(\(E_{\mathrm{pol}} = -0.12040946414707571\) eV,
\(F_{A,x} = \pm 0.040341072868572643\) eV/Å at `thole_a = 0.39`) are the same
fixed point under damping; an independent closed-form re-derivation agrees to
~1e-14 relative.

**Gate B — dynamics** (8-ion perturbed 2×2×2 rocksalt-parity toy, L = 20 Å,
150 × 0.5 fs velocity Verlet from rest, cutoff 10 Å, SCC tol 1e-12 in both
twins, shared `thole_a = 0.39`; \(E_{tot} = \mathrm{KE} + E_{\mathrm{pol}}\)
[eV]):

| Gate | Bound | Notes |
| ---- | ----- | ----- |
| \(\max_t \max_i \lVert\mu^{XL}_i(t) - \mu^{SCF}_i(t)\rVert\) | < 1e-6 e·Å | XL tracks the BO dipole surface |
| total warm SCC iterations | < cold-twin total (strict) | the point of iEL/SCF |
| \(\lvert E_{tot}(t_{end}) - E_{tot}(0)\rvert\), XL twin | < 5e-6 eV | no secular pumping |
| \(\lvert E_{tot}(t_{end}) - E_{tot}(0)\rvert\), SCF twin | < 5e-6 eV | control |

Both twins start from the identical state
(\(\lvert E_0^{XL} - E_0^{SCF}\rvert < 10^{-10}\) eV).

### Performance

`perf_exempt` until a fair XL / mass-zero dipole wall twin is timed; LAMMPS has
no iEL/SCF induced-dipole twin (`ratio_atv_over_oracle` n/a, `within_10x`
exempt). The algorithmic speed claim gated today: total warm SCC iterations
strictly below the cold-SCF twin's on the same trajectory.

### Limitations

- **Guess dynamics only.** Forces/energy are always SCC-converged
  Hellmann–Feynman; the truncated-SCC ("SCF-less") iEL/0-SCF mode is not
  exposed.
- **Not MaZe.** MetalWalls' mass-zero SHAKE-constrained dipole propagation is
  not implemented (the MaZe route exists for electrode charges:
  [`maze-symplectic-charge-dynamics`](#maze-symplectic-charge-dynamics)).

### References

- Albaugh, Demerdash & Head-Gordon, *J. Chem. Phys.* **143**, 174104 (2015) — iEL/SCF
- Niklasson et al., *J. Chem. Phys.* **130**, 214109 (2009), Table I — dissipative recursion coefficients
- Niklasson, *Phys. Rev. Lett.* **100**, 123004 (2008) — time-reversible XL
- Coretti, Bonella & Ciccotti, *J. Chem. Phys.* **149**, 191102 (2018) — MaZe for polarizable models (MetalWalls' route)

## qeq-fluctuating-charge — QEq fluctuating-charge equilibration (LAMMPS `fix qeq/point`) {#qeq-fluctuating-charge}

### Summary

Charge equilibration (electronegativity equalization, Rappé & Goddard 1991):
per-atom fluctuating charges minimize a second-order charge energy with
per-type electronegativity \(\chi\) [eV] and hardness \(\eta\) [eV/e²] under a
total-charge constraint. Unlike the constant-potential electrode family
(`ConstPotBO` / `ConstPotCG` / `ConstPotMaZe`, fixed-Ψ Gaussian electrode
charges), QEq equilibrates **every parameterized atom** against its own
electronegativity. `cpu::md::QEq` binds at `md::Phase::pre_force` (order −50,
the slot of the electrode solvers) and writes `kAtom_charge` before the pair /
k-space forces run, so downstream Coulomb styles see equilibrated charges every
step. The section pins per-atom charges, the QEq objective and the downstream
`coul/cut` energy against an offline LAMMPS run, plus a 1000-solve perf twin
(ratio 0.45).

### Method

Rappé & Goddard (1991). The energy model is the LAMMPS `fix qeq/point` identity
(verified against `src/QEQ/fix_qeq_point.cpp`):

\[
E(q)=\sum_i\Bigl(\chi_i q_i+\tfrac12\eta_i q_i^2\Bigr)
 +\sum_{i<j,\;r_{ij}\le r_c}\frac{q_i q_j}{r_{ij}},
\qquad \sum_i q_i = q_{\mathrm{tot}}
\]

with LAMMPS's **bare** \(1/r\) coupling: no qqrd2e prefactor on the
off-diagonal (`compute_H` fills `0.5/r` on a full neighbor list). The physical
Coulomb energy of the resulting charges is computed downstream by the pair
style with \(k_e = 14.399645\) eV·Å/e². Stationarity gives

\[
(\eta_i\delta_{ij}+J_{ij})\,q=-\chi+\lambda\mathbf 1,
\]

solved with the two-vector scheme of LAMMPS `calculate_Q`: \(Hs=-\chi\),
\(Ht=-\mathbf 1\), \(q=s-((\Sigma s-q_{\mathrm{tot}})/\Sigma t)\,t\) — so
\(\Sigma q = q_{\mathrm{tot}}\) holds exactly. Atomiverse factors the dense
matrix once per launch (Cholesky) and substitutes both right-hand sides;
distances use the box minimum image; the matrix is rebuilt every launch, so
charges follow the moving geometry.

Formaldehyde-like 4-atom cluster, LAMMPS `boundary f f f`, 40 Å cube,
`units metal`:

| id | type | element | x [Å] | y [Å] | z [Å] |
| -- | ---- | ------- | ----- | ----- | ----- |
| 1 | 1 | C | 20.0 | 20.0 | 20.0 |
| 2 | 2 | O | 20.0 | 20.0 | 21.205 |
| 3 | 3 | H | 20.943 | 20.0 | 19.412 |
| 4 | 3 | H | 19.057 | 20.0 | 19.412 |

| type | χ [eV] | η [eV/e²] |
| ---- | -----: | --------: |
| 1 (C) | 5.343 | 10.126 |
| 2 (O) | 8.741 | 13.364 |
| 3 (H) | 4.528 | 13.8904 |

Oracle command: `fix q all qeq/point 1 10.0 1.0e-10 500 param.qeq` with
`pair_style coul/cut 10.0`, `run 0` (setup equilibrates charges before output);
inputs `in.qeq_point`, `in.qeq_perf`, `param.qeq`.

**Out of scope:** shielded / Slater kernels (`qeq/shielded`, `qeq/slater`,
`qeq/dynamic`); Ewald-summed QEq; a sparse / CG solve path.

### Usage

```cpp
cpu::Frame frame(4);                   // H2CO toy: fill x/y/z, type, charge=0
frame.box = cpu::Box::cubic(40.0);

cpu::md::QEq qeq(/*cutoff [Å]=*/10.0);
qeq.set_type(/*type=*/1, /*chi [eV]=*/5.343, /*eta [eV/e²]=*/10.126); // C
qeq.set_type(2, 8.741, 13.364);                                      // O
qeq.set_type(3, 4.528, 13.8904);                                     // H
// qeq.qtot = 0.0;  // constrained total charge [e] (default 0, LAMMPS behavior)

cpu::md::MD sim;
sim.set_neighbor(cpu::md::NeighborList<>{10.0, 0.5});
sim.set_evaluator(cpu::PairCoulCut{/*cutoff [Å]=*/10.0});  // sees equilibrated q
sim.set_integrator(cpu::md::VelocityVerlet{0.0});
sim.set_observer_ref(qeq);       // pre_force −50
sim.build();
sim.run(frame, /*nsteps=*/1);   // charges re-equilibrated every step

const auto& q = qeq.charges();   // [e], frame-row order (also on kAtom_charge)
const double e_qeq = qeq.qeq_energy();   // [eV], bare-1/r kernel objective
const double net = qeq.net_charge();     // [e], equals qtot exactly
```

No fallback: every atom type in the frame must be registered via `set_type` —
`build()` throws on an unparameterized type; `cutoff ≤ 0` and `eta ≤ 0` throw.
Files: `qeq.h`, `src/cpu/dynamics/qeq.cpp` (dense \(H=\eta\delta+1/r\) +
Cholesky + two-vector constraint); unit tests in
`tests/cpu/dynamics/test_qeq.cpp` pin the two-atom closed form
\(q=(\chi_2-\chi_1)/(\eta_1+\eta_2-2/r)\), constraint conservation,
beyond-cutoff decoupling, the energy closed form and the throw contracts.

### Validation (oracle: LAMMPS)

LAMMPS 30 Mar 2026 development (patch_30Mar2026-1074-g7f680de296), QEQ package.

| Quantity | Atomiverse gate | LAMMPS pin | Tol | Notes |
| -------- | --------------: | ---------: | --- | ----- |
| \(q(\mathrm C)\) [e] | `charges()[0]` | **+0.046279970820** | abs ≤ **1e-6** | |
| \(q(\mathrm O)\) [e] | `charges()[1]` | **−0.230795571743** | abs ≤ **1e-6** | |
| \(q(\mathrm H)\) ×2 [e] | `charges()[2..3]` | **+0.092257800461** | abs ≤ **1e-6** | symmetric H pair |
| \(\Sigma q\) [e] | two-vector constraint | **0** | abs ≤ **1e-12** | exact by construction |
| \(E_{\mathrm{QEq}}\) [eV] | `qeq_energy()` | **−0.467311783767** | abs ≤ **1e-6** | bare-1/r objective, derived from pinned q |
| \(E_{\mathrm{coul}}\) (`coul/cut`) [eV] | `PairCoulCut` on equilibrated q | **−0.254699311955** | abs ≤ **1e-6** | LAMMPS thermo `ecoul`, qqr2e = 14.399645 |

Offline repro: `lmp -in in.qeq_point` (charges + ecoul pins), `lmp -in
in.qeq_perf` (1000-solve wall).

### Performance

`in.qeq_perf`: 1000 repeated `fix qeq/point` solves (dt = 0) vs 1000 `QEq`
equilibrations (dense rebuild + Cholesky each launch, continuable
`MD::run(f, 1)` with a zero `FixAddForce` as energy source) on the same
geometry, x86_64 epyc login node, 1 thread / 1 proc. The section prints the
ratio; it is reported, not asserted.

| Twin | \(T_{\mathrm{LAMMPS}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{LAMMPS}}\) | ≤10× | Caveat |
| ---- | --------------------------: | -----------------------: | -------------------------------------------: | ---- | ------ |
| 1000 QEq solves, H₂CO N=4 | 0.000610 | 0.000275 | **0.45** | yes | micro-workload (both sides sub-ms) |

`perf_exempt` false. At N = 4 a dense factor + two substitutions beats LAMMPS's
warm-started CG; do not extrapolate 0.45 to production N (dense is
\(O(N^3)\)).

### Limitations

- **Dense Cholesky** rebuilt every launch — small-N scope (≤ a few hundred
  atoms), matching the constant-potential dense path.
- **Point kernel only** (`qeq/point`).
- **Primary-image minimum-image coupling**; no Ewald-summed QEq.

### References

- A. K. Rappé, W. A. Goddard III, *J. Phys. Chem.* **95**, 3358 (1991), DOI [10.1021/j100161a070](https://doi.org/10.1021/j100161a070)
- LAMMPS QEQ package, `src/QEQ/fix_qeq_point.cpp`
- Related electrode family: [`const-pot-bo-parallel-plate`](md-constant-potential.md#const-pot-bo-parallel-plate)

## maze-symplectic-charge-dynamics — Mass-zero constrained (MaZe) electrode charge dynamics {#maze-symplectic-charge-dynamics}

### Summary

MaZe evolves constant-potential electrode charges as auxiliary degrees of
freedom with zero mass, constrained every step to the constant-potential
condition — the SHAKE-like formulation of Coretti, Scalfi, Bacon, Rotenberg,
Vuilleumier, Salanne, Ciccotti, Bonella (2020). `cpu::md::ConstPotMaZe`
(members `symplectic` (default `true`), `maze_tol` [V], `max_maze_iters`;
accessors `constraint_residual()`, `last_maze_iters()`, `max_dQ()`, `solver()`)
runs the symplectic mass-zero dynamics:

- Verlet extrapolation \(\tilde Q = 2Q_n - Q_{n-1}\) gives the time-reversible
  initial state;
- SHAKE corrections along the constraint gradients
  (\(Q_{n+1} = \tilde Q + A\gamma\), CG on \(A^2\gamma = -\sigma(\tilde Q)\))
  drive the constraint \(\sigma_i = (AQ - (\Psi - b))_i - \chi\) to `maze_tol`
  every step (\(\chi\) = uniform potential shift of the simultaneous
  electroneutrality constraint, along the cached \(u = A^{-1}\mathbf 1\));
- the CG residual of the SHAKE system **is** the constraint violation, so the
  stopping rule bounds \(\lVert\sigma\rVert_\infty\) directly.

Solved tightly, the electrode charges are a function of the ionic positions
only: NVE ionic dynamics is time-reversible and the grand energy
\(KE + U - \sum_i\Psi_iQ_i\) shows no secular drift. `symplectic = false`
selects the CG + Kolafa ASPC adiabatic path on `ConstPotBO` (the
"loose twin" below). `max_maze_iters` is a hard cap — the constraint CG throws
if exceeded.

### Method

For the linear constant-potential constraint the SHAKE iteration reduces to a
linear solve driven to near machine precision (Coretti 2020 §II.C), giving
three algorithmic invariants:

1. **Constraint at solver precision every step:**
   \(\lVert AQ-(\Psi-b)-\chi\mathbf 1\rVert_\infty \le\) `maze_tol`.
2. **Time reversibility** (§II.B, §III): an NVE velocity-flip palindrome
   retraces the ionic trajectory, unlike dissipative predictor / loose-SCF
   adiabatic propagation.
3. **No secular drift** of \(E = KE + U - \sum_i \Psi_i Q_i\) (MetalWalls
   eq. 6), while a loose-tolerance adiabatic twin drifts (Fig. 2 discussion).

The converged charges themselves (MaZe = dense BO solution of the same KKT
system) are pinned in the unit tests against dense Cholesky `ConstPotBO`, which
[`const-pot-bo-parallel-plate`](md-constant-potential.md#const-pot-bo-parallel-plate)
pins against the plate-capacitor algebra. LAMMPS ELECTRODE ships conp/conq
matrix and CG solvers but no mass-zero charge dynamics.

**System.** Parallel-plate capacitor as an isolated 10-atom cluster in an
orthorhombic 30 × 30 × 40 Å cell, `PairGaussianCoul` cutoff 14 Å (> 12.73 Å
cluster diameter, < half of every cell edge, so every pair interacts exactly
once — the minimum-image model `ConstPotBO`'s elastance uses with
`xy_image_range = 0`):

| Region | type | N | z [Å] | η [1/Å] | Ψ [V] | mass [amu] |
| ------ | ---: | ---: | ---: | ---: | ---: | ---: |
| Bottom plate | 1 | 4 | 4 | 2.0 | 0.0 | static |
| Top plate | 2 | 4 | 16 | 2.0 | +1.0 | static |
| Ion + | 3 | 1 | 8.5 | point | — | 40.0 |
| Ion − | 3 | 1 | 11.5 | point | — | 40.0 |

Plate sites at xy ∈ {1.5, 4.5}²; ions ±0.5 e at (2.1, 2.1, 8.5) and
(4.2, 4.2, 11.5) Å (lateral separation 2.97 Å), zero initial velocity. NVE
velocity Verlet, dt = 0.25 fs, 400 steps forward, velocity flip, 400 steps back
(palindrome); grand-energy drift sampled after a 5-step settle. The dissipative
contrast is the identical trajectory with `symplectic = false`, `cg_tol = 1e-3`
and ASPC order 2. MetalWalls' twins were also run on the periodic
6 × 6 × 20 Å cell this cluster was cut from (maximum min-image distance
10.9 Å); a 12 Å cutoff there would count up to two images per direction while
the elastance counts one, so that cell is not used for the dynamics.

**Out of scope:** full electrode Ewald / MetalWalls reciprocal matrix;
electrolyte beyond the two probe ions; thermostat (NVE by design).

### Usage

```cpp
cpu::md::ConstPotMaZe maze(/*eta [1/Å]=*/2.0);
maze.add_electrode(/*type=*/1, /*Psi [V]=*/0.0);
maze.add_electrode(/*type=*/2, /*Psi [V]=*/1.0);
maze.maze_tol = 1e-10;          // ‖σ‖∞ constraint tolerance [V]
// maze.symplectic = false;     // CG+ASPC adiabatic path

std::vector<double> etas = {0.0, 2.0, 2.0, 0.0};   // walls Gaussian, ions point
cpu::PairGaussianCoul pg(/*cutoff=*/14.0, etas, /*include_self=*/true);

cpu::md::MD sim;
sim.set_neighbor(cpu::md::NeighborList<>{14.0, 0.5});
sim.set_evaluator_ref(pg);
sim.set_integrator(cpu::md::VelocityVerlet{0.25});
sim.set_observer_ref(maze);     // pre_force −50
sim.set_observer(cpu::md::FixFreeze{1});
sim.set_observer(cpu::md::FixFreeze{2});
sim.build();
sim.run(frame, 400);
maze.constraint_residual();     // ≤ maze_tol [V]
maze.last_maze_iters();         // 0 when the Verlet guess already satisfies σ
maze.max_dQ();                  // max |Q_i(t) − Q_i(t−1)| [e]
```

MD sorts rows spatially, so the section resolves the ions by tag. The
underlying BO machinery (electrode registration, eta, images, neutrality,
`cg_tol`, `predictor_order`) is reached through `solver()`. Files:
`const_pot_maze.h`, `src/cpu/dynamics/const_pot_maze.cpp` (Verlet
extrapolation + SHAKE constraint CG + χ neutrality shift); unit tests in
`tests/cpu/dynamics/test_const_pot_maze_symplectic.cpp` (dense-BO match per
site, neutrality, zero-iteration re-solve, warm start, adiabatic twin
unchanged, validation).

### Validation (oracle: MetalWalls c8ad175d + Coretti 2020 invariants)

Theory: Coretti et al. 2020, DOI 10.1063/5.0007192; MetalWalls JOSS 2020.

**External pin — dilute-limit electrode charge.** The same 10-atom cluster in a
200 Å cell (cutoff 60 Å) against MetalWalls' Ewald on the identical cluster:

| MW cell L [Å] | MW Q_top [e] | rel Δ vs ATV |
| ---: | ---: | ---: |
| 40 | +0.1267873919608 | +6.301e-4 |
| 60 | +0.1267203095694 | +1.006e-4 |
| 100 | +0.1267091472474 | +1.255e-5 |
| **200** | **+0.1267076937357** | **+1.076e-6** |
| Atomiverse (isolated) | **+0.1267075573679** | — |

Pin: Q_top = +0.1267076937357 e, rel ≤ 1e-4 (MetalWalls' own image residual).
The grand energies agree to 7.473e-6 relative at L = 200 Å (MW −0.9831291093381
eV vs ATV −0.9831217623464 eV), 4e-7 after 1/L³ extrapolation.

**Invariants** (gates carry 10×+ margins over the measurements; Release, GCC
14.3, x86_64 epyc):

| Quantity | Atomiverse (MaZe, measured) | MetalWalls (same invariant) | Tol (gate) | Notes |
| -------- | --------------------------: | --------------------------: | ---------- | ----- |
| Constraint residual \(\lVert\sigma\rVert_\infty\), every step [V] | **9.99e−11** (max) | 2.83e-16 | ≤ **1e−10** | CG stop rule bounds it directly |
| Electrode electroneutrality \(\lvert\sum Q\rvert\), every step [e] | **5.6e−17** (max) | exact | ≤ **1e−9** | χ shift along \(u=A^{-1}\mathbf 1\) |
| Palindrome ion-position return error [Å] | **1.4e−14** | 2.29e-7 | ≤ **1e−6** | 2 × 400 steps, velocity flip |
| Palindrome electrode-charge return error [e] | **9.0e−13** | 5.95e-10 | ≤ **1e−6** | vs first-solve charges |
| MaZe vs loose CG+ASPC twin, position return | 1.4e−14 vs **1.07e−05** | 2.29e-7 vs 1.07e-5 | ratio < **0.1** | reversible vs dissipative |
| NVE grand-energy drift (after 5-step settle) [eV] | **7.57e−08** | 6.76e-7 | ≤ **1e−6** | \(KE+U-\sum\Psi_iQ_i\) |
| MaZe drift vs loose (cg_tol 1e−3) twin drift [eV] | 7.57e−08 vs **1.76e−06** | — | MaZe < twin | 23× better conserving |
| \(Q_{\mathrm{top}}\) first solve, 6 × 6 × 20 Å twin [e] | **+0.135822** | +0.15296 (3D Ewald) | > 0 at ΔΨ = +1 V | twin agrees: +0.135821 |
| Min ion–ion separation [Å] | **3.798** | — | > **1.8** | scenario validity |

MetalWalls' palindrome figure is an upper bound (its text restart file carries
15 significant digits); its constraint holds with `maze inversion` in one exact
solve, and MaZe ≡ Born–Oppenheimer there to 1e-17 e per site.

### Performance

`perf_exempt` with a technical reason: MetalWalls needs 7.6–9.0 s for the
800-step run against ≈ 0.015 s for Atomiverse's MaZe leg (raw ratio ≈ 0.002),
but the codes do different work per step — MetalWalls converges a full 3D Ewald
sum with a very fine k-mesh (the 6 Å lateral cell caps the real-space cutoff),
Atomiverse 45 minimum-image pairs and no reciprocal space. The ratio is not an
engine benchmark. The section's 2 × 800-step NVE on the 10-atom cell (8×8
elastance) plus the dilute solve takes 0.03 s (< 1 s).

### Limitations

- **Minimum-image elastance** on a small textbook cell; no full electrode Ewald.
- **Two probe ions, NVE, static walls** — no electrolyte bath, thermostat or
  piston (compose with `const-pot-il-edl-box`, `piston-electrode-pz`).
- **Dense \(O(N_{\mathrm{elec}}^2)\) mat-vecs** in the constraint CG (\(A^2\gamma\)
  via two applications of \(A\)); large-electrode scaling:
  [`const-pot-cg-large-electrode`](md-constant-potential.md#const-pot-cg-large-electrode).
- The CG+ASPC adiabatic path (`symplectic = false`) is documented under
  [`electrode-charge-predictor`](md-constant-potential.md#electrode-charge-predictor).

### References

- Coretti, Scalfi, Bacon, Rotenberg, Vuilleumier, Salanne, Ciccotti, Bonella, *Mass-zero constrained molecular dynamics for electrode charges in simulations of electrochemical systems*, *J. Chem. Phys.* **152**, 194701 (2020), DOI [10.1063/5.0007192](https://doi.org/10.1063/5.0007192)
