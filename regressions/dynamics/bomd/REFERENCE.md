# dynamics/bomd

One case; every section below is run by `atv_regressions --case dynamics/bomd` and keeps its own oracle record.


## Reference — H₂ / STO-3G short RHF BOMD (multi-device)

### Method

Short NVE BOMD: concrete **RHF** (STO-3G) as the force evaluator on
`md::MD` + VelocityVerlet. **One regression case**; device selected at run
time (`--device=cpu|cuda|both`), PyTorch-style — not a separate `*_cuda` case.

### Acceptance

After 5 × 0.5 fs steps:

1. `frame.meta.get(kEnergy)` is finite and **bound** (PE < 0).
2. PE lies in a **literature-class envelope** only: about −1.117 Ha
   (≈ −30.4 eV) for H₂ RHF/STO-3G near R = 0.74 Å — **wide** band
   [−35, −25] eV so a few NVE steps off the minimum still pass.
3. **Never** an ATV self-golden PE-drift snapshot.

### Architecture checks

- No `AimdDriver` — composition only: `MD` + `set_evaluator(RHF)`.
- No public `SCFSolver` / `*Config` / user Context.
- Frame is the only I/O token.

### Provenance

H₂ RHF/STO-3G equilibrium energy class is standard textbook / CCCBDB-class
(≈ −1.117 Ha). This case does **not** pin PE drift against Atomiverse itself.


## Reference — Ar₂ Born–Oppenheimer NVE with DFT-D3(BJ) dispersion

### Why this case exists

**It exists because the forces were not the gradient of the energy.**

`MeanFieldMethod` folds the geometry-only DFT-D3(BJ) (and DFT-D4) dispersion
energy into the total it reports — `e_total_ += e_disp_` in
`src/cpu/scf/mean_field_method.cpp` — and until **2026-07-28**
`internal::ScfGradient` carried **no dispersion term at all**. For every run
with `d3bj = true` or `d4 = true`, the reported force was the gradient of a
*different function* than the reported energy:

- a single-point energy looks perfectly fine (it is);
- a geometry optimisation converges to the stationary point of the **vacuum**
  surface, not the dispersion-corrected one;
- a Born–Oppenheimer NVE trajectory **does not conserve energy** — it drifts by
  the dispersion work done along the path.

AO Car–Parrinello dodged the bug by refusing `d3bj`/`d4` outright (the refusal
is still there, for the now-mirrored reason: the CP energy functional carries no
E_disp — see `validate_cpmd_config_`). Plain BOMD had no such guard. This case
is the standing witness that the leg exists and is correct.

### Method / surface

**E(R) = E_RHF/def2-SVP(R) + E_D3(BJ)(R)**, isolated (non-periodic) Ar₂.

DFT-D3 with Becke–Johnson (rational) damping, two-body C6/C8, at the **PBE**
parameter set — the `cpu::scf::D3BJ` defaults that the `d3bj` flag uses:
s6 = 1, s8 = 0.7875, a1 = 0.4289, a2 = 4.4407 Bohr; original Grimme
coordination numbers; no three-body ATM.

```
E_disp = −Σ_{A<B} [ s6 C6^AB/(R^6 + f^6) + s8 C8^AB/(R^8 + f^8) ],
f = a1 √(C8^AB/C6^AB) + a2
```

Public API: `cpu::scf::RHF` with `rhf.d3bj = true`, driven by `cpu::Evaluate`
(force gate) and `cpu::md::MD` + `VelocityVerlet` (trajectory). The dispersion
term is also read directly through `cpu::scf::D3BJ::energy` /
`cpu::scf::D3BJ::forces`.

Primary literature: S. Grimme, J. Antony, S. Ehrlich, H. Krieg,
*J. Chem. Phys.* **132**, 154104 (2010) (D3); S. Grimme, S. Ehrlich,
L. Goerigk, *J. Comput. Chem.* **32**, 1456 (2011) (BJ damping).

### Why Ar₂

Because at this level **dispersion is the only thing that binds it**, and the
dispersion force is *larger than the total force*: at 3.60 Å the analytic total
force is 1.11e-2 eV/Å and the dispersion part alone is 1.09e-2 eV/Å. Removing
the leg does not perturb the trajectory, it unbinds the dimer — the pair leaves
the well and never returns, which the separation window below catches
immediately. Two atoms, 36 spherical AOs, ~0.22 s per force evaluation.

### Reference values

#### 1. The dispersion term itself — external oracle (hard-coded)

Ar₂ at **R = 3.400000 Å** (the trajectory start), PBE-D3(BJ):

| Quantity | Value |
|---|---|
| E_disp | **−0.0005527440863375172 Ha** |
| ∂E_disp/∂z on atom 0 | **−2.0714714378968497e-04 Ha/Bohr** |
| ∂E_disp/∂z on atom 1 | **+2.0714714378968497e-04 Ha/Bohr** |

Source: offline **simple-dftd3** through **pyscf-dispersion 1.5.0** (PySCF
2.14.0). External-engine oracle — not a golden snapshot of Atomiverse's own
output, and nothing in CMake or the regression imports it.

Measured agreement (2026-07-28): energy **4.3e-19 Ha**, gradient **5.4e-20
Ha/Bohr** — i.e. bit-level. The gates are set at 1e-12 to leave room for
libm/ULP differences across platforms.

The **gradient** row is the load-bearing one: it is an external check on
exactly the machinery that was missing, including the coordination-number
chain dC6/dCN · dCN/dR.

#### 2. The wiring — analytic force vs central difference of the total energy

At R = 3.600000 Å, h = 0.005 Å, all six degrees of freedom, through
`cpu::Evaluate` (public driver, not the internal gradient class):

**worst |F_analytic − (−ΔE_total/2h)| = 4.31e-06 eV/Å**, against a dispersion
force of 1.087e-02 eV/Å at the same geometry.

The residual is pure O(h²) truncation of the central difference, not gradient
error: at h = 0.01 Å it is 1.767e-05 eV/Å, exactly 4× larger. Gate: 5e-5 eV/Å.
**Before the fix this residual was 1.09e-02 eV/Å** — the whole dispersion
force. (The pure dispersion leg is checked far tighter, at 2e-11 eV/Å, by the
unit test `tests/cpu/scf/test_dispersion_gradient.cpp`.)

#### 3. The trajectory — NVE conservation

80 steps × 8.0 fs = 640 fs, Velocity Verlet, from rest at R = 3.40 Å
(≈ one full period of the dispersion well; the harmonic estimate is 615 fs).

| Measured (2026-07-28) | Value | Gate |
|---|---|---|
| separation range | [3.4000, 4.4434] Å | must stay in [3.30, 4.60] |
| worst \|E_tot − E_tot(step 1)\| | **1.5583e-05 eV** | ≤ 1e-4 eV |
| E_disp span along the trajectory | **9.6647e-03 eV** | ≥ 5e-3 eV |
| span / drift | **620** | ≥ 100 |

The drift is Velocity-Verlet discretization, not force/energy inconsistency: it
scales as dt² (3.93e-06 eV at dt = 4 fs, 1.56e-05 eV at dt = 8 fs).

**The last row is the witness.** E_disp changes by 9.66e-03 eV between the
turning points; dispersion is conservative, so that swing *is* the work the
missing force term would have failed to account for. The conserved energy holds
620× tighter than the size of the term. Without the leg the drift would be of
the order of the swing — and the dimer would not be there to measure, having
left the well on the first outward swing.

Identical numbers at `OMP_NUM_THREADS=1` and unpinned (both drivers call
`set_threads(1)`; 2 atoms make fork/join dominate otherwise).

### Geometry

Cubic 24 Å cell (isolated-molecule SCF; the box only keeps the dimer off the
faces), dimer centred on z:

| Leg | Atom | x | y | z |
|---|---|---|---|---|
| trajectory (R = 3.40 Å) | Ar | 12.000000 | 12.000000 | 10.300000 |
| | Ar | 12.000000 | 12.000000 | 13.700000 |
| force gate (R = 3.60 Å) | Ar | 12.000000 | 12.000000 | 10.200000 |
| | Ar | 12.000000 | 12.000000 | 13.800000 |

Masses 39.948 amu; initial velocities zero. Basis **def2-SVP** (spherical,
36 AOs); SCF `tol = 1e-10` Ha (the FD of the total energy needs it).

### What is NOT claimed

- **This is not a benchmark of Ar₂ binding.** RHF paired with the *PBE*-D3(BJ)
  damping parameters is not a published functional combination — the `d3bj`
  flag has no user-facing parameter surface today, so it always applies the
  PBE set (`scf::D3BJ` defaults). Grimme's HF-D3(BJ) parameters
  (s8 = 0.9171, a1 = 0.3385, a2 = 2.8830) are **not** what this case runs.
  Exposing per-method damping parameters is owed work; until then no
  interaction-energy oracle applies to the *total* here, and none is pinned.
- **The experimental Ar₂ well (D_e = 99.2 cm⁻¹ ≈ 12.3 meV at R_e = 3.757 Å;
  R. A. Aziz, *J. Chem. Phys.* **99**, 4518 (1993)) is context, not a pin.**
  This surface puts its minimum near 3.7 Å with a ~9 meV well relative to
  4.6 Å, which is in the right neighbourhood — but def2-SVP BSSE alone is worth
  ~1.4 meV of spurious binding here, so agreement would be luck, not validation.
- **What IS pinned against an external engine is the dispersion term**
  (energy *and* gradient, §1). The rest of the surface is gated by internal
  consistency (§2) and by conservation (§3), which is exactly the property that
  was broken.

### Oracle: pyscf-dispersion 1.5.0 / simple-dftd3 (offline)

Interpreter: `~/work/.x86_64/bin/python` (PySCF 2.14.0, pyscf-dispersion 1.5.0).
Nothing in CMake or ctest imports it.

```bash
~/work/.x86_64/bin/python - <<'EOF'
import json
from pyscf import gto
from pyscf.dispersion import dftd3 as d3mod
for r in (3.40, 3.60):
    mol = gto.M(atom=f"Ar 0.0 0.0 0.0\nAr 0.0 0.0 {r:.6f}", basis="def2-svp",
                unit="Angstrom", cart=False, verbose=0)
    res = d3mod.DFTD3Dispersion(mol, xc="pbe", version="d3bj").get_dispersion(grad=True)
    print(r, res["energy"], res["gradient"].tolist())
EOF
```

Output pinned 2026-07-28:

```
3.4 -0.0005527440863375172 [[0.0, 0.0, -0.00020714714378968497],
                            [0.0, 0.0,  0.00020714714378968497]]
3.6 -0.00047316652855053566 [[0.0, 0.0, -0.0002113265866359706],
                             [0.0, 0.0,  0.0002113265866359706]]
```

(The 3.60 Å row is the geometry of the force gate; the case checks the 3.40 Å
row numerically and uses the 3.60 Å magnitude, 1.0867e-02 eV/Å, only to report
how large the leg it is testing actually is.)

### Runtime

~21 s single-threaded on an x86_64 login node (12 SCF solves for the FD gate,
80 for the trajectory). Budget 60 s.

### Related

- `tests/cpu/scf/test_dispersion_gradient.cpp` — the unit tier: `D3BJ`/`D4`
  `forces()` vs central differences of their own `energy()` (2 ms, no SCF).
- `regressions/scf/dispersion`, `regressions/scf/dispersion` —
  the dispersion **energies** vs the same oracle family.
- `regressions/gpw/d3` — the periodic twin of the same analytic
  D3(BJ) gradient (`cpu/gpw/internal/dftd3.cpp` is the only copy of the D3 math
  in the tree; the molecular path calls it).


## Reference — GPW BOMD short trajectory (H₂O NVE)

### Method

Born–Oppenheimer molecular dynamics (BOMD) with Gaussian-and-plane-waves
(GPW / Quickstep) forces:

- Outer loop: `cpu::md::MD` + `VelocityVerlet` (NVE; no thermostat)
- Force/energy: `cpu::gpw::GPWEvaluator` with `compute_forces = true`
  (analytic Hellmann–Feynman + Pulay + grid/core; product path)
- Fixed cubic cell (GPW rejects box changes after `build()`)
- Units on the public surface: energy eV, length Å, time fs, force eV/Å

Literature basis for the **observable** (total-energy conservation on NVE when
\(\mathbf{F} \approx -\nabla E\)):

- Frenkel & Smit, *Understanding Molecular Simulation*, 2nd ed., §4.3.4
  (Velocity Verlet is symplectic; energy error bounded/oscillatory).
- Allen & Tildesley, *Computer Simulation of Liquids*, 2nd ed., §3.2.
- Marx & Hutter, *Ab Initio Molecular Dynamics* (CUP) — BOMD chapter:
  energy conservation is the primary short-trajectory diagnostic when the
  electronic structure is solved self-consistently each step.
- VandeVondele et al., *Comput. Phys. Commun.* **167**, 103 (2005) — GPW
  Quickstep energy/force framework (same method family as CP2K).

**Force quality** for this geometry/method is already offline-pinned vs
**CP2K 2026.1** in `regressions/gpw/rks` (max |ΔF| ≈ 3.4×10⁻⁴
Ha/Bohr). This slug pins **MD binding + NVE conservation** on the product
`GPWEvaluator` path, not a second absolute force table.

This is **not** a golden snapshot of Atomiverse total energy. The pin is the
**envelope** on relative drift that a working force↔energy binding must obey.

### System

| Quantity | Value |
| -------- | ----- |
| Species | H₂O (O + 2 H), isolated / MT |
| Geometry | CP2K getting-started (same as `gpw-forces-water`) |
| Masses | O 15.999 amu; H 1.008 amu |
| Cell | cubic \(L = 10\) Å; `PERIODIC NONE` / `nonperiodic = true` |
| XC | PBE |
| Basis | DZVP-MOLOPT-GTH |
| PP | GTH-PBE-q6 (O), GTH-PBE-q1 (H) |
| CUTOFF | **200 Ry** (BOMD envelope; forces pin uses 400 Ry) |
| SCF | `tol = 1e-5`, `max_iter = 60`, `mix = 0.3` |
| Ensemble | NVE |
| `dt` | 0.5 fs |
| Initial velocity | COM-preserving H stretch \(v_x(\mathrm{H1})=+0.0015\), \(v_x(\mathrm{H2})=-0.0015\) Å/fs |
| Steps | 1 warm-up + 4 production (chunked) |

| Atom | x | y | z (Å) |
| ---- | - | - | - |
| O | 5.0000 | 5.0000 | 5.0000 |
| H1 | 5.7586 | 5.0000 | 5.5043 |
| H2 | 4.2414 | 5.0000 | 5.5043 |

**Why cheaper cutoff than the forces pin?** One ENERGY_FORCE at 400 Ry is
~40 s wall on login-class ATV (see `gpw-forces-water` REFERENCE). A multi-step
BOMD at that cost exceeds the ≤60 s regression gate. 200 Ry keeps the same
method family (PBE / MOLOPT / GTH / MT / analytic forces) while staying inside
the time budget; absolute E/F parity remains the forces-pin’s job.

### Offline oracle (CP2K)

**Status:** no `cp2k` / `cp2k.psmp` / `cp2k.sopt` binary and no CP2K Lmod
module on the worker host used for this pin (Arrhenius login-class node,
2026-07-23 inventory). A CP2K twin would be short NVE AIMD / MD with
Quickstep GPW on the same H₂O cell (see suggested input under
`tools/cp2k_oracle/inputs/` when added); when a binary is available, re-run
offline via `tools/cp2k_oracle/` and replace this section with version string
+ measured drift envelope + fair wall.

Until then the regression asserts the **literature NVE/BOMD conservation
envelope** and finite/bounded trajectory diagnostics, not a CP2K absolute
trajectory energy series. Force twin remains `gpw-forces-water` (CP2K 2026.1).

### Observable and tolerance

Let \(E(t) = E_\mathrm{pot}(t) + E_\mathrm{kin}(t)\) [eV], with \(E_0\) taken
after a single warm-up step (energies populated).

\[
\delta = \max_t \frac{|E(t) - E_0|}{\max(1, |E_0|)} \le 0.05
\]

Additional gates:

- PE finite and bound (\(< -100\) eV after warm-up for PBE water)
- KE finite and \(\ge 0\), non-trivial after warm-up
- All coordinates finite
- Soft O–H bond ceiling: both \(r_{\mathrm{OH}} < 1.8\) Å after production

### Perf gate

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| Reason | No offline CP2K binary on host for a fair wall twin of multi-step AIMD |
| `ratio_atv_over_cp2k` | n/a |
| `within_10x` | n/a (exempt) |

When a CP2K twin becomes available, record wall times for the same step
count / comparable settings and require \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}
\le 10\) or document a blocked-speed status after one optimize pass.

Note: single-point force wall vs CP2K already measured in
`gpw-forces-water` (ratio **3.56**, within_10x).

### Dual gate summary

| Gate | Result |
| ---- | ------ |
| Correctness | Regression `reg_gpw-bomd-water` asserts A4–A6 of
  `.claude/specs/gpw-bomd-water/spec.md` |
| Speed | `perf_exempt=true` (no CP2K binary for AIMD twin wall) |

**ATV wall (login-class, OMP=4, 2026-07-23):** ≈ **35 s** for 1 warm-up + 4
production steps at CUTOFF 200 Ry — within A10 (≤ 60 s). Measured relative
drift \(\delta \approx 1.2\times 10^{-5}\) ≪ 0.05.


## Reference — plane-wave BOMD foothold (H atom NVE)

### Method

Born–Oppenheimer molecular dynamics (BOMD) with plane-wave DFT forces:

- Outer loop: `cpu::md::MD` + `VelocityVerlet` (NVE; no thermostat)
- Spin: one valence electron is a doublet (\(S=1/2\)), so the SCF is collinear
  spin-polarized (`spin_polarized = true`, `multiplicity = 2` ⇒
  \(N_\uparrow=1\), \(N_\downarrow=0\)). The restricted, fixed-occupation path
  rejects an odd electron count outright — `internal::require_closed_shell`
  (`src/cpu/planewave/internal/pw_setup.cpp`), the one gate `PW::run`,
  `pw_forces_analytic` and `pw_stress` all call. The default
  `multiplicity = 1` fails a *different* guard, the collinear path's parity
  check: \(N_e + (M-1)\) must be even, which 1 + 0 is not.
- Force/energy: `cpu::pw::PWEvaluator` with `with_gradient(true)`
  (analytic local Hellmann–Feynman + Ewald). A force-producing step is **two**
  SCF solves on today's code path — `PW::run` publishes the energy, then
  `pw_forces_analytic` converges its own density for the forces
  (`src/cpu/planewave/pw_evaluator.cpp`) — so the "5 iterations per step"
  measured below is per solve, and a step pays for two of them. The
  pseudopotential is the
  published analytic **GTH-PBE-q1** parameterization for hydrogen, evaluated
  in-tree from `cpu/gpw/basis_data/gpw_tables.h` — no external file. That
  entry carries no nonlocal projectors, so the KB term is legitimately zero
  here (it is not "switched off").
- Units on the public surface: energy eV, length Å, time fs, force eV/Å

Literature basis for the **observable** (total-energy conservation on NVE when
\(\mathbf{F} \approx -\nabla E\)):

- Frenkel & Smit, *Understanding Molecular Simulation*, 2nd ed., §4.3.4
  (Velocity Verlet is symplectic; energy error bounded/oscillatory).
- Allen & Tildesley, *Computer Simulation of Liquids*, 2nd ed., §3.2.
- Marx & Hutter, *Ab Initio Molecular Dynamics* (CUP) — BOMD chapter:
  energy conservation is the primary short-trajectory diagnostic when the
  electronic structure is solved self-consistently each step.

**Symmetry pin** for residual force (same as `regressions/pw/rks`):
a single ion at a cubic high-symmetry site has \(\mathbf{F}=\mathbf{0}\) by
lattice symmetry — a statement about the lattice, independent of which
pseudopotential is used. Residual from short SCF is bounded by
\(|F| < 0.5\) eV/Å.

This is **not** a golden snapshot of Atomiverse total energy. The pin is the
**envelope** on relative drift that a working force↔energy binding must obey.

### System

| Quantity | Value |
| -------- | ----- |
| Species | 1 × H (Z = 1) at the origin |
| Mass | 1.008 amu |
| Cell | cubic \(L = 6\) Å |
| Pseudopotential | `GthPseudopotential{"GTH-PBE-q1"}` (analytic GTH; same foothold as `h_force_*`) |
| \(E_\mathrm{cut}\) | 5 Ry |
| k-mesh | Γ-only (`k_mesh = 1`) |
| SCF | `tol = 1e-4`, `max_iter = 12`, `mix_beta = 0.5` |
| Spin | collinear `spin_polarized = true`, `multiplicity = 2` (doublet) |
| Ensemble | NVE |
| `dt` | 0.5 fs |
| Initial velocity | \(v_x = 0.002\) Å/fs (hand-set; `InitVelocity` COM-zeros N=1) |
| Steps | 1 warm-up + 10 production (chunked) |

**Why not H₂?** At the 5 Ry cutoff this slug uses, the H 1s state is far
from represented, so a two-proton cell has no meaningful bonding curve and a
short NVE traj is not a useful energy-drift pin. The product force math is
already covered by
`h_force_analytic` / FD unit tests on dimers; this slug pins **MD binding +
NVE conservation** on the symmetry-stable single-H foothold. A production
Si/UPF BOMD traj is a follow-on once `pw-lda-si` re-pins bulk energy.

### Offline oracle (CP2K)

**Status:** no `cp2k` / `cp2k.psmp` / `cp2k.sopt` binary and no CP2K Lmod
module on the worker host used for this pin (Arrhenius login-class node,
2026-07-23 inventory). A CP2K twin would be a short NVE AIMD for a single H
(or H₂ with a real PP) in a comparable cell; when a binary is available,
re-run offline via `tools/cp2k_oracle/` and replace this section with version
string + measured drift envelope.

Until then the regression asserts the **literature NVE/BOMD conservation
envelope** and the **symmetry force residual**, not a CP2K absolute energy.

### Observable and tolerance

Let \(E(t) = E_\mathrm{pot}(t) + E_\mathrm{kin}(t)\) [eV], with \(E_0\) taken
after a single warm-up step (energies populated).

\[
\delta = \max_t \frac{|E(t) - E_0|}{\max(1, |E_0|)} \le 0.05
\]

Additional gates:

- PE finite; KE finite and \(\ge 0\)
- All coordinates finite
- \(|F| < 0.5\) eV/Å after warm-up (symmetry residual)

Measured 2026-08-15 (spin-polarized doublet, 4 OpenMP threads):
\(\delta = 2.17\times10^{-8}\) (tol 0.05), \(|F_0| = 1.13\times10^{-6}\) eV/Å
(tol 0.5), \(E_0 = -12.0275\) eV (\(E_\mathrm{pot,0} = -12.0277\) eV,
\(E_\mathrm{kin,0} = 2.089\times10^{-4}\) eV); SCF converges in 5 iterations
per step with \(m = 1\).

### Perf gate

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| Reason | No offline CP2K (or QE) binary on host for a fair wall twin |
| `ratio_atv_over_cp2k` | n/a |
| `within_10x` | n/a (exempt) |

When a CP2K twin becomes available, record wall times for the same step
count / comparable settings and require \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}
\le 10\) or document a blocked-speed status after one optimize pass.

### Dual gate summary

| Gate | Result |
| ---- | ------ |
| Correctness | Regression `reg_pw-bomd-h` asserts A4–A6 of
  `.claude/specs/pw-bomd-h/spec.md` |
| Speed | `perf_exempt=true` (no CP2K binary) |
