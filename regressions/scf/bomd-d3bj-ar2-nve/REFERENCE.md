# Reference — Ar₂ Born–Oppenheimer NVE with DFT-D3(BJ) dispersion

## Why this case exists

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

## Method / surface

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

## Why Ar₂

Because at this level **dispersion is the only thing that binds it**, and the
dispersion force is *larger than the total force*: at 3.60 Å the analytic total
force is 1.11e-2 eV/Å and the dispersion part alone is 1.09e-2 eV/Å. Removing
the leg does not perturb the trajectory, it unbinds the dimer — the pair leaves
the well and never returns, which the separation window below catches
immediately. Two atoms, 36 spherical AOs, ~0.22 s per force evaluation.

## Reference values

### 1. The dispersion term itself — external oracle (hard-coded)

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

### 2. The wiring — analytic force vs central difference of the total energy

At R = 3.600000 Å, h = 0.005 Å, all six degrees of freedom, through
`cpu::Evaluate` (public driver, not the internal gradient class):

**worst |F_analytic − (−ΔE_total/2h)| = 4.31e-06 eV/Å**, against a dispersion
force of 1.087e-02 eV/Å at the same geometry.

The residual is pure O(h²) truncation of the central difference, not gradient
error: at h = 0.01 Å it is 1.767e-05 eV/Å, exactly 4× larger. Gate: 5e-5 eV/Å.
**Before the fix this residual was 1.09e-02 eV/Å** — the whole dispersion
force. (The pure dispersion leg is checked far tighter, at 2e-11 eV/Å, by the
unit test `tests/cpu/scf/test_dispersion_gradient.cpp`.)

### 3. The trajectory — NVE conservation

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

## Geometry

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

## What is NOT claimed

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

## Oracle: pyscf-dispersion 1.5.0 / simple-dftd3 (offline)

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

## Runtime

~21 s single-threaded on an x86_64 login node (12 SCF solves for the FD gate,
80 for the trajectory). Budget 60 s.

## Related

- `tests/cpu/scf/test_dispersion_gradient.cpp` — the unit tier: `D3BJ`/`D4`
  `forces()` vs central differences of their own `energy()` (2 ms, no SCF).
- `regressions/scf/d3bj-molecular-pbe`, `regressions/scf/dft-d4-h2o-pbe` —
  the dispersion **energies** vs the same oracle family.
- `regressions/gpw/gpw-d3bj-forces` — the periodic twin of the same analytic
  D3(BJ) gradient (`cpu/gpw/internal/dftd3.cpp` is the only copy of the D3 math
  in the tree; the molecular path calls it).
