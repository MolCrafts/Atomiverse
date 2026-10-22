# Polarizable continuum solvation (PCM)

A polarizable continuum model embeds the molecule in a cavity inside a
dielectric; apparent surface charges \(q_k\) on the cavity tesserae respond
to the solute potential \(V_k\) and add the electrostatic free energy
\(\Delta G_{\mathrm{el}} = \tfrac12 \sum_k q_k V_k\) (stabilizing ⇒
negative) to the SCF. The public wrapper is
`cpu::scf::WithSolvent<Method>{std::move(mf), SolventModel, eps}` around a
concrete mean-field method (here `cpu::scf::RHF`), with members
`radii_scale` and `x` (`x = 0` selects C-PCM); `run(frame)` performs the
solvated SCF, `e_solvent()` returns \(\Delta G_{\mathrm{el}}\) [Ha], `mf`
exposes the wrapped method, and `forces(frame)` returns the analytic
free-energy gradient (`cpu::scf::GradientResult`).

All sections run on CPU: `h2o_rhf_631gs_cpcm` (C-PCM energy; oracle record
in the case `REFERENCE.md`), `ief-pcm-h2o` (IEF-PCM energy) and
`pcm-grad-h2o` (C-PCM forces).

Case: `ctest -R '^scf/solvation$'` — sources and oracle record in [`regressions/scf/solvation/`](../../../regressions/scf/solvation/). One section of the case per `##` heading below: `atv_regressions --case scf/solvation --section <name>`.

## Common setup

- System: H₂O, O (0, 0, 0.1173), H (0, ±0.7572, −0.4692) Å; RHF / 6-31G*
  (`Pople631Gs`, Cartesian d shells).
- Solvent: dielectric **ε = 78.4**, cavity modified Bondi × 1.2
  (`radii_scale = 1.2`), Lebedev 302 points per sphere. Only the
  electrostatic term is included (no cavitation / dispersion / repulsion).
- Surface: Atomiverse discards buried tesserae (hard bury-delete) and uses
  point-charge AO potentials; PySCF uses SWIG switching and Gaussian-charge
  `int3c2e`. Surface discretization must match the pin protocol in the case
  `REFERENCE.md`.
- Oracle: offline PySCF **2.14.0** `solvent.PCM`; never imported by ctest.

## ief-pcm-h2o — IEF-PCM, H₂O / 6-31G* {#ief-pcm-h2o}

Integral-equation formalism PCM (Cancès–Mennucci–Tomasi).

### Theory

IEF-PCM solves a more complete boundary-integral problem than conductor-like
PCM (finite dielectric). The observable `e_solv` ≡ `e_solvent()` ≡
\(\Delta G_{\mathrm{el}}\).

### Practice

```cpp
cpu::scf::RHF sol_mf{cpu::scf::BasisSet::Pople631Gs};
sol_mf.tol = 1e-10;
cpu::scf::WithSolvent<cpu::scf::RHF> sol{
    std::move(sol_mf), cpu::scf::SolventModel::IEFPCM, /*eps=*/78.4};
sol.radii_scale = 1.2;
sol.run(frame);
double e_solv = sol.e_solvent();   // ΔG_el [Ha]
```

### Results

#### Accuracy vs PySCF 2.14.0 (`method='IEF-PCM'`)

| Quantity | Pin (Ha) | Tol |
| -------- | -------: | --- |
| \(\Delta G_{\mathrm{el}}\) | **−0.01265803490853** | **1e-4** (~0.06 kcal/mol; surface model residual) |
| \(E_{\mathrm{gas}}\) | −76.010504988256 | diagnostic |
| \(E_{\mathrm{IEF}}\) total | −76.022025167150 | diagnostic |

#### Speed

| Twin | \(T_{\mathrm{PySCF}}\) | \(T_{\mathrm{ATV}}\) | \(r\) | ≤10× |
| ---- | ---------------------: | -------------------: | ----: | ---- |
| vacuum `rhf` | 0.66 s | 6.01 s | **9.08** | ✓ (just under 10) |

The twin omits solvent work, so the ratio is an **upper bound** on how
expensive IEF is vs vacuum RHF, not a pure solver A/B.

## pcm-grad-h2o — C-PCM free-energy forces, H₂O / 6-31G* {#pcm-grad-h2o}

Nuclear gradient of the **C-PCM solvation free energy**
\(G = E_{\mathrm{HF}}[P] + \Delta G_{\mathrm{el}}[P,R]\).

### Theory

Forces require consistent derivatives of both the vacuum HF pieces and the
solvent contribution. The gate is **analytic vs finite difference of
\(G\)** (central FD, \(h = 10^{-3}\) Bohr), not PySCF PCM forces at 1e-5 —
the surface discretizations differ (see the case `REFERENCE.md`).

### Practice

```cpp
cpu::scf::RHF rhf{cpu::scf::BasisSet::Pople631Gs};
rhf.tol = 1e-12;
cpu::scf::WithSolvent<cpu::scf::RHF> sol{
    std::move(rhf), cpu::scf::SolventModel::CPCM, /*eps=*/78.4};
sol.radii_scale = 1.2;
sol.x = 0.0;                                       // C-PCM
const cpu::scf::GradientResult res = sol.forces(frame);   // [eV/Å]
```

### Results

#### Accuracy

| Check | Criterion |
| ----- | --------- |
| Analytic vs FD of \(G\) | \(\le 10^{-5}\) Ha/Bohr element-wise |
| PySCF PCM force table | documented in `REFERENCE.md`; **not** the hard gate |

#### Speed

| Twin | \(T_{\mathrm{PySCF}}\) | \(T_{\mathrm{ATV}}\) | \(r\) | ≤10× |
| ---- | ---------------------: | -------------------: | ----: | ---- |
| vacuum `rhf` | 0.67 s | 42.4 s | **62.9** | **✗** |

This is open **speed debt**: the section does full solvent + FD-consistent
work while the twin is vacuum RHF. Optimizing PCM forces is the priority
follow-up; \(r\sim 63\) is not a pass.
