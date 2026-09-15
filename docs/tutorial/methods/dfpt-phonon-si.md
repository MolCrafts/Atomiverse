# DFPT phonons at Γ — diamond Si (PW LSDA, NC-UPF)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `dfpt-phonon-si` |
| Regression | `regressions/response/dfpt-phonon-si/` |
| Public API | `cpu::pw::DfptPhonon` (members) + `cpu::pw::DfptPhononResult` |
| Roadmap | [dft.md](../../roadmap/dft.md) §11 DFPT — Γ-point foothold ✓ (dispersions ○) |
| Dep | `fd-phonon-supercell` ✓ (FD twin machinery); enables `phonon-dispersion-si` |

## Summary

**Density-functional perturbation theory** computes the dynamical matrix
analytically from self-consistent first-order response instead of `6N`
finite-displacement SCF solves. This foothold delivers **q = 0** for the
plane-wave method `cpu::pw::PW` (restricted LSDA, UPF norm-conserving
pseudopotential, fixed occupations): Sternheimer conduction-space CG per
(k, band), Hartree + LDA-kernel screening of the perturbation, analytic
Ewald second derivatives, KB nonlocal first/second derivative terms.
Reference: Baroni, de Gironcoli, Dal Corso, Giannozzi, *Rev. Mod. Phys.*
**73**, 515 (2001).

## Usage

```cpp
cpu::pw::PW pw;
pw.upf_path = "tests/data/pw/Si_lda_nc.upf";
pw.ecut_ry = 10.0;
pw.k_mesh = 2;
pw.xc = cpu::scf::XCType::LSDA;   // DFPT kernel is LSDA-only
pw.tol = 1e-8;

cpu::pw::DfptPhonon dfpt;         // members: response_tol, cg_tol, mix_beta, …
auto r = dfpt.compute(pw, frame); // ω [cm⁻¹] ascending; C [eV/Å²]
```

`compute` throws `std::invalid_argument` for non-LSDA XC, the toy potential
(empty `upf_path`), metallic smearing, or collinear spin — those regimes are
follow-ons, not silent approximations.

## Validation (regression `reg_dfpt-phonon-si`, ≈ 25 s)

Diamond Si primitive cell (a = 5.431 Å, 10 Ry, 2×2×2 MP), all pins
hard-coded from `REFERENCE.md`:

1. **Energy-FD twin (sharp):** the diagonal force constant matches the
   offline central second difference of the SCF total energy at identical
   settings to **0.07 %** (17.2277 vs 17.2393 eV/Å²). This is the true
   discrete Hessian: the v1 `pw_forces_analytic` gradient deviates from it
   by ≈ 2 % (egg-box/aliasing, documented in REFERENCE.md), so FD-of-forces
   is *not* the oracle.
2. **Acoustic sum rule emerges** (never imposed): |ω_acoustic| ≤ 4.4 cm⁻¹,
   force-constant row sums ≤ 0.0022 eV/Å².
3. **Optical triplet** 575.8–578.5 cm⁻¹, degeneracy spread 2.7 cm⁻¹, inside
   the literature window around the DFPT-LDA / Raman value 517 cm⁻¹
   (Giannozzi et al., PRB **43**, 7231 (1991)); the +12 % offset is the
   documented minimal-settings bias (10 Ry, 2×2×2, experimental a).

Unit tests (`test_cpu_pw_dfpt`): analytic Ewald dynamical matrix vs FD of
Ewald forces (≤ 5e-6 Ha/Bohr²), Sternheimer CG exact on the free-electron
Hamiltonian, and the guard-rail throws. No SCF runs under `tests/`.

## Boundaries

- **q = 0 only** — full dispersions are `phonon-dispersion-si` (queue).
- LSDA kernel only (no GGA f_xc), insulators only (no Fermi-surface terms),
  restricted spin, UPF NC only.
- No LO–TO / Born effective charges (nonpolar Si does not need them).
- CPU only; CUDA out of scope for this loop.
