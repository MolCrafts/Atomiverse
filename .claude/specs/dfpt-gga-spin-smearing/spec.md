---
slug: dfpt-gga-spin-smearing
status: approved
created: 2026-08-01
revised: 2026-08-01
domain: pw / response (DFPT phonons)
queue: cp2k-method-queue
depends_on: [dfpt-gga-spin-si, dfpt-spin-smearing, dfpt-gga-si]
unblocks: []
grilled: false
---

# Spec: `dfpt-gga-spin-smearing` — Γ DFPT, joint collinear spin × pure GGA × metallic smear

## Problem

`cpu::pw::DfptPhonon` ships pairwise-pinned products for each axis of the
Γ response:

- `dfpt-gga-spin-si` — collinear spin × pure PBE (fixed occupations)
- `dfpt-spin-smearing` — collinear spin × Fermi–Dirac (LSDA kernel)
- `dfpt-gga-si` — restricted pure PBE

The three-way product path — spin-resolved Sternheimer with a pure-GGA
\(K_{\mathrm{xc}}^{\sigma\sigma'}\) **and** metallic occupation weights
\(f_{\sigma nk}\in[0,1]\) — is accepted by the product API
(`dfpt_gamma_spin_impl` with `use_smear` + `use_gga`) but was **unpinned**:
no offline dual gate, no dedicated regression. A foothold is not a closed
capability.

## Product

One regression `regressions/response/dfpt-gga-spin-smearing/` exercising the
public path only:

```cpp
cpu::pw::PW pw;
pw.xc = cpu::scf::XCType::PBE;
pw.spin_polarized = true;
pw.multiplicity = 1;
pw.smearing = Smearing::fermi_dirac;
pw.degauss = 0.02;           // Ry
pw.extra_bands = 0;
cpu::pw::DfptPhonon dfpt;    // Γ
auto r = dfpt.compute(pw, frame);
```

Geometry: diamond Si twin shared with all DFPT siblings. No new public API.
Gapped fill keeps \(f\in\{0,1\}\), so the path must match the fixed-occ
spin×GGA parent to roundoff (cross-parent consistency).

## Oracle / pins (offline dual gate)

1. **Sharp same-Hamiltonian:** free-energy second difference of collinear
   spin-PBE + FD SCF, h = 0.03 Å, tol 1e-9.
2. **Physics window (CP2K):** CP2K 2026.1 UKS-PBE + SMEAR
   `VIBRATIONAL_ANALYSIS` on the same cell (GTH/MOLOPT twin); optical Γ
   modes inside [450, 650] cm⁻¹; fair wall r ≤ 10.

ctest never launches CP2K. Fair wall measured on the same twin vib job.

## Acceptance

See `acceptance.md`. Green = regression builds under `ATV_BUILD_PW` +
`ATV_BUILD_REGRESSIONS`, all gates pass, wall ≤ 60 s, `within_10x` from a
real CP2K run.

## Out of scope

- Finite-q spin×GGA×smear (separate residual if queued).
- Hybrids; M ≠ 0 open-shell cells; CUDA.
- Live CP2K at ctest time.
