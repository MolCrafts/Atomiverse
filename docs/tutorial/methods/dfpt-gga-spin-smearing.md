# DFPT phonons at Γ, joint collinear spin × pure GGA × metallic smear — diamond Si (PW)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `dfpt-gga-spin-smearing` |
| Regression | `regressions/response/dfpt-gga-spin-smearing/` |
| Public API | `cpu::pw::DfptPhonon` + `PW` members (`xc=PBE`, `spin_polarized`, `smearing=fermi_dirac`) |
| Roadmap | [dft.md](../../roadmap/dft.md) §11 DFPT — **joint GGA × spin × smear** ✓ |
| Queue | `docs/developer/cp2k-method-queue.md` → **done** (worker wave 4, 2026-08-01); fair ratio **0.21** |
| Dep | `dfpt-gga-spin-si` ✓; `dfpt-spin-smearing` ✓; `dfpt-gga-si` ✓ |

## Summary

Pairwise parents pin spin×GGA (fixed occ), spin×smear (LSDA), and restricted
GGA. This product pins the **three-way** Γ path: spin-resolved Sternheimer
with a pure-GGA \(K_{\mathrm{xc}}^{\sigma\sigma'}\) **and** Fermi–Dirac
occupation weights. Diamond Si minimal twin (10 Ry, \(2\times2\times2\) MP,
in-tree NC UPF). Dual gate: same-Hamiltonian free-energy second difference
plus a real CP2K 2026.1 UKS-PBE+SMEAR `VIBRATIONAL_ANALYSIS` physics anchor
(fair wall ratio 0.21). Gapped fill keeps \(f\in\{0,1\}\), so numbers match
the fixed-occ spin×GGA parent to four decimals.

## Method

Per Cartesian displacement the Sternheimer equation is solved per spin
channel with spinor occupations \(f_{\sigma nk}\in[0,1]\) (shared \(\mu\)),
and the induced spin densities are screened by Hartree + the spin-GGA
kernel from FD of `XCBuilder::step_spin`. Hybrids are rejected. Cite:
Baroni et al., *Rev. Mod. Phys.* **73**, 515 (2001); de Gironcoli, *Phys.
Rev. B* **51**, 6773 (1995); PBE, *Phys. Rev. Lett.* **77**, 3865 (1996).

| Item | Setting |
| ---- | ------- |
| Spin | collinear, multiplicity 1 |
| XC | PBE pure spin-GGA |
| Smear | Fermi–Dirac, degauss 0.02 Ry, extra_bands 0 |
| Structure | diamond Si, a = 5.43 Å, 2-atom fcc primitive |
| PP | `tests/data/pw/Si_lda_nc.upf` |
| Cutoff / k | 10 Ry / 2×2×2 MP |
| \(\mathbf{q}\) | Γ |

## Implementation

```cpp
PW method;
method.upf_path = "tests/data/pw/Si_lda_nc.upf";
method.ecut_ry = 10.0;
method.k_mesh = 2;
method.xc = cpu::scf::XCType::PBE;
method.spin_polarized = true;
method.multiplicity = 1;
method.smearing = Smearing::fermi_dirac;
method.degauss = 0.02;
method.extra_bands = 0;
method.tol = 1e-8;

DfptPhonon dfpt;
dfpt.response_tol = 1e-5;
dfpt.cg_tol = 1e-9;
dfpt.mix_beta = 0.7;
const DfptPhononResult r = dfpt.compute(method, frame);
```

```bash
cmake --build build -j"$(nproc)" --target atomiverse reg_dfpt-gga-spin-smearing
cmake --build build --target run_reg_dfpt-gga-spin-smearing
```

## Validation (vs offline pins)

Source: `regressions/response/dfpt-gga-spin-smearing/REFERENCE.md`.
Measured 2026-08-01 (Release, OMP=1):

| Gate | Value |
| ---- | ----- |
| Free-energy FD C00 | 17.3454 eV/Å² |
| DFPT C00 | 17.3711 eV/Å² (\|Δ\| ≤ 0.30) |
| Optical Γ | 579.82 / 579.82 / 579.96 cm⁻¹ ∈ [450, 650] |
| Acoustic | 12.09 cm⁻¹ (≤ 30) |
| CP2K optical anchor | 463.98 / 534.80 / 534.80 cm⁻¹ |

## Performance

| Item | Value |
| ---- | ----- |
| ATV | 48.3 s (serial) |
| CP2K twin | 232.5 s (1 MPI × OMP=1) |
| ratio | **0.21** (`within_10x`, not `perf_exempt`) |

## Limitations

- Gapped-Si twin exercises the occupation-weight machinery near the
  integer-fill limit; true multi-band metals need denser k/ecut and a
  separate product pin.
- Finite-q three-way combo not gated here.
- Hybrids / CUDA out of scope.

## References

- `regressions/response/dfpt-gga-spin-smearing/REFERENCE.md`
- `tools/cp2k_oracle/inputs/dfpt-gga-spin-smearing.note`
- Baroni et al., *Rev. Mod. Phys.* **73**, 515 (2001)
