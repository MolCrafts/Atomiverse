# Reference — finite-q DFPT phonons, joint collinear spin × pure GGA × metallic smear (diamond Si at X, PW)

## Method

Density-functional perturbation theory at finite phonon wavevector
\(\mathbf{q}=X=(1/2,0,0)\) for the concrete plane-wave method `cpu::pw::PW`
with the **four-way joint** product path:

- collinear spin (`spin_polarized = true`, `multiplicity = 1` ⇒ \(M=0\)),
- pure GGA kernel (`xc = XCType::PBE`),
- Fermi–Dirac metallic occupation smearing (`smearing = fermi_dirac`,
  `degauss = 0.02` Ry, `extra_bands = 0`),
- commensurate finite \(\mathbf{q}\) via 2×1×1 supercell Γ-DFPT + IFC Fourier:

\[
\Delta V_{\mathrm{Hxc},\sigma}[\Delta n_\uparrow,\Delta n_\downarrow]
  = \Delta V_{\mathrm{H}}[\Delta n]
  + \sum_{\sigma'} K_{\mathrm{xc}}^{\sigma\sigma'}[\Delta n_{\sigma'}],
\]

with spinor occupations \(f_{\sigma nk}\in[0,1]\) (shared \(\mu\)), Hartree
from total \(\Delta n\), and \(K_{\mathrm{xc}}^{\sigma\sigma'}\) from central
FD of the **spin-GGA** XC operator (`XCBuilder::step_spin`). Gapped fill at
these settings keeps near-integer FD occupations (\(f\in\{0,1\}\)), so free
energy \(F\approx E_{\mathrm{KS}}\) and results track the fixed-occ
`dfpt-gga-spin-finite-q-si` parent.

```cpp
cpu::pw::PW pw;
pw.xc = cpu::scf::XCType::PBE;
pw.spin_polarized = true;
pw.multiplicity = 1;
pw.smearing = Smearing::fermi_dirac;
pw.degauss = 0.02;
pw.extra_bands = 0;
cpu::pw::DfptPhonon dfpt;
dfpt.q_frac = {0.5, 0.0, 0.0};   // X → 2×1×1 supercell Γ-DFPT
auto r = dfpt.compute(pw, frame);
```

Cite: Baroni, de Gironcoli, Dal Corso, Giannozzi, *Rev. Mod. Phys.* **73**,
515 (2001); de Gironcoli, *Phys. Rev. B* **51**, 6773 (1995); Giannozzi et
al., *Phys. Rev. B* **43**, 7231 (1991); Perdew, Burke, Ernzerhof, *Phys.
Rev. Lett.* **77**, 3865 (1996).

Siblings: `dfpt-gga-spin-smearing` (Γ, spin×GGA×smear),
`dfpt-gga-spin-finite-q-si` (X, spin×GGA, fixed occ),
`dfpt-finite-q-spin-smearing` (direct q, spin×LSDA×smear).

## System (regression twin)

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si, 2-atom fcc primitive cell, a = **5.43 Å** (experimental) |
| \(\mathbf{q}\) | \(X=(1/2,0,0)\) → **2×1×1** supercell (4 atoms) |
| Pseudopotential | `tests/data/pw/Si_lda_nc.upf` (ONCVPSP NC, in-tree) |
| XC | `XCType::PBE` (true spin-GGA kernel) |
| Spin | collinear, multiplicity = 1 (\(N_\uparrow=N_\downarrow=8\) on the supercell) |
| Smearing | Fermi–Dirac, `degauss = 0.02` Ry, `extra_bands = 0` |
| Cutoff | **10 Ry** (minimal-twin budget) |
| k-mesh | **1×1×1** on the supercell (Γ+X of primitive folded) |
| SCF | tol 1e-8 (density L2), linear mix on spin channels |
| Response | `response_tol` 1e-4, `mix_beta` 0.25, `cg_tol` 1e-8, max_iter 120 |

## Offline pins (no live CP2K at regression time)

| Field | Value |
| ----- | ----- |
| Oracle 1 (physics anchor) | **CP2K 2026.1** (rev `git:5e54ba2`, source build `…/cp2k/install/bin/cp2k.psmp`, env `…/install/env.sh`, this host, 2026-08-01) — `VIBRATIONAL_ANALYSIS` (FD of analytic UKS k-point forces, `FULLY_PERIODIC T`, DX 0.01 bohr, `EPS_SCF 1e-8`) on the **2×1×1 supercell** (4 atoms): LSD + `MULTIPLICITY 1` + PBE + `&SMEAR FERMI_DIRAC` at 3157.75 K (≡ 0.02 Ry), ADDED_MOS 4, DZVP-MOLOPT-SR-GTH / GTH-PBE-q4, CUTOFF 300 Ry, 1×2×2 MP (≡ 2×2×2 on the primitive). Input: `tools/cp2k_oracle/inputs/dfpt-gga-spin-finite-q-smearing/si4_uks_pbe_smear_vib.inp` |
| Oracle 1 output lines | `VIB\|Frequency (cm^-1)    98.223723           130.533918           364.045396` / `427.180368           452.075468           458.190888` / `536.999556           537.000981           541.694572` — supercell Γ FD folds primitive Γ ∪ X; modes 1–6 (cells out of phase) = **X branch: 98.22, 130.53 (TA, FD-split doublet), 364.05, 427.18 (LA/LO), 452.08, 458.19 (TO) cm⁻¹**; modes 7–9 = Γ optical 537.00, 537.00, 541.69. Matches the fixed-occ spin×GGA supercell twin at \(M=0\) to the expected gapped-smear digit. |
| Pin 2 (sharp, same-Hamiltonian) | central **second difference of the collinear-spin PBE free energy** on the **2×1×1 supercell** under FD smear: C[(0x),(0x)] = (F(+h) − 2F₀ + F(−h))/h², h = 0.03 Å, SCF tol 1e-9, `mix_beta` 0.3, identical ecut/k/XC/UPF/spin/FD |
| Pin 3 (cross-parent identity) | \(M=0\) + gapped FD must reproduce the fixed-occ joint spin×GGA finite-q parent (`dfpt-gga-spin-finite-q-si`: DFPT C00 21.1911 eV/Å², \|ω\| ≈ 109, 109, 239, 451, 669, 669 cm⁻¹) to spin-doubling / near-integer-fill roundoff |

The CP2K twin necessarily differs in PP/basis representation (GPW
GTH/MOLOPT FD-supercell at 300 Ry vs the LDA-generated NC-UPF plane-wave twin
at a 10 Ry minimal cutoff with a Γ-only supercell mesh), so it anchors the
**physics window**, not the digit. The same \|ω\| windows the finite-q
parents use — low 3 ∈ [50, 400], high 3 ∈ [300, 850] cm⁻¹ — contain both
the CP2K X branch (98, 131, 364 \| 427, 452, 458) and the biased ATV minimal
twin (109, 109, 239 \| 451, 669, 669). The free-energy FD pin is the sharp
correctness gate for the joint spin-GGA×smear supercell kernel.

### Measured pins (this host, 2026-08-01, Release, OMP=1)

| Quantity | Value |
| -------- | ----- |
| Free-energy FD C[(0x),(0x)] (h = 0.03 Å, spin-PBE + FD on supercell, tol 1e-9, mix_beta 0.3) | **21.2006 eV/Å²** (F₀ = −15.121055430501 Ha, F± = −15.120904142901 / −15.120505520962 Ha; \|TS\| ~ 2e-15 Ry ⇒ F=E; mag=0; digit-identical to fixed-occ supercell parent) |
| DFPT C[(0x),(0x)] (regression settings) | **21.1911 eV/Å²** → gate `\|Δ\| ≤ 0.30` vs FD pin (~0.04 %); identity vs fixed-occ parent 21.1911 → gate `\|Δ\| ≤ 0.05` |
| ATV ω(X) [cm⁻¹] | **−108.83, −108.32, 239.12, 451.33, 668.84, 668.90** |
| \|ω\| sorted | 109, 109, 239, 451, 669, 669 → gates [50, 400] (low 3), [300, 850] (high 3) |
| n_imag | 2 (soft TA pair under minimal ecut/k — accepted ≤ 2, same as both finite-q GGA parents) |
| Pre-symmetrization asymmetry | 9.50e-6 eV/Å² → gate ≤ 1e-4 |
| Regression wall | ≈ **148 s** serial (family 60 s aspirational exceeded by spin+smear doubling of the already-tight finite-q GGA twin; fair ratio is the hard perf gate) |

Cross-parent consistency (\(M=0\) gapped-FD limit): joint spin×GGA×smear
DFPT C00 21.1911 = fixed-occ spin×GGA finite-q parent 21.1911 (4-decimal
identical); spectrum matches the parent to the soft-TA digit. The four-way
product adds no spurious response relative to the fixed-occ joint kernel
while exercising the full spin-resolved supercell Sternheimer + `step_spin`
FD + FD occupation weights path.

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **false** |
| ATV wall | **148.1 s** (serial, OMP=1; 12 supercell responses × 2 spins, Γ-only mesh, spin-GGA kernel FD + smear weights) |
| CP2K wall (same observable, comparable job: UKS k-point force evals + FD Hessian + SMEAR on the 4-atom supercell; committed input, serial 1 MPI × OMP=1) | **585.9 s** |
| `ratio_atv_over_cp2k` | **0.25** (148.1 / 585.9) |
| `within_10x` | **true** |
