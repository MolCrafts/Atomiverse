# Reference — finite-q DFPT phonons with joint collinear spin × pure GGA (diamond Si at X, PW)

## Method

Density-functional perturbation theory at finite phonon wavevector
\(\mathbf{q}=X=(1/2,0,0)\) for the concrete plane-wave method `cpu::pw::PW`
with the **joint** collinear-spin × pure-GGA response kernel
(`spin_polarized = true`, `multiplicity = 1` ⇒ \(M=0\),
`xc = XCType::PBE`). Commensurate product path: rational \(\mathbf{q}\) is
evaluated as Γ-point DFPT on the 2×1×1 supercell (self-consistent
spin-resolved Sternheimer, same engine as `dfpt-gga-spin-si`) followed by
IFC Fourier (`PhononDispersion::frequencies_at_q`).
\(K_{\mathrm{xc}}^{\sigma\sigma'}\) is the central FD of the spin-GGA
`XCBuilder::step_spin` operator the ground-state SCF converged — the triple
product of the two shipped joint parents
`regressions/response/dfpt-gga-spin-si` (Γ, spin × PBE) and
`regressions/response/dfpt-gga-finite-q-si` (X supercell, restricted PBE).

```cpp
cpu::pw::PW pw;                  // ecut 10 Ry, k_mesh 1, tol 1e-8
pw.xc = cpu::scf::XCType::PBE;   // pure GGA kernel
pw.spin_polarized = true;        // collinear, M = 0
pw.multiplicity = 1;
cpu::pw::DfptPhonon dfpt;
dfpt.q_frac = {0.5, 0.0, 0.0};   // X → 2×1×1 supercell Γ-DFPT
auto r = dfpt.compute(pw, frame);
```

Cite: Baroni, de Gironcoli, Dal Corso, Giannozzi, *Rev. Mod. Phys.* **73**,
515 (2001); Giannozzi et al., *Phys. Rev. B* **43**, 7231 (1991); Perdew,
Burke, Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996).

## System (regression twin)

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si, 2-atom fcc primitive cell, a = **5.43 Å** (experimental) |
| \(\mathbf{q}\) | \(X=(1/2,0,0)\) → **2×1×1** supercell (4 atoms) |
| Pseudopotential | `tests/data/pw/Si_lda_nc.upf` (ONCVPSP NC, in-tree; same GGA-smoke PP as `pw-pbe-gga-si`) |
| XC | `XCType::PBE` (true spin-GGA kernel) |
| Spin | collinear, multiplicity = 1 (\(N_\uparrow=N_\downarrow=8\) on the supercell) |
| Cutoff | **10 Ry** (minimal-twin budget) |
| k-mesh | **1×1×1** on the supercell (Γ+X of primitive folded) |
| SCF | tol 1e-8 (density L2), linear mix on spin channels |
| Response | `response_tol` 1e-4, `mix_beta` 0.25, `cg_tol` 1e-8, max_iter 120 |

## Offline pins (no live CP2K at regression time)

| Field | Value |
| ----- | ----- |
| Oracle 1 (physics anchor) | **CP2K 2026.1** (rev `85b8a9b49c`, `~/.local/bin/cp2k`, this host, 2026-07-31) — `VIBRATIONAL_ANALYSIS` (FD of analytic UKS k-point forces, `FULLY_PERIODIC T`, DX 0.01 bohr, `EPS_SCF 1e-8`) on the **2×1×1 supercell** (4 atoms): LSD + `MULTIPLICITY 1` + PBE, DZVP-MOLOPT-SR-GTH / GTH-PBE-q4, CUTOFF 300 Ry, 1×2×2 MP (≡ 2×2×2 on the primitive). Input: `tools/cp2k_oracle/inputs/dfpt-gga-spin-finite-q-si/si4_uks_pbe_vib.inp` |
| Oracle 1 output lines | `VIB\|Frequency (cm^-1)  9.7979171441E+01 1.3126605303E+02 3.7058085178E+02` / `4.2823632737E+02 4.5748022016E+02 4.5868391248E+02` / `5.3831837242E+02 5.3831998162E+02 5.4364489119E+02` — supercell Γ FD folds primitive Γ ∪ X; eigenvector phases identify modes 1–6 (cells out of phase = X branch) → **X: 97.98, 131.27 (TA, FD-split doublet), 370.58, 428.24 (LA/LO), 457.48, 458.68 (TO) cm⁻¹**; modes 7–9 = Γ optical 538.32, 538.32, 543.64. Digit-identical to the restricted supercell twin at \(M=0\). |
| Pin 2 (sharp, same-Hamiltonian) | central **second difference of the collinear-spin PBE PW SCF total energy on the 2×1×1 supercell**: C[(0x),(0x)] = (E(+h) − 2E₀ + E(−h))/h², h = 0.03 Å, SCF tol 1e-9, identical ecut/k/XC/UPF/spin |
| Pin 3 (cross-parent M=0 identity) | joint C00 and \|ω(X)\| must reproduce the restricted finite-q GGA parent (`dfpt-gga-finite-q-si`: C00 21.1974 eV/Å² class, \|ω\| ≈ 108, 108, 239, 451, 669, 669 cm⁻¹) to spin-doubling roundoff |

The CP2K twin necessarily differs in PP/basis representation (GPW GTH/MOLOPT
FD-supercell at 300 Ry vs the LDA-generated NC-UPF plane-wave twin at a
10 Ry minimal cutoff with a Γ-only supercell mesh), so it anchors the
**physics window**, not the digit. The same \|ω\| windows the finite-q
parents use — low 3 ∈ [50, 400], high 3 ∈ [300, 850] cm⁻¹ — contain both
the CP2K X branch (98, 131, 371 \| 428, 457, 459) and the biased ATV minimal
twin (109, 109, 239 \| 451, 669, 669). The energy-FD pin is the sharp
correctness gate for the joint spin-GGA supercell kernel.

### Measured pins (this host, 2026-07-31, Release)

| Quantity | Value |
| -------- | ----- |
| Energy-FD C[(0x),(0x)] (h = 0.03 Å, spin-PBE SCF on supercell, tol 1e-8, mix_beta 0.3) | **21.2006 eV/Å²** (E₀ = −15.121055565028 Ha, E± = −15.120904278764 / −15.120505654960 Ha; digit-identical to restricted supercell parent) |
| DFPT C[(0x),(0x)] (regression settings) | **21.1911 eV/Å²** → gate `\|Δ\| ≤ 0.30` vs FD pin (~0.04 %); identity vs restricted parent 21.1974 → gate `\|Δ\| ≤ 0.05` |
| ATV ω(X) [cm⁻¹] | **−108.83, −108.32, 239.12, 451.33, 668.84, 668.90** |
| \|ω\| sorted | 109, 109, 239, 451, 669, 669 → gates [50, 400] (low 3), [300, 850] (high 3) |
| n_imag | 2 (soft TA pair under minimal ecut/k — accepted ≤ 2, same as both parents) |
| Pre-symmetrization asymmetry | 9.50e-6 eV/Å² → gate ≤ 1e-4 |
| Regression wall | ≈ **108 s** serial / **101 s** OMP=8 (family 60 s aspirational exceeded by spin doubling of the already-tight finite-q GGA twin; fair ratio is the hard perf gate) |

Cross-parent consistency (\(M=0\) limit): joint spin×GGA DFPT C00
21.1911 ≈ restricted parent 21.1974 (6e-3 eV/Å²); spectrum matches the
restricted finite-q GGA twin to the soft-TA digit. The joint kernel adds no
spurious spin response at \(M=0\) while exercising the full spin-resolved
supercell Sternheimer + `step_spin` FD path.

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **false** |
| ATV wall | **107.9 s** (serial, OMP=1; 12 supercell responses × 2 spins, Γ-only mesh, spin-GGA kernel FD) |
| CP2K wall (same observable, comparable job: 25 UKS k-point force evals, FD Hessian; committed input, serial 1 thread) | **577.8 s** |
| `ratio_atv_over_cp2k` | **0.19** (107.9 / 577.8) |
| `within_10x` | **true** |
