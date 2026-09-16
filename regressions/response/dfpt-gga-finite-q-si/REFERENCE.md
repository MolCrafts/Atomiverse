# Reference — finite-q DFPT phonons with pure GGA kernel (diamond Si at X, PW PBE)

## Method

Density-functional perturbation theory at finite phonon wavevector
\(\mathbf{q}=X=(1/2,0,0)\) for the concrete plane-wave method `cpu::pw::PW`
with the **pure GGA (PBE) exchange-correlation kernel**. Commensurate product
path: rational \(\mathbf{q}\) is evaluated as Γ-point DFPT on the 2×1×1
supercell (self-consistent Sternheimer, same engine as `dfpt-gga-si`)
followed by IFC Fourier (`PhononDispersion::frequencies_at_q`) —
analytically equivalent to finite-\(q\) DFPT for Born–von Karman
commensurate \(q\). \(K_{\mathrm{xc}}\) is the second functional derivative
of the same `XCBuilder` GGA operator the ground-state SCF converged (central
FD with spectral \(\nabla\rho\) and GGA flux) — the joint product of the two
shipped single-axis pins `regressions/response/dfpt-finite-q-si`
(commensurate X path, LSDA kernel) and `regressions/response/dfpt-gga-si`
(Γ, PBE kernel).

```cpp
cpu::pw::PW pw;                  // ecut 10 Ry, k_mesh 1, tol 1e-8
pw.xc = cpu::scf::XCType::PBE;   // pure GGA kernel
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
| XC | `XCType::PBE` (true GGA kernel) |
| Cutoff | **10 Ry** (minimal-twin budget) |
| k-mesh | **1×1×1** on the supercell (Γ+X of primitive folded) |
| SCF | tol 1e-8 (density L2) |
| Response | `response_tol` 1e-4, `mix_beta` 0.25, `cg_tol` 1e-8, max_iter 120 |

## Offline pins (no live CP2K at regression time)

| Field | Value |
| ----- | ----- |
| Oracle 1 (physics anchor) | **CP2K 2026.1** (rev `85b8a9b49c`, `~/.local/bin/cp2k`, this host, 2026-07-31) — `VIBRATIONAL_ANALYSIS` (FD of analytic RKS k-point forces, `FULLY_PERIODIC T`, DX 0.01 bohr, `EPS_SCF 1e-8`) on the **2×1×1 supercell** (4 atoms): PBE, DZVP-MOLOPT-SR-GTH / GTH-PBE-q4, CUTOFF 300 Ry, 1×2×2 MP (≡ 2×2×2 on the primitive). Input: `tools/cp2k_oracle/inputs/dfpt-gga-finite-q-si/si4_pbe_vib.inp` |
| Oracle 1 output lines | `VIB\|Frequency (cm^-1)  9.7979172103E+01 1.3126606201E+02 3.7058084483E+02` / `4.2823632486E+02 4.5748021881E+02 4.5868390952E+02` / `5.3831838074E+02 5.3831995025E+02 5.4364491240E+02` — the supercell Γ FD spectrum folds primitive Γ ∪ X; eigenvector phases identify modes 1–6 (cells out of phase = X branch) → **X: 97.98, 131.27 (TA, FD-split doublet), 370.58, 428.24 (LA/LO), 457.48, 458.68 (TO) cm⁻¹**; modes 7–9 = Γ optical 538.32, 538.32, 543.64 |
| Pin 2 (sharp, same-Hamiltonian) | central **second difference of the restricted-PBE PW SCF total energy on the 2×1×1 supercell**: C[(0x),(0x)] = (E(+h) − 2E₀ + E(−h))/h², h = 0.03 Å, SCF tol 1e-9, identical ecut/k/XC/UPF |
| Pin 3 (cross-parent consistency) | PBE spectrum must sit near the LSDA parent (`dfpt-finite-q-si`: \|ω\| 130, 130, 230, 448, 673, 673 cm⁻¹) — an XC-kernel change, not new lattice dynamics |

The CP2K twin necessarily differs in PP/basis representation (GPW GTH/MOLOPT
FD-supercell at 300 Ry vs the LDA-generated NC-UPF plane-wave twin at a
10 Ry minimal cutoff with a Γ-only supercell mesh), so it anchors the
**physics window**, not the digit. The same \|ω\| windows the LSDA parent
uses — low 3 ∈ [50, 400], high 3 ∈ [300, 850] cm⁻¹ — contain both the CP2K
X branch (98, 131, 371 | 428, 457, 459) and the biased ATV minimal twin
(108, 108, 239 | 451, 669, 669). The energy-FD pin is the sharp correctness
gate for the finite-q GGA kernel.

### Measured pins (this host, 2026-07-31, Release)

| Quantity | Value |
| -------- | ----- |
| Supercell SCF energy (4-atom, PBE) | −30.242110831216 Ry (−15.121055415608 Ha) |
| Energy-FD C[(0x),(0x)] (h = 0.03 Å, tol 1e-9) | **21.2006 eV/Å²** (E₀ = −15.121055415608 Ha, E± = −15.120904127510 / −15.120505505958 Ha) |
| DFPT C[(0x),(0x)] (regression settings) | **21.1974 eV/Å²** → gate `\|Δ\| ≤ 0.30` (~0.015 % agreement) |
| ATV ω(X) [cm⁻¹] | **−108.49, −107.98, 239.26, 451.36, 668.88, 668.94** |
| \|ω\| sorted | 108, 108, 239, 451, 669, 669 → gates [50, 400] (low 3), [300, 850] (high 3) |
| n_imag | 2 (soft TA pair under minimal ecut/k — accepted ≤ 2, same as the LSDA parent) |
| Pre-symmetrization asymmetry | 7.30e-6 eV/Å² → gate ≤ 1e-4 |
| Regression wall | ≈ **50 s** (≤ 60 s budget; identical numbers at OMP=1 and OMP=8) |

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **false** |
| ATV wall | **49.8 s** (serial, OMP=1; 12 supercell responses, Γ-only mesh, GGA kernel FD) |
| CP2K wall (same observable, comparable job: 25 RKS k-point force evals, FD Hessian; committed input, serial 1 thread) | **288.7 s** |
| `ratio_atv_over_cp2k` | **0.17** (49.8 / 288.7) |
| `within_10x` | **true** |
