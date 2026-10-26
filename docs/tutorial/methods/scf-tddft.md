# Linear-response TDDFT and spin-flip response

> The regression runner has been removed. Regression commands below are historical; case sources and oracle records are retained as archived evidence, not current executable checks.

Linear-response time-dependent DFT computes vertical excitation energies from
the Casida response of a converged Kohn–Sham (or Hartree–Fock) reference. ATV
exposes each response flavour as a class template wrapping the converged
ground-state method (`tdhf.h`), returning `ExcitedStatesResult` (`energy_ev`
[eV], `oscillator_strength` (length gauge), and the `tda` / `alda` flags that
record which path ran):

- `cpu::scf::TDA<RKS>{rks}.run(n, ExcitationChannel::singlet)` — Tamm–Dancoff
  (A only), including hybrid exact-exchange kernels;
- `cpu::scf::TDDFT<RKS>{rks}.run(n, ExcitationChannel::singlet)` and
  `cpu::scf::TDDFT<UKS>{uks}.run(n)` — full Casida A/B (RPA);
- `cpu::scf::SfTDA<UHF>{uhf}.run(n)` and `cpu::scf::SfTDHF<UHF>{uhf}.run(n)` —
  collinear spin-flip TDA (SF-CIS) and full spin-flip RPA (SF-TDHF).

Parameters are constructor arguments / members of the SCF object
(`RKS{basis, xc}`, `UKS{basis, n_alpha, n_beta, xc}`, `tol`, `max_iter`,
`grid_level`); there is no public `*Config`. The sections cover CPU only:
closed-shell H₂/STO-3G (B3LYP TDA and RPA, PBE RPA) and triplet O₂/STO-3G
(UKS-LSDA RPA, SF-RPA, SF-TDA). The case also runs sections `ch2o_tdhf`,
`h2_full_grid_alda` and `uks-tda-o2-lsda`, whose pins live in the case
`REFERENCE.md`.

Case: `ctest -R '^scf/tddft$'` — sources and oracle record in [`regressions/scf/tddft/`](../../../regressions/scf/tddft/). One section of the case per `##` heading below: `atv_regressions --case scf/tddft --section <name>`.

## Common setup

### Theory

Full linear response couples excitations \(\mathbf{X}\) and de-excitations
\(\mathbf{Y}\):

$$
\begin{pmatrix} \mathbf{A} & \mathbf{B} \\ \mathbf{B}^* & \mathbf{A}^* \end{pmatrix}
\begin{pmatrix} \mathbf{X} \\ \mathbf{Y} \end{pmatrix}
= \omega
\begin{pmatrix} \mathbf{X} \\ \mathbf{Y} \end{pmatrix}.
$$

The Tamm–Dancoff approximation drops **B** and solves
\(\mathbf{A}\,\mathbf{X} = \omega\,\mathbf{X}\). For pure functionals the
kernel \(f^{xc}\) is semi-local (adiabatic); hybrids scale HF-exchange blocks
by \(\alpha_{\mathrm{hf}}\) in **A** (and **B** for RPA). For real MOs and pure
DFT, \(A-B=\mathrm{diag}(\varepsilon_a-\varepsilon_i)\) and the dense path
solves the Hermitian reduction
\((\mathbf{A}-\mathbf{B})^{1/2}(\mathbf{A}+\mathbf{B})(\mathbf{A}-\mathbf{B})^{1/2}\mathbf{Z}=\omega^{2}\mathbf{Z}\)
(`src/cpu/scf/tdhf.cpp`; A/B and \(f^{xc}\) in the private
`internal/orbital_hessian.h`). Full RPA
lies below TDA on these systems when de-excitation is kept.

### Systems

- **H₂** (closed shell): \(r(\mathrm{H–H})=0.74\) Å, `Box::cubic(20.0)`.
  The oracle uses H(0,0,0), H(0,0,0.74); the regressions place the bond on +x
  (same \(r=0.74\) Å). One singles channel: \(n_{\mathrm{occ}}=1\),
  \(n_{\mathrm{virt}}=1\), \(n_{\mathrm{ov}}=1\). SCF `tol = 1e-10`,
  `max_iter = 100`.
- **O₂** (triplet, \(2S=2\)): O(0,0,0), O(0,0,1.2075) Å, Z=8, cubic box
  20 Å; \((n_\alpha, n_\beta) = (9, 7)\). UHF/UKS on O₂ has multiple SCF
  minima, so every O₂ section warm-starts from a restricted open-shell density
  (same pattern as [ump2-o2-sto3g](scf-mp2.md#ump2-o2-sto3g)):

```cpp
cpu::scf::ROHF rohf(cpu::scf::BasisSet::STO3G, /*n_alpha=*/9, /*n_beta=*/7);
rohf.tol = 1e-12;
rohf.max_iter = 200;
rohf.run(frame);

using Spin = cpu::scf::SlaterDeterminant::Spin;
cpu::scf::SCFInitialGuess guess;
guess.P_alpha = rohf.determinant().density(Spin::alpha);
guess.P_beta  = rohf.determinant().density(Spin::beta);
guess.has_alpha = guess.has_beta = true;

cpu::scf::UHF uhf(cpu::scf::BasisSet::STO3G, 9, 7);
uhf.tol = 1e-12;
uhf.max_iter = 200;
uhf.run(frame, &guess);
```

Build with `-DATV_BUILD_HF=ON -DATV_BUILD_REGRESSIONS=ON` (CPU login-shell
recipe in `docs/developer/project-contract.md`, `module load buildenv-gcc/2026.03-mpich`).

### Oracle

PySCF **2.14.0**, offline pins only — PySCF is never imported by CMake/ctest;
protocol in `docs/tutorial/appendix-pyscf-oracle.md`. Common settings: STO-3G,
Cartesian (`cart=True`; no \(d\) shells, so cart/sph is a no-op), SCF
`conv_tol=1e-12`, Ha→eV 27.211386245988 (CODATA ≡ `units::kHartreeInEv`, so
the gate does not mix conversion constants). Ground-state energies are a
sanity gate at \(1\times 10^{-4}\) Ha; oscillator strengths, where pinned, use
a generous 0.15 band.

Fair wall: medians of ≥3 runs, \(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}}\),
gate \(r\le 10\):

```bash
python <pyscf fair-bench> --method <twin> --geom <h2|o2> --basis sto-3g \
  [--spin 2] [--xc <xc>] --nstates <n> --repeat <3|5> -- <section binary>
```

The regression binary includes pin checks and guards while the PySCF side is
the twin kernel only; these micro-systems do not stress dense RPA scaling.

### Shared limitations

- Dense solvers only; no automatic Davidson fallback on these surfaces.
- No restricted excitation windows, CVS or real-time TDDFT.
- Analytic CIS/TDA forces and NACs live on `RHF`; no analytic gradients/NACs
  for the hybrid, UKS or spin-flip response paths here.
- CPU only; roadmap in
  [`docs/roadmap/excited-states.md`](../../roadmap/excited-states.md).

### References

- M. E. Casida, in *Recent Advances in Density Functional Methods*, Part I
  (World Scientific, 1995) — Casida equations (A,B).
- S. Hirata, M. Head-Gordon, *Chem. Phys. Lett.* **314**, 291 (1999) — TDA vs
  full.
- A. Dreuw, M. Head-Gordon, *Chem. Rev.* **105**, 4009 (2005) — TDDFT/TDHF
  response review.
- Q. Sun et al., *WIREs Comput. Mol. Sci.* **8**, e1340 (2018) — PySCF.
- Y. Shao, M. Head-Gordon, A. I. Krylov, *J. Chem. Phys.* **118**, 4807
  (2003) — SF-TDDFT/CIS (spin-flip sections).

## h2-b3lyp-hybrid-tda — Hybrid TDDFT-TDA (B3LYP), H₂ / STO-3G {#h2-b3lyp-hybrid-tda}

Tamm–Dancoff response on a **hybrid** RKS reference: Casida **A** includes the
exact-exchange kernel \(\alpha_{\mathrm{hf}}\) plus semi-local \(f^{xc}\).
Pure ALDA-TDA is not enough for hybrids, because B3LYP mixes HF exchange into
both the ground-state Fock and the response kernel. Related:
[Ch. 6 Hybrid DFT](../ch06-hybrid-dft.md); full RPA on the same reference is
[h2-b3lyp-rpa-tddft](#h2-b3lyp-rpa-tddft).

```cpp
cpu::scf::RKS rks(cpu::scf::BasisSet::STO3G, cpu::scf::XCType::B3LYP);
rks.tol = 1e-10;
rks.max_iter = 100;
rks.run(f);
const cpu::scf::ExcitedStatesResult es =
    cpu::scf::TDA<cpu::scf::RKS>{rks}.run(1, cpu::scf::ExcitationChannel::singlet);
// es.energy_ev[0] [eV]; es.oscillator_strength[0]; es.tda == true
```

### Pins

| Quantity | Pin | Tol |
| -------- | --: | --- |
| \(E_{\mathrm{RKS}}\) (B3LYP) | −1.165418410673 Ha | 1e-4 Ha (agreement ~1e-8) |
| S₁ (TDA) | **25.92836482 eV** (0.952849832367 Ha) | **0.05 eV** |
| \(f\) (length) | 1.100141468478 | 0.15 |

The 0.05 eV band is an end-to-end sanity band for the hybrid exact-exchange
kernel path, not a milli-eV acceptance test.

### Performance

| Twin | \(T_{\mathrm{PySCF}}\) | \(T_{\mathrm{ATV}}\) | \(r\) | ≤10× |
| ---- | ---------------------: | -------------------: | ----: | ---- |
| `rks_tda` B3LYP | 3.68 s | 4.66 s | **1.27** | ✓ |

## h2-b3lyp-rpa-tddft — Hybrid full-LR TDDFT (B3LYP RPA), H₂ / STO-3G {#h2-b3lyp-rpa-tddft}

Closed-shell full Casida **A,B** response on RKS/B3LYP. Unlike hybrid TDA,
de-excitation (**B**) is retained; the exact-exchange fraction
\(\alpha_{\mathrm{hf}}\) scales HF-exchange blocks on **both** A and B, and the
adiabatic semi-local \(f^{xc}\) comes from a finite difference of \(V_{xc}\) on
the KS grid, so `es.tda == false` and `es.alda == false`. Dense path,
\(n_{\mathrm{ov}}\le 512\) (larger spaces throw). The finite-difference
\(f^{xc}\) is calibrated for the singlet channel; the triplet non-LSDA path
omits that kernel. Closed-shell only: UKS full-LR is
[uks-full-lr-o2-lsda](#uks-full-lr-o2-lsda) (pure LSDA); open-shell hybrids and
spin-flip full TDDFT are not on this path.

```cpp
const cpu::scf::ExcitedStatesResult es =
    cpu::scf::TDDFT<cpu::scf::RKS>{rks}.run(1, cpu::scf::ExcitationChannel::singlet);
```

### Pins

Oracle: RKS/B3LYP + `tdscf.TDDFT` (full RPA, not TDA), `nstates=3` requested
(1 physical root); repro
`python <pyscf pin> rks-tddft --geom h2 --basis sto-3g --xc b3lyp --nstates 3 --conv-tol 1e-12`.

| Quantity | Pin | Tol | Notes |
| -------- | --: | --- | ----- |
| \(E_{\mathrm{RKS}}\) [Ha] | **−1.165418410673** | \(1\times 10^{-4}\) Ha | ground-state sanity |
| S₁ [eV] (primary) | **25.30779734533265** | **\(1\times 10^{-4}\) eV** | gate |
| S₁ [Ha] | **0.9300443981998157** | \(\approx 3.7\times 10^{-6}\) Ha | consistency via Ha→eV |
| \(f\) (length) | **0.8608891208128704** | 0.15 | optional band |
| `tda` / `alda` | false / false | — | regression rejects TDA / ALDA flags |
| Guard | ≠ 25.928364822972 eV | — | must not match the hybrid-TDA pin |

Context (same geometry/basis; not gates): CIS/RHF-TDA ≈ 25.81 eV; B3LYP hybrid
TDA ≈ 25.93 eV; **B3LYP full RPA ≈ 25.31 eV** — S₁ ≈ 25.928 eV (TDA) vs
≈ 25.308 eV (full RPA).

### Performance

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) |
| ---- | -------------------------: | -----------------------: | ----: |
| `rks_tddft` B3LYP H₂/STO-3G | 0.172 | 0.074 | **0.430** |

`OMP_NUM_THREADS=1` both sides; `--xc b3lyp --nstates 3 --repeat 3
--max-ratio 10.0`. This hybrid RPA twin is not interchangeable with the TDA
twin used for [full-lr-tddft-h2-pbe](#full-lr-tddft-h2-pbe) (\(r\approx 0.27\)).

### References

- A. D. Becke, *J. Chem. Phys.* **98**, 5648 (1993) — B3LYP hybrid form (PySCF
  `xc='B3LYP'`).

## full-lr-tddft-h2-pbe — Full linear-response TDDFT (PBE), H₂ / STO-3G {#full-lr-tddft-h2-pbe}

Casida **A/B** (RPA) response on pure-GGA RKS (PBE), beyond TDA; \(f^{xc}\) is
semi-local without \(\alpha_{\mathrm{hf}}\). Hybrid TDA:
[h2-b3lyp-hybrid-tda](#h2-b3lyp-hybrid-tda).

```cpp
cpu::scf::RKS rks(cpu::scf::BasisSet::STO3G, cpu::scf::XCType::PBE);
rks.tol = 1e-10;
rks.max_iter = 100;
rks.run(f);
const cpu::scf::ExcitedStatesResult es =
    cpu::scf::TDDFT<cpu::scf::RKS>{rks}.run(1, cpu::scf::ExcitationChannel::singlet);
```

### Pins

Oracle: PySCF `TDDFT` / PBE on H₂ 0.74 Å.

| Quantity | Pin | Tol | Notes |
| -------- | --: | --- | ----- |
| \(E_{\mathrm{RKS}}\) | −1.152072795159 Ha | 1e-4 Ha | ground-state sanity |
| S₁ (full LR) | **25.519450223124366 eV** (0.937822497995188 Ha) | **1e-4 eV** | gate |
| \(f\) (length) | 0.8583699347665188 | 0.15 | optional band |
| Guard | ≠ 26.210883282417658 eV | — | must not match the PBE-TDA S₁ |

### Performance

| Twin | \(T_{\mathrm{PySCF}}\) | \(T_{\mathrm{ATV}}\) | \(r\) | ≤10× |
| ---- | ---------------------: | -------------------: | ----: | ---- |
| `rks_tda` PBE | 5.46 s | 1.45 s | **0.27** | ✓ |

The twin is TDA (conservative for comparing full RPA cost).

## uks-full-lr-o2-lsda — UKS full-LR TDDFT (Casida A/B), O₂ / LSDA / STO-3G {#uks-full-lr-o2-lsda}

Open-shell spin-conserving full Casida response on a UKS/LSDA triplet O₂
reference (Slater exchange + VWN5 correlation; `XCType::LSDA` ≡ PySCF
`xc="LDA,VWN"`):

\[
A_{ia\sigma,jb\sigma'}
=
\delta_{\sigma\sigma'}\delta_{ij}\delta_{ab}
(\varepsilon_a^\sigma-\varepsilon_i^\sigma)
+(ia\sigma|jb\sigma')
+(ia\sigma|f^{\mathrm{xc}}_{\sigma\sigma'}|jb\sigma'),
\qquad
B_{ia\sigma,jb\sigma'}
=
(ia\sigma|bj\sigma')
+(ia\sigma|f^{\mathrm{xc}}_{\sigma\sigma'}|jb\sigma'),
\]

with pure DFT \(\alpha_{\mathrm{hf}}=0\) and adiabatic spin-resolved
\(f^{\mathrm{xc}}\) from LSDA second derivatives on the KS grid (PySCF
`tdscf.TDDFT` on `dft.UKS`). The Coulomb/fxc blocks of **A** and **B**
coincide for real MOs, so the Hermitian Casida reduction applies (same path as
`TDDFT<RKS>`). Observable: the five lowest Ms-preserving roots;
`es.tda == false`, `es.alda == true`. Pure LSDA only — no hybrid UKS full-LR.

The section uses ROKS → UKS (`XCType::LSDA`, `grid_level =
GridLevel::MatchPySCF` on both) in place of the ROHF → UHF warm-start shown in
Common setup:

```cpp
cpu::scf::UKS uks(cpu::scf::BasisSet::STO3G, 9, 7, cpu::scf::XCType::LSDA);
uks.tol = 1e-12;
uks.max_iter = 200;
uks.grid_level = cpu::scf::GridLevel::MatchPySCF;
uks.run(frame, &guess);                     // guess from ROKS/LSDA
const cpu::scf::ExcitedStatesResult es = cpu::scf::TDDFT<cpu::scf::UKS>{uks}.run(5);
```

### Pins

Oracle: UKS/LSDA (`LDA,VWN`) + `tdscf.TDDFT`, `nstates=5`; repro
`python <pyscf pin> uks-tddft --geom o2 --basis sto-3g --spin 2 --xc LDA,VWN --nstates 5 --conv-tol 1e-12`.

| Quantity | Pin | Tol | Notes |
| -------- | --: | --- | ----- |
| \(E_{\mathrm{UKS}}\) [Ha] | **−147.193805922283** | \(1\times 10^{-4}\) Ha | ground-state sanity; S²≈2.000202 |
| ω₀ [eV] | **7.620917323839** | **\(1\times 10^{-4}\) eV** | gate |
| ω₁ [eV] | **7.642419562717** | **\(1\times 10^{-4}\) eV** | gate |
| ω₂ [eV] | **7.642420073101** | **\(1\times 10^{-4}\) eV** | gate (near-degenerate with ω₁) |
| ω₃ [eV] | **8.386321248270** | **\(1\times 10^{-4}\) eV** | gate |
| ω₄ [eV] | **8.386321248273** | **\(1\times 10^{-4}\) eV** | gate (degenerate with ω₃) |
| `tda` / `alda` | false / true | — | regression rejects TDA flag; requires ALDA on LSDA |
| Guard | ≠ TDA root 3 **8.403285** eV | — | must not match `uks-tda-o2-lsda` root 3 |

Contrast with UKS TDA (same geometry/basis/XC; not a gate): full RPA leaves
root 0 essentially unchanged and **lowers** roots 3–4 by ~0.017 eV through
**B**:

| Theory | root 0 [eV] | root 3 [eV] |
| ------ | ----------: | ----------: |
| **UKS full RPA TDDFT** | **7.620917** | **8.386321** |
| UKS TDA | 7.620917 | 8.403285 |

### Performance

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) |
| ---- | -------------------------: | -----------------------: | ----: |
| `uks_tddft` LSDA O₂/STO-3G | 0.229 | 0.672 | **2.937** |

`OMP_NUM_THREADS=1` both sides; median of 3 runs; `perf_exempt=false`. The
regression binary includes the ROKS warm-start, pin checks and TDA guard.

## sf-tddft-o2 — Collinear spin-flip full TDDFT (SF-RPA), O₂ UHF / STO-3G {#sf-tddft-o2}

`SfTDHF<UHF>{uhf}.run(n_states)` (after `UHF::run`) solves collinear
spin-flip full linear response (SF-TDHF / SF-RPA) on the high-spin UHF
reference. The response space is the same Ms-decreasing singles manifold as
SF-TDA (majority occupied → minority virtual; for \(n_\alpha\ge n_\beta\)):

\[
\{i_\alpha \to a_\beta\},\qquad
n_{\mathrm{ov}} = n_\alpha^{\mathrm{occ}}\,n_\beta^{\mathrm{virt}}
\]

(STO-3G O₂: \(9\times 3 = 27\)). Coulomb vanishes for pure spin-flip; with the
collinear HF kernel only exchange remains (Shao, Head-Gordon & Krylov 2003):

\[
A_{ia,jb}
=
\delta_{ij}\delta_{ab}
\bigl(\varepsilon_a^\beta - \varepsilon_i^\alpha\bigr)
- (ij|ab),
\qquad
B_{ia,jb}
=
-(ib|aj),
\]

with \(i,j\) majority-occupied and \(a,b\) minority-virtual. The non-Hermitian
RPA

\[
\begin{bmatrix}A&B\\-B&-A\end{bmatrix}
\begin{bmatrix}X\\Y\end{bmatrix}
=
\omega
\begin{bmatrix}X\\Y\end{bmatrix}
\]

is solved densely (`dgeev` on the \(2n_{\mathrm{ov}}\) matrix). On this
reference \((A-B)\) is **not** positive definite
(\(\lambda_{\min}(A-B)\approx -0.46\) Ha), so the Hermitian Casida reduction
used by closed-shell `TDDFT<RKS>` is undefined; complex / non-positive poles can
appear in general open-shell SF-RPA, and the result reports the lowest
**positive real** roots [eV] (`tda == false`, `alda == false`). Oscillator
strengths are zero (spin-forbidden from the high-spin reference).

Scope: HF-exchange kernel only — no SF-TDDFT XC kernel, no noncollinear
spin-flip, no UKS SF path; hard cap \(n_{\mathrm{ov}}\le 256\) (doubled space
≤ 512, larger throws). Compared with [spin-flip-tda-o2](#spin-flip-tda-o2) on
the same reference, the \(B\) block reorders the spectrum: the near-zero
multiplet roots of SF-TDA are no longer the lowest positive RPA poles, so pins
are not interchangeable between the two methods.

### Pins

PySCF mainline has no product SF-TDDFT; the twin rebuilds \(A,B\) from PySCF
2.14.0 UHF MOs with `ao2mo` (`pin_scf.py sf-tddft` / the verbatim script in the
case `REFERENCE.md`). Gate \(\lvert\omega_i - \mathrm{pin}_i\rvert < 10^{-4}\)
eV on five roots; \(n_{\mathrm{ov}}=27\).

| Quantity | Pin | Tol | Notes |
| -------- | --: | --- | ----- |
| \(E_{\mathrm{UHF}}\) [Ha] | −147.633946785450 | \(10^{-4}\) Ha (sanity) | \(S^2\approx 2.003411\) |
| \(a_{\min}(A-B)\) [Ha] | −0.461612348195 | (diagnostic) | Casida Hermitian form undefined |
| \(\omega_0\) [eV] | 4.334691644974 | \(10^{-4}\) eV | lowest positive SF-RPA root |
| \(\omega_1\) [eV] | 7.155381562193 | \(10^{-4}\) eV | degenerate pair with \(\omega_2\) |
| \(\omega_2\) [eV] | 7.155381562193 | \(10^{-4}\) eV | |
| \(\omega_3\) [eV] | 8.181367981325 | \(10^{-4}\) eV | degenerate pair with \(\omega_4\) |
| \(\omega_4\) [eV] | 8.181367981325 | \(10^{-4}\) eV | |

SF-TDA roots on the same reference (0.045 / 1.568 / 1.568 / 3.038 / 7.849 eV)
are **not** interchangeable with these SF-RPA poles.

### Performance

Median of 5 walls, `OMP_NUM_THREADS=1 OMP_PROC_BIND=close`; fair twin = ROHF
warm-start → UHF + dense SF \(A,B\) non-Hermitian RPA (`--method sf_tddft
--spin 2 --nstates 5 --repeat 5`).

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) |
| ---- | -------------------------: | -----------------------: | ----: |
| O₂ UHF/STO-3G SF-RPA (`sf_tddft`) | **0.0412** | **0.392** | **9.51** |

The twin is custom SF \(A,B\) + `eigvals`, not a product PySCF SF-TDDFT;
micro-system walls are noisy and ATV runs the full ROHF + UHF + SF path in the
binary. \(r\approx 9.51\) passes the 10× gate but is near the ceiling; SCF/ERI
cost dominates. Open roadmap axes: SF XC kernels, iterative solvers / larger
bases, CUDA.

## spin-flip-tda-o2 — Collinear spin-flip TDA (SF-CIS), O₂ UHF / STO-3G {#spin-flip-tda-o2}

`SfTDA<UHF>{uhf}.run(n_states)` (after `UHF::run`) solves collinear spin-flip
Tamm–Dancoff (SF-CIS / SF-TDA) on the high-spin UHF reference: Ms-decreasing
singles \(i_\alpha\to a_\beta\) (or \(\beta\to\alpha\) when
\(n_\beta>n_\alpha\)), same space and exchange-only \(A\) as
[sf-tddft-o2](#sf-tddft-o2), with no \(B\) block. Dense Hermitian solve
\(A X = \omega X\) (`eigh` of full \(A\)), hard cap \(n_{\mathrm{ov}}\le 512\)
(larger throws); `tda == true`, `alda == false`. Use for open-shell multiplet
reconstruction and spin-flip roots when a product SF-TDDFT path is not
required. Root 0 may sit near zero (Ms multiplet reconstruction under mild UHF
spin contamination) — not a spectroscopic "first excited state" without
analysis. No SF-TDDFT XC kernel, no noncollinear SF, no UKS SF path; energies
only.

```cpp
const cpu::scf::ExcitedStatesResult es =
    cpu::scf::SfTDA<cpu::scf::UHF>{uhf}.run(/*n_states=*/5);
```

### Pins

PySCF mainline has no product SF-TDA; the twin rebuilds \(A\) from PySCF
2.14.0 UHF MOs + `ao2mo` SF-CIS matrix (`pin_scf.py sf-tda` / the verbatim
script in the case `REFERENCE.md`). Gate
\(\lvert\omega_i - \mathrm{pin}_i\rvert < 10^{-4}\) eV on five roots.

| Quantity | Pin | Tol | Notes |
| -------- | --: | --- | ----- |
| \(E_{\mathrm{UHF}}\) [Ha] | −147.633946785450 | \(10^{-4}\) Ha (sanity) | \(S^2\approx 2.003411\) |
| \(\omega_0\) [eV] | 0.045146996469 | \(10^{-4}\) eV | near-zero multiplet / contamination |
| \(\omega_1\) [eV] | 1.567707515056 | \(10^{-4}\) eV | degenerate pair with \(\omega_2\) |
| \(\omega_2\) [eV] | 1.567707515056 | \(10^{-4}\) eV | |
| \(\omega_3\) [eV] | 3.038448959259 | \(10^{-4}\) eV | |
| \(\omega_4\) [eV] | 7.848628442263 | \(10^{-4}\) eV | |

Related: EOM-SF-CCSD on the same molecule,
[eom-sf-ccsd-o2](scf-eom-cc.md#eom-sf-ccsd-o2).

### Performance

Median of 5 walls, `OMP_NUM_THREADS=1 OMP_PROC_BIND=close`; fair twin = ROHF
warm-start → UHF + dense SF-CIS \(A\) (`--method sf_tda --spin 2 --nstates 5
--repeat 5`).

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) |
| ---- | -------------------------: | -----------------------: | ----: |
| O₂ UHF/STO-3G SF-CIS (`sf_tda`) | **0.0326** | **0.311** | **9.56** |

The twin is custom SF-CIS \(A\), not a product PySCF SF-TDA; micro-system walls
are noisy and ATV runs the full ROHF + UHF + SF path in the binary.
\(r\approx 9.56\) passes the 10× gate but is near the ceiling; SCF/ERI cost
dominates. Roadmap: SF-TDDFT (XC kernel) ○, full SF-LR, larger-basis /
iterative solvers, CUDA.
