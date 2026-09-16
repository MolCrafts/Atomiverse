# Reference — collinear spin-polarized PW band structure (bulk silicon)

## Method

Collinear spin-polarized plane-wave Kohn–Sham DFT with LDA (Slater + PW92)
on the PW real-space grid (`spin_polarized = true`, `multiplicity = 1` ⇒
\(M = N_\uparrow - N_\downarrow = 0\)). Norm-conserving ONCVPSP LDA Si UPF.
Self-consistent spin densities from a uniform Monkhorst–Pack mesh, then
**non-self-consistent** diagonalization of each spin channel on a frozen
\(V_{\mathrm{eff},\sigma}\) along a high-symmetry **k-path**.

Public path:

```cpp
cpu::pw::PW method;
method.spin_polarized = true;
method.multiplicity = 1;
method.upf_path = "..."; method.ecut_ry = 15; method.k_mesh = 2;
method.xc = cpu::scf::XCType::LSDA;

cpu::pw::Bands bands;
bands.n_bands = 8;
bands.set_diamond_path(4);
auto br = bands.compute(method, frame);
// br.nspin == 2; br.eigenvalues_ry (α), br.eigenvalues_beta_ry (β)
```

No public `*Config` bag; no factories.

Sibling closed-shell product: `regressions/pw/bands-dos-si`.

## Pseudopotential

| Item | Value |
| ---- | ----- |
| File | `tests/data/pw/Si_lda_nc.upf` (in-tree) |
| Family | PseudoDojo ONCVPSP-LDA-SR-noCC v0.4 |
| Cite | Hamann, *Phys. Rev. B* **88**, 085117 (2013); van Setten et al., *Comput. Phys. Commun.* **226**, 39 (2018) |

## Literature pin (indirect Γ→X gap)

Bulk Si is an **indirect** gap semiconductor: VBM at \(\Gamma\), CBM near \(X\).

| Quantity | Value | Source |
| -------- | ----- | ------ |
| Experimental \(E_g\) | ~1.12 eV | textbook |
| Converged LDA \(E_g\) (Γ\(_v\)→ near-\(X_c\)) | **~0.45 eV** | Yin & Cohen, *Phys. Rev. B* **26**, 5668 (1982) |

With \(M=0\) collinear occupations the α and β channels are degenerate;
the scientific pins are the same textbook transition and α≈β parity.

## Twin geometry and settings

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si, 2-atom fcc primitive cell |
| Lattice constant \(a\) | 5.43 Å |
| Basis atoms | (0,0,0), \((a/4,a/4,a/4)\) |
| SCF \(E_\mathrm{cut}\) | 15 Ry |
| SCF k-mesh | \(2\times2\times2\) (`PW::k_mesh = 2`) |
| XC | `XCType::LSDA` |
| Spin | collinear, multiplicity = 1 (\(N_\uparrow=N_\downarrow=4\)) |
| SCF tol | \(10^{-6}\) (density L2) |
| Bands | diamond path Γ–X–W–L–Γ–K, `n_seg=4`, `n_bands=8` per spin |

## Acceptance

1. SCF converges; `nspin == 2`.
2. Eigenvalues per (spin, k) sorted ascending; 8 bands each.
3. `|magnetization| ≤ 1e-6`.
4. Every local same-\(k\) gap (α and β, \(n_\mathrm{occ}=4\)) \(> 0\).
5. Path includes \(\Gamma\) and \(X\); \(\varepsilon_c^\alpha(X) < \varepsilon_c^\alpha(\Gamma)\).
6. **Γ→X gap window** (eV, α channel):
   \[
   0.05 \le E_{\Gamma\to X}^\mathrm{ATV} \le 2.50
   \]
   with \(E_{\Gamma\to X}=(\varepsilon_c^\alpha(X)-\varepsilon_v^\alpha(\Gamma))\times 13.605693\).
7. α≈β at Γ VBM and X CBM: \(\lvert\varepsilon^\alpha-\varepsilon^\beta\rvert \le 5\times 10^{-4}\) Ry.

## CP2K note

CP2K Quickstep `BAND_STRUCTURE` with `LSD` / UKS uses **GTH + Gaussian**
(DZVP-GTH), not ONCV plane waves. Absolute \(\varepsilon_n\) are **not**
comparable. Character (indirect Si, \(\varepsilon_c(X)<\varepsilon_c(\Gamma)\)),
\(M=0\), and the Yin–Cohen LDA Γ→X window are the scientific pins.
Offline twin input: `tools/cp2k_oracle/inputs/bands-spin-si.note`.

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K binary / module on the build host for a fair wall twin this shift |
| ATV twin wall (informative) | ~16 s (Release, OMP=8, ecut 15 / k=2 + collinear path, 2026-07-26) |
| `ratio_atv_over_cp2k` | n/a (exempt) |
| `within_10x` | n/a (exempt; correctness gate only) |

When a CP2K binary is available: time the same geometry with the oracle note
input, append \(r = T_\mathrm{atv}/T_\mathrm{cp2k}\) here, require \(r \le 10\).
