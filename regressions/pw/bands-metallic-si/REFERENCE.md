# Reference — metallic occupation band structure (diamond Si, PW LSDA + FD)

## Method

Spin-unpolarized plane-wave Kohn–Sham DFT with **Fermi–Dirac metallic
occupation smearing** on the SCF density, then **non-self-consistent**
diagonalization on a high-symmetry **k-path** (`cpu::pw::Bands`).

Public path (no `*Config`, no factories):

```cpp
cpu::pw::PW method;
method.smearing = Smearing::fermi_dirac;
method.degauss = 0.02;   // Ry
method.extra_bands = 4;
// … ecut, k_mesh, upf, xc=LSDA …

cpu::pw::Bands bands;
bands.n_bands = 8;
bands.set_diamond_path(4);
const BandsResult br = bands.compute(method, frame);
// br.eigenvalues_ry, br.e_fermi_ry, br.gap_ry, br.energy_ry (free energy F)
```

Sibling insulator bands: `regressions/pw/bands-dos-si`.  
Sibling metallic DOS: `regressions/pw/dos-metallic-si`.  
Sibling spin bands (fixed occ): `regressions/pw/bands-spin-si`.

## Pseudopotential

| Item | Value |
| ---- | ----- |
| File | `tests/data/pw/Si_lda_nc.upf` (in-tree) |
| Family | PseudoDojo ONCVPSP-LDA-SR-noCC v0.4 |
| Cite | Hamann, *Phys. Rev. B* **88**, 085117 (2013) |

## Twin geometry and settings

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si, 2-atom fcc primitive cell |
| Lattice constant \(a\) | 5.43 Å |
| Basis atoms | (0,0,0), \((a/4,a/4,a/4)\) |
| SCF \(E_\mathrm{cut}\) | 15 Ry |
| SCF k-mesh | \(2\times2\times2\) |
| XC | `XCType::LSDA` |
| Smearing | Fermi–Dirac, `degauss = 0.02` Ry, `extra_bands = 4` |
| SCF tol | \(10^{-6}\) |
| Bands | diamond path Γ–X–W–L–Γ–K, `n_seg=4`, `n_bands=8` |

Gapped Si at these settings keeps near-integer FD occupations; \(E_F\) sits
in the gap. The product gate is the **metallic bands path** (no throw) +
open spectrum gap + finite \(E_F\), not a metal-like vanishing gap.

## Acceptance pins

| Gate | Criterion |
| ---- | --------- |
| SCF | `scf_converged` |
| Spectrum | 8 bands/k, sorted ascending; path includes Γ and X |
| Local gaps | same-\(k\) \(\varepsilon_{4}-\varepsilon_{3} > 0\) for all path k |
| Integer-fill gap | `gap_ry > 0` |
| \(E_F\) | finite, \(\lvert E_F\rvert < 10^3\) Ry |
| \(E_F\) placement | within \(\pm 4\,\mathrm{degauss}\) of \([E_\mathrm{vbm},E_\mathrm{cbm}]\) |

## CP2K offline twin

CP2K Quickstep `BAND_STRUCTURE` uses **GTH + Gaussian**, not ONCV plane
waves — absolute \(\varepsilon_n\) are **not** comparable. Document the
product counterpart (`&SMEAR METHOD FERMI_DIRAC` + `BAND_STRUCTURE` path
print). Offline note:

`tools/cp2k_oracle/inputs/bands-metallic-si.note`

| Item | Value |
| ---- | ----- |
| Preferred oracle | CP2K 2026.1+ Quickstep PADE + SMEAR FERMI_DIRAC + BAND_STRUCTURE on diamond Si |
| CP2K binary this host | **not available** (`module avail cp2k` empty at pin time) |
| Scientific pin | metallic bands path enabled; open Si gap; \(E_F\) in gap |

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K binary/module on the build host for a fair wall twin this shift |
| ATV twin wall (informative) | **≈ 14.3 s** (Release, OMP=8, ecut 15 / SCF k=2 + path n_seg=4 + FD) |
| Measured (2026-07-26) | `e_fermi ≈ 0.465` Ry, gap ≈ 0.043 Ry, SCF 6 iters, `F ≈ −15.686` Ry |
| `ratio_atv_over_cp2k` | n/a (exempt) |
| `within_10x` | n/a (exempt; correctness gate only) |

When a CP2K binary is available: time comparable geometry with the oracle
note input, append \(r = T_\mathrm{atv}/T_\mathrm{cp2k}\), require \(r\le 10\).
