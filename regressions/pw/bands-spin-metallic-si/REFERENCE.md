# Reference — collinear spin + metallic occupation bands (diamond Si, PW)

## Method

Collinear spin-polarized plane-wave Kohn–Sham DFT with LDA (Slater + PW92)
**and** Fermi–Dirac metallic occupation smearing, then non-self-consistent
diagonalization on frozen \(V_{\mathrm{eff},\sigma}\) along a high-symmetry
**k-path** (`cpu::pw::Bands`).

Public path (no `*Config`, no factories):

```cpp
cpu::pw::PW method;
method.spin_polarized = true;
method.multiplicity = 1;           // seed M=0; free under smear
method.smearing = Smearing::fermi_dirac;
method.degauss = 0.02;             // Ry
method.extra_bands = 4;
method.upf_path = "..."; method.ecut_ry = 15; method.k_mesh = 2;
method.xc = cpu::scf::XCType::LSDA;

cpu::pw::Bands bands;
bands.n_bands = 8;
bands.set_diamond_path(4);
const BandsResult br = bands.compute(method, frame);
// br.nspin == 2; br.eigenvalues_ry (α), br.eigenvalues_beta_ry (β)
// br.e_fermi_ry from SCF μ; br.magnetization free under smear
```

Siblings:

| Regression | Covers |
| ---------- | ------ |
| `bands-spin-si` | collinear spin, fixed occupations |
| `bands-metallic-si` | restricted metallic FD bands |
| `pw-uks-fe-smearing` | UKS + FD SCF (no path) |

## Pseudopotential

| Item | Value |
| ---- | ----- |
| File | `tests/data/pw/Si_lda_nc.upf` (in-tree) |
| Family | PseudoDojo ONCVPSP-LDA-SR-noCC v0.4 |
| Cite | Hamann, *Phys. Rev. B* **88**, 085117 (2013) |

## Literature pin (indirect Γ→X gap)

Bulk Si is an **indirect** gap semiconductor: VBM at \(\Gamma\), CBM near \(X\).

| Quantity | Value | Source |
| -------- | ----- | ------ |
| Experimental \(E_g\) | ~1.12 eV | textbook |
| Converged LDA \(E_g\) (Γ\(_v\)→ near-\(X_c\)) | **~0.45 eV** | Yin & Cohen, *Phys. Rev. B* **26**, 5668 (1982) |

With \(M=0\) seed and mild FD smear, α and β stay nearly degenerate; \(E_F\)
sits in the open Si gap (near-integer occupations). The product gate is the
**spin+smear bands path** (no throw) + scientific character pins.

## Twin geometry and settings

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si, 2-atom fcc primitive cell |
| Lattice constant \(a\) | 5.43 Å |
| Basis atoms | (0,0,0), \((a/4,a/4,a/4)\) |
| SCF \(E_\mathrm{cut}\) | 15 Ry |
| SCF k-mesh | \(2\times2\times2\) |
| XC | `XCType::LSDA` |
| Spin | collinear, multiplicity = 1 (seed \(M=0\)) |
| Smearing | Fermi–Dirac, `degauss = 0.02` Ry, `extra_bands = 4` |
| SCF tol | \(10^{-6}\) |
| Bands | diamond path Γ–X–W–L–Γ–K, `n_seg=4`, `n_bands=8` per spin |

## Acceptance pins

| Gate | Criterion |
| ---- | --------- |
| SCF | `scf_converged`; `nspin == 2` |
| Spectrum | 8 bands/k α and β, sorted ascending; path includes Γ and X |
| Local gaps | same-\(k\) \(\varepsilon_{4}-\varepsilon_{3} > 0\) both spins |
| Integer-fill gap | `gap_ry > 0` |
| \(E_F\) | finite, \(\lvert E_F\rvert < 10^3\) Ry; within \(\pm 4\,\mathrm{degauss}\) of \([E_\mathrm{vbm},E_\mathrm{cbm}]\) |
| Magnetization | \(\lvert M\rvert \le 10^{-3}\) |
| α≈β | \(\lvert\varepsilon^\alpha-\varepsilon^\beta\rvert \le 10^{-3}\) Ry at Γ VBM and X CBM |
| Γ→X (α) | \(0.05 \le E_{\Gamma\to X} \le 2.50\) eV; \(\varepsilon_c^\alpha(X)<\varepsilon_c^\alpha(\Gamma)\) |

## CP2K offline twin

CP2K Quickstep `BAND_STRUCTURE` with `LSD` + `SMEAR FERMI_DIRAC` uses
**GTH + Gaussian**, not ONCV plane waves — absolute \(\varepsilon_n\) are
**not** comparable. Character (indirect Si, open gap, \(E_F\) in gap,
α≈β) is the scientific pin. Offline note:

`tools/cp2k_oracle/inputs/bands-spin-metallic-si.note`

| Item | Value |
| ---- | ----- |
| Preferred oracle | CP2K 2026.1+ Quickstep PADE + LSD + SMEAR FERMI_DIRAC + BAND_STRUCTURE on diamond Si |
| CP2K binary this host | **not available** (`module avail cp2k` empty at pin time) |
| Scientific pin | spin+smear bands path enabled; open Si gap; \(E_F\) in gap; α≈β |

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K binary/module on the build host for a fair wall twin this shift |
| ATV twin wall (informative) | **≈ 45 s** (Release, ecut 15 / SCF k=2 + path n_seg=4 + collinear FD; measured 2026-07-26) |
| Measured (2026-07-26) | `e_fermi ≈ 0.465` Ry, gap ≈ 0.043 Ry, mag=0, Γ→X(α)≈0.58 eV, `F ≈ −15.686` Ry |
| `ratio_atv_over_cp2k` | n/a (exempt) |
| `within_10x` | n/a (exempt; correctness gate only) |

When a CP2K binary is available: time comparable geometry with the oracle
note input, append \(r = T_\mathrm{atv}/T_\mathrm{cp2k}\), require \(r\le 10\).
