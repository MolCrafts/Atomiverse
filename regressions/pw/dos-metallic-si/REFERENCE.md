# Reference — metallic occupation DOS (diamond Si, PW LSDA + FD smear)

## Method

Spin-unpolarized plane-wave Kohn–Sham DFT with **Fermi–Dirac metallic
occupation smearing** on the SCF density, then **non-self-consistent**
diagonalization on a denser Monkhorst–Pack mesh and a **Gaussian-broadened
DOS**. The chemical potential \(\mu\) is recomputed on the DOS mesh; the
occupied integral is

\[
n_{\mathrm{elec}}(E_F)=\int_{-\infty}^{E_F} D(E)\,dE \approx n_e.
\]

Public path (no `*Config`, no factories):

```cpp
cpu::pw::PW method;
method.smearing = Smearing::fermi_dirac;
method.degauss = 0.02;   // Ry
method.extra_bands = 4;
// … ecut, k_mesh, upf, xc=LSDA …

cpu::pw::Dos dos;
dos.k_mesh = 4;
dos.n_bands = 8;
dos.sigma_ry = 0.05;
const DosResult dr = dos.compute(method, frame);
// dr.e_fermi_ry, dr.n_elec_to_vbm (= integral to E_F), dr.dos
```

Sibling insulator DOS (fixed occupations): `regressions/pw/bands-dos-si`.
Sibling metallic SCF (energy only): `regressions/pw/pw-smearing-al`.
Sibling metallic DFPT: `regressions/response/dfpt-metallic-si`.

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
| DOS | MP `k_mesh=4`, `n_bands=8`, \(\sigma=0.05\) Ry, 200 bins |

Gapped Si at these settings keeps near-integer FD occupations; \(E_F\) sits
in the gap. The product gate is the **metallic DOS path** (no throw) +
electron count from \(\int^{E_F} D(E)\,dE\), not a metal-like vanishing gap.

## Acceptance pins

| Gate | Criterion |
| ---- | --------- |
| SCF | `scf_converged` |
| \(E_F\) | finite, \(\lvert E_F\rvert < 10^3\) Ry |
| \(n_e(E_F)\) | \(\lvert n_{\mathrm{elec\_to\_vbm}} - 8\rvert \le 1.0\) |
| Gap | still open on DOS mesh (`gap_ry > 0`) for this twin |
| \(E_F\) placement | within \(\pm 4\,\mathrm{degauss}\) of \([E_\mathrm{vbm},E_\mathrm{cbm}]\) |

## CP2K offline twin

CP2K Quickstep `BAND_STRUCTURE` / DOS use **GTH + Gaussian**, not ONCV
plane waves — absolute \(\varepsilon_n\) / \(D(E)\) are **not** comparable.
Document the product counterpart (`&SMEAR METHOD FERMI_DIRAC` + denser
k-mesh DOS print). Offline note:

`tools/cp2k_oracle/inputs/dos-metallic-si.note`

| Item | Value |
| ---- | ----- |
| Preferred oracle | CP2K 2026.1+ Quickstep PADE + SMEAR FERMI_DIRAC on diamond Si |
| CP2K binary this host | **not available** (`module avail cp2k` empty at pin time) |
| Scientific pin | occupation integral \(\approx n_e\); metallic path enabled |

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K binary/module on the build host for a fair wall twin this shift |
| ATV twin wall (informative) | **≈ 23.5 s** (Release, OMP=8, ecut 15 / SCF k=2 + DOS k=4 + FD) |
| Measured (2026-07-26) | `n_elec_to_E_F ≈ 8.01`, `e_fermi ≈ 0.465` Ry, SCF 6 iters, `F ≈ −15.686` Ry |
| `ratio_atv_over_cp2k` | n/a (exempt) |
| `within_10x` | n/a (exempt; correctness gate only) |

When a CP2K binary is available: time comparable geometry with the oracle
note input, append \(r = T_\mathrm{atv}/T_\mathrm{cp2k}\), require \(r\le 10\).
