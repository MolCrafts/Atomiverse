# Reference — collinear spin + metallic FD DOS (diamond Si, PW LSDA)

## Method

Collinear spin-polarized plane-wave Kohn–Sham DFT with **Fermi–Dirac
metallic occupation smearing** on the SCF density (`spin_polarized = true`,
`multiplicity = 1` seeds \(M=0\); magnetization is free under smear), then
**non-self-consistent** diagonalization of each spin channel on a denser
MP mesh with frozen \(V_{\mathrm{eff},\sigma}\), and a **Gaussian-broadened
total DOS** \(D = D_\alpha + D_\beta\) (spin-orbital weight 1). The chemical
potential \(\mu\) is recomputed on the DOS mesh; the occupied integral is

\[
n_{\mathrm{elec}}(E_F)=\int_{-\infty}^{E_F} D(E)\,dE \approx n_e.
\]

Public path (no `*Config`, no factories):

```cpp
cpu::pw::PW method;
method.spin_polarized = true;
method.multiplicity = 1;
method.smearing = Smearing::fermi_dirac;
method.degauss = 0.02;   // Ry
method.extra_bands = 4;
// … ecut, k_mesh, upf, xc=LSDA …

cpu::pw::Dos dos;
dos.k_mesh = 4;
dos.n_bands = 8;
dos.sigma_ry = 0.05;
const DosResult dr = dos.compute(method, frame);
// dr.nspin==2; e_fermi_ry; n_elec_to_vbm ≈ 8 (integral to E_F)
```

Sibling collinear fixed-occ DOS: `regressions/pw/dos-spin-si`.  
Sibling restricted metallic DOS: `regressions/pw/dos-metallic-si`.  
Sibling collinear spin bands: `regressions/pw/bands-spin-si`.  
Sibling metallic bands: `regressions/pw/bands-metallic-si`.  
Bands spin+smear remains a separate product row (`bands-spin-metallic-si`).

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
| Spin | collinear, multiplicity = 1 (seed \(N_\uparrow=N_\downarrow=4\)) |
| Smearing | Fermi–Dirac, `degauss = 0.02` Ry, `extra_bands = 4` |
| SCF tol | \(10^{-6}\) |
| DOS | MP `k_mesh=4`, `n_bands=8`, \(\sigma=0.05\) Ry, 200 bins |

Gapped Si at these settings keeps near-integer FD occupations; \(E_F\) sits
in the gap. The product gate is the **spin + metallic DOS path** (no throw)
+ electron count from \(\int^{E_F} D(E)\,dE\), not a metal-like vanishing gap.

## Acceptance pins

| Gate | Criterion |
| ---- | --------- |
| SCF | `scf_converged` |
| Spin | `nspin == 2`, \(\lvert M\rvert \le 0.1\) |
| \(E_F\) | finite, \(\lvert E_F\rvert < 10^3\) Ry |
| \(n_e(E_F)\) | \(\lvert n_{\mathrm{elec\_to\_vbm}} - 8\rvert \le 1.0\) |
| Gap | still open on DOS mesh (`gap_ry > 0`) for this twin |
| \(E_F\) placement | within \(\pm 4\,\mathrm{degauss}\) of \([E_\mathrm{vbm},E_\mathrm{cbm}]\) |
| Channels | `dos_alpha` / `dos_beta` same length as total `dos` |
| Consistency | \(\lvert\int(D_\alpha+D_\beta)-\int D\rvert\) small to \(E_F\) |

## CP2K offline twin

CP2K Quickstep PDOS with LSD + SMEAR uses **GTH + Gaussian**, not ONCV
plane waves — absolute \(D(E)\) is **not** comparable. Character pins:
\(M\approx 0\), occupied integral ≈ 8, open gap, path enabled. Offline note:

`tools/cp2k_oracle/inputs/dos-spin-metallic-si.note`

| Item | Value |
| ---- | ----- |
| Preferred oracle | CP2K 2026.1+ Quickstep PADE + LSD + SMEAR FERMI_DIRAC on diamond Si |
| CP2K binary this host | **not available** (`module avail cp2k` empty at pin time) |
| Scientific pin | \(\int^{E_F} D \approx n_e\); collinear + metallic DOS path enabled |

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K binary/module on the build host for a fair wall twin this shift |
| ATV twin wall (informative) | **≈ 36.6 s** (Release, OMP=8, ecut 15 / SCF k=2 + DOS k=4 + collinear + FD) |
| Measured (2026-07-26) | `n_elec_to_E_F ≈ 8.01`, `e_fermi ≈ 0.473` Ry, gap ≈ 0.035 Ry, SCF 13 iters, `E ≈ −15.672` Ry, `mag=0` |
| `ratio_atv_over_cp2k` | n/a (exempt) |
| `within_10x` | n/a (exempt; correctness gate only) |

When a CP2K binary is available: time comparable geometry with the oracle
note input, append \(r = T_\mathrm{atv}/T_\mathrm{cp2k}\), require \(r\le 10\).
