# Reference — collinear-spin PW DOS (diamond Si)

## Method

Collinear spin-polarized plane-wave Kohn–Sham DFT with LDA (Slater + PW92)
on the PW real-space grid (`spin_polarized = true`, `multiplicity = 1` ⇒
\(M = N_\uparrow - N_\downarrow = 0\)). Norm-conserving ONCVPSP LDA Si UPF.
Self-consistent spin densities from a uniform Monkhorst–Pack mesh, then
**non-self-consistent** diagonalization of each spin channel on a denser
MP mesh with frozen \(V_{\mathrm{eff},\sigma}\), and a **Gaussian-broadened
total DOS** \(D = D_\alpha + D_\beta\) (spin-orbital weight 1). The
occupied integral is

\[
n_{\mathrm{elec}}(E_\mathrm{vbm})=\int_{-\infty}^{E_\mathrm{vbm}} D(E)\,dE
\approx n_e.
\]

Public path (no `*Config`, no factories):

```cpp
cpu::pw::PW method;
method.spin_polarized = true;
method.multiplicity = 1;
method.upf_path = "..."; method.ecut_ry = 15; method.k_mesh = 2;
method.xc = cpu::scf::XCType::LSDA;

cpu::pw::Dos dos;
dos.k_mesh = 4;
dos.n_bands = 8;
dos.sigma_ry = 0.05;
const DosResult dr = dos.compute(method, frame);
// dr.nspin == 2; dr.dos / dos_alpha / dos_beta; n_elec_to_vbm ≈ 8
```

Sibling closed-shell insulator DOS: `regressions/pw/bands-dos-si`.  
Sibling metallic (restricted) DOS: `regressions/pw/dos-metallic-si`.  
Sibling collinear spin **bands**: `regressions/pw/bands-spin-si`.

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
| Spin | collinear, multiplicity = 1 (\(N_\uparrow=N_\downarrow=4\)) |
| SCF tol | \(10^{-6}\) |
| DOS | MP `k_mesh=4`, `n_bands=8`, \(\sigma=0.05\) Ry, 200 bins |

## Acceptance pins

| Gate | Criterion |
| ---- | --------- |
| SCF | `scf_converged` |
| Spin | `nspin == 2`, \(\lvert M\rvert \le 10^{-6}\) |
| \(n_e\) | \(\lvert n_{\mathrm{elec\_to\_vbm}} - 8\rvert \le 1.0\) |
| Gap | `gap_ry > 0` on DOS mesh |
| Channels | `dos_alpha` / `dos_beta` same length as total `dos` |
| Consistency | \(\lvert\int(D_\alpha+D_\beta)-\int D\rvert\) small on shared grid |

## CP2K offline twin

CP2K Quickstep PDOS / LSD uses **GTH + Gaussian**, not ONCV plane waves —
absolute \(D(E)\) is **not** comparable. Character pins: \(M=0\), occupied
integral ≈ 8, open gap. Offline note:

`tools/cp2k_oracle/inputs/dos-spin-si.note`

| Item | Value |
| ---- | ----- |
| Preferred oracle | CP2K 2026.1+ Quickstep PADE + LSD on diamond Si |
| CP2K binary this host | **not available** (`module avail cp2k` empty at pin time) |
| Scientific pin | \(\int^{E_\mathrm{vbm}} D \approx n_e\); collinear path enabled |

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K binary/module on the build host for a fair wall twin this shift |
| ATV twin wall (informative) | **≈ 37.5 s** (Release, OMP=8, ecut 15 / SCF k=2 + DOS k=4 + collinear, 2026-07-26) |
| Measured (2026-07-26) | `n_elec_to_vbm ≈ 7.96`, `gap ≈ 0.035` Ry, SCF 13 iters, `E ≈ −15.672` Ry, `mag=0` |
| `ratio_atv_over_cp2k` | n/a (exempt) |
| `within_10x` | n/a (exempt; correctness gate only) |

When a CP2K binary is available: time comparable geometry with the oracle
note input, append \(r = T_\mathrm{atv}/T_\mathrm{cp2k}\), require \(r\le 10\).
