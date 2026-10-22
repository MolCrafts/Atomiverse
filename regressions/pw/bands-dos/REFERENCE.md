# pw/bands-dos

One case; every section below is run by `atv_regressions --case pw/bands-dos` and keeps its own oracle record.


## Reference — PW band structure + DOS (bulk silicon)

### Method

Spin-unpolarized plane-wave Kohn–Sham DFT with LDA (Slater + PW92) on the PW
real-space grid. Norm-conserving ONCVPSP LDA Si UPF. Self-consistent density
from a uniform Monkhorst–Pack mesh, then **non-self-consistent**
diagonalization on a high-symmetry **k-path** (bands) and a denser MP mesh
(DOS, Gaussian broadening).

Public path:

```cpp
cpu::pw::PW method{cpu::pw::UpfPseudopotential{"..."}, /*ecut_ry=*/15.0,
                   /*k_mesh=*/2, cpu::scf::XCType::LSDA};  // + knobs: tol, max_iter, …
cpu::pw::Bands bands;        // members: n_bands, kfrac / set_diamond_path
auto br = bands.compute(method, frame);
cpu::pw::Dos dos;            // members: k_mesh, n_bands, sigma_ry, …
auto dr = dos.compute(method, frame);
```

No public `*Config` bag; no factories.

### Pseudopotential

| Item | Value |
| ---- | ----- |
| File | `tests/data/pw/Si_lda_nc.upf` (in-tree) |
| Family | PseudoDojo ONCVPSP-LDA-SR-noCC v0.4 |
| Cite | Hamann, *Phys. Rev. B* **88**, 085117 (2013); van Setten et al., *Comput. Phys. Commun.* **226**, 39 (2018) |

Same PP family as `regressions/pw/rks`.

### Literature pin (indirect Γ→X gap)

Bulk Si is an **indirect** gap semiconductor: valence-band maximum at
\(\Gamma\), conduction-band minimum near \(X\).

| Quantity | Value | Source |
| -------- | ----- | ------ |
| Experimental \(E_g\) | ~1.12 eV | textbook |
| Converged LDA \(E_g\) (Γ\(_v\)→ near-\(X_c\)) | **~0.45 eV** | Yin & Cohen, *Phys. Rev. B* **26**, 5668 (1982) |

The regression pins the **textbook Si transition**
\[
E_{\Gamma\to X}
  = \varepsilon_{c}(X) - \varepsilon_{v}(\Gamma)
\]
(not the global \(\max\varepsilon_v - \min\varepsilon_c\) over a sparse path,
which at incomplete \(E_\mathrm{cut}\) can spuriously go slightly negative
from path-point scatter while every **local** same-\(k\) gap stays open).

### Twin geometry and settings

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si, 2-atom fcc primitive cell |
| Lattice constant \(a\) | 5.43 Å |
| Basis atoms | (0,0,0), \((a/4,a/4,a/4)\) |
| SCF \(E_\mathrm{cut}\) | 15 Ry |
| SCF k-mesh | \(2\times2\times2\) (`PW::k_mesh = 2`) |
| XC | `XCType::LSDA` |
| SCF tol | \(10^{-6}\) |
| Bands | diamond path Γ–X–W–L–Γ–K, `n_seg=4`, `n_bands=8` |
| DOS | MP `k_mesh=4`, `n_bands=8`, \(\sigma=0.05\) Ry, 200 bins |

### Acceptance (incomplete basis)

1. SCF for both `Bands` and `Dos` converges.
2. Eigenvalues per k sorted ascending; 8 bands.
3. Every **local** (same-\(k\)) gap \(\varepsilon_{n_\mathrm{occ}}(k)-\varepsilon_{n_\mathrm{occ}-1}(k) > 0\).
4. Path includes \(\Gamma\) and \(X\); **indirect character**
   \(\varepsilon_c(X) < \varepsilon_c(\Gamma)\).
5. **Γ→X gap window** (eV):
   \[
   0.05 \le E_{\Gamma\to X}^\mathrm{ATV} \le 2.50
   \]
   with \(E_{\Gamma\to X}=(\varepsilon_c(X)-\varepsilon_v(\Gamma))\times 13.605693\).
   Measured on this host (2026-07-24, Release): \(\approx 0.48\) eV —
   near Yin–Cohen ~0.45 eV despite tutorial ecut/k.
6. Direct \(\Gamma\) gap \(\ge\) Γ→X − 0.05 eV.
7. DOS occupied integral \(n_\mathrm{elec\_to\_vbm}\) within **1.0** of
   \(n_e = 8\).

### CP2K note

CP2K Quickstep `BAND_STRUCTURE` / DOS use **GTH + Gaussian** (DZVP-GTH), not
ONCV plane waves. Absolute \(\varepsilon_n\) are **not** comparable.
Character (indirect Si, \(\varepsilon_c(X)<\varepsilon_c(\Gamma)\)) and the
Yin–Cohen LDA Γ→X window are the scientific pins. Offline twin input:
`tools/cp2k_oracle/inputs/si_bands_dos_note.inp`.

### Performance — measured fair wall (oracle: qe)

**Measured 2026-08-09** against the offline Quantum ESPRESSO twin
(`Program PWSCF v.7.6`, source build of 2026-08-09,
`/nobackup/proj/disk/teoroo/personal/jicli594/work/qe/install/bin/pw.x` +
`dos.x`; install record `tools/qe_oracle/pin_notes.md`). Twin job = bands +
DOS for this system in each engine's native workflow, same NC UPF / cell /
\(E_\mathrm{cut}=15\) Ry: QE runs scf (\(2^3\)) → `bands` NSCF on **exactly
ATV's 16-point** Γ–X–W–L–Γ–K list (`set_diamond_path(4)`, 8 bands) → `nscf`
on the \(4^3\) DOS mesh → `dos.x` (degauss 0.05 Ry, ΔE 0.1 eV ≈ 200 bins).
Inputs: `tools/qe_oracle/inputs/bands-dos-si/` (`si_scf.in`, `si_bands.in`,
`si_nscf_dos.in`, `si_dos.in`).

| Item | Value |
| ---- | ----- |
| \(T_{\mathrm{atv}}\) | **21.68 s** (`reg_bands-dos-si`; median of 3) |
| \(T_{\mathrm{qe}}\) | **1.32 s** (0.34 scf + 0.44 bands + 0.37 nscf + 0.17 dos.x, printed WALLs, median of 3) |
| \(T_{\mathrm{atv}}/T_{\mathrm{qe}}\) | **16.4** |
| within_10x | **false — open PW perf debt** (was the stated target when exempt) |

Protocol: Arrhenius x86_64 epyc login node (48-core, shared; load avg 7–24
during the window), one shell, 3 interleaved repetitions, medians; both sides
single process, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, QE as
`mpirun -np 1`. ATV binary: Release, `-march=native`, this commit.

QE key output lines (provenance, not ctest pins):

```text
si_scf.out : !    total energy              =     -15.67051079 Ry   (7 iters)
si_nscf.out:      highest occupied, lowest unoccupied level (ev): 6.3310  6.8061
```

Recorded cost asymmetries: QE computes the SCF density **once** and reuses it
for both NSCF steps, while ATV's `Bands::compute` and `Dos::compute` each run
their own SCF (public-API cost structure, counted as-is); QE also reduces the
\(2^3\) SCF mesh to 3 IBZ points and the \(4^3\) DOS mesh to 8, vs ATV's full
8 and 64.


## Reference — metallic occupation band structure (diamond Si, PW LSDA + FD)

### Method

Spin-unpolarized plane-wave Kohn–Sham DFT with **Fermi–Dirac metallic
occupation smearing** on the SCF density, then **non-self-consistent**
diagonalization on a high-symmetry **k-path** (`cpu::pw::Bands`).

Public path (no `*Config`, no factories):

```cpp
cpu::pw::PW method{cpu::pw::UpfPseudopotential{"..."}, /*ecut_ry=*/15.0,
                   /*k_mesh=*/2, cpu::scf::XCType::LSDA};
method.smearing = Smearing::fermi_dirac;
method.degauss = 0.02;   // Ry
method.extra_bands = 4;

cpu::pw::Bands bands;
bands.n_bands = 8;
bands.set_diamond_path(4);
const BandsResult br = bands.compute(method, frame);
// br.eigenvalues_ry, br.e_fermi_ry, br.gap_ry, br.energy_ry (free energy F)
```

Sibling insulator bands: `regressions/pw/bands-dos`.  
Sibling metallic DOS: `regressions/pw/bands-dos`.  
Sibling spin bands (fixed occ): `regressions/pw/bands-dos`.

### Pseudopotential

| Item | Value |
| ---- | ----- |
| File | `tests/data/pw/Si_lda_nc.upf` (in-tree) |
| Family | PseudoDojo ONCVPSP-LDA-SR-noCC v0.4 |
| Cite | Hamann, *Phys. Rev. B* **88**, 085117 (2013) |

### Twin geometry and settings

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

### Acceptance pins

| Gate | Criterion |
| ---- | --------- |
| SCF | `scf_converged` |
| Spectrum | 8 bands/k, sorted ascending; path includes Γ and X |
| Local gaps | same-\(k\) \(\varepsilon_{4}-\varepsilon_{3} > 0\) for all path k |
| Integer-fill gap | `gap_ry > 0` |
| \(E_F\) | finite, \(\lvert E_F\rvert < 10^3\) Ry |
| \(E_F\) placement | within \(\pm 4\,\mathrm{degauss}\) of \([E_\mathrm{vbm},E_\mathrm{cbm}]\) |

### CP2K offline twin

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

### Performance

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


## Reference — collinear spin-polarized PW band structure (bulk silicon)

### Method

Collinear spin-polarized plane-wave Kohn–Sham DFT with LDA (Slater + PW92)
on the PW real-space grid (`spin_polarized = true`, `multiplicity = 1` ⇒
\(M = N_\uparrow - N_\downarrow = 0\)). Norm-conserving ONCVPSP LDA Si UPF.
Self-consistent spin densities from a uniform Monkhorst–Pack mesh, then
**non-self-consistent** diagonalization of each spin channel on a frozen
\(V_{\mathrm{eff},\sigma}\) along a high-symmetry **k-path**.

Public path:

```cpp
cpu::pw::PW method{cpu::pw::UpfPseudopotential{"..."}, /*ecut_ry=*/15.0,
                   /*k_mesh=*/2, cpu::scf::XCType::LSDA};
method.spin_polarized = true;
method.multiplicity = 1;

cpu::pw::Bands bands;
bands.n_bands = 8;
bands.set_diamond_path(4);
auto br = bands.compute(method, frame);
// br.nspin == 2; br.eigenvalues_ry (α), br.eigenvalues_beta_ry (β)
```

No public `*Config` bag; no factories.

Sibling closed-shell product: `regressions/pw/bands-dos`.

### Pseudopotential

| Item | Value |
| ---- | ----- |
| File | `tests/data/pw/Si_lda_nc.upf` (in-tree) |
| Family | PseudoDojo ONCVPSP-LDA-SR-noCC v0.4 |
| Cite | Hamann, *Phys. Rev. B* **88**, 085117 (2013); van Setten et al., *Comput. Phys. Commun.* **226**, 39 (2018) |

### Literature pin (indirect Γ→X gap)

Bulk Si is an **indirect** gap semiconductor: VBM at \(\Gamma\), CBM near \(X\).

| Quantity | Value | Source |
| -------- | ----- | ------ |
| Experimental \(E_g\) | ~1.12 eV | textbook |
| Converged LDA \(E_g\) (Γ\(_v\)→ near-\(X_c\)) | **~0.45 eV** | Yin & Cohen, *Phys. Rev. B* **26**, 5668 (1982) |

With \(M=0\) collinear occupations the α and β channels are degenerate;
the scientific pins are the same textbook transition and α≈β parity.

### Twin geometry and settings

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

### Acceptance

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

### CP2K note

CP2K Quickstep `BAND_STRUCTURE` with `LSD` / UKS uses **GTH + Gaussian**
(DZVP-GTH), not ONCV plane waves. Absolute \(\varepsilon_n\) are **not**
comparable. Character (indirect Si, \(\varepsilon_c(X)<\varepsilon_c(\Gamma)\)),
\(M=0\), and the Yin–Cohen LDA Γ→X window are the scientific pins.
Offline twin input: `tools/cp2k_oracle/inputs/bands-spin-si.note`.

### Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K binary / module on the build host for a fair wall twin this shift |
| ATV twin wall (informative) | ~16 s (Release, OMP=8, ecut 15 / k=2 + collinear path, 2026-07-26) |
| `ratio_atv_over_cp2k` | n/a (exempt) |
| `within_10x` | n/a (exempt; correctness gate only) |

When a CP2K binary is available: time the same geometry with the oracle note
input, append \(r = T_\mathrm{atv}/T_\mathrm{cp2k}\) here, require \(r \le 10\).


## Reference — collinear spin + metallic occupation bands (diamond Si, PW)

### Method

Collinear spin-polarized plane-wave Kohn–Sham DFT with LDA (Slater + PW92)
**and** Fermi–Dirac metallic occupation smearing, then non-self-consistent
diagonalization on frozen \(V_{\mathrm{eff},\sigma}\) along a high-symmetry
**k-path** (`cpu::pw::Bands`).

Public path (no `*Config`, no factories):

```cpp
cpu::pw::PW method{cpu::pw::UpfPseudopotential{"..."}, /*ecut_ry=*/15.0,
                   /*k_mesh=*/2, cpu::scf::XCType::LSDA};
method.spin_polarized = true;
method.multiplicity = 1;           // seed M=0; free under smear
method.smearing = Smearing::fermi_dirac;
method.degauss = 0.02;             // Ry
method.extra_bands = 4;

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

### Pseudopotential

| Item | Value |
| ---- | ----- |
| File | `tests/data/pw/Si_lda_nc.upf` (in-tree) |
| Family | PseudoDojo ONCVPSP-LDA-SR-noCC v0.4 |
| Cite | Hamann, *Phys. Rev. B* **88**, 085117 (2013) |

### Literature pin (indirect Γ→X gap)

Bulk Si is an **indirect** gap semiconductor: VBM at \(\Gamma\), CBM near \(X\).

| Quantity | Value | Source |
| -------- | ----- | ------ |
| Experimental \(E_g\) | ~1.12 eV | textbook |
| Converged LDA \(E_g\) (Γ\(_v\)→ near-\(X_c\)) | **~0.45 eV** | Yin & Cohen, *Phys. Rev. B* **26**, 5668 (1982) |

With \(M=0\) seed and mild FD smear, α and β stay nearly degenerate; \(E_F\)
sits in the open Si gap (near-integer occupations). The product gate is the
**spin+smear bands path** (no throw) + scientific character pins.

### Twin geometry and settings

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

### Acceptance pins

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

### CP2K offline twin

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

### Performance

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


## Reference — metallic occupation DOS (diamond Si, PW LSDA + FD smear)

### Method

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
cpu::pw::PW method{cpu::pw::UpfPseudopotential{"..."}, /*ecut_ry=*/15.0,
                   /*k_mesh=*/2, cpu::scf::XCType::LSDA};
method.smearing = Smearing::fermi_dirac;
method.degauss = 0.02;   // Ry
method.extra_bands = 4;

cpu::pw::Dos dos;
dos.k_mesh = 4;
dos.n_bands = 8;
dos.sigma_ry = 0.05;
const DosResult dr = dos.compute(method, frame);
// dr.e_fermi_ry, dr.n_elec_to_vbm (= integral to E_F), dr.dos
```

Sibling insulator DOS (fixed occupations): `regressions/pw/bands-dos`.
Sibling metallic bands: `regressions/pw/bands-dos`.
Sibling metallic DFPT: `regressions/response/dfpt`.

### Pseudopotential

| Item | Value |
| ---- | ----- |
| File | `tests/data/pw/Si_lda_nc.upf` (in-tree) |
| Family | PseudoDojo ONCVPSP-LDA-SR-noCC v0.4 |
| Cite | Hamann, *Phys. Rev. B* **88**, 085117 (2013) |

### Twin geometry and settings

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

### Acceptance pins

| Gate | Criterion |
| ---- | --------- |
| SCF | `scf_converged` |
| \(E_F\) | finite, \(\lvert E_F\rvert < 10^3\) Ry |
| \(n_e(E_F)\) | \(\lvert n_{\mathrm{elec\_to\_vbm}} - 8\rvert \le 1.0\) |
| Gap | still open on DOS mesh (`gap_ry > 0`) for this twin |
| \(E_F\) placement | within \(\pm 4\,\mathrm{degauss}\) of \([E_\mathrm{vbm},E_\mathrm{cbm}]\) |

### CP2K offline twin

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

### Performance

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


## Reference — collinear-spin PW DOS (diamond Si)

### Method

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
cpu::pw::PW method{cpu::pw::UpfPseudopotential{"..."}, /*ecut_ry=*/15.0,
                   /*k_mesh=*/2, cpu::scf::XCType::LSDA};
method.spin_polarized = true;
method.multiplicity = 1;

cpu::pw::Dos dos;
dos.k_mesh = 4;
dos.n_bands = 8;
dos.sigma_ry = 0.05;
const DosResult dr = dos.compute(method, frame);
// dr.nspin == 2; dr.dos / dos_alpha / dos_beta; n_elec_to_vbm ≈ 8
```

Sibling closed-shell insulator DOS: `regressions/pw/bands-dos`.  
Sibling metallic (restricted) DOS: `regressions/pw/bands-dos`.  
Sibling collinear spin **bands**: `regressions/pw/bands-dos`.

### Pseudopotential

| Item | Value |
| ---- | ----- |
| File | `tests/data/pw/Si_lda_nc.upf` (in-tree) |
| Family | PseudoDojo ONCVPSP-LDA-SR-noCC v0.4 |
| Cite | Hamann, *Phys. Rev. B* **88**, 085117 (2013) |

### Twin geometry and settings

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

### Acceptance pins

| Gate | Criterion |
| ---- | --------- |
| SCF | `scf_converged` |
| Spin | `nspin == 2`, \(\lvert M\rvert \le 10^{-6}\) |
| \(n_e\) | \(\lvert n_{\mathrm{elec\_to\_vbm}} - 8\rvert \le 1.0\) |
| Gap | `gap_ry > 0` on DOS mesh |
| Channels | `dos_alpha` / `dos_beta` same length as total `dos` |
| Consistency | \(\lvert\int(D_\alpha+D_\beta)-\int D\rvert\) small on shared grid |

### CP2K offline twin

CP2K Quickstep PDOS / LSD uses **GTH + Gaussian**, not ONCV plane waves —
absolute \(D(E)\) is **not** comparable. Character pins: \(M=0\), occupied
integral ≈ 8, open gap. Offline note:

`tools/cp2k_oracle/inputs/dos-spin-si.note`

| Item | Value |
| ---- | ----- |
| Preferred oracle | CP2K 2026.1+ Quickstep PADE + LSD on diamond Si |
| CP2K binary this host | **not available** (`module avail cp2k` empty at pin time) |
| Scientific pin | \(\int^{E_\mathrm{vbm}} D \approx n_e\); collinear path enabled |

### Performance

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


## Reference — collinear spin + metallic FD DOS (diamond Si, PW LSDA)

### Method

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
cpu::pw::PW method{cpu::pw::UpfPseudopotential{"..."}, /*ecut_ry=*/15.0,
                   /*k_mesh=*/2, cpu::scf::XCType::LSDA};
method.spin_polarized = true;
method.multiplicity = 1;
method.smearing = Smearing::fermi_dirac;
method.degauss = 0.02;   // Ry
method.extra_bands = 4;

cpu::pw::Dos dos;
dos.k_mesh = 4;
dos.n_bands = 8;
dos.sigma_ry = 0.05;
const DosResult dr = dos.compute(method, frame);
// dr.nspin==2; e_fermi_ry; n_elec_to_vbm ≈ 8 (integral to E_F)
```

Sibling collinear fixed-occ DOS: `regressions/pw/bands-dos`.  
Sibling restricted metallic DOS: `regressions/pw/bands-dos`.  
Sibling collinear spin bands: `regressions/pw/bands-dos`.  
Sibling metallic bands: `regressions/pw/bands-dos`.  
Bands spin+smear remains a separate product row (`bands-spin-metallic-si`).

### Pseudopotential

| Item | Value |
| ---- | ----- |
| File | `tests/data/pw/Si_lda_nc.upf` (in-tree) |
| Family | PseudoDojo ONCVPSP-LDA-SR-noCC v0.4 |
| Cite | Hamann, *Phys. Rev. B* **88**, 085117 (2013) |

### Twin geometry and settings

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

### Acceptance pins

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

### CP2K offline twin

CP2K Quickstep PDOS with LSD + SMEAR uses **GTH + Gaussian**, not ONCV
plane waves — absolute \(D(E)\) is **not** comparable. Character pins:
\(M\approx 0\), occupied integral ≈ 8, open gap, path enabled. Offline note:

`tools/cp2k_oracle/inputs/dos-spin-metallic-si.note`

| Item | Value |
| ---- | ----- |
| Preferred oracle | CP2K 2026.1+ Quickstep PADE + LSD + SMEAR FERMI_DIRAC on diamond Si |
| CP2K binary this host | **not available** (`module avail cp2k` empty at pin time) |
| Scientific pin | \(\int^{E_F} D \approx n_e\); collinear + metallic DOS path enabled |

### Performance

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
