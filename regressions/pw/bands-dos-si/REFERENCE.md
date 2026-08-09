# Reference — PW band structure + DOS (bulk silicon)

## Method

Spin-unpolarized plane-wave Kohn–Sham DFT with LDA (Slater + PW92) on the PW
real-space grid. Norm-conserving ONCVPSP LDA Si UPF. Self-consistent density
from a uniform Monkhorst–Pack mesh, then **non-self-consistent**
diagonalization on a high-symmetry **k-path** (bands) and a denser MP mesh
(DOS, Gaussian broadening).

Public path:

```cpp
cpu::pw::PW method;          // members: upf, ecut, k_mesh, xc=LSDA, …
cpu::pw::Bands bands;        // members: n_bands, kfrac / set_diamond_path
auto br = bands.compute(method, frame);
cpu::pw::Dos dos;            // members: k_mesh, n_bands, sigma_ry, …
auto dr = dos.compute(method, frame);
```

No public `*Config` bag; no factories.

## Pseudopotential

| Item | Value |
| ---- | ----- |
| File | `tests/data/pw/Si_lda_nc.upf` (in-tree) |
| Family | PseudoDojo ONCVPSP-LDA-SR-noCC v0.4 |
| Cite | Hamann, *Phys. Rev. B* **88**, 085117 (2013); van Setten et al., *Comput. Phys. Commun.* **226**, 39 (2018) |

Same PP family as `regressions/pw/pw-lda-si`.

## Literature pin (indirect Γ→X gap)

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

## Twin geometry and settings

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

## Acceptance (incomplete basis)

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

## CP2K note

CP2K Quickstep `BAND_STRUCTURE` / DOS use **GTH + Gaussian** (DZVP-GTH), not
ONCV plane waves. Absolute \(\varepsilon_n\) are **not** comparable.
Character (indirect Si, \(\varepsilon_c(X)<\varepsilon_c(\Gamma)\)) and the
Yin–Cohen LDA Γ→X window are the scientific pins. Offline twin input:
`tools/cp2k_oracle/inputs/si_bands_dos_note.inp`.

## Performance — measured fair wall (oracle: qe)

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
