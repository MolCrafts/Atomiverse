# Plane-wave band structure and density of states

> The regression runner has been removed. Regression commands below are historical; case sources and oracle records are retained as archived evidence, not current executable checks.

Non-self-consistent band structures and Gaussian-broadened densities of states
on top of a plane-wave Kohn–Sham SCF. Two post-processors take a configured
`cpu::pw::PW` and a Frame: `cpu::pw::Bands::compute(method, frame)`
diagonalizes on a high-symmetry k-path and returns a `BandsResult`;
`cpu::pw::Dos::compute(method, frame)` diagonalizes on a denser
Monkhorst–Pack mesh and returns a `DosResult`. Both first run the `PW` SCF
(restricted or collinear — `spin_polarized`, `multiplicity` — with fixed or
Fermi–Dirac metallic occupations — `smearing`, `degauss` [Ry],
`extra_bands`), freeze \(V_\mathrm{eff}\) (per spin), then recompute
eigenvalues. Parameters are members (`Bands::n_bands`,
`Bands::set_diamond_path(n_seg)`; `Dos::k_mesh`, `n_bands`, `sigma_ry`,
`n_bins`); no `*Config` bag, no factories. Results are owned structs in Ry,
not Frame meta. The sections cover diamond Si on the CPU backend: bands and DOS
for every combination of restricted / collinear spin and fixed / metallic
occupations.

Case: `ctest -R '^pw/bands-dos$'` — sources and oracle record in [`regressions/pw/bands-dos/`](../../../regressions/pw/bands-dos/). One section of the case per `##` heading below: `atv_regressions --case pw/bands-dos --section <name>`.

| | Fixed occupations | Fermi–Dirac smearing |
| --- | --- | --- |
| **Bands, restricted** | [bands-dos-si](#bands-dos-si) | [bands-metallic-si](#bands-metallic-si) |
| **Bands, collinear** | [bands-spin-si](#bands-spin-si) | [bands-spin-metallic-si](#bands-spin-metallic-si) |
| **DOS, restricted** | [bands-dos-si](#bands-dos-si) | [dos-metallic-si](#dos-metallic-si) |
| **DOS, collinear** | [dos-spin-si](#dos-spin-si) | [dos-spin-metallic-si](#dos-spin-metallic-si) |

Roadmap: [dft.md](../../roadmap/dft.md) § bands/DOS. Requires `ATV_BUILD_PW=ON`
(CPU build environment `buildenv-gcc/2026.03-mpich`).

## Method

**SCF.** Periodic plane-wave Kohn–Sham DFT from the same PW stack as
[pw-lda-si](pw-rks.md#pw-lda-si) (Hartree in \(G\)-space, LDA XC — Slater +
PW92, `XCType::LSDA` — on the FFT grid, Davidson diagonalization, NC UPF).
Collinear runs (nspin=2) use Hartree from total
\(\rho=\rho_\uparrow+\rho_\downarrow\), separate \(V_{\mathrm{xc}}^\sigma\) and
Davidson per spin, as in
[pw-spin-polarized-o2](pw-collinear-spin.md#pw-spin-polarized-o2). Fixed
fillings follow from multiplicity: \(N_\uparrow - N_\downarrow = m - 1\),
\(N_\uparrow + N_\downarrow = N_\mathrm{val}\); for Si
(\(Z_\mathrm{val}=4\) × 2 atoms, \(n_e = 8\)) `multiplicity = 1` gives
\(N_\uparrow=N_\downarrow=4\), and the α and β spectra are degenerate within
numerical noise.

**Metallic occupations.** With `Smearing::fermi_dirac`,

\[
f(x)=\frac{1}{1+e^{x}},\qquad
x=\frac{\varepsilon-\mu}{\sigma},\qquad
\sigma=\texttt{degauss}\ \mathrm{[Ry]},
\]

doubled to \(f_b\in[0,2]\) in the restricted (closed-shell) convention and with
spin-orbital weight 1 per channel when collinear. A shared \(\mu\) is fixed by
electron-count bisection (\(n_e = Z_\mathrm{val}N_\mathrm{atoms} = 8\)); empty
bands near \(E_F\) come from `extra_bands`. With smearing on, the SCF total is
the **Mermin free energy** \(F=E_\mathrm{KS}-TS\), with FD entropy in band form
\(s = -2[f\ln f+(1-f)\ln(1-f)]\) and
\(E_{\mathrm{entropy}} = -TS = -\sigma\sum_{k,b} w_k\, s_{kb}\). Under smear the
collinear magnetization is free; `multiplicity` only seeds it. The occupation
kinds (`gaussian`, `methfessel_paxton` with `mp_order`) are the same as on
[pw-smearing-al](pw-smearing.md#pw-smearing-al); the gates here use
Fermi–Dirac.

**Bands (k-path, NSCF).** After SCF each \(V_{\mathrm{eff},\sigma}\) is frozen;
only the k-dependent kinetic/projector terms change. Fractional path
\(\Gamma\)–\(X\)–\(W\)–\(L\)–\(\Gamma\)–\(K\) via `Bands::set_diamond_path(n_seg)`.
For each path \(\mathbf{k}\) and spin \(\sigma\),

\[
H[V_{\mathrm{eff},\sigma}]\, \psi_{n\mathbf{k}}^\sigma
  = \varepsilon_n^\sigma(\mathbf{k})\, \psi_{n\mathbf{k}}^\sigma,
\qquad
\tfrac12|\mathbf{k}+\mathbf{G}|^2 \le E_\mathrm{cut},
\]

with `n_bands` states per spin (valence + some conduction). Path k-points are
**not** a BZ mesh: under smearing `BandsResult::e_fermi_ry` is the **SCF**
\(\mu\), not re-solved from path weights.

**Gap definitions.** With \(n_\mathrm{occ}=4\) (integer fill per spin, also used
when SCF occupations were fractional),

\[
E_{\Gamma\to X}
  = \bigl(\varepsilon_{c}(X) - \varepsilon_{v}(\Gamma)\bigr)
    \times 13.605693~\mathrm{eV},
\qquad
\varepsilon_{v}(\Gamma)=\varepsilon_{n_\mathrm{occ}-1}(\Gamma),
\quad
\varepsilon_{c}(\mathbf{k})=\varepsilon_{n_\mathrm{occ}}(\mathbf{k}),
\]

taken on the α channel when collinear (β is checked for parity), and

\[
E_g = E_\mathrm{cbm}-E_\mathrm{vbm}
    = \min_{n\ge n_\mathrm{occ},\mathbf{k}}
      \varepsilon_n(\mathbf{k})
      - \max_{n<n_\mathrm{occ},\mathbf{k}}
      \varepsilon_n(\mathbf{k})
    \quad [\mathrm{Ry}].
\]

The textbook Γ→X transition is pinned rather than the global
\(\min\varepsilon_c-\max\varepsilon_v\), which over a **sparse** path can go
slightly negative at incomplete \(E_\mathrm{cut}\) from path-point scatter.
Local same-\(\mathbf{k}\) gaps
\(\varepsilon_{n_\mathrm{occ}}^\sigma(\mathbf{k})-\varepsilon_{n_\mathrm{occ}-1}^\sigma(\mathbf{k})\)
must stay positive (no inverted spectrum) and indirect character requires
\(\varepsilon_c(X)<\varepsilon_c(\Gamma)\). Under smearing \(E_F\) must sit
near the gap window:

\[
E_\mathrm{vbm}-4\sigma \le E_F \le E_\mathrm{cbm}+4\sigma,
\qquad \sigma=\texttt{degauss}.
\]

**DOS.** Uniform MP mesh of size `Dos::k_mesh`³, Gaussian broadening of the NSCF
spectrum (`gaussian_dos` in `bands_dos.h`). Restricted, with the closed-shell
weight 2 absorbed into the kernel:

\[
D(E_i)
  = \sum_{n\mathbf{k}} w_{\mathbf{k}}\,
    \frac{2}{\sqrt{2\pi}\,\sigma}\,
    \exp\!\Bigl(-\frac{(E_i-\varepsilon_{n\mathbf{k}})^2}{2\sigma^2}\Bigr);
\]

collinear, per spin with spin-orbital weight 1:

\[
D_\sigma(E_i)
  = \sum_{n\mathbf{k}} w_{\mathbf{k}}\,
    \frac{1}{\sqrt{2\pi}\,\sigma}\,
    \exp\!\Bigl(-\frac{(E_i-\varepsilon_{n\mathbf{k}}^\sigma)^2}{2\sigma^2}\Bigr),
\qquad
D(E) = D_\alpha(E) + D_\beta(E).
\]

Units: states / Ry / cell. The occupied integral
`DosResult::n_elec_to_vbm` is \(\int_{-\infty}^{E_\ast} D(E)\,dE \approx n_e = 8\)
with \(E_\ast = E_\mathrm{vbm}\) (joint VBM of both spins) for fixed occupations
and \(E_\ast = E_F\) under smearing — there `Dos` recomputes \(\mu\) from the
denser-mesh eigenvalues so that \(\sum w f = n_e\) (`DosResult::e_fermi_ry`).
`DosResult::gap_ry` is \(E_\mathrm{cbm}-E_\mathrm{vbm}\) over the DOS-mesh
eigenvalues of both spins.

**Gapped Si under smearing.** At the tutorial ecut/k diamond Si stays an
open-gap semiconductor: FD occupations remain near-integer and \(E_F\) sits in
the gap. The metallic sections therefore gate the **metallic-occupation code
path** (no throw) plus an open spectrum and a physically placed \(E_F\) — not a
vanishing metal-like gap, and not a claim that Si is a metal or a ferromagnet.
True-metal free energies are [pw-smearing-al](pw-smearing.md#pw-smearing-al) /
[pw-uks-fe-smearing](pw-smearing.md#pw-uks-fe-smearing).

**Observables.**

| Symbol | Definition | Unit |
| ------ | ---------- | ---- |
| \(\varepsilon_n^\alpha(\mathbf{k})\), \(\varepsilon_n^\beta(\mathbf{k})\) | `BandsResult::eigenvalues_ry` / `eigenvalues_beta_ry`; `kfrac` | Ry |
| \(n_\mathrm{spin}\) | `BandsResult::nspin` / `DosResult::nspin` | — |
| \(M\) | `magnetization` \(=N_\uparrow-N_\downarrow\) | e⁻ |
| \(E_F\) | `e_fermi_ry` (SCF μ for bands, DOS-mesh μ for DOS) | Ry |
| \(E_\mathrm{vbm}\), \(E_\mathrm{cbm}\), \(E_g\) | `e_vbm_ry` / `e_cbm_ry` / `gap_ry` | Ry |
| \(E_{\Gamma\to X}\) | \((\varepsilon_c(X)-\varepsilon_v(\Gamma))\times 13.605693\) | eV |
| \(D(E)\), \(D_\alpha\), \(D_\beta\) | `DosResult::dos` / `dos_alpha` / `dos_beta` on bins `energy_ry` | 1/Ry/cell |
| \(n_\mathrm{elec}\) | `DosResult::n_elec_to_vbm` (to VBM, or to \(E_F\) under smear) | electrons/cell |
| \(E_\mathrm{SCF}\) / \(F\) | `BandsResult::energy_ry` / `DosResult::scf_energy_ry` | Ry |

Conversion: `cpu/planewave/qe_units.h` (`rydberg_to_ev = 13.605693`).
Implementation `src/cpu/planewave/bands_dos.cpp` (SCF → freeze
\(V_{\mathrm{eff},\sigma}\) → NSCF Davidson on path / DOS mesh → \(\mu\) +
Gaussian DOS); occupations in `src/cpu/planewave/internal/smearing.*`. No unit
test in the tree covers the path endpoints, Gaussian integral identity,
spin-degeneracy kernel, `integrate_dos_to`, synthetic gap finder or the
occupation closed forms.

## Common setup

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si, 2-atom fcc primitive (Z=14) |
| \(a\) | 5.43 Å |
| Basis atoms | \((0,0,0)\), \((a/4,a/4,a/4)\) |
| Cell | \((a/2)(0,1,1)\), \((a/2)(1,0,1)\), \((a/2)(1,1,0)\) |
| PP | `tests/data/pw/Si_lda_nc.upf` (in-tree), PseudoDojo ONCVPSP-LDA-SR-noCC v0.4, ONCVPSP 3.3.0 (D. R. Hamann), \(Z_\mathrm{val}=4\), no NLCC — same family as `pw-lda-si` |
| SCF \(E_\mathrm{cut}\) | 15 Ry (tutorial; ONCV production ~60 Ry) |
| SCF k-mesh | \(2\times 2\times 2\) (`PW::k_mesh = 2`) |
| XC | `XCType::LSDA` |
| SCF tol | \(10^{-6}\) (density change) |
| Bands path | Γ–X–W–L–Γ–K, `n_seg=4`, `n_bands=8` per spin |
| DOS | MP `k_mesh=4`, `n_bands=8`, \(\sigma=0.05\) Ry, 200 bins |
| Smearing (metallic sections) | Fermi–Dirac, `degauss = 0.02` Ry, `extra_bands = 4` |

```cpp
constexpr double kA = 5.43;       // Å
cpu::Frame frame(2);
frame.span(kAtom_atomic_number)[0] = 14;
frame.span(kAtom_atomic_number)[1] = 14;
frame.span(kAtom_x)[1] = kA / 4.0;
frame.span(kAtom_y)[1] = kA / 4.0;
frame.span(kAtom_z)[1] = kA / 4.0;
const double h = kA / 2.0;
Mat3 cell;                        // fcc primitive: (0,h,h), (h,0,h), (h,h,0)
cell(0, 0) = 0; cell(0, 1) = h; cell(0, 2) = h;
cell(1, 0) = h; cell(1, 1) = 0; cell(1, 2) = h;
cell(2, 0) = h; cell(2, 1) = h; cell(2, 2) = 0;
frame.box.set_cell(cell);

PW method{std::make_shared<UpfPseudopotential>(upf), /*ecut Ry*/ 15.0, /*k_mesh 2×2×2*/ 2,
          cpu::scf::XCType::LSDA};
method.tol = 1e-6;
// per section: max_iter, mix_beta, spin_polarized / multiplicity,
//              smearing = Smearing::fermi_dirac, degauss = 0.02, extra_bands = 4

Bands bands;
bands.n_bands = 8;
bands.set_diamond_path(4);              // Γ–X–W–L–Γ–K
const BandsResult br = bands.compute(method, frame);

Dos dos;
dos.k_mesh = 4;
dos.n_bands = 8;
dos.sigma_ry = 0.05;
dos.n_bins = 200;
const DosResult dr = dos.compute(method, frame);
```

## Oracle

**Literature pin (indirect gap).** Bulk Si is an **indirect** gap semiconductor:
VBM at \(\Gamma\), CBM near \(X\).

| Quantity | Value | Source |
| -------- | ----- | ------ |
| Experimental \(E_g\) | ~1.12 eV | textbook (context only; LDA underestimates) |
| Converged LDA \(E_g\) (Γ\(_v\) → near-\(X_c\)) | **~0.45 eV** | Yin & Cohen, *Phys. Rev. B* **26**, 5668 (1982) |

The band sections gate a **literature-anchored window** \(0.05 \le
E_{\Gamma\to X} \le 2.50\) eV around the Yin–Cohen value: the SCF ecut and k
are deliberately light (same class as `pw-lda-si`), and the wide window allows
residual path scatter while rejecting a metallic collapse or a multi-eV
disaster. The DOS sections gate character (electron count, open gap, channel
consistency, \(E_F\) placement). None is a freeze of Atomiverse-only digits.

**CP2K.** Quickstep `BAND_STRUCTURE` / `PRINT%PDOS` (with `LSD` /
`MULTIPLICITY 1` for spin and `&SMEAR METHOD FERMI_DIRAC` for metallic
occupations) uses **GTH-PADE + DZVP-GTH**, not ONCV plane waves, so absolute
\(\varepsilon_n\) / \(D(E)\) are **not** comparable; the shared content is the
character (indirect Si, \(M=0\), open gap, \(E_F\) in the gap, occupied
integral ≈ 8). The offline CP2K inputs use the same geometry, PADE,
GTH-PADE-q4 + DZVP-GTH-q4, MP \(2\times2\times2\), path Γ–X–W–L–Γ–K with
`NPOINTS 4` or `PDOS` with `NLUMO 4`, and for smearing
\(k_B T \approx 0.02\) Ry (`ELECTRONIC_TEMPERATURE [K] 3157.75`, ≈ 3158 K) with
`ADDED_MOS ≥ 4`; the preferred oracle is CP2K 2026.1+ Quickstep. No CP2K binary
is available on the build host (`module avail cp2k` empty), and nothing runs
CP2K at ctest time.

**Performance (all sections): `perf_exempt`** — no CP2K binary / module on the
build host for a fair wall twin, and absolute eigenvalues are not a CP2K twin;
\(T_\mathrm{CP2K}\), \(r = T_\mathrm{ATV}/T_\mathrm{CP2K}\) and the ≤10× check
are n/a. Every section is inside the ≤ 60 s single-rank CPU budget. When a CP2K
binary is available offline, time the same geometry, record \(r\) in
REFERENCE.md and require \(r\le 10\) (same-geometry wall only, not
absolute-eigenvalue parity, GTH+Gaussian vs NC-UPF PW).

## Limitations (all sections)

- Tutorial ecut / SCF k only — not production-converged band gaps or dense DOS;
  production \(E_\mathrm{cut}\) (~60 Ry) / dense k break the ≤ 60 s budget.
- Absolute \(\varepsilon_n\) / \(D(E)\) not comparable to CP2K Quickstep GTH decks.
- DOS-mesh gap values are a **positivity** gate, not a Yin–Cohen window (that
  lives on the restricted Γ→X band path).
- No GPW / Gaussian k-path (`gpw-kpoints-si`); no noncollinear / spinor KS; no
  CUDA PW bands / DOS; no MPI multi-rank scaling gate
  ([mpi-pw-scaling-4rank](pw-rks.md#mpi-pw-scaling-4rank) covers the SCF).
- Berry phase, Wannier and DFPT response are separate methods
  ([berry-spin-si](pw-berry-phase.md#berry-spin-si),
  [periodic-wannier-si](pw-wannier.md#periodic-wannier-si),
  [dfpt-metallic-si](response-dfpt.md#dfpt-metallic-si), `dfpt-spin-si`).
- Empty `upf_path` selects a toy \(V_\mathrm{loc}\) (unphysical); every section
  uses the in-tree ONCV Si UPF.

## References

- M. T. Yin and M. L. Cohen, *Phys. Rev. B* **26**, 5668 (1982)
- D. R. Hamann, *Phys. Rev. B* **88**, 085117 (2013)
- M. J. van Setten et al., *Comput. Phys. Commun.* **226**, 39 (2018)
- M. Methfessel and A. T. Paxton, *Phys. Rev. B* **40**, 3616 (1989)
- N. D. Mermin, *Phys. Rev.* **137**, A1441 (1965)
- [Chapter 8 — Plane-Wave DFT](../ch08-plane-wave.md)

## bands-dos-si — Restricted band structure + DOS of bulk Si {#bands-dos-si}

Restricted LDA SCF with fixed occupations (`max_iter = 50`), then `Bands` on the
path and `Dos` on the \(4^3\) mesh. Observables: \(\varepsilon_n(\mathbf{k})\),
the Γ→X gap and the occupied DOS integral to the VBM.

| Quantity | Atomiverse / pin | External | Tol / band | Notes |
| -------- | ---------------: | -------: | ---------- | ----- |
| \(E_{\Gamma\to X}\) | band gate | Yin–Cohen ~0.45 eV | **[0.05, 2.50] eV** | |
| Measured (informative) | **≈ 0.48 eV** | ~0.45 eV | inside window | Release; near the Yin–Cohen value despite the tutorial settings |
| Local same-\(k\) gap | \(> 0\) | open insulator | strict | every path k |
| Indirect character | \(\varepsilon_c(X)<\varepsilon_c(\Gamma)\) | Si textbook | boolean | path samples include Γ and X |
| Direct \(\Gamma\) vs Γ→X | \(E_g^\mathrm{dir}(\Gamma) \ge E_{\Gamma\to X}-0.05\) eV | Si | −0.05 eV slack | direct larger than indirect |
| \(n_\mathrm{elec\_to\_vbm}\) | \(\approx 8\) | \(n_e=8\) | **±1.0** | Gaussian DOS on tutorial mesh |
| SCF | `scf_converged == true` | — | — | both `Bands` and `Dos` |
| Eigenvalue order | sorted ascending per k | — | — | 8 bands |

ATV wall (informative): ~22 s (Release, 8 OMP, ecut 15 / k=2 + path + DOS k=4).

Design record: git `034fed2e` (`.claude/specs/bands-dos-si/`).

## bands-metallic-si — Metallic-occupation band structure of Si {#bands-metallic-si}

Restricted LDA SCF with Fermi–Dirac smearing (`max_iter = 80`,
`mix_beta = 0.5`, `degauss = 0.02` Ry, `extra_bands = 4`), then `Bands` on the
path. \(E_g\) uses the integer-fill cut; \(E_F\) is the SCF \(\mu\).

```cpp
method.max_iter = 80;
method.mix_beta = 0.5;
method.smearing = Smearing::fermi_dirac;
method.degauss = 0.02;            // Ry
method.extra_bands = 4;
const BandsResult br = bands.compute(method, frame);
// br.e_fermi_ry; br.gap_ry; br.e_vbm_ry / e_cbm_ry; br.energy_ry = Mermin F
```

| Quantity | Atomiverse / pin | External | Tol / band | Notes |
| -------- | ---------------: | -------: | ---------- | ----- |
| SCF | `scf_converged == true` | — | — | metallic free-energy SCF |
| `nspin` | 1 | restricted | strict | |
| Path endpoints | includes Γ and X | diamond path | boolean | `set_diamond_path` |
| Eigenvalue order | sorted ascending per k | — | — | 8 bands |
| Local same-\(k\) gap | \(\varepsilon_4-\varepsilon_3 > 0\) | open insulator | strict | every path k |
| Integer-fill \(E_g\) | `gap_ry > 0` | open Si | strict | not a metal-collapse gate |
| \(E_F\) finite | \(\lvert E_F\rvert < 10^3\) Ry | SCF μ | strict | `e_fermi_ry` |
| \(E_F\) placement | in \([E_\mathrm{vbm},E_\mathrm{cbm}]\pm 4\sigma\) | gapped FD | \(\sigma=0.02\) Ry | near-integer fill |
| Measured (informative) | \(E_F\approx 0.465\) Ry, \(E_g\approx 0.043\) Ry, \(F\approx -15.686\) Ry | SCF 6 iters | — | Release OMP=8 |

ATV wall (informative): ~14.3 s (Release, 8 OMP, ecut 15 / k=2 + path + FD).

## bands-spin-si — Collinear-spin band structure of Si {#bands-spin-si}

Collinear LDA SCF with fixed occupations (`spin_polarized = true`,
`multiplicity = 1` ⇒ \(M=0\), `max_iter = 80`), then `Bands` per spin. The
Γ→X window is taken on α; at Γ VBM and X CBM the closed-shell degeneracy
requires \(\bigl|\varepsilon^\alpha - \varepsilon^\beta\bigr| \le 5\times
10^{-4}\,\mathrm{Ry}\).

```cpp
method.max_iter = 80;
method.spin_polarized = true;
method.multiplicity = 1;          // M = 0 closed-shell collinear
const BandsResult br = bands.compute(method, frame);
// br.nspin == 2; br.eigenvalues_ry (α); br.eigenvalues_beta_ry (β); br.magnetization ≈ 0
```

| Quantity | Atomiverse / pin | External | Tol / band | Notes |
| -------- | ---------------: | -------: | ---------- | ----- |
| \(E_{\Gamma\to X}\) (α) | band gate | Yin–Cohen ~0.45 eV | **[0.05, 2.50] eV** | |
| \(n_{\mathrm{spin}}\) | **2** | collinear | exact | `BandsResult::nspin` |
| \(\lvert M\rvert\) | \(\le 10^{-6}\) | \(M=0\) fixed occ | **1e-6** | |
| Local same-\(k\) gap (α and β) | \(> 0\) | open insulator | strict | every path k, both spins |
| Indirect character | \(\varepsilon_c^\alpha(X)<\varepsilon_c^\alpha(\Gamma)\) | Si textbook | boolean | |
| α≈β at Γ VBM, X CBM | \(\lvert\Delta\varepsilon\rvert\le 5\times 10^{-4}\) Ry | \(M=0\) degeneracy | **5e-4 Ry** | |
| SCF | `scf_converged == true` | — | — | |
| Eigenvalue order | sorted ascending per (spin, k) | — | — | 8 bands each channel |

ATV wall (informative): ~16 s (Release, OMP=8, ecut 15 / k=2 + collinear path).

## bands-spin-metallic-si — Collinear-spin metallic-occupation band structure of Si {#bands-spin-metallic-si}

Collinear LDA SCF with Fermi–Dirac smearing (`spin_polarized = true`,
`multiplicity = 1` seeding \(M=0\), `max_iter = 80`, `mix_beta = 0.5`,
`degauss = 0.02` Ry, `extra_bands = 4`), then `Bands` per spin. Under smear
\(M\) is free and must relax to \(\lvert M\rvert\le 10^{-3}\); α≈β parity at Γ
VBM and X CBM is \(\lvert\varepsilon^\alpha-\varepsilon^\beta\rvert \le
10^{-3}\) Ry.

```cpp
method.max_iter = 80;
method.mix_beta = 0.5;
method.spin_polarized = true;
method.multiplicity = 1;              // seed M=0; free under smear
method.smearing = Smearing::fermi_dirac;
method.degauss = 0.02;
method.extra_bands = 4;
const BandsResult br = bands.compute(method, frame);
// br.nspin == 2; br.e_fermi_ry; br.gap_ry; br.magnetization; br.energy_ry = Mermin F
```

| Quantity | Atomiverse / pin | External | Tol / band | Notes |
| -------- | ---------------: | -------: | ---------- | ----- |
| SCF | `scf_converged == true` | — | — | collinear metallic free-energy SCF |
| `nspin` | 2 | collinear | strict | α + β channels |
| \(\lvert M\rvert\) | \(\le 10^{-3}\) | seed \(M=0\) | strict | free under smear |
| Path endpoints | includes Γ and X | diamond path | boolean | |
| Eigenvalue order | sorted ascending per (k, spin) | — | — | 8 bands/spin |
| Local same-\(k\) gap | \(\varepsilon_4^\sigma-\varepsilon_3^\sigma > 0\) | open insulator | strict | every path k, both spins |
| Integer-fill \(E_g\) | `gap_ry > 0` | open Si | strict | not a metal-collapse gate |
| \(E_F\) finite | \(\lvert E_F\rvert < 10^3\) Ry | SCF μ | strict | |
| \(E_F\) placement | in \([E_\mathrm{vbm},E_\mathrm{cbm}]\pm 4\sigma\) | gapped FD | \(\sigma=0.02\) Ry | |
| Γ→X (α) | band gate | Yin–Cohen window | **[0.05, 2.50] eV** | |
| Indirect character | \(\varepsilon_c^\alpha(X)<\varepsilon_c^\alpha(\Gamma)\) | Si textbook | boolean | |
| α≈β | \(\lvert\Delta\varepsilon\rvert\le 10^{-3}\) Ry | \(M\approx 0\) | at Γ VBM, X CBM | |
| Measured (informative) | \(E_F\approx 0.465\) Ry, \(E_g\approx 0.043\) Ry, mag=0, Γ→X(α)≈0.58 eV, \(F\approx -15.686\) Ry | — | — | Release |

ATV wall (informative): ~45 s (Release, ecut 15 / k=2 + path + collinear FD).
A finite-magnetization metal is not pinned here.

## dos-metallic-si — Metallic-occupation DOS of Si {#dos-metallic-si}

Restricted LDA SCF with Fermi–Dirac smearing (`max_iter = 80`,
`mix_beta = 0.5`, `degauss = 0.02` Ry, `extra_bands = 4`), then `Dos` on the
\(4^3\) mesh with \(\mu\) recomputed there; the occupied integral is to \(E_F\):
\(n_{\mathrm{elec}}(E_F)=\int_{-\infty}^{E_F} D(E)\,dE \approx n_e\).

```cpp
method.max_iter = 80;
method.mix_beta = 0.5;
method.smearing = Smearing::fermi_dirac;
method.degauss = 0.02;            // Ry
method.extra_bands = 4;
const DosResult dr = dos.compute(method, frame);
// dr.e_fermi_ry, dr.n_elec_to_vbm (integral to E_F), dr.gap_ry, dr.dos
```

| Quantity | Atomiverse / pin | External | Tol / band | Notes |
| -------- | ---------------: | -------: | ---------- | ----- |
| SCF | `scf_converged == true` | — | — | FD free-energy SCF |
| \(E_F\) | finite | finite | \(\lvert E_F\rvert < 10^3\) Ry | DOS-mesh \(\mu\) |
| \(n_e(E_F)\) | \(\approx 8\) | \(n_e=8\) | **±1.0** | `n_elec_to_vbm` = \(\int^{E_F} D\) |
| Measured \(n_e(E_F)\) (informative) | **≈ 8.01** | 8 | inside tol | Release |
| Measured \(E_F\) (informative) | **≈ 0.465 Ry** | — | — | sits in gap |
| Measured \(F\) (informative) | **≈ −15.686 Ry** | — | not a CP2K twin | SCF free energy, 6 iters |
| Gap on DOS mesh | `gap_ry > 0` | open insulator | strict | gapped Si + FD |
| \(E_F\) placement | in \([E_\mathrm{vbm},E_\mathrm{cbm}]\) | textbook Si | **±4×degauss** | \(\mathrm{degauss}=0.02\) Ry |
| DOS grid | non-empty, `energy_ry.size()==dos.size()` | — | — | 200 bins |

ATV wall (informative): ~23.5 s (Release, 8 OMP, ecut 15 / k=2 + DOS k=4 + FD).
Metallic DFPT is [dfpt-metallic-si](response-dfpt.md#dfpt-metallic-si); GPW
`&SMEAR` is a separate GPW path.

## dos-spin-si — Collinear-spin DOS of Si {#dos-spin-si}

Collinear LDA SCF with fixed occupations (`spin_polarized = true`,
`multiplicity = 1` ⇒ \(M=0\), `max_iter = 80`, `mix_beta = 0.5`), then `Dos`
per spin. \(n_\mathrm{elec\_to\_vbm}\) integrates to the joint VBM; on \(M=0\)
each channel holds \(\approx 4\) electrons.

```cpp
method.max_iter = 80;
method.mix_beta = 0.5;
method.spin_polarized = true;
method.multiplicity = 1;            // M = 0; N↑ = N↓ = 4
const DosResult dr = dos.compute(method, frame);
// dr.nspin == 2; dr.dos / dos_alpha / dos_beta; dr.n_elec_to_vbm ≈ 8; dr.gap_ry > 0
```

| Quantity | Atomiverse / pin | External | Tol / band | Notes |
| -------- | ---------------: | -------: | ---------- | ----- |
| SCF | `scf_converged == true` | — | — | collinear LDA, 13 iters |
| \(nspin\) | **2** | collinear | exact | `DosResult::nspin` |
| \(M = N_\uparrow-N_\downarrow\) | **0** | singlet Si | **\(\lvert M\rvert\le 10^{-6}\)** | fixed occupations |
| \(n_\mathrm{elec\_to\_vbm}\) | **≈ 7.96** (measured) | \(n_e=8\) | **±1.0** | |
| Per-spin occupied | \(n_\alpha, n_\beta \approx 4\) | \(M=0\) | **±1.0** each | trap integral to joint VBM |
| \(D\) vs \(D_\alpha+D_\beta\) | \(\lvert\int(D_\alpha+D_\beta)-\int D\rvert\) | identity | **≤ 0.05** | shared energy grid |
| Gap on DOS mesh | **> 0** (`gap ≈ 0.035` Ry measured) | open insulator | strict `gap_ry > 0` | |
| \(E_\mathrm{SCF}\) (informative) | **≈ −15.672** Ry | — | — | not a CP2K energy twin |
| Channel lengths | `\|dos_alpha\|=\|dos_beta\|=\|dos\|` | — | exact | 200 bins |

ATV wall (informative): ~37.5 s (Release, 8 OMP, ecut 15 / k=2 + DOS k=4 +
collinear).

## dos-spin-metallic-si — Collinear-spin metallic-occupation DOS of Si {#dos-spin-metallic-si}

Collinear LDA SCF with Fermi–Dirac smearing (`spin_polarized = true`,
`multiplicity = 1` seeding \(N_\uparrow=N_\downarrow=4\), `max_iter = 80`,
`mix_beta = 0.5`, `degauss = 0.02` Ry, `extra_bands = 4`), then `Dos` per spin
with \(\mu\) recomputed on the DOS mesh; `n_elec_to_vbm` is the integral to
\(E_F\) and each channel holds \(\approx 4\) electrons to the same \(E_F\).

```cpp
method.max_iter = 80;
method.mix_beta = 0.5;
method.spin_polarized = true;
method.multiplicity = 1;                 // seed M = 0; N↑ = N↓ = 4
method.smearing = Smearing::fermi_dirac;
method.degauss = 0.02;
method.extra_bands = 4;
const DosResult dr = dos.compute(method, frame);
// dr.nspin == 2; dr.n_elec_to_vbm ≈ 8 (to E_F); dr.e_fermi_ry; |dr.magnetization| ≤ 0.1
```

| Quantity | Atomiverse / pin | External | Tol / band | Notes |
| -------- | ---------------: | -------: | ---------- | ----- |
| SCF | `scf_converged == true` | — | — | collinear FD free-energy SCF, 13 iters |
| \(nspin\) | **2** | collinear | exact | |
| \(M = N_\uparrow-N_\downarrow\) | **0** (measured) | singlet Si | **\(\lvert M\rvert\le 0.1\)** | free under smear; seed \(M=0\) |
| \(n_\mathrm{elec}(E_F)\) | **≈ 8.01** (measured) | \(n_e=8\) | **±1.0** | `n_elec_to_vbm` |
| Per-spin occupied | \(n_\alpha, n_\beta \approx 4\) | \(M\approx 0\) | **±1.0** each | trap integral to \(E_F\) |
| \(D\) vs \(D_\alpha+D_\beta\) | \(\lvert\int(D_\alpha+D_\beta)-\int D\rvert\) to \(E_F\) | identity | **≤ 0.05** | |
| Gap on DOS mesh | **> 0** (`gap ≈ 0.035` Ry measured) | open insulator | strict `gap_ry > 0` | |
| \(E_F\) finite | \(\lvert E_F\rvert < 10^3\) Ry | DOS-mesh \(\mu\) | strict | |
| \(E_F\) placement | **≈ 0.473** Ry in \([E_\mathrm{vbm},E_\mathrm{cbm}]\pm 4\sigma\) | gapped FD | \(\sigma=0.02\) Ry | |
| \(F\) (informative) | **≈ −15.672** Ry | — | — | not a CP2K energy twin |
| Channel lengths | `\|dos_alpha\|=\|dos_beta\|=\|dos\|` | — | exact | 200 bins |

ATV wall (informative): ~36.6 s (Release, 8 OMP, ecut 15 / k=2 + DOS k=4 +
collinear + FD).
