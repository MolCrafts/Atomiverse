# Plane-wave restricted Kohn–Sham DFT

> The regression runner has been removed. Regression commands below are historical; case sources and oracle records are retained as archived evidence, not current executable checks.

Spin-unpolarized periodic plane-wave Kohn–Sham DFT with norm-conserving
pseudopotentials and Monkhorst–Pack k-point sampling. The concrete method is
`cpu::pw::PW{std::make_shared<UpfPseudopotential>(path), ecut_ry, k_mesh, xc}` with members
(`tol`, `max_iter`, `mix_beta`, `pulay_depth`, …) — no `*Config`, no factories.
`build(frame)` → `run(frame)` returns a `PWResult` (energy in Ry / Ha,
`converged`, `iterations`, `n_kpts`, `n_kpts_local`); `PW::forces(frame)`
writes analytic ionic forces to `kAtom_f*` [eV/Å] and `kEnergy` [eV];
`PW::stress(frame)` returns the Nielsen–Martin Voigt stress [eV/Å³] and writes
\(W = \sigma\Omega\) [eV] to `kStress`; `cpu::single_point(frame, method)`
(Evaluate driver) writes `kEnergy` [eV]. A Driver can mount a moved `PW` as a
nested node; it then writes `kStress` when the plan consumes stress. The
sections cover CPU energy (LDA and true PBE GGA), symmetry forces, analytic
stress and MPI k-point scaling of bulk silicon.

The case also carries sections without a tutorial page; their oracle records
live in the case `REFERENCE.md`: `h_force_analytic` / `h_force_fd` (analytic
and FD PW forces vanish for a single atom by symmetry), `h_stress_cubic`
(cubic monoatomic stress off-diagonals vanish), `pw-cuda-lda-si` (the
[pw-lda-si](#pw-lda-si) literature window run with `--device=cpu|cuda|both`),
`cuda-pw-forces` (stretched Si dimer, same-settings
max\(|F_\mathrm{cuda}-F_\mathrm{cpu}|<5\times10^{-3}\) eV/Å) and
`cuda-pw-stress` / `cuda-pw-stress-pbe` (diamond Si LDA / PBE stress,
max\(|\sigma_\mathrm{cuda}-\sigma_\mathrm{cpu}|<5\times10^{-2}\) eV/Å³ plus
cubic symmetry).

Case: `ctest -R '^pw/rks$'` — sources and oracle record in [`regressions/pw/rks/`](../../../regressions/pw/rks/). One section of the case per `##` heading below: `atv_regressions --case pw/rks --section <name>`.

Roadmap: [dft.md](../../roadmap/dft.md) § Periodic DFT / plane-wave; § Periodic
response — analytic stress. Requires `ATV_BUILD_PW=ON` (CPU build environment
`buildenv-gcc/2026.03-mpich`); the MPI section also needs an MPI build.

## Method

Bloch states expand in plane waves truncated by the kinetic cutoff:

\[
\psi_{n\mathbf{k}}(\mathbf{r})
  = \sum_{\mathbf{G}} c_{n\mathbf{k}}(\mathbf{G})\,
    e^{i(\mathbf{k}+\mathbf{G})\cdot\mathbf{r}},
\qquad
\tfrac12|\mathbf{k}+\mathbf{G}|^2 \le E_\mathrm{cut}.
\]

The SCF loop (owned by `PW`) is: Poisson
\(V_\mathrm{H}(\mathbf{G}) = 4\pi\rho(\mathbf{G})/G^2\) (\(G\neq 0\)) → XC on the
FFT real-space grid → \(V_\mathrm{eff}\) → Davidson diagonalization → density +
Pulay mix → residual. Ion–ion energy is 3D Ewald. Core electrons are replaced by
a **norm-conserving** pseudopotential (UPF). Spin-unpolarized, fixed
closed-shell occupations (`Smearing::none`); uniform Monkhorst–Pack mesh of
size `k_mesh`³. Total energy is in Rydberg on `PWResult`; the Frame gets eV with
\(E[\mathrm{eV}] = E[\mathrm{Ry}]\times 13.605693\) (`cpu/planewave/qe_units.h`).

`PW`'s XC default may be GGA-PBE; pure LDA (Slater + PW92,
`cpu::scf::XCType::LSDA`) must be passed to the constructor. Each section passes its in-tree UPF in the list of element providers.

## Common setup

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si, 2-atom fcc primitive (Z=14, mass 28.0855) |
| \(a\) | 5.43 Å (experimental Si) |
| Basis atoms | \((0,0,0)\), \((a/4,a/4,a/4)\) |
| Cell | \((a/2)(0,1,1)\), \((a/2)(1,0,1)\), \((a/2)(1,1,0)\) |
| \(E_\mathrm{cut}\) | 15 Ry (tutorial; ONCV production ~60 Ry) |
| k-mesh | \(2\times 2\times 2\) (`k_mesh = 2`, 8 k-points) |

**Pseudopotential.** `tests/data/pw/Si_lda_nc.upf` (in-tree), PseudoDojo
ONCVPSP-LDA-SR-noCC v0.4, generated with ONCVPSP 3.3.0 (D. R. Hamann; LDA,
iexc=3, PZ), \(Z_\mathrm{val}=4\), no NLCC. Upstream:
<https://github.com/PseudoDojo/ONCVPSP-LDA-SR-noCCv0.4> (`Si/Si.upf`).

```cpp
constexpr double kA = 5.43;      // Å
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
```

## Oracle

**ABINIT-class same-pseudo pin.** From PseudoDojo `Si/Si.djrepo` deltafactor
key `"8.0"`, `num_sites = 2` (converged limit of the same pseudopotential):

| Quantity | Value |
| -------- | ----: |
| \(E_\mathrm{min}\) | **−230.158202 eV/cell** = **−16.916316 Ry/cell** |
| \(v_0\) | 20.43445 Å³/atom → \(a_0 \approx 5.468\) Å (conventional cubic) |
| \(B_0\) | 89.96 GPa |

The \(2\times2\times2\) / 15 Ry setup is deliberately under-converged, so
\(E_\mathrm{ATV}\) is **higher** (less bound) than \(E_\mathrm{min}\). The
energy gate is the band \(E_\mathrm{min} \le E_\mathrm{ATV} \le
E_\mathrm{min}+1.50\) Ry, i.e. \([-16.916316,\,-15.416316]\) Ry: the lower edge
is the external pin and the 1.50 Ry headroom rejects unit disasters while
allowing the known incomplete-basis elevation at \(a=5.43\) Å — not a golden
of Atomiverse output.

**CP2K.** Quickstep bulk-Si decks use **GTH-PADE + DZVP-GTH**; absolute total
energies and `STRESS_TENSOR` values are **not** comparable to ONCV plane waves —
the fair cross-code science metric is the EOS (\(a_0\), \(B_0\)) / structural
response. The CP2K inputs (same geometry / \(2\times 2\times 2\) MP) are offline
wall-only skeletons. Nothing runs CP2K, ABINIT or QE at ctest time.

**Performance (all CPU single-rank sections): `perf_exempt`** — no CP2K / QE /
ABINIT binary on the build host for a fair NC-UPF PW wall twin;
\(T_\mathrm{CP2K}\), \(r = T_\mathrm{ATV}/T_\mathrm{CP2K}\) and the ≤10× check
are n/a. When a plane-wave oracle binary is available offline, time the same
geometry, record \(r\) in REFERENCE.md and require \(r\le 10\) (same-geometry
wall only, not absolute-energy parity).

## Limitations (all sections)

- Tutorial ecut/k only — not production-converged absolute energies, stress
  magnitudes or EOS; production ONCV cutoffs (~60 Ry) and dense k-meshes exceed
  the ≤ 60 s e2e budget.
- Restricted closed shell only: collinear spin is
  [pw-spin-polarized-o2](pw-collinear-spin.md#pw-spin-polarized-o2), metallic
  smearing [pw-smearing-al](pw-smearing.md#pw-smearing-al), PW BOMD
  `pw-bomd-h`.
- No hybrid / meta-GGA XC on PW.
- The CUDA PW surface is covered only by the CPU↔CUDA sections listed above.

## References

- J. P. Perdew, K. Burke, M. Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996) — PBE
- O. H. Nielsen and R. M. Martin, *Phys. Rev. B* **32**, 3780 (1985) — stress
- D. R. Hamann, *Phys. Rev. B* **88**, 085117 (2013) — ONCVPSP
- M. J. van Setten et al., *Comput. Phys. Commun.* **226**, 39 (2018) — PseudoDojo
- PseudoDojo ONCVPSP-LDA-SR-noCCv0.4 `Si.djrepo` deltafactor \(E_\mathrm{min}\) / EOS
- [Chapter 8 — Plane-Wave DFT](../ch08-plane-wave.md)

## pw-lda-si — LDA total energy of bulk Si (k-mesh) {#pw-lda-si}

### Summary

Plane-wave Kohn–Sham LDA total energy of diamond Si on the
[Common setup](#common-setup), run through the Evaluate driver:
Frame in, `PW` members, energy on `PWResult` [Ry/Ha] and
`frame.meta.get(kEnergy)` [eV], gated against the PseudoDojo / ABINIT-class
band (not CP2K Quickstep absolute energy).

```cpp
PW method{{std::make_shared<UpfPseudopotential>(ATV_REG_DATA_DIR "/pw/Si_lda_nc.upf")}, /*ecut Ry*/ 15.0,
          /*k_mesh 2×2×2*/ 2, cpu::scf::XCType::LSDA};
method.tol = 1e-6;
method.max_iter = 50;

cpu::single_point(frame, method);

const auto& r = method.last_result();         // r.energy [Ry], r.converged, r.iterations
const double e_ev = frame.meta.get(kEnergy);  // [eV]
```

### Validation

SCF tol \(10^{-6}\) (density change), `max_iter` 50.

| Quantity | Atomiverse / pin | External | Tol / band | Notes |
| -------- | ---------------: | -------: | ---------- | ----- |
| \(E_\mathrm{min}\) (converged PP) | — | **−16.916316 Ry** | literature pin | PseudoDojo / ABINIT-class |
| \(E_\mathrm{tot}\) | band gate | same \(E_\mathrm{min}\) | \([-16.916316,\,-15.416316]\) Ry | \(E_\mathrm{min} \le E_\mathrm{ATV} \le E_\mathrm{min}+1.50\) Ry |
| Measured (informative) | **≈ −15.670898 Ry** (5 iters) | — | inside band (~1.25 Ry above \(E_\mathrm{min}\)) | Release, 16 OMP |
| Frame \(E\) [eV] | \(E_\mathrm{Ry}\times 13.605693\) | engine conversion | \(10^{-4}\) eV | `qe_units.h` |
| SCF | `converged == true` | — | — | |

### Performance

ATV wall ≈ 7–8 s (informative), inside the ≤ 60 s budget; `perf_exempt`.

### Limitations

Energy only; forces / stress / GGA are the other sections.

## pw-pbe-gga-si — True PBE GGA on bulk Si (energy + symmetry forces) {#pw-pbe-gga-si}

### Summary

Plane-wave KS with the **true GGA PBE** functional on the FFT grid: density
gradients enter both \(E_{\mathrm{xc}}\) and \(v_{\mathrm{xc}}\) — not PBE called
with \(\sigma = 0\). No CP2K / QE binary is available on the pin host
(`module avail cp2k` / `qe` empty; no `cp2k.psmp` / `pw.x` on `PATH`), so
correctness uses crystal-symmetry force zeros plus a PBE–LSDA energy-gap
activation pin; no absolute total energy is claimed.

### Method

When `is_gga(xc)` (e.g. `XCType::PBE`), spectral \(\nabla\rho\) is built from
\(\rho(G)\); pointwise \(e_{\mathrm{xc}}(\rho,\sigma)\) and potential
contributions go through the shared `cpu::scf::evaluate_xc` (libxc-parity
tree). Closed shell:

\[
v_{\mathrm{xc}}
=
v_\rho
-
\nabla\cdot\Big[
\big(v_{\sigma_{\alpha\alpha}}+\tfrac12 v_{\sigma_{\alpha\beta}}\big)
\nabla\rho
\Big].
\]

`XCType::LSDA` selects the pointwise density-only path. **Forces:** analytic
Hellmann–Feynman (plus PP projectors) via `pw.forces(frame)`; on ideal diamond
sites lattice symmetry requires \(\mathbf{F}=\mathbf{0}\) for every atom.

```cpp
PW pbe{{std::make_shared<UpfPseudopotential>(upf)}, 15.0, 2, cpu::scf::XCType::PBE};  // true GGA (spectral ∇ρ)
pbe.tol      = 1e-5;
pbe.max_iter = 40;
pbe.mix_beta = 0.5;

pbe.build(frame);
const PWResult r_pbe = pbe.run(frame);  // copy: forces() overwrites last_result()
pbe.forces(frame);                      // own SCF; kAtom_fx/fy/fz [eV/Å], kEnergy [eV]

PW lda{{std::make_shared<UpfPseudopotential>(upf)}, 15.0, 2, cpu::scf::XCType::LSDA};  // activation twin
```

Internal XC: `xc_builder.h`, `src/cpu/planewave/internal/xc_builder.cpp`
(spectral \(\nabla\rho\), divergence \(v_{\mathrm{xc}}\)). Recorded, with no
unit test in the tree gating it: uniform density → PBE
\(E_{\mathrm{xc}}\) matches PBE@\(\sigma=0\) within \(10^{-10}\) (Ha-scale grid
integral); nonuniform density → GGA differs from \(\sigma=0\) by a meaningful
threshold.

### Validation

Mixing: Pulay, `mix_beta = 0.5`, `tol = 1e-5`, `max_iter = 40`. The shipped Si
UPF is LDA-generated; here it is a fixed NC valence potential for the GGA path.

| Quantity | Gate | Reference | Tol / bound | Measured (Release, single rank) |
| -------- | ---- | --------- | ----------- | ------------------------------- |
| \(E_{\mathrm{PBE}}\) | finite (SCF converged) | — | finite | −15.70481672 Ry (6 iters) |
| \(E_{\mathrm{LSDA}}\) | finite (SCF converged) | — | finite | −15.67175235 Ry (5 iters) |
| \(\lvert F_i\rvert\) (both Si) | symmetry residual | crystal symmetry \(\mathbf{F}=\mathbf{0}\) | \(< 0.5\) eV/Å | \(\sim 0.0015\) eV/Å |
| \(\lvert E_{\mathrm{PBE}}-E_{\mathrm{LSDA}}\rvert\) | GGA activation | internal dual run | \(> 10^{-4}\) Ry | 0.0330644 Ry |

### Performance

Informational wall (PBE + forces + LSDA dual): \(\sim 27\) s; `perf_exempt`.

### Limitations

- **No absolute energy oracle.** Pins are symmetry forces + the internal
  GGA–LDA gap, not a cited CP2K/QE total energy at matched cutoff/PP.
- **PP inconsistency.** LDA-generated ONCVPSP UPF under a PBE functional — not
  a production PBE-generated PP pair; production Si typically wants
  \(\sim 30\) Ry and denser meshes, and absolute energies are not transferable
  to PseudoDojo deltafactor tables.
- No analytic cell stress under PBE on CPU here (CPU↔CUDA in
  `cuda-pw-stress-pbe`).

## pw-analytic-stress-si — Nielsen–Martin analytic stress of bulk Si (LDA) {#pw-analytic-stress-si}

### Summary

Nielsen–Martin cell stress for plane-wave LDA on the
[Common setup](#common-setup) (same twin as [pw-lda-si](#pw-lda-si)).
`PW::stress`: one SCF, then Hellmann–Feynman cell derivatives under symmetric
strain at fixed fractional coordinates, returning Voigt \(\sigma\) in
**eV/Å³**. The multi-SCF strain finite difference `pw_stress_fd` is the
dual-gate oracle only (not the hot path); the section gates the analytic stress
against an offline multi-SCF FD pin, not a freeze of analytic output.

### Method

After one converged SCF on the unstrained cell, the stress is the cell
derivative of the total energy at fixed fractional coordinates (Nielsen &
Martin, *Phys. Rev. B* **32**, 3780 (1985)). Engine decomposition: kinetic
(bands), Hartree, local PP form-factor \(G\)-derivative, Ewald ion–ion, LDA XC
volume term, and Kleinman–Bylander nonlocal (UPF projectors). Sign convention
(mechanical / CP2K):

\[
\sigma_{\alpha\beta}
  = -\frac{1}{\Omega}\,
    \frac{\partial E}{\partial\varepsilon_{\alpha\beta}},
\qquad
\sigma = (\sigma_{xx},\sigma_{yy},\sigma_{zz},\sigma_{xy},\sigma_{xz},\sigma_{yz})
\quad [\mathrm{eV}/\mathrm{\AA}^3].
\]

`pw_stress_fd` rebuilds a full SCF for each of 12 central-difference strained
cells (\(h = 2\times 10^{-3}\) default).

| Symbol | Definition | Unit |
| ------ | ---------- | ---- |
| \(\sigma_v\) | Analytic Voigt stress from `PW::stress` | eV/Å³ |
| \(\sigma_v^\mathrm{FD}\) | Multi-SCF strain-FD from `pw_stress_fd` / offline pin | eV/Å³ |
| `kStress` | Frame meta write (virial σΩ) by `PW::stress`, or by a mounted `PW` when the plan consumes stress | eV |

```cpp
PW method{{std::make_shared<UpfPseudopotential>(ATV_REG_DATA_DIR "/pw/Si_lda_nc.upf")}, 15.0, 2,
          cpu::scf::XCType::LSDA};
method.tol = 1e-6;
method.max_iter = 50;

// Analytic Nielsen–Martin stress [eV/Å³], Voigt xx,yy,zz,xy,xz,yz.
// Also writes W = σ·Ω [eV] to kStress and E [eV] to kEnergy.
const std::array<double, 6> san = method.stress(frame);

// Dual-gate oracle only (12 SCFs): pw_stress_fd(frame, method, /*h=*/2e-3);
// Driver binding: opt.on(minimizer::Phase::force).add(std::move(method)); σΩ lands
// in kStress when the plan consumes stress (barostat, cell relaxer,
// StressConverged / PressureConverged).
```

Private implementation `pw_stress.h`; no unit test in the tree compares
analytic and FD stress (H toy, symmetry).

### Validation

Offline multi-SCF strain-FD (`pw_stress_fd`, \(h=2\times 10^{-3}\)) on the same
NC-UPF twin, Arrhenius epyc, Release, `OMP_NUM_THREADS=16`; SCF tol
\(10^{-6}\), `max_iter` 50, LSDA.

| Quantity | Atomiverse / pin | Oracle | Tol | Notes |
| -------- | ---------------: | -----: | --- | ----- |
| \(\sigma_{xx},\sigma_{yy},\sigma_{zz}\) | dual-gate | **+0.05755** eV/Å³ (mean FD diagonals) | 0.05 eV/Å³ | the energy-derivative sign \(+\frac1\Omega\partial E/\partial\varepsilon\) gives −0.05755 |
| \(\sigma_{xy},\sigma_{xz},\sigma_{yz}\) | dual-gate | **0** | 0.05 eV/Å³ | crystal symmetry zero |
| Offline \(\max_v\lvert\sigma^\mathrm{an}-\sigma^\mathrm{FD}\rvert\) | **\(7.3\times 10^{-5}\)** eV/Å³ | same twin | ≪ 0.05 | measured offline, not in ctest |
| Soft symmetry \(\lvert\sigma_{xy,xz,yz}\rvert\) | — | 0 | **0.025** eV/Å³ | incomplete ecut/k residual |
| Soft symmetry \(\lvert\sigma_{xx}-\sigma_{yy}\rvert,\ldots\) | — | 0 | **0.025** eV/Å³ | cubic isotropy of the diagonal |
| Sign of hydrostatic \(\sigma_{xx}\) | compressive **> 0** | EOS context | qualitative | \(a=5.43\) Å compressed vs PseudoDojo \(a_0\approx 5.468\) Å (\(B_0\) = 89.96 GPa) |

### Performance

ATV analytic wall ≈ **30** s (Release, 16 threads; 1 SCF + stress pass),
inside the ≤ 60 s budget; the analytic + full FD dual run takes ≈ 116 s and is
offline only. `perf_exempt`.

### Limitations

- Hydrostatic stress magnitude is incomplete-basis; not an equilibrium-pressure
  product (cell relaxation is [pw-variable-cell-si](pw-variable-cell.md#pw-variable-cell-si)).
- The 0.05 eV/Å³ dual-gate tolerance is wide; the \(\sim 10^{-4}\) an↔FD
  residual is an offline measurement.
- GGA (PBE) stress against an external oracle, MPI stress scaling and GPW
  strain-FD stress (`gpw-stress-water`) are not covered here.

## mpi-pw-scaling-4rank — MPI k-point scaling at 4 ranks {#mpi-pw-scaling-4rank}

### Summary

With MPI, `PW` distributes Monkhorst–Pack **k-points round-robin** over
`MPI_COMM_WORLD`: rank \(r\) owns \(\{r,\, r+n_\mathrm{procs},\, \ldots\}\),
diagonalizes those subspaces with Davidson, then **Allreduces** the real-space
density and band-energy sum every SCF iteration, so every rank holds the
identical mixed density and takes identical convergence decisions. The Davidson
initial guess is seeded by subspace index (not rank), so the SCF trajectory is
**rank-count invariant** by construction. The section runs the
[pw-lda-si](#pw-lda-si) twin at **np = 1** and **np = 4** (sibling of
`mpi-gpw-scaling-4rank`).

### Observables

| Symbol | Member | Meaning |
| ------ | ------ | ------- |
| \(n_k\) | `PWResult::n_kpts` | Total MP k-points of the run (`k_mesh`³ set) |
| \(n_k^\mathrm{local}\) | `PWResult::n_kpts_local` | k-points owned by **this** rank under RR |
| \(E\) | `PWResult::energy` | Total energy [Ry] |
| \(T_\mathrm{run}\) | section wall | `MPI_Wtime` around `run()` (max over ranks) |

Serial / MPI-inactive: \(n_k^\mathrm{local} = n_k\). Partition identity:

\[
\sum_r n_k^\mathrm{local}(r) = n_k,
\qquad
\max_r n_k^\mathrm{local} - \min_r n_k^\mathrm{local} \le 1 .
\]

For \(k_\mathrm{mesh}=2\) (8 k-points) at 4 ranks each rank owns exactly 2.

```cpp
PW method{{std::make_shared<UpfPseudopotential>(ATV_REG_DATA_DIR "/pw/Si_lda_nc.upf")}, 15.0, 2};
method.tol = 1e-6;
method.max_iter = 50;
method.build(frame);
const PWResult& r = method.run(frame);   // under mpiexec -n N
// r.n_kpts (global), r.n_kpts_local (this rank)
```

### Validation

Equal `OMP_NUM_THREADS` per rank in both launches. The section self-launches
via the build-configured `mpiexec` (compile-time `ATV_MPIEXEC`).

| Check | Bound |
| ----- | ----- |
| Energy (np=1 and np=4) | \([-16.916316,\, -16.916316 + 1.50]\) Ry (PseudoDojo ABINIT \(E_\mathrm{min}\)) |
| Rank-count invariance | \(\lvert E(1)-E(4)\rvert \le 10^{-8}\) Ry; identical SCF iters |
| 4-way split | `nk_local_min == nk_local_max == 2` at np=4 |
| Speedup | \(T(1)/T(4) \ge 1.2\) (weak floor; grid work is still replicated) |
| Budget | total wall ≤ 60 s |

No MPI unit test in the tree checks the split at n = 1/2/4
(`Allreduce SUM(n_kpts_local) == n_kpts` and `max − min ≤ 1` after one SCF pass,
`max_iter = 1`); the gates above are the only check.

### Limitations

- **k-point RR only** — no G-vector / band / domain decomposition; grid and
  density work remain replicated on every rank, only the per-k Davidson
  diagonalization parallelizes, hence the deliberately weak speedup floor.
- CPU plane-wave path only; no CUDA PW MPI. GPW MPI scaling is
  `mpi-gpw-scaling-4rank`.
