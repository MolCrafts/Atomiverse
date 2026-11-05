# Plane-wave collinear spin-polarized Kohn–Sham

> The regression runner has been removed. Regression commands below are historical; case sources and oracle records are retained as archived evidence, not current executable checks.

Collinear spin-polarized plane-wave Kohn–Sham (nspin=2) on the concrete method
`cpu::pw::PW` (and `cuda::pw::PW`): set `spin_polarized = true` and
`multiplicity` on the method (no `*Config` bag, no factory), then
`build(frame)` → `run(frame)`; `PWResult` reports `energy` [Ry], `energy_ha`
[Ha] (`energy = 2 × energy_ha`), `magnetization`
\(M = N_\uparrow - N_\downarrow\), `converged` and `iterations`. The k-mesh is
the third constructor argument and the XC functional the fourth
(`PW{pp, ecut_ry, k_mesh, xc}`); with `XCType::PBE` the spin path is true
collinear spin-GGA. Restricted default (`spin_polarized = false`) is unchanged.
Internals (density / XC / workspace spin buffers, `XCBuilder::step_spin`) stay
under `cpu/planewave/internal/`. The sections cover the O₂ triplet at Γ on CPU:
energy + magnetization, and the GGA activation against LSDA.

The case also carries CUDA sections without a tutorial page, each a same-settings
CPU↔CUDA dual gate on the O₂ triplet: `cuda-pw-collinear-spin` (LSDA energy +
\(M\) identity; the device collinear path is fixed-occupation, no smear),
`cuda-pw-spin-pbe` (spin-GGA PBE energy, \(M\) and GGA activation — the device
twin of [pw-spin-gga](#pw-spin-gga)), `cuda-pw-spin-forces` (LSDA analytic ionic
forces, stretched bond, max\(|F_\mathrm{cuda}-F_\mathrm{cpu}|<5\times10^{-3}\)
eV/Å) and `cuda-pw-spin-stress` (LSDA Nielsen–Martin cell stress). Their oracle
records live in the case `REFERENCE.md`.

Case: `ctest -R '^pw/collinear-spin$'` — sources and oracle record in [`regressions/pw/collinear-spin/`](../../../regressions/pw/collinear-spin/). One section of the case per `##` heading below: `atv_regressions --case pw/collinear-spin --section <name>`.

Roadmap: [dft.md](../../roadmap/dft.md) §10 PW collinear spin / true collinear
spin-GGA; § k-point periodic UKS. Requires `ATV_BUILD_PW=ON` (CPU build
environment `buildenv-gcc/2026.03-mpich`).

## Method

**Level.** Collinear spin-DFT (von Barth–Hedin / LSD channel on the PW grid):

- Separate \(\alpha/\beta\) KS orbitals and densities \(\rho_\uparrow\),
  \(\rho_\downarrow\) at every \(\mathbf{k}\) of the MP mesh.
- Hartree from total density \(\rho = \rho_\uparrow + \rho_\downarrow\).
- Spin-dependent XC potentials
  \(V_{\mathrm{xc}}^\sigma(\rho_\uparrow,\rho_\downarrow)\) via the shared
  `cpu::scf::evaluate_xc` (libxc-parity tree).
- Dual Davidson diagonalizations per **k**-point and spin channel with spin
  occupations of 1 (no closed-shell factor of 2); k-weights from
  `generate_mp_mesh`.

**Spin / occupations.** Non-metallic fixed fillings from multiplicity \(2S+1\):

\[
N_\uparrow - N_\downarrow = 2S = m - 1,\qquad
N_\uparrow + N_\downarrow = N_{\mathrm{val}}.
\]

Oxygen NC-PP \(Z_{\mathrm{val}} = 6\) → \(N_{\mathrm{val}} = 12\) for O₂;
multiplicity 3 → \(N_\uparrow = 7\), \(N_\downarrow = 5\), magnetization

\[
M = N_\uparrow - N_\downarrow = 2 .
\]

The O₂ ground state is the triplet \(^3\Sigma_g^-\) (Herzberg, *Molecular
Spectra and Molecular Structure* I; standard open-shell computational test).
With fixed integer occupations and multiplicity 3, \(M = 2\) is an **identity**
of the occupation assignment, independent of the k-mesh — the hard literature
gate \(|M-2|\le 1\times10^{-9}\) of every section. Metallic occupations with
spin are the [pw-uks-fe-smearing](pw-smearing.md#pw-uks-fe-smearing) section.

**Observables**

| Symbol | Definition | Unit |
| ------ | ---------- | ---- |
| \(E\) | Total KS energy (`PWResult::energy`) | Ry |
| \(E_{\mathrm{Ha}}\) | \(E/2\) (`PWResult::energy_ha`) | Ha |
| \(M\) | \(N_\uparrow - N_\downarrow\) (`PWResult::magnetization`) | e⁻ |
| — | SCF convergence flag / iteration count | — |

## Common setup

| Setting | Value |
| ------- | ----- |
| System | O₂, \(R(\mathrm{O{-}O}) = 1.2075\) Å (same as the molecular O₂ SCF pins) |
| Geometry | O \((0,0,0)\); O \((0,0,1.2075)\) [Å] |
| PP | SG15 ONCV PBE O (Hamann ONCVPSP), `tests/data/pw/O_ONCV_PBE-1.2.upf` |
| Multiplicity | 3 (\(^3\Sigma_g^-\)); fixed \(N_\uparrow=7\), \(N_\downarrow=5\) |

```cpp
cpu::Frame frame(2);
for (int i = 0; i < 2; ++i) {
    frame.span(kAtom_atomic_number)[i] = 8;       // Z(O)
    frame.span(kAtom_mass)[i] = 15.999;
    frame.span(kAtom_x)[i] = 0.0;
    frame.span(kAtom_y)[i] = 0.0;
}
frame.span(kAtom_z)[0] = 0.0;
frame.span(kAtom_z)[1] = 1.2075;         // R(O–O) [Å]
frame.box = cpu::Box::cubic(8.0);

PW pw{std::make_shared<UpfPseudopotential>(ATV_REG_DATA_DIR "/pw/O_ONCV_PBE-1.2.upf"),
      12.0, 1 /* Γ */, cpu::scf::XCType::PBE};   // true collinear spin-GGA
pw.tol = 5e-4;                           // density L2
pw.max_iter = 60;
pw.mix_beta = 0.3;
pw.pulay_depth = 5;
pw.spin_polarized = true;
pw.multiplicity = 3;                     // triplet → M = 2

pw.build(frame);
const PWResult& r = pw.run(frame);
// r.energy_ha [Ha], r.energy [Ry], r.magnetization, r.converged
```

The sections call `PW` directly; a Driver can mount a moved `PW` as a nested
node, in which case the public Frame energy is eV. Files:
`cpu/planewave/pw_solver.h` (API), `src/cpu/planewave/pw_solver.cpp`
(SCF / spin × k branch), `src/cpu/planewave/internal/{density,xc}_builder.cpp`
and `xc_builder.h` (density / spin-GGA XC). No unit test in the tree covers
the spin path (including multi-k magnetization).

## Oracle

No section runs CP2K or Quantum ESPRESSO at ctest time. The host CP2K 2026.1
module is listed but unloadable (dependency chain / `hwloc`) and no `pw.x` is on
PATH, so absolute \(E_{\mathrm{tot}}\) is a **fixed-settings offline method
pin** for regression stability (same PP, ecut, cell, geometry), not a
literature chemical-accuracy claim; absolute PW total energies shift with
cutoff, k-mesh and PP family. CP2K Quickstep is **GPW/GTH**, not NC-UPF PW;
re-pin against QE `pw.x` with this UPF when an offline twin exists.
Magnetization remains the hard literature gate.

**Performance (all sections): `perf_exempt`** — no loadable CP2K / `pw.x`
binary on the host for a fair single-rank NC-UPF PW wall twin (CP2K Quickstep
is a different basis/PP class); \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) is
n/a.

## Limitations (all sections)

- **Fixed occupations only** — metallic smearing is orthogonal
  ([pw-smearing-al](pw-smearing.md#pw-smearing-al)); spin + smearing is
  [pw-uks-fe-smearing](pw-smearing.md#pw-uks-fe-smearing).
- **No CPU spin-polarized forces/stress gate**: analytic force/stress APIs are
  gated for nspin=2 only CPU↔CUDA at LSDA (`cuda-pw-spin-forces`,
  `cuda-pw-spin-stress`), not against an external oracle and not for spin-GGA.
- **Energy twin debt:** absolute \(E_{\mathrm{tot}}\) is a frozen offline method
  pin, not a CP2K/QE re-pin.
- **Modest ecut** for the wall budget; production O₂/ONCV needs higher cutoff
  for basis completeness, so absolute energies are not transferable.
- **Out of scope:** noncollinear spin, SOC, hybrid PW, meta-GGA (\(\tau\)
  channel blocked on the PW/GPW engines), GPW UKS (`gpw-uks-o2-gamma`,
  `gpw-uks-pbe`, [gpw-uks-blyp](gpw-uks.md#gpw-uks-blyp)).

## References

- G. Herzberg, *Molecular Spectra and Molecular Structure* I (O₂
  \(^3\Sigma_g^-\) ground multiplet)
- J. P. Perdew, K. Burke, M. Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996)
  — PBE
- D. R. Hamann, optimized norm-conserving Vanderbilt pseudopotentials; SG15
  ONCV library (UPF used offline)
- [Chapter 8 — Plane-Wave DFT](../ch08-plane-wave.md),
  [Chapter 9 — GPW](../ch09-gpw.md)
- Related: [pw-lda-si](pw-rks.md#pw-lda-si),
  [pw-pbe-gga-si](pw-rks.md#pw-pbe-gga-si) (closed-shell true GGA)

## pw-spin-polarized-o2 — Collinear spin-polarized O₂ triplet (Γ) {#pw-spin-polarized-o2}

### Summary

Energy + magnetization of the O₂ triplet at Γ with the production spin path
(`XCType::PBE`, true collinear spin-GGA) — the [Common setup](#common-setup)
snippet verbatim.

### Settings

| Setting | Value |
| ------- | ----- |
| Cell | cubic \(L = 8.0\) Å |
| \(E_{\mathrm{cut}}\) | 12 Ry |
| **k**-mesh | \(1\times1\times1\) (Γ, `k_mesh = 1`) |
| XC | `XCType::PBE` (true collinear spin-GGA) |
| SCF | density L2 tol \(5\times10^{-4}\), `max_iter=60`, `mix_beta=0.3`, `pulay_depth=5` |

### Validation

| Quantity | Atomiverse / pin | Reference | Tol | Notes |
| -------- | ---------------: | --------: | --- | ----- |
| \(M\) [e⁻] | **2** | 2 (fixed occupations / Herzberg multiplet) | \(1\times10^{-9}\) | `PWResult::magnetization` |
| \(E_{\mathrm{tot}}\) [Ha] | **−28.7585210635** | offline method pin (same PP/ecut/cell/geom) | \(1\times10^{-3}\) Ha | same value and settings as [pw-spin-gga](#pw-spin-gga) |
| SCF | must converge | — | — | fails if `!converged` |

Other energies of this setup that are **not** the reference: −28.6267426458 Ha
is the PBE@\(\sigma=0\) (LSDA-limit) channel without spin gradients, and
**−28.3392565801** Ha is the value produced when the Davidson subspace
Hamiltonian is truncated to its real part and the G-sphere enumeration clips
sphere-edge plane waves.

### Performance

\(T_{\mathrm{ATV}}\) ~40 s (order of magnitude, single rank; not for a ratio).
`perf_exempt` (see [Oracle](#oracle)).

### Limitations

**Γ-only molecular cell** (`k_mesh = 1`); multi-k collinear UKS is the
[pw-uks-kmesh-fe](#pw-uks-kmesh-fe) variant.

## pw-spin-gga — True collinear spin-GGA PBE, O₂ triplet (Γ) {#pw-spin-gga}

### Summary

Dedicated GGA-activation section: the same O₂ triplet / Γ / 12 Ry settings as
[pw-spin-polarized-o2](#pw-spin-polarized-o2) with `XCType::PBE`, plus an LSDA
spin twin on the same geometry and occupations. Gates: \(M=2\), the offline
\(E_{\mathrm{tot}}\) pin, and an internal PBE–LSDA energy gap proving that the
spin path is not stuck in the density-only LDA limit.

### Method

Separate \(\rho_\uparrow/\rho_\downarrow\) channels with spectral density
gradients \(\nabla\rho_\sigma\) (from \(\rho_\sigma(G)\)) enter the
reduced-gradient invariants

\[
\sigma_{\sigma\sigma'} = (\nabla\rho_\sigma)\cdot(\nabla\rho_{\sigma'})
\]

in `cpu::scf::evaluate_xc`, giving pointwise
\(e_{\mathrm{xc}}(\rho_\uparrow,\rho_\downarrow,\{\sigma\})\), and the spin
potentials (`XCBuilder::step_spin`) include the full divergence term, of the
same form as GPW `xc_grid_gga_spin`:

\[
v_\sigma
=
v_{\rho_\sigma}
-
\nabla\cdot\Big(
  2\,v_{\sigma_{\sigma\sigma}}\,\nabla\rho_\sigma
  +
  v_{\sigma_{\uparrow\downarrow}}\,\nabla\rho_{\bar\sigma}
\Big).
\]

This is **not** PBE called with forced \(\sigma=0\) per spin channel.

```cpp
PW pbe{std::make_shared<UpfPseudopotential>(upf), 12.0, 1, cpu::scf::XCType::PBE};    // members as in Common setup
PW lsda{std::make_shared<UpfPseudopotential>(upf), 12.0, 1, cpu::scf::XCType::LSDA};  // GGA activation twin
const double de = std::fabs(r_pbe.energy_ha - r_lsda.energy_ha);    // gate: > 1e-4 Ha
```

### Validation

Settings as [pw-spin-polarized-o2](#pw-spin-polarized-o2) (cubic \(L = 8.0\) Å,
12 Ry, Γ, density L2 tol \(5\times10^{-4}\), `max_iter=60`, `mix_beta=0.3`).

| Quantity | Atomiverse / pin | Reference | Tol / bound | Notes |
| -------- | ---------------: | --------: | ----------- | ----- |
| \(M\) [e⁻] | **2** | 2 (fixed occupations / Herzberg multiplet) | \(1\times10^{-9}\) | `PWResult::magnetization` |
| \(E_{\mathrm{tot}}\) [Ha] | **−28.7585210635** | offline method pin (same PP/ecut/cell/geom; true spin-GGA) | \(1\times10^{-3}\) Ha | no loadable CP2K/`pw.x` on the pin host |
| \(\lvert E_{\mathrm{PBE}}-E_{\mathrm{LSDA}}\rvert\) [Ha] | \(\approx 0.129\) | internal dual run | \(> 10^{-4}\) Ha | GGA activation |
| \(E_{\mathrm{LSDA}}\) [Ha] | \(\approx -28.62946347\) | dual run | finite + converged | informational |
| SCF (both runs) | must converge | — | — | fails if `!converged` |

True spin-GGA lies ~0.13 Ha below the PBE@\(\sigma=0\) / LSDA-like energy
(near −28.63 Ha) at these settings.

### Performance

\(T_{\mathrm{ATV}}\) ~60 s (order of magnitude, single rank, PBE + LSDA runs;
not for a ratio). `perf_exempt` (see [Oracle](#oracle)).

### Limitations

- **Γ-only molecular cell**; multi-k collinear UKS
  ([pw-uks-kmesh-fe](#pw-uks-kmesh-fe)) is not audited for spin-GGA activation
  digits.
- Absolute digit match to CP2K Quickstep is out of scope (GPW/GTH ≠ NC-UPF PW).

## pw-uks-kmesh-fe — Multi-k collinear UKS, O₂ (Monkhorst–Pack) {#pw-uks-kmesh-fe}

*No section of the case runs this variant.*

### Summary

Collinear UKS on a multi-k Monkhorst–Pack mesh: `PW` with
`spin_polarized = true`, `multiplicity = 3` and the `k_mesh` constructor
argument **≥ 2** (no public members beyond the Γ path). The target is the O₂
triplet in a small cubic cell, not bulk Fe: an Fe metal needs spin and metallic
smearing together, which is the
[pw-uks-fe-smearing](pw-smearing.md#pw-uks-fe-smearing) section. The values
below are offline pins on the CPU path; no section gates them.

### Settings

| Setting | Value |
| ------- | ----- |
| Cell | cubic \(L = 5.5\) Å |
| \(E_{\mathrm{cut}}\) | 5 Ry (modest tutorial ecut for the multi-k × dual-spin wall budget) |
| **k**-mesh | \(2\times 2\times 2\) (`k_mesh = 2`) |
| XC | constructor default |
| SCF | density L2 tol \(2\times10^{-3}\), `max_iter=20`, `mix_beta=0.5`, `pulay_depth=3` |

```cpp
frame.box = cpu::Box::cubic(5.5);
PW pw{std::make_shared<UpfPseudopotential>("tests/data/pw/O_ONCV_PBE-1.2.upf"), 5.0, 2};  // multi-k UKS
pw.tol = 2e-3;
pw.max_iter = 20;
pw.mix_beta = 0.5;
pw.pulay_depth = 3;
pw.spin_polarized = true;
pw.multiplicity = 3;
```

### Validation

| Quantity | Atomiverse / pin | Reference | Tol | Notes |
| -------- | ---------------: | --------: | --- | ----- |
| \(M\) [e⁻] | **2** | 2 (fixed occupations, independent of k-mesh) | \(1\times10^{-9}\) | `PWResult::magnetization` |
| \(E_{\mathrm{tot}}\) [Ha] | **−24.2395602747** | offline method pin (same PP/ecut/cell/geom/k-mesh) | \(5\times10^{-3}\) Ha | Arrhenius x86_64; ~5 SCF iters |
| SCF | must converge | — | — | |

CP2K documentation twin (algorithm only, not energy-compared): Quickstep UKS +
`KPOINTS` Monkhorst–Pack \(2\times2\times2\) on the same O₂ cell with GTH +
MOLOPT.

### Performance

\(T_{\mathrm{ATV}}\) **~31 s** (OMP=8, single rank), inside the ≤ 60 s
single-rank CPU budget. `perf_exempt` (see [Oracle](#oracle)).

### Limitations

- **O₂ cell, not bulk Fe**; no Fe NC-UPF fixed-occupation path.
- **Modest ecut (5 Ry)** and small cell: absolute energy is not chemically
  converged.
- Γ-only UKS is [pw-spin-polarized-o2](#pw-spin-polarized-o2); GPW k-points are
  `gpw-kpoints-si`; restricted multi-k is [pw-lda-si](pw-rks.md#pw-lda-si).
