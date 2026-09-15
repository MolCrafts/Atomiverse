# DFPT phonons: collinear spin + metallic smearing — diamond Si (Γ, PW LSDA)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `dfpt-spin-smearing` |
| Regression | `regressions/response/dfpt-spin-smearing/` |
| Public API | `cpu::pw::DfptPhonon` + `cpu::pw::DfptPhononResult`; ground state via `cpu::pw::PW` members `spin_polarized`, `multiplicity`, `smearing`, `degauss` [Ry], `extra_bands` |
| Roadmap | [dft.md](../../roadmap/dft.md) §11 DFPT — collinear spin + metallic smear ✓ (`reg_dfpt-spin-smearing`) |
| Dep | siblings `dfpt-spin-si` ✓, `dfpt-metallic-si` ✓; Γ foothold `dfpt-phonon-si` ✓ |

## Summary

**Density-functional perturbation theory** at **q = 0** for the plane-wave
method with **collinear spin** and **Fermi–Dirac metallic occupation
smearing** in one ground-state + Sternheimer path. Product surface is
`cpu::pw::DfptPhonon::compute(method, frame)` on a `cpu::pw::PW` whose
members set spin and smear — **no** public `*Config` bag and **no**
`make_*` factory. Density response uses spin-resolved occupations
\(f_{\sigma nk}\in[0,1]\), a shared chemical potential \(\mu\), and the
collinear spin XC kernel; optional \(\delta\mu\) restores charge neutrality
when \(\partial N/\partial\mu\) is finite (de Gironcoli). Textbook pin:
**diamond Si** primitive cell, NC UPF, LSDA, \(M=0\), FD `degauss = 0.02`
Ry. Regression: `reg_dfpt-spin-smearing`. Offline CP2K documents the
product counterpart (LSD + `&SMEAR` + vibrational analysis); absolute
\(\omega/E\) are **same-Hamiltonian ATV free-energy FD** pins (GTH-GPW ≠
NC-UPF).

## Method

**Level.** Γ-point DFPT dynamical matrix for `cpu::pw::PW` (norm-conserving
UPF, pure LSDA kernel on this pin). Collinear spin + metallic smear is
allowed at Γ and at **commensurate** finite \(\mathbf{q}\) (supercell Γ
path). Direct incommensurate finite-\(q\) + spin+smear **throws**.

**Occupations (spin-orbitals).** Fermi–Dirac (or Gaussian / Methfessel–Paxton
via `method.smearing`) with width `degauss` [Ry]:

\[
f_{\sigma nk}\in[0,1],\qquad
n_{\mathrm{elec}}
  = \sum_{\sigma nk} w_k\, f_{\sigma nk}
  = N_\uparrow + N_\downarrow
\]

with a **shared** \(\mu\) over spin channels (multiplicity \(M=1\) on the
Si twin ⇒ \(N_\uparrow=N_\downarrow=4\) valence spin-orbitals). Gapped Si at
minimal twin settings keeps near-integer FD fill
(\(f_{\sigma nk}\in\{0,1\}\)), so the Mermin free energy
\(F=E_{\mathrm{KS}}-TS\) collapses to \(E_{\mathrm{KS}}\) and the force
constants track the collinear fixed-occupation twin (`dfpt-spin-si`).

**Density response.**

\[
\Delta n_\sigma
  = \sum_{nk} w_k\, f_{\sigma nk}\,
    2\,\mathrm{Re}\,\psi^*_{\sigma nk}\,\Delta\psi_{\sigma nk}.
\]

Sternheimer CG projects out bands with \(f\ge 10^{-8}\). Hartree screens
total \(\Delta n\); the XC kernel is the collinear LSDA
\(f_{\mathrm{xc}}^{\sigma\sigma'}\) table (same as `dfpt-spin-si`; pure GGA
uses FD of `XCBuilder::step_spin` on other pins).

**Screening / neutrality.** At Γ, a Fermi-level shift \(\delta\mu\) may
restore \(\int\Delta n=0\) when \(\partial N/\partial\mu\) is non-negligible
(de Gironcoli, *Phys. Rev. B* **51**, 6773 (1995)).

**Dynamical matrix.** Analytic Ewald second derivatives + KB nonlocal
first/second derivative terms; mass-weight and diagonalize to \(\omega\)
[cm⁻¹]. Acoustic sum rule is **not** imposed — residual \(|\omega_{\mathrm{ac}}|\)
and row sums are acceptance gates.

**Excluded (this slug).** Direct finite-\(q\) spin+smear; hybrids; cold /
Marzari–Vanderbilt smear enum on the DFPT pin path; GPW DFPT; LO–TO /
Born charges; CUDA DFPT. Restricted-only smear: `dfpt-metallic-si`.
Spin-only fixed-occ: `dfpt-spin-si`. Finite-\(q\) + smear (restricted):
`dfpt-finite-q-smearing`.

Cite: Baroni, de Gironcoli, Dal Corso, Giannozzi, *Rev. Mod. Phys.* **73**,
515 (2001).

## Implementation

Public surface only: construct `cpu::pw::PW`, set members (including spin +
smear), construct `cpu::pw::DfptPhonon` with response members, call
`compute`. Headers: `include/atomiverse/cpu/planewave/dfpt_phonon.h`,
`pw_solver.h`. Frame is not mutated; spectra live on
`DfptPhononResult` (Frame purity).

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/dfpt_phonon.h"
#include "atomiverse/cpu/planewave/pw_solver.h"

using namespace ATV;
using namespace ATV::cpu::pw;

// diamond Si primitive, a = 5.43 Å (REFERENCE.md)
cpu::Frame frame(2);
frame.span(kAtom_type)[0] = frame.span(kAtom_type)[1] = 14;
frame.span(kAtom_mass)[0] = frame.span(kAtom_mass)[1] = 28.0855;
frame.span(kAtom_x)[1] = frame.span(kAtom_y)[1] = frame.span(kAtom_z)[1] =
    5.43 / 4.0;
const double half = 5.43 / 2.0;
Mat3 cell;
cell(0, 0) = 0;    cell(0, 1) = half; cell(0, 2) = half;
cell(1, 0) = half; cell(1, 1) = 0;    cell(1, 2) = half;
cell(2, 0) = half; cell(2, 1) = half; cell(2, 2) = 0;
frame.box.set_cell(cell);

PW method;
method.upf_path = "tests/data/pw/Si_lda_nc.upf";
method.ecut_ry = 10.0;
method.k_mesh = 2;
method.xc = cpu::scf::XCType::LSDA;
method.spin_polarized = true;
method.multiplicity = 1;
method.smearing = Smearing::fermi_dirac;
method.degauss = 0.02;   // Ry
method.extra_bands = 0;  // valence Sternheimer manifold (gapped twin)
method.tol = 1e-8;
method.max_iter = 200;

DfptPhonon dfpt;
dfpt.response_tol = 1e-5;
dfpt.cg_tol = 1e-9;
dfpt.mix_beta = 0.7;
// default q_frac = {0,0,0} = Γ
const DfptPhononResult r = dfpt.compute(method, frame);
// r.frequencies [cm⁻¹] length 6; r.force_constants [eV/Å²] shape [6,6]
// r.scf_converged, r.response_converged, r.max_asymmetry
```

`compute` throws `std::invalid_argument` for hybrids, empty `upf_path`, or
direct finite-\(q\) with spin **and** smear. Restricted smear and collinear
fixed-occ remain the sibling paths above.

```bash
# configure (login shell + Lmod — see AGENTS.md)
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT=$OPENBLASDIR
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_PW=ON \
        -DATV_BUILD_REGRESSIONS=ON -DATV_BUILD_CUDA=OFF \
        -DBLA_VENDOR=OpenBLAS
  cmake --build build -j$(nproc) --target atomiverse reg_dfpt-spin-smearing
  ./build/regressions/reg_dfpt-spin-smearing
  # or: cmake --build build --target run_reg_dfpt-spin-smearing
'
```

Source consumer:
`regressions/response/dfpt-spin-smearing/reg_dfpt-spin-smearing.cpp`.

## Validation (CP2K / offline pins)

Source:
[`regressions/response/dfpt-spin-smearing/REFERENCE.md`](../../../regressions/response/dfpt-spin-smearing/REFERENCE.md)
(offline pins only; **never** live CP2K in ctest).

### ATV system (same-Hamiltonian gates)

| Setting | Value |
| ------- | ----- |
| Geometry | diamond Si, 2-atom fcc primitive, \(a=5.43\) Å |
| PP | `tests/data/pw/Si_lda_nc.upf` (ONCVPSP NC, in-tree) |
| XC | `XCType::LSDA` (collinear LSDA kernel) |
| Spin | collinear, multiplicity = 1 (\(M=0\)) |
| Smearing | Fermi–Dirac, `degauss = 0.02` Ry, `extra_bands = 0` |
| \(E_{\mathrm{cut}}\) | 10 Ry (minimal-twin budget) |
| k-mesh | Monkhorst–Pack \(2\times2\times2\) (unshifted) |
| SCF | density L2 tol \(10^{-8}\), `max_iter=200` |
| Response | `response_tol=10^{-5}`, `mix_beta=0.7`, CG tol \(10^{-9}\) |
| Pin date | 2026-07-26 (Release, OMP=8) |

**Sharp pin.** Central second difference of the **collinear-spin free energy**
\(F\) at identical ecut/k/XC/UPF/spin/FD:

\[
C[(0x),(0x)]
  = \frac{F(+h)-2F_0+F(-h)}{h^2},\qquad h=0.03\,\mathrm{Å}.
\]

Gapped FD fill \(\Rightarrow F\approx E_{\mathrm{KS}}\); the discrete Hessian
matches the `dfpt-spin-si` energy-FD pin (**17.3390 eV/Å²**).

| Quantity | Atomiverse / pin | Tol | Notes |
| -------- | ---------------: | --- | ----- |
| Free-energy FD \(C[(0x),(0x)]\) | **17.3390 eV/Å²** | — | offline central FD (sharp oracle) |
| DFPT \(C[(0x),(0x)]\) | **17.3415 eV/Å²** | \(\lvert\Delta\rvert\le 0.30\) | ~0.015 % vs FD pin |
| Acoustic \(\omega\) (no ASR) | 1.33, 1.34, 1.34 cm⁻¹ | \(\lvert\omega\rvert\le 30\) | emerges, not imposed |
| Optical triplet | **579.45, 579.45, 579.59** cm⁻¹ | \([500,650]\), spread \(\le 10\) | LDA minimal-settings window (lit. ~517 cm⁻¹ Giannozzi *et al.*) |
| Max row sum (ASR) | 0.0002 eV/Å² | \(\le 0.05\) | translational invariance |
| Pre-symmetrization \(\lvert C-C^\top\rvert/2\) | \(7.3\times10^{-10}\) eV/Å² | \(\le 10^{-5}\) | force-constant symmetry |
| Regression wall | ≈ **35 s** | \(\le 60\) s | OMP=8 Release |

### Offline CP2K twin (algorithm / SMEAR + vib documentation)

**Not an absolute-\(\omega\) twin** (GTH-GPW ≠ NC-UPF PW). Documents that CP2K
Quickstep can combine `VIBRATIONAL_ANALYSIS` (or FD vib) with LSD and
`&SMEAR METHOD FERMI_DIRAC` on a Si-like solid.

| Item | Value |
| ---- | ----- |
| Preferred product | CP2K `VIBRATIONAL_ANALYSIS` + LSD + `&SMEAR` |
| Sketch | `tools/cp2k_oracle/inputs/dfpt-spin-smearing.note` |
| Structure | diamond Si primitive \(a=5.43\) Å, 2 atoms |
| Method sketch | GPW, PADE/LDA, GTH-PADE + DZVP-MOLOPT-SR-GTH (Si q4) |
| Spin / smear | LSD / unrestricted, mult. 1; FERMI_DIRAC \(\sim 0.02\) Ry \(k_B T\) (~3158 K); `ADDED_MOS` ≥ 4 |
| k-points | Monkhorst–Pack \(2\times2\times2\) |
| CP2K binary this pin host | **not available** (`module avail cp2k` empty) |

Oracle path: `tools/cp2k_oracle/README.md` (offline only). When a CP2K binary
is available, record \(\omega(\Gamma)\) [cm⁻¹] + wall and optional fair
ratio \(r=T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) (require \(\le 10\)).

## Performance

Source: REFERENCE + handoff. Gate: \(r=T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\le 10\)
unless exempted.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| CP2K vib+LSD+SMEAR vs `reg_dfpt-spin-smearing` | n/a | ≈ **30–35** | n/a | **perf_exempt** | no CP2K module on pin host; correctness gates only |

`perf_exempt = true` — reason: no CP2K binary/module for a fair DFPT
spin+smear wall twin this shift. Handoff records **perf_exempt**
(the case `REFERENCE.md`). Informative ATV wall: ~30 s
(2-atom Si, 6 responses × 2 spins, 8 k-points, OMP=8).

## Limitations

- **Γ / commensurate \(\mathbf{q}\) only for spin+smear.** Direct
  incommensurate finite-\(q\) with collinear spin **and** metallic smear is
  rejected at the API boundary.
- **Gapped twin, not an open metal.** Si at 10 Ry / \(2^3\) keeps near-integer
  FD occupations; true multi-band metals need denser k/ecut and
  `extra_bands > 0` (not the regression pin).
- **Absolute \(\omega/E\) are ATV free-energy FD, not CP2K-matched.** GTH-GPW
  and NC-UPF PW are different Hamiltonians; CP2K documents product settings.
- **LSDA kernel on this pin** (GGA spin kernel productized on other DFPT
  slugs; hybrids out of scope).
- **No LO–TO splitting / Born effective charges** (nonpolar Si).
- **CPU only**; CUDA DFPT out of scope for this loop.
- **Speed.** Dual-gate wall vs CP2K deferred until an offline CP2K vib run is
  recorded on a host with the binary; regression stays ≤ 60 s.

## References

- Offline pin: `regressions/response/dfpt-spin-smearing/REFERENCE.md`
- Regression: `regressions/response/dfpt-spin-smearing/reg_dfpt-spin-smearing.cpp`
- CP2K twin note: `tools/cp2k_oracle/inputs/dfpt-spin-smearing.note`
- Oracle notes: `tools/cp2k_oracle/README.md`
- Handoff **perf_exempt**: the case `REFERENCE.md`
- Roadmap: `docs/roadmap/dft.md` §11 DFPT spin × smear productized
- Public headers: `include/atomiverse/cpu/planewave/dfpt_phonon.h`,
  `pw_solver.h`
- Siblings: [dfpt-phonon-si](dfpt-phonon-si.md) (Γ restricted LSDA);
  `regressions/response/dfpt-spin-si`, `dfpt-metallic-si`,
  `dfpt-finite-q-smearing`
- Literature: Baroni *et al.*, *Rev. Mod. Phys.* **73**, 515 (2001);
  de Gironcoli, *Phys. Rev. B* **51**, 6773 (1995); Giannozzi *et al.*,
  *Phys. Rev. B* **43**, 7231 (1991) (Si optical Γ LDA window)
- Narrative PW background: [ch08-plane-wave](../ch08-plane-wave.md);
  metallic occupations: [pw-smearing-al](pw-smearing-al.md)
