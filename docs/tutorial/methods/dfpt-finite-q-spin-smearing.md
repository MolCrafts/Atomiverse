# Finite-q DFPT — collinear spin + metallic smearing (diamond Si, PW LSDA)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `dfpt-finite-q-spin-smearing` |
| Regression | `regressions/response/dfpt-finite-q-spin-smearing/` |
| Public API | `cpu::pw::DfptPhonon` + `cpu::pw::DfptPhononResult`; GS members on `cpu::pw::PW` (`spin_polarized`, `multiplicity`, `smearing`, `degauss` [Ry], `extra_bands`) |
| Roadmap | [dft.md](../../roadmap/dft.md) §11 DFPT — finite-q / spin / smear product ✓ |
| Dep | `dfpt-spin-incommensurate` ✓, `dfpt-finite-q-smearing` ✓, `dfpt-spin-smearing` ✓ |

## Summary

**Density-functional perturbation theory** at a **fractional phonon wavevector**
\(\mathbf{q}\) for the plane-wave method `cpu::pw::PW` with **collinear spin**
and **Fermi–Dirac metallic occupations**. This slug pins the **direct**
Sternheimer product path
(`internal::dfpt_q_force_constants` → `dfpt_q_spin_impl`) that previously
rejected the spin+smear combination at incommensurate \(\mathbf{q}\).

Public surface: set members on `PW` and `DfptPhonon` (no `*Config` bag, no
`make_*` factory). Result owns \(\omega\) [cm⁻¹] and \(C(\mathbf{q})\) [eV/Å²];
Frame is not mutated. Regression binary: `reg_dfpt-finite-q-spin-smearing`.
Offline CP2K documents the product counterpart (phonon / DFPT + LSD +
`&SMEAR`); absolute \(\omega\) / energy are **not** a GTH-GPW ↔ NC-UPF twin —
gates freeze ATV spectra on the same Hamiltonian as the fixed-occupation spin
incommensurate twin.

## Method

**Level.** Self-consistent DFPT phonons (Baroni et al., *Rev. Mod. Phys.*
**73**, 515 (2001)) on NC-UPF plane-wave KS DFT. Pure LSDA kernel in this
regression twin (pure GGA kernels are accepted by the API; hybrids are
rejected).

**Ground state.** Collinear spin (`spin_polarized = true`, `multiplicity = 1`
⇒ \(M=0\)) with Fermi–Dirac smearing:

\[
f_{\sigma nk}\in[0,1],\qquad
n_{\mathrm{elec}}
  = \sum_{\sigma nk} w_k\, f_{\sigma nk}(\varepsilon_{\sigma nk},\mu,\sigma_{\mathrm{deg}}),
\]

shared chemical potential \(\mu\) over spin-orbitals, width `degauss` [Ry]
inside the SCF. Free energy \(F = E_{\mathrm{KS}} - TS\) when occupations are
fractional; gapped Si at the minimal twin keeps near-integer fill
(\(F\approx E_{\mathrm{KS}}\)).

**Response (direct finite-\(\mathbf{q}\)).** For incommensurate
\(\mathbf{q}\) (here \((0.3,0,0)\) under supercell product \(\le 6\)):

- Spin-resolved Sternheimer with projector on bands with \(f\ge 10^{-8}\) at
  \(\mathbf{k}+\mathbf{q}\)
- Hartree from total \(\Delta n\); spin \(K_{\mathrm{xc}}\) (local LSDA
  \(f_{\mathrm{xc}}^{\sigma\sigma'}\) table)
- Occupation weights use ground-state \(f_{\sigma nk}\) at the shared \(\mu\)
- **No** \(\delta\mu\) charge-neutrality correction at \(q\neq 0\) (de Gironcoli
  \(\delta\mu\) remains a Γ-only option when \(\partial N/\partial\mu\) is
  non-negligible)

Density response (schematic):

\[
\Delta n_\sigma
  = \sum_{nk} w_k\, f_{\sigma nk}\,
    2\,\mathrm{Re}\,\psi^*_{\sigma nk}\,\Delta\psi_{\sigma nk}.
\]

**Commensurate path (not this pin).** Rational
\(\mathbf{q}=(h/n_a,k/n_b,l/n_c)\) with small supercell (e.g. \(X\)) uses
supercell Γ-DFPT (spin+smear Sternheimer, same engine as
`dfpt-spin-smearing`) plus IFC Fourier. Covered by sibling slugs; this
regression deliberately takes the **direct** incommensurate path.

**Excluded (this slug).** Hybrids / meta-GGA / GAPW; LO–TO / Born charges;
CUDA DFPT; live CP2K in ctest; absolute GTH-GPW energy twin; metallic
true Fermi-surface DFPT on Al/Fe (Si stays gapped at these settings).

## Implementation

Public headers: `include/atomiverse/cpu/planewave/dfpt_phonon.h`,
`include/atomiverse/cpu/planewave/pw_solver.h`. Occupations /
Sternheimer internals under `planewave/internal/` (not user types).

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/dfpt_phonon.h"
#include "atomiverse/cpu/planewave/pw_solver.h"

using namespace ATV;
using namespace ATV::cpu::pw;

// Diamond Si primitive, a = 5.43 Å (REFERENCE.md)
cpu::Frame frame(2);
frame.span(kAtom_type)[0] = frame.span(kAtom_type)[1] = 14;
frame.span(kAtom_mass)[0] = frame.span(kAtom_mass)[1] = 28.0855;
frame.span(kAtom_x)[1] = frame.span(kAtom_y)[1] = frame.span(kAtom_z)[1]
    = 5.43 / 4.0;
const double h = 5.43 / 2.0;
Mat3 cell;
cell(0, 0) = 0; cell(0, 1) = h; cell(0, 2) = h;
cell(1, 0) = h; cell(1, 1) = 0; cell(1, 2) = h;
cell(2, 0) = h; cell(2, 1) = h; cell(2, 2) = 0;
frame.box.set_cell(cell);

PW method;
method.upf_path = "tests/data/pw/Si_lda_nc.upf";
method.ecut_ry = 10.0;
method.k_mesh = 1;
method.xc = cpu::scf::XCType::LSDA;
method.spin_polarized = true;
method.multiplicity = 1;
method.smearing = Smearing::fermi_dirac;
method.degauss = 0.02; // Ry
method.extra_bands = 0;
method.tol = 1e-6;
method.mix_beta = 0.4;
method.max_iter = 200;

DfptPhonon dfpt;
dfpt.q_frac = {0.3, 0.0, 0.0}; // incommensurate → direct Sternheimer
dfpt.response_tol = 1e-4;
dfpt.response_max_iter = 120;
dfpt.cg_tol = 1e-8;
dfpt.mix_beta = 0.2;

const DfptPhononResult r = dfpt.compute(method, frame);
// r.frequencies [cm⁻¹] ascending (imag → negative);
// r.force_constants / force_constants_imag [eV/Å²];
// r.scf_converged, r.response_converged, r.max_asymmetry
```

```bash
# configure (login shell + Lmod — see AGENTS.md)
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT=$OPENBLASDIR
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_REGRESSIONS=ON \
        -DATV_BUILD_HF=ON -DATV_BUILD_PW=ON -DATV_BUILD_GPW=ON \
        -DATV_BUILD_CUDA=OFF -DBLA_VENDOR=OpenBLAS
  cmake --build build -j$(nproc) --target atomiverse reg_dfpt-finite-q-spin-smearing
  ./build/regressions/reg_dfpt-finite-q-spin-smearing
  # or: cmake --build build --target run_reg_dfpt-finite-q-spin-smearing
'
```

Source consumer:
`regressions/response/dfpt-finite-q-spin-smearing/reg_dfpt-finite-q-spin-smearing.cpp`.
Unit coverage for spin+smear / finite-q acceptance:
`tests/cpu/planewave/test_dfpt.cpp` → `test_cpu_pw_dfpt` (no e2e SCF under
`tests/`).

## Validation (CP2K)

Source:
[`regressions/response/dfpt-finite-q-spin-smearing/REFERENCE.md`](../../../regressions/response/dfpt-finite-q-spin-smearing/REFERENCE.md)
(offline pins only; **never** live CP2K in ctest).

### ATV system (spectrum gates)

| Setting | Value |
| ------- | ----- |
| Geometry | diamond Si primitive, \(a=5.43\) Å, 2 atoms |
| \(\mathbf{q}\) | **(0.3, 0, 0)** fractional (incommensurate under \(n\le 6\)) |
| PP | `tests/data/pw/Si_lda_nc.upf` (NC, LDA) |
| XC | LSDA |
| Spin | collinear, multiplicity = 1 (\(M=0\)) |
| Smearing | Fermi–Dirac, `degauss = 0.02` Ry, `extra_bands = 0` |
| \(E_{\mathrm{cut}}\) | 10 Ry |
| k-mesh | \(1\times1\times1\) |
| Response | tol \(10^{-4}\), mix 0.2, max_iter 120 |
| Pin date | 2026-07-26 (Release, OMP=8) |

| Quantity | Pin | Tol / gate | Notes |
| -------- | --: | ---------- | ----- |
| \(\omega(\mathbf{q})\) [cm⁻¹] | **692.90, 726.85, 726.85, 944.07, 1023.46, 1023.46** | — | measured ATV print |
| \(\lvert\omega\rvert\) low 3 | 693, 727, 727 | ∈ **[400, 900]** | unit-disaster window |
| \(\lvert\omega\rvert\) high 3 | 944, 1023, 1023 | ∈ **[700, 1300]** | same |
| \(n_{\mathrm{imag}}\) | **0** | \(\le 0\) | no soft modes on this twin |
| Asymmetry \(\max\lvert C-C^\dagger\rvert/2\) | **29.6 eV/Å²** | \(\le 50\) | pre-Hermitian projection |
| Twin match | identical print to `dfpt-spin-incommensurate` | printed precision | gapped FD fill |

Minimal twin (10 Ry, Γ-only k) biases the spectrum high vs literature mid-ΓX;
windows reject unit disasters, not claim quantitative LDA dispersion
(Giannozzi et al., *Phys. Rev. B* **43**, 7231 (1991)).

### Offline CP2K twin (algorithm / product documentation)

**Not an absolute-\(\omega\) twin** (GTH-GPW ≠ NC-UPF PW). Documents that
CP2K Quickstep can combine fractional-q phonon analysis with UKS/LSD and
electronic smearing.

| Item | Value |
| ---- | ----- |
| Code | CP2K (when available offline) |
| Note | `tools/cp2k_oracle/inputs/dfpt-finite-q-spin-smearing.note` |
| Method sketch | Quickstep GPW, PADE/LDA, GTH-PADE + DZVP-MOLOPT-SR-GTH (Si q4) |
| Spin | LSD / unrestricted, multiplicity 1 |
| SMEAR | FERMI_DIRAC, \(T_e \approx 3158\) K (\(\approx 0.02\) Ry \(k_B T\)) |
| \(\mathbf{q}\) | fractional ≈ (0.3, 0, 0) or closest commensurate mesh |
| Host status (pin shift) | **CP2K binary not available** (`module avail cp2k` empty) |

Oracle path: `tools/cp2k_oracle/README.md` (offline only). Preferred oracle
label when a binary exists: CP2K phonon / DFPT + LSD + `&SMEAR`.

## Performance

Source: same REFERENCE + handoff wave 11. Gate:
\(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \le 10\) when a fair twin exists;
this slug is **`perf_exempt`**.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| CP2K DFPT spin+smear @ q≈(0.3,0,0) vs `reg_dfpt-finite-q-spin-smearing` | n/a | **≈ 11** | **perf_exempt** | n/a | no CP2K wall twin on pin host |

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K DFPT spin+smear wall twin on the build host this shift |
| ATV wall | ≈ 11 s (2-atom primitive, 6 responses × 2 spin, OMP=8; ≤ 60 s budget) |
| `ratio_atv_over_oracle` | n/a (exempt) |
| `within_10x` | n/a (exempt; correctness gates only) |

Handoff: the case `REFERENCE.md` (wave 11 row
`dfpt-finite-q-spin-smearing` → **perf_exempt**).

## Limitations

- **Gapped Si fill, not a true metal pin.** FD occupations stay near-integer
  at these settings; spectra track fixed-occ `dfpt-spin-incommensurate`. Real
  metallic DFPT (Fermi-surface terms on Al/Fe) is a different product claim.
- **No \(\delta\mu\) at \(q\neq 0\).** Charge-neutrality \(\delta\mu\) is Γ-only
  when \(\partial N/\partial\mu\) is large.
- **Absolute energy / \(\omega\) are ATV-frozen, not CP2K-matched.** GTH-GPW and
  NC-UPF PW differ; CP2K note documents settings / algorithm only.
- **Minimal ecut/k.** 10 Ry + \(1^3\) k biases \(\omega\) high; gates are
  wide windows, not literature Raman targets.
- **LSDA kernel in the regression** (API also accepts pure GGA kernels;
  hybrids rejected).
- **CPU only.** CUDA DFPT out of scope for this loop.
- **Commensurate supercell path** not exercised here (see
  `dfpt-finite-q-smearing` at \(X\), `dfpt-spin-smearing` at Γ).
- **Speed.** `perf_exempt` until an offline CP2K DFPT spin+smear wall is
  recorded in the oracle note.

## References

- Offline pin: `regressions/response/dfpt-finite-q-spin-smearing/REFERENCE.md`
- Regression: `regressions/response/dfpt-finite-q-spin-smearing/reg_dfpt-finite-q-spin-smearing.cpp`
- CP2K twin note: `tools/cp2k_oracle/inputs/dfpt-finite-q-spin-smearing.note`
- Oracle notes: `tools/cp2k_oracle/README.md`
- Roadmap: `docs/roadmap/dft.md` §11 DFPT
- Public header: `include/atomiverse/cpu/planewave/dfpt_phonon.h`
- GS method: `include/atomiverse/cpu/planewave/pw_solver.h`
- Unit tests: `tests/cpu/planewave/test_dfpt.cpp`
- Siblings: `dfpt-spin-incommensurate`, `dfpt-finite-q-smearing`,
  `dfpt-spin-smearing`, `dfpt-phonon-si`, `dfpt-incommensurate-si`
- Literature: Baroni et al., *Rev. Mod. Phys.* **73**, 515 (2001); de
  Gironcoli, *Phys. Rev. B* **51**, 6773 (1995); Giannozzi et al.,
  *Phys. Rev. B* **43**, 7231 (1991); Mermin, *Phys. Rev.* **137**, A1441
  (1965)
- Related tutorials: [dfpt-phonon-si](dfpt-phonon-si.md),
  [pw-smearing-al](pw-smearing-al.md),
  [fd-phonon-supercell](fd-phonon-supercell.md)
