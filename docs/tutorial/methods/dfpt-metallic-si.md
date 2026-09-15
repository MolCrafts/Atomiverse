# DFPT phonons + metallic smearing — diamond Si (PW LSDA, NC-UPF)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `dfpt-metallic-si` |
| Regression | `regressions/response/dfpt-metallic-si/` |
| Public API | `cpu::pw::DfptPhonon` / `DfptPhononResult` + `PW` members `smearing`, `degauss` [Ry], `extra_bands` |
| Roadmap | [dft.md](../../roadmap/dft.md) §11 DFPT — metallic occupation smearing ✓ |
| Dep | `dfpt-phonon-si` ✓ (Γ fixed-occ); `pw-smearing-al` ✓ (Mermin GS); siblings: GGA / spin / finite-\(q\) DFPT |

## Summary

**Density-functional perturbation theory** at \(\mathbf{q}=0\) for the
plane-wave method `cpu::pw::PW` with **restricted metallic occupation
smearing** (Fermi–Dirac / Methfessel–Paxton / Gaussian). Ground-state SCF
uses the Mermin free energy \(F = E_{\mathrm{KS}} - TS\); the Sternheimer
response projects out bands with \(f \ge 10^{-8}\) and restores
\(\int\Delta n = 0\) with a Fermi-level shift \(\delta\mu\) when
\(\partial N/\partial\mu\) is non-negligible (de Gironcoli). Product gate:
`reg_dfpt-metallic-si` — diamond Si NC-UPF LSDA twin with FD
`degauss = 0.02` Ry. Sharp acceptance is the **same-Hamiltonian** free-energy
central second difference for \(C[(0x),(0x)]\), not a GTH-GPW absolute-\(E\)
match to CP2K. Performance is **`perf_exempt`** (no CP2K DFPT+`&SMEAR` wall
on the pin host).

## Method

**Level.** Γ-point DFPT phonons (Baroni et al.) over a restricted PW KS
ground state with fractional occupations. Pseudopotential path is
**norm-conserving UPF**; XC for this pin is **LSDA** (pure GGA DFPT is a
sibling product, `dfpt-gga-si`). Public calculator: `cpu::pw::DfptPhonon`
with members `response_tol`, `cg_tol`, `mix_beta`, `q_frac` (default Γ).

**Occupations (closed-shell).** With \(\sigma =\) `degauss` [Ry] and reduced
energy \(x = (\varepsilon-\mu)/\sigma\), band occupation \(f_b\in[0,2]\)
doubles the spinless Fermi–Dirac fill (same convention as
[`pw-smearing-al`](pw-smearing-al.md)). SCF total when smearing is on is the
**Mermin free energy** \(F\); `extra_bands` pads empty states above the
valence manifold (open metals; this twin uses `extra_bands = 0` because
gapped Si keeps near-integer FD occupations at 2×2×2 / 10 Ry).

**Response density.** Linear-response occupation-weighted density change:

\[
\Delta n
  = \sum_{nk} w_k\, f_{nk}\,
    2\,\mathrm{Re}\bigl(\psi_{nk}^*\,\Delta\psi_{nk}\bigr).
\]

Sternheimer CG is solved in the subspace orthogonal to occupied (or
fractionally occupied) bands. At Γ, when the density of states at \(\mu\) is
non-zero, a Lagrange multiplier \(\delta\mu\) enforces charge neutrality of
\(\Delta n\) (de Gironcoli, *Phys. Rev. B* **51**, 6773 (1995)).

**Dynamical matrix.** Self-consistent first-order density → Hartree +
\(f_{\mathrm{xc}}\) screening → analytic Ewald + KB nonlocal second
derivatives → Cartesian force constants \(C\) [eV/Å²] → mass-weighted
Hermitian diagonalization → \(\omega\) [cm⁻¹] ascending (imaginary modes as
negative \(\omega\)). Acoustic sum rule is **emergent**, never projected.

**What is included**

| Piece | Status |
| ----- | ------ |
| Restricted FD / Gaussian / MP smear at Γ | product |
| Commensurate finite \(\mathbf{q}\) + smear | product path (supercell Γ) |
| Same-Hamiltonian free-energy FD pin | regression gate |
| LSDA kernel (this twin) | product |

**What is excluded (throws `std::invalid_argument`, not silent)**

| Piece | Notes |
| ----- | ----- |
| Spin + smear | collinear DFPT is fixed-occ only (`dfpt-spin-si`) |
| Incommensurate \(\mathbf{q}\) + smear | fixed-occ / spin incommensurate siblings |
| Hybrid XC / empty `upf_path` | same guards as fixed-occ DFPT |
| LO–TO / Born charges | nonpolar Si; not required |
| CUDA DFPT | out of this loop |

## Implementation

Public surface only: construct `cpu::pw::PW` and `cpu::pw::DfptPhonon`, set
members, call `compute(method, frame)`. **No** public `*Config` bag, **no**
`make_*` / `build_*` factory. Headers:
`include/atomiverse/cpu/planewave/dfpt_phonon.h`,
`include/atomiverse/cpu/planewave/pw_solver.h`. Spectra and force constants
live on `DfptPhononResult` (Frame purity — Frame is geometry I/O only).

Minimal sequence (matches `reg_dfpt-metallic-si.cpp`):

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/dfpt_phonon.h"
#include "atomiverse/cpu/planewave/pw_solver.h"

using namespace ATV;
using namespace ATV::cpu::pw;

// Diamond Si fcc primitive, a = 5.43 Å, 2 atoms (REFERENCE.md)
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
method.smearing = Smearing::fermi_dirac;
method.degauss = 0.02;   // Ry
method.extra_bands = 0;  // gapped twin; open metals use > 0
method.tol = 1e-8;
method.max_iter = 200;

DfptPhonon dfpt;
dfpt.response_tol = 1e-5;
dfpt.cg_tol = 1e-9;
dfpt.mix_beta = 0.7;
const DfptPhononResult r = dfpt.compute(method, frame);
// r.frequencies [cm⁻¹] length 6; r.force_constants [eV/Å²] 6×6
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
        -DATV_BUILD_TESTS=ON -DATV_BUILD_PW=ON \
        -DATV_BUILD_REGRESSIONS=ON -DATV_BUILD_CUDA=OFF \
        -DBLA_VENDOR=OpenBLAS
  cmake --build build -j$(nproc) --target atomiverse reg_dfpt-metallic-si
  ./build/regressions/reg_dfpt-metallic-si
'
# or: cmake --build build --target run_reg_dfpt-metallic-si
```

Source consumer:
`regressions/response/dfpt-metallic-si/reg_dfpt-metallic-si.cpp`.
Unit surface for DFPT primitives stays under `tests/cpu/planewave/` (no e2e
SCF under `tests/`).

## Validation (CP2K)

Source:
[`regressions/response/dfpt-metallic-si/REFERENCE.md`](../../../regressions/response/dfpt-metallic-si/REFERENCE.md)
(offline pins only; **never** live CP2K in ctest).

### ATV system (regression twin)

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si, 2-atom fcc primitive, \(a = 5.43\) Å |
| PP | `tests/data/pw/Si_lda_nc.upf` (ONCVPSP NC, in-tree) |
| XC | `XCType::LSDA` |
| Smearing | Fermi–Dirac, `degauss = 0.02` Ry, `extra_bands = 0` |
| Cutoff | **10 Ry** (minimal-twin budget) |
| k-mesh | Monkhorst–Pack **2×2×2** (unshifted) |
| SCF | density L2 tol \(10^{-8}\), `max_iter = 200` |
| Response | `response_tol = 10^{-5}`, `mix_beta = 0.7`, CG tol \(10^{-9}\) |
| Pin date | 2026-07-26 (Arrhenius x86_64, Release, OMP=8) |

Gapped Si at these settings keeps near-integer FD occupations, so
\(F \approx E_{\mathrm{KS}}\) and results track the fixed-occupation sibling
`dfpt-phonon-si`. Open metals need denser k / ecut and `extra_bands > 0`.

### Accuracy table (hard-coded pins)

| Quantity | Atomiverse / pin | Oracle / twin | Tol | Notes |
| -------- | ---------------: | ------------: | --- | ----- |
| Free-energy FD \(C[(0x),(0x)]\) | **17.2393 eV/Å²** | same-Hamiltonian central FD of \(F\), \(h=0.03\) Å | pin definition | identical ecut/k/XC/UPF/FD |
| DFPT \(C[(0x),(0x)]\) | **17.3416 eV/Å²** | FD pin 17.2393 | \(\lvert\Delta\rvert\le 0.30\) | ~0.6 % agreement |
| Acoustic \(\omega\) (no ASR) | −2.01, −2.00, −2.00 cm⁻¹ | 0 (translational) | \(\lvert\omega\rvert\le 30\) | emergent ASR |
| Optical triplet | **579.45, 579.45, 579.60** cm⁻¹ | LDA / Raman window ~517 cm⁻¹ | \([500,650]\), spread ≤ 10 | +12 % = documented minimal-settings bias (10 Ry, \(2^3\) k) |
| Max row sum (ASR) | 0.0004 eV/Å² | 0 | ≤ 0.05 | translational invariance |
| Pre-symmetrization asymmetry | \(3.4\times10^{-10}\) eV/Å² | 0 | ≤ \(10^{-5}\) | \(C\) Hermitian to roundoff |

Literature anchor (not a regression pin): Giannozzi et al., *Phys. Rev. B*
**43**, 7231 (1991) DFPT-LDA optical Γ ≈ 517 cm⁻¹.

### Offline CP2K twin (algorithm / SMEAR documentation)

**Not an absolute-energy or absolute-\(\omega\) twin** (GTH-GPW ≠ NC-UPF PW).
Documents that CP2K Quickstep can combine `VIBRATIONAL_ANALYSIS` (or FD
phonons) with `&SMEAR` on a Si-like solid. Absolute acceptance stays on the
ATV free-energy FD Hessian.

| Item | Value |
| ---- | ----- |
| Preferred oracle | CP2K `VIBRATIONAL_ANALYSIS` + `&SMEAR METHOD FERMI_DIRAC` |
| Host status | CP2K module/binary **not available** on the Arrhenius login node used for this pin (`module avail cp2k` empty) |
| Sketch | diamond Si primitive, a ≈ 5.43 Å; GPW PADE + GTH-PADE + DZVP-MOLOPT-SR-GTH (Si q4); \(T_e\) ≈ 3158 K for `degauss ≈ 0.02` Ry; `ADDED_MOS` ≥ 4; MP \(2\times2\times2\) |
| Note path | `tools/cp2k_oracle/inputs/dfpt-metallic-si.note` |

Oracle path: `tools/cp2k_oracle/README.md` (offline only). When a CP2K binary
is available, record \(\omega(\Gamma)\) [cm⁻¹] + wall + optional fair ratio
\(r = T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) (require \(\le 10\)).

## Performance

Source: same REFERENCE + handoff. Gate:
\(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \le 10\) unless exempted.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| DFPT+`&SMEAR` vs `reg_dfpt-metallic-si` | n/a | **≈ 16.5** | n/a | n/a | **`perf_exempt`** — no CP2K binary on pin host |

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K DFPT+smear wall twin on the build host this shift |
| ATV wall (informative) | ≈ 16.5 s (2-atom Si, 6 responses, 8 k-points, OMP=8) |
| Budget | ≤ 60 s regression |
| `ratio_atv_over_cp2k` | n/a (exempt) |
| `within_10x` | n/a (exempt; correctness gates only) |

Handoff: the case `REFERENCE.md` (`dfpt-metallic-si` →
**perf_exempt**).

## Limitations

- **Textbook twin is gapped Si with FD smear**, not a dense multi-band metal.
  Near-integer occupations make \(F \approx E\); open metals need denser
  k/ecut and `extra_bands > 0`.
- **Absolute \(\omega\) / \(E\) are not CP2K-matched.** GTH-GPW and NC-UPF PW
  are different Hamiltonians; CP2K documents SMEAR + vib settings only until
  a fair offline twin is recorded.
- **Spin + smear** and **incommensurate \(\mathbf{q}\) + smear** remain
  rejected (follow-ons). Collinear fixed-occ DFPT is product elsewhere.
- **Minimal settings bias** (+12 % vs experimental Raman / production DFPT)
  is intentional for the ≤ 60 s gate — do not treat optical cm⁻¹ as
  production-converged.
- **CPU only**; CUDA DFPT out of scope for the CP2K capability loop.
- **No public Config / factories**; parameters are members on `PW` and
  `DfptPhonon`.
- **Speed.** Informative ATV wall only; no fair \(r\) until CP2K DFPT+smear
  is available offline.

## References

- Offline pin: `regressions/response/dfpt-metallic-si/REFERENCE.md`
- Regression: `regressions/response/dfpt-metallic-si/reg_dfpt-metallic-si.cpp`
- CP2K twin note: `tools/cp2k_oracle/inputs/dfpt-metallic-si.note`
- Oracle rules: `tools/cp2k_oracle/README.md`
- Handoff (**perf_exempt**): the case `REFERENCE.md`
- Roadmap: `docs/roadmap/dft.md` §11 metallic DFPT ✓
- Public headers: `include/atomiverse/cpu/planewave/dfpt_phonon.h`,
  `pw_solver.h`
- Fixed-occ sibling: [`dfpt-phonon-si`](dfpt-phonon-si.md)
- Metallic GS sibling: [`pw-smearing-al`](pw-smearing-al.md)
- Literature: Baroni, de Gironcoli, Dal Corso, Giannozzi, *Rev. Mod. Phys.*
  **73**, 515 (2001); de Gironcoli, *Phys. Rev. B* **51**, 6773 (1995);
  Giannozzi et al., *Phys. Rev. B* **43**, 7231 (1991); Mermin,
  *Phys. Rev.* **137**, A1441 (1965)
- Narrative PW background: [ch08-plane-wave](../ch08-plane-wave.md)
