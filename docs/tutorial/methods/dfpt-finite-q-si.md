# Finite-q DFPT phonons at X — diamond Si (PW LSDA, NC-UPF)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `dfpt-finite-q-si` |
| Regression | `regressions/response/dfpt-finite-q-si/` |
| Public API | `cpu::pw::DfptPhonon` (member `q_frac`) + `cpu::pw::DfptPhononResult` |
| Roadmap | [dft.md](../../roadmap/dft.md) §11 DFPT — finite-q product path ✓ |
| Dep | `dfpt-phonon-si` ✓ (Γ Sternheimer); `phonon-dispersion-si` ✓ (IFC Fourier consumer) |
| Oracle | CP2K offline twin note (`tools/cp2k_oracle/inputs/dfpt-finite-q-si.note`); no live CP2K in ctest |

## Summary

**Density-functional perturbation theory at fractional phonon wavevector
\(\mathbf{q}\)** extends the Γ-only `cpu::pw::DfptPhonon` foothold to
commensurate points of the primitive Brillouin zone. For rational
\(\mathbf{q}=(h/n_a,k/n_b,l/n_c)\) the product path is **supercell Γ-DFPT +
IFC Fourier** — analytically equivalent to direct finite-\(q\) Sternheimer for
Born–von Karman commensurate \(q\). This slug pins diamond Si at the **X
point** \(X=(1/2,0,0)\) (2×1×1 supercell, 4 atoms, 6 branches). Same
restrictions as Γ DFPT: restricted LSDA, NC-UPF, fixed occupations. Cite:
Baroni, de Gironcoli, Dal Corso, Giannozzi, *Rev. Mod. Phys.* **73**, 515
(2001); Giannozzi et al., *Phys. Rev. B* **43**, 7231 (1991).

## Method

**Level.** Phonon dynamical matrix \(C(\mathbf{q})\) of a periodic crystal from
self-consistent first-order density response, not \(6N\) finite-displacement
SCF solves. At \(\mathbf{q}=\mathbf{0}\) the engine assembles the Cartesian
force constants via Sternheimer conduction-space CG, Hartree + LDA-kernel
screening, analytic Ewald second derivatives, and KB nonlocal derivative
terms (see `dfpt-phonon-si`).

**Commensurate finite \(\mathbf{q}\).** A phonon at rational

\[
\mathbf{q}
  = \tfrac{h}{n_a}\mathbf{b}_1
  + \tfrac{k}{n_b}\mathbf{b}_2
  + \tfrac{l}{n_c}\mathbf{b}_3
\]

is exactly a **Γ phonon of the \(n_a\times n_b\times n_c\) supercell**, folded
back by the Bloch phase \(e^{i\mathbf{q}\cdot\mathbf{R}}\). Implementation:

1. Resolve the smallest positive integers \(n_a,n_b,n_c\le 6\) with product
   \(\le 8\) such that \(n_a q_a\in\mathbb{Z}\) (etc.); throw if incommensurate
   (direct Sternheimer at arbitrary \(q\) is a follow-on).
2. Expand the primitive `Frame` with `vib::expand_supercell`.
3. Run the proven Γ assembly `internal::dfpt_gamma_force_constants` on the
   supercell (same response tolerances as Γ DFPT).
4. Fourier-transform the supercell IFC with `PhononDispersion::frequencies_at_q`
   at the requested `q_frac` of the **primitive** cell.
5. Mass-weight and report \(\omega\) [cm⁻¹] for the primitive \(3N\) branches
   (imaginary eigenvalues as negative \(\omega\)).

**X-point twin.** \(X=(1/2,0,0)\) ⇒ \(n_a=2\), \(n_b=n_c=1\) (4 Si atoms).
Six branches: TA (doublet), LA, LO, TO (doublet) in the literature LDA
ordering (Giannozzi 1991). Minimal tutorial settings (10 Ry, Γ on the SC)
admit a soft/imaginary TA pair; gates use \(|\omega|\) windows and
\(n_\mathrm{imag}\le 2\).

**Observables (on `DfptPhononResult`, never on Frame).**

| Symbol | Definition | Unit |
| ------ | ---------- | ---- |
| \(\omega_k\) | `frequencies` (length \(3N\), ascending) | cm⁻¹ |
| \(\mathrm{Re}\,C(q)\) | `force_constants` | eV/Å² |
| \(\mathrm{Im}\,C(q)\) | `force_constants_imag` (≈0 at Γ) | eV/Å² |
| — | `max_asymmetry` before Hermitian projection | eV/Å² |
| — | `scf_converged` / `response_converged` | bool |

**Key exclusions.**

- Direct (non-supercell) Sternheimer at incommensurate \(q\).
- LO–TO / Born effective charges (nonpolar Si does not need them at X).
- GGA \(f_{xc}\), metallic smearing / Fermi-surface terms, collinear spin.
- Toy potential (`upf_path` empty); UPF NC only.
- Full production dispersion path at converged ecut/k — use denser settings
  outside the ≤ 60 s e2e budget.
- Live CP2K / QE / ABINIT at ctest time (offline pins only).

Also fixed in this slug: the **Ewald real-space lattice sum** previously used
the transpose of the cell matrix (harmless for the symmetric diamond
primitive, fatal for non-symmetric supercells). `EwaldSummation` and DFPT
Ewald Hessians now build \(L = h\mathbf{a}+k\mathbf{b}+l\mathbf{c}\) from
**rows**.

## Implementation

Public surface only (`cpu::Frame` + concrete method members; no `*Config`, no
`make_*` / `build_*`). Headers:
`include/atomiverse/cpu/planewave/dfpt_phonon.h`,
`pw_solver.h`; internals under `cpu/planewave/internal/dfpt.h` (private).

Minimal sequence (matches `reg_dfpt-finite-q-si.cpp`):

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/dfpt_phonon.h"
#include "atomiverse/cpu/planewave/pw_solver.h"

using namespace ATV;
using namespace ATV::cpu::pw;

constexpr double kA = 5.43;       // Å
constexpr double kEcutRy = 10.0;
constexpr int kMesh = 1;          // Γ on 2×1×1 SC ≡ fold of primitive X

cpu::Frame frame(2);
frame.span(kAtom_type)[0] = 14;
frame.span(kAtom_type)[1] = 14;
frame.span(kAtom_mass)[0] = 28.0855;
frame.span(kAtom_mass)[1] = 28.0855;
frame.span(kAtom_x)[1] = kA / 4.0;
frame.span(kAtom_y)[1] = kA / 4.0;
frame.span(kAtom_z)[1] = kA / 4.0;
const double h = kA / 2.0;
Mat3 cell;
// fcc primitive rows: (0,h,h), (h,0,h), (h,h,0)
cell(0, 0) = 0; cell(0, 1) = h; cell(0, 2) = h;
cell(1, 0) = h; cell(1, 1) = 0; cell(1, 2) = h;
cell(2, 0) = h; cell(2, 1) = h; cell(2, 2) = 0;
frame.box.set_cell(cell);

PW method;
method.upf_path = "tests/data/pw/Si_lda_nc.upf";
method.ecut_ry = kEcutRy;
method.k_mesh = kMesh;
method.xc = cpu::scf::XCType::LSDA;   // DFPT kernel is LSDA-only
method.tol = 1e-8;
method.max_iter = 200;

DfptPhonon dfpt;
dfpt.q_frac = {0.5, 0.0, 0.0};       // X of the primitive cell
dfpt.response_tol = 1e-4;
dfpt.response_max_iter = 120;
dfpt.cg_tol = 1e-8;
dfpt.mix_beta = 0.25;
const DfptPhononResult r = dfpt.compute(method, frame);
// r.frequencies [cm⁻¹], length 6; r.force_constants [eV/Å²]
```

`compute` throws `std::invalid_argument` for non-LSDA XC, empty `upf_path`,
metallic smearing, collinear spin, or incommensurate / oversized \(q\)
(supercell product > 8). The input Frame is not mutated.

```bash
# scoped build (binding) — never bare cmake --build build
cmake --build build -j$(nproc) --target atomiverse reg_dfpt-finite-q-si
./build/regressions/reg_dfpt-finite-q-si
# or: cmake --build build --target run_reg_dfpt-finite-q-si
```

Relevant sources: `src/cpu/planewave/dfpt_phonon.cpp` (commensurate dispatch),
`src/cpu/planewave/internal/dfpt.cpp` (Γ Sternheimer assembly + Ewald rows),
`include/atomiverse/cpu/vibration/phonon_dispersion.h` (IFC Fourier).

## Validation (vs offline pins / literature)

Source: `regressions/response/dfpt-finite-q-si/REFERENCE.md` (hard-coded;
never live oracle in ctest). Twin measured 2026-07-25, Release, OMP=8.

**System.** Diamond Si primitive, a = **5.43 Å**, `Si_lda_nc.upf`, LSDA,
**10 Ry**, k-mesh **1×1×1** on the 2×1×1 supercell, response tol 1e-4 /
mix 0.25 / max_iter 120.

| Quantity | Atomiverse / pin | Reference | Tol / gate | Notes |
| -------- | ---------------: | --------: | ---------- | ----- |
| SCF energy (4-atom SC) | −30.164 Ry | ~2 × primitive scale | finite | internal Ry OK |
| \(\omega(X)\) [cm⁻¹] | −130.3, −130.3, 229.8, 448.2, 672.6, 672.6 | — | hard pin | measured ATV |
| \(\|\omega\|\) sorted | 130, 130, 230, 448, 673, 673 | — | windows below | regression gates |
| Low 3 \(\|\omega\|\) | in [50, 400] | lit. TA/LA biased | hard | soft TA allowed |
| High 3 \(\|\omega\|\) | in [300, 850] | lit. LO/TO biased | hard | incomplete basis |
| Literature LDA X | — | TA≈146, LA≈397, LO≈413, TO≈455 | qualitative | Giannozzi PRB 1991 |
| \(n_\mathrm{imag}\) | 2 | ≤ 2 accepted | hard | minimal ecut/k |
| max_asymmetry | ≤ 1e-3 eV/Å² | Hermitian residual | hard | before projection |
| Wall | ≈ 43 s | ≤ 60 s e2e | hard | OMP=8 |

Gates intentionally use **magnitude windows** rather than signed literature
digits: the 10 Ry / Γ-SC twin is tutorial-class and can report a soft TA
doublet. Production dispersions require denser settings outside the
regression budget.

Unit tests (`test_cpu_pw_dfpt`): analytic Ewald dynamical matrix vs FD of
Ewald forces, Sternheimer CG on free-electron Hamiltonian, guard-rail
throws — **no full SCF** under `tests/`.

CP2K offline note: when a Quickstep DFPT / `VIBRATIONAL_ANALYSIS` twin is
available, record \(\omega(X)\) for all 6 branches + wall under matched OMP
(`tools/cp2k_oracle/inputs/dfpt-finite-q-si.note`). No CP2K binary on the pin
host this shift.

## Performance

Source: the case `REFERENCE.md` (wave 2) + REFERENCE.md.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | -----------------------: | ----: | ---- | ------ |
| Si X finite-q DFPT | n/a (no binary) | ≈ 43 | n/a | n/a | **`perf_exempt`** |

Gate: \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \le 10\) when a fair twin
exists. Here: **`perf_exempt`** — no CP2K DFPT wall recorded on the build
host this shift. Re-bench with OMP matched when the oracle binary lands.

## Limitations

- **Commensurate \(q\) only** (product \(n_a n_b n_c \le 8\)); direct
  incommensurate Sternheimer is open follow-on work.
- LSDA kernel only; insulators only; restricted spin; UPF NC only.
- Minimal twin (10 Ry, SC Γ) is **not** production-converged X-point
  frequencies — soft TA pair is accepted with \(n_\mathrm{imag}\le 2\).
- No LO–TO splitting / Born charges path.
- CPU only; CUDA DFPT out of scope for the CP2K capability loop.
- Fair speed vs CP2K still missing (offline note only).

## References

- Offline pin: `regressions/response/dfpt-finite-q-si/REFERENCE.md`
- Regression: `regressions/response/dfpt-finite-q-si/reg_dfpt-finite-q-si.cpp`
- Oracle note: `tools/cp2k_oracle/inputs/dfpt-finite-q-si.note`
- Oracle rules: `tools/cp2k_oracle/README.md` (never live in ctest)
- Γ foothold tutorial: [dfpt-phonon-si](dfpt-phonon-si.md)
- FD supercell machinery: [fd-phonon-supercell](fd-phonon-supercell.md)
- Roadmap: [dft.md](../../roadmap/dft.md) §11
- Baroni, de Gironcoli, Dal Corso, Giannozzi, *Rev. Mod. Phys.* **73**, 515
  (2001)
- Giannozzi, de Gironcoli, Pavone, Baroni, *Phys. Rev. B* **43**, 7231 (1991)
