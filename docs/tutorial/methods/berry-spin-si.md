# Collinear-spin Berry-phase polarization — bulk Si (PW)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `berry-spin-si` |
| Regression | `regressions/pw/berry-spin-si/` |
| Public API | `cpu::pw::PW` (`spin_polarized`, `multiplicity`) + `cpu::pw::BerryPolarization` / `BerryPolarizationResult` |
| Roadmap | [dft.md](../../roadmap/dft.md) §11 Berry-phase polarization (collinear spin ✓) |
| Wave | CP2K capability loop — wave 10 / merged 2026-07-26 |
| Dual gate | Literature symmetry pins (centrosymmetric lattice + elemental \(Z^*=0\)); speed **`perf_exempt`** |

## Summary

**King–Smith–Vanderbilt (KSV) formal polarization** on the **collinear spin**
plane-wave path: `cpu::pw::PW` with `spin_polarized = true` and fixed
occupations, then `cpu::pw::BerryPolarization::compute` on frozen
spin-resolved \(V_\mathrm{eff}\). Each spin channel owns its occupied
manifold; string-averaged Berry phases \(\varphi_{i,\uparrow}\) and
\(\varphi_{i,\downarrow}\) are combined with **spin factor 1 per channel**
and total polarization is wrapped with quantum \(1\) (nspin=2 convention).
The product twin is diamond silicon (multiplicity 1, \(N_\uparrow=N_\downarrow=4\))
with the same **basis-independent** scientific pins as closed-shell
`berry-polarization`: centrosymmetric formal \(p_i\) on the half-quantum
lattice (Spaldin) and Born effective charge \(Z^*(\mathrm{Si})=0\) for an
elemental crystal (Pick–Cohen–Martin). Absolute phases are **not** compared
to CP2K Quickstep (GTH + Gaussian). Product regression:
`reg_berry-spin-si`. Performance: **`perf_exempt`** (no fair CP2K/QE PW
wall twin on the pin host).

## Method

**Level.** Modern theory of polarization on periodic plane-wave KS DFT
(King-Smith & Vanderbilt, *Phys. Rev. B* **47**, 1651 (1993); Resta,
*Rev. Mod. Phys.* **66**, 899 (1994)). After a collinear SCF on
`cpu::pw::PW` members, \(V_\mathrm{eff}^\sigma(\mathbf r)\) is frozen per
spin \(\sigma\in\{\uparrow,\downarrow\}\). Occupied Bloch states are
diagonalized **non-self-consistently** on strings of \(\mathbf k\)-points
along each reciprocal lattice vector \(\mathbf b_i\).

**String geometry.** For direction \(i\) (lattice axis 0,1,2), a
\(N_\perp\times N_\perp\) mesh of strings is laid in the transverse plane;
each string has \(J\) points with fractional component
\(k_i = j/J\), \(j=0,\ldots,J-1\) (`k_perp` \(=N_\perp\),
`k_string` \(=J\)). Link matrices are occupied–occupied overlaps between
adjacent string points; the closing link \(k_{J-1}\to k_0+\mathbf b_i\) uses
the periodic-gauge coefficient shift
\(c_{n,k_0+\mathbf b}(\mathbf G)=c_{n,k_0}(\mathbf G+\mathbf b)\).

**Collinear electronic polarization.** Per direction \(i\) and spin
\(\sigma\), the string-averaged Berry phase is

\[
\varphi_{i,\sigma}
  = \operatorname{Im}\ln\prod_j \det S^{(j)}_{i,\sigma},
\]

with unimodular link determinants (phase of the product). Electronic
polarization (fractional elementary charge \(e\)) is

\[
p^\mathrm{el}_i
  = \frac{\varphi_{i,\uparrow}+\varphi_{i,\downarrow}}{2\pi}
\quad\text{(factor \(f=1\) per channel)}.
\]

`BerryPolarizationResult::phase_el[i]` stores the **mean** phase
\((\varphi_{i,\uparrow}+\varphi_{i,\downarrow})/2\) so that, when
\(N_\uparrow=N_\downarrow\), the closed-shell identity
\(p^\mathrm{el}=2\varphi/2\pi\) holds with that mean. Total formal
polarization wraps into \((-0.5,0.5]\) (quantum \(1\)).

**Closed-shell contrast.** Restricted path (`spin_polarized == false`):
spin factor \(f=2\), \(p^\mathrm{el}=2\varphi/2\pi\), wrap quantum \(2\).
Unchanged by this slug — see `regressions/pw/berry-polarization`.

**Ionic term.** Spin-independent:

\[
p^\mathrm{ion}_i
  = \sum_a Z^\mathrm{val}_a\, \tau_{a,i},
\qquad
\tau_{a,i}\in[0,1).
\]

For diamond Si with \(Z^\mathrm{val}=4\) at fractional sites
\((0,0,0)\) and \(\tfrac14(1,1,1)\) (ideal), each lattice component is
exactly \(1\).

**Born effective charge (elemental ASR).** Finite sublattice displacement
\(\delta\) of atom 1 along \(\mathbf a_3\) defines

\[
Z^*
  = \frac{\operatorname{wrap}\bigl(p_3(\delta)-p_3(0);\, q=1\bigr)}{\delta}.
\]

For a monoatomic crystal, the acoustic sum rule requires
\(Z^*=0\) (Pick, Cohen & Martin, *Phys. Rev. B* **1**, 910 (1970)).

**XC / occupations.** `XCType::LSDA` on the PW grid; collinear fixed
occupations from `multiplicity` (here \(m=1\) → \(N_\uparrow=N_\downarrow\)).
**Metallic smearing is rejected** (`BerryPolarization::compute` throws) —
formal Berry polarization requires an insulator.

**Observables.**

| Symbol | Definition | Unit |
| ------ | ---------- | ---- |
| \(\varphi_i\) | `phase_el[i]` (mean of spin phases) | rad |
| \(p^\mathrm{el}_i\) | `p_el_frac[i]` | \(e\) (fractional) |
| \(p^\mathrm{ion}_i\) | `p_ion_frac[i]` | \(e\) (fractional) |
| \(p_i\) | `p_frac[i]` total, wrapped (quantum 1) | \(e\) (fractional) |
| \(\mathbf d\) | `dipole_e_ang` \(=\sum_i p_i\mathbf a_i\) | e Å |
| \(E_\mathrm{SCF}\) | `energy_ry` | Ry |

Internal energy remains Ry on the result; public Frame energy I/O for
energy-only drivers is still eV when written via the Evaluate / single-point
path — this post-processor returns an owned result struct, not Frame meta.

**Key exclusions (this slug).**

- Closed-shell Berry only → `berry-polarization`.
- Metallic Berry / Fermi-surface corrections.
- Noncollinear spin / SOC.
- GPW / Gaussian Berry strings.
- Absolute \(p_i\) / phase parity with CP2K Quickstep GTH+DZVP (different
  PP and basis class).
- Production-converged \(E_\mathrm{cut}\) / dense strings when they break
  the ≤ 60 s e2e budget.
- Live CP2K / QE at ctest time (offline pins only).
- Public `*Config` bags; `make_*` / `build_*` factories.

## Implementation

Public surface only (`cpu::Frame` + concrete method / post-processor
members; no `*Config`, no factories). Headers:
`include/atomiverse/cpu/planewave/berry_polarization.h`,
`pw_solver.h`. Implementation:
`src/cpu/planewave/berry_polarization.cpp` (collinear SCF → freeze
\(V_\mathrm{eff}^\sigma\) → NSCF string phases per spin → wrap quantum 1).

Minimal sequence (matches `reg_berry-spin-si.cpp`):

```cpp
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/berry_polarization.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cpu/scf/xc_functional.h"

using namespace ATV;
using namespace ATV::cpu::pw;

constexpr double kA = 5.43;       // Å
constexpr double kEcutRy = 15.0;  // tutorial incomplete basis
constexpr int kMeshScf = 2;       // 2×2×2 MP SCF
constexpr double kDelta = 0.02;   // fractional displacement along a_3

// diamond Si, 2-atom fcc primitive; optional delta_frac on atom 1 along a_3
cpu::Frame make_si_diamond(double a_ang, double delta_frac_a3) {
    cpu::Frame frame(2);
    frame.span(kAtom_type)[0] = 14;
    frame.span(kAtom_type)[1] = 14;
    frame.span(kAtom_mass)[0] = 28.0855;
    frame.span(kAtom_mass)[1] = 28.0855;
    const double h = a_ang / 2.0;
    Mat3 cell;
    cell(0, 0) = 0; cell(0, 1) = h; cell(0, 2) = h;
    cell(1, 0) = h; cell(1, 1) = 0; cell(1, 2) = h;
    cell(2, 0) = h; cell(2, 1) = h; cell(2, 2) = 0;
    frame.box.set_cell(cell);
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_y)[0] = 0.0;
    frame.span(kAtom_z)[0] = 0.0;
    frame.span(kAtom_x)[1] = a_ang / 4.0 + delta_frac_a3 * cell(2, 0);
    frame.span(kAtom_y)[1] = a_ang / 4.0 + delta_frac_a3 * cell(2, 1);
    frame.span(kAtom_z)[1] = a_ang / 4.0 + delta_frac_a3 * cell(2, 2);
    return frame;
}

PW method;
method.upf_path = "tests/data/pw/Si_lda_nc.upf";
method.ecut_ry = kEcutRy;
method.k_mesh = kMeshScf;
method.tol = 1e-6;
method.max_iter = 50;
method.xc = cpu::scf::XCType::LSDA;
method.spin_polarized = true;
method.multiplicity = 1;          // N↑ = N↓ = 4 (8 valence e⁻)

BerryPolarization berry;
berry.k_perp = 2;
berry.k_string = 4;

const auto ideal = make_si_diamond(kA, 0.0);
const BerryPolarizationResult r0 = berry.compute(method, ideal);
// r0.p_frac[d], r0.p_ion_frac[d], r0.phase_el[d], r0.scf_converged

BerryPolarization berry_z;
berry_z.k_perp = 2;
berry_z.k_string = 4;
berry_z.directions = {false, false, true};  // only a_3 for Z*
const auto disp = make_si_diamond(kA, kDelta);
const BerryPolarizationResult r1 = berry_z.compute(method, disp);
const double dp = wrap_centered(r1.p_frac[2] - r0.p_frac[2], /*quantum=*/1.0);
const double z_born = dp / kDelta;
```

`compute` throws `std::invalid_argument` for metallic smearing or invalid
string mesh (`k_perp < 1`, `k_string < 2`). Collinear spin is an **explicit
member path**, not a silent fallback.

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT="$OPENBLASDIR"
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_PW=ON \
        -DATV_BUILD_REGRESSIONS=ON -DATV_BUILD_CUDA=OFF \
        -DBLA_VENDOR=OpenBLAS
  cmake --build build -j$(nproc) --target atomiverse reg_berry-spin-si
  ./build/regressions/reg_berry-spin-si
'
```

Unit (no full SCF e2e): `test_cpu_pw_berry` — string layout, direction axes,
link-det phase math, `wrap_centered`, invalid-argument guards.

**File pointers**

| Role | Path |
| ---- | ---- |
| Public API | `include/atomiverse/cpu/planewave/berry_polarization.h` |
| Implementation | `src/cpu/planewave/berry_polarization.cpp` |
| PW method / collinear SCF | `include/atomiverse/cpu/planewave/pw_solver.h` |
| Unit test | `tests/cpu/planewave/test_berry_polarization.cpp` |
| Regression | `regressions/pw/berry-spin-si/reg_berry-spin-si.cpp` |
| Offline pin | `regressions/pw/berry-spin-si/REFERENCE.md` |
| CP2K oracle note | `tools/cp2k_oracle/inputs/si_berry_spin_note.inp` |
| Closed-shell twin | `regressions/pw/berry-polarization/` |

## Validation (vs literature / CP2K note)

Source: [`regressions/pw/berry-spin-si/REFERENCE.md`](../../../regressions/pw/berry-spin-si/REFERENCE.md)
(offline pin; **never live CP2K or QE in ctest**). Scientific product is
**symmetry-exact literature**, not absolute-phase parity with GTH Quickstep.

### Pseudopotential

| Item | Value |
| ---- | ----- |
| File | `tests/data/pw/Si_lda_nc.upf` (in-tree) |
| Family | PseudoDojo ONCVPSP-LDA-SR-noCC v0.4 |
| \(Z_\mathrm{val}\) | 4 |
| Same family as | `pw-lda-si`, `berry-polarization`, `bands-dos-si` |

### Twin geometry and settings

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si, 2-atom fcc primitive cell |
| Lattice constant \(a\) | 5.43 Å |
| Basis atoms (ideal) | \((0,0,0)\), \((a/4)(1,1,1)\) |
| Displaced twin | atom 1 \(+0.02\,\mathbf a_3\) |
| Cell | \((a/2)(0,1,1)\), \((a/2)(1,0,1)\), \((a/2)(1,1,0)\) |
| SCF \(E_\mathrm{cut}\) | 15 Ry (tutorial; ONCV production ~60 Ry) |
| SCF k-mesh | \(2\times2\times2\) |
| XC | `XCType::LSDA` |
| Spin | collinear, multiplicity 1 (\(N_\uparrow=N_\downarrow=4\)) |
| SCF tol | \(10^{-6}\) |
| Strings | `k_perp = 2`, `k_string = 4` |

### Accuracy table

| Quantity | Atomiverse / pin | Reference | Tol | Notes |
| -------- | ---------------: | --------: | --- | ----- |
| SCF (ideal + displaced) | `scf_converged` | must converge | — | both collinear runs |
| \(p^\mathrm{ion}\) (ideal) | \((1,1,1)\) | exact ion bookkeeping | \(1\times10^{-9}\) | \(Z_\mathrm{val}=4\), diamond sites |
| \(\Delta p^\mathrm{ion}_3\) (displaced) | \(0.08\) | \(Z_\mathrm{val}\times\delta=4\times0.02\) | \(1\times10^{-9}\) | only \(a_3\) moved |
| Centrosymmetric \(p_i\) (ideal) | distance to \(\{0,\tfrac12\}\) lattice | Spaldin *JSSC* **195**, 2 (2012) | **&lt; 0.05** | half-quantum lattice; quantum-1 branch |
| \(\|Z^*(\mathrm{Si})\|\) | \(\|\Delta p_3\|/\delta\) | Pick–Cohen–Martin *PRB* **1**, 910 (1970) | **&lt; 0.5** | elemental ASR |
| Host twin (informative, 2026-07-26) | \(\varphi=0\), \(p=0\), \(Z^*=0\) | same pins | exact to print | Release ecut 15 / k=2 |

**Why literature, not CP2K digits.** CP2K `&MOMENTS PERIODIC T` under
`LSD` / UKS documents the same modern-theory observable on
**GTH-PADE + DZVP-MOLOPT-SR-GTH**, not ONCV plane waves. Absolute Berry
phases and formal \(p_i\) branch choices are **not** comparable across
PP/basis. The regression therefore hard-codes **parameter-free symmetry
pins** shared with closed-shell `berry-polarization`. With multiplicity 1
and even \(N_e\), collinear KSV must recover the same physics modulo the
smaller nspin=2 quantum.

**CP2K offline note.** Input echo (never executed at ctest time):
`tools/cp2k_oracle/inputs/si_berry_spin_note.inp` — same diamond geometry,
PADE / LSD, multiplicity 1, `&MOMENTS PERIODIC T`. Re-pin absolute phases
only against a fair NC-UPF PW twin (e.g. QE) when available offline.

## Performance

Source: `regressions/pw/berry-spin-si/REFERENCE.md`; queue / handoff mark
this slug **`perf_exempt`**.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| Si collinear KSV (ecut 15 / k=2; 2 SCF + strings) | n/a | **32.05** (informative) | n/a | n/a (**exempt**) | No loadable CP2K/QE PW binary on pin host; GTH phases not a fair twin |

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No loadable CP2K/QE PW binary on the build host for a fair wall twin this shift (same host constraint as `berry-polarization`) |
| ATV twin wall (informative) | **32.05 s** (Release; 2 collinear SCF + string sweeps) |
| `ratio_atv_over_cp2k` | n/a (exempt) |
| `within_10x` | n/a (exempt; correctness gate only) |
| Wall budget | e2e ≤ 60 s single-rank CPU (PASS) |

When a fair NC-UPF PW oracle is available offline: time the same geometry
and string density, append \(r=T_\mathrm{ATV}/T_\mathrm{oracle}\) to
REFERENCE.md, require \(r\le 10\). Wall comparison is same-geometry only,
not absolute-phase parity with GTH Quickstep.

## Limitations

- **Tutorial ecut / k / string mesh only** — not production-converged
  polarization magnitudes or dense-string branch resolution.
- **Insulators only** — metallic smearing is a hard throw; no
  Fermi-surface Berry corrections.
- **Collinear nspin=2 only** on this slug — noncollinear / SOC out of
  scope; closed-shell is the sibling regression.
- **Multiplicity 1 Si** — open-shell magnets (unequal \(N_\uparrow,N_\downarrow\))
  exercise the same code path but are not dual-gated here.
- **No GPW Berry**, no hybrid / meta-GGA XC, no CUDA PW Berry.
- **Energy / phase not Frame-meta I/O** — result is an owned struct; public
  energy on Frame remains the Evaluate / method path in eV.
- **Perf exempt** until a loadable fair PW twin exists on the pin host.

## References

- Offline pin: `regressions/pw/berry-spin-si/REFERENCE.md`
- Regression: `regressions/pw/berry-spin-si/reg_berry-spin-si.cpp`
- Closed-shell sibling: `regressions/pw/berry-polarization/REFERENCE.md`
- CP2K oracle note: `tools/cp2k_oracle/inputs/si_berry_spin_note.inp`
- Roadmap: [dft.md](../../roadmap/dft.md) §11 Berry-phase polarization
- King-Smith & Vanderbilt, *Phys. Rev. B* **47**, 1651 (1993)
- Resta, *Rev. Mod. Phys.* **66**, 899 (1994)
- Spaldin, *J. Solid State Chem.* **195**, 2 (2012) — centrosymmetric lattice
- Pick, Cohen & Martin, *Phys. Rev. B* **1**, 910 (1970) — elemental \(Z^*=0\)
