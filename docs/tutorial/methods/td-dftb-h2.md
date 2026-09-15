# Linear-response TD-DFTB (Casida) — H₂ singlet / triplet

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `td-dftb-h2` |
| Regression | `regressions/semiempirical/td-dftb-h2/` |
| Public API | `cpu::semiempirical::DFTB2::td_casida` → `TdDftbResult` |
| Track | TB / SE method loop wave 4 () |
| Roadmap | TD-DFTB — ✓ foothold (energies only) |
| Family oracle | **DFTB+** (never OpenMOPAC / xtb) |
| Pin oracle | `literature` — **formalism only**; no absolute value pinned |

## Summary

Vertical **singlet** and **triplet** excitation energies from the
tight-binding formulation of time-dependent density-functional response
theory, on a converged restricted **SCC-DFTB2** ground state. The concrete
method `cpu::semiempirical::DFTB2` gains a `const` query method
`td_casida(frame, n_states, …)`; every Casida quantity (transition charges,
coupling kernel, Ω) is private workspace inside it. Frame in, result out —
no public `*Config` bag, no `*System`, no `make_*` / `build_*` factory.

This is the tree's house shape for linear response: `RHF::tda`, `RKS::tddft`,
`UKS::tda` and `UHF::sf_tda` are all member queries on the concrete
ground-state method that return a result-only struct
(`cpu/scf/tdhf.h`). `cpu/semiempirical/td_dftb2.h` mirrors it for the
tight-binding family.

**Energies only.** No oscillator strengths, no transition dipoles, no
excited-state gradients or non-adiabatic couplings.

## Method

Ground state: SCC-DFTB2 (Elstner et al., *Phys. Rev. B* **58**, 7260 (1998)),
restricted closed shell, Γ point, `DftbSkSet::MioSketch`.

Response: Niehaus, Suhai, Della Sala, Lugli, Elstner, Seifert, Frauenheim,
*Phys. Rev. B* **63**, 085108 (2001). Mulliken transition charges [e]

\[
q^{ia}_A = \tfrac12\sum_{\mu\in A}
\Big(c_{\mu i}\,(Sc_a)_\mu + c_{\mu a}\,(Sc_i)_\mu\Big)
\]

contract the coupling kernel [eV]

\[
K^{S}_{ia,jb} = \sum_{AB} q^{ia}_A\,\gamma_{AB}\,q^{jb}_B
\qquad\text{(singlet)},\qquad
K^{T}_{ia,jb} = \sum_{A} q^{ia}_A\,W_A\,q^{jb}_A
\qquad\text{(triplet)}
\]

— the singlet kernel is the **same** \(\gamma_{AB}\) the second-order SCC
energy uses; the triplet magnetization kernel is on-site,
\(\gamma^{m}_{AB}=\delta_{AB}W_A\) with the atom-averaged collinear spin
constant \(W_A\) [eV]. The Hermitian Casida eigenproblem [eV²]

\[
\Omega_{ia,jb} = \delta_{ij}\delta_{ab}(\varepsilon_a-\varepsilon_i)^2
+ 2\sqrt{\varepsilon_a-\varepsilon_i}\;K_{ia,jb}\;
\sqrt{\varepsilon_b-\varepsilon_j},
\qquad \Omega F = \omega^2 F
\]

is solved densely (LAPACK `dsyevd`), and \(\omega_n=\sqrt{\lambda_n}\).

`td_casida` is **stateless**: it runs its own converged SCC at the geometry
in `frame`, through the same routine `launch()` uses, so it never depends on
whether or where `launch()` last ran. Cost is one extra SCC solve.

## Implementation

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/dftb2.h"
#include "atomiverse/cpu/semiempirical/td_dftb2.h"

using namespace ATV;

cpu::Frame h2(2);                       // H2 at 0.74 A in a 20 A cubic box
h2.span(kAtom_type)[0] = h2.span(kAtom_type)[1] = 1;
h2.span(kAtom_x)[1] = 0.74;
h2.box = cpu::Box::cubic(20.0);

cpu::semiempirical::DFTB2 dftb;          // parameters are MEMBERS
dftb.scc_tol = 1e-12;
dftb.max_scc = 120;

const auto s = dftb.td_casida(h2, /*n_states=*/1);
const auto t = dftb.td_casida(h2, 1,
                              cpu::semiempirical::TdDftbChannel::triplet);

s.energy_ev[0];                          // lowest singlet omega [eV]
t.energy_ev[0];                          // lowest triplet omega [eV]
s.ks_gap_ev[0];                          // eps_a - eps_i [eV]
s.transition_charge[0];                  // q^{ia}_{H1} [e]
```

Optional arguments narrow the active window and scale the kernel:

```cpp
// HOMO -> LUMO only, coupling switched off (non-interacting limit).
const auto bare = dftb.td_casida(h2o, 1,
                                 cpu::semiempirical::TdDftbChannel::singlet,
                                 /*n_occ_window=*/1, /*n_virt_window=*/1,
                                 /*coupling_scale=*/0.0);
// bare.energy_ev[0] == bare.ks_gap_ev[0] to solver precision.
```

## Validation

Gates and measured values: `regressions/semiempirical/td-dftb-h2/REFERENCE.md`,
binary `reg_td-dftb-h2` (13 ms, single-threaded).

| Gate | Tolerance | Measured |
| ---- | --------- | -------- |
| λ = 0 ⇒ ω = ε_a − ε_i (H₂, H₂O) | 1e-10 eV | 0.000e+00 |
| Ω symmetric as assembled (H₂O, 8×8) | 1e-12 eV² | 3.553e-15 |
| Σ_A q^{ia}_A = 0 (all 8 H₂O pairs) | 1e-10 e | 2.776e-16 |
| H₂: ω_T < ε_a−ε_i < ω_S | strict | 15.010 < 15.441 < 22.418 |
| H₂: q_H1 = −q_H2, \|q\| > ½ | analytic | ±0.6515851161 |
| H₂: ω_S red-shifts on stretching | strict | 10.566 (1.20 Å) < 22.418 |

Unit gates: `tests/cpu/semiempirical/test_td_dftb2.cpp` (13 tests).

**There is no absolute excitation-energy oracle.** No `dftb+` binary exists on
the pin host, and no published TD-DFTB/mio-1-1 value for H₂ exists — DFTB+'s
own test suite and Recipes have no H₂ linear-response case, and Humeniuk &
Mitrić state the reason directly (a minimal valence basis cannot describe
H₂'s Rydberg-like excited states). `S1 = 22.42 eV` is therefore **not** a
physical excitation energy (experiment: B¹Σ⁺_u ≈ 12.3–12.7 eV vertical); it is
pinned only as a drift guard and labelled as such in the source.

## Limitations

- Restricted closed shell, molecular Γ point, `MioSketch` only — ThreeOb,
  multi-k and open shell all throw.
- Dense solver, ≤ 512 occupied→virtual pairs; no Davidson.
- No oscillator strengths / transition dipoles, no excited-state gradients.
- No long-range correction, no on-site correction — charge-transfer states are
  the known failure mode of plain TD-DFTB.
- Spin constants \(W_A\) are sketches (\(W_\mathrm{H}=-0.50\) eV vs the
  DFTB.org mio \(\approx-1.96\) eV), so triplet red shifts are ~4× too small.
- \(\gamma_{AB}\) carries a **pre-existing units defect** shared with the
  DFTB2/DFTB3 ground state (`src/cpu/semiempirical/dftb2.cpp:118`,
  `src/cpu/semiempirical/dftb3_scc.cpp:41`) that scales every singlet coupling.
  See `REFERENCE.md` → "Blocking defect found in the ground state".

## References

- T. A. Niehaus, S. Suhai, F. Della Sala, P. Lugli, M. Elstner, G. Seifert,
  Th. Frauenheim, "Tight-binding approach to time-dependent density-functional
  response theory", *Phys. Rev. B* **63**, 085108 (2001),
  DOI [10.1103/PhysRevB.63.085108](https://doi.org/10.1103/PhysRevB.63.085108).
- M. Elstner, D. Porezag, G. Jungnickel, J. Elsner, M. Haugk, Th. Frauenheim,
  S. Suhai, G. Seifert, *Phys. Rev. B* **58**, 7260 (1998),
  DOI [10.1103/PhysRevB.58.7260](https://doi.org/10.1103/PhysRevB.58.7260).
- M. E. Casida, in *Recent Advances in Density Functional Methods*, Part I
  (World Scientific, 1995).
- R. Rüger, E. van Lenthe, T. Heine, L. Visscher, *J. Chem. Phys.* **144**,
  184103 (2016), DOI [10.1063/1.4948647](https://doi.org/10.1063/1.4948647).
