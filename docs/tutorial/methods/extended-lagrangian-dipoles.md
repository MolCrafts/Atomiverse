# Extended-Lagrangian PIM dipole dynamics (iEL/SCF) — bulk polarizable ions

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `extended-lagrangian-dipoles` |
| Regression | `regressions/md/extended-lagrangian-dipoles/` |
| Public API | `cpu::PimXlDipole` (`Potential`; on `cpu::md::MD` at `force`, or `cpu::Evaluate`) |
| Roadmap / queue | `docs/developer/metalwalls-method-queue.md` wave 5; map §2 "XL dipoles" ~ → ✓ |
| Oracle | **MetalWalls c8ad175d** for stationary PIM (`thole_a=0`); XL dynamics = scheme self-check (Albaugh 2015 / Niklasson 2009) |

## Summary

Cold-starting the induced-dipole self-consistent cycle (SCC) from
\(\boldsymbol{\mu}=0\) every MD step wastes iterations, and a *truncated*
SCC pumps energy into the trajectory because the converged dipole always
lags the nuclei. `cpu::PimXlDipole` is the **iEL/SCF extended-Lagrangian**
cure (Albaugh, Demerdash & Head-Gordon 2015) — the classical-polarization
analogue of XL-BOMD: an auxiliary dipole set \(\boldsymbol{\mu}^{aux}\) gets
its own time-reversible dynamics via the dissipative **Niklasson recursion**
and serves as the SCC warm start each step. Energy and forces written to
Frame are always the owned `PimInducedDipole`'s SCC-converged
Hellmann–Feynman values — the scheme changes the *guess dynamics*, never the
physics, so the fixed point at a stationary geometry **is** the plain SCF
PIM answer. Still a **Component** on `cpu::md::MD` — no new Driver category.
MetalWalls' own mass-zero route is MaZe (SHAKE-like constrained
propagation); that variant is deferred exactly as `ConstPotMaZe` defers it
for electrode charges.

## Method

The auxiliary dipoles are propagated by the time-reversible dissipative
Niklasson recursion (Table-I coefficients, default order K = 5):

\[
\mu^{aux}_{n+1} = 2\mu^{aux}_n - \mu^{aux}_{n-1}
  + \kappa\,(\mu^{SCC}_n - \mu^{aux}_n)
  + \alpha \sum_{k=0}^{K} c_k\, \mu^{aux}_{n-k}
\]

Each MD step: (1) the SCC of the owned `PimInducedDipole` (Thole-damped
bulk PIM, see the [`pim-induced-dipole-bulk`](pim-induced-dipole-bulk.md)
page) is warm-started from \(\boldsymbol{\mu}^{aux}\) and converged to
`pol_tol`; (2) \(E_{\mathrm{pol}}\) [eV] and Hellmann–Feynman forces land on
Frame; (3) the auxiliary dynamics advances with the converged
\(\boldsymbol{\mu}^{SCC}\). Implementation composes two shipped pieces:
`cpu::PimInducedDipole` (SCC + forces) and `cpu::electronic::XlBomd`
(Niklasson recursion) acting element-wise on the nlocal×3 dipole array.

**What is not included:** permanent charge–charge Coulomb (pair it with
`PairCoulCut` / PME separately, as for the plain PIM component); the true
MaZe mass-zero constrained propagation (deferred); Drude oscillators /
permanent multipoles; anisotropic \(\alpha\).

**Defining iEL/SCF properties** (the regression's dynamics gates): the XL
trajectory tracks the Born–Oppenheimer (cold-SCF) dipole surface, the warm
starts are strictly cheaper than cold SCC, and the total energy
\(\mathrm{KE} + E_{\mathrm{pol}}\) is conserved as well as the SCF twin's.

## Implementation

Public surface only — constructor arguments and members, no `*Config` bag:
`cpu::PimXlDipole{cutoff [Å], alpha_by_type [Å³], thole_a, xl_order,
pol_tol [e·Å], pol_max_iter}`. Mutate the owned SCC solver through
`solver()` (`thole_a`, `tt_b`, `pol_tol`, …); `reset()` drops the auxiliary
history at a geometry discontinuity (atom-count changes reset
automatically).

```cpp
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/potentials/pim_xl_dipole.h"

using namespace ATV;

// Bulk polarizable ions: ±1 e Frame charges, α by type [Å³]
cpu::PimXlDipole xl(/*cutoff [Å]=*/10.0,
                    /*alpha_by_type=*/{0.0, 1.0, 1.5},
                    /*thole_a=*/0.39,
                    /*xl_order=*/5,          // Niklasson K ∈ [3, 9]
                    /*pol_tol [e·Å]=*/1e-12,
                    /*pol_max_iter=*/200);

cpu::md::MD sim;
sim.set_integrator(cpu::md::VelocityVerlet{/*dt [fs]=*/0.5});
sim.add_ref(xl);                       // Potential — default force binding
// permanent q–q Coulomb is NOT included: add PairCoulCut / PME separately
sim.build();
sim.run(frame, 150);

xl.last_pol_energy();    // E_pol [eV] (also on frame.meta.get(kEnergy))
xl.last_pol_iters();     // SCC iterations — small once warm
xl.last_warm();          // false only on the first launch / after reset()
xl.last_aux_error();     // max|μ_aux − μ_scc| [e·Å] — distance to BO surface
xl.last_mu();            // converged dipoles, packed xyz [e·Å]
```

The regression itself uses a specialized out-of-library NVE harness (not the
product `MD` driver) so it can run the XL twin and the cold-SCF
Born–Oppenheimer twin (`PimInducedDipole`) on bitwise-identical
trajectories and difference them step by step. User code still binds
`PimXlDipole` on `cpu::md::MD`; Frame remains the only I/O token.

Build and run the regression (MD-only configure; CPU login pattern from
`AGENTS.md` / metalwalls loop):

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT="$OPENBLASDIR"
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_MD=ON -DATV_BUILD_KSPACE=ON \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_REGRESSIONS=ON \
        -DATV_BUILD_HF=OFF -DATV_BUILD_PW=OFF -DATV_BUILD_GPW=OFF \
        -DATV_BUILD_CUDA=OFF -DBLA_VENDOR=OpenBLAS
  cmake --build build --target reg_extended-lagrangian-dipoles -j"$(nproc)"
  ./build/regressions/reg_extended-lagrangian-dipoles
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/potentials/pim_xl_dipole.h` | Public `PimXlDipole` |
| `include/atomiverse/cpu/potentials/pim_induced_dipole.h` | Owned SCC solver (`solver()`) |
| `include/atomiverse/cpu/electronic/xl_bomd.h` | Niklasson propagator (`propagator()`) |
| `regressions/md/extended-lagrangian-dipoles/reg_extended-lagrangian-dipoles.cpp` | Product gate |
| `tests/cpu/potentials/test_pim_xl_dipole.cpp` | Unit surface |

## Validation (vs MetalWalls-form offline pins)

Source:
[`regressions/md/extended-lagrangian-dipoles/REFERENCE.md`](../../../regressions/md/extended-lagrangian-dipoles/REFERENCE.md)
(offline literature-form pins; **never** live MetalWalls in ctest). An
independent closed-form re-derivation (no Atomiverse code) lives at
`tools/metalwalls_oracle/inputs/metalwalls/extended-lagrangian-dipoles/pin_two_site.py`
and agrees to ~1e-14 relative.

**Oracle:** MetalWalls JOSS **5**, 2373 (2020) — capability map §2
"XL dipoles"; pinned in literature form (iEL/SCF: Albaugh et al. 2015;
recursion coefficients: Niklasson et al. 2009 Table I). No MW binary on the
agent host, and LAMMPS has no iEL/SCF induced-dipole twin (its polarizable
paths are Drude/QEq).

### Accuracy

**Pin 1 — stationary two-site fixed point** (geometry of
`pim-induced-dipole-bulk`: q = ±1 e, α = 1.0/1.5 Å³, r = 3 Å,
thole_a = 0.39, L = 50 Å). The XL fixed point at a stationary geometry IS
the SCF solution, cold and warm launches alike:

| Quantity | Atomiverse / pin | Oracle pin | Tol | Notes |
| -------- | ---------------: | ---------: | --- | ----- |
| \(E_{\mathrm{pol}}\) [eV] | **−0.12040946414707571** | same hard pin (closed-form Applequist/Thole) | \(10^{-9}\) | cold **and** warm launches |
| \(F_{A,x}\) [eV/Å] | **+0.040341072868572643** | same | \(10^{-8}\) | action–reaction with B |
| \(F_{B,x}\) [eV/Å] | **−0.040341072868572643** | same | \(10^{-8}\) | \(F_{A,x}+F_{B,x}=0\) |
| warm SCC iterations | < cold, and ≤ 2 | iEL/SCF property | exact | `last_warm()` flagged |
| warm aux deviation [e·Å] | < 1e-10 | 0 at a fixed point | — | `last_aux_error()` |

**Pin 2 — dynamics gates** (8-ion perturbed 2×2×2 rocksalt-parity toy,
L = 20 Å, 150 × 0.5 fs velocity-Verlet from rest, cutoff 10 Å, SCC tol
1e-12 in both twins; \(E_{tot} = \mathrm{KE} + E_{\mathrm{pol}}\) [eV]):

| Gate | Bound | Notes |
| ---- | ----- | ----- |
| \(\max_t \max_i \lVert\mu^{XL}_i(t) - \mu^{SCF}_i(t)\rVert\) | < 1e-6 e·Å | XL tracks the BO dipole surface |
| total warm SCC iterations | < cold-twin total (strict) | the point of iEL/SCF |
| \(\lvert E_{tot}(t_{end}) - E_{tot}(0)\rvert\), XL twin | < 5e-6 eV | no secular pumping |
| \(\lvert E_{tot}(t_{end}) - E_{tot}(0)\rvert\), SCF twin | < 5e-6 eV | control |

Both twins start from the bitwise-identical state
(\(\lvert E_0^{XL} - E_0^{SCF}\rvert < 10^{-10}\) eV in the gate).

Related unit surface (not the e2e pin):
`tests/cpu/potentials/test_pim_xl_dipole.cpp`.

## Performance

Source: REFERENCE.md; handoff
[`docs/developer/metalwalls-method-handoff.md`](../../developer/metalwalls-method-handoff.md)
(wave 5, 2026-07-27).

| Twin | \(T_{\mathrm{MetalWalls}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------------: | -----------------------: | ----: | ---- | ------ |
| iEL/SCF bulk-PIM wall | — | — | **perf_exempt** | n/a | No MetalWalls binary on PATH for a fair XL/mass-zero dipole wall twin; LAMMPS has no iEL/SCF induced-dipole twin (Drude/QEq only) |

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| `perf_exempt_reason` | No MetalWalls binary on PATH for a fair XL/mass-zero dipole wall twin, and LAMMPS has no iEL/SCF induced-dipole twin (its polarizable paths are Drude/QEq). Same pin path as `pim-induced-dipole-bulk`. |
| `ratio_atv_over_oracle` | **n/a** |
| `within_10x` | exempt |

Do **not** invent wall-clock ratios. When a MetalWalls twin binary is
available, record \(T_{\mathrm{atv}}/T_{\mathrm{MW}}\) in REFERENCE and
require \(\le 10\). The *algorithmic* speed claim that is gated today: total
warm SCC iterations strictly below the cold-SCF twin's on the same
trajectory (Pin 2).

## Limitations

- **Guess dynamics only.** Forces/energy are always SCC-converged
  Hellmann–Feynman; the truncated-SCC ("SCF-less") operating mode of
  iEL/0-SCF is not exposed.
- **Not MaZe.** MetalWalls' mass-zero SHAKE-constrained dipole propagation
  is deferred (same status as `ConstPotMaZe`'s dipole analogue); this is the
  Niklasson-recursion warm-start scheme.
- **No permanent \(q\)–\(q\).** Callers add Coulomb pair / k-space
  separately; double-counting is a user composition error.
- **Inherits the bulk-PIM foothold scope:** all-pairs \(O(N^2)\), isotropic
  \(\alpha\) by type, real-space cutoff (no polarization Ewald).
- **No fair speed ratio** until a MetalWalls twin binary is on the pin host.

## References

- Offline pin: [`regressions/md/extended-lagrangian-dipoles/REFERENCE.md`](../../../regressions/md/extended-lagrangian-dipoles/REFERENCE.md)
- Product gate: [`regressions/md/extended-lagrangian-dipoles/reg_extended-lagrangian-dipoles.cpp`](../../../regressions/md/extended-lagrangian-dipoles/reg_extended-lagrangian-dipoles.cpp)
- Public header: [`include/atomiverse/cpu/potentials/pim_xl_dipole.h`](../../../include/atomiverse/cpu/potentials/pim_xl_dipole.h)
- Queue / handoff: [`docs/developer/metalwalls-method-queue.md`](../../developer/metalwalls-method-queue.md), [`docs/developer/metalwalls-method-handoff.md`](../../developer/metalwalls-method-handoff.md), [`docs/developer/metalwalls-method-loop.md`](../../developer/metalwalls-method-loop.md)
- Sibling pages: [`pim-induced-dipole-bulk`](pim-induced-dipole-bulk.md) (owned SCC), [`maze-symplectic-charge-dynamics`](maze-symplectic-charge-dynamics.md) (the MaZe route, for electrode charges)
- Literature: Albaugh, Demerdash & Head-Gordon, *J. Chem. Phys.* **143**,
  174104 (2015) — iEL/SCF; Niklasson et al., *J. Chem. Phys.* **130**,
  214109 (2009), Table I — dissipative recursion coefficients; Niklasson,
  *Phys. Rev. Lett.* **100**, 123004 (2008) — time-reversible XL; Coretti,
  Bonella & Ciccotti, *J. Chem. Phys.* **149**, 191102 (2018) — MaZe for
  polarizable models (MW's route); Thole, *Chem. Phys.* **59**, 341 (1981);
  Applequist, Carl & Fung, *J. Am. Chem. Soc.* **94**, 2952 (1972) — damped
  point-dipole SCC; Marin-Laflèche et al., *J. Open Source Softw.* **5**,
  2373 (2020) — MetalWalls
