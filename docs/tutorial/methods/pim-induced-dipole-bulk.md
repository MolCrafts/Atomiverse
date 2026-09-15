# Bulk PIM self-consistent induced dipoles — two-site ion pair (E4)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `pim-induced-dipole-bulk` |
| Regression | `regressions/md/pim-induced-dipole-bulk/` |
| Public API | `cpu::PimInducedDipole` (`Potential`; typically on `cpu::Evaluate` or MD `force`) |
| Roadmap | map slice E4 |
| Oracle | **MetalWalls** bulk PIM multipole form (offline analytic pin; not LAMMPS) |

## Summary

Classical **Polarizable Ion Model (PIM)** for bulk electrolytes / molten salts:
each atom type carries an isotropic polarizability \(\alpha\) [Å³]. Induced
dipoles \(\boldsymbol{\mu}\) are solved self-consistently from the permanent
field of Frame point charges plus mutual induction through a
**Thole**-damped (optionally **Tang–Toennies**-damped) dipole tensor.
Polarization energy \(E_{\mathrm{pol}}\) [eV] and Hellmann–Feynman forces
(charge–dipole + dipole–dipole) are written to Frame. Permanent charge–charge
Coulomb is **not** included — pair it with `PairCoulCut` / PME separately.
Public surface is `cpu::PimInducedDipole` — a **`Potential` Component**, not a
Driver category and not inside `KSpace*`. This regression is the **E4
foothold**: a two-site charged ion pair with offline Applequist/Thole pins in
the MetalWalls bulk-PIM form. **Oracle is MetalWalls** (MW-only classical PIM
SCC); LAMMPS polarizable paths (Drude / permanent multipoles) are not this
physics. Fair wall vs MetalWalls is **`perf_exempt`** (no MW binary on the pin
host).

## Method

**Physics.** Applequist isotropic induced-dipole SCC. For sites \(i\):

\[
\boldsymbol{\mu}_i=\alpha_i\mathbf{E}_i^{\mathrm{tot}},\quad
\mathbf{E}_i^{\mathrm{tot}}
  =\mathbf{E}_i^{\mathrm{perm}}
  +\sum_{j\neq i}\mathbf{T}_{ij}\cdot\boldsymbol{\mu}_j
\]

\[
E_{\mathrm{pol}}
  =-\tfrac12\sum_i\boldsymbol{\mu}_i\cdot\mathbf{E}_i^{\mathrm{perm}}
  \quad\text{[eV via `units::coulomb`]}
\]

Permanent field \(\mathbf{E}^{\mathrm{perm}}\) is the Coulomb field of Frame
point charges (min-image, real-space cutoff). Mutual induction uses the dipole
interaction tensor with **Thole** exponential screening
(\(u = a\,r/(\alpha_i\alpha_j)^{1/6}\), factors \(\lambda_3,\lambda_5,\lambda_7\);
Thole, *Chem. Phys.* **59**, 341 (1981)). Optional **Tang–Toennies** damping
\(f_n(br)\) multiplies the same kernels when `tt_b > 0`. Forces are
Hellmann–Feynman at converged SCC (charge–dipole + dipole–dipole); permanent
charge–charge is excluded so the component stays the polarization piece only.

This is the **MetalWalls bulk PIM** electrostatic polarization path (capability
map **E4**). LAMMPS is **not** the oracle: its polarizable stack is Drude /
permanent multipoles, not this SCF bulk PIM.

**What this pin includes**

| Item | Setting |
| ---- | ------- |
| Geometry | 2 sites: A at origin, B at \((3,0,0)\) Å |
| Charges | \(q_A=+1\,e\), \(q_B=-1\,e\) |
| Types / \(\alpha\) | type 1 → \(\alpha=1.0\) Å³; type 2 → \(\alpha=1.5\) Å³ (`alpha_by_type={0,1.0,1.5}`) |
| Box | cubic \(L=50\) Å (min-image ≡ free space at this spacing) |
| Cutoff | 20 Å |
| Thole \(a\) | 0.39 |
| Tang–Toennies \(b\) | 0 (off) |
| SCC | `pol_tol = 10^{-12}` e·Å, `pol_max_iter = 200`; pin uses 8 iterations |
| Binding | `cpu::Evaluate` + `add_ref` at `evaluate::Phase::evaluate` |

**Key exclusions (non-goals of this slug)**

- Permanent charge–charge Coulomb (`PairCoulCut` / PME).
- Electrode coupling / joint dipole⊕\(Q\) solve (queue **E5**
  `dipole-electrode-joint`).
- Neighbor-list / production \(O(N)\) pair path (v1 is all-pairs \(O(N^2)\)
  foothold).
- LAMMPS Drude or permanent multipole polarizable models as oracle.
- Live MetalWalls binary inside ctest (offline pin only).
- Public `*Config` bags or `make_*` / `build_*` factories.

## Implementation

Public surface only (`cpu::Frame` + `PimInducedDipole` ctor args / members; no
Config bag). Minimal Evaluate path matching the regression:

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/potentials/pim_induced_dipole.h"

using namespace ATV;

cpu::Frame frame(2);
frame.span(kAtom_x)[0] = 0.0;
frame.span(kAtom_y)[0] = 0.0;
frame.span(kAtom_z)[0] = 0.0;
frame.span(kAtom_x)[1] = 3.0;
frame.span(kAtom_y)[1] = 0.0;
frame.span(kAtom_z)[1] = 0.0;
frame.span(kAtom_charge)[0] = 1.0;
frame.span(kAtom_charge)[1] = -1.0;
frame.span(kAtom_type)[0] = 1;
frame.span(kAtom_type)[1] = 2;
frame.span(kAtom_mass)[0] = frame.span(kAtom_mass)[1] = 1.0;
frame.span(kAtom_tag)[0] = 1;
frame.span(kAtom_tag)[1] = 2;
// zero forces / energy before launch
frame.box = cpu::Box::cubic(50.0);
frame.meta.get(kEnergy) = 0;

cpu::PimInducedDipole pim(/*cutoff [Å]=*/20.0,
                          /*alpha_by_type=*/{0.0, 1.0, 1.5},
                          /*thole_a=*/0.39,
                          /*pol_tol [e·Å]=*/1e-12,
                          /*pol_max_iter=*/200);
// optional: pim.tt_b = 0.0;  // Tang–Toennies off by default

cpu::Evaluate eval;
eval.on(evaluate::Phase::evaluate).add_ref(pim);
eval.build();
eval.run(frame);

const double E = frame.meta.get(kEnergy);           // [eV]
const double e_pol = pim.last_pol_energy();         // [eV]
const int iters = pim.last_pol_iters();
const bool ok = pim.last_pol_converged();
// induced dipoles: pim.last_mu() packed xyz [e·Å]
// forces: frame.span(kAtom_fx/fy/fz)
```

MD-style composition (polarization + permanent Coulomb as separate
potentials):

```cpp
cpu::md::MD sim;
sim.set_evaluator(cpu::PimInducedDipole{20.0, {0.0, 1.0, 1.5}, 0.39});
sim.set_evaluator(cpu::PairCoulCut{/*cutoff=*/20.0});  // permanent q–q
sim.set_neighbor(cpu::md::NeighborList<>{20.0, 0.3});
sim.set_integrator(cpu::md::VelocityVerlet{/*dt [fs]=*/1.0});
sim.build();
sim.run(frame, /*nsteps=*/1);
```

Architecture rules:

- **Driver stays `cpu::Evaluate` or `cpu::md::MD`.** PIM is a Component
  (`Potential`); never a new Driver category.
- Parameters are **constructor arguments and public members** — no
  `PimInducedDipoleConfig`.
- Frame is the sole I/O currency; \(E_{\mathrm{pol}}\) lands on
  `kEnergy`, forces on `kAtom_f*`; diagnostics via `last_pol_*` / `last_mu()`.

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
  cmake --build build --target reg_pim-induced-dipole-bulk -j"$(nproc)"
  ./build/regressions/reg_pim-induced-dipole-bulk
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/potentials/pim_induced_dipole.h` | Public `PimInducedDipole` |
| `src/cpu/potentials/pim_induced_dipole.cpp` | SCC, Thole/TT, \(E_{\mathrm{pol}}\), HF forces |
| `include/atomiverse/cpu/evaluate/evaluate.h` | Host preset `cpu::Evaluate` |
| `regressions/md/pim-induced-dipole-bulk/reg_pim-induced-dipole-bulk.cpp` | Product gate |
| `tests/cpu/potentials/test_pim_induced_dipole.cpp` | Unit: undamped closed form, FD forces, Thole |

## Validation (vs MetalWalls-form offline pin)

Source:
[`regressions/md/pim-induced-dipole-bulk/REFERENCE.md`](../../../regressions/md/pim-induced-dipole-bulk/REFERENCE.md)
(offline pin; **never** live MetalWalls in ctest). Oracle tooling notes:
[`tools/metalwalls_oracle/README.md`](../../../tools/metalwalls_oracle/README.md).

**Oracle:** MetalWalls JOSS **5**, 2373 (2020) bulk PIM multipole form; Thole
*Chem. Phys.* **59**, 341 (1981) exponential damping; **offline analytic pin**
of that formula (no MetalWalls binary on the agent host).

**Why MetalWalls (not LAMMPS):** classical SCF bulk PIM is MW-only in this
loop’s oracle priority. LAMMPS polarizable paths are Drude / permanent
multipoles — different physics; they are not used as the pin for E4.

### Accuracy

| Quantity | Atomiverse / pin | Oracle pin | Tol | Notes |
| -------- | ---------------: | ---------: | --- | ----- |
| \(E_{\mathrm{pol}}\) [eV] | **−0.12040946414707571** | same hard pin | \(\lvert\Delta E\rvert < 10^{-9}\) | dual: `kEnergy` and `last_pol_energy()` |
| \(F_{A,x}\) [eV/Å] | **+0.040341072868572643** | same | \(\lvert\Delta F_x\rvert < 10^{-8}\) | action–reaction with \(B\) |
| \(F_{B,x}\) [eV/Å] | **−0.040341072868572643** | same | \(\lvert\Delta F_x\rvert < 10^{-8}\) | \(F_{A,x}+F_{B,x}=0\) |
| \(F_{*,y}\), \(F_{*,z}\) [eV/Å] | 0 | 0 | \(\lvert F\rvert < 10^{-12}\) | collinear geometry |
| \(\mu_{A,x}\) [e·Å] | 0.082882545215708481 | REFERENCE | diagnostic | not a hard ctest assert in reg main |
| \(\mu_{B,x}\) [e·Å] | 0.12342697961661356 | REFERENCE | diagnostic | packed in `last_mu()` |
| SCC iterations | 8 | — | converged | `pol_tol = 10^{-12}` e·Å |
| \(E_{\mathrm{pol}} < 0\) | true | expected for \(\alpha\ge 0\) | exact | polarization stabilizes |

**Dual gate:** energy **and** forces. Unit tests additionally check that FD of
\(E_{\mathrm{pol}}\) matches analytic \(F\) within \(5\times 10^{-6}\) eV/Å,
and undamped two-site SCC matches the closed-form Applequist solution.

Geometry, \(\alpha\), Thole \(a\), cutoffs, and tolerances: **Method** table
and REFERENCE (match `reg_pim-induced-dipole-bulk`).

Related unit surface (not the e2e pin):
`tests/cpu/potentials/test_pim_induced_dipole.cpp`.

## Performance

Source: REFERENCE.md; handoff
the case `REFERENCE.md`
(wave 2, 2026-07-24).

| Twin | \(T_{\mathrm{MetalWalls}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------------: | -----------------------: | ----: | ---- | ------ |
| bulk PIM two-site / multipole wall | — | — | **perf_exempt** | n/a | No MetalWalls binary on PATH for a fair bulk-PIM wall twin; pin is published multipole + Thole formula |

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| `perf_exempt_reason` | No MetalWalls binary on PATH for a fair bulk-PIM wall twin; pin is the published multipole + Thole formula (same path as gaussian-electrode-charge literature pins) |
| `ratio_atv_over_oracle` | **n/a** |
| `within_10x` | exempt |

Do **not** invent wall-clock ratios. When a MetalWalls twin binary is
available, record \(T_{\mathrm{atv}}/T_{\mathrm{MW}}\) in REFERENCE and require
\(\le 10\).

## Limitations

- **Bulk only.** No electrode charge solve and no joint dipole⊕electrode
  coupling (E5 `dipole-electrode-joint`, unblocked after this slug).
- **No permanent \(q\)–\(q\).** Callers must add Coulomb pair / k-space
  separately; double-counting is a user composition error.
- **All-pairs \(O(N^2)\) foothold.** NeighborConsumer path and production
  scaling deferred.
- **Isotropic \(\alpha\) by type.** No anisotropic polarizability, no Drude
  oscillators, no permanent multipoles on sites beyond Frame charges.
- **No fair speed ratio** until a MetalWalls bulk-PIM twin is on the pin host.
- **Scope of E4 wave only** for this page: Fumi–Tosi / Born / Steele and
  electrode stack siblings have their own pages when written.

## References

- Offline pin: [`regressions/md/pim-induced-dipole-bulk/REFERENCE.md`](../../../regressions/md/pim-induced-dipole-bulk/REFERENCE.md)
- Product gate: [`regressions/md/pim-induced-dipole-bulk/reg_pim-induced-dipole-bulk.cpp`](../../../regressions/md/pim-induced-dipole-bulk/reg_pim-induced-dipole-bulk.cpp)
- Public header: [`include/atomiverse/cpu/potentials/pim_induced_dipole.h`](../../../include/atomiverse/cpu/potentials/pim_induced_dipole.h)
- Oracle notes: [`tools/metalwalls_oracle/README.md`](../../../tools/metalwalls_oracle/README.md)
- Literature: Thole, *Chem. Phys.* **59**, 341 (1981); Applequist isotropic
  induced-dipole SCC; Marin-Laflèche et al., *J. Open Source Softw.* **5**,
  2373 (2020) (MetalWalls classical PIM)
