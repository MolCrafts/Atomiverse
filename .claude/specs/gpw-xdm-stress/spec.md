---
slug: gpw-xdm-stress
status: done
created: 2026-07-31
revised: 2026-07-31
domain: gpw / cell stress / dispersion
queue: cp2k-method-loop
deferred: false
depends_on: []
unblocks: [gpw-xdm-stress]
---

# Spec: `gpw-xdm-stress` — Γ GPW analytic stress folding free-atom XDM, dedicated pins

**Slug:** `gpw-xdm-stress`
**Domain:** GPW cell stress (CPU), Γ fixed-\(P\) Lagrangian
**Queue row:** `gpw-xdm-stress` (GapCensus wave 15, assigned wave 2)

## Summary

The Γ analytic fixed-\(P\) stress Lagrangian (`energy_fixed_P`,
`src/cpu/gpw/internal/gpw_stress.cpp:863`) and the multi-k Lagrangian
(`:520`) already rebuild the geometry-only free-atom **XDM**
\(E_{\mathrm{disp}}\) at every strained cell — the energy **folds**, but no
dedicated pin, regression, or unit gate exists for the XDM stress
contribution. Wave-15 thin **pin-only** sibling of `gpw-ts-stress`: no
solver change; deliver strained-cell dual pure-Python pins + the
`gpw_stress_fd` dual gate + the missing sibling-parity unit gates.

## Physics

Free-atom XDM (Becke–Johnson combination rules, QE NC-PBE BJ damping
a1=0.4283, a2=2.4690 Å, rcut 50 Å real-space lattice sum) is
**geometry-only**: \(E_{\mathrm{disp}}(\{Z\},\{\mathbf R\},\mathbf h)\) does
not depend on the SCF density. Under the fixed-\(P\) Hellmann–Feynman cell
theorem its contribution to \(\sigma_{aa}\) is exactly
\(-\Omega_0^{-1}\,\Delta E_{\mathrm{disp}}/(2h)\) with
\(E_{\mathrm{disp}}\) rebuilt at each strained cell (positions at fixed
fractional coordinates, cell axis scaled, rcut fixed at 50 Å). Holding
\(E_{\mathrm{disp}}\) fixed would differentiate a different functional than
the SCF converged (the ZnO 0.53 Ha/Bohr failure mode).

## Change surface

**No solver change.** `GpwDft::xdm` + `gpw_stress` / `gpw_stress_fd` /
`GPWEvaluator{compute_stress}` already exist and fold XDM. Additions only:

1. `tools/cp2k_oracle/inputs/gpw-xdm-stress/pin_xdm_strain.py` + note —
   offline dual pure-Python free-atom XDM lattice sum (independent
   implementation: α from CRC 88th Ed., M1–M3 re-inverted from the cited
   literature homoatomic C6/C8/C10, never copied from ATV headers), printing
   E_disp unstrained + at ±h per axis and the per-axis σ_XDM FD terms.
2. `regressions/gpw/gpw-xdm-stress/` (REFERENCE.md + reg) + CMake row.
3. `tests/cpu/gpw/test_cpu_gpw_stress.cpp`: `AnalyticMatchesFdWithXdm` +
   `XdmChangesGammaDiagonal` (cheap MT H₂ fixture — the exact sibling gates
   D4/D3BJ/MBD already have; XDM is the one dispersion fold without them).

Coordination: `gpw_stress.cpp` is **not edited**, so the in-flight
`wf/gpw-d3-stress` worker cannot conflict.

## Oracle

Same family as the shipped `gpw-xdm-forces` / `gpw-xdm-ice` pins: CP2K
2026.1 stock `PAIR_POTENTIAL TYPE` has no `TYPE XDM`
(only DFTD2/D3/D3BJ/D4), so the offline oracle is the **dual pure-Python
free-atom XDM reference** (precedented; full density-dependent XDM oracles —
postg/critic2/QE — stay out of scope with the free-atom foothold).
A pin is a number the python script printed; never seeded from ATV.
Provenance cross-check at pin time: the unstrained value must reproduce the
shipped `gpw-xdm-ice` periodic pin −2.1148157612879029e−3 Ha.

## Regression (e2e) — `regressions/gpw/gpw-xdm-stress/`

Periodic water dimer, 12 Å cube (the `gpw-xdm-ice` / `gpw-mbd-stress` twin
geometry), PBE, DZVP-MOLOPT-SR-GTH, GTH-PBE-q1/q6, Γ, `xdm = true`,
`compute_stress = true`, h = 2e-3. Gates:

| Gate | Oracle | Tol |
| --- | --- | --- |
| A. \(E_{\mathrm{disp}}\) unstrained (product SCF payload via Frame) | dual pure-Python pin (≡ shipped `gpw-xdm-ice` pin) | 1e-8 Ha (Frame Å→Bohr vs pin constant; sibling ice/forces tier) |
| B. \(E_{\mathrm{disp}}\) **under strain** (±h per axis, fixed fractional coords, rcut 50 Å) | fresh dual pure-Python pins at each of the 6 strained cells | 1e-12 Ha (direct `xdm_pbe` + pin Å→Bohr) |
| C. analytic fixed-\(P\) σ vs multi-SCF strain-FD (`gpw_stress_fd_impl`) | internal dual gate | 2e-5 Ha/Bohr³ (Γ dispersion-stress reg tier) |
| D. \(\sigma(\mathrm{xdm}) − \sigma(\mathrm{plain})\) per axis equals the pinned XDM strain-FD term \(-\Delta E_{\mathrm{disp}}/(2h\Omega_0)\) | pins from gate B | 1e-8 Ha/Bohr³ |
| E. off-diagonals exactly 0; `kStress` unit conversion 1e-9; wall ≤ 60 s | structural | — |

Gate B is checked on the same `xdm_pbe` call + rcut the stress Lagrangian
uses (strain-pin parity, 1e-12 Ha class — deterministic geometry-only sum).

## Perf

`perf_exempt` — no XDM-capable oracle binary exists to time against (CP2K
stock has no XDM; same policy as `gpw-xdm-ice` / `gpw-xdm-forces` /
`gpw-mbd-stress`). Product wall gated ≤ 60 s.

## Acceptance

- Pin script committed + note; unstrained value bit-reproduces the shipped
  `gpw-xdm-ice` pin (provenance cross-check recorded in REFERENCE.md).
- All regression gates green; sibling unit gates green
  (`test_cpu_gpw_stress`); no edit to `gpw_stress.cpp`.
- Scoped build + test only; builds wiped on exit.
