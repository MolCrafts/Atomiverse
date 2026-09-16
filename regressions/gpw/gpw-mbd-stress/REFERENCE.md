# Reference — Γ GPW analytic cell stress with periodic MBD@rsSCS

## Method

Γ-point **GPW** (PBE) on a periodic water dimer with **MBD@rsSCS** many-body
dispersion (free-atom polarizabilities, PBE \(\beta = 0.83\), Fermi steepness
\(a = 6\), MBD lattice sum on a 2×2×2 mesh).

**Product cell stress** is the Hellmann–Feynman fixed-density-matrix path:

1. One SCF at the unstrained cell → \(P\), energy-weighted \(W\).
2. Central FD of the non-self-consistent Lagrangian
   \(E_L[P;W;\varepsilon]\) under orthorhombic diagonal strain at fixed
   fractional coordinates, with the geometry-only
   \(E_{\mathrm{disp}}^{\mathrm{MBD}}(\{\mathbf R\},\mathbf h)\) **rebuilt at
   every strained cell** (same `mbd_rsscs_pbe` call as the energy path).
3. \(\sigma_{aa} = -\Omega^{-1}\,\partial E_L/\partial\varepsilon_{aa}\)
   (CP2K `STRESS_TENSOR` sign; \(E\) in Hartree, \(\Omega\) in Bohr³).

Rebuilding at each strain is the whole point: holding \(E_{\mathrm{disp}}\)
fixed would differentiate a different functional than the SCF converged — the
failure mode that shipped once as a 0.53 Ha/Bohr ZnO force error.

Public path: `cpu::gpw::GPWEvaluator` with `mbd = true`,
`mbd_k_mesh = {2,2,2}`, `compute_stress = true`, `stress_use_fd = false`
→ Frame `kStress` [eV/Å³]; compared here via `last_stress_ha_bohr3()`.

## Oracles

| Gate | Oracle |
| ---- | ------ |
| \(E_{\mathrm{disp}}\) | offline **pure-Python pymbd** `mbd_energy` — the same pin `gpw-mbd-periodic` uses, so the energy being differentiated is externally validated |
| \(\sigma\) | **multi-SCF strain-FD** (`gpw_stress_fd_impl`): full re-SCF at each strained cell, MBD folded through the energy path |
| MBD actually folded | \(\sigma(\text{mbd}) \ne \sigma(\text{plain})\) |

The stress itself has no external twin: CP2K ships no MBD@rsSCS stress class,
and pymbd exposes energies rather than a GPW cell derivative. The strain-FD
oracle is therefore the correct correctness statement for the *new* capability
("the analytic stress is the strain derivative of the functional we converged"),
with the functional pinned externally. Same structure as `gpw-d4-stress`.

## System

Same geometry as `gpw-mbd-periodic` / `gpw-d4-stress`:

| Item | Value |
|------|-------|
| System | water dimer, **6 atoms**, order **O H H O H H** |
| Cell | 12 × 12 × 12 Å, `PERIODIC XYZ` |
| Coordinates (Å) | O (0,0,0); H (0.758602, 0, ±0.504284); O (3,0,0); H (3.758602, 0, ±0.504284) |
| Basis | `DZVP-MOLOPT-SR-GTH` |
| PP | O `GTH-PBE-q6`, H `GTH-PBE-q1` → **16** valence e⁻ (`n_occ = 8`) |
| XC | PBE |
| Dispersion | MBD@rsSCS, free-atom volumes, \(\beta = 0.83\), \(a = 6\), MBD k-mesh 2×2×2 |
| Cutoff | 200 Ry |
| Strain step | \(h = 2\times10^{-3}\) |

Placement is verbatim: a periodic cell wraps by minimum image. (An isolated
MT run would have to be centred — see `gpw-d3-forces/REFERENCE.md`.)

## Pin

**\(E_{\mathrm{disp}}\) = −1.4965720704989138 × 10⁻³ Ha**

Source: offline pure-Python **pymbd** `mbd_energy` (libmbd tree), periodic,
MBD k-mesh 2×2×2, Å→Bohr as in Atomiverse. Identical value and provenance as
`regressions/gpw/gpw-mbd-periodic`; twin notes under
`tools/cp2k_oracle/inputs/gpw-mbd-periodic/`. Never launched at ctest time.

## Measured results (2026-07-26)

| Gate | Tol | Measured |
| ---- | --- | -------- |
| \(E_{\mathrm{disp}}\) vs pymbd | 1e−8 Ha | **1.85e−10** |
| \(\sigma_{\mathrm{diag}}\) analytic vs multi-SCF FD | 2e−5 Ha/Bohr³ | **7.39e−07** |
| \(\lvert\sigma\rvert\) nonzero | > 1e−10 | **5.82e−05** |
| MBD shift \(\lvert\sigma(\mathrm{mbd})-\sigma(\mathrm{plain})\rvert\) | > 1e−10 | **1.72e−07** |
| Off-diagonals (orthorhombic) | exactly 0 | **0** |
| `kStress` unit conversion | 1e−9 | exact |
| Product wall | ≤ 60 s | **23.3 s** |

Per-axis: \(\sigma_{xx} = -2.185\times10^{-5}\) (FD \(-2.259\times10^{-5}\)),
\(\sigma_{yy} = 2.007\times10^{-5}\), \(\sigma_{zz} = 5.823\times10^{-5}\)
Ha/Bohr³.

## Tolerance justification

- **\(E_{\mathrm{disp}}\) 1e−8 Ha:** deterministic rsSCS solve against an
  independent implementation of the same algorithm; measured 1.9e−10.
- **Analytic vs FD 2e−5 Ha/Bohr³:** the two differ by the fixed-\(P\)
  (frozen density-matrix) approximation plus SCF residual at `tol = 1e-6`, not
  by the dispersion treatment. Same tier as `gpw-d4-stress`; measured 7.4e−7,
  i.e. 27× inside the gate. The \(xx\) axis carries the largest residual
  because the dimer axis is \(x\), so its strain moves the O–O separation most.
- **MBD shift floor 1e−10:** the MBD stress contribution is ~1.7e−7 Ha/Bohr³
  here; the floor only has to exclude "silently not folded", which is exactly
  what the multi-k path used to do.

## Dual gate

- **Correctness:** the gates above, plus unit gates in
  `tests/cpu/gpw/test_cpu_gpw_stress.cpp` — `AnalyticMatchesFdWithMbd`
  (6.1e−9 Ha/Bohr³ on cheap MT H₂), `MbdChangesGammaDiagonal`, and
  `MbdMultiKStressThrows`.
- **Speed:** no MBD-stress oracle binary exists to time against, so this case is
  **perf_exempt**. For scale: the product analytic pass is 23.3 s versus 23.2 s
  for its own multi-SCF FD twin on this small cell — the fixed-\(P\) path's
  advantage grows with atom count, since the FD twin needs a full re-SCF per
  strain sample.

## Scope

**Not covered here:** multi-k MBD stress — shipped separately as
`regressions/gpw/gpw-mbd-kpoint-stress` (the k-summed fixed-\(P_k\)
Lagrangian folds the same geometry-only \(E_{\mathrm{disp}}\), gated by
strained-cell pymbd pins + the multi-SCF strain-FD dual gate).

## Notes

- Sibling MBD products: `gpw-mbd-ice` (isolated energy), `gpw-mbd-periodic`
  (periodic energy), `gpw-mbd-forces`, `gpw-mbd-kpoint-forces`,
  `gpw-mbd-hirshfeld`.
- Sibling dispersion stress: `gpw-d4-stress`.
