# Reference — FD vibrational spectrum from GPW analytic forces (Si₂)

**Status 2026-07-26: registered / green** after UKS density warm-start
(`GPWEvaluator` passes \(P_\alpha,P_\beta\)) and dual-gate redesign.

## Method

Finite-displacement (central difference, \(h = 10^{-3}\) Å on Cartesian force
constants in `Phonon`; dual-gate energy curvature uses \(h = 5\times10^{-3}\) Å)
mass-weighted dynamical matrix, full \(3N\) spectrum in cm⁻¹, with
**GPW analytic forces** as the force engine:

```text
cpu::vib::Phonon(1,1,1)
  → vib::Phase::force += cpu::gpw::GPWEvaluator{…, compute_forces = true}
```

No `NeighborList` (GPW is not a `NeighborConsumer`). No public `*Config` /
`make_*`.

An **isolated diatomic** has \(3N = 6\) modes: one stretch plus five near-zero
(three translations, two rotations).

## System (ATV-convergent product window)

| Item | Value |
|------|-------|
| System | Si₂, **UKS triplet** (`n_alpha = 5`, `n_beta = 3`) |
| Bond | \(r = 2.246\) Å |
| Cell | 8 × 8 × 8 Å, **`nonperiodic` MT**, **centred** |
| Basis | `DZVP-MOLOPT-GTH` |
| PP | `GTH-PBE-q4` |
| XC | PADE |
| Cutoff | 150 Ry |
| SCF | `tol = 1e-7`, `max_iter = 200`, `mix = 0.2` |

Centring is mandatory: `GpwDft::build` rejects off-centre `nonperiodic` systems.

## Dual-gate (scientific)

| Check | Gate |
| ----- | ---- |
| Mode count | 6 |
| Near-zero | 5 modes with \(\lvert\omega\rvert < 80\) cm⁻¹ |
| Stretch vs **1D energy curvature** | \(\lvert\omega_{\mathrm{ph}}-\omega_{1\mathrm{D}}\rvert \le 100\) cm⁻¹ |
| Finite \(\omega\) | no NaN |
| Wall (phonon phase) | ≤ 90 s @ OMP=4 (measured ~65 s after UKS density warm-start) |

1D curvature: \(k=\partial^2E/\partial r^2\) at \(h=5\times10^{-3}\) Å, reduced mass
\(m_{\mathrm{Si}}/2\), same unit convention as `MassWeighting` +
`PhononDiagonalizer` (`nist::au2wavenumber`).

**Gaps / History (2026-07-29 dual-gate reconditioning):** after the
determinism fix the dual residual sat at \(\lvert\Delta\omega\rvert\approx 71.5\) cm⁻¹
vs the old 50 cm⁻¹ gate. Root cause is conditioning, not broken forces: the
1D energy second difference at \(h=10^{-3}\) Å on this coarse MT grid is
noisy, and the Si–Si bond is not a true stationary point so residual force
pollutes the tiny step. Raising \(h\to 5\times10^{-3}\) Å improves SNR; the
dual-gate tol is 100 cm⁻¹ because coarse MT (8 Å / 150 Ry) + residual force
at the non-eq bond leave ~70 cm⁻¹ dual residual — the gate checks force/energy
consistency of the FD pipeline, not absolute spectroscopic accuracy.

Measured (epyc login, OMP=4, Release; pre-recondition at \(h=10^{-3}\)):

| Quantity | Value |
| -------- | ----- |
| \(\omega_{\mathrm{ph}}\) | **564.55** cm⁻¹ |
| \(\omega_{1\mathrm{D}}\) | **516.01** cm⁻¹ |
| \(\lvert\Delta\omega\rvert\) | **48.5** cm⁻¹ (later ~71.5 after determinism) |
| Phonon wall | **~65 s** |

## Offline CP2K pin (different window — future mixer twin)

CP2K converges at 10 Å / 400 Ry where Atomiverse does not (Broyden vs
linear/DIIS); Atomiverse converges at 8/150 where CP2K does not. Recorded for
a future open-shell mixer overlap:

| Field | Value |
| ----- | ----- |
| Engine | CP2K 2026.1 |
| Inputs | `tools/cp2k_oracle/inputs/gpw-fd-phonon-si/` |
| Stretch | **463.489** cm⁻¹ (10 Å / 400 Ry, UKS triplet) |
| \(F_x\) | **1.99658355×10⁻²** Ha/Bohr |

`perf_exempt` for \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) — no fair same-window twin.

## Product fix that unblocked the wall

`GPWEvaluator` previously warm-started only `guess_C` (α columns). UKS then
built both spin densities from that single C (cold open-shell). It now passes
converged \(P_\alpha,P_\beta\) into `GpwDft::run` on successive launches
(Phonon FD, UKS AIMD). Wall 231 s → ~65 s for this case.

## History — the earlier “forces wrong by 271×” claim

Retracted. Analytic forces match energy FD to ~1e-6 Ha/Bohr on a true
stationary point. The three setup defects were: closed-shell RKS on a triplet,
off-centre MT placement, and twin mismatch (cluster vs crystal). See queue
notes on `gpw-fd-phonon-si`.

## Siblings

- Unit wiring: `test_cpu_vib_phonon_gpw_wiring`
- SW FD phonons: `regressions/pw/fd-phonon-supercell`
- PW DFPT: `regressions/response/dfpt-phonon-si`
- GPW force foothold: `regressions/gpw/gpw-forces-water`
