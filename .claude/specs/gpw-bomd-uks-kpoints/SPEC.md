# Spec — gpw-bomd-uks-kpoints

## Goal

Close multi-k collinear **UKS plain Born–Oppenheimer MD** on GPW: full SCF
each step with the previous step's dual-spin complex densities
\(P_k^\alpha, P_k^\beta\) as the SCF initial guess (k-path analogue of Γ
`guess_P`/`guess_P_beta`; CP2K `EXTRAPOLATION USE_PREV_P` on the UKS +
k-mesh path). Not XL-BOMD, not CPMD, not SGCP.

## Capability surface

| Piece | Location | Status expected |
| ----- | -------- | --------------- |
| Multi-k UKS SCF + forces | `GpwDft` / `gpw_forces_*` | product (`gpw-kpoints-uks`, `gpw-kpoint-uks-forces`) |
| Plain multi-k RKS \(P_k\) warm start | `GPWEvaluator::launch` | product (`gpw-bomd-kpoints`) |
| Dual-spin multi-k warm start | same evaluator block: when `n_α+n_β>0` and `P_k_beta` full, pass both | product unit-pinned (`GpwBomd.PlainMultiKUksRelaunchWarmStartsBothSpins`) |
| E2E NVE pin vs CP2K UKS AIMD k-mesh | this slug | **this delivery** |

No new algorithm: the evaluator already seeds dual \(P_k^\sigma\). This slug
is the **matched CP2K AIMD pin + dual gate** that closes the queue row.

## Twin (CP2K 2026.1 Quickstep)

- System: periodic H₂, cubic 12 Bohr, bond 1.0 Bohr along \(x\), closed-shell
  collinear UKS (\(N_\alpha=N_\beta=1\), `LSD` + `MULTIPLICITY 1`) so both
  spin channels are non-empty and the molecule stays bound.
- Basis/PP/XC: DZV-GTH-PADE / GTH-PADE-q1 / PADE; CUTOFF 300 Ry; NGRIDS 1.
- k: MONKHORST-PACK 1×1×2, `FULL_GRID .TRUE.` (2 k, \(w=1/2\)); ATV
  `k_time_reversal=false`.
- MD: NVE, dt 0.5 fs, 4 steps; `EXTRAPOLATION USE_PREV_P`; antisymmetric
  stretch \(v_x=\mp0.001\) Å/fs.
- Input: `tools/cp2k_oracle/inputs/gpw-bomd-uks-kpoints/h2_uks_bomd_kpoints_nve.inp`

## Acceptance

1. Offline CP2K run records step-0 energy/force and per-step Pot from `.ener`.
2. `reg_gpw-bomd-uks-kpoints` asserts ATV vs those pins (same-method band as
   RKS multi-k BOMD residual) + dual \(P_k^\sigma\) payload + warm-start
   iters ≤ cold + NVE drift envelope.
3. Fair wall on the **same twin MD job** (1 MPI × OMP=8): \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}} \le 10\).
4. Unit gate `test_cpu_gpw_bomd` (existing dual-spin warm start) still green.

## Non-goals

- High-spin unbound H₂ NVE, O₂ multi-k BOMD product depth, XL/CPMD/SGCP,
  thermostat combo (`cpmd-uks-thermostat-kpoints`).
