# cuda-gpw-device-payload

method=cuda | P3 | staffed Assign wave 3

## Lever

| Field | Value |
| --- | --- |
| File | `src/cuda/gpw/gpw_solver.cu` final Γ RKS payload block; `AoDiisEigenDevice` |
| Function | drop host `store_converged_state` → `diagonalize_fock`; use device eigen of final `d_H_work` + boundary D2H C/eps/P/H |
| Metric | zero host re-diag of KS matrix after mid-SCF device `dsyevd`; energy dual-gate \|dE\| vs CPU |

## Defect

After mid-SCF `AoDiisEigenDevice` already runs device `dsyevd`, final
`store_converged_state` still host-rediags `H0 + V` (`diagonalize_fock`). That is
payload-boundary waste and a scientific H densify path that belongs on device.

## Acceptance

1. GH200 dual-gate PASS: same-settings CPU Γ RKS H₂ energy, \|dE\| ≤ 5e-6 Ha
2. Diff greps: no mid-path scientific matrix DH; final C/eps/H/P D2H only at
   post-convergence boundary
3. Unit: `eigen_from_device` matches host `diagonalize_fock` on toy H (eig + densified P)
4. Skills named in worker summary (av-perf residency; reuse existing syevd/gemm)
5. **perf_exempt** small-cell architectural residency fix

## Out of scope

UKS payload (`uks-payload-device`), multi-k payload (`kpoint-payload-device`),
cold-start densify (`device-cold-start`), force AO Pulay, stress.

## Twin

oracle: cpu-twin — `cpu::gpw::GpwDft` H₂ Γ RKS PADE energy (same settings as
`cuda-gpw-device-diis-eigen`).
