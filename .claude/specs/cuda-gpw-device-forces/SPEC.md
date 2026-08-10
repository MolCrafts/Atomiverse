# cuda-gpw-device-forces

## Lever

| Field | Value |
| --- | --- |
| File | `src/cuda/gpw/gpw_solver.cu` (`gpw_forces_from_scf` / `gpw_forces_analytic`) + `cuda/gpw/internal/gpw_forces.{cuh,cu}` |
| Function | Free-fn forces: host AO Pulay + device grid HF product (replaces CPU `gpw_forces_from_scf`) |
| Metric | max\|F_cuda − F_cpu\| [Ha/Bohr] on H₂ Γ RKS PADE dual-gate; unit kernel vs host twin |

## Acceptance

1. `cuda::gpw::gpw_forces_*` **does not** call `cpu::gpw::gpw_forces_from_scf` for the first-slice path.
2. Grid force product on device: V_eff · dχ · Pχ grid term + ionic HF (∫ V_es ∂n_c) + local-PP HF; host keeps AO Pulay (W·dS, P·dT, P·dVnl) matching hybrid STVnl energy.
3. Public free-function shape unchanged (`gpw_forces_analytic` / `gpw_forces_from_scf`).
4. GH200 dual-gate: energy \|ΔE\| ≤ 5e-6 Ha, max\|ΔF\| ≤ 5e-5 Ha/Bohr (same class as collocate/full-product).
5. Unit: device ionic / local-PP / grid-veff kernels vs host twin.
6. **perf_exempt** small-cell correctness foothold (not fair-wall vs CP2K).
7. Fail-closed: first-slice only (RKS Γ, PADE/PBE, local_pp_on_grid, ngrids=1).

## Non-goals

UKS / multi-k / multigrid / stress / PBE force dual-gate product (separate queue rows).
OT / dispersion / DFT+U force fold.

## Result

**acceptance_met=true** (wave-8 rebase GH200 **1127167** / prior **1126434**,
**1125926**): units 2/2 PASS; \|ΔE\|=**1.617e-7** Ha; max\|ΔF\|=**2.056e-6**
Ha/Bohr (≤ 5e-5); **perf_exempt** small cell. Rebase onto `dev` @
`ee4059e9` (post multigrid-run + uks-gamma land) kept MG/UKS energy paths;
device free-fn forces fail-closed on UKS + `ngrids!=1`.