# cuda-pw-smear-stress

method=cuda | twin: same-settings CPU restricted metallic-smear `pw_stress`

## Lever

| Item | Value |
| --- | --- |
| File | `src/cuda/planewave/internal/pw_stress.cu` + `src/cuda/planewave/pw_solver.cu` |
| Twin | `src/cpu/planewave/internal/pw_stress.cpp` (`run_scf_smear_restricted` + fractional NSCF) |
| Function | `pw_stress` / `pw_stress_analytic_impl` / `run_device_scf` / `nscf_energy_ry` |
| Metric | max‖σ_cuda − σ_cpu‖ [eV/Å³] under restricted FD smear (Mermin free energy) |

Parent `cuda-pw-smear-forces` landed (Γ Al₂ FD dual-gate). Free-fn / public
`cuda::pw::pw_stress` still rejects `smearing != none` while energy + forces
accept restricted metallic smear. Lift the stress reject and assemble
Nielsen–Martin stress of the **Mermin free energy** under fractional
occupations on both backends (honest dual-gate):

1. Reference SCF: restricted FD/Gaussian/MP smear (not fixed-occ).
2. NSCF under ±h strain: remapped ρ, re-diagonalize, re-find μ / f_b, free
   energy \(F = E_{\mathrm{KS}} - \sigma S\).
3. \(\sigma = -(1/\Omega)\,\partial F/\partial\varepsilon\).

Does **not** own: spin∧smear, multi-k smear stress (`kmesh-smear-stress`),
smear-PBE stress, forces (already landed).

## Acceptance

- GH200 CUDA↔CPU restricted metallic-smear cell stress dual-gate with stated
  max‖Δσ‖ (and energy hold if co-gated)
- device-only kernels in `.cu` for any new device work; no host loops in `.cu`
- no soft tols; same-settings CPU oracle; **perf_exempt** small cell
- no dual-gate matrix explosion (spin/multi-k/PBE joints stay child rows)
- unit: restricted smear no longer throws on stress path; open-shell without
  smear still rejects; degauss≤0 with smear throws

## Tolerances (design-time)

| Quantity | Tol |
| --- | --- |
| max\|σ_cuda − σ_cpu\| | 5e-2 eV/Å³ (same class as `reg_cuda-pw-stress`) |
