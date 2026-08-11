# cuda-pw-smear-forces

method=cuda | twin: CPU forces under metallic occ + dual-gate

## Lever
`cuda::pw::pw_forces_analytic` in `src/cuda/planewave/pw_solver.cu` rejects
`smearing != none` while restricted metallic smear **energy** is landed.
Lift force reject for restricted (non-spin) smear; fractional occupation
force assembly.

## Acceptance
- GH200 dual-gate max|ΔF| stated; energy smear path unchanged
- spin+smear forces stay reject (child `spin-smear-forces`)
- no soft tols; same-settings CPU oracle
