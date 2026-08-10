# cuda-gpw-nondefault-stream

## Lever
`GpwDft::run` owns a non-default `cudaStream_t` for mid-SCF H2D/D2H +
`KsChainDevice` / `KsMultiGridChainDevice` (file `gpw_solver.cu`). Stream-
scoped fences; dual-gate tols unchanged.

## Acceptance
- GH200 dual-gate |ΔE|≤5e-6 Ha max|ΔF|≤5e-5 Ha/Bohr
- acceptance_met=false until GH200 green

## acceptance_met
true
