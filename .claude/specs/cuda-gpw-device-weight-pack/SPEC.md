# cuda-gpw-device-weight-pack

method=cuda | twin: same-settings CPU Γ RKS energy dual-gate

## Lever
`build_H_and_energy` in `src/cuda/gpw/gpw_solver.cu` — mid-SCF every iter
still host `product_density_weight` + H2D `DeviceProduct` pack + D2H integrate
acc + host `scatter_product_acc` after device DIIS. Device P→weight and/or
device acc→V_AO scatter keeping P resident.

## Acceptance
- GH200 energy dual-gate after `device-diis-eigen`; stated tol; **perf_exempt**
- Does **not** own STVnl (`device-stvnl`) or DIIS/eigen
- geometry re-H2D residual → child `device-product-residency`
