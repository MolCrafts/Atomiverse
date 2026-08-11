# cuda-gpw-device-weight-pack

method=cuda | twin: same-settings CPU Γ RKS energy dual-gate

## Lever

| | |
| --- | --- |
| **File** | `src/cuda/gpw/gpw_solver.cu` (`build_H_and_energy` / UKS `build_uks_H`) |
| **Function** | mid-SCF host `product_density_weight` + full `DeviceProduct` H2D + D2H integrate acc + host `scatter_product_acc`; RKS also `download_P` every DIIS iter only to re-pack weights |
| **Metric** | CUDA↔CPU \|dE\| ≤ 5e-6 Ha (H₂ Γ RKS PADE); P resident mid-SCF under DIIS (no per-iter `download_P` for packing) |

**Device path (this slug):**

1. Geometry-fixed product meta + prims/contribs H2D **once** per `run`.
2. Each SCF iter: device kernel P→`DeviceProduct.weight` from resident `d_P` (RKS `AoDiisEigenDevice::d_P()`; UKS uploads host Pa/Pb).
3. Device KS chain (already landed).
4. Device acc→V_AO scatter; D2H V only (for host residual / H0+V).
5. Device Frobenius tr(P·T), tr(P·Vpp) so energy does not force a P download.

Does **not** own STVnl (`device-stvnl`) or DIIS/eigen. Geometry re-H2D residual of full packs is reduced here (weight-only update into resident `d_prods`); further residency polish is child `device-product-residency`.

## Acceptance

- Unit: device weight vs host `product_density_weight`; device scatter vs host `scatter_product_acc` (toy prims/products).
- Regression: `reg_cuda-gpw-device-weight-pack` H₂ Γ RKS PADE energy dual-gate vs `cpu::gpw::GpwDft`, \|dE\| ≤ 5e-6 Ha.
- GH200 log required; **perf_exempt** (small-cell correctness foothold; not fair-wall).
- Public API unchanged (`cuda::gpw::GpwDft`).

## Out of scope

- Device STVnl / Löwdin
- UKS device DIIS (`uks-device-diis`)
- Force-stream / AO-Pulay
- Full product residency without any geometry touch (child row)
