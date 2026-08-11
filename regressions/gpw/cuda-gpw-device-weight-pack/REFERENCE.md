# cuda-gpw-device-weight-pack

## What

CUDA↔CPU energy dual-gate for mid-SCF **device P→product weight** and
**acc→V_AO scatter** on `cuda::gpw::GpwDft` (Γ RKS DIIS PADE). Geometry-fixed
product meta + prim/contrib tables are uploaded once; each SCF iteration
refreshes `DeviceProduct.weight` from resident `d_P` and scatters integrate
acc on device. Under DIIS, P is not downloaded mid-loop for packing.

## Twin

Same-settings `cpu::gpw::GpwDft` on H₂ (DZV-GTH-PADE / GTH-PADE-q1), cell 12
Bohr, cutoff 80 Ry, n_occ=1, diis_depth=6, PADE, periodic, local PP on grid.

## Tolerance

|dE| ≤ **5e-6** Ha (same class as `cuda-gpw-device-diis-eigen`).

## Oracle

`oracle: cpu-twin` — same-settings CPU GpwDft energy (not a CP2K fair wall).

## Perf

**perf_exempt** — small-cell correctness / residency foothold; not a fair-wall
claim. Child `device-product-residency` may deepen geometry residency.

## Gate

GH200 log required. Build/test only on GPU node with CUDA ON.
