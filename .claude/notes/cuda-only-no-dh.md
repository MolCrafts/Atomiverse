<!-- mol:note:topic:cuda-only-no-dh -->
# CUDA-only iron law — no mid-path Device↔Host exchange

Binding for all **claimed CUDA product** paths (RHF covered envelope, PW,
`cuda::gpw::GpwDft`, and any future ES CUDA method). Enforced by
`es-cpu-cuda-opt` extra_law + code review. Current understanding only.

## Law

**Mid-path is device-only.** Once a CUDA method has entered its SCF (or
force/stress body for that step), scientific state stays on the device.

| | Rule |
| --- | --- |
| **Forbidden** | Mid-iter `cudaMemcpy` / `cudaMemcpyAsync` of scientific fields either **Device→Host** or **Host→Device** (ρ, \(V_{\mathrm{eff}}\), \(V_{\mathrm{AO}}\), \(H\), \(P\), \(C\), ε, forces, stress, per-iter χ/weight packs). Host for-loops that reimplement that science. Silent CPU SCF fallback. |
| **Allowed boundaries** | **Once** before the first SCF iteration: geometry / basis / fixed \(H_0\) / one-time tables **H2D**. **Once** after convergence (or force finalization): Frame / `last_result` payload **D2H**. |
| **Scalars** | Convergence metrics may use device reductions to a **device scalar** then a single small D2H of that scalar if the host driver must decide stop — not full matrices. Prefer mapped flags / device-side criteria when possible (RHF style). |

Violations are **defects**, not “hybrid footholds to dual-gate around”. Staff a
residency fix; do not open more energy dual-gates that leave the exchange in
place.

## Why

CLAUDE ES CUDA order: rewrite hot paths in **CUDA programming style** —
device-only, tiling, **minimal H2D**. Mid-SCF PCIe of dense AO matrices
destroys any large-system story and is not “CUDA product”.

## Audit hotspots (known debt — fix order)

| Priority | Surface | Status |
| --- | --- | --- |
| P0 | Γ RKS mid-SCF | **fixed** — device \(H=H_0+V\), `residual_device`, no V_AO D2H |
| P0 | Γ UKS mid-SCF | **fixed** — pack from `d_P`, device dual \(H\), no mid-SCF P D2H |
| P0 | Multi-k collocate/P | **fixed** — `collocate_k_device_cm` + no P ping-pong; device Re⟨P,T/Vnl⟩ |
| P0 | Multi-k residual H | **fixed earlier** — `ingest_H_k_device_rm` + `residual_from_device` |
| P0 | Γ RKS single-grid KS | **fixed** — device product KS + PBC min-image collocate/integrate (`product-collocate-pbc`); cbg retained force-only |
| P1 | Forces free-fn | open — full grid re-H2D + host densify ρ + host AO Pulay (`force-device-handoff` / `device-ao-pulay`) |
| P1 | Γ payload | **fixed** — RKS `device-payload` (**1182615**); UKS `uks-payload-device` merge `f2be952a` / **1186414** dual `eigen_from_device` |
| P1 | Multi-k payload / cold-start | open — host re-diag / densify (`kpoint-payload-device` / `kpoint-device-cold-start` / `device-cold-start`) |
| P1 | Multi-k setup residency | open — host χ / Löwdin / KB Vnl / H0 rematerialize (`kpoint-device-chi` / `orth` / `vnl` / `h0-residency`) |
| P1 | DIIS residual | **fixed** — device max-\|e\| scalar only |
| P2 | Stress free-fn | open — host assembly shell (`device-stress`) |
| P2 | Si MG energy dual-gate | **landed** (`88d8ec59` / **1180883**); iron-law mid-path closed by monom default |
| P2 | Si MG device monom KS | **fixed** `multigrid-device-ks` merge `3730ada7` / **1186912** — device pack→MG chain→scatter default; host collint opt-in `ATV_GPW_CUDA_HOST_MG=1` only |

RHF s/p and PW density-side residency are the **acceptance bar**.

## Worker checklist (CUDA slug)

1. Load CUDA kernel skills (see workflow `extra_law` / RHF note skill table).
2. Grep the diff for `DeviceToHost`, `HostToDevice`, `download_*host`, mid-loop
   `cudaMemcpy` — zero scientific mid-path hits required for green.
3. State lever + which skill guided launch/tiling/memory.
4. GH200 dual-gate still required for correctness; **does not** excuse DH.

## Claims discipline

- No “CUDA wins” on H₂/H₂O.
- No “device-only product” claim while any P0 row above remains.
- Large-cell walls only after mid-path DH is gone on that method.
