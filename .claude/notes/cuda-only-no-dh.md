<!-- mol:note:topic:cuda-only-no-dh -->
# CUDA-only iron law — no mid-path device↔host exchange

Binding for every claimed CUDA ES product path (RHF covered envelope, PW,
`cuda::gpw::GpwDft`, future methods).

## Law

Once a CUDA method enters its SCF (or the force/stress body of that step),
scientific state stays on the device.

| | Rule |
| --- | --- |
| **Forbidden** | Mid-iteration D2H or H2D of scientific fields (ρ, V_eff, V_AO, H, P, C, ε, forces, stress, per-iteration χ / weight packs); host loops reimplementing that science; silent CPU SCF fallback |
| **Allowed** | **Once** before the first iteration: geometry / basis / fixed H0 / tables H2D. **Once** after convergence or force finalization: Frame / `last_result` payload D2H |
| **Scalars** | Convergence metrics reduce to a device scalar; one small D2H of that scalar is fine if the host must decide stop (prefer mapped flags / device criteria) |

Violations are defects, not "hybrid footholds": staff a residency fix rather
than more dual-gates around the exchange. RHF s/p and PW density-side residency
are the acceptance bar.

## Open violations (fix order)

| Priority | Surface | Fix slug |
| --- | --- | --- |
| P1 | GPW force free-fn: grid re-H2D, host densify ρ, host AO Pulay | `force-device-handoff` / `device-ao-pulay` |
| P1 | GPW multi-k payload / cold-start (host re-diag / densify) | `kpoint-payload-device` / `kpoint-device-cold-start` / `device-cold-start` |
| P1 | GPW multi-k setup residency (host χ / Löwdin / KB Vnl / H0) | `kpoint-device-{chi,orth,vnl}` / `h0-residency` |
| P2 | GPW stress free-fn host assembly | `device-stress` |

## Worker checklist (CUDA slug)

1. Load the CUDA kernel skills ([cuda-rhf-device-only.md](cuda-rhf-device-only.md)).
2. Grep the diff for `DeviceToHost`, `HostToDevice`, `download_*host`, mid-loop
   `cudaMemcpy` — zero scientific mid-path hits for green.
3. State the lever and which skill guided launch/tiling/memory.
4. A GH200 dual-gate is still required for correctness; it never excuses an
   exchange.

## Claims

- No "CUDA wins" on H₂/H₂O.
- No "device-only product" claim while a P0/P1 row is open on that path.
- Large-cell walls only after the method's mid-path exchange is gone.
