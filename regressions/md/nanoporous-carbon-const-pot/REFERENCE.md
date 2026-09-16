# nanoporous-carbon-const-pot

oracle: metalwalls
oracle_version: MetalWalls git **c8ad175d** (product composition gate; empty-slit
soft pin optional when twin archived)
oracle_kind: real binary available; product NVT gates are self-consistency of
the const-pot + freeze + occupancy composition

## Physics

Map §9 nanoporous product: **slit-pore carbon** electrodes (honeycomb sheets)
held at fixed Ψ by `ConstPotBO`, frozen walls, mobile electrolyte inside the
pore. Occupancy proxy = ion count with \(z \in (z_{\mathrm{bot}}+\delta,
z_{\mathrm{top}}-\delta)\).

## Geometry

| | value |
|---|---|
| walls | honeycomb C, 4×7 rect cells → 112 sites/wall |
| planes | z = 8 / 32 Å in Lz = 40 Å (pore gap 24 Å) |
| Ψ | ∓1 V, η = 1.805132 Å⁻¹ (MW graphite) |
| ions | 4 Na⁺ + 4 Cl⁻ in pore mid-band |
| image sum | `xy_image_range = 40` |
| MD | 200 fs NVT, freeze walls, LJ + CoulLong/PME |

## Assertions

| check | bound |
|-------|-------|
| empty polarity / neutrality | Q_L<0<Q_R, \|sum\|≤1e-8 |
| filled polarity / neutrality | same after step 1 and chunks |
| freeze | max \|Δz_wall\| ≤ 1e-10 Å |
| occupancy | ≥ N_ions/2 after short NVT |
| finite thermo | E,K finite |

## Perf

`perf_exempt` — product box composition, not a kernel microbench.
