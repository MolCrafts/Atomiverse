# Reference — constant-potential MD NVT short trajectory

oracle: metalwalls
oracle_version: MetalWalls release 20.05, git **c8ad175d** (2026-06-18),
binary at `/nobackup/proj/disk/teoroo/personal/jicli594/work/opt/metalwalls/mw`
oracle_kind: real binary (re-pinned 2026-07-29)

## Map slice

E1 product: fluctuating electrode charges at fixed potential, frozen walls,
classical Coulomb MD on `cpu::md::MD`.

## Geometry

- Box 14×14×20 Å; walls 2×2 at z = 2 / 18 Å (types 2/3); 4 Na⁺ + 4 Cl⁻
- Ψ = ∓0.5 V, Gaussian w = 0.5 Å (η = 2 Å⁻¹)
- Soft LJ + CoulLong/PME (α = 0.3 Å⁻¹); NVT 400 × 0.5 fs at 300 K

## Pin (MetalWalls c8ad175d)

Twin: `tools/metalwalls_oracle/inputs/metalwalls/const-pot-md-nvt-short/mw_twin/`
(`runtime_pbc3.inpt`, `num_pbc 3` matches ATV's 3D PME cell).

From `mw_output/total_charges.out`:

| step | Q_C1 [e] | Q_C2 [e] |
|------|----------|----------|
| 0 | **−9.711155787094686×10⁻²** | **+9.711155787094686×10⁻²** |
| 1 | −9.742254974821957×10⁻² | +9.742254974821957×10⁻² |

## Assertions

| Check | Bound |
| ----- | ----- |
| Polarity after step 1 | Q_L < 0 < Q_R |
| Neutrality | \|Q_L+Q_R\| ≤ 1e-8 e |
| Soft \|Q\| vs MW step-0 | 5 % rel (PME vs Ewald residual ~3 %) |
| Finite E/K/charges over NVT | no NaN |
| Freeze walls | Δx ≤ 1e-10 Å |
| End polarity | Q_L < 0 < Q_R |

## Perf

`perf_exempt` — short product trajectory; fair wall ratio deferred.
