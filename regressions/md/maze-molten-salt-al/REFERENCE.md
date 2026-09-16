# maze-molten-salt-al

oracle: metalwalls
oracle_version: MetalWalls git **c8ad175d** product scenario (MaZe + Fumi–Tosi
footholds closed wave-4/5; this slug is the MW molten-salt + Al wall composition)
oracle_kind: product composition gates; binary available for future absolute pins

## Physics

Map §9 MW example: **molten salt between Al electrodes** with MaZe charge
dynamics. ATV composition:

- `cpu::md::ConstPotMaZe` (symplectic) on Al wall types
- `cpu::PairBorn` multi-type Fumi–Tosi NaCl + soft Al–ion Born exclusion
- `cpu::PairCoulCut` + freeze + short NVT at 800 K

## Geometry

| | value |
|---|---|
| box | 12 × 12 × 24 Å |
| Al walls | 3×3 per plate at z = 3 / 21 Å, Ψ = 0 / 1 V, η = 1.5 Å⁻¹ |
| salt | 4 Na⁺ + 4 Cl⁻ mid-slab |
| MD | 160 × 0.5 fs NVT, freeze walls |

## Assertions

| check | bound |
|-------|-------|
| polarity | Q_L < 0 < Q_R |
| neutrality | \|sum Q\| ≤ 1e-5 during NVT |
| freeze | max \|Δx_wall\| ≤ 1e-10 Å |
| ion separation proxy | min ion–ion r > 1 Å |
| finite thermo | E, K finite |

## Perf

`perf_exempt` — product scenario, not a kernel microbench.
