# `cuda-gpw-uks-o2-forces`

## What

Γ collinear UKS O₂ **analytic forces + cell stress** dual-gate. Energy parent
is `cuda-gpw-uks-o2-gate`. CUDA SCF payload + host `gpw_forces_from_scf` /
`gpw_stress` assembly.

## Twin

Same-settings CPU O₂ triplet (n_α=7, n_β=5, SZV-GTH-PADE, 12 Bohr, 100 Ry,
Padé LSD). Oracle: CPU twin.

## Gates

| Gate | Tol |
| --- | ---: |
| Energy \|dE\| | **5×10⁻⁵ Ha** |
| max\|ΔF\| | **5×10⁻⁴ Ha/Bohr** |
| max‖Δσ‖ | **5×10⁻⁵ Ha/Bohr³** |

**Claim: ◐** — CPU-twin dual-gate; not an independent CP2K pin.
**perf_exempt**.
