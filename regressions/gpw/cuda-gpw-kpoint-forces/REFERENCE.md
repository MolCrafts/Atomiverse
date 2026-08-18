# `cuda-gpw-kpoint-forces`

## What

Multi-k RKS H₂ **analytic forces + cell stress** dual-gate. Energy parent
is `cuda-gpw-kpoints` (`k_mesh={2,1,1}`, TR off). CUDA SCF payload + host
k-path assembly.

## Twin

Same-settings CPU H₂ DZV-GTH-PADE, 12 Bohr, 80 Ry, Padé, `n_occ=1`.

## Gates

| Gate | Tol |
| --- | ---: |
| Energy \|dE\| | **5×10⁻⁶ Ha** |
| max\|ΔF\| | **5×10⁻⁵ Ha/Bohr** |
| max‖Δσ‖ | **5×10⁻⁵ Ha/Bohr³** |

**perf_exempt**.
