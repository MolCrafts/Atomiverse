# extended-lagrangian-dipoles

oracle: metalwalls
oracle_version: MetalWalls git **c8ad175d** for stationary PIM (thole_a=0);
iEL/SCF dynamics are literature-form (Albaugh 2015 / Niklasson 2009)
oracle_kind: real binary for Gate A; dynamics gates are scheme self-consistency

## Gate A — stationary XL fixed point = SCF PIM (MetalWalls)

Same twin as `pim-induced-dipole-bulk` (two-site, L=50 Å, α=1.0/1.5 Å³,
`thole_a=0` — MW has no Thole):

| Observable | MetalWalls c8ad175d |
|------------|---------------------|
| E_pol | **−2.4397739572766×10⁻¹ eV** (−8.9660039191731e-3 Ha) |
| F_A,x = −F_B,x | **+3.4926913376021×10⁻¹ eV/Å** |

Cold and warm XL launches must reproduce these (fixed point of the XL
recursion). Supersedes the wave-5 Thole `0.39` number mis-attributed to MW.

## Gate B — dynamics (XL tracks BO twin)

On 8-ion bulk toy, 150 × 0.5 fs: μ_XL tracks cold SCF to <1e-6 e·Å; warm
XL needs fewer SCC iters; energy drift bounded. Uses `thole_a=0.39` as a
shared XL/SCF damping (not a MW claim).

## Perf

`perf_exempt` until a fair mass-zero/XL wall twin is timed.
