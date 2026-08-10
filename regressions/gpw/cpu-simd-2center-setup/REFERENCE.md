# Reference — portable xsimd analytic 2-center S/T (slab pure-s)

## Method

Γ-point lattice-summed contracted overlap \(S\) and kinetic \(T\) for **pure
s–s** Gaussian shells — the GPW hard one-electron fragment optimized by
`cpu-simd-2center-setup` (`contracted_ST_periodic` / image-axis portable
xsimd when the live image count fills ≥1 native vector).

## Oracle

**Literature closed form** (standard s–s McMurchie–Davidson / Helgaker
molecular integrals; same formulae as `tests/cpu/gpw/test_cpu_gpw_one_electron.cpp`):

\[
S_{pq} = c_p c_q \left(\frac{\pi}{p}\right)^{3/2} e^{-\mu R^2},\quad
T_{pq} = \mu(3-2\mu R^2)\,S_{pq}
\]

with \(p=\alpha_p+\alpha_q\), \(\mu=\alpha_p\alpha_q/p\), and
\(R^2=|\mathbf{R}_A-\mathbf{R}_B-\mathbf{T}|^2\). Periodic:

\[
S = \sum_{\mathbf{T}}\sum_{pq} S_{pq}(\mathbf{T}),\quad
T = \sum_{\mathbf{T}}\sum_{pq} T_{pq}(\mathbf{T}).
\]

No CP2K twin (fragment is not a full FORCE_EVAL). Offline only — no live
oracle in ctest.

## System (slab fragment)

| | |
| --- | --- |
| Shells | 4-prim pure-s bra × 4-prim pure-s ket (shared centers) |
| Centers | \(\mathbf{R}_A=(0,0,0)\), \(\mathbf{R}_B=(0.8,0.2,-0.3)\) Bohr |
| Cell | \(L=(8,8,5)\) Bohr; image half-counts \(m=(2,2,5)\) |
| Reach | 50 Bohr (essentially all images live → SIMD image path) |

## Gate

| Check | Tol |
| --- | --- |
| \(S,T\) vs closed-form lattice sum | rel **1e-12** · max(1,\|ref\|) |
| Fused `contracted_ST_periodic` vs separate image sum of `prim_ST` | same |
| Wall | printed microbench only; **perf_exempt** vs fair twins (slab-class) |
