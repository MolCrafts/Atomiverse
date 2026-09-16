# Reference — portable xsimd mixed-angular 2-center S/T (Hermite)

## Method

Γ-point lattice-summed contracted overlap \(S\) and kinetic \(T\) for
**mixed-angular** (p/d) Cartesian Gaussian shells — the GPW hard one-electron
Hermite path optimized by `cpu-simd-2center-angular`
(`contracted_ST_periodic` / image-axis portable xsimd over McMurchie–Davidson
\(E_t^{i,j}\) when live images fill ≥1 native vector and cart \(L\le 2\)).

## Oracle

**Independent Hermite path** — sum of `prim_overlap` / `prim_kinetic` over
each lattice image (scalar memoized Hermite, same physics as Helgaker
9.5.6–7 / McMurchie–Davidson). Not a CP2K FORCE_EVAL twin. Offline only.

## System (slab fragment)

| | |
| --- | --- |
| Shells | contracted p mates + one d monom (shared centers per shell) |
| Centers | \(\mathbf{R}_A=(0,0,0)\), \(\mathbf{R}_B=(0.7,-0.15,0.25)\) Bohr |
| Cell | \(L=(6,6,4)\) Bohr; image half-counts \(m=(2,2,4)\) |
| Reach | 40 Bohr (essentially all images live → SIMD image path) |

## Gate

| Check | Tol |
| --- | --- |
| \(S,T\) fused vs prim image sum | rel **1e-11** · max(1,\|ref\|) |
| public `overlap`/`kinetic` vs fused | **1e-14** |
| Wall | printed microbench only; **perf_exempt** vs fair twins |
