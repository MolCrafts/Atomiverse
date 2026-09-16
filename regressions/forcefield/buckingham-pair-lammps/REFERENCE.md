# Reference — Buckingham pair dimer (LAMMPS offline pin)

## Oracle

```text
oracle: lammps
oracle_version: 30 Mar 2026 - Development (develop / patch_30Mar2026-1074-g7f680de296)
```

Binary used offline: `/home/jicli594/.local/x86_64/bin/lmp` (local build).
Twin input: `tools/metalwalls_oracle/inputs/lammps/in.buckingham_dimer`.

**ctest never launches LAMMPS.**

## Formula (LAMMPS `pair_style buck`)

\[
E(r) = A\,e^{-r/\rho} - \frac{C}{r^{6}},\quad r < r_c
\]

Energy shift (`pair_modify shift yes`):

\[
E_{\mathrm{shift}}(r) = E(r) - E(r_c)
\]

Force (shift is constant): \(\mathrm{d}E/\mathrm{d}r = -(A/\rho)\,e^{-r/\rho}+6C/r^{7}\).

Units: **metal** (eV, Å) — same as Atomiverse public surface.

## Geometry / parameters

| Field | Value |
| --- | --- |
| Atoms | 2 (single type) |
| Separation \(r\) | 2.5 Å along \(x\) |
| Box | cubic 40 Å, PBC |
| \(A\) | 1000.0 eV |
| \(\rho\) | 0.3 Å |
| \(C\) | 50.0 eV·Å⁶ |
| \(r_c\) | 8.0 Å |
| Shift | yes |

## Offline pins (2026-07-23)

From `lmp -in in.buckingham_dimer` (run 0; metal units):

| Quantity | Value |
| --- | --- |
| \(E\) [eV] | `0.03576020865970150342` |
| \(F_{1x}\) [eV/Å] | `-0.30971158806504689` |
| \(F_{2x}\) [eV/Å] | `+0.30971158806504689` |
| \(F_{1y}=F_{1z}=F_{2y}=F_{2z}\) | `0` |

Independent analytic check (same algebra) agrees to machine precision with the
LAMMPS dump (`forces.dump` %.16e format).

## Tolerance (regression)

| Quantity | Tol |
| --- | --- |
| \(\|E - E_{\mathrm{pin}}\|\) | \(1\times 10^{-10}\) eV |
| \(\|F_x - F_{x,\mathrm{pin}}\|\) | \(1\times 10^{-10}\) eV/Å |

## Performance

Fair wall: 2-atom single-point is not a meaningful bulk twin (both engines
finish in sub-millisecond noise). Recorded once for completeness:

| Engine | Wall (single run 0 / evaluate) | Note |
| --- | --- | --- |
| LAMMPS `run 0` | ~ms (startup-dominated) | full MD stack spin-up |
| ATV `Evaluate` dimer | ~µs–ms (startup-dominated) | O(N²) foothold |

```text
perf_exempt: true
perf_exempt_reason: ion-dimer single-point pin; wall dominated by process startup, not pair kernel. Fair bulk NL twin deferred with Fumi–Tosi multi-type + NeighborList path.
ratio_atv_over_oracle: n/a
within_10x: exempt
```

Optional bulk microbench (not a ctest gate): when a production NL
`PairBuck` lands, re-measure vs LAMMPS `pair_style buck` on the same lattice.
