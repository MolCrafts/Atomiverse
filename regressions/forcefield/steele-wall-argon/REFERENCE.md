# Reference — Steele 10-4-3 wall, single Ar (LAMMPS offline pin)

## Oracle

```text
oracle: lammps
oracle_version: 30 Mar 2026 - Development (develop / patch_30Mar2026-1074-g7f680de296)
```

Binary used offline: `/home/jicli594/.local/x86_64/bin/lmp` (local build).
Twin input: `tools/metalwalls_oracle/inputs/lammps/in.steele_wall_argon`.

**ctest never launches LAMMPS.**

> Queue note: wave-2 rationale wrote `wall/lj93` for Steele 10-4-3. LAMMPS
> `wall/lj93` is the **9-3** form; the **10-4-3 Steele** potential is
> `fix wall/lj1043`. This pin uses `wall/lj1043`.

## Formula (LAMMPS `fix wall/lj1043`)

\[
U(r)=2\pi\varepsilon\left[
  \frac{2}{5}\left(\frac{\sigma}{r}\right)^{10}
  -\left(\frac{\sigma}{r}\right)^{4}
  -\frac{\sqrt{2}\,\sigma^{3}}{3\bigl(r+(0.61/\sqrt{2})\sigma\bigr)^{3}}
\right]-U(r_c),\qquad 0 < r < r_c
\]

Force along the wall normal follows LAMMPS `wall_particle` (energy shifted
at \(r_c\)). Units: **metal** (eV, Å) — same as Atomiverse public surface.

Ref: Steele, *Surf. Sci.* **36**, 317 (1973); LAMMPS `fix_wall` doc.

## Geometry / parameters

| Field | Value |
| --- | --- |
| Atoms | 1 Ar |
| Position | \((10, 10, 4)\) Å |
| Box | \(20\times 20\times 30\) Å, boundary `p p f` |
| Wall | `zlo 0.0` |
| \(\varepsilon\) | 0.010338 eV |
| \(\sigma\) | 3.40 Å |
| \(r_c\) | 12.0 Å |
| Pair | `pair_style zero` (wall only) |

## Offline pins (2026-07-24)

From `lmp -in in.steele_wall_argon` (`run 0`; metal units; `fix_modify energy yes`):

| Quantity | Value |
| --- | --- |
| \(E\) [eV] | `-0.035247965699900359` |
| \(F_z\) [eV/Å] | `-0.025162258813859549` |
| \(F_x = F_y\) | `0` |

Independent analytic evaluation of the same closed form agrees to ≥ 1e-15
with the LAMMPS dump (`forces.dump` %.16e).

Cross-check heights (same params, not regression gates):

| \(r\) [Å] | \(E\) [eV] | \(F_z\) [eV/Å] |
| --- | --- | --- |
| 3.0 | `-2.892345800523042e-02` | `+1.508268784423809e-01` |
| 4.0 | `-3.524796569990036e-02` | `-2.516225881385955e-02` |
| 5.0 | `-1.687854854072667e-02` | `-1.207703871038502e-02` |
| ≥ \(r_c\) | `0` | `0` |

## Tolerance (regression)

| Quantity | Tol |
| --- | --- |
| \(\|E - E_{\mathrm{pin}}\|\) | \(1\times 10^{-10}\) eV |
| \(\|F_z - F_{z,\mathrm{pin}}\|\) | \(1\times 10^{-10}\) eV/Å |
| \(\|F_x\|,\|F_y\|\) | \(1\times 10^{-12}\) eV/Å |

## Performance

Fair wall: 1-atom single-point is not a meaningful bulk twin (both engines
finish in sub-millisecond noise; startup dominates).

```text
perf_exempt: true
perf_exempt_reason: single-atom wall single-point pin; wall dominated by process startup, not kernel. Fair multi-atom slab twin deferred.
ratio_atv_over_oracle: n/a
within_10x: exempt
```
