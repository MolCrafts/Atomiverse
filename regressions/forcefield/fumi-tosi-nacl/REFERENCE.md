# Reference — Fumi–Tosi multi-type Born + Coulomb (LAMMPS offline pin)

## Oracle

```text
oracle: lammps
oracle_version: 30 Mar 2026 - Development (develop / patch_30Mar2026-1074-g7f680de296)
```

Binary used offline: `/home/jicli594/.local/x86_64/bin/lmp` (local build).
Twin input: `tools/metalwalls_oracle/inputs/lammps/in.fumi_tosi_nacl4`.

**ctest never launches LAMMPS.**

## Formula

### Born–Mayer–Huggins (LAMMPS `pair_style born`, shift yes)

\[
E_{\mathrm{born}}(r)=A\,e^{(\sigma-r)/\rho}-\frac{C}{r^{6}}+\frac{D}{r^{8}}-E(r_c),\quad r < r_c
\]

`pair_coeff` order: `A rho sigma C D` (metal: eV, Å).

### Coulomb (LAMMPS `coul/cut`, unshifted)

\[
E_{\mathrm{coul}}(r)=\mathrm{qqr2e}\,\frac{q_i q_j}{r},\quad \mathrm{qqr2e}=14.399645\ \mathrm{eV\cdotÅ}
\]

### Twin composition

```text
pair_style hybrid/overlay born 12.0 coul/cut 12.0
pair_modify pair born shift yes
```

Atomiverse: `cpu::PairBorn` (multi-type, always shifted) + `cpu::PairCoulCut`.

## Geometry / parameters

| Field | Value |
| --- | --- |
| Atoms | 4 (2 Na + 2 Cl planar square) |
| Edge \(d\) | 2.82 Å |
| Positions | Na(10,10,20), Cl(12.82,10,20), Na(12.82,12.82,20), Cl(10,12.82,20) |
| Box | cubic 40 Å, PBC |
| Types | Na = frame type 0 / LAMMPS type 1; Cl = 1 / 2 |
| Charges | \(q_{\mathrm{Na}}=+1\), \(q_{\mathrm{Cl}}=-1\) |
| \(r_c\) | 12.0 Å (Born and Coulomb) |
| Born shift | yes |

### Multi-type Born table (metal)

| Pair | \(A\) [eV] | \(\rho\) [Å] | \(\sigma\) [Å] | \(C\) [eV·Å⁶] | \(D\) [eV·Å⁸] |
| --- | --- | --- | --- | --- | --- |
| Na–Na | 25.4435 | 0.317 | 2.340 | 101.1719 | 48.1771 |
| Cl–Cl | 25.4435 | 0.317 | 3.170 | 278.122 | 694.480 |
| Na–Cl | 20.3548 | 0.317 | 2.755 | 48.1771 | 48.1771 |

## Offline pins (2026-07-24)

From `lmp -in in.fumi_tosi_nacl4` (run 0; metal units):

| Quantity | Value |
| --- | --- |
| \(E\) [eV] | `54.7707332302457` |
| \(E_{\mathrm{vdwl}}\) (Born) [eV] | `67.97442356628899` |
| \(E_{\mathrm{coul}}\) [eV] | `-13.20369033604329` |
| \(F_{0x}=F_{0y}\) (Na) [eV/Å] | `-5.1254154966280602e+01` |
| \(F_{1x}=-F_{1y}\) (Cl) [eV/Å] | `+5.5205811735300870e+01` |
| \(F_{iz}\) | `0` |

## Tolerance (regression)

| Quantity | Tol |
| --- | --- |
| \(\|E - E_{\mathrm{pin}}\|\) | \(1\times 10^{-9}\) eV |
| \(\|F - F_{\mathrm{pin}}\|\) | \(1\times 10^{-8}\) eV/Å |

## Performance

Fair wall: 4-atom single-point is not a meaningful bulk twin (both engines
finish in sub-millisecond noise dominated by process startup).

```text
perf_exempt: true
perf_exempt_reason: 4-ion single-point pin; wall dominated by process startup, not pair kernel. Fair bulk NL twin deferred with production NeighborList-backed PairBorn.
ratio_atv_over_oracle: n/a
within_10x: exempt
```

Optional bulk microbench (not a ctest gate): when a production NL multi-type
`PairBorn` lands, re-measure vs LAMMPS `born` (+ `coul/cut` or `coul/long`)
on a molten-salt box.
