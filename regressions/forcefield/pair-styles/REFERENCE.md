# forcefield/pair-styles

One case; every section below is run by `atv_regressions --case forcefield/pair-styles` and keeps its own oracle record.


## Reference — Born repulsion / BMH pair dimer (LAMMPS offline pin)

### Oracle

```text
oracle: lammps
oracle_version: 30 Mar 2026 - Development (develop / patch_30Mar2026-1074-g7f680de296)
```

Binary used offline: `/home/jicli594/.local/x86_64/bin/lmp` (local build).
Twin input: `tools/metalwalls_oracle/inputs/lammps/in.born_dimer`.

**ctest never launches LAMMPS.**

### Formula (LAMMPS `pair_style born`)

Binary-verified form (coeff order `A rho sigma C D`; \(\rho\) softness length):

\[
E(r) = A\,e^{(\sigma-r)/\rho} - \frac{C}{r^{6}} + \frac{D}{r^{8}},\quad r < r_c
\]

Energy shift (`pair_modify shift yes`):

\[
E_{\mathrm{shift}}(r) = E(r) - E(r_c)
\]

Force (shift is constant):
\(\mathrm{d}E/\mathrm{d}r = -(A/\rho)\,e^{(\sigma-r)/\rho}+6C/r^{7}-8D/r^{9}\).

Units: **metal** (eV, Å) — same as Atomiverse public surface.

### Geometry / parameters

| Field | Value |
| --- | --- |
| Atoms | 2 (single type) |
| Separation \(r\) | 2.5 Å along \(x\) |
| Box | cubic 40 Å, PBC |
| \(A\) | 1000.0 eV |
| \(\rho\) | 3.0 Å |
| \(\sigma\) | 1.0 Å |
| \(C\) | 50.0 eV·Å⁶ |
| \(D\) | 20.0 eV·Å⁸ |
| \(r_c\) | 8.0 Å |
| Shift | yes |

### Offline pins (2026-07-24)

From `lmp -in in.born_dimer` (run 0; metal units) + analytic cross-check:

| Quantity | Value |
| --- | --- |
| \(E\) [eV] | `509.3671885909988` |
| \(F_{1x}\) [eV/Å] | `-2.0172730961087780e+02` |
| \(F_{2x}\) [eV/Å] | `+2.0172730961087780e+02` |
| \(F_{1y}=F_{1z}=F_{2y}=F_{2z}\) | `0` |

Independent analytic check (same algebra) agrees with the LAMMPS dump
(`forces.dump` %.16e format) and thermo PE.

### Tolerance (regression)

| Quantity | Tol |
| --- | --- |
| \(\|E - E_{\mathrm{pin}}\|\) | \(1\times 10^{-10}\) eV |
| \(\|F_x - F_{x,\mathrm{pin}}\|\) | \(1\times 10^{-10}\) eV/Å |

### Performance

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
`PairBorn` lands, re-measure vs LAMMPS `pair_style born` on the same lattice.


## Reference — Buckingham pair dimer (LAMMPS offline pin)

### Oracle

```text
oracle: lammps
oracle_version: 30 Mar 2026 - Development (develop / patch_30Mar2026-1074-g7f680de296)
```

Binary used offline: `/home/jicli594/.local/x86_64/bin/lmp` (local build).
Twin input: `tools/metalwalls_oracle/inputs/lammps/in.buckingham_dimer`.

**ctest never launches LAMMPS.**

### Formula (LAMMPS `pair_style buck`)

\[
E(r) = A\,e^{-r/\rho} - \frac{C}{r^{6}},\quad r < r_c
\]

Energy shift (`pair_modify shift yes`):

\[
E_{\mathrm{shift}}(r) = E(r) - E(r_c)
\]

Force (shift is constant): \(\mathrm{d}E/\mathrm{d}r = -(A/\rho)\,e^{-r/\rho}+6C/r^{7}\).

Units: **metal** (eV, Å) — same as Atomiverse public surface.

### Geometry / parameters

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

### Offline pins (2026-07-23)

From `lmp -in in.buckingham_dimer` (run 0; metal units):

| Quantity | Value |
| --- | --- |
| \(E\) [eV] | `0.03576020865970150342` |
| \(F_{1x}\) [eV/Å] | `-0.30971158806504689` |
| \(F_{2x}\) [eV/Å] | `+0.30971158806504689` |
| \(F_{1y}=F_{1z}=F_{2y}=F_{2z}\) | `0` |

Independent analytic check (same algebra) agrees to machine precision with the
LAMMPS dump (`forces.dump` %.16e format).

### Tolerance (regression)

| Quantity | Tol |
| --- | --- |
| \(\|E - E_{\mathrm{pin}}\|\) | \(1\times 10^{-10}\) eV |
| \(\|F_x - F_{x,\mathrm{pin}}\|\) | \(1\times 10^{-10}\) eV/Å |

### Performance

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


## Reference — Fumi–Tosi multi-type Born + Coulomb (LAMMPS offline pin)

### Oracle

```text
oracle: lammps
oracle_version: 30 Mar 2026 - Development (develop / patch_30Mar2026-1074-g7f680de296)
```

Binary used offline: `/home/jicli594/.local/x86_64/bin/lmp` (local build).
Twin input: `tools/metalwalls_oracle/inputs/lammps/in.fumi_tosi_nacl4`.

**ctest never launches LAMMPS.**

### Formula

#### Born–Mayer–Huggins (LAMMPS `pair_style born`, shift yes)

\[
E_{\mathrm{born}}(r)=A\,e^{(\sigma-r)/\rho}-\frac{C}{r^{6}}+\frac{D}{r^{8}}-E(r_c),\quad r < r_c
\]

`pair_coeff` order: `A rho sigma C D` (metal: eV, Å).

#### Coulomb (LAMMPS `coul/cut`, unshifted)

\[
E_{\mathrm{coul}}(r)=\mathrm{qqr2e}\,\frac{q_i q_j}{r},\quad \mathrm{qqr2e}=14.399645\ \mathrm{eV\cdotÅ}
\]

#### Twin composition

```text
pair_style hybrid/overlay born 12.0 coul/cut 12.0
pair_modify pair born shift yes
```

Atomiverse: `cpu::PairBorn` (multi-type, always shifted) + `cpu::PairCoulCut`.

### Geometry / parameters

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

#### Multi-type Born table (metal)

| Pair | \(A\) [eV] | \(\rho\) [Å] | \(\sigma\) [Å] | \(C\) [eV·Å⁶] | \(D\) [eV·Å⁸] |
| --- | --- | --- | --- | --- | --- |
| Na–Na | 25.4435 | 0.317 | 2.340 | 101.1719 | 48.1771 |
| Cl–Cl | 25.4435 | 0.317 | 3.170 | 278.122 | 694.480 |
| Na–Cl | 20.3548 | 0.317 | 2.755 | 48.1771 | 48.1771 |

### Offline pins (2026-07-24)

From `lmp -in in.fumi_tosi_nacl4` (run 0; metal units):

| Quantity | Value |
| --- | --- |
| \(E\) [eV] | `54.7707332302457` |
| \(E_{\mathrm{vdwl}}\) (Born) [eV] | `67.97442356628899` |
| \(E_{\mathrm{coul}}\) [eV] | `-13.20369033604329` |
| \(F_{0x}=F_{0y}\) (Na) [eV/Å] | `-5.1254154966280602e+01` |
| \(F_{1x}=-F_{1y}\) (Cl) [eV/Å] | `+5.5205811735300870e+01` |
| \(F_{iz}\) | `0` |

### Tolerance (regression)

| Quantity | Tol |
| --- | --- |
| \(\|E - E_{\mathrm{pin}}\|\) | \(1\times 10^{-9}\) eV |
| \(\|F - F_{\mathrm{pin}}\|\) | \(1\times 10^{-8}\) eV/Å |

### Performance

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


## Reference — Steele 10-4-3 wall, single Ar (LAMMPS offline pin)

### Oracle

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

### Formula (LAMMPS `fix wall/lj1043`)

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

### Geometry / parameters

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

### Offline pins (2026-07-24)

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

### Tolerance (regression)

| Quantity | Tol |
| --- | --- |
| \(\|E - E_{\mathrm{pin}}\|\) | \(1\times 10^{-10}\) eV |
| \(\|F_z - F_{z,\mathrm{pin}}\|\) | \(1\times 10^{-10}\) eV/Å |
| \(\|F_x\|,\|F_y\|\) | \(1\times 10^{-12}\) eV/Å |

### Performance

Fair wall: 1-atom single-point is not a meaningful bulk twin (both engines
finish in sub-millisecond noise; startup dominates).

```text
perf_exempt: true
perf_exempt_reason: single-atom wall single-point pin; wall dominated by process startup, not kernel. Fair multi-atom slab twin deferred.
ratio_atv_over_oracle: n/a
within_10x: exempt
```


## Reference — soft-core Coulomb dimer

Beutler one-leg soft-core Coulomb (appearance):

\[
r_\lambda=(\alpha\sigma_C^6(1-\lambda)^p+r^6)^{1/6},\quad
V=\lambda\,k_C q_i q_j\Bigl(\frac{1}{r_\lambda}-\frac{1}{r_c}\Bigr),\quad
F_r=\lambda\,k_C q_i q_j\,\frac{r^5}{r_\lambda^7}.
\]

- Beutler et al., *Chem. Phys. Lett.* **222**, 529 (1994)
- \(k_C = \) `units::coulomb` = 14.399645 eV·Å

### Parameters / goldens (2026-07-23)

| Symbol | Value |
|---|---|
| q₁,q₂ | +1, −1 |
| r, r_c | 3.0, 10.0 Å |
| λ, α, p, σ_C | 0.5, 0.5, 1, 1.0 Å |
| E | −1.6798214399374538 eV |
| \|F_r\| | 0.799660331374428 eV/Å |


## Reference — soft-core LJ dimer

### Formula

Beutler / GROMACS one-leg soft-core LJ (appearance):

\[
r_\lambda=(\alpha\sigma^6(1-\lambda)^p+r^6)^{1/6},\quad
V=\lambda\bigl[V_{\mathrm{LJ}}(r_\lambda)-V_{\mathrm{shift}}\bigr]
\]

Hard shift \(V_{\mathrm{shift}}=V_{\mathrm{LJ}}(r_c)\) (PairLJCut convention).

- Beutler et al., *Chem. Phys. Lett.* **222**, 529 (1994),
  DOI 10.1016/0009-2614(94)00397-1
- GROMACS free-energy interactions (one-leg soft-core)

**Not** LAMMPS `lj/cut/soft` algebra.

### Parameters

| Symbol | Value |
|---|---|
| ε | 0.5 eV |
| σ | 3.0 Å |
| r_c | 8.0 Å |
| r | 3.0 Å |
| λ | 0.5 |
| α | 0.5 |
| p | 1 |

### Goldens (offline, 2026-07-23)

| Quantity | Value |
|---|---|
| r_λ | 3.11367244666864 Å |
| E | −0.15722681917773984 eV |
| F_r | 0.7680000000000029 eV/Å |
