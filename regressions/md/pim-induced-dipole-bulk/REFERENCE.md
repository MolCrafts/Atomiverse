# Reference — bulk PIM self-consistent induced dipoles (E4)

oracle: metalwalls
oracle_version: **MetalWalls git c8ad175d** (release branch, 2026-06-18), run
locally as `mpirun -n 1 mw`, `num_steps 0`
verdict: **repinned** (2026-07-29) — the wave-1 pin was a *Thole* number and
MetalWalls implements no Thole damping; a real-MetalWalls gate was added and
the literature pin was re-labelled. Details in **Provenance / what changed**.
perf_exempt: false

## Physics

Classical Polarizable Ion Model (bulk, no electrodes):

\[
\boldsymbol{\mu}_i=\alpha_i\mathbf{E}_i^{\mathrm{tot}},\quad
\mathbf{E}_i^{\mathrm{tot}}
  =\mathbf{E}_i^{\mathrm{perm}}
  +\sum_{j\neq i}\mathbf{T}_{ij}\cdot\boldsymbol{\mu}_j
\]

\[
E_{\mathrm{pol}}
  =-\tfrac12\sum_i\boldsymbol{\mu}_i\cdot\mathbf{E}_i^{\mathrm{perm}}
  \quad\text{[eV via `units::coulomb`]}
\]

Permanent field from Frame point charges (min-image, cutoff). Mutual induction
uses the dipole tensor, optionally screened by **Thole** exponential damping
(\(u=a r/(\alpha_i\alpha_j)^{1/6}\), \(\lambda_3,\lambda_5,\lambda_7\)) or by
**Tang–Toennies** \(f_n(br)\) (`tt_b>0`). Forces: Hellmann–Feynman at SCF
(charge–dipole + dipole–dipole); permanent charge–charge is **not** included
(use `PairCoulCut` / PME separately).

MetalWalls capability map **E4**. LAMMPS is not the oracle here (its
polarizable path is Drude / permanent multipoles, not this SCF bulk PIM).

### Where MetalWalls and Atomiverse agree exactly, and where they cannot

`E_pol = -½ Σ μ·E^perm` is MetalWalls' own bulk-PIM energy: MetalWalls
*minimises* `U = U_qq + U_qμ + U_μμ + Σ|μ|²/2α`, and at the stationary point
that functional equals `U_qq - ½ Σ μ·E^perm`. So

> **E_pol(MetalWalls) = Coulomb(polarizable run) − Coulomb(same geometry, α = 0)**

which is how every MetalWalls number below was extracted (two runs, identical
geometry / box / Ewald parameters, one with `polarizability` set and one with
`polarizability 0.0`). The same subtraction on `forces.out` gives the
polarization force, matching what `PimInducedDipole` writes to Frame.

Two things do **not** carry over and are handled explicitly:

1. **Thole damping does not exist in MetalWalls.** `grep -ri thole` over the
   whole MetalWalls tree at c8ad175d returns **zero hits**. Its only
   short-range multipole screening is Tang–Toennies applied to the
   charge → dipole field (`damping` / `tt_pair <A> <B> b order c`, see
   `src/coulomb_sr_pim_3DPBC.inc`), and even that is a different functional
   form from `PimInducedDipole::tt_b` (MetalWalls carries a per-ordered-pair
   intensity `c` and screens only the charge→dipole kernel; Atomiverse applies
   \(f_1,f_3,f_5,f_7\) to every multipole kernel with no intensity). So a
   `thole_a = 0.39` number **cannot** be produced by MetalWalls at all. The
   MetalWalls-expressible configuration is `thole_a = 0`, `tt_b = 0`.
2. **MetalWalls is Ewald-periodic; `PimInducedDipole` is min-image/cutoff.**
   At L = 50 Å and r = 3 Å only the direct pair falls inside the 20 Å cutoff,
   so Atomiverse computes the *isolated* pair while MetalWalls computes the
   full 3D lattice sum with a neutralising background. The difference is the
   background term \(-\tfrac{2\pi}{3V}r^2\) in the effective pair potential and
   is **not** a disagreement — it scales exactly as \(1/L^3\) (verified below)
   and is removed by extrapolating MetalWalls to \(L\to\infty\).

## Geometry (both gates)

Cubic box \(L=50\) Å, cutoff 20 Å.

| Site | type | \(q\) [e] | \(\alpha\) [Å³] | \((x,y,z)\) [Å] |
| ---- | ---- | --------- | --------------- | --------------- |
| A | 1 | +1 | 1.0 | (0, 0, 0) |
| B | 2 | −1 | 1.5 | (3, 0, 0) |

MetalWalls twin (atomic units — `tools/metalwalls_oracle/inputs/metalwalls/pim-induced-dipole-bulk/{pol,nopol}/`):

| quantity | Atomiverse | MetalWalls input |
| --- | --- | --- |
| r(A–B) | 3 Å | 5.669178373877311 a₀ |
| L | 50 Å | 94.486306231288509 a₀ |
| α(A) | 1.0 Å³ | 6.748334494600375 a₀³ |
| α(B) | 1.5 Å³ | 10.122501741900562 a₀³ |
| cutoff | 20 Å | `coulomb_rcut 47.0` a₀ (Ewald; rtol = ktol = 1e-15) |
| SCC | Jacobi, tol 1e-14 e·Å | `dipoles_minimization cg 1.0e-14 500` |
| damping | `thole_a = 0`, `tt_b = 0` | `tt_pair … c = 0.0` (see note) |

> **MetalWalls input gotcha:** the `damping` block is *mandatory* whenever
> `polarizability > 0`. `system%tt` is allocated only inside
> `read_tt_parameters`, while `coulomb_sr_pim` reads `tt%order/b/c`
> unconditionally when `pimrun` is true — omitting the block SIGSEGVs in
> `qmelt2mumelt_electricfield`. The twin therefore declares a block with
> intensity `c = 0.0`, which is *exactly* undamped
> (`dampfunc = -Σf·exp·c·r⁻³ = 0`).

## Raw MetalWalls output (unconverted, atomic units)

Box ladder, both runs at each L (`energies_breakdown.out` column *Coulomb
energy*, `dipoles.out` column *mux*, `forces.out` column *fx_total*):

| L [Å] | Coulomb(pol) [Ha] | Coulomb(α=0) [Ha] | μ_A [e·a₀] | μ_B [e·a₀] | Fx_A(pol) [Ha/a₀] | Fx_A(α=0) [Ha/a₀] |
| --- | --- | --- | --- | --- | --- | --- |
| 50 | -1.854230777074360e-01 | -1.764726282995376e-01 | 2.350425216726092e-01 | 3.408115027774546e-01 | 3.787501788985436e-02 | 3.108582740719812e-02 |
| 100 | -1.853664657312579e-01 | -1.764023917148879e-01 | 2.352116583319346e-01 | 3.410568997826066e-01 | 3.790257607718792e-02 | 3.111075171513420e-02 |
| 200 | -1.853594140368846e-01 | -1.763936508951200e-01 | 2.352326278473861e-01 | 3.410873102287844e-01 | 3.790599571004718e-02 | 3.111383989835044e-02 |
| 400 | -1.853585333394395e-01 | -1.763925595028977e-01 | 2.352352436798666e-01 | 3.410911033311218e-01 | 3.790642238375128e-02 | 3.111422506724607e-02 |
| 800 | -1.853584232759848e-01 | -1.763924231166841e-01 | 2.352355704920252e-01 | 3.410915772132902e-01 | 3.790647569364988e-02 | 3.111427318667677e-02 |
| 1600 | -1.853584095187919e-01 | -1.763924060695891e-01 | 2.352356113383368e-01 | 3.410916364405839e-01 | 3.790648235662846e-02 | 3.111427920077185e-02 |
| 3200 | -1.853584077991659e-01 | -1.763924039387391e-01 | 2.352356164439632e-01 | 3.410916438437465e-01 | 3.790648318947710e-02 | 3.111427995250770e-02 |

**Sanity check on the MetalWalls convention** (independent of Atomiverse): at
L = 50 Å the α = 0 run gives −1.764726282995376e-01 Ha, while the isolated pair
is −1/r = −0.1763924036343333 Ha. The difference is
\(-\tfrac{2\pi}{3V}r^2 = -7.980\times10^{-5}\) Ha with V = L³ = 843 509 a₀³ —
i.e. −0.1764722 Ha, matching to 4e-7. Same for the force:
\(1/r^2 - \tfrac{4\pi}{3V}r = 0.0310862\) vs MetalWalls 0.03108583 Ha/a₀. The
neutralising-background Ewald convention is confirmed.

### Derived MetalWalls polarization quantities

`E_pol = ΔCoulomb`, `F_pol,A = ΔFx_A`:

| L [Å] | E_pol [Ha] | F_pol,A [Ha/a₀] |
| --- | --- | --- |
| 50 | -8.950449407898381e-03 | 6.789190482656242e-03 |
| 100 | -8.964074016370005e-03 | 6.791824362053722e-03 |
| 200 | -8.965763141764616e-03 | 6.792155811696739e-03 |
| 400 | -8.965973836541796e-03 | 6.792197316505214e-03 |
| 800 | -8.966000159300713e-03 | 6.792202506973113e-03 |
| 1600 | -8.966003449202786e-03 | 6.792203155856612e-03 |
| 3200 | -8.966003860426813e-03 | 6.792203236969399e-03 |

### Extrapolation to the isolated limit

Richardson on \(f(L)=f_\infty+c/L^3\), \(f_\infty=(8f(2L)-f(L))/7\). The ladder
is stationary to 8e-13 over the last three doublings, so the extrapolant is
converged, not fitted:

| pair (L, 2L) | μ_A [e·a₀] | E_pol [Ha] | F_pol,A [Ha/a₀] |
| --- | --- | --- | --- |
| 400 / 800 | 2.3523561717948e-01 | -8.9660039196948e-03 | 6.7922032484685e-03 |
| 800 / 1600 | 2.3523561717352e-01 | -8.9660039191888e-03 | 6.7922032485543e-03 |
| 1600 / 3200 | 2.3523561717334e-01 | -8.9660039191731e-03 | 6.7922032485569e-03 |

**Oracle values pinned in the regression (gate A, `thole_a = 0`):**

| Observable | MetalWalls c8ad175d, isolated limit | eV / Å units (×27.211386245988, ×0.529177210903) |
| ---------- | ----------------------------------- | ------------------------------------------------ |
| μ_A | **+2.3523561717334e-01** e·a₀ | +1.2448132780408e-01 e·Å |
| μ_B | **+3.4109164490134e-01** e·a₀ | +1.8049792531618e-01 e·Å |
| E_pol | **−8.9660039191731e-03** Ha | −2.4397739572766e-01 eV |
| F_A,x | **+6.7922032485569e-03** Ha/a₀ | +3.4926913376021e-01 eV/Å |
| F_B,x | −F_A,x | −F_A,x |

Conversions used, spelled out (CODATA-2018, the same constants MetalWalls
prints in its own file headers — `1 Eh = 2625.499640 kJ/mol`,
`1 au dipole = 2.541746 D`, `1 au force = 51.421 eV/angstrom`):
1 Ha = 27.211386245988 eV, 1 a₀ = 0.529177210903 Å, 1 Ha/a₀ = 51.42208619083 eV/Å.
The regression converts **Atomiverse → atomic units** and compares there, so
the pinned digits are literally MetalWalls' own output.

### Measured agreement, and the one residual

Atomiverse column = what `reg_pim-induced-dipole-bulk` actually prints
(14 SCC iterations at tol 1e-14 e·Å):

| Observable | Atomiverse (thole_a = 0) | MetalWalls (isolated limit) | relative |
| --- | --- | --- | --- |
| μ_A [e·a₀] | 2.3523561717333177e-01 | 2.3523561717334e-01 | **−3.5e-14** |
| μ_B [e·a₀] | 3.4109164490133109e-01 | 3.4109164490134e-01 | **−2.6e-14** |
| E_pol [Ha] | -8.9660036212596272e-03 | -8.9660039191731e-03 | **+3.323e-08** |
| F_A,x [Ha/a₀] | 6.7922030228728730e-03 | 6.7922032485569e-03 | **−3.323e-08** |

(The E_pol sign is `+` because E_pol is negative — |E_pol| is *smaller* in
Atomiverse, i.e. both E and F are scaled down by the same factor.)

The dipoles match to machine precision; the energy and the force are low by a
constant **3.3227e-08** relative. That constant is not physics — it is
`core/units.h`:

```cpp
inline constexpr double coulomb = 14.399645;   // eV·Å/e²
```

versus the exact \(e^2/4\pi\varepsilon_0 = 1\,\mathrm{Ha}\cdot a_0 =
14.399645478456\ \mathrm{eV\,\AA/e^2}\); the deficit is
\((14.399645-14.399645478456)/14.399645478456 = -3.3227\times10^{-8}\), to four
digits the observed offset. It shows up in E and F and not in μ because
`units::coulomb` multiplies only at the end — the SCC itself runs in Gaussian
units (μ = αE with μ in e·Å), where the constant never appears. Rescaling
MetalWalls by `14.399645 / 14.399645478456` reproduces the Atomiverse energy to
**6e-14** and the force to **2e-14**.

So: **the Atomiverse bulk-PIM kernel is MetalWalls' bulk-PIM kernel to machine
precision.** The 8-digit `units::coulomb` is a repo-wide constant (every eV pin
in the tree is built on it) and is left alone here; it only sets the floor of
this gate.

## Pin values (hard-coded in the regression)

### Gate A — MetalWalls c8ad175d oracle (`thole_a = 0`)

| Observable | Pin (atomic units) | Tolerance | Measured |
| ---------- | ------------------ | --------- | -------- |
| μ_A | +2.3523561717334e-01 e·a₀ | 1e-9 rel | 2.6e-14 |
| μ_B | +3.4109164490134e-01 e·a₀ | 1e-9 rel | 2.7e-14 |
| E_pol | −8.9660039191731e-03 Ha | 1e-7 rel | 3.32e-08 |
| F_A,x = −F_B,x | +6.7922032485569e-03 Ha/a₀ | 1e-7 rel | 3.32e-08 |

The 1e-7 on E and F is **not** slack for numerical noise — the computation is
deterministic double precision. It is exactly one decade above the 3.32e-08
`units::coulomb` floor derived above, which is a fixed, understood, non-PIM
offset. Tighten it the day `units::coulomb` gains its remaining digits; never
widen it.

### Gate B — Thole/Applequist closed form (`thole_a = 0.39`), **not** MetalWalls

| Observable | Value |
| ---------- | ----- |
| \(E_{\mathrm{pol}}\) [eV] | **−0.12040946414707571** |
| \(F_{A,x}\) [eV/Å] | **+0.040341072868572643** |
| \(F_{B,x}\) [eV/Å] | **−0.040341072868572643** |
| \(F_{*,y}\), \(F_{*,z}\) | 0 |
| \(\mu_{A,x}\) [e·Å] | 0.082882545215708481 |
| \(\mu_{B,x}\) [e·Å] | 0.12342697961661356 |
| SCC iterations | 8 (tol \(10^{-12}\) e·Å) |

**Tol:** \(|E - E_{\mathrm{pin}}| < 10^{-9}\) eV;
\(|F_x - F_{\mathrm{pin}}| < 10^{-8}\) eV/Å; transverse \(|F| < 10^{-12}\).

These are the values a two-site Applequist SCC with van Duijnen/Swart
exponential λ gives in closed form:
\(u = a r/(\alpha_A\alpha_B)^{1/6}\), \(\lambda_3 = 1-e^{-u^3}\),
\(\lambda_5 = 1-(1+u^3)e^{-u^3}\); \(E^{\mathrm{perm}} = \lambda_3/r^2\),
\(K = 3\lambda_5/r^3 - \lambda_3/r^3\),
\(\mu_A = \alpha_A(E+K\alpha_B E)/(1-\alpha_A\alpha_B K^2)\), and
\(E_{\mathrm{pol}} = -\tfrac12 k(\mu_A+\mu_B)E\). Re-derived independently
from the paper formulae (not from Atomiverse output) on 2026-07-29 and
reproducing every digit above. **Kept as a literature pin, not an oracle pin.**

**API:** `cpu::Evaluate` + `cpu::PimInducedDipole{20.0, {0,1.0,1.5}, thole_a}`.

## Provenance / what changed

- 2026-07-24 (wave 1): pinned offline from published formulae because no
  MetalWalls binary existed on the host. `oracle_version` claimed the
  MetalWalls JOSS paper for a `thole_a = 0.39` number.
- 2026-07-29 (this re-pin, real binary): **that attribution was wrong.**
  MetalWalls has no Thole damping in any released version, so no MetalWalls
  run can produce −0.12040946 eV. The physics claim (`E_pol = -½Σμ·E^perm`,
  Applequist mutual induction, Hellmann–Feynman charge–dipole + dipole–dipole
  force) was nonetheless correct, and is now pinned against the binary in the
  configuration MetalWalls *can* express. **No Atomiverse code was changed and
  no tolerance was widened**; the Thole pin's values are unchanged, only its
  provenance label.

References:
- MetalWalls git **c8ad175d** — `src/coulomb_sr_pim*.{f90,inc}`,
  `src/ions_dipoles.F90`, `src/damping.f90`.
- Marin-Laflèche *et al.*, *J. Open Source Softw.* **5**, 2373 (2020) — MetalWalls.
- Thole, *Chem. Phys.* **59**, 341 (1981); van Duijnen & Swart,
  *J. Phys. Chem. A* **102**, 2399 (1998) — exponential damping (gate B only).
- Applequist *et al.*, *J. Am. Chem. Soc.* **94**, 2952 (1972) — isotropic SCC.

## Fair wall (perf gate)

The two-site case is far below any timer resolution on both sides, so the wall
twin is a **512-ion disordered rocksalt PIM single point** with the same
species as this regression (q = ±1, α = 1.0 / 1.5 Å³, L = 22.56 Å). Single
rank, single thread, best of 7 cold runs:

| | wall | iterations |
| --- | --- | --- |
| MetalWalls c8ad175d (`Melt dipoles computation` internal timer) | 8.0124e-02 s | 17 (preconditioned CG, Ewald) |
| Atomiverse `PimInducedDipole` launch | 6.17e-02 s | 35 (Jacobi, cutoff O(N²)) |

**ratio_atv_over_mw = 0.77** (gate ≤ 10).

Not an equal-accuracy benchmark: the two E_pol differ by 3.0 % because
Atomiverse truncates at 11 Å with no reciprocal-space dipole path while
MetalWalls does the full Ewald sum. 0.77 is therefore a *lower bound* on the
cost of real parity. Method, generators and the full caveat:
`tools/metalwalls_oracle/pin_notes_pim-induced-dipole-bulk.md`.
