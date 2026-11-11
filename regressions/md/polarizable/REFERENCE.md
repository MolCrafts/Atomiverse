# md/polarizable

One case; every section below is run by `atv_regressions --case md/polarizable` and keeps its own oracle record.


## Reference — bulk PIM self-consistent induced dipoles (E4)

oracle: metalwalls
oracle_version: **MetalWalls git c8ad175d** (release branch, 2026-06-18), run
locally as `mpirun -n 1 mw`, `num_steps 0`
verdict: **repinned** (2026-07-29) — the wave-1 pin was a *Thole* number and
MetalWalls implements no Thole damping; a real-MetalWalls gate was added and
the literature pin was re-labelled. Details in **Provenance / what changed**.
perf_exempt: false

### Physics

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

#### Where MetalWalls and Atomiverse agree exactly, and where they cannot

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

### Geometry (both gates)

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

### Raw MetalWalls output (unconverted, atomic units)

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

#### Derived MetalWalls polarization quantities

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

#### Extrapolation to the isolated limit

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

#### Measured agreement, and the one residual

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

### Pin values (hard-coded in the regression)

#### Gate A — MetalWalls c8ad175d oracle (`thole_a = 0`)

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

#### Gate B — Thole/Applequist closed form (`thole_a = 0.39`), **not** MetalWalls

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

### Provenance / what changed

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

### Fair wall (perf gate)

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


## Reference — bulk polarizable IL under NPT (PIM product twin, map §9)

oracle: literature (Thole/Applequist PIM), **MetalWalls-form-validated**  
oracle_version: pin values from the published PIM/Thole equations, offline
numpy — `tools/metalwalls_oracle/inputs/pim_npt_polarizable_il_pin.py`,
numpy 2.5.1. Charge–dipole kernel and polarization-energy convention
cross-validated against the **real MetalWalls binary, git c8ad175d**
(release branch 2026-06-18) — see *MetalWalls re-pin* below.
Thole, *Chem. Phys.* **59**, 341 (1981) exponential damping;
Applequist induced-dipole SCF. NPT integrator: MTK (Tuckerman,
*J. Phys. A* **39**, 5629 (2006); Shinoda, *Phys. Rev. B* **69**, 134103
(2004)), LAMMPS `fix npt iso` convention.  
mw_expressible: **false** — see *Why MetalWalls cannot express this pin*.  
perf_exempt: false — wall-clock twin measured, see *Performance*.

### What this regression pins

The **PIM-NPT product twin**: the polarizable-ion-model virial must reach
the shared stress meta (`kStress`) every barostat reads, so a bulk
polarizable ionic melt barostats on the **full** pressure including the
polarization contribution
\(W=\tfrac12\sum_{i\neq j}\mathbf r_{ij}\otimes\mathbf f_{ij}\)
(Hellmann–Feynman pair forces of the Thole-damped induced-dipole form,
Voigt, eV, LAMMPS sign).

### Geometry — deterministic toy polarizable IL

4×4×4 simple-cubic rock-salt lattice, d = 2.35 Å (compressed start), cubic
box L = 9.4 Å, N = 64: +1 cations (type 0, α = 0) / −1 anions (type 1,
α = 3.0 ų) alternating as (−1)^(ix+iy+iz); deterministic displacement
`disp[a] = 0.15·sin(1.7·(i+1) + 2.3·a)` Å breaks centro-symmetry (a perfect
rock-salt lattice has E_pol ≡ 0). PIM: **real-space cutoff 4.0 Å,
minimum image**, thole_a = 0.39, tt_b = 0. k = 14.399645 eV·Å/e².

### Pin values (hard-coded; offline independent numpy implementation)

| Observable | Value | Units | Tol |
| --- | --- | --- | --- |
| E_pol | −8.695749870555906 | eV | 1e-8 |
| W_xx | −10.66184373345355 | eV | 1e-7 |
| W_yy | −12.125914211727187 | eV | 1e-7 |
| W_zz | −9.657852793291474 | eV | 1e-7 |
| W_xy | −0.2325271422922529 | eV | 1e-7 |
| W_xz | −2.2469237837238 | eV | 1e-7 |
| W_yz | +1.375774440282161 | eV | 1e-7 |
| P_pol = tr(W)/(3V) | −0.013021203851134545 | eV/ų | 1e-10 |

These are literature-formula evaluations (published Applequist/Thole
induced-dipole equations, independent implementation) of the **truncated**
Hamiltonian `cpu::PimInducedDipole` implements — **not** a golden dump of
Atomiverse output, and **not** what MetalWalls computes for this geometry
(see below).

---

### MetalWalls re-pin (binary git c8ad175d, 2026-07-29)

The waves-1–5 pin was written when no MetalWalls binary existed and cited
"MetalWalls JOSS **5**, 2373 (2020) bulk PIM + Thole damping" as its
provenance. With the binary installed, both halves of that claim were
checked. **The Thole half was wrong** (MetalWalls has no Thole damping) and
the numbers are **not** MetalWalls numbers. Corrected here.

MW invocation (all cases): `mpirun -n 1 mw` with `num_steps 0`, `num_pbc 3`,
explicit zero velocities in `data.inpt` (so kinetic energy and kinetic
stress vanish and the polarizable − non-polarizable difference is purely
configurational). Inputs are generated by
`tools/metalwalls_oracle/inputs/metalwalls/pim-npt-polarizable-il/gen_mw_twin.py`
and live beside it.

Unit conversions used, applied explicitly (MW is atomic units throughout):

```
1 Bohr    = 0.529177210903 A          alpha: 3.0 A^3 = 20.2450034838011 Bohr^3
1 Hartree = 27.211386245988 eV        L:     9.4 A   = 17.7634255714822 Bohr
1 Ha/Bohr^3 (pressure) x V[Bohr^3] x 27.211386245988 = virial W [eV]
```

#### 1. Form check — ATV's charge–dipole kernel IS MetalWalls' (PASS)

Case `dimer_L<NN>/`: one +1 cation (α = 0) and one −1 anion (α = 3.0 ų) at
d = 2.35 Å in a cubic box of side NN Bohr, `pol` vs `nopol`. Only one site
is polarizable ⇒ no dipole–dipole term, no damping of any kind ⇒ MW and ATV
evaluate literally the same kernel, and as L grows the Ewald images die as
1/L³ onto the exact isolated-pair value
\(E_{\rm pol}=-\tfrac12 k\alpha/d^4 = -0.708225558187103\) eV.

Raw MW `energies_breakdown.out` potential energies [Ha], and the difference:

| L [Bohr] | E_pot(pol) | E_pot(nopol) | ΔE [Ha] | ΔE [eV] | residual [Ha] |
| --- | --- | --- | --- | --- | --- |
| 30 | −2.521198744858342E-01 | −2.267615435503378E-01 | −0.025358330935496 | −0.690035337639 | +6.685e-04 |
| 45 | −2.514772905267299E-01 | −2.256416228860870E-01 | −0.025835667640643 | −0.703024331093 | +1.911e-04 |
| 60 | −2.513217982578419E-01 | −2.253745703150123E-01 | −0.025947227942830 | −0.706060041565 | +7.958e-05 |
| 90 | −2.512421127227655E-01 | −2.252386551210688E-01 | −0.026003457601697 | −0.707590128531 | +2.335e-05 |

The residual falls exactly as 1/L³ (observed ratios 0.286 / 0.416 / 0.293 vs
(L₁/L₂)³ = 0.296 / 0.422 / 0.296). Richardson extrapolation in 1/L³ on
L = 60, 90 gives

```
E_pol^MW(L -> inf) = -0.026027133247535 Ha = -0.708234375674 eV
exact               = -0.026026809210851 Ha = -0.708225558187 eV
relative deviation  = 1.2e-05   (the two-point extrapolation residual)
```

**Conclusion: ATV's induced-dipole convention is confirmed against real
MetalWalls** — μᵢ = αᵢE_i^tot, E_pol = −½ Σ μᵢ·E_i^perm (which is the total
polarization energy *including* the Σμ²/2α self term at self-consistency),
and k = e²/4πε₀. Sign, factor of ½ and prefactor are all right.

#### 2. The pinned lattice number — MW disagrees by 5.17×

Case `lattice64/`: the **exact** regression start configuration (all 64
positions, charges, α, box), MW's own Hamiltonian (Ewald, undamped).

```
raw MW energies_breakdown.out (step 0, potential energy column, Hartree)
  pol    -1.266373589282987E+001
  nopol  -1.260191731612654E+001
  ΔE     -6.181857670332924E-002 Ha  x 27.211386245988  =  -1.682169167852 eV

raw MW stress_tensor.out (step 0, Ha/Bohr^3, Voigt xx yy zz xy xz yz)
  pol    -7.742498448544866E-04 -7.616120787429479E-04 -7.649262962239600E-04
         -1.658541076225775E-05 -1.089946648650244E-05 -1.288240192104498E-05
  nopol  -7.418989467597009E-04 -7.620790870295351E-04 -7.443333729899221E-04
         -8.461396735889022E-06 -7.068378590011610E-06 -5.716127366761310E-06
  Δσ     -3.235090E-05 +4.670083E-07 -2.059292E-05
         -8.124014E-06 -3.831088E-06 -7.166275E-06
  W = Δσ · V (V = 5605.0586578632 Bohr^3) · 27.211386245988 [eV]
       -4.934204789  +0.071228765  -3.140861813
       -1.239086123  -0.584322951  -1.093010343
  P_pol = tr(W)/(3V) = -3.212132602e-03 eV/A^3
```

| Observable | old pin (kept) | real MW, same geometry | rel. discrepancy |
| --- | --- | --- | --- |
| E_pol [eV] | −8.695749870555906 | −1.682169167852 | **4.169** (ratio 5.169) |
| W_xx [eV] | −10.661843733 | −4.934204789 | 1.161 |
| W_yy [eV] | −12.125914212 | +0.071228765 | — (sign differs) |
| W_zz [eV] | −9.657852793 | −3.140861813 | 2.075 |
| P_pol [eV/ų] | −1.3021203851e-02 | −3.2121326023e-03 | 3.054 |

The gap is **not** an ATV coding error — §1 proves the kernel matches MW.
It is the real-space truncation. The same numpy oracle, same geometry, only
the cutoff and Thole switch varied:

| oracle setting | E_pol [eV] |
| --- | --- |
| rc = 4.0 Å, thole_a = 0.39 (**the pin**) | −8.695749870556 |
| rc = 4.0 Å, thole_a = 0 | −8.771469108688 |
| rc = 4.7 Å = L/2 (max min-image), thole_a = 0 | −4.892485851954 |
| rc = 4.7 Å, thole_a = 0.39 | −2.668325687238 |
| MetalWalls, full Ewald, undamped | −1.682169167852 |

Thole damping moves the rc = 4.0 Å number by only 0.9 %. The **cutoff**
moves it by 1.8× between 4.0 Å and 4.7 Å, and the reciprocal-space tail
supplies the remaining ~2.9×. In a rock-salt lattice the charge–dipole
field is a cancelling alternating sum: truncating it at any finite radius
leaves an unbalanced shell, so the truncated E_pol is not a converged
physical quantity. `cpu::PimInducedDipole` documents itself as a
real-space-cutoff foothold (`cutoff` is a hard multipole truncation, not an
Ewald splitting radius) and there is **no long-range path for the induced
dipole field** — pairing it with `CoulLong`/PME extends only the permanent
charge–charge term. That capability gap is real and is reported upstream;
it is out of scope for this slug and is *not* a defect in the pinned math.

#### Why MetalWalls cannot express this pin

Verdict **not_expressible**: the pinned observables belong to a Hamiltonian
MetalWalls has no way to write down. Two independent, specific gaps:

1. **MetalWalls has no Thole damping.** `grep -ri thole` over the whole
   MetalWalls source tree and `doc/` returns **zero** hits. Its only
   induced-dipole damping is Tang–Toennies on the **charge–dipole** term:
   `src/damping.f90` stores per-ordered-pair (b, order n, intensity c) from
   `tt_pair`, and `src/coulomb_sr_pim_3DPBC.inc:514–536` applies
   `dampfunc = −f_n(b·r)·e^{−b·r}·c / r³` to the *charge* field only. The
   **dipole–dipole** tensor carries no short-range damping at all. The pin's
   λ₃/λ₅/λ₇ = 1−e^{−u³}, 1−(1+u³)e^{−u³}, 1−(1+u³+0.6u⁶)e^{−u³} with
   u = a·r/(αᵢαⱼ)^{1/6} is the Thole / van Duijnen–Swart *atomic
   polarizability* form, which is simply absent from MetalWalls.
2. **MetalWalls has no truncated-Coulomb mode.** It supports `num_pbc` 2 or
   3 only (the kernels exist solely as `*_2DPBC.inc` / `*_3DPBC.inc`; there
   is no 0-D/cluster path), and electrostatics is *always* Ewald-split:
   `coulomb_rcut` is the real-space splitting radius of erfc(αr)/r, paired
   with the reciprocal sum, not a truncation of the physical interaction.
   There is no setting under which MW evaluates a bare minimum-image
   charge–dipole/dipole–dipole sum cut at 4.0 Å.

Accordingly the pinned numbers are **kept unchanged** (they are the correct
evaluation of the model the code implements, they are what the regression's
product invariant needs, and ATV reproduces them to < 1e-14 eV), the
tolerances are **kept unchanged**, and the oracle label is corrected from
`metalwalls` to `literature (Thole/Applequist), MetalWalls-form-validated`.

#### Bug found in MetalWalls (reported for the record)

A PIM run whose `runtime.inpt` omits the `damping` section **segfaults**:
`system%tt%b / %order / %c` are allocated only inside
`read_damping_parameters` (`src/configuration.F90:3264`), but
`src/coulomb_sr_pim_3DPBC.inc:514` dereferences them unconditionally. The
twins above therefore ship an explicit `damping` block with intensity
`c = 0` (identity damping). Separately, `tt_pair … 0 …` divides by zero at
`src/configuration.F90:3307` (`tt_order/int(tt_order)`), so the order field
is set to 4 with `c = 0`.

### Performance

Wall-clock twin `npt250/run/`: the **same** 64-ion box as the regression's
NPT product leg (d = 2.9 Å, L = 11.6 Å), 250 steps, dt = 1 fs, T = 800 K,
τ_T = 100 fs (4134.1 a.u.), P = 0, τ_P = 500 fs (20670.5 a.u.), LJ
(0.01 eV → 0.964853 kJ/mol, σ = 3.0 Å, rc = 4.0 Å), MW at the production
Ewald/dipole tolerances its own `example/NaCl_PIM_NPT` ships
(rtol 1.63e-5, ktol 1e-7, CG dipoles 1e-7). `mpirun -n 1` vs ATV single
rank, `OMP_NUM_THREADS=1`, same node.

| | wall clock | MD loop only |
| --- | --- | --- |
| MetalWalls (best of 3) | 0.83 s | 0.330 s |
| ATV `reg_pim-npt-polarizable-il` (best of 5, all 3 legs) | 0.06 s | ≈0.055 s |
| **ratio ATV/MW** | **0.072** | **0.17** |

Caveat, stated because the number flatters ATV: MW's per step work is a
full Ewald sum over charges, charge–dipole and dipole–dipole plus a CG
dipole solve; ATV's is a 4.0 Å-truncated minimum-image O(N²) Jacobi SCC
plus LJ + coul/cut. ATV is cheaper here largely because it is doing
strictly less electrostatics (§2), so this is a throughput datapoint, not a
like-for-like kernel benchmark.

### Legs

1. **PIM pin:** `Evaluate` + `PimInducedDipole` alone on the start config —
   E_pol, all six Voigt virial components, and P_pol vs the table above.
2. **Stress additivity:** trace(kStress) of the LJ+coul/cut+PIM stack minus
   the LJ+coul/cut stack equals the pinned PIM trace — the polarization
   pressure reaches the meta the barostat reads.
3. **Short NPT product run:** same lattice **stretched** to d = 2.9 Å
   (start tension P₀ ≈ −0.12 eV/ų — a stretched ionic melt contracts onto
   the stiff r⁻¹² wall, which is self-limiting; the compressed pin-leg
   density would expand ballistically through the soft side).
   `cpu::md::MD` + `PairLJCut{0.01 eV, 3.0 Å}` + `PairCoulCut` (both
   virial-correct) + `PimInducedDipole` +
   `NoseHooverNPT{T=800 K, τ_T=100 fs, P=0, τ_P=500 fs, dt=1 fs}` +
   `Virial` (kinetic stress) + `InitVelocity` + an out-of-library
   `TrackVolume` Criterion, 250 steps. Invariants: per-step SCC convergence
   (the component throws otherwise), finite MTK conserved quantity,
   contraction happened (V_min < 0.97·V₀), bounded
   (0.5·V₀ < V_min, V_max < 1.2·V₀).

Runtime ≪ 60 s (N = 64, O(N²) SCC per step).


## extended-lagrangian-dipoles

oracle: metalwalls
oracle_version: MetalWalls git **c8ad175d** for stationary PIM (thole_a=0);
iEL/SCF dynamics are literature-form (Albaugh 2015 / Niklasson 2009)
oracle_kind: real binary for Gate A; dynamics gates are scheme self-consistency

### Gate A — stationary XL fixed point = SCF PIM (MetalWalls)

Same twin as `pim-induced-dipole-bulk` (two-site, L=50 Å, α=1.0/1.5 Å³,
`thole_a=0` — MW has no Thole):

| Observable | MetalWalls c8ad175d |
|------------|---------------------|
| E_pol | **−2.4397739572766×10⁻¹ eV** (−8.9660039191731e-3 Ha) |
| F_A,x = −F_B,x | **+3.4926913376021×10⁻¹ eV/Å** |

Cold and warm XL launches must reproduce these (fixed point of the XL
recursion). Supersedes the wave-5 Thole `0.39` number mis-attributed to MW.

### Gate B — dynamics (XL tracks BO twin)

On 8-ion bulk toy, 150 × 0.5 fs: μ_XL tracks cold SCF to <1e-6 e·Å; warm
XL needs fewer SCC iters; energy drift bounded. Uses `thole_a=0.39` as a
shared XL/SCF damping (not a MW claim).

### Perf

`perf_exempt` until a fair mass-zero/XL wall twin is timed.


## qeq-fluctuating-charge

oracle: lammps
oracle_version: LAMMPS 30 Mar 2026 development (patch_30Mar2026-1074-g7f680de296), QEQ package

### What is pinned

QEq / fluctuating-charge equilibration (`cpu::md::QEq`, the LAMMPS
`fix qeq/point` twin) on a formaldehyde-like H₂CO 4-atom organic toy:
per-atom equilibrated charges, the QEq objective, and the `coul/cut`
Coulomb energy of the equilibrated system. Units: eV, Å, e (`units metal`).

Oracle run offline 2026-07-27 (`lmp -in in.qeq_point`; inputs preserved at
`tools/metalwalls_oracle/inputs/lammps/qeq-fluctuating-charge/`, procedure in
`tools/metalwalls_oracle/pin_notes_qeq-fluctuating-charge.md`). ctest never
launches LAMMPS.

### System

`boundary f f f`, 40 Å cube; C (20, 20, 20), O (20, 20, 21.205),
H (20.943, 20, 19.412), H (19.057, 20, 19.412).
QEq params (χ [eV], η [eV/e²]): C 5.343/10.126, O 8.741/13.364,
H 4.528/13.8904. `fix qeq/point 1 10.0 1.0e-10 500 param.qeq`,
`pair_style coul/cut 10.0`, `run 0`.

### Pinned values (hard-coded in the regression, tol 1e-6)

| quantity | LAMMPS pin |
|---|---|
| q(C) | +0.046279970820 e |
| q(O) | −0.230795571743 e |
| q(H) ×2 | +0.092257800461 e |
| Σq | 0 (two-vector constraint, tol 1e-12) |
| E_coul (`coul/cut`, qqr2e = 14.399645) | −0.254699311955 eV |
| E_qeq = Σχq + ½qᵀ(ηδ + 1/r)q (derived from pinned q; bare-1/r kernel) | −0.467311783767 eV |

Kernel identity verified against LAMMPS source `src/QEQ/fix_qeq_point.cpp`:
off-diagonal bare `1/r` (no qqrd2e), diagonal η, RHS −χ, constraint
`q = s − (Σs/Σt)t`.

### Fair perf twin

`in.qeq_perf`: 1000 repeated `fix qeq/point` solves (dt = 0), LAMMPS
`Loop time of 0.000610294 s` (1 proc). The regression times 1000
`QEq::launch` equilibrations (dense rebuild + Cholesky each launch) on the
same geometry and prints `T_atv`, `T_lmp`, and the ratio to stdout — reported,
not asserted (micro-workload; both sides are sub-ms).

Measured 2026-07-27 (x86_64 epyc login node, 1 thread):
`T_atv = 0.000275 s`, `T_lmp = 0.000610 s` ⇒ **ratio 0.45** (ATV within 10×;
dense factor+two substitutions beats warm-started CG at N = 4).


## maze-molten-salt-al

oracle: metalwalls
oracle_version: MetalWalls git **c8ad175d** product scenario (MaZe + Fumi–Tosi
footholds closed wave-4/5; this slug is the MW molten-salt + Al wall composition)
oracle_kind: product composition gates; binary available for future absolute pins

### Physics

Map §9 MW example: **molten salt between Al electrodes** with MaZe charge
dynamics. ATV composition:

- `cpu::md::ConstPotMaZe` (symplectic) on Al wall types
- `cpu::PairBorn` multi-type Fumi–Tosi NaCl + soft Al–ion Born exclusion
- `cpu::PairCoulCut` + freeze + short NVT at 800 K

### Geometry

| | value |
|---|---|
| box | 12 × 12 × 24 Å |
| Al walls | 3×3 per plate at z = 3 / 21 Å, Ψ = 0 / 1 V, η = 1.5 Å⁻¹ |
| salt | 4 Na⁺ + 4 Cl⁻ mid-slab |
| MD | 160 × 0.5 fs NVT, freeze walls |

### Assertions

| check | bound |
|-------|-------|
| polarity, step 1 | Q_L < 0 < Q_R |
| BO tracking, end | \|Q_L(MaZe) − Q_L(dense ConstPotBO, same configuration)\| ≤ 1e-6 e |
| field-driven polarity, end | Q_L(Ψ) − Q_L(Ψ = 0) < 0 on the end configuration |
| neutrality | \|sum Q\| ≤ 1e-5 during NVT |
| freeze | max \|Δx_wall\| ≤ 1e-10 Å |
| ion separation proxy | min ion–ion r > 1 Å |
| finite thermo | E, K finite |

The end state is not tested by the sign of the total charge: in this 8-ion
cell the ions' induced charge (~0.1 e) can exceed the 1 V response (~0.04 e),
so one snapshot's sign depends on where the ions happen to be. The
potential-driven part is deterministic (the BO solve is linear in Ψ).

### Perf

`perf_exempt` — product scenario, not a kernel microbench.


## maze-symplectic-charge-dynamics

oracle: metalwalls
oracle_version: **MetalWalls, git c8ad175d** (release branch, 2026-06-18), run
locally as `/nobackup/proj/disk/teoroo/personal/jicli594/work/opt/metalwalls/mw`,
`mpirun -n 1`. Supporting theory: Coretti et al., J. Chem. Phys. **152**,
194701 (2020), DOI 10.1063/5.0007192; Marin-Laflèche et al., JOSS **5**, 2373
(2020), DOI 10.21105/joss.02373.

Waves 1–5 pinned this case from the Coretti/JOSS **formulas** because no
MetalWalls binary existed on this host. The binary now exists; every pin below
has been re-checked against it. Verdict: **confirmed** — MetalWalls satisfies
every invariant this regression asserts, and reproduces Atomiverse's electrode
charges to **1.1e-6 relative** in the one configuration where the two codes
evaluate the same electrostatic model. One new pin (a real MW number) was
added; nothing was loosened.

### Unit bridge (stated explicitly — MW is atomic units, ATV is metal units)

| Quantity | MetalWalls | Atomiverse | Conversion |
| --- | --- | --- | --- |
| length | Bohr | Å | 1 Bohr = 0.529177210903 Å |
| energy | Hartree | eV | 1 Ha = 27.211386245988 eV |
| charge | e | e | identity |
| potential | Eh·e⁻¹ | V | 1 Eh/e = 27.211386245988 V |
| force | Ha/Bohr | eV/Å | 27.211386245988 / 0.529177210903 = 51.4220675 eV/Å |
| time | ħ/Eh | fs | 1 fs = 41.341373335182 a.u. |
| Gaussian width η | Bohr⁻¹ | Å⁻¹ | η[Bohr⁻¹] = η[Å⁻¹] × 0.529177210903 |

Input values actually written into MetalWalls (from `gen_inputs.py`):
η = 2.0 Å⁻¹ → **1.058354421806 Bohr⁻¹**; Ψ_top = 1 V → **0.036749322175655 Eh/e**;
dt = 0.25 fs → **10.335343333795 a.u.**; box 6 × 6 × 20 Å →
11.33835674775462 × 11.33835674775462 × 37.79452249251541 Bohr.
Dimensional check: 1 V is far below 1 Eh/e = 27.2 V, so the a.u. potential must
be ≪ 1 — it is (0.0367). η is an inverse length, so going Å⁻¹ → Bohr⁻¹
*shrinks* the number (Bohr < Å) — it does (2.0 → 1.058). MetalWalls' own
`bohr2angstrom` (`src/constants.F90`) is 0.52917721067, 4.4e-10 relative from
the value above — five orders below any tolerance here.

### Geometry

Parallel-plate capacitor as an isolated 10-atom cluster in an orthorhombic
30 × 30 × 40 Å cell, `PairGaussianCoul` cutoff 14 Å (the MetalWalls twins
below were run on the periodic 6 × 6 × 20 Å cell this cluster was cut from):

| Region | type | N | z [Å] | η [1/Å] | Ψ [V] | mass [amu] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bottom plate | 1 | 4 | 4 | 2.0 | 0.0 | static |
| Top plate | 2 | 4 | 16 | 2.0 | +1.0 | static |
| Ion + (+0.5 e) | 3 | 1 | 8.5 | point | — | 40.0 |
| Ion − (−0.5 e) | 3 | 1 | 11.5 | point | — | 40.0 |

Plate sites at xy ∈ {1.5, 4.5}², ions at (2.1, 2.1, 8.5) and (4.2, 4.2, 11.5) Å,
zero initial velocity. Explicit NVE velocity Verlet, dt = 0.25 fs, 400 steps
forward, velocity flip, 400 steps back (palindrome).

### Which electrostatic model the dynamics case runs

Pair styles sum explicit periodic image rows within their cutoff (LAMMPS
parity), while `ConstPotBO` builds the elastance with `xy_image_range = 0`,
one nearest image per pair and no reciprocal space. The two describe the same
model only when every pair interacts exactly once: the cutoff (14 Å) exceeds
the cluster diameter (12.73 Å, opposite plate corners) and stays below half of
every cell edge (30 / 40 Å), so no image is inside it. That is the **isolated
pairwise Gaussian-Coulomb** model — the one MetalWalls' Ewald converges to in
the dilute twin below (1.1e-6 relative at L = 200 Å), and the dynamics case's
first-solve Q_top (+0.126708 e) is that value.

On the periodic 6 × 6 × 20 Å cell a 12 Å cutoff would count up to two images
per direction while the elastance counts one, and pairs at exactly 2L sit on
the cutoff, so that cell is not used for the dynamics. MetalWalls' 3D-Ewald
answer for it is kept here as context:

| Q on the +1 V plate, 6 × 6 × 20 Å cell | value [e] |
| --- | ---: |
| Minimum-image truncation of that cell (not run) | +0.1358218636749 |
| MetalWalls c8ad175d, 3D Ewald (`twinA_ewald_6x6x20`) | **+0.1529601174306** |

### The kernel-matched twin — where MW and ATV compute the same thing

Take the identical 10-atom cluster and grow the cell. MetalWalls' Ewald then
converges to the **isolated pairwise Gaussian-Coulomb sum**, which is exactly
what the minimum-image path computes once the cell exceeds twice the cluster
extent. Atomiverse's answer is box-independent there (bit-identical at
L = 40, 60, 200 Å); MetalWalls approaches it monotonically as its residual
periodic images die off:

| MW cell L [Å] | MW Q_top [e] | rel Δ vs ATV | MW grand energy [eV] | rel Δ | max ion-force Δ [eV/Å] |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 40 | +0.1267873919608 | +6.301e-4 | −0.9840388431722 | +9.328e-4 | 4.38e-4 |
| 60 | +0.1267203095694 | +1.006e-4 | −0.9833929872312 | +2.759e-4 | 1.30e-4 |
| 100 | +0.1267091472474 | +1.255e-5 | −0.9831803436266 | +5.959e-5 | 2.82e-5 |
| **200** | **+0.1267076937357** | **+1.076e-6** | **−0.9831291093381** | **+7.473e-6** | **3.53e-6** |
| Atomiverse (isolated) | **+0.1267075573679** | — | **−0.9831217623464** | — | — |

Raw MetalWalls output, `twinB_dilute_200/charges.out` (already in e, no
conversion — MW writes "1 au = 1 e"):

```
## step        0
-4.234061692847945E-002      <- bottom plate, 4 sites
-3.112908052013959E-002
-3.112908052014043E-002
-2.210891576695224E-002
+2.118516355288748E-002      <- top plate (+1 V), 4 sites
+3.065933680220784E-002
+3.065933680220688E-002
+4.420385657840947E-002
```
Σ(top 4) = **+0.1267076937357 e**; Σ(all 8) = 0 to the last digit
(`charge_neutrality true electrodes`).

Raw `twinB_dilute_200/energies.out` (Hartree):

```
## step        Hamiltonian             ...            Potential energy
       0 -3.612932837933060E-002 ...            -3.612932837933060E-002
```
−3.612932837933060e-2 Ha × 27.211386245988 eV/Ha = **−0.9831291093381 eV**.
Atomiverse's grand energy for the same cluster,
E = U_pair − Σ Ψ_i Q_i = −0.8564142049784 − 1.0 × 0.1267075573679 =
**−0.9831217623464 eV** — MW's remaining +7.5e-6 relative gap is its own image
tail, which extrapolates (1/L³ Richardson on L = 100/200) to −0.9831214 eV,
4e-7 from Atomiverse.

### Hard-coded pins (asserted in the regression source)

| Pin | Value | Source | Checked against real MW |
| --- | --- | --- | --- |
| Dilute-limit Q_top | +0.1267076937357 e, rel ≤ 1e-4 | **MetalWalls c8ad175d, `twinB_dilute_200`** | measured 1.076e-6 |
| Constraint residual ‖σ‖∞, every step | ≤ 1e-10 V | Coretti 2020 §II.C | MW: 2.83e-16 V ✓ |
| Electrode electroneutrality \|ΣQ\|, every step | ≤ 1e-9 e | KKT constraint | MW: exact ✓ |
| Palindrome ion-position return (MaZe) | ≤ 1e-6 Å | Coretti 2020 time-reversibility | MW: 2.29e-7 Å ✓ |
| Palindrome electrode-charge return (MaZe) | ≤ 1e-6 e | Coretti 2020 time-reversibility | MW: 5.95e-10 e ✓ |
| MaZe return error < 0.1 × loose CG+ASPC twin | ratio | reversible vs dissipative | MW 2.29e-7 vs 1.07e-5 ✓ |
| NVE grand-energy drift (MaZe, after 5-step settle) | ≤ 1e-6 eV | Coretti 2020 no-secular-drift | MW: 6.76e-7 eV ✓ |
| MaZe drift < loose (cg_tol 1e-3) CG+ASPC twin drift | ratio | Coretti 2020 vs loose solve | — (ATV-internal twin) |
| Q_top > 0 at ΔΨ = +1 V | polarity | electrostatics sanity | MW: +0.15296 e ✓ |
| Ion min separation > 1.8 Å | validity | bound smooth trajectory | — (ATV-internal) |

Units: eV, Å, fs, amu, V, e (Atomiverse public surface).

#### How each invariant was verified against the binary

* **Constraint at solver precision.** `twinA_ewald_6x6x20_md/run.out` prints,
  every step, `|maze| Maximum of the constraints: 1.04083E-17` (atomic units,
  Eh/e; the single-solve `twinA_ewald_6x6x20` case reports 5.89806E-17) —
  × 27.211386245988 V per Eh/e = **2.83e-16 V** (resp. 1.61e-15 V), five orders
  inside the 1e-10 V pin, and `|maze| number of iterations: 0` because
  `maze inversion` reaches the linear constraint in one exact solve. MetalWalls also confirms MaZe ≡
  Born–Oppenheimer: `twinB_dilute_60` (`maze inversion`) and
  `twinB_dilute_60_inv` (`matrix_inversion`) agree to 1e-17 e per site.
* **No secular drift.** `twinA_ewald_6x6x20_md`, 800 MaZe NVE steps at
  dt = 0.25 fs on the regression geometry: max |H − H(step 5)| = 2.4834e-8 Ha
  = **6.758e-7 eV**, linear slope 1.863e-11 Ha/step = 5.07e-10 eV/step
  (4.06e-7 eV over the whole run). Inside the 1e-6 eV pin. Atomiverse's
  7.57e-8 eV is smaller because its truncated forces move the ions less
  (MW's peak KE reaches 0.778 eV on the same budget); the pinned *bound* is
  what both codes must respect, and both do.
* **Time reversibility.** `twinA_pal_fwd` runs 400 steps and dumps
  `restart.000000400.out`; the velocity block is negated in place and rerun as
  `twinA_pal_back` for 400 more. Ion displacement on the forward leg is
  0.2854 Å; the palindrome returns to **2.288e-7 Å** and **5.95e-10 e**. Both
  inside the 1e-6 pins, and 47× better than the loose CG+ASPC twin's 1.07e-5 Å.
  MetalWalls' figure is an *upper bound*: its restart file is text with 15
  significant digits, so the round trip seeds ~1e-14 Bohr which this locally
  unstable ion pair amplifies over 400 steps. Atomiverse's in-memory flip has
  no such floor, hence 1.4e-14 Å.

### Reproducing

Inputs: `tools/metalwalls_oracle/inputs/metalwalls/maze-symplectic-charge-dynamics/`
(`gen_inputs.py` writes every case; `atv_twin.cpp` is the matching
out-of-library Atomiverse consumer used for the dilute comparison).

```bash
python3 gen_inputs.py
bash -lc 'module load buildenv-gcc/2026.03-mpich
          export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
          cd twinB_dilute_200 && mpirun -n 1 .../metalwalls/mw > run.stdout 2>&1'
```

Note: `electrode_charges maze <alg> <tol>` needs at least two `# charges`
sections in `data.inpt` (its Verlet history) — zeros are fine, the constraint
is linear so `inversion` lands on it from any guess. MetalWalls' random
velocity initializer divides by the instantaneous temperature, so a zero-T
start must supply an explicit all-zero `# velocities` block or MW aborts with
"Temperature is NaN or meaningless". At r_cut ≤ 2.5 Å in this cell MW's MaZe
cannot reach a 1e-14 tolerance ("M-SHAKE failed…"); use `matrix_inversion`
there, or keep r_cut 2.9 Å.

### Measured at re-pin time (2026-07-29, Release, GCC 14.3, x86_64 epyc)

| Metric | MaZe symplectic | loose (1e-3) CG+ASPC twin |
| --- | ---: | ---: |
| Grand-energy drift [eV] | 7.571e-08 | 1.755e-06 |
| Palindrome position return [Å] | 1.421e-14 | 1.068e-05 |
| Palindrome charge return [e] | 9.015e-13 | 5.092e-05 |
| Max constraint residual [V] | 9.990e-11 | — |
| Max \|ΣQ\| [e] | 5.551e-17 | 5.551e-17 |
| Q_top first solve, 6 × 6 × 20 Å twin [e] | +0.135822 | +0.135821 |
| Q_top dilute limit [e] | +0.1267075573679 | — |
| min ion separation [Å] | 3.798 | 3.798 |

### Runtime and the MetalWalls wall-clock comparison

Two 800-step NVE runs plus one dilute solve: **0.03 s** (≪ 60 s).

MetalWalls needs **7.6–9.0 s** for the single 800-step twin-A run
(`twinA_perf`, output every 800 steps, `mpirun -n 1`), against ≈ 0.015 s for
Atomiverse's 800-step MaZe leg — a raw ratio of ≈ 0.002. **That ratio is not
an engine benchmark and must not be quoted as one.** The two codes are not
doing the same work per step: MetalWalls converges a full 3D Ewald sum whose
k-mesh is forced very fine (the 6 Å lateral cell caps the real-space cutoff at
< 3 Å, pushing α up and the k-space mode count to O(10⁵)), while Atomiverse
evaluates 45 minimum-image pairs and no reciprocal space at all. The cost gap
is the model gap documented above, so this case carries **`perf_exempt` with a
technical reason**, not "no binary".
