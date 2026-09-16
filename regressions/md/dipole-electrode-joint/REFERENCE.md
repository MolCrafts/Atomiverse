# Reference — coupled PIM dipoles ⊕ electrode \(Q\) (E5)

oracle: metalwalls
oracle_version: MetalWalls git **c8ad175d** (release branch, 2026-06-18), binary
`/nobackup/proj/disk/teoroo/personal/jicli594/work/opt/metalwalls/mw`, run
`mpirun -n 1`. Twin inputs:
`tools/metalwalls_oracle/inputs/metalwalls/dipole-electrode-joint/{runtime,data}.inpt`
mw_expressible: **false** — see *Why MetalWalls cannot express this system*
perf_exempt: true
perf_exempt_reason: Measured, and the measurement is meaningless. MW twin
(`num_pbc 3`, `mpirun -n 1`): 0.52 / 0.73 / 0.67 s. Whole ATV regression
(`OMP_NUM_THREADS=1`, four joint solves + one MD step + thermo/timing report):
0.04 / 0.04 / 0.04 s. Both are ~100 % fixed overhead for a 4-atom system — MW's
is MPI startup plus Ewald table setup (a 16×16×33 k-grid and a 67-point
z-integral table built for 4 atoms), ATV's is process startup; the joint solve
itself is a 2×2 Cholesky costing microseconds. The two are also not solving the
same linear system (MW: preconditioned CG over
\(3N_{\text{ion}}+N_{\text{elec}}\) Ewald-coupled unknowns; ATV: direct
min-image 2×2). The resulting ~0.07 ratio measures startup, not the solver, and
must not be quoted as a speed result. A fair number needs an electrode-slab twin
(≥10³ atoms) that ATV cannot yet express (no Ewald in this component) — see
*Debt* below.

## Physics

Self-consistent joint of:

1. **Constant-potential BO** electrode charges \(Q\) with Gaussian elastance \(A\)
   (same kernels as `cpu::md::ConstPotBO`).
2. **PIM induced dipoles** \(\boldsymbol{\mu}\) on polarizable electrolyte sites
   with Thole damping (same kernels as `cpu::PimInducedDipole`).

Coupling:

\[
A Q = \Psi - \Phi^{\mathrm{perm}} - \Phi^{\mu},\quad
\Phi_i^{\mu} = k_e\sum_j\frac{\boldsymbol{\mu}_j\cdot\mathbf{r}_{ij}}{r_{ij}^3}
\]

\[
\boldsymbol{\mu}_i = \alpha_i\mathbf{E}_i^{\mathrm{tot}}(q_{\mathrm{all}},\boldsymbol{\mu})
\]

Outer alternating solve until \(\max|\Delta Q|\) and \(\max|\Delta\mu|\) below
`joint_tol`. **Oracle is MetalWalls** — LAMMPS has no SCF PIM ⊕ const-pot
joint twin (ELECTRODE is non-polarizable; Drude is a different model).

## Geometry

Orthorhombic \(L_x=L_y=10\) Å, \(L_z=20\) Å; \(\eta=2\) Å⁻¹; \(\Psi_{\mathrm{bot}}=0\),
\(\Psi_{\mathrm{top}}=1\) V; electroneutrality on; `xy_image_range=0` (min-image);
`thole_a=0.39`.

| Site | type | \(q\) [e] | \(\alpha\) [Å³] | \((x,y,z)\) [Å] |
| ---- | ---- | --------- | --------------- | --------------- |
| bot graphene wall | 1 | solved | 0 | (0, 0, 2) |
| top graphene wall | 2 | solved | 0 | (0, 0, 18) |
| IL cation | 3 | +1 | 1.0 | (0, 0, 8) |
| IL anion | 4 | −1 | 1.5 | (0, 0, 12) |

### Pin values (hard-coded in regression)

| Observable | Value |
| ---------- | ----- |
| \(Q_{\mathrm{bot}}\) [e] | **−7.3285008958883610e−02** |
| \(Q_{\mathrm{top}}\) [e] | **+7.3285008958883610e−02** |
| \(\mu_{\mathrm{cation},z}\) [e·Å] | **+5.8951916148990660e−02** |
| \(\mu_{\mathrm{anion},z}\) [e·Å] | **+8.7432701391015569e−02** |
| \(\mu_{*,x}\), \(\mu_{*,y}\) | 0 |
| \(\lvert\sum Q\rvert\) | 0 (within \(10^{-12}\)) |
| \(\alpha=0\) \(Q_{\mathrm{top}}\) [e] | **+7.5339627935767609e−02** |

**Tol (dual gate):**
relative \(|Q-Q_{\mathrm{pin}}|/|Q_{\mathrm{pin}}| \le 10^{-6}\);
\(|\mu_z-\mu_{\mathrm{pin}}| \le 10^{-8}\) e·Å; transverse \(|\mu|<10^{-10}\);
neutrality \(|\sum Q|\le 10^{-10}\) e.

These are **ATV's own minimum-image algebra**, not MetalWalls output. See the
re-pin section below for exactly what that means and what MetalWalls says.

**API:** `cpu::md::DipoleElectrodeJoint{2.0, {0,0,0,1.0,1.5}}` +
`add_electrode(1,0).add_electrode(2,1)`.

---

## Real-MetalWalls re-pin (2026-07-29)

Waves 1–5 pinned this case from published MetalWalls *formulas* because no MW
binary was installed. A verified binary now exists, so the MW twin was actually
built and run. This section records what came back. **Verdict: not_expressible**
— the pinned numbers are kept, the tolerances are kept, and the provenance claim
is corrected.

### The MW twin

`tools/metalwalls_oracle/inputs/metalwalls/dipole-electrode-joint/` reproduces the
regression's system exactly — same 4 sites, same \(z\), same permanent charges,
same masses, same \(\eta\), same box, same \(\Psi\) — converted Å/eV/V → Bohr/Ha/(Ha·e⁻¹)
with **MetalWalls' own** `bohr2angstrom = 0.52917721067` (`src/constants.F90`):

| quantity | regression (ATV units) | MW input (atomic units) |
| --- | --- | --- |
| \(L_x, L_y\) | 10 Å | 18.897261254578282 Bohr |
| \(L_z\) | 20 Å | 37.794522509156565 Bohr |
| \(z\) bot elec | 2 Å | 3.7794522509156563 Bohr |
| \(z\) cation | 8 Å | 15.117809003662625 Bohr |
| \(z\) anion | 12 Å | 22.676713505493936 Bohr |
| \(z\) top elec | 18 Å | 34.015070258240904 Bohr |
| \(\eta\) | 2.0 Å⁻¹ | 1.05835442134 Bohr⁻¹ |
| \(\alpha_{\mathrm{cation}}\) | 1.0 Å³ | 6.748334503514373 Bohr³ |
| \(\alpha_{\mathrm{anion}}\) | 1.5 Å³ | 10.12250175527156 Bohr³ |
| \(\Psi_{\mathrm{top}}\) | 1 V | 0.03674932217565499 Ha/e (= 1/27.211386245988) |
| \(\Psi_{\mathrm{bot}}\) | 0 V | 0.0 |

MW block: `dipoles_and_electrodes / algorithm cg 1.0e-13 5000 / preconditioner
jacobi / charge_neutrality true`, `coulomb_rtol 1e-12`, `coulomb_rcut 9.4` Bohr
(the \(L_{\min}/2 = 9.4486\) Bohr ceiling), `coulomb_ktol 1e-12`.

Two MW-side gotchas, both handled in the shipped input and worth remembering:

- A `damping` block is **mandatory** whenever dipoles are on. `system%tt` is
  allocated only by `read_tt_parameters` (`src/configuration.F90`), yet
  `coulomb_sr_pim` dereferences `tt%order/b/c` unconditionally → SIGSEGV in
  `qmelt2mumelt_electricfield` without it. A `tt_pair … c = 0` entry allocates
  it with every damping term identically zero.
- With `temperature 0` the `data.inpt` must carry a `# velocities` section for
  **all** atoms (electrode rows included, and they must be exactly 0), otherwise
  `print_step` aborts on `Temperature is NaN or meaningless`.

### Raw MetalWalls output

`charges.out` / `dipoles.out`, atomic units, `num_pbc 3`:

```
# Electrode atom charges in atomic unit
# step        0
-6.707261457445883E-002
+6.707261457445883E-002

# Dipoles on mobile particles in atomic unit
#          mux                     muy                     muz
# step        0
+6.745304354600813E-018 -3.094316960449654E-018 +1.326729243171588E-001
+2.064507384972561E-018 -4.018619627801056E-018 +1.957837940515414E-001
```

and with the dipoles switched off (`electrode_charges cg`, the \(\alpha=0\) branch):

```
-6.913534940031944E-002
+6.913534940031944E-002
```

For reference, the same twin under `num_pbc 2` (slab Ewald) gives
\(Q_{\mathrm{top}} = +1.302179493406125\times10^{-1}\) (joint) and
\(+1.362412229597930\times10^{-1}\) (\(\alpha=0\)). `num_pbc 3` is the closer
analogue of ATV's min-image, which wraps all three axes.

**Unit conversion.** Charge is already in \(e\) (MW: "1 au = 1 e"), so no
conversion. Dipole is \(e\cdot a_0\) → \(e\cdot\)Å by \(\times 0.52917721067\):

- \(\mu_{\mathrm{cat},z}\): \(0.1326729243171588 \times 0.52917721067 = 7.0207488022\times10^{-2}\) e·Å
- \(\mu_{\mathrm{an},z}\): \(0.1957837940515414 \times 0.52917721067 = 1.0360432203\times10^{-1}\) e·Å

Dimensional check (not number-matching): MW's \(\mu = \alpha E\) is in atomic
units where \(k_e \equiv 1\); ATV's \(\mu[e\,\text{Å}] = \alpha[\text{Å}^3]\,q[e]/r^2[\text{Å}^2]\)
is the same \(k_e\)-absorbed convention with polarizability as a volume. So the
dipole conversion is length-only (\(a_0\to\)Å) and \(\alpha\) converts as a volume
(\(a_0^3\to\)Å³), which is what the twin table does.

### Pin vs MetalWalls

| Observable | ATV pin | MW `num_pbc 3` | (pin − MW)/\|MW\| |
| --- | --- | --- | --- |
| \(Q_{\mathrm{top}}\) joint | +7.3285008958883610e−02 | +6.7072614574458830e−02 | **+9.26 %** |
| \(Q_{\mathrm{bot}}\) joint | −7.3285008958883610e−02 | −6.7072614574458830e−02 | **−9.26 %** |
| \(Q_{\mathrm{top}}\), \(\alpha=0\) | +7.5339627935767609e−02 | +6.9135349400319443e−02 | **+8.97 %** |
| \(\mu_{\mathrm{cat},z}\) | +5.8951916148990660e−02 | +7.0207488022e−02 | **−16.03 %** |
| \(\mu_{\mathrm{an},z}\) | +8.7432701391015569e−02 | +1.0360432203e−01 | **−15.61 %** |

Nine percent, against a \(10^{-6}\) gate. The pin is **not** confirmed by MW.

### Where the gap comes from — measured, not asserted

`tools/metalwalls_oracle/inputs/metalwalls/dipole-electrode-joint/ewald_check.py`
re-implements **the ATV kernels verbatim** — Gaussian self term
\(k_e\eta\sqrt{2/\pi}\), Gaussian–Gaussian \(k_e\operatorname{erf}(\eta r/\sqrt2)/r\),
Gaussian–point \(k_e q\operatorname{erf}(\eta r)/r\), neutrality projection,
\(AQ=\Psi-b\) — with one switch: the real-space sum is either **minimum image**
(what ATV does) or a **full 3D Ewald lattice sum**. The \(\alpha=0\) branch has no
dipoles, so it isolates the electrode algebra:

| summation convention | \(Q_{\mathrm{top}}\) [e] |
| --- | --- |
| ATV kernels, minimum image (**= the pin**) | 7.5339627935767609e−02 |
| ATV kernels, 3D Ewald | 6.9135350235340731e−02 |
| **MetalWalls c8ad175d, `num_pbc 3`** | **6.9135349400319443e−02** |

ATV's own formulas under a matched lattice sum reproduce MetalWalls to
**1.21 × 10⁻⁸ relative**, stable to 1e-15 across five independent Ewald
splittings (\(\alpha_{\mathrm{Ew}} = 0.25\ldots0.45\) Å⁻¹, real cutoff 10–14 cells,
k-cutoff 18–30). That residual is not numerical noise — it is
`ATV::units::coulomb = 14.399645`, which is 3.28 × 10⁻⁸ below the exact
\(1\,\mathrm{Ha}\cdot a_0 = 14.399645472\) eV·Å; propagating that constant through
\(Q=[\Delta\Psi-\Delta b]/[2(A_{11}-A_{12})]\) predicts +1.1 × 10⁻⁸, which is what
is observed.

**Conclusion: the ATV kernels are MetalWalls' kernels, exactly.** The 9 % is the
minimum-image truncation of a conditionally convergent lattice sum, which is a
*deliberate, documented* property of `DipoleElectrodeJoint`
(`xy_image_range = 0`), not a coding error. The larger μ gap (16 %) is the same
truncation inside the PIM SCC field — `scc_mu_` always uses `box.min_image`
regardless of `xy_image_range` — plus the damping-model difference below.

### Why MetalWalls cannot express this system (`mw_expressible: false`)

Two specific, checkable reasons — neither is "it was hard":

1. **MetalWalls has no minimum-image / non-Ewald electrostatics.** Every Coulomb
   entry point dispatches `select case (num_PBC)` with exactly `case(2)` and
   `case(3)` and no default (`src/coulomb.F90:136`,
   `src/coulomb_sr.f90:47,78,108,144,173,204,235`). `num_pbc 0` or `1` silently
   returns zero potential. The regression's system is defined by a *different
   boundary condition* — min-image real-space Coulomb — which is not a parameter
   MW exposes, not a tolerance that can be loosened, and not something MW can be
   configured into.
2. **MetalWalls has no Thole damping.** `grep -ri thole src/` returns nothing.
   MW's only induced-dipole damping is Tang–Toennies on the *charge–dipole* term
   (`src/damping.f90`, `tt_pair b order c`), and there is **no dipole–dipole
   damping at all**. The regression pins `thole_a = 0.39` applied to charge–dipole
   *and* dipole–dipole (λ₃/λ₅). The MW twin was therefore run undamped (`c = 0`),
   i.e. the ATV `thole_a = 0` limit — a different model from the one the pin
   describes. For scale, at matched min-image summation ATV gives
   \(Q_{\mathrm{top}} = 7.3285\times10^{-2}\) (Thole 0.39) vs
   \(7.3162\times10^{-2}\) (undamped): damping is a 0.17 % effect on \(Q\) but a
   ~6 % effect on \(\mu\).

The pinned values are therefore **kept** and the tolerances are **kept**. They
remain what they always were: ATV's own min-image algebra — now *independently
validated in form* against a real MetalWalls binary. What changed is that this
file no longer claims the numbers came out of MetalWalls.

### Debt — named, not buried

`cpu::md::DipoleElectrodeJoint` computes a **min-image** electrode elastance and
a **min-image** PIM field. For a real electrode that is not a converged model:
the electrode lattice sum is conditionally convergent, and min-image truncation
costs ~9 % on \(Q\) and ~16 % on \(\mu\) for this 10×10×20 Å cell.
`xy_image_range > 0` only adds a bare real-space \(xy\) image sum to \(A\) and
\(b\); it adds no \(k\)-space term and does not touch `scc_mu_` at all, so it does
not converge to the Ewald answer either. Closing E5 against MetalWalls for real
requires wiring this component to the existing `longrange/` Ewald/PME machinery.
Until then this regression is an **algebra pin**, not an electrode-physics pin,
and the numbers above are the measured size of the gap.

Second, smaller smell found while building the twin: the cation–top-electrode and
anion–bottom-electrode separations are **exactly \(L_z/2 = 10\) Å**, i.e. on the
minimum-image tie. Magnitudes are unambiguous, so \(A\) and the permanent-charge
\(b\) are safe, but the sign of \(\Delta z\) in the \(\boldsymbol\mu\)-to-electrode
term \(k_e(\boldsymbol\mu\cdot\mathbf r)/r^3\) is decided purely by the rounding
mode. `cpu::Box::min_image` uses `std::nearbyint` (round-half-to-**even** in the
default FP environment) and `pin_joint.py` uses Python `round` (also
round-half-to-even), so the two agree today — but under round-half-away-from-zero
those two contributions would flip sign. A future geometry tweak should move a
site off the \(L/2\) tie rather than rely on that coincidence.

## Provenance

- MetalWalls c8ad175d — twin inputs + raw output, this file (2026-07-29).
- Coretti et al., *J. Chem. Phys.* (2022) — polarizable electrolyte + metallic
  electrodes (MetalWalls `dipoles_and_electrodes`).
- Marin-Laflèche et al., JOSS **5**, 2373 (2020).
- Thole, *Chem. Phys.* **59**, 341 (1981).
- Pin generator (ATV algebra, min-image):
  `tools/metalwalls_oracle/inputs/metalwalls/dipole-electrode-joint/pin_joint.py`.
- Boundary-condition decomposition:
  `tools/metalwalls_oracle/inputs/metalwalls/dipole-electrode-joint/ewald_check.py`.
- Map: MetalWalls capability **E5** (`dipole-electrode-joint`).
