# semiempirical/dftb2

One case; every section below is run by `atv_regressions --case semiempirical/dftb2` and keeps its own oracle record.


## H2O SCC-DFTB2 (mio-1-1 static SK)

```text
oracle: dftb+
oracle_version: 24.1 + mio-1-1
```

### Method

Self-consistent-charge DFTB2 (Elstner et al., *Phys. Rev. B* **58**, 7260
(1998), DOI [10.1103/PhysRevB.58.7260](https://doi.org/10.1103/PhysRevB.58.7260)).

### Data

In-tree static tables for **all ordered H/C/N/O pairs** transcribed from
DFTB.org / dftbparams mio-1-1 `*.skf` into `basis_data/dftb_mio_sk_tables.h`
(generator: `tools/gen_dftb_mio_tables.py`). Interpolation = DFTB+
`polyInterU1` + `distFudge` (`internal/mio_sk_table.h`). No runtime `.skf`
parser.

### Geometry

Water (Å), cubic box 20 Å (isolated):

| Atom | x    | y    | z   |
|------|------|------|-----|
| O    | 0.00 | 0.00 | 0.0 |
| H    | 0.96 | 0.00 | 0.0 |
| H    |−0.24 | 0.93 | 0.0 |

Same twin as `tools/tb_oracle/inputs/dftbplus/h2o_mio_scc/`.

### Pins / gates

| Quantity | Value | Source |
| -------- | ----- | ------ |
| Total energy (DFTB+) | **−110.9590 eV** (−4.0776668457 Ha) | DFTB+ 24.1, re-run 2026-07-29 |
| ATV absolute tol | **0.05 eV** | covers residual ~0.037 eV |
| Force vs FD | **1×10⁻⁵ eV/Å** | analytic SCC gradient |

### Residual vs DFTB+ (~0.037 eV)

Order-of-magnitude and sign match after real mio tables (sketch era was
~−94.5 eV). Remaining meV-scale gap is under investigation (single-shell
Hubbard U from Us vs DFTB+ shell handling, mixer, onsite SPE). H₂ still
matches DFTB+ to ~1e-8 eV (`dftb2-dftbplus-energy-pin`).

### Re-run oracle

```bash
source tools/tb_oracle/setup_dftbplus_env.sh
tools/tb_oracle/run_dftbplus_pin.sh tools/tb_oracle/inputs/dftbplus/h2o_mio_scc mio
```


## Reference — DFTB2 H₂ energy vs DFTB+ (mio-1-1)

```text
oracle: dftb+
oracle_version: 24.1
```

### Method

SCC-DFTB2 (Elstner et al., *Phys. Rev. B* **58**, 7260 (1998)).
Hamiltonian / SK set: **mio-1-1** H–H (static transcription of
`H-H.skf` into `basis_data/dftb_mio_hh_sk.h`; no runtime `.skf` I/O).

### Geometry

Isolated H₂, r(H–H) = **0.74 Å**, charge 0, closed shell. Cubic box 30 Å
(isolation only).

### Offline pin (DFTB+ 24.1)

| Quantity | Value | Unit |
| -------- | ----- | ---- |
| Total energy | −0.6749509324 | Ha |
| Total energy | **−18.36635052** | eV |
| Conversion | 1 Ha = 27.211386245988 eV | — |
| Band energy | −0.6807883427 | Ha |
| Repulsive | +0.0058374104 | Ha |
| SCC energy | 0 (Δq = 0) | Ha |

Inputs: `tools/tb_oracle/inputs/dftbplus/h2_mio_scc/`.

### Gate

|E_ATV − (−18.36635052 eV)| ≤ **1×10⁻³ eV** on `frame.meta[kEnergy]` via
`cpu::Evaluate` + `cpu::semiempirical::DFTB2` (`with_forces = false`).

### Perf (optional, not ctest)

| Timer | Wall [s] |
| ----- | -------- |
| ATV `reg_dftb2-dftbplus-energy-pin` (Evaluate+DFTB2, forces off) | ~0.06 |
| DFTB+ 24.1 cold (same geometry; includes I/O) | ~0.15 |

Ratio \(T_{\mathrm{atv}}/T_{\mathrm{oracle}}\) ≈ **0.4** (≤ 10). Not asserted in ctest.
Exact |E_ATV − E_pin| on this host: ~3×10⁻⁹ eV.


## Reference — CH4 SCC-DFTB2 with static 3ob-3-1 SK

### Oracle

```text
oracle: dftb+
oracle_version: 24.1
```

Offline pin only — ctest never launches DFTB+.

### Method

- **Hamiltonian:** SCC-DFTB2 (Elstner et al., *Phys. Rev. B* **58**, 7260
  (1998); DOI [10.1103/PhysRevB.58.7260](https://doi.org/10.1103/PhysRevB.58.7260)).
- **SK set:** DFTB.org **3ob-3-1** organic H/C/N/O (Gaus, Goez, Elstner,
  *J. Chem. Theory Comput.* **9**, 338 (2013); DOI
  [10.1021/ct300849w](https://doi.org/10.1021/ct300849w)). Static tables in
  `basis_data/dftb_3ob_tables.h` (CC-BY-SA 4.0 transcription; no runtime `.skf`).
- **Pin Hamiltonian twin:** DFTB+ `SCC = Yes`, **no** `ThirdOrder` (pure DFTB2
  with 3ob SK files). 3ob was trained for DFTB3; this case pins **SK fidelity**
  under DFTB2.

### Geometry

Methane, tetrahedral, \(r(\mathrm{C{-}H}) = 1.09\) Å:

| Atom | x [Å] | y [Å] | z [Å] |
|------|-------|-------|-------|
| C | 0 | 0 | 0 |
| H | \(s\) | \(s\) | \(s\) |
| H | \(s\) | \(-s\) | \(-s\) |
| H | \(-s\) | \(s\) | \(-s\) |
| H | \(-s\) | \(-s\) | \(s\) |

with \(s = 1.09 / \sqrt{3}\). Isolated (cubic box 20 Å, molecular path).

### Pin

| Quantity | Value | Units |
|----------|-------|-------|
| Total energy (DFTB+ 24.1) | **−3.2291103259** | Ha |
| Total energy (public ATV) | **−87.86856831** | eV |

Conversion: \(E[\mathrm{eV}] = E[\mathrm{Ha}] \times 27.211386245988\).

DFTB+ detailed components (documentation only):

| Term | Ha | eV |
|------|-----|-----|
| Electronic | −3.1685644194 | −86.2210 |
| SCC | 0.0011039499 | 0.0300 |
| Repulsive | −0.0605459065 | −1.6475 |
| **Total** | **−3.2291103259** | **−87.8686** |

### Gate

- Public API: `cpu::Evaluate` + `cpu::semiempirical::DFTB2` with
  `sk_set = DftbSkSet::ThreeOb`, `with_forces = false`.
- Assert `|E_ATV − (−87.86856831)| ≤ 0.05` eV (SK tables + Elstner
  γ vs DFTB+ poly-interp / γ implementation residual).

### Unmoved by the 2026-07-28 SCC repair — this is the control

`E = −87.84300546 eV` vs the DFTB+ 24.1 pin `−87.86856831 eV`
(|ΔE| = 2.6e-02 eV) is byte-unchanged across the charge-mixing and
double-counting fixes. That is the point: the 3ob path **already** assembled
`E = Tr(ρH⁰) + E² + E_rep` (the correct route) and its SCC **already**
converged, so the only in-tree DFTB2 number with a real DFTB+ oracle behind it
is the one number that must not move — and did not. Only the MioSketch path,
which was on the `E_band − E²` route, moved. The 3ob analytic *gradient* did
change (wrong-sign γ term), but this case runs with `with_forces = false`.

### Float → double re-emission (reproducibility, not accuracy)

`tools/gen_dftb_3ob_tables.py` now transcribes 3ob-3-1 `.skf` into
`basis_data/dftb_3ob_tables.h` with `Dftb3obSkPoint` stored as eight
`double`s at 17 significant digits (was `float`, 9 digits). This is a
**reproducibility** fix — a generator and a full-precision table — not an
accuracy fix. The float32 storage bound on this Hamiltonian is
≤ 2.03e-06 Ha (5.53e-05 eV) and cannot explain a 9.4e-04 Ha
(~2.6e-02 eV) residual (~470×).

Measured CH₄ energy on the same public `DFTB2` / `DftbSkSet::ThreeOb` /
`DftbBoundary::cluster` launch used by this regression:

| Table | \(E\) [eV] | vs DFTB+ pin −87.86856831 eV |
|-------|------------|------------------------------|
| committed `float` 9-digit | −87.868568739305 | −4.3e-07 eV |
| re-emitted `double` 17-digit | −87.868568232331 | +7.8e-08 eV |

**Movement from this re-emission: +5.07e-07 eV** (float table → double
table), well under the expected 1e-04 eV. The regression golden
(DFTB+ pin, 0.05 eV gate) is unchanged.

The historical ATV snapshot \(E = -87.84300546\) eV in the section above
(|ΔE| = 2.6e-02 eV vs DFTB+) is already stale on current code: even the
committed float tables sit on the DFTB+ pin to ~4e-07 eV. That 2.6e-02 eV
gap is not present on this Hamiltonian today and was not closed by
widening storage.

### Perf (optional, not ctest)

DFTB+ wall for this molecule: ~0.12 s (single core, release). ATV expected
well under 10× (small dense eigenproblem).


## Reference — DFTB2 analytic nuclear forces (H2O)

oracle: literature
oracle_version: Hourahine et al., J. Chem. Phys. 152, 124101 (2020); Elstner
  et al., Phys. Rev. B 58, 7260 (1998) — closed-form SCC-DFTB2 nuclear gradient
family_oracle: dftb+

No DFTB+ run produced an absolute energy or force for this geometry. The
gate is analytic \(F\) vs re-SCC FD of the same ATV energy. Do not read
`oracle: dftb+` into that identity.

### Method

SCC-DFTB2 (Elstner et al., Phys. Rev. B **58**, 7260 (1998)):

\[
E = \sum_i n_i\varepsilon_i - \tfrac12\sum_{AB}\Delta q_A\gamma_{AB}\Delta q_B + E_{\mathrm{rep}}
\]

Nuclear forces are the **closed-form** gradient at the SCC density (Hellmann–Feynman
+ Pulay energy-weighted density + Hamiltonian shift on \(\partial S\) + fixed-\(\Delta q\)
\(-\partial E_{\mathrm{SCC}}\) + repulsive) — **not** a re-SCC finite difference.
This is the DFTB+ force module boundary.

### Hamiltonian / tables

- ATV class: `cpu::semiempirical::DFTB2`
- SK / Hubbard / repulsive: static `basis_data/dftb_mio_tables.h` (mio-**style**
  sketch for H/C/N/O). **Not** bit-identical to DFTB.org mio-1-1.

### Geometry (Å)

| atom | Z | x | y | z |
| ---- | - | - | - | - |
| O | 8 | 0.00 | 0.00 | 0.00 |
| H | 1 | 1.05 | 0.00 | 0.00 |
| H | 1 | −0.26 | 1.00 | 0.00 |

Isolated molecule, cubic box 20 Å. `scc_tol = 1e-10`.

### Pins (hard-coded gates)

1. **Energy:** finite, \(|E| > 10^{-3}\) eV.
2. **Forces:** all components finite; \(|F|^2 > 10^{-8}\) at off-eq geometry.
3. **Translation:** \(|\sum_i F_i| \le 10^{-6}\) eV/Å per Cartesian.
4. **FD identity:** analytic \(F\) vs central **re-SCC** FD of the same \(E\)
   (\(h=10^{-4}\) Å) on \(F_{O,x}\) and \(F_{H1,y}\) within **1×10⁻⁵ eV/Å**
   (measured 2.5e-07 / 4.9e-09 eV/Å). A converged SCC-DFTB2 gradient is
   stationary, so this is FD-exact to the central-difference truncation error.

Offline pin note: `tools/tb_oracle/inputs/dftbplus/dftb2_analytic_forces_note.md`.

### Closed — the polar-gradient defect (found 2026-07-28, fixed 2026-07-28)

The gate used to read **1.0 eV/Å** and pass at \(|d| = 0.23\) / \(0.75\)
eV/Å, i.e. ~4× looser than the error it was meant to catch, with the standing
explanation "Mulliken charge response beyond fixed-\(\Delta q\) HF". That
explanation was wrong — a converged SCC-DFTB2 gradient is stationary and needs
no coupled-perturbed correction. Three real defects were stacked on top of each
other, which is why the earlier A/B on the \(\gamma\) term alone
(shipped 0.228, term dropped 1.020, sign flipped 2.268 eV/Å) concluded the
shipped form was "best of three" and closed the wrong door:

1. **The SCC never converged.** The undamped Picard iteration was a period-2
   limit cycle, so both the analytic force and the FD reference were sampled
   off different branches of a cycle. Fixed by Anderson charge mixing
   (`internal::ChargeMixer`, public `mix_beta`, default 0.3); a budget-exhausted
   SCC now throws.
2. **The reported energy was not the functional the gradient differentiates.**
   `E = E_band − E_dc` carried a spurious \(\sum_A V_A q^0_A\) (**7.85 eV**
   on this geometry) because \(E_{\mathrm{band}}\) contracts \(V_A\) with the
   **gross** Mulliken population. Now
   \(E = \mathrm{Tr}(\rho H^0) + E^{(2)} + E_{\mathrm{rep}}\).
3. **The \(\gamma\) gradient term had the wrong sign.** With
   \(E^{(2)} = \tfrac12\sum_{AB}\gamma_{AB}\Delta q_A\Delta q_B\) and
   bond \(= R_B - R_A\), the ordered double sum gives the pair term
   \(\gamma_{AB}\Delta q_A\Delta q_B\) and \(\partial r/\partial\) bond
   \(= +\hat r\), so \(\partial E^{(2)}/\partial\) bond
   \(= +\Delta q_A\Delta q_B\,d\gamma/dr\). The code carried a minus.
   Present in all three assemblies (`analytic_forces_restricted`,
   `analytic_forces_unrestricted`, `analytic_forces_3ob`).

Residual after the fix, measured on this case:

| case | \(\Delta q\) | analytic vs re-SCC FD (before → after) |
| ---- | ------------ | -------------------------------------- |
| O₂ triplet (`reg_dftb2-spin-analytic-forces`) | 0 | 3.0e-07 → 3.0e-07 eV/Å |
| H₂O MioSketch (this case) | ≠ 0 | 0.23 / 0.75 → **2.5e-07 / 4.9e-09** eV/Å |
| H₂O 3ob (measured at the fix; no standing case) | ≠ 0 | 0.84 / 0.24 → **4.4e-07 / 2.0e-09** eV/Å |
| H₂O stress (`reg_dftb2-analytic-stress`) | ≠ 0 | 1.3e-04 → **1.1e-10** eV/Å³ |

The energy of this case moved from −44.60971806 to **−91.70909683 eV** as a
consequence of (1) and (2): the old number was a limit-cycle snapshot over a
double-counting route that was wrong at second order. It is an ATV twin
snapshot, not an oracle value — no DFTB+ identity is claimed for it.

**Closed 2026-07-29:** the owed open-shell \(\Delta q \ne 0\) gate landed as
the **NO doublet** in `regressions/semiempirical/dftb2`,
and `analytic_forces_unrestricted` / `analytic_forces_restricted` were folded
into one spin-resolved `analytic_forces_stress`, so a sign defect in the
\(\gamma\) or Pulay term can no longer survive in one copy while being fixed in
another.

### Perf

`perf_exempt: true` — no DFTB+ absolute-force twin was run for this
geometry (FD identity only). Analytic path replaces \(6N\) full re-SCC
force evaluations with one post-SCC contraction.


## Reference — DFTB2 analytic cell stress (H2O)

oracle: literature
oracle_version: Hourahine et al., J. Chem. Phys. 152, 124101 (2020); Elstner
  et al., Phys. Rev. B 58, 7260 (1998) — pair-virial cell stress of SCC-DFTB2
family_oracle: dftb+

No DFTB+ run produced an absolute energy or stress for this geometry. The
gate is analytic \(\sigma\) vs strain-FD of the same ATV energy. Do not read
`oracle: dftb+` into that identity.

### Method

SCC-DFTB2 (Elstner et al., Phys. Rev. B **58**, 7260 (1998)):

\[
E = \sum_i n_i\varepsilon_i - \tfrac12\sum_{AB}\Delta q_A\gamma_{AB}\Delta q_B + E_{\mathrm{rep}}
\]

Intensive cell stress (mechanical / DFTB+ / PW–GPW convention):

\[
\sigma_{\alpha\beta} = -\frac{1}{\Omega}\frac{\partial E}{\partial\varepsilon_{\alpha\beta}}
\]

closed-form at the SCC density via the **pair virial** of the same analytic
nuclear force terms (Hellmann–Feynman + Pulay + fixed-\(\Delta q\) SCC \(\gamma\)
+ repulsive) — **not** a multi-SCC strain finite difference on the product path.

Voigt order on Frame `kStress`: `(xx, yy, zz, xy, xz, yz)` in **eV/Å³**.

### Hamiltonian / tables

- ATV class: `cpu::semiempirical::DFTB2`
- SK / Hubbard / repulsive: static `basis_data/dftb_mio_tables.h` (mio-**style**
  sketch for H/C/N/O). **Not** bit-identical to DFTB.org mio-1-1.

### Geometry (Å)

| atom | Z | x | y | z |
| ---- | - | - | - | - |
| O | 8 | 0.00 | 0.00 | 0.00 |
| H | 1 | 1.05 | 0.00 | 0.00 |
| H | 1 | −0.26 | 1.00 | 0.00 |

Isolated molecule, cubic box 20 Å. `scc_tol = 1e-10`. `with_stress = true`.

### Pins (hard-coded gates)

1. **Energy:** finite, \(|E| > 10^{-3}\) eV.
2. **Stress:** all six Voigt components finite.
3. **Strain-FD identity:** analytic \(\sigma_{xx}\), \(\sigma_{yy}\) vs
   multi-SCC affine strain-FD of the same \(E\) (\(h=2\times10^{-4}\)) within
   **1×10⁻⁸ eV/Å³**. The stress is the pair virial of the same bond terms as
   the analytic force, and a converged SCC-DFTB2 gradient is stationary, so
   this is FD-exact to the strain-FD truncation error.

Offline pin note: `tools/tb_oracle/inputs/dftbplus/dftb2_analytic_stress_note.md`.

Measured after the gradient + double-counting fixes (2026-07-28):

```text
E = -91.70909683 eV
sigma = (-1.93986571e-03, -1.71915651e-03, 0.0,
          1.02900333e-03, -7.8e-20, 3.6e-20) eV/A^3
sigma_xx: analytic -1.93986571e-03  strain-FD -1.93986582e-03  |d| 1.147e-10
sigma_yy: analytic -1.71915651e-03  strain-FD -1.71915660e-03  |d| 8.937e-11
```

\(\sigma_{zz}=0\) exactly and the \(xz,yz\) shears are ~1e-20 — correct for a
planar molecule in the \(xy\) plane.

### Closed — the 1.3e-4 eV/Å³ residual was the force defect (fixed 2026-07-28)

The gate used to read **5×10⁻³ eV/Å³** and pass at 1.3e-4 / 1.0e-4, i.e. ~7%
relative, described as a "fixed-\(\Delta q\) vs re-SCC residual". It was not:
this stress is by construction the pair virial of the same bond terms as
`analytic_forces_restricted`, and it inherited that function's wrong-sign
\(\gamma\) gradient. With the sign corrected (and the SCC actually converged,
and the energy on the \(\mathrm{Tr}(\rho H^0) + E^{(2)} + E_{\mathrm{rep}}\)
route) the residual is **1.1e-10 / 8.9e-11 eV/Å³** — six orders of magnitude
tighter, and the gate is tightened to 1e-8 to match.

Both \(E\) and \(\sigma\) moved (the stress even changed sign): the old
numbers were a limit-cycle snapshot over a double-counting route that was wrong
at second order, differentiated by a gradient with a sign error. Neither the old
nor the new value is an oracle number — no DFTB+ identity is claimed.

### Gaps (still open)

1. ~~**D3 dispersion is absent from the stress.**~~ **Closed 2026-07-29:** with
   `d3=true` and `with_stress=true`, D3 is folded into `last_stress_` /
   `kStress` from the analytic pair virial of `dftd3_pbe_bj`, image-aware when
   `d3_periodic=true`. Unit gates in
   `tests/cpu/semiempirical/test_dftb2_dispersion_d3.cpp`
   (`StressIncludesD3`, `StressTraceMatchesVolumeDerivative`).
2. ~~**Open shell falls back to strain-FD.**~~ **Closed 2026-07-29:**
   `analytic_forces_restricted` / `analytic_forces_unrestricted` were folded
   into one spin-resolved `analytic_forces_stress` that also emits the virial;
   this case's numbers are unchanged
   (`regressions/semiempirical/dftb2`).
3. ~~**No periodic case.**~~ **Closed 2026-07-29** (unit-gated, see below).
   This gate's geometry is still a molecule in a 20 Å box, and the numbers above
   are bit-unchanged — but the defect it was pointing at was real and is fixed:
   the pair virial walked ordered pairs \(a<b\) only, so it dropped the
   self-image \(A\)–\((A+\mathbf T)\) bond terms. Those carry no nuclear force
   (the bond is \(\mathbf T\), independent of every nuclear coordinate) but do
   scale with the cell, so they belong in the virial and only in the virial.
   Invisible here (every self-image is beyond \(r_{\mathrm{cut}}\)); on a
   periodic C-chain cell (\(L_x = 2.8\) Å < \(r_{\mathrm{cut}} = 4.5\) Å) it was
   a 14% error, \(\sigma_{xx}\) analytic \(-0.31716963\) vs strain-FD
   \(-0.37019871\) eV/Å³. With the self-image \(H^0/S/\gamma/E_\mathrm{rep}\)
   virial terms added, the residual is \(<10^{-6}\) eV/Å³ and the periodic case
   is unit-gated by `tests/cpu/semiempirical/test_dftb2.cpp`
   `PeriodicAnalyticStressMatchesStrainFdCChain`.

   The self-image virial is carried on the **spin-resolved** assembly, so it
   applies to the open-shell path as well as the restricted one.

### Perf

`perf_exempt: true` — no DFTB+ absolute-stress twin was run for this
geometry (strain-FD identity only). Analytic path replaces 12 multi-SCC
energy evaluations (6 Voigt × ±h) with one post-SCC pair contraction.


## dftb2-ewald-gamma — Ewald-summed periodic SCC γ vs DFTB+

```text
oracle: dftb+
oracle_version: 25.1 development (commit a23bfb2) + mio-1-1
```

Offline pin only — ctest never launches DFTB+.

### Why this case exists

The earlier periodic DFTB2 checks (Γ energy, Γ forces, multi-k energy and
forces) all used the same homonuclear 1-D carbon chain. There Δq ≡ 0 by symmetry, so
\(E^{(2)}=\tfrac12\sum\gamma\Delta q\Delta q\) vanishes **identically** and the
SCC charge kernel is never evaluated at all. The periodic γ was therefore
untested for as long as that chain was the only periodic geometry, and it was
wrong: the \(1/R\) half of \(\gamma\) was truncated at the Slater–Koster
cutoff, which is not an approximation of a lattice sum but a different,
non-positive-definite operator. On this cell that gave \(E^{(2)} = -2.19\) Ha
against DFTB+'s \(+0.138\) Ha and a 37 eV error in the total.

This case is polar by construction and asserts \(\max|\Delta q| > 0.1\,e\), so
it cannot silently degenerate into the same blind spot.

### Method

SCC-DFTB2 (Elstner et al., *Phys. Rev. B* **58**, 7260 (1998), DOI
[10.1103/PhysRevB.58.7260](https://doi.org/10.1103/PhysRevB.58.7260)),
`boundary = lattice`: image-summed \(H^0\)/\(S\)/repulsive within the SK
cutoffs, and \(\gamma = 1/R - S(\tau_A,\tau_B,R)\) with the \(1/R\) half taken
as an Ewald sum with a neutralising background
(`internal/dftb_gamma_lattice.h`) and the exponential \(S\) half truncated
where it falls below \(10^{-12}\) eV.

SK set: mio-1-1 H/O (`basis_data/dftb_mio_sk_tables.h`). No runtime `.skf`.

### Geometry

8 water molecules, r(O–H) = 0.96 Å, ∠HOH = 104.5°, oxygens on a 2×2×2
sublattice with random rigid-body orientations (seed 20260902) and a 0.15 Å
positional jitter. Cubic cell **L = 7.0 Å** (≈ 0.81 g/cm³). Minimum
inter-atomic distance under the minimum image is 0.96 Å (the intramolecular
O–H bond), i.e. no intermolecular contact is closer than a bond length.

Full coordinates are in the regression source and in
`tools/tb_oracle/inputs/dftbplus/dftb2-ewald-gamma/geo.gen`.

### DFTB+ settings

`SCC = Yes`, `SCCTolerance = 1e-10`, `MaxAngularMomentum = {O = p; H = s}`,
`KPointsAndWeights = SuperCellFolding {1 0 0  0 1 0  0 0 1  0 0 0}` (Γ),
`Analysis { CalculateForces = Yes }`, `ParserVersion = 8`.

### Pins

| Quantity | Value | Unit |
| -------- | ----- | ---- |
| Band energy | −30.0466782447 | Ha |
| **Energy SCC** | **+0.1379943876** | Ha |
| Repulsive energy | +0.6186556656 | Ha |
| **Total energy** | **−32.6355809899** | Ha |
| Total energy | **−888.0593996784** | eV |
| Energy SCC | +3.7550185808 | eV |

Conversion 1 Ha = 27.211386245988 eV. Per-atom forces [eV/Å] are transcribed
into the regression source from `detailed.out`.

### Gates

| Gate | Tolerance |
| ---- | --------- |
| Total energy vs DFTB+ | 1×10⁻³ eV |
| **SCC energy vs DFTB+** (on its own) | 1×10⁻³ eV |
| max per-component force vs DFTB+ | 1×10⁻³ eV/Å |
| max \|Δq\| (the cell is actually polar) | > 0.1 e |

Observed at the time of writing: |ΔE| = 1.9×10⁻⁶ eV, |ΔE_SCC| = 1.8×10⁻⁶ eV,
max |ΔF| = 3.4×10⁻⁶ eV/Å, max |Δq| ≈ 0.6 e.

### Re-run oracle

```bash
source tools/tb_oracle/setup_dftbplus_env.sh
tools/tb_oracle/run_dftbplus_pin.sh tools/tb_oracle/inputs/dftbplus/dftb2-ewald-gamma mio
```


## dftb2-scc-guess — ASPC extrapolation of the SCC charges

```text
oracle: literature
oracle_version: Kolafa ASPC (formula identity); DFTB+ not applicable
family_oracle: dftb+
```

This case has **no absolute oracle pin and needs none**. It gates an
optimisation, and an optimisation has exactly two things to prove: that it
speeds something up, and that it changes nothing. Both are internal
comparisons against the same code with the feature off. DFTB+ is not a twin
here because its own charge restart (`ReadInitialCharges` / `charges.bin`) is a
different scheme with different iteration counts; comparing the two counts
would measure the mixers, not this feature.

### Method

Kolafa's always-stable predictor–corrector (Kolafa, *J. Comput. Chem.* **25**,
335 (2004), DOI [10.1002/jcc.10385](https://doi.org/10.1002/jcc.10385)),
order K = 3, applied to the Mulliken \(\Delta q\) of SCC-DFTB2 (Elstner et al.,
*Phys. Rev. B* **58**, 7260 (1998)). Implementation is the shared
`cpu::electronic::ASPCPredictor` — the same type Gaussian SCF and GPW use on
the density matrix — driven through `DFTB2::predictor(order)`.

The extrapolated \(\Delta q\) seeds the SCC. The residual gate is untouched, so
the converged point cannot move; only the number of iterations to reach it can.

### System

The `dftb2-ewald-gamma` cell: 8 water molecules, cubic 7.0 Å, Γ, mio-1-1,
`boundary = lattice`, `scc_tol = 1e-8`. Polar by construction — extrapolating
\(\Delta q\) is worth nothing on a cell where \(\Delta q \equiv 0\).

16 launches along a smooth scripted displacement,
\(\delta_{i\alpha}(s) = 0.01\,\mathrm{Å} \cdot s \cdot \sin(0.37 i + 1.13
\alpha)\). Scripted rather than driven by `md::MD` so the case builds in the
semiempirical-only configuration (`ATV_BUILD_MD=OFF`); the predictor's contract
is per-launch, which is what this exercises.

### Gates

| Gate | Tolerance | Observed |
| ---- | --------- | -------- |
| **Converged energy unchanged** | 1×10⁻⁶ eV | **2.3×10⁻¹³ eV** |
| SCC iteration speedup | ≥ 1.25× | **1.90×** (188 → 99) |

The energy gate is the load-bearing one: a predictor that moved the answer
would be a bug wearing a speedup's clothes.

The speedup gate sits well under the observed value on purpose. Two things cap
what a guess can buy:

- The first K+1 = 4 launches have no history and run cold.
- Past that the floor is the **mixer's** convergence rate, not the guess. A
  better starting point lowers the initial residual; getting from there to
  `scc_tol` still costs Anderson/Pulay its own iterations. Raising the ceiling
  further means changing the mixer (DFTB+ defaults to Broyden), not the guess.

### Trajectory dependence — read this before quoting 1.90×

The speedup depends on how smooth the charge history is:

| trajectory | iterations off → on | speedup |
| --- | --- | --- |
| smooth scripted displacement (this case) | 188 → 99 | 1.90× |
| `md::MD` + `VelocityVerlet`, 0.2 fs, Maxwell-Boltzmann 300 K, 16 steps | 176 → 119 | 1.48× |
| same, 40 steps | 440 → 306 | 1.44× |

Thermal velocities make \(\Delta q(t)\) less polynomial than a scripted ramp,
so a production NVT run should be expected around **1.4–1.5×**, not 1.9×.

### Re-run

```bash
ctest --test-dir <build> -R '^semiempirical/dftb2$'
```


## Reference — DFTB2 + D3(BJ) water dimer dispersion

### Method

SCC-DFTB2 electronic + repulsive (ATV static mio-style SK sketch) with
optional **Grimme DFT-D3 Becke–Johnson** two-body correction (`DFTB2::d3`).

Damping parameters = published **DFTB3-D3(BJ)** set (DFTB+ / Grimme list;
`s-dftd3 --bj dftb3`):

| param | value |
| ----- | ----- |
| \(s_6\) | 1.0 |
| \(s_8\) | 0.5883 |
| \(a_1\) | 0.5719 |
| \(a_2\) | 3.6017 Bohr |

CN form: molecular Grimme (simple-dftd3 / PySCF path), **not** CP2K tanh.
ATM off. Non-periodic molecular evaluation.

### Integration note (merge onto dev, 2026-07-28)

This feature was authored against the pre-`sk_set` DFTB2, where the total energy
was assembled inline and forces were a central FD of that total — so folding D3
into the FD evaluator sufficed. On current `dev`, `DFTB2::launch` dispatches to
`launch_mio` / `launch_3ob`, and the closed-shell mio path uses **analytic**
electronic forces with no FD evaluator to hook into. The feature was therefore
**ported**, not merged textually: D3 is now applied once in `DFTB2::launch`
after the SK path returns, contributing

- its energy to `energy_` and `frame.meta[kEnergy]` (deliberately **not** to
  `kEnergyElectronic` — dispersion is an empirical geometric term), and
- its **own analytic gradient** to fx/fy/fz.

This makes `F = −∇E` hold for the total on *every* path (closed-shell analytic,
open-shell FD, 3ob FD) instead of only on the FD ones. Verified out-of-tree by
isolating `F_D3 = F(d3=true) − F(d3=false)` against a central FD of `E_disp`
alone: worst deviation **7.5e-11 eV/Å** on this geometry (max |F_D3| =
0.0067 eV/Å). No in-tree gate covers the D3 force path yet — see Gaps.

Neither gate below depends on the SK parameters, so the 2026-07-27 upgrade of
the H onsite/U rows from sketch values to transcribed mio-1-1 does not move this
pin. Measured after the port: `E_disp = −0.045967885155 eV` against the
s-dftd3 pin `−0.045967885157 eV`, `|Δ| = 1.5e-12 eV`.

**The pinned quantity is unmoved by the 2026-07-28 SCC repair** (D3 is a purely
geometric additive term): `E_disp = −0.045967885155 eV`, `|Δ| = 1.5e-12 eV`.
The printed *total* energies did move — `E(no D3) = −83.6622617788` →
**−188.7287268170 eV**, `E(D3) = −83.7082296640` → **−188.7746947021 eV** —
because the electronic MioSketch part was a non-convergent Picard limit-cycle
snapshot over a double-counting route that was wrong at second order. Neither
total is pinned or claimed against an oracle; the gate is on `E_disp` and on
`E_tot(d3) − E_tot(off) = E_disp` (still 1e-8 eV), both of which held.
Before the repair this water dimer did not converge at all: with linear
damping it needed >50 iterations, which is what drove the switch to Anderson
mixing (it now converges in 14 at `mix_beta = 0.3`).

### Gaps

- ~~**D3 forces are ungated.**~~ **Closed 2026-07-29:**
  `tests/cpu/semiempirical/test_dftb2_dispersion_d3.cpp`
  `ForcesMatchCentralFDOfEDisp` now gates \(F_{D3}\) against a central FD of
  \(E_{\mathrm{disp}}\) in-tree (< 1e-6 eV/Å), molecular and periodic. D3 stress
  is gated too (`StressIncludesD3`, `StressTraceMatchesVolumeDerivative`).
- ~~**No periodic D3.**~~ **Closed 2026-07-29:** `DFTB2::d3_periodic` sums
  lattice images over an orthorhombic cell, pinned against offline s-dftd3 1.2.1
  in `regressions/semiempirical/dftb2/`. This case stays
  the **cluster** pin; the periodic one owns the image term. Note that the
  periodic value depends on `DFTB2::d3_cutoff` (new member, default unchanged at
  94.2 Bohr) — the molecular pin here does not.
- **Triclinic cells still unsupported** — `d3_periodic` throws on a
  non-orthorhombic box.

### Oracle

```text
oracle: literature
oracle_version: Simple-DFTD3/s-dftd3 1.2.1 (offline)
```

Primary numerical pin is **pure \(E_{\mathrm{disp}}\)** from offline
**s-dftd3 1.2.1** (Simple-DFTD3). DFTB+ is the family oracle for the electronic
DFTB2 ladder; for this geometry-only D3 correction the literature/simple-dftd3
engine is the correct pin (same model DFTB+ uses for D3). ctest never launches
s-dftd3 or DFTB+.

### Hard gate — E_disp

**E_disp = −0.0016892886213669401 Ha = −0.04596788515696838 eV**

Source: offline s-dftd3 1.2.1 on the geometry below:

```text
s-dftd3 --bj dftb3 --json pin.json dimer.xyz
## Dispersion energy:      -1.6892886213669E-03 Eh
```

Conversion: CODATA / ATV `kHartreeInEv = 27.211386245988`.

Absolute total DFTB2 energy is **not** pinned to DFTB+ (ATV SK is a mio-style
sketch). Regression gates:

1. `|E_disp_ATV − E_disp_pin| ≤ 1e-4 eV`
2. `E_tot(d3) − E_tot(off) = E_disp` within 1e-8 eV

### Geometry (Å)

| Atom | x | y | z |
| ---- | - | - | - |
| O | 0.000 | 0.000 | 0.000 |
| H | 0.757 | 0.000 | 0.586 |
| H | −0.757 | 0.000 | 0.586 |
| O | 0.000 | 0.000 | 2.900 |
| H | 0.757 | 0.000 | 3.486 |
| H | −0.757 | 0.000 | 3.486 |

Cubic box 20 Å (isolated; D3 path ignores PBC).

### Offline reproduce

```bash
export PATH=/software/sse2/el9_epyc9005/easybuild/prefix/software/Simple-DFTD3/1.2.1-gfbf-2025b/bin:$PATH
## twin: tools/tb_oracle/inputs/dftbplus/dftb2-dispersion-d3/dimer.xyz
s-dftd3 --bj dftb3 --json pin.json dimer.xyz
```

Oracle inputs: `tools/tb_oracle/inputs/dftbplus/dftb2-dispersion-d3/`.

### Perf

`perf_exempt: true` — pure D3 pin is microseconds; fair \(T_{\mathrm{atv}}/T_{\mathrm{oracle}}\)
vs s-dftd3 on a 6-atom cluster is not a product gate. Full DFTB2 SCC wall is
electronics-dominated.


## Reference — DFTB2 Γ-PBC lattice-sum D3(BJ)

### Method

SCC-DFTB2 (ATV static mio-style SK sketch) with the opt-in **Grimme DFT-D3
Becke–Johnson** two-body correction, evaluated as a **lattice sum over
orthorhombic cell images** instead of an isolated cluster
(`DFTB2::d3_periodic = true`).

Damping = published **DFTB3-D3(BJ)** set (`s-dftd3 --bj dftb3`), same as the
molecular sibling `dftb2-dispersion-d3`:

| param | value |
| ----- | ----- |
| \(s_6\) | 1.0 |
| \(s_8\) | 0.5883 |
| \(a_1\) | 0.5719 |
| \(a_2\) | 3.6017 Bohr |

CN form: molecular Grimme (simple-dftd3 path), **not** CP2K tanh. ATM off.

### Why this is not a second molecular pin

The molecular sibling pins the isolated-cluster sum. Here the *same* geometry
is evaluated both ways in one run and the gate is on **all three** of:

1. the cluster sum (must still reproduce the molecular pin),
2. the lattice sum (absolute periodic pin), and
3. **their difference — the image term**, which a cluster-only implementation
   reports as exactly `0`.

The image term is what this feature adds; gate 3 is the discriminating one.

### The cutoff is part of the pinned quantity

A lattice sum is only defined up to its real-space truncation: the
\(C_6/R^6\) tail beyond the cutoff is discarded, and D3 engines disagree on
where to stop. ATV's `DFTB2::d3_cutoff` defaults to **94.2 Bohr** (legacy
`dftd3` `rthr` convention, unchanged); **simple-dftd3 truncates the two-body
term at 60 Bohr** and exposes no CLI flag to change it.

Measured on this geometry (cluster term identical in every row, so the spread
is pure image tail):

| `d3_cutoff` [Bohr] | ATV \(E_{\mathrm{disp}}\) [Ha] |
| ------------------ | ------------------------------ |
| 40.0 | −1.6956613842920e−03 |
| 60.0 | −1.6959658556667e−03 |
| 94.2 (default) | −1.6960673778506e−03 |
| 160.0 | −1.6960985763136e−03 |

Against the oracle's −1.6959579071183e−03, the default cutoff looks 3.0e−6 eV
"wrong" and the matched 60 Bohr cutoff agrees to **2.2e−7 eV**. The
regression therefore sets `d3_cutoff = 60 Bohr` so the comparison is
like-for-like. `d3_cutoff` was added as a **public member** for exactly this
reason — before it, the periodic sum could not be pinned against any external
engine. Default behaviour is bit-identical to the previous hard-coded 94.2
Bohr (gated in `test_dftb2_dispersion_d3.cpp`).

The residual 2.2e−7 eV is the **coordination-number** cutoff: ATV drives CN
and the pair sum from one radius, simple-dftd3 uses a separate 40 Bohr CN
cutoff. CN enters only through the interpolated \(C_6(\mathrm{CN})\), so the
effect is third-order small. See Gaps.

### Oracle

```text
oracle: literature
oracle_version: Simple-DFTD3/s-dftd3 1.2.1 (offline)
```

DFTB+ is the family oracle for the electronic DFTB ladder; for this
geometry-only dispersion correction the simple-dftd3 engine is the correct pin
— it is the same D3 model DFTB+ calls. ctest never launches s-dftd3 or DFTB+.

### Hard gates

**Lattice (periodic) — \(E_{\mathrm{disp}} = -1.6959579071183\times10^{-3}\) Ha**
**Cluster (molecular) — \(E_{\mathrm{disp}} = -1.6892886213669\times10^{-3}\) Ha**
**Image term — difference \(= -6.6692857514\times10^{-6}\) Ha**

Conversion: ATV `kHartreeInEv = 27.211386245988`.

1. `|E_disp_cluster − pin_cluster| ≤ 1e-6 eV` (measured 1.5e-12)
2. `|E_disp_lattice − pin_lattice| ≤ 1e-6 eV` (measured 2.2e-7)
3. `|image − pin_image| ≤ 1e-6 eV` (measured 2.2e-7) — **discriminating**
4. `E_disp_lattice < E_disp_cluster` (images must attract)
5. `E_tot(d3 periodic) − E_tot(off) = E_disp_lattice` within 1e-8 eV

### Geometry (Å)

| Atom | x | y | z |
| ---- | - | - | - |
| O | 0.000 | 0.000 | 0.000 |
| H | 0.757 | 0.000 | 0.586 |
| H | −0.757 | 0.000 | 0.586 |
| O | 0.000 | 0.000 | 2.900 |
| H | 0.757 | 0.000 | 3.486 |
| H | −0.757 | 0.000 | 3.486 |

Cubic cell **12 Å** — small enough that the nearest images sit well inside the
D3 cutoff (the 20 Å cell of the molecular sibling would make the image term
~10× smaller and the pin correspondingly less discriminating).

### Offline reproduce

```bash
export PATH=/software/sse2/el9_epyc9005/easybuild/prefix/software/Simple-DFTD3/1.2.1-gfbf-2025b/bin:$PATH
cd tools/tb_oracle/inputs/dftbplus/dftb2-dispersion-d3-periodic
s-dftd3 --bj dftb3 dimer_pbc12.gen   # lattice sum (periodic .gen input)
s-dftd3 --bj dftb3 dimer_mol.xyz     # isolated cluster
```

Oracle inputs: `tools/tb_oracle/inputs/dftbplus/dftb2-dispersion-d3-periodic/`.

### Unit gates (companion, `tests/cpu/semiempirical/test_dftb2_dispersion_d3.cpp`)

The oracle pin fixes the number; these fix the *structure* of the sum:

- `BeyondCutoffBoxEqualsMolecular` — a 120 Å cell (every image past the cutoff)
  collapses onto the cluster sum to 1e-12: no residual image term, no
  double-counted home cell.
- `SupercellEnergyIsExtensive` — \(E(2{\times}2{\times}2) = 8\,E(1{\times}1{\times}1)\)
  to 1e-8 relative. The decisive Bravais-sum identity: a wrong image count, a
  half-counted home cell, or a non-image-aware coordination number all break it.
- `ForcesMatchCentralFDOfEDisp` — periodic analytic \(F_{D3}\) vs central FD of
  the periodic \(E_{\mathrm{disp}}\), < 1e-6 eV/Å.
- `StressTraceMatchesVolumeDerivative` — image-aware virial:
  \(\mathrm{tr}\,\sigma_{D3} = -(3/V)\,\partial E_{\mathrm{disp}}/\partial\ln V\)
  under affine scaling of **cell and positions together**.
- `CutoffTruncatesTheLatticeTail` — monotone, converging tail; default member
  value reproduces the historical 94.2 Bohr result exactly.
- `NonOrthorhombicBoxThrows`, `NonPositiveCutoffThrows`.

### Gaps

1. **Orthorhombic only.** `d3_periodic` reads `box.length(d)` and throws on a
   triclinic cell. General-lattice image enumeration is owed.
2. **One CN cutoff.** ATV drives the coordination number from the same radius
   as the pair sum; simple-dftd3 uses a separate (40 Bohr) CN cutoff. Worth
   2.2e-7 eV here. Splitting the two would close the pin to ~1e-12 like the
   molecular case.
3. **No ATM.** Three-body \(C_9\) is off on both sides (`s9 = 0`), so this
   pins the two-body lattice sum only.
4. **Not gated against a periodic DFTB+ run.** The dispersion model is pinned
   against the engine DFTB+ itself calls, but no end-to-end DFTB+ periodic
   DFTB2+D3 total energy is pinned — no `dftb+` binary exists on this host.

### Perf

`perf_exempt: true` — the D3 lattice sum is microseconds; DFTB2 wall time is
electronics-dominated.


## spin-dftb2-o2 — collinear spin-polarized SCC-DFTB2 on O₂

```text
oracle: dftb+
oracle_version: 24.1
```

Offline twin: `tools/tb_oracle/inputs/dftbplus/spin-dftb2-o2/`
(`geo.gen`, `dftb_in.hsd`, archived `dftb.out.txt` = DFTB+ 24.1 `detailed.out`
verbatim). SK: DFTB.org mio-1-1 `O-O.skf`. `SpinConstants { ShellResolvedSpin
= No; O = { -0.028 } }` — the published PBE table is
\(\{W_{ss},W_{sp},W_{ps},W_{pp}\}=\{-0.035,-0.030,-0.030,-0.028\}\) Ha
(DFTB+ order, \(l'\) fastest); DFTB+ `ShellResolvedSpin = No` accepts a
single constant and fills \(W_{ll'}\) with it, which is ATV's HOAO collapse
(\(W_{\mathrm{pp}}\) for O). `UnpairedElectrons = 2`. Cluster (`C`) geometry.
ctest never launches DFTB+.

Public API: `cpu::semiempirical::DFTB2` with `multiplicity = 3` and
`d.spin_w_ha = basis_data::kDftbSpinConstantsPbeHa;` (`shell_resolved_spin`
stays default `false`). `sk_set` stays DFTB2's default `MioSketch` (transcribed
mio-1-1).

### Method

Collinear unrestricted SCC-DFTB2:

- Charge SCC: Elstner \(\gamma_{AB}\) + Hubbard \(U\) (Elstner et al.,
  *Phys. Rev. B* **58**, 7260 (1998),
  DOI [10.1103/PhysRevB.58.7260](https://doi.org/10.1103/PhysRevB.58.7260)).
- Spin channel: HOAO-collapsed \(W_A\) (Köhler / DFTB+ `ShellResolvedSpin =
  No`). Published PBE constants: DFTB+ manual appendix `app:spinconst`
  (Köhler, PhD thesis, Univ. Paderborn 2004; Köhler et al., *J. Phys. Chem. A*
  **111**, 5622 (2007), DOI 10.1021/jp068802p).
- Energy assembled as \(\mathrm{Tr}(\rho H^0)+E^{(2)}+E_{\mathrm{spin}}+E_{\mathrm{rep}}\)
  [eV], **not** a band sum.

### Geometry

O₂, bond length 1.21 Å (near experimental), isolated cluster:

| Atom | Z | x [Å] | y | z |
|------|---|-------|---|---|
| O    | 8 | 0.00  | 0 | 0 |
| O    | 8 | 1.21  | 0 | 0 |

Charge 0, multiplicity **3** (\(n_\alpha=7\), \(n_\beta=5\)).
`scc_tol = 10^{-10}`, max 80 iterations. Forces off (energy + spin gate).

### Why \(\Delta E_{\mathrm{total}} = \Delta E_{\mathrm{spin}}\) exactly

This is an identity, not an estimate. Four steps, each named in
`src/cpu/semiempirical/dftb2.cpp`:

1. **Total energy is not a band sum.** The assembly is
   \(E=\mathrm{Tr}(\rho H^0)+E^{(2)}+E_{\mathrm{spin}}+E_{\mathrm{rep}}\).
   The spin shift never enters the first term.
2. **The shift is the same constant on both atoms.** O₂ is homonuclear ⇒
   \(\Delta q_A\equiv 0\) ⇒ Coulomb shift vanishes; \(\varepsilon^\sigma_A=\sigma W_A m_A\)
   is one constant \(c\).
3. **`apply_shift` then gives \(H^\sigma=H^0\pm cS\).** Equal on-site shifts
   make `half = ½(ε_a+ε_b)` identically \(c\).
4. **The generalised eigenproblem has the same eigenvectors.**
   \((H^0+cS)C=\varepsilon SC \iff H^0 C=(\varepsilon-c)SC\): eigenvectors
   unchanged, eigenvalues translated by \(c\). Therefore \(\rho^\alpha,\rho^\beta\)
   are unchanged ⇒ \(m_A\) stays self-consistently 1.0, \(\mathrm{Tr}(\rho H^0)\),
   \(E^{(2)}\) and \(E_{\mathrm{rep}}\) are unchanged.

The only term that moves is \(E_{\mathrm{spin}}\). For O₂ triplet,
\(m_A=1\) by symmetry so \(E_{\mathrm{spin}}=W_{\mathrm{pp}}(\mathrm{O},\mathrm{PBE})\).
Sketch \(W=-1\) eV → published \(W=-0.028\) Ha \(=-0.761918814888\) eV ⇒
\(\Delta E_{\mathrm{spin}}=+0.238081185112\) eV. **1e-4 eV is numerical-noise
margin on that equality.** A measured deviation beyond it is an iron-law-9
stop (`path:line` + the measured value) — never a reason to widen the
tolerance.

### Pins (hard-coded gates)

| Quantity | Value | Tol | Unit |
| -------- | ----- | --- | ---- |
| \(E_{\mathrm{spin}}=W_{\mathrm{pp}}(\mathrm{O},\mathrm{PBE})\) | **−0.761918814888** | \(10^{-9}\) | eV |
| Total energy \(E\) | **−177.004501172488** | \(10^{-4}\) | eV |
| \(E-E_{\mathrm{old}}\) vs superseded sketch-W pin −177.2425823576 | **+0.238081185112** | \(10^{-4}\) | eV |
| \(\sum_A m_A\) (Mulliken spin) | 2.0 | 0.25 | e |
| Per-atom \(\lvert m\rvert\) | > 0.3 each | — | e |
| \(\lvert m_0-m_1\rvert\) | < 0.2 | — | e |

Measured ATV (2026-09-04, Release, `OMP_NUM_THREADS=1`):
\(E=-177.004501172479\) eV, \(m=[1,1]\), \(\Delta q=[0,0]\),
\(E-E_{\mathrm{old}}=+0.238081185121\) eV
(\(|\mathrm{shift}-0.238081185112|=9\times10^{-12}\) eV).

### Offline DFTB+ 24.1

From archived `dftb.out.txt` (1 Ha = 27.211386245988 eV):

| Quantity | DFTB+ | Converted |
| -------- | ----- | --------- |
| Total energy | −6.5047954387 H | **−177.004501133607 eV** |
| Energy SPIN | −0.0280000000 H | **−0.761918814888 eV** |
| Energy SCC | 0 (Δq = 0) | 0 |
| Gross atomic spins | 1.000, 1.000 | e |

\(\lvert E_{\mathrm{ATV}}-E_{\mathrm{DFTB+}}\rvert=3.9\times10^{-8}\) eV
(≤ \(10^{-3}\) eV ⇒ `oracle: dftb+`). Spin populations and \(E_{\mathrm{spin}}\)
are bit-identical to the published \(W_{\mathrm{pp}}\).

### Perf

```text
perf_exempt: true
perf_exempt_reason: DFTB+ 24.1 wall is a cold process + I/O (~0.3 s); ATV
  Evaluate+DFTB2 is ~0.02 s. Not a fair kernel-vs-kernel wall.
```

### Units

Public ATV: energy **eV**, positions **Å**, Mulliken populations **e**.


## Reference — collinear spin-DFTB2 analytic nuclear forces (O₂)

```text
oracle: dftb+
oracle_version: 24.1
```

Offline twin: `tools/tb_oracle/inputs/dftbplus/dftb2-spin-analytic-forces/`
(`geo.gen`, `dftb_in.hsd`, archived `dftb.out.txt`). Same spin block as
`spin-dftb2-o2` (PBE \(W_{\mathrm{pp}}(\mathrm{O})=-0.028\) Ha,
`ShellResolvedSpin = No`). Cluster O₂, r = 1.30 Å. ctest never launches DFTB+.

The **module-boundary** gate remains analytic \(F\) vs re-SCC central FD of
the same ATV energy. Absolute \(E\) and \(F_x\) are additionally pinned
against DFTB+ 24.1.

### Method

Collinear unrestricted SCC-DFTB2 (Elstner SCC + Köhler spin channel). Nuclear
forces are the **closed-form** gradient at the SCC density (dual α/β
Hellmann–Feynman + Pulay + spin-dependent Hamiltonian shifts on \(\partial S\)
+ fixed-\(\Delta q\) SCC \(\gamma\) + repulsive). \(W_A\) is
geometry-independent; fixed-\(m\) spin energy contributes no extra nuclear
term. This is the DFTB+ collinear spin force module boundary
(Hourahine et al., *J. Chem. Phys.* **152**, 124101 (2020)).

Energy: \(\mathrm{Tr}(\rho H^0)+E^{(2)}+E_{\mathrm{spin}}+E_{\mathrm{rep}}\).

### Hamiltonian / tables

- ATV class: `cpu::semiempirical::DFTB2`, `sk_set = MioSketch` (transcribed
  mio-1-1), `boundary = cluster`, `multiplicity = 3`,
  `spin_w_ha = basis_data::kDftbSpinConstantsPbeHa`.

### Geometry (Å)

O₂ triplet, **stretched** so \(|F|\neq 0\):

| atom | Z | x | y | z |
| ---- | - | - | - | - |
| O | 8 | 0.00 | 0.00 | 0.00 |
| O | 8 | 1.30 | 0.00 | 0.00 |

Isolated cluster. `charge = 0`, `multiplicity = 3`, `scc_tol = 1e-10`.

### Pins (hard-coded gates)

1. **Energy:** finite, \(|E| > 10^{-3}\) eV.
2. **Forces:** all components finite; \(|F|^2 > 10^{-8}\) at off-eq geometry.
3. **Translation:** \(|\sum_i F_i| \le 10^{-6}\) eV/Å per Cartesian.
4. **Action–reaction:** \(F_{0,x}+F_{1,x}=0\) (within \(10^{-6}\)).
5. **FD identity:** analytic \(F\) vs central **re-SCC** FD of the same \(E\)
   (\(h=10^{-4}\) Å) on \(F_{0,x}\) and \(F_{1,x}\) within **0.05 eV/Å**.

Measured ATV (2026-09-04, Release, 1 thread):

```text
E = -176.77425642 eV
F[0] = ( 4.91147291, 0, 0) eV/Å
F[1] = (-4.91147291, 0, 0) eV/Å
F_0_x analytic=4.91147291  fd=4.91147232  |d|=5.9e-07 eV/Å
```

### Offline DFTB+ 24.1

From archived `dftb.out.txt` (1 Ha = 27.211386245988 eV;
1 Ha/Bohr = 51.42206747467393 eV/Å):

| Quantity | DFTB+ | Converted |
| -------- | ----- | --------- |
| Total energy | −6.4963341184 H | **−176.774256878772 eV** |
| Energy SPIN | −0.0280000000 H | −0.761918814888 eV |
| \(F_{0,x}\) | +0.095512870310 Ha/Bohr | **+4.9114692618 eV/Å** |
| \(F_{1,x}\) | −0.095512870310 Ha/Bohr | −4.9114692618 eV/Å |

\(\lvert E_{\mathrm{ATV}}-E_{\mathrm{DFTB+}}\rvert=4.6\times10^{-7}\) eV,
\(\lvert F_{0,x}^{\mathrm{ATV}}-F_{0,x}^{\mathrm{DFTB+}}\rvert=3.6\times10^{-6}\) eV/Å
(both ≤ \(10^{-3}\) ⇒ `oracle: dftb+`).

Homonuclear O₂ has \(\Delta q\equiv 0\), so this gate does not exercise the
charge-dependent \(\gamma\) gradient. That discriminator is the NO doublet in
`dftb2-spin-analytic-stress`.

### Perf

```text
perf_exempt: true
perf_exempt_reason: DFTB+ 24.1 cold wall includes I/O; the analytic path
  replaces 6N re-SCC force evaluations with one post-SCC contraction
```

### Units

Public ATV: energy **eV**, positions **Å**, forces **eV/Å**.


## Reference — collinear spin-DFTB2 analytic cell stress (O₂ triplet, NO doublet)

```text
oracle: dftb+
oracle_version: 24.1 (energy + forces). Cluster DFTB+ has no cell, so it
  cannot report stress; the stress gate is analytic vs strain-FD of the
  same ATV energy.
```

Offline twin: `tools/tb_oracle/inputs/dftbplus/dftb2-spin-analytic-stress/`
(`geo.gen`, `dftb_in.hsd`, archived `dftb.out.txt`). Cluster (`C`) O₂ with
the regression's tilted bond \(|R|=1.30\) Å along
\(\hat u=(0.6,0.7,\sqrt{0.15})\). ATV uses `DftbBoundary::cluster` and a
cubic 20 Å box only to define \(\Omega\) for the intensive virial — that is
still a cluster Hamiltonian. A DFTB+ `S` geometry would add Ewald images ATV
does not use, so the twin stays `C`. **No DFTB+ stress number is claimed
and none is invented.**

Same spin block as `spin-dftb2-o2`. ctest never launches DFTB+.

### Method

Collinear unrestricted SCC-DFTB2:

\[
E = \mathrm{Tr}(\rho^\alpha H^0)+\mathrm{Tr}(\rho^\beta H^0)
  + \tfrac12\sum_{AB}\gamma_{AB}\,\Delta q_A\Delta q_B
  + \tfrac12\sum_A W_A m_A^2
  + E_{\mathrm{rep}}
\]

Intensive cell stress (mechanical / DFTB+ / PW–GPW convention):

\[
\sigma_{\alpha\beta}=-\frac{1}{\Omega}\frac{\partial E}{\partial\varepsilon_{\alpha\beta}},
\qquad \text{Voigt } (xx,yy,zz,xy,xz,yz)\ \ [\mathrm{eV/\AA^3}]
\]

evaluated **closed form** at the SCC density as the pair virial of the same
bond terms as the analytic nuclear force (Hellmann–Feynman, Pulay including
the \(\pm W_A m_A\) half of \(\varepsilon^\sigma_A\), SCC \(\gamma\),
repulsive). \(E_{\mathrm{spin}}\) has no explicit strain dependence; the
implicit \(m_A(\varepsilon)\) response reduces variationally to that Pulay
term. It is included; it is not assumed to be zero.

### Hamiltonian / tables

- ATV class: `cpu::semiempirical::DFTB2` (`sk_set = MioSketch`, transcribed
  mio-1-1), `boundary = cluster`,
  `spin_w_ha = basis_data::kDftbSpinConstantsPbeHa`.

### Geometry (Å)

Both molecules use the tilted bond
\(\hat u = (0.6,\,0.7,\,\sqrt{0.15})\) so all six Voigt components are
non-trivial.

**Case 1 — O₂ triplet (headline).** `charge = 0`, `multiplicity = 3`,
\(r = 1.30\) Å:

| atom | Z | x | y | z |
| ---- | - | - | - | - |
| O | 8 | 0.00 | 0.00 | 0.00 |
| O | 8 | 0.78 | 0.91 | 0.503487835 |

**Case 2 — NO doublet (discriminator).** `charge = 0`, `multiplicity = 2`,
\(r = 1.20\) Å. O₂ is degenerate: \(\Delta q\equiv 0\) and the spin shift is
uniform, so a gate built on O₂ alone would pass an implementation that never
wrote the spin virial. NO restores both \(\Delta q\neq 0\) and
\(W_{\mathrm{N}}m_{\mathrm{N}}\neq W_{\mathrm{O}}m_{\mathrm{O}}\). No DFTB+
twin is archived for NO (this spec's twin is O₂).

Isolated cluster, cubic 20 Å box (\(\Omega=8000\) Å³; every lattice image is
beyond the SK cutoff). `scc_tol = 1e-12`, `max_scc = 200`,
`with_forces = true`, `with_stress = true`.

### Pins (hard-coded gates)

Per case:

1. **Energy:** finite, \(|E| > 10^{-3}\) eV.
2. **Translation:** \(|\sum_i F_i|\le 10^{-6}\) eV/Å per Cartesian.
3. **Stress:** all six Voigt components finite.
4. **Strain-FD identity:** analytic \(\sigma_v\) vs central affine strain-FD of
   the same converged \(E\) (\(h=2\times10^{-4}\)), **all six** components,
   within **1×10⁻⁸ eV/Å³**.

Measured ATV (2026-09-04, Release, 1 thread):

```text
O2 triplet   E = -176.77425642 eV
  sigma[0] analytic=-2.87321165e-04  fd=-2.87321148e-04  |d|=1.753e-11
  sigma[1] analytic=-3.91076031e-04  fd=-3.91075973e-04  |d|=5.730e-11
  sigma[2] analytic=-1.19717152e-04  fd=-1.19717153e-04  |d|=6.681e-13
  sigma[3] analytic=-6.70416052e-04  fd=-6.70415688e-04  |d|=3.641e-10
  sigma[4] analytic=-3.70930029e-04  fd=-3.70929985e-04  |d|=4.402e-11
  sigma[5] analytic=-4.32751701e-04  fd=-4.32751627e-04  |d|=7.429e-11
NO doublet   E = -152.83740700 eV
  sigma[0] analytic=-2.67630972e-04  fd=-2.67630947e-04  |d|=2.476e-11
  sigma[1] analytic=-3.64275489e-04  fd=-3.64275402e-04  |d|=8.689e-11
  sigma[2] analytic=-1.11512905e-04  fd=-1.11512907e-04  |d|=1.906e-12
  sigma[3] analytic=-6.24472267e-04  fd=-6.24471689e-04  |d|=5.780e-10
  sigma[4] analytic=-3.45510099e-04  fd=-3.45510034e-04  |d|=6.500e-11
  sigma[5] analytic=-4.03095115e-04  fd=-4.03095004e-04  |d|=1.115e-10
```

### Offline DFTB+ 24.1 (O₂ only)

From archived `dftb.out.txt`. Same \(|R|=1.30\) Å as the linear force twin,
so the energy is identical (orientation does not enter a cluster Hamiltonian):

| Quantity | DFTB+ | Converted |
| -------- | ----- | --------- |
| Total energy | −6.4963341184 H | **−176.774256878772 eV** |
| Energy SPIN | −0.0280000000 H | −0.761918814888 eV |
| \(F_1\) | (0.057307719745, 0.066859006948, 0.036991973513) Ha/Bohr | (2.94688143, 3.43802837, 1.90220376) eV/Å |
| \(F_2\) | opposite | opposite |

\(\lvert E_{\mathrm{ATV}}-E_{\mathrm{DFTB+}}\rvert=4.6\times10^{-7}\) eV
(≤ \(10^{-3}\) ⇒ energy/forces `oracle: dftb+`). The DFTB+ force lies along
\(\hat u\) with magnitude 4.91146906 eV/Å, matching the linear 1.30 Å twin.
**No stress tensor is present in `detailed.out` or `results.tag`.**

### Gaps (still open)

1. **D3 dispersion is absent from the stress** unless `d3 = true` (the
   closed-shell D3 virial lives on `dftb2-analytic-stress`).
2. **No periodic open-shell stress twin.** This geometry is a molecule in a
   20 Å box; self-image virial is unit-gated in
   `tests/cpu/semiempirical/test_dftb2.cpp`.
3. **Cluster DFTB+ cannot pin \(\sigma\).** Closing an absolute stress pin
   would need a periodic Hamiltonian ATV actually uses, not a dummy `S` box.

### Citations

- M. Elstner et al., Phys. Rev. B **58**, 7260 (1998). DOI 10.1103/PhysRevB.58.7260.
- C. Köhler, G. Seifert, U. Gerstmann, M. Elstner, H. Overhof, Th. Frauenheim,
  Phys. Chem. Chem. Phys. **3**, 5109 (2001). DOI 10.1039/b105782k.
- C. Köhler, Th. Frauenheim, B. Hourahine, G. Seifert, M. Sternberg,
  J. Phys. Chem. A **111**, 5622 (2007). DOI 10.1021/jp068802p.
- B. Hourahine et al., J. Chem. Phys. **152**, 124101 (2020). DOI 10.1063/1.5143190.

### Perf

```text
perf_exempt: true
perf_exempt_reason: DFTB+ 24.1 cluster run has no stress to wall-clock
  against; analytic virial replaces 12 multi-SCC strain evaluations
```

### Units

Public ATV: energy **eV**, positions **Å**, forces **eV/Å**, stress
**eV/Å³** Voigt \((xx,yy,zz,xy,xz,yz)\).


## dftb-plus-u

**oracle:** literature (formula)
**oracle_version:** Dudarev PRB 57 1505 (1998); design `.claude/notes/dftb-plus-u.md`

### Method

SCC-DFTB2 (MioSketch) water + oxygen \(U_{\mathrm{eff}}=4\) eV. Energy-only.

### Gates

- \(E_U = \tfrac12 U n(1-n/N)\) on O with \(n=n_{\mathrm{val}}-\Delta q\), \(N=2n_{\mathrm{AO}}\)
- \(E(U)-E(0)=E_U\)
- Nuclear forces: central FD of \(E_{\mathrm{SCC}}+E_U\) matches independent energy FD (tol 5e-4 eV/Å)

### Gaps

- No absolute DFTB+ +U pin (Dudarev shell details differ by engine).
- Stress / multi-k / shell-resolved \(U_\ell\) still open.
- Forces are re-SCC FD, not analytic \(\partial n/\partial R\).
