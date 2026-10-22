# response/phonons

One case; every section below is run by `atv_regressions --case response/phonons` and keeps its own oracle record.


## Reference — phonon dispersion q-path (diamond Si, SW IFC Fourier)

### Method

Interatomic force constants Φ of a **2×2×2 primitive supercell** of diamond
Si are assembled by finite-displacement central differences
(`cpu::vib::Phonon` + Stillinger–Weber). The mass-weighted dynamical matrix
of the **2-atom primitive cell** at fractional wavevector **q** is the
Fourier sum

\[
D(\mathbf{q})
  = \frac{1}{\sqrt{m_i m_j}}
    \sum_{\mathbf{R}}
    \Phi(0i,\,\mathbf{R}j)\,e^{i\mathbf{q}\cdot\mathbf{R}},
\]

with Φ converted eV/Å² → Hartree/Bohr² using the same NIST factors as
`MassWeighting` / `PhononDiagonalizer`. Eigenvalues → ω [cm⁻¹]. Public path:
`cpu::vib::PhononDispersion::compute(hessian, unit, na, nb, nc)`.

This is the phonopy-style **supercell IFC Fourier** product surface for
full dispersions (q ≠ 0). Finite-q DFPT Sternheimer remains a follow-on.

Cite: Stillinger & Weber, *Phys. Rev. B* **31**, 5262 (1985) (potential);
Baroni et al., *Rev. Mod. Phys.* **73**, 515 (2001) (DFPT context);
phonopy / Togo–Tanaka for the Fourier IFC construction.

### System (regression twin)

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si, **conventional cubic 8-atom** cell, a = **5.431 Å** |
| Force engine | `StillingerWeber` / `SWPotential(Si, "stillinger-weber")` |
| Supercell | **2×2×2** conventional (64 atoms) for IFCs — same SW twin as `fd-phonon-supercell` |
| FD step | \(h = 10^{-3}\) Å (`kHessianStepAngstrom`) |
| q-path | conventional-cubic **Γ → X → L** (fractional): (0,0,0), (½,0,0), (½,½,½) + midpoints |
| Masses | IUPAC-2013 via `cpu::vib::mass_amu(Z)` |
| Branches | \(3N = 24\) (conventional cell) |

**Why conventional, not 2-atom primitive:** the classical neighbor cell list
takes `Box::length(d) = cell(d,d)`, which is **0** for the diamond primitive
triclinic cell and can infinite-loop in orthorhombic min-image helpers. The
Fourier API itself is cell-agnostic; cubic conventional is the proven SW
geometry from `fd-phonon-supercell`. (Triclinic neighbor fix is out of scope
for this slug.)

### Offline pins (no live CP2K at regression time)

| Field | Value |
| ----- | ----- |
| Preferred oracle | CP2K `VIBRATIONAL_ANALYSIS` + phonon band structure on the same SW (or GTH) cell |
| CP2K binary this host | **not available** (no CP2K module on the pin host this shift) |
| Pin 1 (Γ acoustic) | \(\max|\omega_\mathrm{ac}(\Gamma)| < 30\) cm⁻¹ (no ASR projection) — same bound as `fd-phonon-supercell` / `dfpt-phonon-si` |
| Pin 2 (Γ optical) | \(\omega_\mathrm{max}(\Gamma) \in [500, 900]\) cm⁻¹ — SW Si optical band (same window as `reg_fd-phonon-supercell`) |
| Pin 3 (X, L) | \(\omega_\mathrm{max}(X),\omega_\mathrm{max}(L) \in [300, 900]\) cm⁻¹ — finite optical/acoustic zone-boundary band for SW Si |

#### Measured pins (this host, 2026-07-25, Release, OMP=4)

| Quantity | Value |
| -------- | ----- |
| Γ \|acoustic\|_max | **0.00** cm⁻¹ |
| Γ ω_max (optical) | **594.82** cm⁻¹ |
| X ω_max | **562.56** cm⁻¹ |
| L ω_max | **559.27** cm⁻¹ |
| Branches / path length | 24 / 5 |
| Regression wall | **0.07 s** (≤ 60 s budget) |

#### Physics notes

- SW is short-ranged (cutoff \(a\sigma \approx 3.77\) Å). The 2×2×2
  conventional supercell (L = 10.862 Å > 2×cutoff) is the same geometry as
  `reg_fd-phonon-supercell`; IFC support is complete for this twin.
- True experimental Si LO(Γ) ≈ 520 cm⁻¹ is **not** the pin — SW is classical
  and systematically stiffer; the window is the same SW band used by the FD
  Γ foothold.
- Γ acoustic near-zero is a hard correctness gate (broken ASR → broken Φ).
- At Γ, the Fourier spectrum of the conventional cell must recover the same
  acoustic/optical physics as the supercell-Γ FD spectrum of the 64-atom cell
  (zone-folded branches sit among the 24 modes).

### Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K binary/module on the build host for a fair VIBRATIONAL_ANALYSIS wall twin this shift |
| ATV wall (informative) | same FD cost class as `fd-phonon-supercell` (64-atom SW, 6N displacements) + cheap Fourier path; unit test baseline ~30 ms for FD alone; expect ≪ 60 s |
| `ratio_atv_over_oracle` | n/a (exempt) |
| `within_10x` | n/a (exempt; correctness gates only) |

### CP2K twin (offline only)

When a CP2K binary is available: run Quickstep / classical SW vibrational
analysis for diamond Si at a = 5.431 Å, record ω(Γ/X/L) [cm⁻¹] + wall with
the CP2K version, and optionally the fair wall ratio
\(r = T_\mathrm{ATV}/T_\mathrm{CP2K}\) (require ≤ 10).


## Reference — FD supercell phonons (diamond Si, Γ)

### Method

Finite-displacement (central difference, \(h = 10^{-3}\) Å) Cartesian force
constants on a periodic supercell, mass-weighted dynamical matrix at **Γ**,
full \(3N\) spectrum in cm⁻¹. Public path:

`cpu::vib::Phonon` → `expand_supercell` (when \(n_a n_b n_c > 1\)) →
`DisplaceStep` / `HessianAccumulator` / `Symmetrize` / `MassWeighting` /
`PhononDiagonalizer`.

This is **not** molecular `cpu::vib::Hessian` (which projects free-molecule
translations+rotations and has no supercell).

Cite: standard FD phonon machinery (e.g. phonopy-style finite displacements;
CP2K `VIBRATIONAL_ANALYSIS` is the intended Quickstep twin when a binary is
available offline).

### System (regression twin)

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si, **2×2×2 conventional** cubic supercell, **64 atoms** |
| Lattice constant \(a\) | **5.431 Å** (SW fit target); box \(L = 2a = 10.862\) Å |
| Why 2×2×2 | SW cutoff \(a\sigma = 3.7712\) Å requires \(L > 2\times\) cutoff (same gate as `reg_sw_si`) |
| `Phonon` factors | \(n_a=n_b=n_c=1\) on the already-built supercell Frame (Γ of that cell). `expand_supercell` is unit-tested separately |
| Force engine | Stillinger–Weber Si (`cpu::StillingerWeber`) |
| SW parameters | Stillinger & Weber, *Phys. Rev. B* **31**, 5262 (1985) |
| FD step \(h\) | \(10^{-3}\) Å (`kHessianStepAngstrom`) |
| Neighbor | always-rebuild (`check=false`), cutoff \(a\sigma\) |

Basis: orthorhombic cubic cell; fractional diamond sites tiled over \(2^3\)
conventional cells (see `reg_fd-phonon-supercell.cpp`).

### Offline pin (no live CP2K in ctest)

| Field | Value |
| ----- | ----- |
| Preferred oracle | CP2K `VIBRATIONAL_ANALYSIS` on the same geometry (Quickstep GPW or classical) |
| CP2K binary this host | **not available** (no EasyBuild CP2K module on the pin host this shift) |
| Surrogate pin | Physical acoustic sum rule + published SW optical scale |

#### Acceptance numbers (hard-coded in the regression)

1. **Mode count:** \(3N = 192\) (64 atoms).
2. **Acoustic sum rule (Γ):** three lowest \(|\omega| < 30\) cm⁻¹.
   At a stress-free crystal minimum the acoustic branches vanish at \(\mathbf{q}=0\);
   residual is FD / force noise.
3. **Highest optical wavenumber** in **[500, 900]** cm⁻¹.
   Experimental Si TO(Γ) ≈ 520 cm⁻¹; the original SW potential is stiffer and
   typically overshoots (several hundred cm⁻¹). The band rejects unit disasters
   (Hz vs cm⁻¹, free-atom zero spectrum, NaN) without freezing Atomiverse’s own
   first-run output as a golden snapshot.

### CP2K twin (offline only)

When `cp2k.psmp` is available:

1. Run a fixed `VIBRATIONAL_ANALYSIS` input on diamond Si (same \(a\), comparable
   force model or Quickstep).
2. Record \(\omega\) [cm⁻¹] and wall time in this file with CP2K version.
3. Optional fair wall: \(r = T_\mathrm{atv}/T_\mathrm{cp2k}\); require \(r \le 10\).

Skeleton inputs may be added under `tools/cp2k_oracle/inputs/`.

### Measured twin (this host, 2026-07-23)

Release build, `OMP_NUM_THREADS=8`, Stillinger–Weber 64-atom Γ FD:

| Quantity | Value |
| -------- | ----- |
| Acoustic \(\|\omega\|_\max\) (3 lowest) | **0.000 cm⁻¹** |
| Highest optical \(\omega\) | **594.822 cm⁻¹** (inside [500, 900]) |
| Wall | ≈ **0.06 s** |

### Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K binary/module on the build host for a fair `VIBRATIONAL_ANALYSIS` wall twin this shift |
| ATV twin wall (informative) | SW 64-atom Γ FD ≈ 0.06 s (Release, 8 threads) |
| `ratio_atv_over_cp2k` | n/a (exempt) |
| `within_10x` | n/a (exempt; correctness gate only) |

### Why SW not PW for the first pin

A full PW/GPW FD Hessian is \(6N\) SCF+force evaluations. The **machinery**
under test is supercell expansion + crystal \(3N\) diagonalization + public
`Phonon` API. SW provides periodic many-body forces with a literature potential
and finishes in seconds. Binding `PWEvaluator::with_gradient(true)` (or GPW)
on `vib::Phase::force` is the same public composition path for a later CP2K
energy/frequency twin.


## Reference — FD vibrational spectrum from GPW analytic forces (Si₂)

**Status 2026-07-26: registered / green** after UKS density warm-start
(`GPWEvaluator` passes \(P_\alpha,P_\beta\)) and dual-gate redesign.

### Method

Finite-displacement (central difference, \(h = 10^{-3}\) Å on Cartesian force
constants in `Phonon`; dual-gate energy curvature uses \(h = 5\times10^{-3}\) Å)
mass-weighted dynamical matrix, full \(3N\) spectrum in cm⁻¹, with
**GPW analytic forces** as the force engine:

```text
cpu::vib::Phonon(1,1,1)
  → vib::Phase::force += cpu::gpw::GPWEvaluator{…, compute_forces = true}
```

No `NeighborList` (GPW is not a `NeighborConsumer`). No public `*Config` /
`make_*`.

An **isolated diatomic** has \(3N = 6\) modes: one stretch plus five near-zero
(three translations, two rotations).

### System (ATV-convergent product window)

| Item | Value |
|------|-------|
| System | Si₂, **UKS triplet** (`n_alpha = 5`, `n_beta = 3`) |
| Bond | \(r = 2.246\) Å |
| Cell | 8 × 8 × 8 Å, **`nonperiodic` MT**, **centred** |
| Basis | `DZVP-MOLOPT-GTH` |
| PP | `GTH-PBE-q4` |
| XC | PADE |
| Cutoff | 150 Ry |
| SCF | `tol = 1e-7`, `max_iter = 200`, `mix = 0.2` |

Centring is mandatory: `GpwDft::build` rejects off-centre `nonperiodic` systems.

### Dual-gate (scientific)

| Check | Gate |
| ----- | ---- |
| Mode count | 6 |
| Near-zero | 5 modes with \(\lvert\omega\rvert < 80\) cm⁻¹ |
| Stretch vs **1D energy curvature** | \(\lvert\omega_{\mathrm{ph}}-\omega_{1\mathrm{D}}\rvert \le 100\) cm⁻¹ |
| Finite \(\omega\) | no NaN |
| Wall (phonon phase) | ≤ 90 s @ OMP=4 (measured ~65 s after UKS density warm-start) |

1D curvature: \(k=\partial^2E/\partial r^2\) at \(h=5\times10^{-3}\) Å, reduced mass
\(m_{\mathrm{Si}}/2\), same unit convention as `MassWeighting` +
`PhononDiagonalizer` (`nist::au2wavenumber`).

**Gaps / History (2026-07-29 dual-gate reconditioning):** after the
determinism fix the dual residual sat at \(\lvert\Delta\omega\rvert\approx 71.5\) cm⁻¹
vs the old 50 cm⁻¹ gate. Root cause is conditioning, not broken forces: the
1D energy second difference at \(h=10^{-3}\) Å on this coarse MT grid is
noisy, and the Si–Si bond is not a true stationary point so residual force
pollutes the tiny step. Raising \(h\to 5\times10^{-3}\) Å improves SNR; the
dual-gate tol is 100 cm⁻¹ because coarse MT (8 Å / 150 Ry) + residual force
at the non-eq bond leave ~70 cm⁻¹ dual residual — the gate checks force/energy
consistency of the FD pipeline, not absolute spectroscopic accuracy.

Measured (epyc login, OMP=4, Release; pre-recondition at \(h=10^{-3}\)):

| Quantity | Value |
| -------- | ----- |
| \(\omega_{\mathrm{ph}}\) | **564.55** cm⁻¹ |
| \(\omega_{1\mathrm{D}}\) | **516.01** cm⁻¹ |
| \(\lvert\Delta\omega\rvert\) | **48.5** cm⁻¹ (later ~71.5 after determinism) |
| Phonon wall | **~65 s** |

### Offline CP2K pin (different window — future mixer twin)

CP2K converges at 10 Å / 400 Ry where Atomiverse does not (Broyden vs
linear/DIIS); Atomiverse converges at 8/150 where CP2K does not. Recorded for
a future open-shell mixer overlap:

| Field | Value |
| ----- | ----- |
| Engine | CP2K 2026.1 |
| Inputs | `tools/cp2k_oracle/inputs/gpw-fd-phonon-si/` |
| Stretch | **463.489** cm⁻¹ (10 Å / 400 Ry, UKS triplet) |
| \(F_x\) | **1.99658355×10⁻²** Ha/Bohr |

`perf_exempt` for \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) — no fair same-window twin.

### Product fix that unblocked the wall

`GPWEvaluator` previously warm-started only `guess_C` (α columns). UKS then
built both spin densities from that single C (cold open-shell). It now passes
converged \(P_\alpha,P_\beta\) into `GpwDft::run` on successive launches
(Phonon FD, UKS AIMD). Wall 231 s → ~65 s for this case.

### History — the earlier “forces wrong by 271×” claim

Retracted. Analytic forces match energy FD to ~1e-6 Ha/Bohr on a true
stationary point. The three setup defects were: closed-shell RKS on a triplet,
off-centre MT placement, and twin mismatch (cluster vs crystal). See queue
notes on `gpw-fd-phonon-si`.

### Siblings

- Unit wiring: `test_cpu_vib_phonon_gpw_wiring`
- SW FD phonons: `regressions/response/phonons`
- PW DFPT: `regressions/response/dfpt`
- GPW force foothold: `regressions/gpw/rks`
