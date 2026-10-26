# Plane-wave Berry-phase polarization

> The regression runner has been removed. Regression commands below are historical; case sources and oracle records are retained as archived evidence, not current executable checks.

King–Smith–Vanderbilt (KSV) formal polarization on the periodic plane-wave
KS-DFT path: a `cpu::pw::PW` SCF, then `cpu::pw::BerryPolarization::compute`
(members `k_perp`, `k_string`, `directions`) freezes \(V_\mathrm{eff}\),
diagonalizes the occupied bands non-self-consistently on strings of k-points
along each reciprocal lattice vector, and returns an owned
`BerryPolarizationResult` (phases, electronic / ionic / total fractional
polarization, dipole, SCF energy in Ry) — not Frame meta. Collinear spin is an
explicit member path (`PW::spin_polarized`, `PW::multiplicity`), not a silent
fallback. CPU backend only; the case requires `ATV_BUILD_PW=ON` (CPU build
environment `buildenv-gcc/2026.03-mpich`).

The case also carries the closed-shell section `berry-polarization` (restricted
LDA Si, same symmetry pins), which has no tutorial page; its oracle record lives
in the case `REFERENCE.md`. The page below documents the collinear-spin section.

Case: `ctest -R '^pw/berry-phase$'` — sources and oracle record in [`regressions/pw/berry-phase/`](../../../regressions/pw/berry-phase/). One section of the case per `##` heading below: `atv_regressions --case pw/berry-phase --section <name>`.

Roadmap: [dft.md](../../roadmap/dft.md) §11 Berry-phase polarization.

## Oracle

The gates are **parameter-free literature symmetry pins** — a centrosymmetric
formal \(p_i\) on the half-quantum lattice (Spaldin) and the Born effective
charge \(Z^*=0\) of an elemental crystal (Pick–Cohen–Martin) — hard-coded in the
section sources; nothing runs CP2K or QE at ctest time. CP2K
`&MOMENTS PERIODIC T` documents the same modern-theory observable on
**GTH-PADE + DZVP-MOLOPT-SR-GTH**, not ONCV plane waves, so absolute Berry
phases and formal \(p_i\) branch choices are **not** comparable across PP/basis.
Absolute phases are re-pinned only against a fair NC-UPF PW twin (e.g. QE) when
one is available offline.

## References

- R. D. King-Smith and D. Vanderbilt, *Phys. Rev. B* **47**, 1651 (1993)
- R. Resta, *Rev. Mod. Phys.* **66**, 899 (1994)
- N. A. Spaldin, *J. Solid State Chem.* **195**, 2 (2012) — centrosymmetric lattice
- P. Pick, M. H. Cohen and R. M. Martin, *Phys. Rev. B* **1**, 910 (1970) — elemental \(Z^*=0\)

## berry-spin-si — Collinear-spin Berry-phase polarization of bulk Si {#berry-spin-si}

### Summary

`PW` with `spin_polarized = true` and fixed occupations, then
`BerryPolarization::compute` on frozen spin-resolved \(V_\mathrm{eff}\). Each
spin channel owns its occupied manifold; string-averaged Berry phases
\(\varphi_{i,\uparrow}\) and \(\varphi_{i,\downarrow}\) are combined with
**spin factor 1 per channel** and the total polarization is wrapped with quantum
\(1\) (nspin=2 convention). The system is diamond silicon (multiplicity 1,
\(N_\uparrow=N_\downarrow=4\)) with the same basis-independent pins as the
closed-shell `berry-polarization` section. Performance: **`perf_exempt`**.

### Method

**String geometry.** For direction \(i\) (lattice axis 0,1,2), a
\(N_\perp\times N_\perp\) mesh of strings is laid in the transverse plane;
each string has \(J\) points with fractional component
\(k_i = j/J\), \(j=0,\ldots,J-1\) (`k_perp` \(=N_\perp\),
`k_string` \(=J\)). Link matrices are occupied–occupied overlaps between
adjacent string points; the closing link \(k_{J-1}\to k_0+\mathbf b_i\) uses
the periodic-gauge coefficient shift
\(c_{n,k_0+\mathbf b}(\mathbf G)=c_{n,k_0}(\mathbf G+\mathbf b)\).

**Collinear electronic polarization.** Per direction \(i\) and spin
\(\sigma\in\{\uparrow,\downarrow\}\), the string-averaged Berry phase is

\[
\varphi_{i,\sigma}
  = \operatorname{Im}\ln\prod_j \det S^{(j)}_{i,\sigma},
\]

with unimodular link determinants (phase of the product). Electronic
polarization (fractional elementary charge \(e\)) is

\[
p^\mathrm{el}_i
  = \frac{\varphi_{i,\uparrow}+\varphi_{i,\downarrow}}{2\pi}
\quad\text{(factor \(f=1\) per channel)}.
\]

`BerryPolarizationResult::phase_el[i]` stores the **mean** phase
\((\varphi_{i,\uparrow}+\varphi_{i,\downarrow})/2\) so that, when
\(N_\uparrow=N_\downarrow\), the closed-shell identity
\(p^\mathrm{el}=2\varphi/2\pi\) holds with that mean. Total formal
polarization wraps into \((-0.5,0.5]\) (quantum \(1\)). The restricted path
(`spin_polarized == false`, section `berry-polarization`) uses spin factor
\(f=2\), \(p^\mathrm{el}=2\varphi/2\pi\), wrap quantum \(2\). With multiplicity
1 and even \(N_e\), collinear KSV recovers the same physics modulo the smaller
nspin=2 quantum.

**Ionic term.** Spin-independent:

\[
p^\mathrm{ion}_i
  = \sum_a Z^\mathrm{val}_a\, \tau_{a,i},
\qquad
\tau_{a,i}\in[0,1).
\]

For diamond Si with \(Z^\mathrm{val}=4\) at fractional sites
\((0,0,0)\) and \(\tfrac14(1,1,1)\) (ideal), each lattice component is
exactly \(1\).

**Born effective charge (elemental ASR).** Finite sublattice displacement
\(\delta\) of atom 1 along \(\mathbf a_3\) defines

\[
Z^*
  = \frac{\operatorname{wrap}\bigl(p_3(\delta)-p_3(0);\, q=1\bigr)}{\delta}.
\]

For a monoatomic crystal the acoustic sum rule requires \(Z^*=0\).

**XC / occupations.** `XCType::LSDA` on the PW grid; collinear fixed
occupations from `multiplicity` (here \(m=1\) → \(N_\uparrow=N_\downarrow\)).
**Metallic smearing is rejected** — formal Berry polarization requires an
insulator.

**Observables.**

| Symbol | Definition | Unit |
| ------ | ---------- | ---- |
| \(\varphi_i\) | `phase_el[i]` (mean of spin phases) | rad |
| \(p^\mathrm{el}_i\) | `p_el_frac[i]` | \(e\) (fractional) |
| \(p^\mathrm{ion}_i\) | `p_ion_frac[i]` | \(e\) (fractional) |
| \(p_i\) | `p_frac[i]` total, wrapped (quantum 1) | \(e\) (fractional) |
| \(\mathbf d\) | `dipole_e_ang` \(=\sum_i p_i\mathbf a_i\) | e Å |
| \(E_\mathrm{SCF}\) | `energy_ry` | Ry |

### Implementation

```cpp
constexpr double kA = 5.43;       // Å
constexpr double kDelta = 0.02;   // fractional displacement of atom 1 along a_3

// diamond Si, 2-atom fcc primitive (type 14, mass 28.0855); atom 1 at
// a/4.0 (1,1,1) + delta_frac * a_3
cpu::Frame make_si_diamond(double a_ang, double delta_frac_a3);

PW method{std::make_shared<UpfPseudopotential>(ATV_REG_DATA_DIR "/pw/Si_lda_nc.upf"), /*ecut Ry*/ 15.0,
          /*k_mesh 2×2×2*/ 2, cpu::scf::XCType::LSDA};
method.tol = 1e-6;
method.max_iter = 50;
method.spin_polarized = true;
method.multiplicity = 1;          // N↑ = N↓ = 4 (8 valence e⁻)

BerryPolarization berry;
berry.k_perp = 2;
berry.k_string = 4;
const BerryPolarizationResult r0 = berry.compute(method, make_si_diamond(kA, 0.0));
// r0.p_frac[d], r0.p_ion_frac[d], r0.phase_el[d], r0.scf_converged

BerryPolarization berry_z;
berry_z.k_perp = 2;
berry_z.k_string = 4;
berry_z.directions = {false, false, true};  // only a_3 for Z*
const BerryPolarizationResult r1 = berry_z.compute(method, make_si_diamond(kA, kDelta));
const double dp = wrap_centered(r1.p_frac[2] - r0.p_frac[2], /*quantum=*/1.0);
const double z_born = dp / kDelta;
```

`compute` throws `std::invalid_argument` for metallic smearing or an invalid
string mesh (`k_perp < 1`, `k_string < 2`). Headers:
`cpu/planewave/berry_polarization.h`, `cpu/planewave/pw_solver.h`;
implementation `src/cpu/planewave/berry_polarization.cpp` (collinear SCF →
freeze \(V_\mathrm{eff}^\sigma\) → NSCF string phases per spin → wrap quantum 1).
No unit test in the tree covers string layout, direction axes, link-det phase
math, `wrap_centered` or the invalid-argument guards.

### Validation

#### Pseudopotential

`tests/data/pw/Si_lda_nc.upf` (in-tree), PseudoDojo ONCVPSP-LDA-SR-noCC v0.4,
\(Z_\mathrm{val}=4\) — same family as [pw-lda-si](pw-rks.md#pw-lda-si),
`berry-polarization` and [bands-dos-si](pw-bands-dos.md#bands-dos-si).

#### Geometry and settings

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si, 2-atom fcc primitive cell |
| Lattice constant \(a\) | 5.43 Å |
| Basis atoms (ideal) | \((0,0,0)\), \((a/4)(1,1,1)\) |
| Displaced twin | atom 1 \(+0.02\,\mathbf a_3\) |
| Cell | \((a/2)(0,1,1)\), \((a/2)(1,0,1)\), \((a/2)(1,1,0)\) |
| SCF \(E_\mathrm{cut}\) | 15 Ry (tutorial; ONCV production ~60 Ry) |
| SCF k-mesh | \(2\times2\times2\) |
| XC | `XCType::LSDA` |
| Spin | collinear, multiplicity 1 (\(N_\uparrow=N_\downarrow=4\)) |
| SCF tol | \(10^{-6}\) |
| Strings | `k_perp = 2`, `k_string = 4` |

#### Accuracy table

| Quantity | Atomiverse / pin | Reference | Tol | Notes |
| -------- | ---------------: | --------: | --- | ----- |
| SCF (ideal + displaced) | `scf_converged` | must converge | — | both collinear runs |
| \(p^\mathrm{ion}\) (ideal) | \((1,1,1)\) | exact ion bookkeeping | \(1\times10^{-9}\) | \(Z_\mathrm{val}=4\), diamond sites |
| \(\Delta p^\mathrm{ion}_3\) (displaced) | \(0.08\) | \(Z_\mathrm{val}\times\delta=4\times0.02\) | \(1\times10^{-9}\) | only \(a_3\) moved |
| Centrosymmetric \(p_i\) (ideal) | distance to \(\{0,\tfrac12\}\) lattice | Spaldin *JSSC* **195**, 2 (2012) | **&lt; 0.05** | half-quantum lattice; quantum-1 branch |
| \(\|Z^*(\mathrm{Si})\|\) | \(\|\Delta p_3\|/\delta\) | Pick–Cohen–Martin *PRB* **1**, 910 (1970) | **&lt; 0.5** | elemental ASR |
| Measured (informative) | \(\varphi=0\), \(p=0\), \(Z^*=0\) | same pins | exact to print | Release, ecut 15 / k=2 |

The CP2K input echo (same diamond geometry, PADE / LSD, multiplicity 1,
`&MOMENTS PERIODIC T`) is kept offline and never executed at ctest time.

### Performance

**`perf_exempt`**: no loadable CP2K/QE PW binary on the build host for a fair
wall twin (same host constraint as `berry-polarization`), and GTH phases are not
a fair twin; \(T_\mathrm{CP2K}\), the ratio and the ≤10× check are n/a. ATV wall
(informative): **32.05 s** (Release; 2 collinear SCF + string sweeps, ecut 15 /
k=2), inside the ≤ 60 s single-rank CPU budget. When a fair NC-UPF PW oracle is
available offline, time the same geometry and string density, record
\(r=T_\mathrm{ATV}/T_\mathrm{oracle}\) in REFERENCE.md and require \(r\le 10\).

### Limitations

- **Tutorial ecut / k / string mesh only** — not production-converged
  polarization magnitudes or dense-string branch resolution; production
  \(E_\mathrm{cut}\) / dense strings break the ≤ 60 s budget.
- **Insulators only** — metallic smearing is a hard throw; no
  Fermi-surface Berry corrections.
- **Collinear nspin=2 only** — noncollinear / SOC out of scope; closed-shell is
  the `berry-polarization` section.
- **Multiplicity 1 Si** — open-shell magnets (unequal \(N_\uparrow,N_\downarrow\))
  exercise the same code path but are not gated here.
- **No GPW / Gaussian Berry strings**, no hybrid / meta-GGA XC, no CUDA PW Berry.
- **Energy / phase are not Frame-meta I/O** — the result is an owned struct;
  public energy on Frame remains the Evaluate / method path in eV.
