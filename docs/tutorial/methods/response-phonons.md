# Finite-displacement crystal phonons

> The regression runner has been removed. Regression commands below are historical; case sources and oracle records are retained as archived evidence, not current executable checks.

Finite-displacement (FD) crystal phonons assemble Cartesian interatomic force
constants (IFCs) \(\Phi\) on a periodic supercell by central differences of
forces, mass-weight them into a dynamical matrix, and diagonalize it to
phonon wavenumbers in cm⁻¹. Two public products share the FD loop:
`cpu::vib::Phonon` (a preset over `Driver<vib::Phase>`, ctor `Phonon(na, nb, nc)`
or public members `na`/`nb`/`nc`) returns the full \(3N\) spectrum at the
**supercell Γ point** on a `cpu::vib::PhononResult`; `cpu::vib::PhononDispersion`
(public `qfrac` member, `compute(hessian, unit, na, nb, nc)`) Fourier-transforms
those supercell IFCs onto the **unit-cell** dynamical matrix at arbitrary
fractional \(\mathbf{q}\) and returns a `cpu::vib::PhononDispersionResult`
(phonopy-style IFC Fourier, not a second force loop and not finite-\(q\) DFPT).
`cpu::vib::expand_supercell` builds the supercell Frame.

Force engines are **user-bound** on `vib::Phase::force` (classical
`Potential`, PW, GPW, …) — the same composition rule as AIMD and molecular
vibrations. The sections below pin the classical Stillinger–Weber (SW) force
engine on diamond Si on the CPU backend: the Γ spectrum of a 64-atom supercell
(`fd-phonon-supercell`) and the Γ → X → L dispersion from the same IFCs
(`phonon-dispersion-si`). The case also carries a `gpw-fd-phonon-si` section
(Si₂ FD spectrum with GPW analytic forces); its oracle record is in the case
`REFERENCE.md`. Analytic Γ and finite-\(q\) DFPT live on
[response-dfpt.md](response-dfpt.md#dfpt-phonon-si).

Case: `ctest -R '^response/phonons$'` — sources and oracle record in [`regressions/response/phonons/`](../../../regressions/response/phonons/). One section of the case per `##` heading below: `atv_regressions --case response/phonons --section <name>`.
The SW sections need `ATV_BUILD_REGRESSIONS=ON` with the VIB and MD modules
(standard CPU configure in module environment `buildenv-gcc/2026.03-mpich`).

## Common setup

### Force constants and dynamical matrix

Central differences with \(h = 10^{-3}\) Å (`kHessianStepAngstrom`, the same
step as the molecular Hessian):

\[
\Phi_{i\alpha,j\beta}
  = -\frac{F_{j\beta}(+\mathbf{e}_{i\alpha}h) - F_{j\beta}(-\mathbf{e}_{i\alpha}h)}{2h}
  \quad[\mathrm{eV}/\mathrm{\AA}^2],
\]

then \(\Phi \leftarrow (\Phi+\Phi^\top)/2\). \(\Phi\) is converted
eV/Å² → Hartree/Bohr² with the NIST factors of `MassWeighting` /
`PhononDiagonalizer`; eigenvalues become wavenumbers [cm⁻¹] via
`nist::au2wavenumber()` (CODATA-2014 path shared with molecular vib). All
\(3N\) modes are reported ascending; imaginary modes (\(\lambda < 0\)) appear as
**negative** \(\omega\).

Frequencies and force constants live on the result objects
(`PhononResult`, `PhononDispersionResult`), never on Frame (Frame purity).
Forces on Frame remain eV/Å during the FD loop.

### Relation to other phonon products

| Product | What it computes | Surface |
| ------- | ---------------- | ------- |
| `cpu::vib::Hessian` | free-molecule FD Hessian + TR project + thermochem | molecular |
| `cpu::vib::Phonon` | supercell FD Φ → \(\omega\) at **supercell Γ** | crystal Γ (folded) |
| `cpu::vib::PhononDispersion` | Fourier of supercell Φ → \(\omega(\mathbf{q})\) on unit cell | crystal **q-path** |
| `cpu::pw::DfptPhonon` | analytic DFPT dynamical matrix at **q = 0** (PW LSDA) | Γ DFPT |

### Stillinger–Weber diamond Si twin

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si, conventional cubic 8-atom cells |
| Lattice constant \(a\) | **5.431 Å** (SW fit target) |
| IFC supercell | **2×2×2** conventional → **64 atoms**, box \(L = 2a = 10.862\) Å |
| Why 2×2×2 | SW cutoff \(a\sigma = 3.7712\) Å requires \(L > 2\times\) cutoff (same gate as the `md/manybody-potentials` section `sw_si`) |
| Force engine | Stillinger–Weber Si (`cpu::StillingerWeber{cpu::SWPotential(Si, "stillinger-weber")}`) |
| SW parameters | Stillinger & Weber, *Phys. Rev. B* **31**, 5262 (1985) |
| FD step \(h\) | \(10^{-3}\) Å (`kHessianStepAngstrom`) |
| Neighbor | always-rebuild (`check=false`), cutoff \(a\sigma\) |
| Masses | IUPAC-2013 via `cpu::vib::mass_amu(Z)` (regression mass 28.0855 amu) |
| `Phonon` factors | \(n_a=n_b=n_c=1\) on the already-built supercell (Γ of that cell) |

Fractional diamond sites are tiled over \(2^3\) conventional 8-atom cells;
`expand_supercell` is unit-tested in
`tests/cpu/vibration/test_phonon_dispersion.cpp`
(`PhononDispersion.SupercellReplicatesColumnsOfNonSymmetricCell`: lattice
columns and atom enumeration order); the acoustic sum and the missing-force
reject have no unit test in the tree.

Shared FD sequence (both sections):

```cpp
cpu::SWPotential pot(molrs::Element::Si, "stillinger-weber");
const double cutoff = pot.cutoff();  // a*σ ≈ 3.7712 Å; need L > 2×cutoff

cpu::vib::Phonon phonon(/*na=*/1, /*nb=*/1, /*nc=*/1);  // Γ of the 64-atom SC
phonon.set_neighbor(cpu::md::NeighborList<>{cutoff, 0.3, 1, 0, false});
phonon.on(ATV::vib::Phase::force).add(cpu::StillingerWeber{pot});
// PW/GPW: .add(std::move(pw));  // cpu::pw::PW mounted directly
phonon.build();
cpu::vib::PhononResult pr = phonon.run(sc);
// pr.frequencies [cm^-1], length 3*N; pr.hessian [eV/Å²]; pr.n_atoms
```

**Why SW, not PW/GPW.** A full ab initio FD Hessian is \(6N_\mathrm{sc}\)
SCF+force evaluations. The machinery under test is supercell expansion, the
crystal \(3N\) diagonalization, the IFC Fourier and the public
`Phonon` / `PhononDispersion` API. SW supplies periodic many-body forces with a
literature potential and finishes well inside the ≤ 60 s e2e budget. Mounting
`cpu::pw::PW` (or GPW) on `vib::Phase::force` is the same public composition
path for a CP2K energy/frequency twin.

### Oracle

Preferred oracle is offline **CP2K `VIBRATIONAL_ANALYSIS`** (plus phonon band
structure) on diamond Si at the same \(a\), comparable force model or
Quickstep; never live CP2K in ctest. The pin host has no CP2K module, so the
surrogate oracle is the **Γ acoustic sum rule** (an `analytic` identity: at a
stress-free crystal minimum the three acoustic branches vanish at
\(\mathbf{q}=0\); residual \(|\omega|\) is FD / force noise) plus the published
**SW optical scale**. The optical band rejects unit disasters (Hz vs cm⁻¹,
free-atom zero spectrum, NaN) without freezing Atomiverse's own digits as a
golden snapshot. Experimental Si TO/LO(Γ) ≈ 520 cm⁻¹ is **not** the pin — SW is
classical and systematically stiffer.

When `cp2k.psmp` is available offline: run a fixed `VIBRATIONAL_ANALYSIS` /
phonon band input on diamond Si, record \(\omega\) [cm⁻¹] and wall time in
`REFERENCE.md` with the CP2K version, and time a comparable twin for
\(r = T_\mathrm{ATV}/T_\mathrm{CP2K}\), requiring \(r \le 10\). Skeleton inputs
may live under `tools/cp2k_oracle/inputs/`.

**Performance.** Both sections are `perf_exempt` = **true**: no CP2K
module/binary on the build host for a fair wall twin;
`ratio_atv_over_cp2k` and `within_10x` are n/a (correctness gates only). Wall
budget: e2e ≤ 60 s single-rank CPU (PASS; twin ≪ 1 s).

### Common exclusions and limitations

- The pins are **classical SW**, not PW/GPW FD frequencies vs CP2K Quickstep
  (the composition path for ab initio forces is public; that pin is future
  work).
- Acoustic sum-rule **projection** is not a product step (full \(3N\)
  branches; near-zero acoustics at Γ are an acceptance check, not imposed).
- Molecular thermochemistry / IR / free-molecule TR projection stay on
  `cpu::vib::Hessian`.
- No public `*Config` bags or `make_*` / `build_*` factories; no CUDA phonon
  path; no MPI multi-rank FD scaling gate.
- SW optical \(\omega\) is systematically stiffer than experimental Si
  (~520 cm⁻¹); compare SW literature, not Raman experiment.

### References

- F. H. Stillinger and T. A. Weber, *Phys. Rev. B* **31**, 5262 (1985).
- Standard FD phonon machinery (phonopy / Togo–Tanaka finite displacements and
  supercell IFC Fourier construction; CP2K `VIBRATIONAL_ANALYSIS` as intended
  Quickstep twin).
- Roadmap: [dft.md](../../roadmap/dft.md) §11 phonons / DFPT.
- Background: [Chapter 8 — Plane-Wave DFT](../ch08-plane-wave.md),
  [Chapter 9 — GPW](../ch09-gpw.md),
  [Chapter 14 — Vibrations](../ch14-vibrations.md).

## phonon-dispersion-si — Phonon dispersion ω(q), diamond Si (SW IFC Fourier) {#phonon-dispersion-si}

Phonon band structure \(\omega(\mathbf{q})\) from the 2×2×2 supercell IFCs of
[fd-phonon-supercell](#fd-phonon-supercell), Fourier-transformed onto the
conventional cubic 8-atom unit cell along **Γ → X → L**. Orthogonal to
[dfpt-phonon-si](response-dfpt.md#dfpt-phonon-si) (Γ DFPT, not this Fourier
path).

### Method

Given \(\Phi\) of an \(n_a\times n_b\times n_c\) supercell (atom order =
`cpu::vib::expand_supercell`), the mass-weighted dynamical matrix of the
**unit cell** at fractional \(\mathbf{q}\) is

\[
D_{\alpha\beta}(\kappa\kappa';\mathbf{q})
  = \frac{1}{\sqrt{m_\kappa m_{\kappa'}}}
    \sum_{\ell'}
    \Phi_{\alpha\beta}(0\kappa,\,\ell'\kappa')
    \,e^{i\mathbf{q}\cdot\mathbf{R}_{\ell'}}
  \qquad[\mathrm{a.u.}].
\]

Branches at each \(\mathbf{q}\) are sorted ascending. \(\mathbf{q}\) is given
in fractional crystal coordinates of the unit cell. The regression path:

| Label | Fractional \(\mathbf{q}\) |
| ----- | ------------------------: |
| Γ | \((0,0,0)\) |
| mid Γ–X | \((1/4,0,0)\) |
| X | \((1/2,0,0)\) |
| mid X–L | \((1/2,1/4,1/4)\) |
| L | \((1/2,1/2,1/2)\) |

`PhononDispersion::set_diamond_path(n_seg)` gives the FCC/diamond
Γ–X–W–L–Γ–K path (same waypoints as `cpu::pw::Bands::set_diamond_path`); an
empty `qfrac` at `compute` time fills `set_diamond_path(4)`.

**Why the conventional cubic cell (not the 2-atom primitive).** The classical
neighbor cell list uses `Box::length(d) = cell(d,d)`, which is **0** for the
diamond primitive triclinic cell and can hang orthorhombic min-image helpers.
The Fourier API itself is cell-agnostic; the conventional 8-atom cell is the
SW geometry of `fd-phonon-supercell`. At \(\mathbf{q}=0\) the Fourier spectrum
of the conventional cell recovers the same acoustic / optical physics as the
supercell-Γ FD spectrum of the 64-atom cell (zone-folded branches sit among the
\(3N_\mathrm{unit}=24\) modes).

**Observables.**

| Symbol | Definition | Unit |
| ------ | ---------- | ---- |
| \(\omega_b(\mathbf{q})\) | `PhononDispersionResult::frequencies_cm[iq][ibranch]` | cm⁻¹ |
| \(\mathbf{q}_\mathrm{frac}\) | `PhononDispersionResult::qfrac` | crystal |
| \(N\) | `n_atoms_unit` (unit cell); branches \(= 3N\) | — |
| \(n_a,n_b,n_c\) | supercell factors that built Φ | — |
| \(\Phi\) | upstream `PhononResult::hessian` | eV/Å² |

### Code

Header `phonon_dispersion.h`, implementation
`src/cpu/vibration/phonon_dispersion.cpp` (Fourier \(D(\mathbf{q})\) +
eigensolve). After the shared FD sequence on
`sc = cpu::vib::expand_supercell(unit, 2, 2, 2)`:

```cpp
cpu::vib::PhononDispersion path;
path.qfrac = {
    {0.0, 0.0, 0.0},   // Γ
    {0.25, 0.0, 0.0},
    {0.5, 0.0, 0.0},   // X
    {0.5, 0.25, 0.25},
    {0.5, 0.5, 0.5},   // L
};
// or: path.set_diamond_path(/*n_seg=*/4);

const cpu::vib::PhononDispersionResult r =
    path.compute(pr.hessian, unit, /*na=*/2, /*nb=*/2, /*nc=*/2);
// r.frequencies_cm[iq][ibranch] [cm^-1]; r.n_branches == 24
```

### Accuracy

Branches \(3N = 24\) (conventional unit cell, \(N=8\)); q-path Γ → mid → X →
mid → L (5 points). Measured pins (Release, `OMP_NUM_THREADS=4`):

| Quantity | Atomiverse / pin | External / rule | Tol / band | Notes |
| -------- | ---------------: | --------------- | ---------- | ----- |
| Branches / path length | **24 / 5** | \(3N\), \(N=8\); 5 q | exact | hard-coded |
| Γ \(\lvert\omega\rvert_\mathrm{ac,max}\) (3 lowest) | **0.00** cm⁻¹ | Γ acoustic sum rule | **&lt; 30** cm⁻¹ | no ASR projection |
| Γ \(\omega_\mathrm{max}\) (optical) | **594.82** cm⁻¹ | SW optical scale | **[500, 900]** cm⁻¹ | same window as FD Γ twin |
| X \(\omega_\mathrm{max}\) | **562.56** cm⁻¹ | SW zone-boundary band | **[300, 900]** cm⁻¹ | finite optical/acoustic |
| L \(\omega_\mathrm{max}\) | **559.27** cm⁻¹ | SW zone-boundary band | **[300, 900]** cm⁻¹ | finite optical/acoustic |
| Sorted / finite | yes | every path q | exact | NaN / disorder fail |
| Frame pollution | none | result-only | — | \(\omega\) on `PhononDispersionResult` |

### Performance

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| SW 64-atom FD IFC + Fourier Γ/X/L | n/a | ≈ **0.07** (informative) | n/a | n/a (**exempt**) | No CP2K binary on pin host |

Cost class: the FD cost of `fd-phonon-supercell` (64-atom SW, \(6N\)
displacements) plus a cheap Fourier path.

### Limitations

- **IFC Fourier**, not finite-\(\mathbf{q}\) DFPT Sternheimer; analytic
  \(q\neq 0\) response is on [response-dfpt.md](response-dfpt.md#dfpt-finite-q-si).
- Conventional cubic unit cell for the SW neighbor path; triclinic primitive
  diamond hangs classical min-image (infra debt).
- No LO–TO splitting / Born effective charges (nonpolar Si does not need them;
  polar materials are future work).

### References

- S. Baroni, S. de Gironcoli, A. Dal Corso, P. Giannozzi, *Rev. Mod. Phys.*
  **73**, 515 (2001) (DFPT context).

## fd-phonon-supercell — Finite-displacement supercell phonons, diamond Si (Γ, SW) {#fd-phonon-supercell}

Γ-point phonons of the 64-atom diamond Si supercell: the full \(3N\) spectrum
of `cpu::vib::Phonon` gated by the acoustic sum rule and the SW optical band.

### Method

Γ-point phonons of an \(n_a\times n_b\times n_c\) supercell of a periodic unit
cell. Lattice vectors are the rows of `frame.box.cell()` [Å]:

\[
\mathbf{a}' = n_a\mathbf{a},\quad
\mathbf{b}' = n_b\mathbf{b},\quad
\mathbf{c}' = n_c\mathbf{c}.
\]

Image \((i,j,k)\) places every unit-cell atom at
\(\mathbf{R} + i\mathbf{a} + j\mathbf{b} + k\mathbf{c}\)
(\(i\in[0,n_a)\), etc.). Types and masses replicate; tags renumber uniquely.
`expand_supercell` does **not** copy topology (bonds/angles) — force engines
that need connectivity rebuild it. The dynamical matrix at Γ is mass-weighted
through the shared `MassWeighting` path (Hartree/Bohr²/amu):

\[
D_{i\alpha,j\beta}
  = \frac{\Phi_{i\alpha,j\beta}}{\sqrt{m_i m_j}}.
\]

**Pipeline (structural components, fixed budget).** Same phase enum as
`Hessian` (`setup` → `displace` → `force` → `accumulate` → … → `diagonalize`);
`expand_supercell` (when \(n_a n_b n_c > 1\)) → `DisplaceStep` /
`HessianAccumulator` / `Symmetrize` / `MassWeighting` / `PhononDiagonalizer`.
Termination is \(6N_\mathrm{sc}\) displacements + one finalize pass. The
crystal path uses `PhononDiagonalizer` (full \(3N\)) instead of the molecular
TR projector + thermochemistry.

| | Molecular `cpu::vib::Hessian` | Crystal `cpu::vib::Phonon` |
| - | --- | --- |
| System | free molecule | crystal (PBC forces) |
| Cell | ignored for modes | supercell expansion |
| Diagonalizer | TR project \(3N-6/5\) | full \(3N\) spectrum |
| Output | vib + thermochem | \(\omega\) [cm⁻¹] (+ Hessian / modes on result) |

**Observables.**

| Symbol | Definition | Unit |
| ------ | ---------- | ---- |
| \(\omega_k\) | `PhononResult::frequencies` (length \(3N\), ascending) | cm⁻¹ |
| \(\Phi\) | `PhononResult::hessian` | eV/Å² |
| \(N\) | `n_atoms` (supercell); `n_unit` before expansion | — |
| \(n_a,n_b,n_c\) | supercell factors on the result | — |

### Code

Headers `phonon.h`, `phonon_result.h`, `supercell.h`, `phonon_diagonalizer.h`;
implementation `src/cpu/vibration/phonon.cpp` (`run()` / expand + FD loop).
The Frame passed to the shared FD sequence already is the 2×2×2 conventional
supercell (`diamond_conventional(5.431, 2)`, \(N = 64\)), so `Phonon(1,1,1)`
takes Γ of that cell. To expand a true unit cell instead, set
`phonon.na = phonon.nb = phonon.nc = 2` or call
`cpu::vib::expand_supercell(unit, 2, 2, 2)`.

### Accuracy

Measured twin (Release, `OMP_NUM_THREADS=8`):

| Quantity | Atomiverse / pin | External / rule | Tol / band | Notes |
| -------- | ---------------: | --------------- | ---------- | ----- |
| Mode count | **192** | \(3N\), \(N=64\) | exact | hard-coded in regression |
| Acoustic \(\lvert\omega\rvert_\max\) (3 lowest) | **0.000 cm⁻¹** (measured) | Γ acoustic sum rule | **&lt; 30 cm⁻¹** | residual = FD / force noise |
| Highest optical \(\omega\) | **594.822 cm⁻¹** (measured) | SW optical scale | **[500, 900] cm⁻¹** | exp. Si TO(Γ) ≈ 520 cm⁻¹; SW overshoots |
| Frame pollution | none | result-only | — | \(\omega\), \(\Phi\) on `PhononResult` |

### Performance

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| SW 64-atom Γ FD | n/a | ≈ **0.06** (informative) | n/a | n/a (**exempt**) | No CP2K `VIBRATIONAL_ANALYSIS` binary on pin host |

### Limitations

- Γ of the supercell only — the finite-\(q\) path is
  [phonon-dispersion-si](#phonon-dispersion-si).
- No DFPT ([dfpt-phonon-si](response-dfpt.md#dfpt-phonon-si)).
