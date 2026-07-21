# Chapter 2: Atomic Orbitals to Single-Point RHF

This chapter builds the first real quantum-chemical result in the manual — the
total energy of a molecule — and it does so in two deliberate steps. First it
isolates the one idea that makes electronic structure hard: the matrix you want
to diagonalize depends on the eigenvectors that come out of that same
diagonalization, so one solve is never enough. Then it replaces the abstract
version of that loop with the smallest genuine molecular problem in the
repository, closed-shell Restricted Hartree-Fock (RHF) on H₂.

| | |
|---|---|
| **Examples** | [`01_scf.cpp`](https://github.com/MolCrafts/Atomiverse/blob/dev/examples/01_scf.cpp), [`02_rhf.cpp`](https://github.com/MolCrafts/Atomiverse/blob/dev/examples/02_rhf.cpp) |
| **Results JSON** | [`examples/results/02_rhf.json`](https://github.com/MolCrafts/Atomiverse/blob/dev/examples/results/02_rhf.json) |
| **Reference** | Szabo & Ostlund, *Modern Quantum Chemistry*, Table 3.7 |

## The molecule

H₂ at \(R = 0.74\) Å in the minimal STO-3G basis — two electrons, two basis
functions, one occupied MO.

```molvis {format="xyz" representation="ball-and-stick" controls="view" height="280px"}
2
name=H2 Connct="[0,1]"
H  0.0000  0.0000  0.0000
H  0.7400  0.0000  0.0000
```

## Why SCF Has to Iterate

Suppose there is a fixed symmetric matrix $H_0$, a current density-like matrix
$P$, and a scalar coupling strength $\alpha$. A stripped-down model of the
self-consistent field (SCF) problem says the matrix we solve at iteration $k$ is

$$ H^{(k)} = H_0 + \alpha P^{(k)}. $$

Once $H^{(k)}$ is available, we diagonalize it,

$$ H^{(k)} C^{(k)} = C^{(k)} \varepsilon^{(k)}, $$

take the occupied columns of $C^{(k)}$, and rebuild the next density as

$$ P^{(k+1)} = C_{\mathrm{occ}}^{(k)} \left(C_{\mathrm{occ}}^{(k)}\right)^T. $$

Nothing in this loop is chemically realistic yet, and that is intentional — it
makes the fixed-point structure obvious. The current density changes the next
matrix, the next matrix changes the occupied subspace, and the occupied subspace
changes the next density. The calculation stops only when the density hardly
moves anymore:

$$ \left\|P^{(k+1)} - P^{(k)}\right\|_F < \mathrm{tol}. $$

If you keep only one picture from the first half of this chapter, keep this one:
**SCF is an attempt to make the matrix used for the solve consistent with the
orbitals that the solve itself produces.**

## Reading the Bare Loop

[`examples/01_scf.cpp`](https://github.com/MolCrafts/Atomiverse/blob/dev/examples/01_scf.cpp) is deliberately stripped down
— a realistic molecular example would hide the SCF pattern under too many new
ideas at once. It builds a fixed matrix $H_0$, sets the first density guess to
zero, and then runs exactly the equations above: form $H^{(k)}$, diagonalize,
keep the occupied columns, rebuild the density. The most important object is not
$H_0$ but the projector $P = C_{\mathrm{occ}} C_{\mathrm{occ}}^T$ — the
mathematical record of which states are occupied, and the thing fed back into
the next iteration.

The example prints an energy-like scalar $E = \mathrm{Tr}(H P)$ and the density
residual at every step; here those are only diagnostics that let you watch the
iteration settle. Build and run it:

```bash
cmake -B build -DATV_BUILD_EXAMPLES=ON
cmake --build build --target 01_scf
./build/examples/01_scf
```

The residual should shrink until the loop stops changing. That is the whole
lesson of the first half: SCF is not a single diagonalization, it is a search for
a density that reproduces itself.

## What Changes When the Problem Becomes Molecular

Now keep the loop and replace the introductory matrix with real physics: H₂ in
the minimal STO-3G basis — tiny enough to inspect by eye, rich enough to
introduce every central object in closed-shell Hartree-Fock.

We are no longer solving for eigenvectors of an arbitrary symmetric matrix. We
solve for molecular orbitals expanded in a finite atomic-orbital basis
$\{\chi_\mu\}$. Because those basis functions overlap, the eigenproblem is not
the ordinary $FC = C\varepsilon$ form. It is the **generalized** eigenproblem

$$ F C = S C \varepsilon, $$

where $S$ is the overlap matrix — a measure of how far the basis is from being
orthonormal. The matrix $H_{\mathrm{core}}$ collects the one-electron physics
(kinetic energy plus electron-nucleus attraction). The four-index tensor
$(\mu\nu|\lambda\sigma)$ stores the electron-electron repulsion integrals. These
fixed quantities come entirely from the geometry and the basis set; once they
are available, the SCF loop can begin.

Closed-shell RHF assumes electrons come in pairs sharing the same spatial
orbital. For H₂ that is exactly right: two electrons, one occupied spatial
orbital, and a density matrix

$$ P = 2\, C_{\mathrm{occ}} C_{\mathrm{occ}}^T. $$

The factor of $2$ is the signature of the closed-shell model — each occupied
spatial orbital carries one $\alpha$ and one $\beta$ electron. The effective
matrix is now the Fock matrix,

$$ F_{\mu\nu} = H_{\mathrm{core},\mu\nu} +
\sum_{\lambda\sigma} P_{\lambda\sigma}
\left[(\mu\nu|\lambda\sigma) - \tfrac{1}{2}(\mu\lambda|\nu\sigma)\right]. $$

This is the RHF version of the introductory update $H = H_0 + \alpha P$. The
form is more elaborate, but the logic is identical: the current density
determines the next Fock matrix, which determines the next occupied orbitals,
which determine the next density.

## Why HF Becomes a Matrix Equation — and When the Basis Goes Bad

**Concept.** The generalized eigenproblem $FC = SC\varepsilon$ exists because
atomic-orbital basis functions overlap. Usually that overlap is benign. But
nothing stops a basis from becoming *nearly redundant* — diffuse functions on
neighboring atoms describing the same region of space, or two centers drifting
close together during a scan or optimization. Then some linear combination of
basis functions is almost the zero function, the overlap matrix $S$ acquires
an eigenvalue near zero, and every solver that implicitly divides by those
eigenvalues starts amplifying noise instead of describing physics.

**Math.** Diagonalize the overlap, $S = U s U^\mathrm{T}$. Löwdin's symmetric
orthogonalization $X = U s^{-1/2} U^\mathrm{T}$ turns Roothaan–Hall into an
ordinary eigenproblem $F' C' = C' \varepsilon$ with $F' = X^\mathrm{T} F X$
and $C = X C'$ — but $s^{-1/2}$ blows up exactly when $s_i \to 0$.
*Canonical* orthogonalization instead **drops** the offending directions:
keep only the eigenpairs with $s_i \ge \tau$ and form the rectangular

$$ X = \tilde U \tilde s^{-1/2} \in \mathbb{R}^{n \times m}, \qquad
   X^\mathrm{T} S X = I_m, \quad m \le n. $$

The molecular-orbital count shrinks to $n_\mathrm{mo} = m$: the coefficient
matrix $C = XC'$ becomes $n \times m$, while the density
$P = 2 C_\mathrm{occ} C_\mathrm{occ}^\mathrm{T}$ stays $n \times n$. Nothing
physical is lost — the dropped directions carried no independent basis
content in the first place.

**Algorithm.** Atomiverse exposes the threshold as
`SCFConfig::lindep_threshold` (default $\tau = 10^{-8}$ on the raw $S$
eigenvalues; PySCF ≥ 2.13 applies $10^{-6}$ by default). Each `run()` builds
the orthogonalizer once: if nothing falls below $\tau$ the solver keeps its
LAPACK `dsygvd` fast path — pre-existing results are bit-identical by
construction — and only when columns are actually dropped does
diagonalization switch to the two-step $X^\mathrm{T} F X$ path. Setting
`lindep_threshold = 0` disables filtering entirely.

A worked pathological case: H₂ at $0.7414$ Å plus a bare proton placed
$10^{-5}$ Å from the first hydrogen (charge $+1$, 6-31G, $n_\mathrm{ao} = 6$).
The two near-coincident centers make two overlap eigenvalues collapse to
$\approx 3.5 \times 10^{-12}$ and $\approx 1.1 \times 10^{-10}$ — the next one
up is a healthy $0.126$. With the default threshold the solver drops exactly
those 2 directions ($n_\mathrm{mo} = 4$ of $6$), converges cleanly, and its
electronic energy matches a PySCF 2.13.1 reference for the identical system
to better than $10^{-6}$ Hartree. The clean H₂/6-31G energy
($-1.1267339671$ Hartree) is untouched to within $10^{-8}$ Hartree.

**Limitations.** Dropping directions makes the *energy* well-defined but the
orbital count basis-dependent: downstream consumers must read
$n_\mathrm{mo}$ from the result (`eigenvalues_alpha.size()`), not assume
$n_\mathrm{ao}$. The threshold acts on raw eigenvalues — a deliberately
simpler convention than PySCF's normalized-overlap variant — and analytic
nuclear gradients through the filtered subspace are not yet validated.

<!-- Every number quoted in this section (3.5e-12, 1.1e-10, 0.126, 2 dropped
     columns, n_mo = 4 of 6, the 1e-6 Ha PySCF agreement, and the pinned
     clean-H2 energy -1.1267339671 Ha) is regression-locked by
     tests/cpu/scf/test_scf_lindep.cpp (test_cpu_scf_lindep) and
     tests/cpu/linalg/test_canonical_orthogonalization.cpp
     (test_cpu_linalg_canonical_orth). Change one side only in lockstep. -->

## Reading the RHF Example

The setup in [`examples/02_rhf.cpp`](https://github.com/MolCrafts/Atomiverse/blob/dev/examples/02_rhf.cpp) is intentionally
short:

```cpp
cpu::Frame frame(2);
frame.span(kAtom_x)[0] = 0;    frame.span(kAtom_y)[0] = 0; frame.span(kAtom_z)[0] = 0;
frame.span(kAtom_x)[1] = 0.74; frame.span(kAtom_y)[1] = 0; frame.span(kAtom_z)[1] = 0;
frame.span(kAtom_type)[0] = 1;
frame.span(kAtom_type)[1] = 1;
frame.simbox = cpu::Box::cubic(100.0);
```

Those lines carry more chemistry than they appear to: two nuclei fixed at the
equilibrium bond length, each of charge $+1$, so two electrons to place into
molecular orbitals. The rest of the file follows the RHF logic directly:

```cpp
cpu::scf::RHF rhf;
rhf.basis = cpu::scf::BasisSet::STO3G;
rhf.tol = tol;
rhf.max_iter = 100;
const auto& result = rhf.run(frame);   // energy (Hartree)
```

> **Element coverage:** the STO-3G integral path supports elements **H through O
> (Z = 1–8)**. Atoms with higher atomic number throw when resolving the basis rather
> than returning a wrong answer (there are "no fallback values" by design).
> Heavier-element basis sets are a planned extension.

The first line prepares the fixed molecular quantities (overlap, one-electron
terms, electron-electron repulsion, nuclear repulsion). The second starts the
SCF cycle — exactly the loop from the first half of this chapter, except the
effective matrix is now the Fock matrix and the energy is a real observable:

$$ E_{\mathrm{RHF}} =
\tfrac{1}{2}\mathrm{Tr}\!\left[P(H_{\mathrm{core}} + F)\right] + V_{\mathrm{nuc}}. $$

If you want the same lesson expressed on CUDA, the companion file is
[`examples/02_rhf.cu`](https://github.com/MolCrafts/Atomiverse/blob/dev/examples/02_rhf.cu), but the CPU example is the
place to learn the method itself.

## The result that convinces you

Build and run:

```bash
cmake -B build -DATV_BUILD_CUDA=OFF -DATV_BUILD_HF=ON -DATV_BUILD_EXAMPLES=ON
cmake --build build --target 02_rhf
./build/examples/02_rhf
```

Numbers below are the live output of that binary, frozen in
[`examples/results/02_rhf.json`](https://github.com/MolCrafts/Atomiverse/blob/dev/examples/results/02_rhf.json)
— not a hand-tuned golden file.

| Quantity | Atomiverse | Reference |
|---|---:|---:|
| \(E_\mathrm{total}\) (Ha) | **−1.1167593074** | −1.1168 (Szabo & Ostlund, Table 3.7) |
| \(E_\mathrm{electronic}\) (Ha) | −1.8318636465 | — |
| \(V_\mathrm{nuc}\) (Ha) | 0.7151043391 | — |
| SCF iterations (\(\mathrm{tol}=10^{-10}\)) | 3 | — |
| \(\varepsilon_0\) (occupied) (Ha) | −0.578554 | — |
| \(\mathrm{Tr}(PS)\) | 2.000000 | 2 |

The absolute error is \(\approx 4\times 10^{-5}\) Ha — well inside the
textbook table's four-decimal rounding. SCF benchmarks pin Atomiverse RHF
energies against PySCF to \(10^{-7}\) Ha (see
[benchmark validation](https://github.com/MolCrafts/Atomiverse/blob/dev/benchmarks/README.md)).
After convergence, \(\mathrm{Tr}(PS)=N_e\) must still hold: the density
represents two electrons in a closed-shell state.

## Electron density (Gaussian cube)

[`examples/16_density_cube.cpp`](https://github.com/MolCrafts/Atomiverse/blob/dev/examples/16_density_cube.cpp)
runs the same H₂ / STO-3G RHF and writes ρ(r) plus the HOMO with
`DensityCubeWriter` / `OrbitalCubeWriter`. Live files:

- [`examples/results/h2_density.cube`](https://github.com/MolCrafts/Atomiverse/blob/dev/examples/results/h2_density.cube)
- [`examples/results/h2_homo.cube`](https://github.com/MolCrafts/Atomiverse/blob/dev/examples/results/h2_homo.cube)

On a 0.3 Bohr mesh, the grid-integrated density is **1.99945** electrons
(target 2.0) — see [`16_density_cube.json`](https://github.com/MolCrafts/Atomiverse/blob/dev/examples/results/16_density_cube.json).

```bash
cmake --build build --target 16_density_cube
./build/examples/16_density_cube examples/results/h2_density.cube examples/results/h2_homo.cube
```

<figure class="molcrafts-web-component-card">
  <figcaption>
    <span class="molcrafts-web-component-card__package">MolVis</span>
    <strong>H₂ electron density (RHF/STO-3G)</strong>
    <small>Gaussian cube from Atomiverse <code>16_density_cube</code>. Drag to rotate; density isosurface from <code>h2_density.cube</code>.</small>
  </figcaption>
  <div class="molcrafts-web-component-preview">
    <molvis-viewer
      format="cube"
      src="../../assets/structures/h2_density.cube"
      representation="ball-and-stick"
      controls="view"
      height="380px"
    ></molvis-viewer>
  </div>
</figure>

```cpp
cpu::scf::SCF scf{config};
auto handle = scf.result_handle();
cpu::Evaluate eval;
eval.on(evaluate::Phase::evaluate).add(std::move(scf).with_gradient(false));
eval.on(evaluate::Phase::finalize)
    .add(cpu::recorder::DensityCubeWriter{handle, "h2_density.cube", 4.0, 0.30});
eval.on(evaluate::Phase::finalize)
    .add(cpu::recorder::OrbitalCubeWriter{
        handle, "h2_homo.cube", /*mo=*/0, cpu::recorder::OrbitalSpin::Alpha,
        4.0, 0.30});
eval.build();
eval.run(frame);
```

## Closed-Shell Validity

**Concept.** RHF is not a general-purpose method with occasional failures — it
is a method whose derivation *assumes* a specific electronic structure: every
occupied spatial orbital holds exactly two electrons of opposite spin. That
assumption has two immediate consequences. The electron count $N_e$ must be
even, and the spin multiplicity must be $M = 1$ (a singlet). Feed RHF anything
else and the failure is not numerical noise; the ansatz itself no longer
describes the state you asked for.

**Math.** The closed-shell density is built from $N_e/2$ doubly-occupied
orbitals:

$$ P_{\mu\nu} = 2 \sum_{i=1}^{N_e/2} C_{\mu i} C_{\nu i}. $$

For odd $N_e$, integer division silently truncates: $\lfloor 9/2 \rfloor = 4$
doubly-occupied orbitals describe $8$ electrons — one electron simply
disappears from the calculation. The resulting energy is perfectly convergent
and perfectly meaningless.

**Algorithm.** Atomiverse guards against this at the single choke point every
Gaussian-SCF path passes through, the solver entry (`SCFSolver::run`). Before
any allocation or iteration it checks that $n_\alpha \ge 0$, $n_\beta \ge 0$,
$n_\alpha + n_\beta = N_e$, and $n_\alpha = n_\beta$ for a restricted spin
treatment, and throws `std::invalid_argument` otherwise. The classic trap now
fails loudly: water cation ($\mathrm{H_2O^+}$, $N_e = 9$) routed through
`rhf_config` yields $n_\alpha = n_\beta = 4$, the sum $8 \ne 9$ trips the
guard, and no SCF iteration ever runs.

```cpp
/* integrals private inside RHF/UHF/RKS/UKS */   // H2O+, 9 electrons
cpu::scf::RHF rhf;
solver.build(ints);
solver.run(ints);             // throws std::invalid_argument — not silence
```

Because the guard is read-only and runs before any work, valid closed-shell
runs take exactly the same code path as the unguarded solver — the
neutral-water energy in this chapter is pinned to the pre-guard baseline at
$10^{-8}$ Ha by the regression test. (The bar is not zero: the same source
reassociates its BLAS reductions differently on x86_64 and aarch64, which
moves the last digits by ~$10^{-12}$ Ha.)

**Limitations.** The guard validates bookkeeping, not physics. A biradical
singlet passes every check and still breaks the RHF ansatz; that failure mode
needs the unrestricted treatment of
[Chapter 3](ch03-open-shell.md), where the companion section "Charge and
Multiplicity Conventions" gives the full $(N_e, M) \to (n_\alpha, n_\beta)$
resolution rules.

<!-- Every number and thrown case in this section is regression-locked by
     tests/cpu/scf/test_spin_resolution.cpp (test_cpu_scf_spin_resolution):
     H2O+ via rhf_config throws (OddElectronRhfThrows); the neutral-H2O
     baseline pin (1e-8 Ha) is NeutralH2ORhfEnergyMatchesBaseline. Change one
     side only in lockstep with the other. -->

## SCF Convergence Is a Numerical Problem

**Concept.** The SCF fixed-point iteration has no built-in guarantee of
convergence. When the HOMO–LUMO gap is small — stretched bonds, transition
metals, radicals — the bare iteration tends to *oscillate*: the occupied and
virtual subspaces swap character between iterations, each density undoing the
previous one. Convergence acceleration is therefore not a luxury bolted onto
Hartree–Fock; it is part of what makes the method usable. Three tools cover
most of practice: damping, DIIS, and the level shift.

**Damping** mixes the new density with the old one,
$P \leftarrow (1-\alpha) P_\mathrm{new} + \alpha P_\mathrm{old}$, trading
speed for stability. One engine-specific convention to note: Atomiverse damps
the **density after diagonalization** (`SCFConfig::damping_factor` = fraction
of *old* density retained), whereas PySCF damps the **Fock matrix before
DIIS**. Both act only while DIIS is inactive; the difference is deliberate
and documented in the solver.

**DIIS** (Pulay's Direct Inversion in the Iterative Subspace) extrapolates
the Fock matrix from its recent history, choosing the combination that
minimizes the commutator error $e = X^\mathrm{T}(FPS - SPF)X$ — zero exactly
at self-consistency. It is the default accelerator (`SCFConfig::use_diis`)
and usually converges equilibrium molecules in 10–20 iterations.

**Level shift** (Saunders & Hillier, *Int. J. Quantum Chem.* **7**, 699
(1973), DOI [10.1002/qua.560070407](https://doi.org/10.1002/qua.560070407))
attacks the oscillation mechanism directly. Shifting the Fock matrix by

$$ F' = F + \lambda\,(S - S\,D_\mathrm{occ}\,S), \qquad
   D_\mathrm{occ} = C_\mathrm{occ} C_\mathrm{occ}^\mathrm{T}, $$

raises every **virtual** orbital eigenvalue by exactly $\lambda$ (Hartree)
while leaving the occupied ones untouched — the widened gap suppresses
occupied/virtual swapping. Atomiverse applies it with PySCF `get_fock`
ordering parity: damping → DIIS extrapolation of the *unshifted* Fock →
level shift → diagonalize. Two consequences follow. The converged **energy**
is $\lambda$-invariant (measured $|E_{\lambda=0.5} - E_{\lambda=0}| <
10^{-9}$ Hartree on H₂O/6-31G; PySCF itself shows $2.8\times10^{-14}$), and
after convergence one extra $\lambda = 0$ diagonalization removes the $+
\lambda$ pollution from the reported virtual energies (final
$\max_i|\Delta\varepsilon_i| \le 10^{-6}$ Hartree; PySCF measures
$2.5\times10^{-7}$).

**A hard case you can reproduce.** Stretch both O–H bonds of water to
$1.7$ Å (6-31G RHF) and strip away every crutch: `init_guess = Hcore`,
`use_diis = false`, `damping_factor = 0`. The bare iteration oscillates
about $4.5$ Hartree above the true energy and is still not converged after
100 iterations. Setting only `level_shift = 0.5` rescues it — convergence in
**57 iterations** to the same energy a DIIS reference run reaches in 15
($-75.6750348$ Hartree, agreement $< 10^{-8}$). The scan shows the
practitioner's trade-off: $\lambda = 0.3$ needs 52 iterations, $0.5$ needs
57, $1.0$ needs 105, $2.0$ needs 145 — too small fails to stabilize, too
large over-damps and crawls.

**Practice (Jensen's rule of thumb).** Start with the defaults (MINAO guess
+ DIIS). If SCF oscillates, add a level shift of $0.3$–$1.0$ Hartree to
stabilize the early iterations, and let DIIS finish the job; for pathological
cases combine a modest shift with damping. Remember that a level shift
slows terminal convergence — it is a stabilizer, not an accelerator — and
always check the *de-shifted* orbital energies, never the shifted ones.

<!-- Every number in this section (the 1e-9 energy invariance, the 1e-6
     de-shift bound, the 100-iteration divergence, λ=0.5 → 57 iterations,
     the λ scan 52/57/105/145, and the <1e-8 DIIS-reference agreement) is
     regression-locked by tests/cpu/scf/test_level_shift.cpp
     (test_cpu_scf_level_shift) and the apply_level_shift unit tests in
     tests/cpu/scf/test_accelerators.cpp. Change one side only in lockstep
     with the other. -->

## Koopmans' Theorem: An Orbital Energy That Predicts an Experiment

**Concept.** So far the orbital energies $\varepsilon_i$ have been internal
bookkeeping — the diagonal of the Fock matrix in the MO basis. Koopmans'
theorem (T. Koopmans, *Physica* **1**, 104 (1934),
DOI [10.1016/S0031-8914(34)90011-2](https://doi.org/10.1016/S0031-8914(34)90011-2))
turns one of them into a measurable quantity. Ionizing a molecule means
removing one electron; the energy it costs is the ionization potential (IP).
Koopmans says the *vertical* IP — removal fast enough that the nuclei and the
other electrons do not move — is simply the negative of the highest-occupied
orbital energy:

$$\mathrm{IP} \;\approx\; -\varepsilon_{\mathrm{HOMO}}.$$

**The frozen-orbital math.** Why should a diagonal Fock element equal an
ionization energy? Take a closed-shell RHF determinant with $N$ electrons and
total energy

$$E(N) \;=\; \sum_i 2 h_{ii} \;+\; \sum_{ij} \left(2 J_{ij} - K_{ij}\right),$$

where $h_{ii}$ is the core (kinetic + nuclear-attraction) integral of orbital
$i$, and $J_{ij}$, $K_{ij}$ are the Coulomb and exchange integrals between
occupied orbitals $i$ and $j$. Now remove the electron in orbital $i$ **without
letting any other orbital change** — the frozen-orbital approximation. The
cation's energy $E_{\mathrm{frozen}}(N-1;\,\text{hole }i)$ loses exactly the
terms that involved that one electron, and the difference telescopes into a
single canonical orbital energy:

$$E(N) - E_{\mathrm{frozen}}(N-1;\,\text{hole }i)
  \;=\; h_{ii} + \sum_j \left(2 J_{ij} - K_{ij}\right)
  \;=\; \varepsilon_i.$$

That last equality is just the definition of the canonical Fock eigenvalue.
Ionizing from the HOMO therefore costs $-\varepsilon_{\mathrm{HOMO}}$, and the
same algebra gives $-\varepsilon_i$ for any occupied orbital.

**The approximation — and why it works better than it should.** The derivation
made two silent assumptions, and both introduce error:

1. **No orbital relaxation.** The remaining $N-1$ electrons *do* relax inward
   toward the extra positive charge, lowering the cation's true energy below
   $E_{\mathrm{frozen}}$. Koopmans therefore **overestimates** the IP.
2. **No electron correlation.** RHF misses correlation energy, and the neutral
   (more electrons, more correlation) loses more of it than the cation. This
   pushes the neutral up relative to the cation and **underestimates** the IP.

The two errors have opposite sign. Near the Hartree-Fock basis-set limit they
partially cancel, which is why $-\varepsilon_{\mathrm{HOMO}}$ lands within an eV
or so of experiment for many small closed-shell molecules — better than either
error term alone would suggest. In a minimal basis, basis-set incompleteness
dominates and the cancellation argument no longer holds.

**The algorithm.** Nothing new is required. A single RHF solve already returns
the sorted orbital energies; the HOMO is the last occupied one:

```cpp
auto ints   = /* IntegralPack is private — use RHF */;
auto config = /* SCFConfig is private — set RHF members */;
config.max_iter = 100;
cpu::scf::RHF rhf;
solver.build(ints);
const auto& result = solver.run(ints);
double eps_homo = result.eigenvalues_alpha(ints.n_occ() - 1); // (Hartree)
double ip_eV    = -eps_homo * 27.211386245988;            // 1 Ha = 27.211386245988 eV
```

For a component-based workflow, `cpu::scf::OrbitalInfoObserver` (bound at the
`finalize` phase of a `cpu::Evaluate` driver and sharing the SCF state via
`scf.result_handle()`) exposes `homo_eV()` / `lumo_eV()` directly. The worked
sweep is `examples/11_koopmans.cpp`.

**The property.** Running RHF single points on five closed-shell systems in two
basis sets gives the Koopmans IP $= -\varepsilon_{\mathrm{HOMO}}$
(eV column $= -\varepsilon_{\mathrm{HOMO}} \times 27.211386245988$):

| system | STO-3G $-\varepsilon_{\mathrm{HOMO}}$ (Ha) | STO-3G IP (eV) | 6-31G $-\varepsilon_{\mathrm{HOMO}}$ (Ha) | 6-31G IP (eV) | exp. IP (eV) |
|---|---|---|---|---|---|
| He  | 0.87603551 | 23.84 | 0.91412663 | 24.87 | 24.59 (vertical) |
| H₂  | 0.57797481 | 15.73 | 0.59539276 | 16.20 | 15.43 (adiabatic) |
| H₂O | 0.39123677 | 10.65 | 0.50136813 | 13.64 | 12.62 (vertical) |
| NH₃ | 0.35252132 |  9.59 | 0.41503216 | 11.29 | 10.8 (vertical) |
| CH₄ | 0.51931445 | 14.13 | 0.54404220 | 14.80 | 13.6 (vertical, Jahn-Teller) |

**Basis sensitivity.** Read the H₂O row across: the Koopmans IP moves from
**10.65 eV** in STO-3G to **13.64 eV** in 6-31G — nearly 3 eV from a basis-set
change alone (0.39123677 Ha → 0.50136813 Ha). The minimal STO-3G basis places
the HOMO far too high; adding the split-valence flexibility of 6-31G pulls it
down and *overshoots* experiment (12.62 eV). This is the error cancellation
failing in the minimal basis and then over-correcting once relaxation is still
neglected — a concrete reminder that a single Koopmans number without a basis
qualifier is meaningless.

**Adiabatic vs vertical.** Koopmans approximates the *vertical* IP: the nuclei
are frozen at the neutral geometry. Experiments often report the *adiabatic*
IP, measured to the relaxed cation minimum, which is lower. For He the two
coincide (an atom has no geometry to relax). For H₂ the tabulated 15.43 eV is
adiabatic; the vertical value is higher (~16 eV), closer to what Koopmans
targets. For CH₄ the cation is Jahn-Teller distorted, so the vertical and
adiabatic IPs differ substantially — comparison with the ~13.6 eV vertical
value can only ever be qualitative.

**Limitations.** Koopmans is a first estimate, not a converged property.
Orbital relaxation and correlation are neglected; the error cancellation is
fortuitous and fails in small bases and for core ionizations (deep orbitals
relax the most). The theorem is derived for *canonical* Hartree-Fock orbitals —
it does **not** transfer cleanly to Kohn-Sham DFT orbital energies (where
$-\varepsilon_{\mathrm{HOMO}}$ is contaminated by the exchange-correlation
derivative discontinuity) nor to open-shell / spin-unrestricted references.
The electron affinity from $-\varepsilon_{\mathrm{LUMO}}$ is a far worse
approximation still, because the virtual orbitals of a neutral RHF calculation
feel the full $N$-electron potential rather than the $N$-electron one the extra
electron would see. For quantitative IPs, use $\Delta$SCF (separate neutral and
cation solves) or a correlated method — but as a zero-extra-cost readout of a
number you already computed, Koopmans is hard to beat.

<!-- Every −ε_HOMO value quoted in this section is transcribed from the pinned
     PySCF 2.13.1 (cart=True, conv_tol=1e-12) table that is the single source
     of truth in tests (Koopmans unit) / literature tables; the
     eV columns are those Ha values × 27.211386245988. The anchors in
     tests/cpu/scf/test_koopmans.cpp lock He/H2/H2O @ STO-3G + He @ 6-31G in
     the default gate. Change one side only in lockstep with the others. -->

### Hands-on: Koopmans vs ΔSCF (Jensen §3.4 style)

Koopmans froze the orbitals; ΔSCF lets them relax. The ΔSCF ionization
potential is simply the difference of **two independently converged SCF
solutions** — the neutral and the cation — at the same (vertical) geometry:

$$\mathrm{IP}_{\Delta\mathrm{SCF}} \;=\; E_{\mathrm{SCF}}(N-1) - E_{\mathrm{SCF}}(N).$$

No new machinery is needed: the cation is configured entirely through the
members that already exist — `charge` on any method, and the explicit
`n_alpha` / `n_beta` electron counts on `UHF` (removing one electron leaves
an open shell, so the cation is a UHF run):

```cpp
// Neutral N2 at the experimental bond length (1.0977 Å): RHF/6-31G*.
cpu::scf::RHF rhf;
rhf.basis = cpu::scf::BasisSet::Pople631Gs;
rhf.tol = 1e-12;
rhf.run(frame);
double e_n2 = rhf.energy();                              // [Hartree]

// Koopmans, read straight off the converged determinant (zero extra cost) —
// or via cpu::scf::OrbitalInfoObserver::ip_koopmans_eV().
double ip_koopmans =
    -rhf.determinant().homo(cpu::scf::SlaterDeterminant::Spin::alpha)
    * 27.211386245988;                                   // [eV]

// Cation N2+: one α electron removed — members only, no new API.
cpu::scf::UHF uhf;
uhf.basis = cpu::scf::BasisSet::Pople631Gs;
uhf.charge = 1;
uhf.n_alpha = 7;   // 13 electrons: doublet
uhf.n_beta = 6;
uhf.tol = 1e-12;
uhf.run(frame);
double ip_delta = (uhf.energy() - e_n2) * 27.211386245988;   // [eV]
```

The N₂ numbers (RHF//UHF 6-31G*, cart, PySCF 2.14.0 cross-checked to
≤1e-3 eV — `regressions/scf/n2_ionization`):

| quantity | value (eV) | comment |
|---|---|---|
| −ε_HOMO (Koopmans) | 16.65 | frozen orbitals; HF-limit literature value ≈ 17.3 (Szabo-Ostlund §3.3) |
| ΔSCF IP | 15.69 | orbital relaxation lowers the cation |
| experiment (vertical) | 15.58 | NIST WebBook |

Two things to read out of the table. First, ΔSCF < Koopmans **always** for
the same solution — relaxation is variational, the cation energy can only go
down. Second, ΔSCF lands within ~0.1 eV of experiment here while Koopmans
overshoots by ~1 eV; part of that improvement is honest relaxation physics
and part is the correlation-error cancellation shifting between the two
recipes, so do not expect 0.1 eV routinely.

> **Warning — electron affinities.** Do **not** compute an EA from
> $-\varepsilon_{\mathrm{LUMO}}$. Koopmans' EA is a *basis-set artifact*: the
> virtual orbitals of the neutral feel the $N$-electron potential rather than
> the $(N{+}1)$-electron one, and as the basis grows the extra electron can
> spread into ever-more-diffuse functions, so $-\varepsilon_{\mathrm{LUMO}}$
> drifts toward 0 (a free electron) instead of converging to the physical
> EA. If you need an affinity, run it as ΔSCF — a separate converged anion
> (`charge = -1`, one more β or α electron) against the neutral — with a
> basis containing diffuse functions, and be suspicious even then when the
> anion is not bound (Jensen §3.4).

ΔSCF here means the *ground-state* cation/anion. Non-Aufbau occupations
(excited-state or core-hole ΔSCF) are a different tool —
`SlaterDeterminant::with_occupation` — and come with their own collapse
problems (MOM); they are out of scope for this chapter.

## What to Take Into Chapter 3

The mental shift is now complete. The iteration still has the same shape it had
in the bare loop, but every object in it has become physical: the density is an
electronic density matrix, the effective matrix is a Fock matrix, and the final
scalar is a real molecular energy. The next chapter keeps this machinery but
drops the closed-shell restriction — where RHF stops being enough and
unrestricted Hartree-Fock begins:
[Chapter 3: Unrestricted Hartree-Fock (UHF)](ch03-open-shell.md).
