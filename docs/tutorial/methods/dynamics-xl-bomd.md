# Extended-Lagrangian BOMD (XL-BOMD)

Extended-Lagrangian Born–Oppenheimer MD propagates an auxiliary AO density
matrix alongside the nuclei and uses it, instead of the last converged density,
as the electronic starting point at every MD step. The auxiliary density is a
*dynamical variable* with almost time-reversible dynamics, so it removes the
systematic energy drift that plain guess reuse causes in long BOMD trajectories.
Two forms are available: **SCF-converged** XL-BOMD (the SCF still runs to `tol`;
only the guess changes) and **zero-SCF linearized** XL-BOMD (no SCF iterations
on warm steps; energy and forces come from Niklasson's shadow Born–Oppenheimer
functional).

XL-BOMD is a wrapper **type** around a concrete method, bound into the force
slot of `cpu::md::MD` (NVE `cpu::md::VelocityVerlet`) with `set_evaluator_ref`:

- **GPW:** `cpu::electronic::XlBomdForce{cpu::gpw::GPWEvaluator{cpu::gpw::GpwDft{…}, /*compute_forces=*/true}, k}`;
  adding the tag `cpu::electronic::XlBomdLinearized{}` as third argument selects
  the zero-SCF shadow-force path. Γ-RKS, Γ-UKS, multi-k RKS and multi-k UKS.
- **Gaussian SCF:** `cpu::electronic::XlBomdMethod<cpu::scf::{RHF,UHF,RKS,UKS,…}>{method, k}`
  (SCF-converged form only).

The sections below cover Γ-RKS H₂O and Γ-UKS O₂ (GPW), multi-k RKS H₂ (GPW),
RHF/STO-3G H₂O, and zero-SCF linearized forces on Γ-RKS H₂O; three further
variants (linearized multi-k, linearized Γ-UKS, SCF-converged multi-k UKS) are
documented without a running section. The case also carries a
`xl-bomd-linearized-stress` section (zero-SCF linearized XL-BOMD with analytic
shadow-density stress on periodic H₂, CP2K-anchored), whose oracle record is in
the case `REFERENCE.md`. CPU only.

Case: `ctest -R '^dynamics/xl-bomd$'` — sources and oracle record in [`regressions/dynamics/xl-bomd/`](../../../regressions/dynamics/xl-bomd/). One section of the case per `##` heading below: `atv_regressions --case dynamics/xl-bomd --section <name>`.

## Theory

**Dissipative XL recursion** (Niklasson, *Phys. Rev. Lett.* **100**, 123004
(2008); dissipative form Niklasson et al., *J. Chem. Phys.* **130**, 214109
(2009)):

\[
P_{n+1} = 2P_n - P_{n-1} + \kappa\,(D_n - P_n)
          + \alpha \sum_{k=0}^{K} c_k P_{n-k},
\]

with \(D_n\) the electronic density at step \(n\) (the SCF-converged density,
or the one-shot diagonalization density on the linearized path) and
\((\kappa,\alpha,c_k)\) the Table-I row for the dissipation order
\(K \in [3,9]\) (history depth \(K+1\)). Default \(K=5\):

| \(K\) | \(\kappa\) | \(\alpha\) | \(c_0\ldots c_K\) |
| ----: | ---------: | ---------: | ---------------- |
| 5 | 1.82 | 0.018 | \(\{-6,\,14,\,-8,\,-3,\,4,\,-1\}\) |

The recursion is time-reversible up to the \(O(\alpha)\) dissipation. Reusing
the last converged density is *not* time-reversible — the guess always lags the
nuclei and the one-sided error pumps energy into the trajectory. Unrestricted
methods propagate α and β independently with identical coefficients; multi-k
paths propagate one complex Hermitian \(P_k\) per sampled Bloch point (real
\((\kappa,\alpha,c)\) act element-wise, so Hermitian history stays Hermitian up
to roundoff). The first MD step seeds the history with the cold SCF density
(stationary start); later steps propagate.

**ASPC ≠ XL.** Kolafa ASPC (`cpu::electronic::ASPCPredictor`, and CP2K MD's
default `EXTRAPOLATION ASPC`) *extrapolates* a polynomial through past
densities; XL *propagates* a dynamical variable. Car–Parrinello
(`cpu::electronic::CarParrinello`) and second-generation CP
(`cpu::electronic::SecondGenCp`) are different electronic propagators again.

**Shadow / linearized functional.** When \(n = P_{\mathrm{aux}}\) is not
self-consistent, the ordinary KS energy and its Hellmann–Feynman force are
inconsistent with the nuclear dynamics. The linearized (shadow) functional

\[
\mathcal{E}(R,n)
  = \mathrm{Tr}\!\bigl(H[n]\,D\bigr)
  - \mathrm{Tr}\!\bigl(H[n]\,n\bigr)
  + E_{\mathrm{KS}}[n],
  \qquad D = q[n]
\]

recovers \(E_{\mathrm{KS}}\) exactly at the fixed point \(D = n\), and
\(\mathrm{Tr}\!\bigl(H[n]\,(D-n)\bigr)\) is the first-order residual heat when
\(D \neq n\). Forces are taken of \(\mathcal{U}(R,n)\) at fixed \(n\) (density
\(D\), KS matrix \(H[n]\)) (Niklasson & Cawkwell, *Phys. Rev. B* **86**, 174308
(2012); Niklasson, *J. Chem. Phys.* **147**, 054103 (2017)). Collinear UKS sums
the trace terms over spin; multi-k weights them by \(w_k\).

**Primary observable.** Every section gates the relative NVE envelope

\[
\delta = \max_t \frac{\lvert E(t)-E_0\rvert}{\max\!\big(1,\,\lvert E_0\rvert\big)},
\]

with \(E(t)=E_{\mathrm{pot}}(t)+E_{\mathrm{kin}}(t)\) [eV] and \(E_0\) after the
warm-up / cold-seed step (XL histories seeded). The budget is \(0.05\) for
SCF-converged XL and \(0.10\) for linearized XL (looser because of one-shot
residual heat). A broken XL implementation (α-only history, swapped channels,
non-density aux) destroys SCF convergence or the envelope immediately. Velocity
Verlet conserves \(E\) up to a bounded non-secular envelope when
\(\mathbf{F}\approx -\nabla E\) (Frenkel & Smit §4.3.4; Marx & Hutter BOMD).

## Public API

```cpp
// GPW (any of Γ-RKS / Γ-UKS / multi-k): the wrapper owns the evaluator
cpu::gpw::GPWEvaluator gpw(std::move(dft), /*compute_forces=*/true);
cpu::electronic::XlBomdForce eval{std::move(gpw), /*k=*/5};            // SCF-converged
// cpu::electronic::XlBomdForce eval{std::move(gpw), /*k=*/5,
//                                   cpu::electronic::XlBomdLinearized{}};  // zero-SCF shadow

cpu::md::MD sim;
sim.set_integrator(cpu::md::VelocityVerlet{/*dt [fs]=*/0.5});
sim.set_evaluator_ref(eval);   // borrow: readbacks stay live after run()
sim.build();
sim.run(frame, 1);             // warm-up / cold seed
sim.run(frame, 4);             // production; frame.meta.get(kEnergy) [eV]
// eval.last_result().{iters, converged, n_kpts, P_k}; eval.method().last_s2()
```

The propagators are public primitives (`cpu/electronic/xl_bomd.h`):

```cpp
cpu::electronic::XlBomd xl(5);      // real; XlBomdCplx for one Bloch point
xl.advance(D_scf);                  // first call seeds; later calls propagate
const auto& guess = xl.aux();       // initial density for the next step
// both throw unless 3 <= K <= 9
```

- XL is a type, not a flag: the mode bits are private on `GPWEvaluator` /
  protected on `MeanFieldMethod`, and the wrappers are the only public way to
  set them. There is no `XlConfig` and no `make_xl_*`. `kinds`, `cutoff_ry` and
  `xc` are `GpwDft` constructor arguments; the other parameters are members
  moved into `GPWEvaluator` with the method. `n_alpha` / `n_beta` select UKS
  (dual propagators); a non-Γ `k_mesh` selects the complex per-k propagators.
- Force binding is `set_evaluator_ref(eval)` (borrow) or
  `set_evaluator(std::move(eval))` (own); `on(md::Phase::force).add_ref(...)`
  remains available when a specific order is needed.
- **Mutual exclusion** (`std::invalid_argument` at `build()` / first nested
  entry, both backends of the seam agree): XL × OT (`use_ot` parameterizes
  orbitals, not the density), XL × CPMD, XL × second-gen CP, and on the SCF
  methods XL × `predictor(...)` (ASPC). The `XlBomdLinearized` constructor sets
  both the XL auxiliary density and `compute_forces` itself, so "linearized
  without XL" and "linearized without forces" cannot be expressed. The
  linearized path refuses `stress_use_fd` (strain-FD re-converges the SCF —
  Born–Oppenheimer stress, not the shadow functional).
- Fixed cell (NVE): GPW rejects box changes after `build()`.
- Public units: eV, Å, fs, eV/Å on Frame; Hartree / Ry and \(\langle S^2\rangle\)
  only through method accessors.

Sources: `xl_bomd.h` + `src/cpu/electronic/xl_bomd.cpp` (propagators),
`xl_bomd_force.h` (`XlBomdForce`, `XlBomdLinearized`), `aimd_method.h`
(`XlBomdMethod`), `gpw_evaluator.h` + `src/cpu/gpw/gpw_evaluator.cpp`
(GPW XL advance, `guess_P_k`, warm shadow paths),
`src/cpu/gpw/internal/gpw_forces.cpp` (`internal::gpw_forces_at_density`,
`assemble_gpw_forces_kpoints`), `src/cpu/gpw/gpw_solver.cpp`
(`run_gpw_scf_kpoints_uks`).

**Unit coverage.** No unit test in the tree gates the XL path (GPW, the
Gaussian-SCF seam, or the propagator coefficients). Its stated properties:
Table-I coefficients for every \(K\in[3,9]\) bitwise (real and complex),
fixed point on a constant density, bounded response under oscillatory driving,
`GpwDft::run` `guess_P` plumbing (converged-density guess reaches the cold
fixed point in ≤ cold iterations); XL + UKS occupations allowed at `build()`,
XL + OT refused, invalid \(K\) refused; illegal linearized combos throw; shadow
fixed-point identities (Γ real, dual-spin, k-weighted complex): when \(D = n\),
\(\mathcal{E} = E_{\mathrm{KS}}\); when \(D \neq n\),
\(\mathcal{E} - E_{\mathrm{KS}} = \mathrm{Tr}(H(D-n))\).

## GPW electronic structure

GPW Kohn–Sham (Quickstep family; Lippert, Hutter, Parrinello, *Mol. Phys.*
**92**, 477 (1997); VandeVondele et al., *Comput. Phys. Commun.* **167**, 103
(2005)): contracted Gaussian AO basis, density collocated on a uniform grid,
FFT Poisson Hartree, grid XC, GTH analytic pseudopotentials. Isolated systems
use Martyna–Tuckerman electrostatics (`nonperiodic = true` / CP2K
`PERIODIC NONE` + `PSOLVER MT`; Martyna & Tuckerman, *J. Chem. Phys.* **110**,
2810 (1999)) with the molecule **centred** in the cell; periodic systems use the
periodic Poisson solver (`nonperiodic = false`). UKS: separate
\(\rho_\alpha,\rho_\beta\) and KS matrices, spin-resolved **PADE (LSD)** XC
(Goedecker, Teter, Hutter, *Phys. Rev. B* **54**, 1703 (1996)), dual-spin DIIS,
explicit integer members `n_alpha` / `n_beta` (no multiplicity bag). Multi-k:
complex Bloch matrices per \(k\), k-summed density and forces (Monkhorst & Pack,
*Phys. Rev. B* **13**, 5188 (1976)). Analytic forces write
`frame.span(kAtom_f*)` [eV/Å].

### Systems

| | H₂O (Γ-RKS) | O₂ triplet (Γ-UKS) | H₂ periodic (multi-k) |
| --- | --- | --- | --- |
| Boundary | isolated / MT, cubic \(L = 10\) Å | isolated / MT, cubic \(L = 12\) Å | periodic, cubic \(L = 12\) Bohr ≈ 6.350 Å |
| Geometry | CP2K getting-started (below) | \(R = 1.2075\) Å along \(z\), centred | \(R = 1.0\) Bohr compressed bond along \(x\), cell-centred |
| Masses [amu] | O 15.999; H 1.008 | 15.999 each | 1.008 each |
| XC | PBE | PADE (LSD) | PADE |
| Basis / PP | DZVP-MOLOPT-GTH; GTH-PBE-q6 (O), GTH-PBE-q1 (H) | DZVP-MOLOPT-GTH; GTH-PBE-q6 | DZV-GTH-PADE; GTH-PADE-q1 |
| Electrons | closed shell | \(n_\alpha=7\), \(n_\beta=5\) (12 valence e⁻, multiplicity 3) | \(n_{\mathrm{occ}}=1\) |
| \(v_0\) [Å/fs] | COM-preserving H stretch \(v_x(\mathrm{H1/H2}) = \pm 0.0015\) | antisymmetric stretch \(v_z(\mathrm{O1/O2}) = \mp 0.0008\) | antisymmetric stretch \(v_x = \mp 0.001\) |
| SCF | `tol = 1e-5`, `max_iter = 60`, `mix = 0.3` | `tol = 1e-5`, `max_iter = 80`, `mix = 0.4` | `tol = 1e-5`, `max_iter = 60`, `mix = 0.3` |

All runs: DIIS diagonalization, \(K=5\), NVE (no thermostat / barostat),
\(\Delta t = 0.5\) fs, 1 warm-up / cold seed + 4 production steps (chunked).

| Atom | x | y | z (Å) |
| ---- | - | - | - |
| O | 5.0000 | 5.0000 | 5.0000 |
| H1 | 5.7586 | 5.0000 | 5.5043 |
| H2 | 4.2414 | 5.0000 | 5.5043 |

| Atom | \(x\) | \(y\) | \(z\) (Å) |
| ---- | ----: | ----: | --------: |
| O1 | 6.0000 | 6.0000 | 5.39625 |
| O2 | 6.0000 | 6.0000 | 6.60375 |

## Oracle and performance policy

The pinned observable is the literature XL-BOMD / NVE conservation envelope plus
method diagnostics, not a golden Atomiverse total energy. CP2K is the oracle
family and is only ever run offline, never inside ctest. For the GPW sections
with a running regression no `cp2k` / `cp2k.psmp` / `cp2k.sopt` binary or CP2K
Lmod module was available on the pin host (Arrhenius login-class inventory), so
the CP2K twin inputs (short NVE Quickstep MD on the same system,
`EXTRAPOLATION ASPC` as CP2K's closest density-guess analogue, `&KPOINTS` where
relevant) are kept with the CP2K oracle tooling but unexecuted; those sections
are **`perf_exempt`** (no fair multi-step AIMD wall). CP2K ASPC is not identical
to Niklasson XL (nor, with low `MAX_SCF`, to the shadow functional with true
single-diagonalization forces): a twin compares workflow class, not a
bit-identical density history. When a binary is available the twin is re-run
offline and the record gains CP2K version, measured drift and a fair wall
(\(T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \le 10\)). The linearized multi-k variant
is CP2K-anchored (CP2K 2026.1 converged-BOMD `.ener` trace) and has a measured
ratio of 1.3. Absolute E/F parity vs CP2K lives on the single-point pages
([gpw-uks-o2-gamma](gpw-uks.md#gpw-uks-o2-gamma), `gpw-uks-forces`,
[gpw-kpoint-forces](gpw-rks.md#gpw-kpoint-forces),
[gpw-kpoint-uks-forces](gpw-uks.md#gpw-kpoint-uks-forces)). Regression wall
budget ≤ 60 s (login-node class CPU). Build: `buildenv-gcc/2026.03-mpich`,
`ATV_BUILD_GPW=ON` (+ HF + PW) or `ATV_BUILD_HF=ON`, with MD.

## Shared limitations

- **Envelope, not absolute energy.** A correct path is not a high-precision
  total-energy twin of CP2K Quickstep.
- **Short trajectories** (1 + 4 steps on GPW): plumbing / conservation checks,
  not equilibration, thermostats, statistics or multi-ps production AIMD.
  Long-time drift — XL-BOMD's headline property — is not demonstrated here.
- **Envelope cutoffs** are cheaper than the absolute E/F pins.
- **Not combined:** XL + OT, + CPMD / second-gen CP, + smearing, + DFT+U,
  hybrid / meta-GGA XC, GAPW; no CUDA.

## References

- A. M. N. Niklasson, *Phys. Rev. Lett.* **100**, 123004 (2008)
- A. M. N. Niklasson et al., *J. Chem. Phys.* **130**, 214109 (2009) — Table I
- A. M. N. Niklasson & M. J. Cawkwell, *Phys. Rev. B* **86**, 174308 (2012)
- A. M. N. Niklasson, *J. Chem. Phys.* **147**, 054103 (2017)
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW
- J. VandeVondele et al., *Comput. Phys. Commun.* **167**, 103 (2005) — GPW
- S. Goedecker, M. Teter, J. Hutter, *Phys. Rev. B* **54**, 1703 (1996) — PADE LSD
- Martyna & Tuckerman, *J. Chem. Phys.* **110**, 2810 (1999) — MT Poisson
- Monkhorst & Pack, *Phys. Rev. B* **13**, 5188 (1976)
- Related: [gpw-bomd-water](dynamics-bomd.md#gpw-bomd-water) (plain BOMD),
  [gpw-bomd-kpoints](dynamics-bomd.md#gpw-bomd-kpoints),
  [cpmd-rhf-h2](dynamics-car-parrinello.md#cpmd-rhf-h2); roadmaps
  [molecular-dynamics.md](../../roadmap/molecular-dynamics.md),
  [dft.md](../../roadmap/dft.md); narrative [ch09-gpw.md](../ch09-gpw.md),
  [ch10-classical-md.md](../ch10-classical-md.md)

## xl-bomd-water — SCF-converged XL-BOMD, Γ-RKS H₂O NVE (GPW) {#xl-bomd-water}

Closed-shell RKS Γ-point diagonalization/DIIS GPW path; the plain-BOMD
[gpw-bomd-water](dynamics-bomd.md#gpw-bomd-water) NVE twin (PBE /
DZVP-MOLOPT-GTH / GTH-PBE / MT, 200 Ry, dt 0.5 fs, 1 + 4 steps) with XL
engaged. SCF still runs to `tol` each step; the XL machinery improves the guess,
and the force-pass SCF also warm-starts from the just-converged MOs.

```cpp
// frame: 3 atoms at the H₂O geometry above, H stretch vx = ±0.0015 Å/fs
frame.box = cpu::Box::cubic(10.0);
cpu::gpw::GpwDft dft{{O, H}, 200.0, cpu::gpw::XcKind::PBE};
dft.max_iter = 60;
dft.tol = 1e-5;
dft.mix = 0.3;
dft.nonperiodic = true;
cpu::gpw::GPWEvaluator gpw(std::move(dft), /*compute_forces=*/true);
cpu::electronic::XlBomdForce eval{std::move(gpw), /*k=*/5};
```

**Gates:** relative total-energy drift \(\delta \le 0.05\), SCF converged at
every sampled step, bounded O–H, finite trajectory. Oracle: literature envelope
(CP2K twin input kept, no binary on the pin host); `perf_exempt`.

## xl-bomd-rhf-water — XL-BOMD on Gaussian SCF, RHF/STO-3G H₂O NVE {#xl-bomd-rhf-water}

### Summary

The same method-agnostic propagator (`cpu::electronic::XlBomd`) behind the
Gaussian SCF methods: `XlBomdMethod<Method>` hands `aux()` to the SCF as the
initial guess each MD step; the SCF still runs to `tol`, so warm steps converge
in fewer iterations **and** without ASPC's corrector bias. The GPW sibling is
[xl-bomd-water](#xl-bomd-water); the Car–Parrinello sibling
[cpmd-rhf-h2](dynamics-car-parrinello.md#cpmd-rhf-h2) is the *other* answer to
"what is the density now" on this seam. The linearized / zero-SCF shadow variant
is GPW-only.

### API

| Public surface | Meaning |
| --- | --- |
| `cpu::electronic::XlBomdMethod<Method>{Method method, int k = 5}` | Move a configured `RHF` / `UHF` / `RKS` / `UKS` in and engage XL auxiliary-density propagation with dissipation order \(K\) (history depth \(K+1\)). \(K \notin [3,9]\) throws `std::invalid_argument` at build (lookup table, no fallback) |
| `int xl_advances() const` (on `wrapper.method()`) | Auxiliary propagations taken — one per **converged nested** run; 0 under plain BOMD. The public witness that the history is on the nuclear clock |

The wrapper (`cpu/electronic/aimd_method.h`) owns `Method` and implements the
nested DriverNode and timestep protocols. Construct the method with its `basis`
and set `tol` / `max_iter` before moving it in; inspect it afterwards through
`wrapper.method()`.

#### The guess ladder

`drive_scf_` resolves **one** ordered chain per step — the entire contract:

| Rank | Source | When it wins |
| --- | --- | --- |
| 1 | the caller's **explicit** `SCFInitialGuess` | whenever `run(frame, &guess)` supplies one |
| 2 | **XL** `aux()` | `xl_bomd` and the history is seeded |
| 3 | **ASPC** `predict()` | `predictor(n)` set and the polynomial is filled |
| 4 | **last-P** reuse | a previous converged density exists |

An explicit caller guess outranks the ASPC prediction (letting ASPC overwrite it
diverges by 2.8e-5 Ha; recorded, no unit test in the tree pins the order).

#### Exclusivity and history integrity

`xl_bomd` is mutually exclusive with `predictor(...)` (ASPC) and with `cpmd`
(Car–Parrinello propagates the *coefficients* and never runs the SCF XL would
be guessing for), in both directions; each pair throws on the first nested
entry, as `GPWEvaluator::build` does. The auxiliary density lives on the
**nuclear** clock:

- it advances exactly **once per converged nested run** (`xl_advances()`
  increments in lockstep with the MD steps);
- a **standalone** `run()` / `forces()` is allowed but leaves the history
  untouched — a single point is somewhere else in configuration space (GPW
  behaves the same);
- an **unconverged** nested run *drops* the history rather than propagating an
  SCF residual into every later guess.

Channel convention: RHF/RKS propagate the factor-2 total density in one
channel; UHF/UKS/ROHF/ROKS propagate α and β independently; GHF propagates the
single 2n-dimensional spinor density.

### Usage

```cpp
cpu::scf::RHF rhf_base{cpu::scf::BasisSet::STO3G};   // or UHF / RKS / UKS
rhf_base.tol = 1e-6;                    // Ha, |dE| SCF exit
cpu::electronic::XlBomdMethod<cpu::scf::RHF> rhf{std::move(rhf_base), /*k=*/5};

cpu::md::MD sim;
sim.set_integrator(cpu::md::VelocityVerlet{0.25});   // fs — see below
sim.set_evaluator_ref(rhf);             // borrow: diagnostics stay readable
sim.build();
sim.run(frame, 20);                     // frame.meta.get(kEnergy) [eV]
rhf.method().xl_advances();             // == 20: one advance per converged step
```

Step 1 is a cold seed (nothing has been propagated yet); every later step is a
normal SCF that starts from `aux()`.

#### Choosing the timestep — κ = ω²Δt² is not decoration

The auxiliary density is a harmonic oscillator whose Verlet coupling
\(\kappa = \omega^2\Delta t^2\) is **fixed by the Table-I row**, so raising
\(\Delta t\) does not "cost accuracy" gradually — it moves the auxiliary
oscillator toward the nuclear frequency and the guess stops tracking. Measured
on H₂O/STO-3G (guess error = one-shot energy minus the converged energy at the
same geometry [Ha]; full table in the case `REFERENCE.md`):

| Δt (fs) | steps per O–H period | err(XL) / err(last-P reuse) |
| --- | --- | --- |
| 0.25 | ≈ 36 | **128–507× better** |
| 0.5 | ≈ 18 | 3.9–7.9× better (one sampled step: 8× *worse*) |
| 1.0 | ≈ 9 | **0.4–1.8× — mostly worse than plain reuse** |

Rule of thumb: XL-BOMD wants Δt well under the fastest vibrational period —
here ≈ 0.25 fs with hydrogens moving. That is why this section runs at 0.25 fs
while its Car–Parrinello siblings run at 0.05 fs. The GPW XL sections run at
0.5 fs because they pin only the drift envelope — no iteration margin — and so
never depend on the tracking regime.

The *iteration count* is a coarser meter than the guess error, because the SCF
exits on \(|\Delta E| < \)`tol`: once a guess is "good enough" the counter
saturates. Warm-step mean over steps 9–20 at Δt = 0.25 fs, `tol` = 1e-6 Ha:

| Guess owner | mean SCF iterations |
| --- | --- |
| XL-BOMD (K = 5) | **5.67** |
| plain BOMD (last-P reuse) | 6.58 |
| ASPC(3) | 6.00 |
| cold start (fresh method each geometry) | 8.25 |

### Pin

- **Contract** (no unit test in the tree gates it): member-surface defaults; both exclusivity throws; \(K \notin [3,9]\) throws;
  advance-on-converged-nested-only (standalone and unconverged runs leave
  `xl_advances()` alone); restricted and unrestricted advance conventions; and
  the four-rung ladder order, probed with a one-shot energy fingerprint
  (acceptance items ac-001 ladder, ac-002 exclusivity, ac-003 history
  integrity, ac-004 the XL signature). The Table-I coefficients have no unit
  test in the tree either.
- **Section** (0.41 s): RHF/STO-3G H₂O NVE, 20 steps at Δt = 0.25 fs, three
  legs on one trajectory (XL / plain-BOMD twin / cold-start twin). Frozen:
  warm-mean iteration gain ≥ 1.50 vs cold (measured 2.58) and ≥ 0.40 vs reuse
  (measured 0.92); relative energy-drift envelope ≤ 2.0e-5 (measured
  1.955e-6); `xl_advances() == 20` and the twin's `== 0`; the two MD legs stay
  within 1e-6 Å. The case `REFERENCE.md` holds the full measured landscape (dt
  scan, tol scan, guess-error table) and the reason for each margin.
- Mode storage, ladder, advance hook and exclusivity gates:
  `mean_field_method.h`, `src/cpu/scf/mean_field_method.cpp`.

### Limitations

- Linearized / zero-SCF shadow XL-BOMD needs the shadow-energy force assembly
  and is GPW-only ([xl-bomd-linearized-forces](#xl-bomd-linearized-forces)).
- No multi-k XL on the AO methods, no CUDA.
- A 20-step section can only guard an *envelope*; a thousand-step drift
  demonstration is production usage, not a ≤ 60 s regression.
- No third-party XL-BOMD twin (`perf_exempt`): SCF iteration counts are not
  comparable across programs, so the pinned observable is a relative property
  of three runs of this code.

## xl-bomd-uks — SCF-converged XL-BOMD, Γ-UKS O₂ triplet NVE (GPW) {#xl-bomd-uks}

### Summary

Open-shell Γ-UKS GPW forces with independent auxiliary densities \(P^\alpha\)
and \(P^\beta\), propagated with the same Table-I recursion
(`xl_` = α, `xl_beta_` = β inside the evaluator) and used as SCF initial spin
densities each step — the dual-spin counterpart of the closed-shell
[xl-bomd-water](#xl-bomd-water). Löwdin \(\langle S^2\rangle\) is computed after
each SCF (ideal high-spin \(S(S+1)=2\)).

```cpp
constexpr double L = 12.0;     // Å, O₂ centred at z = L/2 ∓ R/2
frame.span(kAtom_vz)[0] = -0.0008;   // antisymmetric stretch [Å/fs]
frame.span(kAtom_vz)[1] = 0.0008;
frame.box = cpu::Box::cubic(L);
cpu::gpw::GpwDft dft{{O}, 200.0, cpu::gpw::XcKind::PADE};
dft.max_iter = 80;
dft.tol = 1e-5;
dft.mix = 0.4;
dft.nonperiodic = true;
dft.n_alpha = 7;  // multiplicity 3, 12 valence e⁻
dft.n_beta = 5;
cpu::gpw::GPWEvaluator gpw(std::move(dft), /*compute_forces=*/true);
cpu::electronic::XlBomdForce eval{std::move(gpw), /*k=*/5};
// after warm-up: E0 = kEnergy + kKinetic [eV]; s2 = eval.method().last_s2()
```

**Why 200 Ry?** The absolute UKS energy / force twin is at 400 Ry on
`gpw-uks-o2-gamma` / `gpw-uks-forces`; multi-step UKS AIMD at that cost exceeds
the ≤60 s budget. 200 Ry keeps the same method family inside the time gate;
this section gates **dual-spin XL + MD binding + NVE conservation**, not
absolute E/F parity.

### Validation

| Quantity | Atomiverse / pin | CP2K | Tol | Notes |
| -------- | ---------------: | ---: | --- | ----- |
| Relative total-energy drift \(\delta\) | \(\le 0.05\) (envelope) | n/a (no binary) | **0.05** | A7 |
| SCF converged each step | dual XL spin guesses | n/a | — | Broken α-only / channel-swap fails immediately |
| PE after warm-up | finite; \(< -100\) eV | n/a | bound | O₂ PADE GPW O(−10²) eV |
| KE | finite; \(\ge 0\); non-trivial after warm-up | n/a | — | Antisymmetric stretch seeds KE |
| \(\langle S^2\rangle\) | \([1.5,\,2.5]\) after warm-up and production | n/a | window | Triplet Löwdin window |
| Soft O–O window | \(0.9 < r_{\mathrm{OO}} < 1.6\) Å | n/a | Å | No dissociation / collapse on short traj |
| Coordinates | finite over traj | n/a | — | Trajectory non-explosion |
| Absolute \(E_{\mathrm{tot}}\) | **not pinned** | — | — | Envelope only |

CP2K twin design (`o2_uks_xl_bomd_nve.inp`): short NVE Quickstep UKS MD on the
same O₂ cell (PADE / DZVP-MOLOPT-GTH / GTH-PBE-q6 / MT / MULT 3), unexecuted.
`perf_exempt`; for scale, the single-point UKS energy wall vs CP2K on
`gpw-uks-o2-gamma` has ratio \(\approx 0.98\) (energy only, not this
trajectory).

### Limitations

- 200 Ry / `tol=1e-5` / 80 SCF iters — enough for the NVE envelope inside
  ≤60 s, not production AIMD accuracy or absolute E/F at the UKS single-point
  settings (400 Ry).
- Γ-UKS diagonalization only here; multi-k UKS is
  [xl-bomd-uks-kpoints](#xl-bomd-uks-kpoints), zero-SCF is
  [xl-bomd-linearized-uks](#xl-bomd-linearized-uks).

## xl-bomd-kpoints — SCF-converged multi-k XL-BOMD, periodic H₂ NVE (GPW) {#xl-bomd-kpoints}

### Summary

Multi-k closed-shell RKS GPW forces: one complex auxiliary density \(P_k\) per
sampled Bloch point via `cpu::electronic::XlBomdCplx`, fed as the vector of
guesses into the multi-k SCF (SCF still to `tol`). System: the periodic H₂ of
[gpw-kpoint-forces](gpw-rks.md#gpw-kpoint-forces) at 100 Ry, \(1\times1\times2\)
unreduced mesh (`k_time_reversal=false` → \(n_k=2\)). Multi-k force quality is
not gated here (no standing multi-k force regression; see
[gpw-kpoint-forces](gpw-rks.md#gpw-kpoint-forces)); this section gates **MD
binding + multi-k XL guess plumbing + NVE envelope**.

```cpp
constexpr double bohr2ang = 0.52917720859;
constexpr double L_bohr = 12.0;      // R = 1.0 Bohr along x, cell-centred
frame.span(kAtom_vx)[0] = -0.0010;   // Å/fs antisymmetric stretch
frame.span(kAtom_vx)[1] =  0.0010;
frame.box = cpu::Box::cubic(L_bohr * bohr2ang);
cpu::gpw::GpwDft dft{{H}, 100.0, cpu::gpw::XcKind::PADE};
dft.max_iter = 60;
dft.tol = 1e-5;
dft.mix = 0.3;
dft.k_mesh[0] = 1;
dft.k_mesh[1] = 1;
dft.k_mesh[2] = 2;
dft.k_time_reversal = false;   // two sampled k (unreduced)
cpu::gpw::GPWEvaluator gpw(std::move(dft), /*compute_forces=*/true);
cpu::electronic::XlBomdForce eval{std::move(gpw), /*k=*/5};
// after warm-up: eval.last_result().n_kpts, .P_k, .converged
```

### Validation

| Quantity | Atomiverse / pin | Oracle | Tol / gate | Notes |
| -------- | ---------------: | ------ | ---------- | ----- |
| Relative NVE drift \(\delta\) | \(\le 0.05\) | Literature XL-BOMD/NVE envelope | hard | scale \(\max(1,\|E_0\|)\) |
| SCF converged each chunk | true | product requirement | hard | multi-k XL \(P_k\) guesses |
| \(n_{\mathrm{kpts}}\) | \(\ge 2\) | mesh \(1\times1\times2\) unreduced | hard | `last_result().n_kpts` |
| \(P_k\) payload | `P_k.size() == n_kpts` | multi-k SCF contract | hard | after warm-up |
| PE after warm-up | \(< -5\) eV | H₂ PADE bound | hard | not a golden energy |
| KE after warm-up | \(\ge 0\), \(\gt 10^{-12}\) eV | kinetic non-trivial | hard | stretch seed |
| \(r_{\mathrm{HH}}\) after prod | \((0.4, 1.5)\) Å | soft bond window | hard | no collapse / fly-off |

The Table-I \((\kappa,\alpha,c)\) coefficients at \(K=5\) are exact (Niklasson
*JCP* **130**, 214109 (2009)); no unit test in the tree gates them.

CP2K twin sketch (short NVE Quickstep MD, ASPC-class extrapolation +
`&KPOINTS`) kept unexecuted; `perf_exempt`.

### Limitations

- Diagonalization/DIIS only — OT multi-k energy is a separate path without XL.
- The 100 Ry envelope cutoff is cheap; this is not a multi-k force accuracy pin.

## xl-bomd-linearized-forces — Zero-SCF linearized XL-BOMD, Γ-RKS H₂O NVE (GPW shadow BO) {#xl-bomd-linearized-forces}

### Summary

Nuclear forces of the shadow Born–Oppenheimer potential with **no iterative
SCF** on warm MD steps, on the system of [xl-bomd-water](#xl-bomd-water) (only
the `XlBomdLinearized{}` tag differs). The pin is the literature shadow-XL / NVE
envelope plus **`last_result().iters == 0`** on warm steps.

### Method

| Stage | Action |
| ----- | ------ |
| 0. Cold seed | Full Γ-RKS SCF → `xl_->advance(D_scf)`; history becomes ready |
| 1. Fixed-\(n\) KS | `solver_->max_iter = 0` with `guess_P = n_aux` → \(H[n]\), \(E_{\mathrm{KS}}[n]\) |
| 2. One diag | Generalized Fock diagonalization → \(D = q[n]\), \(C_{\mathrm{occ}}\) |
| 3. Shadow \(E\) | \(\mathcal{E} = \mathrm{Tr}(H D) - \mathrm{Tr}(H n) + E_{\mathrm{KS}}[n]\) → `kEnergy` [eV] |
| 4. Forces | `internal::gpw_forces_at_density(D, C_occ, H[n])` → eV/Å |
| 5. XL advance | `xl_->advance(D)` (Table-I \(K=5\)); `last_result().iters = 0`, `converged = true` |

No DIIS mixing loop and no residual-threshold iteration on warm steps; savings
apply only after `xl_->ready()` (the cold seed still pays full SCF). Warm steps
cost one \(H\) build + one diagonalization vs iterative DIIS; that saving is
not claimed as a measured fair-bench ratio.

```cpp
cpu::gpw::GpwDft dft{{O, H}, 200.0, cpu::gpw::XcKind::PBE};
dft.max_iter = 60;
dft.tol = 1e-5;
dft.mix = 0.3;
dft.nonperiodic = true;
cpu::gpw::GPWEvaluator gpw(std::move(dft), /*compute_forces=*/true);
// The third argument is a tag type, not a bool: it selects the zero-SCF
// shadow-BO force path (and turns forces on) at construction.
cpu::electronic::XlBomdForce eval{std::move(gpw), /*k=*/5,
                                  cpu::electronic::XlBomdLinearized{}};
// sim.run(frame, 1): cold seed, iters > 0; then 2 chunks × 2 production steps
```

### Validation

| Quantity | Atomiverse / pin | Oracle | Tol | Notes |
| -------- | ---------------: | ------ | --- | ----- |
| Relative NVE drift \(\delta\) | \(\le 0.10\) | literature shadow-XL / NVE envelope | 0.10 | looser than SCF-XL 0.05 (residual heat) |
| Warm `last_result().iters` | \(= 0\) | zero iterative SCF | — | product diagnostic |
| Warm `converged` | true | stationary for shadow functional | — | |
| Cold seed `iters` | \(> 0\), converged | full SCF seed | — | history not ready until seed |
| PE after seed | \(< -100\) eV | bound H₂O | — | |
| KE after seed | \(\ge 0\), non-trivial | NVE stretch | — | |
| \(r_{\mathrm{OH}}\) after production | both \(< 1.8\) Å | soft bond window | — | no dissociation |

The shadow identity \(\mathcal{E} = E_{\mathrm{KS}}\) at \(D=n\) (Niklasson
2012/2017), the exact Table-I \(\kappa,\alpha,c_k\) at \(K=5\) (Niklasson 2009
Table I) and the build-time throw on illegal combos (no silent degrade) have no
unit test in the tree.

CP2K twin sketch: short NVE Quickstep MD, ASPC extrapolation, **`MAX_SCF 1`**
as a low-budget stand-in for one-shot KS, same H₂O / PBE / DZVP-MOLOPT-GTH /
200 Ry / MT cell; unexecuted. `perf_exempt`.

### Limitations

- Do not tighten the \(\delta \le 0.10\) window without re-pinning against
  literature/CP2K.

## xl-bomd-linearized-kpoints — Zero-SCF linearized XL-BOMD on the k path, periodic H₂ NVE {#xl-bomd-linearized-kpoints}

*No section of the case runs this variant.*

### Summary

Zero-SCF linearized XL-BOMD on Monkhorst–Pack meshes, RKS and collinear UKS,
with complex Bloch densities: the `XlBomdLinearized` tag on a `GPWEvaluator`
whose `GpwDft` `k_mesh` is non-Γ; one `cpu::electronic::XlBomdCplx` per sampled
(k, σ). The cold seed runs one full multi-k SCF; warm steps do one KS build at
fixed \(\{n_k\}\) (`max_iter = 0`), one generalized diagonalization per (k, σ)
→ \(D_k^\sigma = q[n]\), and the k-weighted shadow energy

\[
\mathcal{E}(R, \{n_k\})
  = \sum_k w_k \sum_\sigma \mathrm{Re}\,\mathrm{Tr}\!\left[
    H_k^\sigma[n]\,\bigl(D_k^\sigma - n_k^\sigma\bigr)\right]
  + E_{\mathrm{KS}}[n],
\]

with k-summed analytic forces at the \((D_k, \varepsilon_k,
C_k^{\mathrm{occ}})\) payload — since the \(C_k\) are exact eigenvectors of
\(H_k[n]\), the Γ off-BO energy-weighted density
\(W = C (C^\dagger H C) C^\dagger\) reduces to the standard
\(\sum_i f \varepsilon_i C C^\dagger\) the k assembly builds. Then
`XlBomdCplx::advance(D_k^σ)` per (k, σ) with Table-I \(K = 5\). The UKS leg uses
the `max_iter == 0` dual-spin fixed-density branch of `run_gpw_scf_kpoints_uks`
(the k-UKS sibling of the multi-k RKS CPMD eval) and the `H_ks_k_beta` payload
of `GpwResult`.

This variant is **CP2K-anchored**: the per-step shadow \(E_{\mathrm{pot}}\) is
compared against the CP2K converged-SCF k-point BOMD `.ener` trace of the
identical twin input (the BO surface the shadow potential tracks to
\(O((D-n)^2)\)), plus an SCF-XL vs linearized A/B on the same trajectory and an
exact UKS spin-degeneracy identity.

### Method

| Stage | Action |
| ----- | ------ |
| 0. Cold seed | Full multi-k SCF → `advance(P_k^σ_scf)` per (k, σ); histories ready |
| 1. Fixed-\(n\) KS | `max_iter = 0` with `guess_P_k^σ = n_k^σ` → \(H_k^\sigma[n]\), \(S_k\), \(E_{\mathrm{KS}}[n]\) |
| 2. Diags | `KOrth(S_k)` + `diagonalize_k(H_k^σ)` per (k, σ) → \(D_k^\sigma\), \(\varepsilon_k^\sigma\) (occ 2 RKS / 1 per spin) |
| 3. Shadow \(E\) | k-weighted formula above → `kEnergy` [eV] |
| 4. Forces | `internal::gpw_forces_from_scf` on the \((D_k, \varepsilon_k, C_k^{\mathrm{occ}})\) payload → eV/Å |
| 5. XL advance | `advance(D_k^σ)` per (k, σ); `last_result().iters = 0`, `converged = true` |

System: the periodic H₂ of [xl-bomd-kpoints](#xl-bomd-kpoints) at the
[gpw-bomd-kpoints](dynamics-bomd.md#gpw-bomd-kpoints) pin settings — CUTOFF
300 Ry, MP 1×1×2 unreduced (2 k, CP2K `FULL_GRID`), cold-seed `tol = 1e-7`,
v₀ = ∓0.001 Å/fs, 1 cold seed + 4 production steps. Three legs: **A** SCF-XL
RKS-k anchor, **B** linearized RKS-k (product under test), **C** linearized
UKS-k with `n_alpha = n_beta = 1` (spin-degeneracy identity vs B). Metallic
smearing on the fixed-density k eval throws (fractional occupations need a
diagonalization inside the eval).

```cpp
cpu::gpw::GpwDft dft{{H}, 300.0, cpu::gpw::XcKind::PADE};
dft.max_iter = 60;
dft.tol = 1e-7;
dft.mix = 0.3;
dft.k_mesh[2] = 2;
dft.k_time_reversal = false;         // two sampled k (CP2K FULL_GRID)
// dft.n_alpha = dft.n_beta = 1;     // collinear UKS leg (dual P_k^σ)
cpu::gpw::GPWEvaluator gpw(std::move(dft), /*compute_forces=*/true);
cpu::electronic::XlBomdForce eval{std::move(gpw), /*k=*/5,
                                  cpu::electronic::XlBomdLinearized{}};
```

### Validation and performance

Oracle: **CP2K 2026.1** (`cp2k.psmp`, EasyBuild `2026.1-gmpflf-2026.03`,
Arrhenius epyc login node) converged-SCF k-point NVE `.ener` trace of the
identical twin input, offline.

Per-step Pot [Ha]:

| Step | CP2K 2026.1 | A (SCF-XL) | B (linearized) | C (UKS lin.) |
| ---- | ----------- | ---------- | -------------- | ------------ |
| 0 | −1.074050184 | −1.074456793 | −1.074456793 | −1.074456793 |
| 1 | −1.097682104 | −1.097482136 | −1.097521796 | −1.097521784 |
| 2 | −1.125544577 | −1.125333978 | −1.125443925 | −1.125443902 |
| 3 | −1.130531455 | −1.130595564 | −1.130587128 | −1.130587132 |
| 4 | −1.121507812 | −1.121740443 | −1.121722124 | −1.121722125 |

| Gate | Band | Measured worst |
| ---- | ---- | -------------- |
| B shadow \(E_{\mathrm{pot}}\) vs CP2K Pot | ± 2e-3 Ha | **4.066e-4 Ha** (≈ the documented 4.07e-4 Ha lattice-sum/PP-tail model band; production rows 0.6–2.1e-4) |
| A/B trace (linearized vs SCF-XL) | ± 1e-3 Ha | **1.099e-4 Ha** |
| C/B spin-degeneracy identity | ± 1e-4 Ha | **2.31e-8 Ha** |
| NVE drift envelope | ≤ 0.05 | B 4.056e-3, C 4.056e-3 (A 4.253e-3; CP2K itself 4.4e-3) |
| Warm steps | `iters == 0`, converged, `n_kpts == 2` | ✓ |
| Cold seed | full SCF, `iters > 0` | A/B 8 iters, C 7 |

Fair wall (same host, OMP=8, single rank):

| Metric | Value |
| ------ | ----- |
| \(T_{\mathrm{cp2k}}\) (4-step converged-BOMD k twin, CUTOFF 300) | **1.49 s** (CP2K-internal 1.045 s, quiet node) |
| \(T_{\mathrm{atv}}\) (leg B: 1 seed single point + 4 linearized steps) | **1.96 s** |
| ratio \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) | **1.3** (within 10×) |
| All three legs | 7.7 s (≤ 60 s budget) |

## xl-bomd-linearized-uks — Zero-SCF linearized XL-BOMD on Γ-UKS, O₂ triplet NVE {#xl-bomd-linearized-uks}

*No section of the case runs this variant.*

### Summary

Zero-SCF linearized XL-BOMD on collinear spin-polarized Γ GPW: the system of
[xl-bomd-uks](#xl-bomd-uks) with only the `XlBomdLinearized{}` tag added (dual
α/β `cpu::electronic::XlBomd` propagators), giving a direct SCF-XL vs
linearized-XL A/B on the same O₂ triplet. The cold seed runs a full dual-spin
UKS SCF (~23 iterations) that seeds **both** auxiliary densities
\(n^\sigma = P^\sigma_{\mathrm{aux}}\); warm steps build \(H^\sigma[n]\) at fixed
auxiliary densities (`max_iter = 0`), one generalized diagonalization **per
spin** \(D^\sigma = q^\sigma[n]\) (occupancy 1), and the dual-spin shadow energy

\[
\mathcal{E}(R, n^\alpha, n^\beta)
  = \sum_{\sigma}\Bigl[\mathrm{Tr}\!\bigl(H^\sigma[n]\,D^\sigma\bigr)
  - \mathrm{Tr}\!\bigl(H^\sigma[n]\,n^\sigma\bigr)\Bigr]
  + E_{\mathrm{KS}}[n^\alpha, n^\beta],
\]

with Hellmann–Feynman forces via the dual-spin fixed-density overload of
`internal::gpw_forces_at_density`, then `XlBomd::advance(D^σ)` per spin. The
dual-spin shadow-energy fixed-point identity is unit-pinned.

| Stage | Action |
| ----- | ------ |
| 0. Cold seed | Full dual-spin UKS SCF → `advance(D^σ_scf)` per spin; both histories ready |
| 1. Fixed-\(n\) KS | `max_iter = 0` with `guess_P^σ = n^σ` → \(H^\sigma[n]\), \(E_{\mathrm{KS}}[n^\alpha,n^\beta]\) |
| 2. Two diags | One generalized diagonalization per spin → \(D^\sigma = q^\sigma[n]\), \(C^\sigma_{\mathrm{occ}}\) |
| 3. Shadow \(E\) | \(\mathcal{E}\) per the dual-spin formula → `kEnergy` [eV] |
| 4. Forces | `internal::gpw_forces_at_density(D^α, C^α, H^α, D^β, C^β, H^β)` → eV/Å |
| 5. ⟨S²⟩ | Löwdin at the shadow densities (Szabo & Ostlund §2.5) |
| 6. XL advance | `advance(D^σ)` per spin; `last_result().iters = 0`, `converged = true` |

**Why O₂ and not an OH radical?** The OH doublet's cold-seed DIIS is
pathological in this tree (locks in only at ~196 iterations at every
`mix`/`tol` probed, PBE; PADE worse). The O₂ triplet seed converges in ~23
iterations and pins the same dual-spin linearized-XL physics.

```cpp
cpu::gpw::GpwDft dft{{O}, 200.0, cpu::gpw::XcKind::PADE};
dft.max_iter = 80;
dft.tol = 1e-5;
dft.mix = 0.4;
dft.nonperiodic = true;
dft.n_alpha = 7;  // multiplicity 3, 12 valence e⁻
dft.n_beta = 5;
cpu::gpw::GPWEvaluator gpw(std::move(dft), /*compute_forces=*/true);
cpu::electronic::XlBomdForce eval{std::move(gpw), /*k=*/5,
                                  cpu::electronic::XlBomdLinearized{}};
// s2 = eval.method().last_s2()  — Löwdin at the shadow densities
```

### Validation and performance

| Quantity | Atomiverse / pin | Oracle | Tol | Notes |
| -------- | ---------------: | ------ | --- | ----- |
| Relative NVE drift \(\delta\) | \(\le 0.10\); measured **1.8e-07** (8 threads) | literature shadow-XL / NVE envelope | 0.10 | looser than SCF-XL-UKS 0.05 (residual heat) |
| Warm `last_result().iters` | \(= 0\) | zero iterative SCF | — | product diagnostic |
| Warm `converged` | true | stationary for shadow functional | — | |
| Cold seed `iters` | \(> 0\) (~23), converged | full dual-spin SCF seed | — | both spin histories seeded |
| ⟨S²⟩ (Löwdin, shadow densities) | \([1.5, 2.5]\); measured 2.0047 (seed) → 2.0047 (final) | triplet \(S(S+1)=2\) | window | at seed and every sampled step |
| PE after seed | \(< -100\) eV | bound O₂ | — | |
| KE after seed | \(\ge 0\), non-trivial | NVE stretch | — | |
| \(r_{\mathrm{OO}}\) after production | \(0.9 < r < 1.6\) Å; measured 1.220 Å | soft bond window | Å | no dissociation / collapse |

The dual-spin shadow identity \(\mathcal{E} = E_{\mathrm{KS}}\) at
\(D^\sigma = n^\sigma\) (Niklasson 2012/2017) and the Table-I
\(\kappa,\alpha,c_k\) for \(K\in[3,9]\), exact per spin (Niklasson 2009 Table I),
have no unit test in the tree.

CP2K twin sketch: short NVE Quickstep UKS MD, low `MAX_SCF` as a stand-in for
one-shot per-spin KS, same O₂ / PADE / DZVP-MOLOPT-GTH / GTH-PBE-q6 / 200 Ry /
MT / MULT 3; unexecuted. `perf_exempt`; recorded wall 40 s for 5 steps (worker
host, 8 threads; ≤60 s budget).

## xl-bomd-uks-kpoints — SCF-converged multi-k UKS XL-BOMD, periodic H₂ NVE {#xl-bomd-uks-kpoints}

*No section of the case runs this variant.*

### Summary

Dual-spin multi-k XL: independent complex auxiliary densities \(P_k^\alpha\)
and \(P_k^\beta\) per sampled \(k\) (one `XlBomdCplx` per \((k,\sigma)\)), used
as the multi-k UKS SCF initial densities each ionic step (SCF still to `tol`).
Multi-k UKS solves \(H_k^\sigma\) per sampled \(k\); complex Bloch densities are
Hermitian and real \((\kappa,\alpha,c)\) act element-wise on
`linalg::CplxMatrix`. System: the periodic H₂ of
[xl-bomd-kpoints](#xl-bomd-kpoints) (100 Ry, `k_mesh = {1,1,2}`,
`k_time_reversal = false`, \(n_k = 2\), PADE (LSD)). Multi-k UKS force math:
[gpw-kpoint-uks-forces](gpw-uks.md#gpw-kpoint-uks-forces) /
[gpw-kpoints-uks](gpw-uks.md#gpw-kpoints-uks).

**Why closed-shell fillings on the UKS path?** High-spin H₂ (\(N_\alpha=2\),
\(N_\beta=0\)) is unbound at this level and cannot pin a short NVE drift
window. The check uses \(N_\alpha = N_\beta = 1\) on the collinear UKS path:
both spin channels are non-empty (dual \(P_k^\alpha / P_k^\beta\) XL engages)
while the molecule stays bound.

```cpp
cpu::gpw::GpwDft dft{{H}, 100.0, cpu::gpw::XcKind::PADE};
dft.max_iter = 60;
dft.tol = 1e-5;
dft.mix = 0.3;
dft.n_alpha = 1;               // closed-shell fillings on UKS path
dft.n_beta  = 1;
dft.k_mesh[0] = 1;
dft.k_mesh[1] = 1;
dft.k_mesh[2] = 2;
dft.k_time_reversal = false;   // n_k = 2
cpu::gpw::GPWEvaluator gpw(std::move(dft), /*compute_forces=*/true);
cpu::electronic::XlBomdForce eval{std::move(gpw), /*k=*/5};
```

### Validation

| Quantity | Atomiverse / pin | Oracle | Tol | Notes |
| -------- | ---------------: | ------ | --- | ----- |
| Relative NVE drift \(\delta\) | \(\le 0.05\) | literature XL-BOMD/NVE envelope | 0.05 | \(E_0\) after warm-up |
| SCF converged every step | true | multi-k UKS + dual XL guesses | — | wrong recursion fails immediately |
| \(n_kpts\) | \(\ge 2\) | mesh \(1\times1\times2\) unreduced | — | payload diagnostic |
| \(\lvert P_k\rvert\), \(\lvert P_k^\beta\rvert\) | \(= n_kpts\) (`P_k.size() == P_k_beta.size() == n_kpts`) | dual-spin multi-k XL | — | α-only / RKS leak fails |
| PE after warm-up | \(< -5\) eV | bound closed-shell H₂ | — | high-spin H₂ excluded |
| KE after warm-up | \(\ge 0\), non-trivial | NVE stretch | — | |
| \(\langle S^2\rangle\) | finite, \(\ge 0\) | integer seed formula \(s_z(s_z+1)+N_\beta\) | — | not full Löwdin on multi-k UKS |
| \(r_{\mathrm{HH}}\) after production | \((0.4, 1.5)\) Å | soft bond window | — | no dissociation / collapse |

The Table-I \(\kappa,\alpha,c_k\) at \(K=5\) are exact (Niklasson 2009 Table I);
no unit test in the tree gates them.

CP2K twin sketch: short NVE Quickstep UKS MD with `&KPOINTS` MONKHORST-PACK
\(1\,1\,2\) and ASPC density-guess extrapolation; unexecuted. `perf_exempt`.

### Limitations

- Closed-shell UKS fillings only; open-shell multi-k UKS XL on a bound system
  (e.g. an O₂ mesh) is not covered.
- The multi-k UKS \(\langle S^2\rangle\) is the integer seed formula, not the
  full Löwdin diagnostic used on Γ UKS.
