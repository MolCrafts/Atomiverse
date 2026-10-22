# Born–Oppenheimer molecular dynamics

Born–Oppenheimer MD (BOMD) integrates the nuclei with a fixed-step symplectic
velocity Verlet and re-solves the electronic problem self-consistently on every
ionic step. There is no fictitious electronic dynamics (Car–Parrinello) and no
extended-Lagrangian or ASPC propagator (see
[XL-BOMD](dynamics-xl-bomd.md#xl-bomd-water) and
[Car–Parrinello](dynamics-car-parrinello.md#cpmd-water) for those).

The workflow is composed, not a dedicated driver: the MD preset
`cpu::md::MD` with an NVE `cpu::md::VelocityVerlet{dt}` integrator, and the
concrete electronic method mounted directly at the force phase with
`sim.on(md::Phase::force).add(std::move(method))` — `cpu::gpw::GpwDft` for GPW
(Quickstep-class Gaussian-and-plane-waves), `cpu::pw::PW` for plane-wave DFT.
The mounted method computes analytic ionic forces because the force phase
requests them, and writes `kEnergy` [eV] and `kAtom_f*` [eV/Å] to the Frame.
Parameters are constructor arguments or members of the method; there is no
`*Config` bag, no categorical electronic-structure setter, and the Frame is the
only I/O currency. The case also carries two sections without a page here:
`h2_rhf_sto3g_aimd` (H₂ RHF/STO-3G BOMD with `cpu::scf::RHF`, run on CPU and
CUDA) and `bomd-d3bj-ar2-nve` (Ar₂ RHF + DFT-D3(BJ) NVE); their oracle records
are in the case `REFERENCE.md`.

Case: `ctest -R '^dynamics/bomd$'` — sources and oracle record in [`regressions/dynamics/bomd/`](../../../regressions/dynamics/bomd/). One section of the case per `##` heading below: `atv_regressions --case dynamics/bomd --section <name>`.

## Common setup

Build: the sections need the modules named in `case.cmake` — GPW sections
`ATV_BUILD_GPW` (with HF + PW), PW sections `ATV_BUILD_PW`, plus `MD` and
`ATV_BUILD_REGRESSIONS=ON` — on the CPU toolchain (`buildenv-gcc/2026.03-mpich`).

### Nuclear dynamics (NVE)

\[
\mathbf{v}\!\left(t+\tfrac{\Delta t}{2}\right)
  = \mathbf{v}(t) + \tfrac{\Delta t}{2}\,\mathbf{a}(t),\quad
\mathbf{r}(t+\Delta t)
  = \mathbf{r}(t) + \Delta t\,\mathbf{v}\!\left(t+\tfrac{\Delta t}{2}\right),
\]

\[
\mathbf{a}(t+\Delta t)
  = \mathbf{F}\!\big(\mathbf{r}(t+\Delta t)\big)/m,\quad
\mathbf{v}(t+\Delta t)
  = \mathbf{v}\!\left(t+\tfrac{\Delta t}{2}\right)
  + \tfrac{\Delta t}{2}\,\mathbf{a}(t+\Delta t).
\]

Total energy \(E(t)=E_{\mathrm{pot}}(t)+E_{\mathrm{kin}}(t)\) is conserved up
to bounded, non-secular error when \(\mathbf{F}\approx -\nabla E\) and the
integrator is symplectic (Frenkel & Smit §4.3.4; Allen & Tildesley §3.2;
Marx & Hutter, *Ab Initio Molecular Dynamics*, BOMD chapter).

### Primary observable (envelope, not absolute energy)

The `gpw-bomd-water` and `pw-bomd-h` sections pin the literature symplectic /
BOMD total-energy conservation envelope, not a golden Atomiverse total energy
and not a live CP2K AIMD trajectory:

\[
\delta = \max_t \frac{\lvert E(t)-E_0\rvert}{\max\!\big(1,\,\lvert E_0\rvert\big)}
  \le 0.05,
\]

with \(E_0 =\) `frame.meta.get(kEnergy) + frame.meta.get(kKinetic)` [eV] taken
after one warm-up step (`sim.run(frame, 1)`, which populates PE/KE/forces);
production then runs in chunks and \(\delta\) is evaluated after each chunk.
Every section also requires finite PE, finite KE \(\ge 0\) and finite
coordinates over the trajectory. Absolute \(E_{\mathrm{tot}}\) is not pinned.

### Oracle status (CP2K)

Pins are offline; ctest never runs CP2K. For the two envelope sections no CP2K
(or QE) binary was available on the pin host (no `cp2k` / `cp2k.psmp` /
`cp2k.sopt`, no CP2K Lmod module on the Arrhenius login-class inventory), so
the pin basis is the literature NVE/BOMD conservation envelope plus the
section's finite/bounded trajectory diagnostics, and speed is `perf_exempt`
(no fair-wall twin; regression wall budget ≤ 60 s on a login-node class CPU).
No absolute total-energy digits or CP2K energy series are invented; when a CP2K
AIMD twin is recorded, the status is replaced by version string + input digest
+ measured \(\delta\), and the wall gate becomes
\(T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \le 10\) (or blocked-speed documented
after one optimize pass). The `gpw-bomd-kpoints` record does carry a CP2K
2026.1 twin.

### Shared limitations

- Short trajectories only — not equilibration, thermostats or statistics.
- Fixed cell NVE: no barostat; the methods reject box changes after `build()`.
  Stress write (only when the plan consumes stress) and multi-rank GPW/PW MD
  are out of scope.
- Changes to the analytic force math are gated elsewhere (force pins), not here.

### References

- Frenkel & Smit, *Understanding Molecular Simulation*, 2nd ed., §4.3.4
- Allen & Tildesley, *Computer Simulation of Liquids*, 2nd ed., §3.2
- Marx & Hutter, *Ab Initio Molecular Dynamics* (CUP) — BOMD chapter
- VandeVondele et al., *Comput. Phys. Commun.* **167**, 103 (2005) — GPW
  Quickstep
- Narrative: [ch08-plane-wave.md](../ch08-plane-wave.md),
  [ch09-gpw.md](../ch09-gpw.md), [ch10-classical-md.md](../ch10-classical-md.md)

## gpw-bomd-water — GPW BOMD, H₂O NVE energy drift {#gpw-bomd-water}

### Method

BOMD with GPW forces on every step: `cpu::gpw::GpwDft` mounted at
`md::Phase::force`. GPW Kohn–Sham (VandeVondele et al. 2005): density matrix
in contracted Gaussians (DZVP-MOLOPT-GTH); Hartree / XC / local short-range GTH
PP on a uniform real-space grid; Martyna–Tuckerman isolated Poisson
(`nonperiodic = true`). Forces are the product analytic path
(Hellmann–Feynman + Pulay + grid/core) written to `frame.span(kAtom_f*)`
[eV/Å]. The SCF may warm-start from the previous step's MO coefficients when
available. Closed-shell Γ RKS only.

The system is isolated H₂O at the CP2K getting-started geometry (same twin
family as [`gpw-forces-water`](gpw-rks.md#gpw-forces-water) and
`gpw-water-pbe-mt`). Absolute force quality vs **CP2K 2026.1** is pinned in
`gpw-forces-water` (max |ΔF| ≈ 3.4×10⁻⁴ Ha/Bohr at 400 Ry); this section gates
the MD binding and NVE conservation on the product method path, which
[`xl-bomd-water`](dynamics-xl-bomd.md#xl-bomd-water) builds on.

### System

| Quantity | Value |
| -------- | ----- |
| Species | H₂O (O + 2 H), isolated / MT |
| Masses | O 15.999 amu; H 1.008 amu |
| Cell | cubic \(L = 10\) Å; `PERIODIC NONE` / `nonperiodic = true` |
| XC / basis / PP | PBE / DZVP-MOLOPT-GTH / GTH-PBE-q6 (O), GTH-PBE-q1 (H) |
| CUTOFF | **200 Ry** (forces pin uses 400 Ry) |
| SCF | `tol = 1e-5`, `max_iter = 60`, `mix = 0.3` |
| \(\Delta t\) | 0.5 fs |
| Initial velocity | COM-preserving H stretch \(v_x(\mathrm{H1})=+0.0015\), \(v_x(\mathrm{H2})=-0.0015\) Å/fs |
| Steps | 1 warm-up + 4 production (2 chunks) |
| Threads | 8 |

| Atom | \(x\) | \(y\) | \(z\) (Å) |
| ---- | ----: | ----: | --------: |
| O | 5.0000 | 5.0000 | 5.0000 |
| H1 | 5.7586 | 5.0000 | 5.5043 |
| H2 | 4.2414 | 5.0000 | 5.5043 |

**Why a cheaper cutoff than the forces pin?** One ENERGY_FORCE at 400 Ry is
~40 s wall on login-class ATV (see the `gpw-forces-water` record), so multi-step
BOMD at that cost exceeds the ≤60 s regression gate. 200 Ry keeps the same
method family (PBE / MOLOPT / GTH / MT / analytic forces) inside the time
budget; absolute E/F parity at 400 Ry / 1e-7 stays the forces pin's job.

```cpp
frame.box = cpu::Box::cubic(10.0);
cpu::gpw::GpwDft dft{{O, H}, 200.0, cpu::gpw::XcKind::PBE};  // kinds, cutoff [Ry], XC
dft.max_iter = 60;  dft.tol = 1e-5;  dft.mix = 0.3;
dft.nonperiodic = true;

cpu::md::MD sim;
sim.set_integrator(cpu::md::VelocityVerlet{/*dt [fs]=*/0.5});
sim.on(md::Phase::force).add(std::move(dft));
sim.build();
sim.run(frame, 1);   // warm-up
sim.run(frame, 2);   // production chunk (×2)
```

No unit test in the tree covers the MD bind (≥2 MD steps, finite PE/KE at
cheap 200 Ry settings); the gates below are the only check.

### Gates

| Quantity | Atomiverse / pin | Tol | Notes |
| -------- | ---------------: | --- | ----- |
| Relative total-energy drift \(\delta\) | measured \(\approx 1.2\times 10^{-5}\) | **0.05** | envelope |
| PE after warm-up | finite; \(< -100\) eV | bound | PBE water is O(−10²) eV |
| KE | finite; \(\ge 0\); non-trivial (> 1e-12 eV) after warm-up | — | |
| Soft O–H ceiling | both \(r_{\mathrm{OH}} < 1.8\) Å | **1.8 Å** | no dissociation on the short trajectory |
| Coordinates | finite over trajectory | — | non-explosion |

Twin design for a future CP2K pin: short NVE MD Quickstep GPW on the same H₂O
cell (PBE / DZVP-MOLOPT-GTH / GTH / MT, input `h2o_bomd_mt.inp`), recording
version + measured drift envelope + fair wall.

### Performance

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | Caveat |
| ---- | ------------------------: | -----------------------: | ----: | ------ |
| NVE drift (AIMD) | — | ≈ **35** (login-class, OMP=4; 1+4 steps @ 200 Ry) | **perf_exempt** | no offline CP2K binary for a multi-step AIMD fair wall |

The single-point force wall vs CP2K is measured in `gpw-forces-water` (ratio
**3.56**, within 10×) — that is ENERGY_FORCE, not this multi-step twin.

### Limitations

- 200 Ry / `tol=1e-5` / 60 SCF iterations suffice for the NVE envelope inside
  ≤60 s, not production AIMD accuracy or absolute E/F vs CP2K at 400 Ry / 1e-7.
- 1 warm-up + 4 production steps.
- No UKS, k-points, hybrid or GAPW; cell stress is `gpw-stress-water`.

## pw-bomd-h — Plane-wave BOMD, H atom NVE {#pw-bomd-h}

### Method

BOMD with `cpu::pw::PW` mounted directly at `md::Phase::force` as a nested
Driver node (plane-wave SCF: Poisson → XC → \(V_{\mathrm{eff}}\) → Davidson →
density mix). The PW method keeps Ry/Ha on `last_result()` only. The
pseudopotential is the analytic in-tree **GTH-PBE-q1**
(`GthPseudopotential{"GTH-PBE-q1"}`, no external file); that entry has no
nonlocal projectors, so the Kleinman–Bylander term is zero. One valence
electron is a doublet, so the SCF is collinear spin-polarized
(`spin_polarized = true`, `multiplicity = 2`; the restricted force path rejects
an odd electron count). Forces come from the `PW::forces` path — local
Hellmann–Feynman + Ewald ionic contribution [eV/Å] — reusing the converged
density, one SCF per step.

A single H at the origin of a cubic cell has \(\mathbf{F}=\mathbf{0}\) by
symmetry, giving an exact force-residual pin; a hand-set \(v_x\) supplies
non-trivial KE so the NVE envelope is meaningful. Product force math is
covered by `h_force_analytic` / FD unit tests; this section pins MD binding +
NVE conservation on the symmetry-stable single-H system.

### System

| Quantity | Value |
| -------- | ----- |
| Species | 1 × H (\(Z=1\)) at the origin, mass 1.008 amu |
| Cell | cubic \(L = 6\) Å |
| Pseudopotential | GTH-PBE-q1 (analytic, local only) |
| \(E_{\mathrm{cut}}\) / k-mesh | 5 Ry / Γ-only (`k_mesh = 1`) |
| SCF | `tol = 1e-4`, `max_iter = 12`, `mix_beta = 0.5` |
| \(\Delta t\) | 0.5 fs |
| Initial velocity | \(v_x = 0.002\) Å/fs (hand-set; `InitVelocity` COM-zeros \(N=1\)) |
| Steps | 1 warm-up + 10 production (5 chunks) |

```cpp
frame.box = cpu::Box::cubic(6.0);
cpu::pw::PW pw{cpu::pw::GthPseudopotential{"GTH-PBE-q1"}, 5.0, 1};  // PP, Ecut [Ry], k_mesh
pw.tol = 1e-4;  pw.max_iter = 12;  pw.mix_beta = 0.5;
pw.spin_polarized = true;  // lone H: N↑ = 1, N↓ = 0
pw.multiplicity = 2;

cpu::md::MD sim;
sim.set_integrator(cpu::md::VelocityVerlet{/*dt [fs]=*/0.5});
sim.on(md::Phase::force).add(std::move(pw));
sim.build();
sim.run(frame, 1);   // warm-up
sim.run(frame, 2);   // production chunk (×5)
```

No unit test in the tree covers the MD bind (≥2 MD steps, finite PE/KE,
residual force bound); the gates below are the only check.

### Gates

| Quantity | Atomiverse / pin | Tol | Notes |
| -------- | ---------------: | --- | ----- |
| Relative total-energy drift \(\delta\) | \(\le 0.05\) | **0.05** | envelope |
| PE, KE | finite; KE \(\ge 0\) | — | after warm-up and each production chunk |
| Warm-up \(\lvert\mathbf{F}\rvert\) | \(< 0.5\) eV/Å | **0.5 eV/Å** | cubic high-symmetry residual; same bound as `h_force_analytic` |
| Coordinates | finite over trajectory | — | non-explosion |

Twin design for a future CP2K pin: short NVE AIMD for single H (or H₂ with a
real PP) in a comparable cell, recording version + measured drift envelope.
Speed: `perf_exempt` (no offline CP2K/QE binary for a fair wall twin).

### Limitations

- Local-only PP: the section pins force/MD binding, not production hydrogen
  chemistry.
- Single ion with \(\mathbf{F}\approx\mathbf{0}\): free-particle NVE with
  near-flat PE; does not stress multi-atom bonding, cell stress, or long-range
  force cancellation on a molecule.
- 5 Ry, 12 SCF iterations, `tol=1e-4`: enough for the envelope, not production
  AIMD accuracy. Si/UPF BOMD, GGA / metallic smearing PW BOMD and thermostats
  are not covered.

## gpw-bomd-kpoints — Multi-k GPW BOMD, periodic H₂ NVE (DZV-GTH-PADE) {#gpw-bomd-kpoints}

*No section of the case runs this variant.*

### Method

Plain BOMD on a Monkhorst–Pack k-mesh: the nuclei follow NVE velocity Verlet,
and at every step the multi-k closed-shell RKS GPW problem is re-converged to
full SCF tolerance, with analytic k-summed forces driving the integrator. No
auxiliary electronic propagator is involved — contrast
[`xl-bomd-kpoints`](dynamics-xl-bomd.md#xl-bomd-kpoints) (extended-Lagrangian
density), [`cpmd-kpoints`](dynamics-car-parrinello.md#cpmd-kpoints) (fictitious
orbital dynamics) and
[`second-gen-cpmd-kpoints`](dynamics-second-gen-cpmd.md#second-gen-cpmd-kpoints).
After a converged step the complex densities \(\{P_k\}\) seed the next SCF
(the k-path analogue of the Γ `guess_C`/`guess_P` block; CP2K's
`EXTRAPOLATION USE_PREV_P`).

At each MD step \(n\) the electronic problem is solved self-consistently on the
sampled k set \(\{k\}\):

$$ F_k[\,\{P_{k'}\}\,]\,C_k = S_k C_k \varepsilon_k, \qquad
   P = \sum_k w_k\, \mathrm{Re}\, P_k , $$

then the nuclei advance on the k-summed analytic force
\(F_I = -\sum_k w_k\, \partial E_k / \partial R_I\). Spin treatment is
restricted (RKS); the UKS multi-k warm start is covered by the unit gate (the
solver rejects a partial one-spin guess — both \(P_k^\sigma\) are seeded or
none). The mesh is MP 1×1×2 **unreduced** (`k_time_reversal = false`, 2 k
points, \(w = 1/2\)), matching CP2K `FULL_GRID .TRUE.`. XC is PADE (LDA); no
dispersion, no smearing, no thermostat. Not included: time-reversal-reduced
meshes, XL/CPMD propagators, k-point stress.

```cpp
cpu::gpw::GpwDft dft{kinds, 300.0, cpu::gpw::XcKind::PADE};
dft.max_iter        = 200;
dft.mix             = 0.3;
dft.k_mesh[2]       = 2;        // 1×1×2
dft.k_time_reversal = false;    // 2 k, w = 1/2 (CP2K FULL_GRID)
dft.tol             = 1e-7;
cpu::gpw::GPWEvaluator gpw(std::move(dft), /*compute_forces=*/true);

cpu::single_point(frame, gpw);  // VV bootstrap: step-0 forces

cpu::md::MD md;
md.set_integrator(cpu::md::VelocityVerlet{0.5});   // fs
md.set_evaluator_ref(gpw);
md.build();
md.run(frame, 4);               // 4 NVE steps
```

Sources: `src/cpu/gpw/` (MD path + \(P_k\) warm start). A same-geometry
relaunch converges in strictly fewer iterations to the same fixed point (RKS
and UKS multi-k; recorded, no unit test in the tree gates it).

### Validation (vs CP2K)

Values recorded when this page was written (no standing regression). Twin:
periodic H₂ (bond 1.0 Bohr, cubic 12 Bohr cell), DZV-GTH-PADE / GTH-PADE-q1 /
PADE, CUTOFF 300 Ry NGRIDS 1, MP 1×1×2 unreduced, NVE dt 0.5 fs × 4 steps, ATV
`tol = 1e-7` ⇔ CP2K `EPS_SCF 1e-7`. **CP2K 2026.1** (`cp2k.psmp`) Quickstep
AIMD (`RUN_TYPE MD` + `&KPOINTS`), single rank, OMP=8.

| Quantity | Atomiverse | CP2K pin | Tol | Notes |
| -------- | ---------: | -------: | --- | ----- |
| Step-0 E_pot [Ha] | −1.074456793 | −1.074050184 | 1e-3 | Δ = 4.07e-4 — Γ-point periodic-GPW model band (see below) |
| Step-0 \(F_x\) atom 0 [Ha/Bohr] | −0.339676 | −0.345535381 | 8e-3 | Δ = 5.86e-3; analytic ≡ FD to 6e-6 ([gpw-kpoint-forces](gpw-rks.md#gpw-kpoint-forces)) |
| E_pot step 1 [Ha] | trace | −1.097682104 | 2e-3 | worst Δ over steps 1–4 = 2.33e-4 |
| E_pot step 2 [Ha] | trace | −1.125544577 | 2e-3 | |
| E_pot step 3 [Ha] | trace | −1.130531455 | 2e-3 | |
| E_pot step 4 [Ha] | trace | −1.121507812 | 2e-3 | |
| NVE drift (rel.) | 4.25e-3 | 4.4e-3 | ≤ 0.05 | dt = 0.5 fs is deliberately aggressive for this compressed bond |
| Multi-k payload | `n_kpts == 2`, converged every step | — | exact | full \(P_k\), warm start never worse (8 ≤ 8 iters; CP2K 14 cold → 8–9 warm) |

The Γ control (600 Ry) shows identical energy/force deltas (4.07e-4 /
5.85e-3), so the residual is the Γ-point periodic-GPW model band on this
compressed periodic H₂, not a k-point defect — the multi-k machinery
contributes < 1e-5 on top.

### Performance

Fair wall, same host, single rank, `OMP_NUM_THREADS=8`; gate
\(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \le 10\), no exemption.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | -----------------------: | ----: | ---- | ------ |
| H₂ NVE, 1 single point + 4 MD steps, CUTOFF 300, 2 k | 33.4 | 2.14 | **0.064** | ✓ | ATV DIIS vs CP2K direct-P mixing; same SCF tol |

### Limitations

- Converged-limit energy/force parity vs CP2K sits on the Γ-point
  periodic-GPW model band (4e-4 Ha / 6e-3 Ha/Bohr on this system).
- Warm start is \(P_k\) reuse only — no multi-step density extrapolation (no
  ASPC on the multi-k path).
- The twin uses an unreduced mesh; time-reversal-reduced BOMD is not checked
  against CP2K.

### References

- Monkhorst & Pack, *Phys. Rev. B* **13**, 5188 (1976)
- Sibling: [gpw-kpoint-forces](gpw-rks.md#gpw-kpoint-forces)
