# Second-generation Car–Parrinello MD

Second-generation Car–Parrinello MD (SGCP; Kühne, Krack, Mohamed & Parrinello,
*Phys. Rev. Lett.* **98**, 066401 (2007), doi:10.1103/PhysRevLett.98.066401)
keeps the nuclei on ordinary NVE velocity Verlet while the electrons use a
**Kolafa ASPC density predictor** plus a **partial SCF corrector** each MD
step. Production steps do not run a full Born–Oppenheimer SCF: once the ASPC
history is warm, the SCF is capped at the wrapper's `corrector_max_iter` (≪ the
method's `max_iter`). There is **no** fictitious electronic mass and **no**
electronic kinetic energy \(T_e\).

The public type is the wrapper
`cpu::electronic::SecondGenCp{cpu::gpw::GPWEvaluator{GpwDft{…},
/*compute_forces=*/true}, aspc_k, corrector_max_iter}`
(`cpu/electronic/second_gen_cp.h`) bound to `cpu::md::MD` + NVE
`cpu::md::VelocityVerlet` with `set_evaluator_ref` (borrow, so `last_result()`
stays readable; `set_evaluator(std::move(eval))` hands ownership over). The
ASPC order \(K\) and corrector budget are constructor arguments (defaults
\(K=3\), budget 5; the constructor throws for \(K < 0\) or a budget \(< 1\)),
the wrapper sets `compute_forces = true` itself, and the SGCP mode bits are
private on `GPWEvaluator`. The corrector is the method's diagonalization/DIIS
SCF, or Orbital Transformation (OT) with `GpwDft::use_ot = true`. The sections
cover closed-shell Γ RKS (H₂O) and Γ-UKS (O₂ triplet) with both correctors, and
multi-k RKS (periodic H₂) with both correctors; all CPU, GPW only.

Case: `ctest -R '^dynamics/second-gen-cpmd$'` — sources and oracle record in [`regressions/dynamics/second-gen-cpmd/`](../../../regressions/dynamics/second-gen-cpmd/). One section of the case per `##` heading below: `atv_regressions --case dynamics/second-gen-cpmd --section <name>`.

## Theory and common setup

### Method map

| Method | Electronics each MD step | Public type |
| ------ | ------------------------ | ----------- |
| BOMD ([gpw-bomd-water](dynamics-bomd.md#gpw-bomd-water)) | full SCF to `tol` / `max_iter` | `GpwDft` mounted at the force phase |
| XL-BOMD ([xl-bomd-water](dynamics-xl-bomd.md#xl-bomd-water)) | full SCF; Niklasson auxiliary density as guess | `cpu::electronic::XlBomdForce{gpw, k}` |
| Classical CPMD ([cpmd-water](dynamics-car-parrinello.md#cpmd-water)) | fixed-density KS + electronic VV/RATTLE on \(C\); \(T_e\) | `cpu::electronic::CarParrinello{gpw, mu_au}` |
| **SGCP (this page)** | ASPC predict + **budgeted** SCF corrector | `cpu::electronic::SecondGenCp{gpw, aspc_k, corrector_max_iter}` |

| Section | Spin | Sampling | Corrector |
| ------- | ---- | -------- | --------- |
| `second-gen-cpmd-water` | RKS | Γ | diag/DIIS partial SCF |
| `second-gen-cpmd-uks` | UKS | Γ | diag/DIIS partial SCF (dual ASPC) |
| `second-gen-cpmd-ot` | RKS | Γ | **OT** partial SCF |
| `second-gen-cpmd-uks-ot` | UKS | Γ | **OT** partial SCF |
| `second-gen-cpmd-kpoints` | RKS | multi-k | diag partial SCF (ASPC on \(P_k\)) |
| `second-gen-cpmd-ot-kpoints` | RKS | multi-k | **OT** partial SCF |
| `second-gen-cpmd-ot-uks-kpoints` | UKS | multi-k | **OT** (no standing section) |

Static siblings: [ot-minimizer-water](gpw-orbital-transformation.md#ot-minimizer-water)
(full OT SCF), [gpw-uks-o2-gamma](gpw-uks.md#gpw-uks-o2-gamma) (O₂ energy /
⟨S²⟩), [cpmd-uks](dynamics-car-parrinello.md#cpmd-uks) (classical CP, not
SGCP), [xl-bomd-uks](dynamics-xl-bomd.md#xl-bomd-uks) (Niklasson XL + full
SCF).

### ASPC predictor and corrector

Kolafa ASPC (*J. Comput. Chem.* **25**, 335 (2004), doi:10.1002/jcc.10385)
of order \(K\) stores \(K+1\) previous densities and builds a Lagrange
extrapolation for the next ionic step,

\[
P_{n+1}^{\mathrm{pred}} = \sum_{k=0}^{K} B_k\, D_{n-k}.
\]

Until the history is full, `predict()` returns false and the method runs a
full SCF (seed phase; \(K+1\) seed steps). After warm-up, each production force
step:

1. ASPC → initial density guess
2. SCF (or OT) with iteration cap `corrector_max_iter`
3. Kolafa corrector blend of SCF and predicted \(P\)
4. Record the density into the ASPC history
5. Analytic forces at the post-corrector density → `frame.span(kAtom_f*)` [eV/Å]

The primitive is public as `cpu::electronic::ASPCPredictor` (real) and
`ASPCPredictorCplx` (complex Bloch densities), `aspc_predictor.h` — the same
primitive used for XL / BOMD warm starts. RKS uses one predictor, UKS two
(α and β), multi-k one per sampled \(k\) (and spin). The CP2K method-family
twin is `&QS EXTRAPOLATION ASPC` + limited `MAX_SCF` (CP2K exercise
`exercises:common:sgcp`).

### Conserved quantity and gates

The conserved quantity is the ordinary nuclear NVE total (public units eV on
the Frame),

\[
E = E_{\mathrm{pot}} + E_{\mathrm{kin}} = T_{\mathrm{n}} + E_{\mathrm{KS}},
\]

not the classical CP \(E_{\mathrm{cons}} = T_n + T_e + E_{\mathrm{KS}}\). Every
section gates

\[
\delta = \max_t \frac{\lvert E(t)-E_0\rvert}{\max\!\big(1,\,\lvert E_0\rvert\big)}
  \le 0.10,
\]

with \(E_0\) after the first ASPC-warm production step. The soft 0.10 window
(vs 0.05 on full-SCF BOMD / XL pins) absorbs partial-SCF heat on a short
trajectory; the gate still fails a broken ASPC guess or a path that silently
falls back to full BOMD. The method-identity gate is production
`last_result().iters ≤ corrector_max_iter` once ASPC is warm. All sections also
require finite PE, finite KE \(\ge 0\) and non-trivial after seed, finite
coordinates, and a soft bond window. Absolute \(E_{\mathrm{tot}}\) is not
pinned (envelope + identity only).

Common protocol: ASPC \(K = 1\) (2 seed steps with full SCF), corrector budget
5, \(\Delta t = 0.5\) fs (BOMD-class; classical CP uses ~0.1 fs), then 1
baseline warm production step + 2 drift steps:

```cpp
cpu::electronic::SecondGenCp eval{std::move(gpw), /*aspc_k=*/1,
                                  /*corrector_max_iter=*/5};
cpu::md::MD sim;
sim.set_integrator(cpu::md::VelocityVerlet{/*dt [fs]=*/0.5});
sim.set_evaluator_ref(eval);
sim.build();
sim.run(frame, 2);   // seed: full SCF fills ASPC history (K+1 = 2)
sim.run(frame, 1);   // first ASPC-warm production step → baseline E0
// assert eval.last_result().iters <= 5
sim.run(frame, 2);   // production drift window
// E = frame.meta.get(kEnergy) + frame.meta.get(kKinetic)  [eV]
```

Build: modules `GPW` (with HF + PW) and `MD`, `ATV_BUILD_REGRESSIONS=ON`, CPU
toolchain (`buildenv-gcc/2026.03-mpich`); sections run at 8 threads, each
within the ≤ 60 s e2e budget.

### Exclusions and guards

- One AIMD wrapper per evaluator: `build()` throws if SGCP is combined with a
  `CarParrinello` or `XlBomdForce` wrapper (`cpmd` / `xl_bomd` mutual
  exclusion).
- `compute_forces = true` is required (the wrapper sets it).
- Propagating modes cannot score reversible MC trials.
- Not covered: electronic / nuclear thermostat, NVT/NPT, cell dynamics,
  metallic smearing or DFT+U under the SGCP corrector, hybrid / meta-GGA XC,
  MPI multi-rank AIMD scaling, CUDA.
- Fixed cell NVE: GPW rejects box changes after `build()`.

### Oracle and speed policy

Pins are offline; ctest never runs CP2K. On the pin host no `cp2k` binary /
CP2K Lmod module was available (Arrhenius login-class inventory), so the Γ and
multi-k diagonalization sections are gated on the literature SGCP method
identity (ASPC + partial SCF) plus the NVE envelope and trajectory diagnostics;
their CP2K inputs are unexecuted sketches (`EXTRAPOLATION ASPC`, limited
`MAX_SCF`). When a binary is available they are re-run offline and the record
gains version + `.ener` drift + fair wall, with
\(T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \le 10\) required (or blocked-speed
documented after one optimize pass). Multi-k OT cannot have a CP2K number at
all (CP2K's OT is Γ-only and its input validation refuses `&OT` with
`&KPOINTS`). Speed is **`perf_exempt`** for every section: correctness is the
NVE drift + corrector-budget identity, not a wall ratio. Single-point force
wall vs CP2K is measured elsewhere (`gpw-forces-water`, vs **CP2K 2026.1**,
ENERGY_FORCE ratio **3.56**, within 10×).

### References

- T. D. Kühne, M. Krack, F. R. Mohamed, M. Parrinello, *Phys. Rev. Lett.*
  **98**, 066401 (2007), doi:10.1103/PhysRevLett.98.066401 — second-generation CP
- J. Kolafa, *J. Comput. Chem.* **25**, 335 (2004), doi:10.1002/jcc.10385 — ASPC
- J. VandeVondele, J. Hutter, *J. Chem. Phys.* **118**, 4365 (2003),
  doi:10.1063/1.1543154 — OT
- T. D. Kühne et al., *J. Chem. Phys.* **152**, 194103 (2020) §VII C,
  doi:10.1063/5.0007045 — CP2K / OT formulation followed in-tree
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW
- J. VandeVondele et al., *Comput. Phys. Commun.* **167**, 103 (2005) — Quickstep GPW
- S. Goedecker, M. Teter, J. Hutter, *Phys. Rev. B* **54**, 1703 (1996) — PADE
- G. J. Martyna, M. E. Tuckerman, *J. Chem. Phys.* **110**, 2810 (1999) — MT
- Car & Parrinello, *PRL* **55**, 2471 (1985) — classical CPMD (contrast)
- Frenkel & Smit §4.3.4; Marx & Hutter, *Ab Initio Molecular Dynamics* — NVE
  envelope
- Roadmap: [`docs/roadmap/molecular-dynamics.md`](../../roadmap/molecular-dynamics.md) §AIMD (§7)

## second-gen-cpmd-water — SGCP, H₂O NVE (diagonalization corrector) {#second-gen-cpmd-water}

Closed-shell RKS Γ-point GPW with the diagonalization/DIIS corrector.

| Quantity | Value |
| -------- | ----- |
| Species | H₂O (O + 2 H), isolated / MT, CP2K getting-started geometry (twin family of `gpw-bomd-water` / `xl-bomd-water` / `cpmd-water`) |
| Masses | O 15.999 amu; H 1.008 amu |
| Cell | cubic \(L = 10\) Å; `nonperiodic = true` (Martyna–Tuckerman) |
| XC / basis / PP | PBE / DZVP-MOLOPT-GTH / GTH-PBE-q6 (O), GTH-PBE-q1 (H) |
| CUTOFF | 200 Ry |
| SCF envelope | `tol = 1e-5`, `max_iter = 60` (seed), `mix = 0.3` |
| ASPC \(K\) / corrector | **1** (needs \(K+1=2\) seed densities) / `corrector_max_iter = 5` |
| \(\Delta t\) | 0.5 fs |
| Initial velocity | COM-preserving H stretch \(v_x(\mathrm{H1})=+0.0015\), \(v_x(\mathrm{H2})=-0.0015\) Å/fs |
| Steps | 2 seed (full SCF) + 1 baseline production + 2 drift production |

| Atom | \(x\) | \(y\) | \(z\) (Å) |
| ---- | ----: | ----: | --------: |
| O | 5.0000 | 5.0000 | 5.0000 |
| H1 | 5.7586 | 5.0000 | 5.5043 |
| H2 | 4.2414 | 5.0000 | 5.5043 |

```cpp
frame.box = cpu::Box::cubic(10.0);
cpu::gpw::GpwDft dft{{O, H}, 200.0, cpu::gpw::XcKind::PBE};
dft.max_iter = 60;  dft.tol = 1e-5;  dft.mix = 0.3;
dft.nonperiodic = true;
cpu::gpw::GPWEvaluator gpw(std::move(dft), /*compute_forces=*/true);
cpu::electronic::SecondGenCp eval{std::move(gpw), /*aspc_k=*/1,
                                  /*corrector_max_iter=*/5};
```

### Gates

| Quantity | Atomiverse / pin | Tol | Notes |
| -------- | ---------------: | --- | ----- |
| Relative total-energy drift \(\delta\) | \(\le 0.10\) (envelope) | **0.10** | partial-SCF heat |
| Production SCF iters | \(\le 5\) once ASPC warm | **`corrector_max_iter`** | method identity: not full BOMD |
| PE after seed | finite; \(< -100\) eV | bound | PBE water O(−10²) eV |
| KE | finite; \(\ge 0\); non-trivial after seed | — | COM-preserving H stretch |
| Soft O–H ceiling | both \(r_{\mathrm{OH}} < 1.8\) Å | **1.8 Å** | no dissociation |
| Coordinates | finite over trajectory | — | non-explosion |

Unit surface: `build()` rejects `cpmd` / `xl_bomd` combinations, forces-off,
and bad \(K\) or budget; accepts default members. Offline CP2K sketch:
`second_gen_cpmd_water_nve.inp` (Quickstep GPW, `EXTRAPOLATION ASPC`, limited
`MAX_SCF`, PBE / MOLOPT / GTH / MT).

### Limitations

- 2 seed + 1 baseline + 2 production steps only.
- \(K=1\) in the section for a fast seed; the wrapper default is \(K=3\) /
  corrector 5 — callers may raise \(K\) for production.

## second-gen-cpmd-uks — SGCP on Γ-UKS, O₂ triplet NVE {#second-gen-cpmd-uks}

Open-shell Γ-UKS GPW with **dual Kolafa ASPC** predictors on independent spin
densities \(P^\alpha, P^\beta\) (two `ASPCPredictor` instances), then a partial
SCF corrector with dual DIIS diagonalization (not OT): seed \(K+1\) full-SCF
steps, predict \(P^\sigma_{n+1}\) per spin, run at most `corrector_max_iter`
SCF iterations, record \(P^\sigma\), evaluate forces at that state. UKS alone
is accepted by `build()`. Triplet O₂ with GTH-q6 → 12 valence e⁻;
\(n_\alpha=7\), \(n_\beta=5\); spin-resolved PADE LSD on the GPW grid.
\(\langle S^2\rangle\) is the Löwdin single-determinant value (ideal \(S=1\):
\(S(S+1)=2\)), read from `eval.method().last_s2()`.

```cpp
// O₂ triplet, R = 1.2075 Å, centered in a 12 Å cubic MT cell
frame.span(kAtom_vz)[0] = -0.0008; // Å/fs antisymmetric stretch
frame.span(kAtom_vz)[1] =  0.0008;
frame.box = cpu::Box::cubic(12.0);
cpu::gpw::GpwDft dft{{O}, 200.0, cpu::gpw::XcKind::PADE};
dft.max_iter = 80;  dft.tol = 1e-5;  dft.mix = 0.4;
dft.nonperiodic = true;
dft.n_alpha = 7;  dft.n_beta = 5;
cpu::gpw::GPWEvaluator gpw(std::move(dft), /*compute_forces=*/true);
cpu::electronic::SecondGenCp eval{std::move(gpw), /*aspc_k=*/1,
                                  /*corrector_max_iter=*/5};
```

System (hard-coded): O₂ (mass 15.999 amu), \(R=1.2075\) Å, \(L=12\) Å cubic,
MT / `nonperiodic`, PADE LSD, DZVP-MOLOPT-GTH / GTH-PBE-q6, 200 Ry, `tol=1e-5`,
seed `max_iter=80`, ASPC \(K=1\), corrector 5, `dt=0.5` fs, 2 seed + 1 baseline
+ 2 drift steps. Dual ASPC + partial-SCF launch path:
`src/cpu/gpw/gpw_evaluator.cpp`.

### Gates

| Quantity | Atomiverse / pin | Oracle | Tol | Notes |
| -------- | ---------------: | ------ | --- | ----- |
| Relative NVE drift \(\delta\) | asserted | literature SGCP envelope | ≤ 0.10 | after first warm production step |
| Production SCF iters | ≤ 5 | CP2K `MAX_SCF` sketch | identity | once ASPC warm — not full BOMD |
| \(\langle S^2\rangle\) | \([1.5, 2.5]\) | Löwdin triplet window | hard | after seed and production |
| PE after seed | \(< -100\) eV | bound | hard | finite trajectory |
| KE after seed | \(> 0\), finite | identity | hard | antisymmetric stretch |
| O–O distance | \(0.9 < r < 1.6\) Å | structural | soft | after production |

Offline CP2K sketch: `UKS`, `MULTIPLICITY 3`, `EXTRAPOLATION ASPC`,
`EXTRAPOLATION_ORDER 1`, `MAX_SCF 5`, PADE, MT. Static UKS energy / force pins
for the same molecule: `gpw-uks-o2-gamma`, `gpw-uks-forces`.

## second-gen-cpmd-ot — SGCP with OT corrector, H₂O NVE {#second-gen-cpmd-ot}

The closed-shell Γ-point path where the corrector is **Orbital Transformation**
(`use_ot = true`) rather than diagonalization/DIIS: ASPC predicts \(P\), the OT
manifold is warm-started from that density (natural-orbital seed, the Γ
`guess_P` OT warm start), OT runs for at most `corrector_max_iter` steps, and
analytic nuclear forces are taken at the post-corrector state. Not classical
CPMD (no \(\mu\,\ddot C\), no \(T_e\), no electronic RATTLE); not DIIS SGCP
(`second-gen-cpmd-water`, `use_ot=false`).

### OT corrector (closed-shell)

OT (VandeVondele & Hutter 2003) optimizes occupied coefficients on
\(C^\top S C = I\) without a dense generalized eigensolve each outer step.
Parameterization for reference \(C_0\) and \(S\)-tangent \(x\):

\[
C(x) = C_0\,\cos U + x\,U^{-1}\sin U,\qquad
U = (x^\top S x)^{1/2}.
\]

Closed-shell band-energy gradient consistent with \(P = 2\,C C^\top\):

\[
g = (I - S C_0 C_0^\top)\,(4 H C_0).
\]

Forces use the OT-exported orbitals / \(H_{\mathrm{KS}}\), not a `max_iter=0`
fixed-density re-eval (OT forbids that route — the solver exports
\(H_{\mathrm{KS}}\) / \(S\) when needed). Seed steps run full-budget OT SCF.
Paths: `ot.h` (OT parameterization, internal); `gpw_evaluator.h` (public
`use_ot` on the method; private SGCP mode bits); ASPC predict/correct and OT
force path in `src/cpu/gpw/gpw_evaluator.cpp`. Unit: RKS OT accepted; mutual
exclusion with the CP / XL wrappers; `CarParrinello` + `use_ot` is refused.

### Setup

| Item | Value |
| ---- | ----- |
| System | Isolated H₂O, charge 0, closed shell |
| Geometry [Å] | O (5, 5, 5); H (5.7586, 5, 5.5043); H (4.2414, 5, 5.5043) |
| Cell | 10 Å cubic, `PERIODIC NONE`, Martyna–Tuckerman |
| Basis / PP / XC | DZVP-MOLOPT-GTH; GTH-PBE-q6 (O), GTH-PBE-q1 (H); PBE |
| Cutoff / SCF | **200 Ry** / `tol = 1e-5`, `max_iter = 60`, `mix = 0.3`, `use_ot = true` |
| dt / ASPC \(K\) | **0.5 fs** / **1** |
| Corrector | OT, `corrector_max_iter = 5` |
| Seed / production | 2 full-OT steps (\(K+1\)) / 1 baseline + 2 drift steps |

### Gates

| Quantity | Atomiverse / pin | CP2K twin | Tol / gate | Notes |
| -------- | ---------------: | --------: | ---------- | ----- |
| Relative \(\lvert E-E_0\rvert/\mathrm{scale}\) | ≤ **0.10** | envelope (re-pin) | ≤ 0.10 | soft vs full-BO 0.05 |
| Production OT `iters` | ≤ **5** once ASPC warm | `MAX_SCF 5` sketch | ≤ `corrector_max_iter` | method identity |
| O–H distances | < **1.8 Å** | — | < 1.8 | no bond explode |
| PE after seed | < **−100 eV** | — | bound | Frame eV |
| Seed KE | > 0 | — | non-trivial | stretch-mode seed |

Offline CP2K sketch (not ctest): `EXTRAPOLATION ASPC`, `EXTRAPOLATION_ORDER 1`,
`&OT`, tight `MAX_SCF`, output `/tmp/cp2k_oracle_h2o_sgcp_ot_nve.out`. The fair
OT-only single-point wall is on
[ot-minimizer-water](gpw-orbital-transformation.md#ot-minimizer-water)
(\(r\sim 9\)); the MD section is not ratioed against a static SCF twin.

### Limitations

- Short trajectory + soft drift tol — not a long-run microcanonical audit or a
  CP2K energy-class match.
- Dense OT preconditioner cost (see `ot-minimizer-water`); not sparse OT.

## second-gen-cpmd-uks-ot — SGCP + OT corrector on Γ-UKS, O₂ triplet NVE {#second-gen-cpmd-uks-ot}

Open-shell Γ-UKS with OT as the partial-SCF corrector. Each warm force step:

1. **Dual Kolafa ASPC** predictors on independent spin densities
   \(P^\alpha\), \(P^\beta\);
2. purification of the predicted densities to **spin-resolved natural
   orbitals** that seed the OT manifolds (the Γ-UKS
   `guess_P`/`guess_P_beta` OT warm start);
3. a **capped OT** corrector (`corrector_max_iter` = 5, not full BOMD);
4. analytic nuclear forces at the post-corrector density.

Production steps spend \(O(\texttt{max\_iter})\) OT iterations with
\(\texttt{max\_iter} \ll\) the full BOMD budget — the SGCP signature. The
histories for \(P^\alpha\) and \(P^\beta\) are independent; a broken spin guess
(α-only, swapped channels, ignored density) fails the production-iter, ⟨S²⟩ or
drift gates immediately. OT runs on the same gapped fixed-point family as full
OT when the budget is raised; the product identity is the **cap**, not milli-Ha
energy parity vs full BO.

### Setup

| Item | Value |
| ---- | ----- |
| System | Isolated O₂, charge 0, UKS triplet |
| Geometry [Å] | \(R = 1.2075\); O at \(z = L/2 \pm R/2\), \(x=y=L/2\), \(L=12\) |
| Masses | 15.999 amu each |
| Cell | 12 Å cubic, `PERIODIC NONE`, Martyna–Tuckerman |
| Spin | \(n_\alpha=7\), \(n_\beta=5\) (GTH-q6, 12 valence e⁻) |
| Basis / PP / XC | DZVP-MOLOPT-GTH; GTH-PBE-q6; PADE (LSD) |
| Cutoff (pin) | **200 Ry** |
| SCF seed | `max_iter=80`, `tol=1e-5`, `mix = 0.4`, **OT** |
| SGCP | ASPC \(K=1\); `corrector_max_iter=5` once warm |
| \(\Delta t\) / initial velocity | 0.5 fs / antisymmetric stretch \(v_z = \pm 0.0008\) Å/fs |
| Steps | 2 seed (warm ASPC) + 1 baseline production + 2 drift steps |
| CP2K twin sketch | `EXTRAPOLATION ASPC`, `EXTRAPOLATION_ORDER 1`, `&OT` + limited `MAX_SCF`, `UKS` / `MULTIPLICITY 3` |

```cpp
dft.n_alpha = 7; // multiplicity 3, 12 valence e⁻
dft.n_beta = 5;
dft.use_ot = true;                     // OT partial-SCF corrector
cpu::gpw::GPWEvaluator gpw(std::move(dft), /*compute_forces=*/true);
cpu::electronic::SecondGenCp eval{std::move(gpw), /*aspc_k=*/1,
                                  /*corrector_max_iter=*/5};
// … seed 2, then production 1 + 2
const double s2 = eval.method().last_s2();
const int iters = eval.last_result().iters;
```

### Gates

| Quantity | Atomiverse / pin | CP2K | Tol | Notes |
| -------- | ---------------: | ---: | --- | ----- |
| Relative NVE drift \(\delta\) | ≤ **0.10** | (sketch; unexecuted) | **0.10** | \(E_0\) after first warm production step |
| Production OT iters | ≤ **5** | limited `MAX_SCF` | = budget | identity: partial OT, not full BOMD |
| PE after seed [eV] | \(< -100\) | — | bound | finite + physical |
| KE after seed [eV] | \(> 0\), finite | — | — | stretch velocity non-trivial |
| ⟨S²⟩ after seed / prod | \([1.5, 2.5]\) | LSD triplet class | window | Löwdin single-det |
| \(r_{\mathrm{OO}}\) after prod [Å] | \((0.9, 1.6)\) | — | soft | no explode/collapse |

Offline CP2K sketch (not ctest): `module load ... CP2K/2026.1`, output
`o2_uks_sgcp_ot_nve.out`, recording `ENERGY|` / `.ener` drift and the fair wall
vs this section. Static multi-k / smearing OT compositions are on
[ot-uks-kpoints](gpw-orbital-transformation.md#ot-uks-kpoints),
[ot-uks-smearing](gpw-orbital-transformation.md#ot-uks-smearing) and
[ot-uks-dftu-smearing](gpw-orbital-transformation.md#ot-uks-dftu-smearing).

## second-gen-cpmd-kpoints — Multi-k SGCP, H₂ NVE (ASPC on \(P_k\)) {#second-gen-cpmd-kpoints}

Multi-k closed-shell RKS GPW over a `GpwDft` with a non-Γ `k_mesh`: per sampled
\(k\), one `cpu::electronic::ASPCPredictorCplx` extrapolates the complex Bloch
density; a partial diagonalization SCF refines it; analytic multi-k forces are
taken at the post-corrector density (multi-k force math:
[gpw-kpoint-forces](gpw-rks.md#gpw-kpoint-forces); density-guess twin with
Niklasson XL instead of ASPC:
[xl-bomd-uks-kpoints](dynamics-xl-bomd.md#xl-bomd-uks-kpoints)).

### ASPC on complex Bloch densities

Real Lagrange coefficients \(\omega_m\) act **element-wise** on Hermitian
\(P_k \in \mathbb{C}^{n_{\mathrm{ao}}\times n_{\mathrm{ao}}}\):

\[
P_k^{\mathrm{pred}}
  = \sum_{m=0}^{K} \omega_m\, P_k^{(n-m)},
\]

requiring \(K+1\) history densities (\(K=1\) → 2 seed SCF steps). After a
partial SCF at the new geometry yields \(P_k^{\mathrm{scf}}\), the Kolafa
corrector blend

\[
P_k^{\mathrm{corr}}
  = \frac{K+2}{2K+3}\, P_k^{\mathrm{scf}}
  + \frac{K+1}{2K+3}\, P_k^{\mathrm{pred}}
\]

is recorded for the next step (`ASPCPredictor` / `ASPCPredictorCplx::correct`):

```cpp
cpu::electronic::ASPCPredictorCplx aspc(/*order K=*/1);
// after cold multi-k SCF: aspc.record(P_k);
// when history_size() >= K+1:
//   aspc.predict(P_pred);  // partial SCF → P_scf
//   aspc.correct(P_scf, P_pred, P_corr);
//   aspc.record(P_corr);
```

### System

| Quantity | Value |
| -------- | ----- |
| Species | H₂ (2 × H), periodic, masses 1.008 amu |
| Geometry | \(R = 1.0\) Bohr compressed bond along \(x\), cell-centred |
| Cell | cubic \(L = 12\) Bohr ≈ 6.350 Å (`bohr2ang = 0.52917720859`); periodic XYZ |
| Spin / XC | closed-shell RKS, \(n_{\mathrm{occ}}=1\); PADE (LDA) |
| Basis / PP | DZV-GTH-PADE / GTH-PADE-q1 |
| CUTOFF | 100 Ry (envelope run) |
| k-mesh | `k_mesh = {1,1,2}`, `k_time_reversal=false` → \(n_k=2\) |
| ASPC \(K\) / corrector | 1 / `corrector_max_iter = 5` after ASPC warm |
| Cold SCF | `tol = 1e-5`, `max_iter = 60`, `mix = 0.3`, DIIS diagonalization |
| \(\Delta t\) / \(v_0\) | 0.5 fs / antisymmetric stretch \(v_x = -0.0010, +0.0010\) Å/fs (\(\pm 0.001\)) |
| Steps | 2 seed + 1 baseline warm ASPC + 2 production |

```cpp
cpu::gpw::GpwDft dft{{H}, 100.0, cpu::gpw::XcKind::PADE};
dft.max_iter = 60;  dft.tol = 1e-5;  dft.mix = 0.3;
dft.nonperiodic = false;
dft.k_mesh[2] = 2;  dft.k_time_reversal = false;   // n_k = 2
cpu::gpw::GPWEvaluator gpw(std::move(dft), /*compute_forces=*/true);
cpu::electronic::SecondGenCp eval{std::move(gpw), 1, /*corrector_budget=*/5};
```

### Gates

| Quantity | Atomiverse / pin | Oracle | Tol | Notes |
| -------- | ---------------: | ------ | --- | ----- |
| Relative NVE drift \(\delta\) | \(\le 0.10\) | literature SGCP/NVE envelope | 0.10 | \(E_0\) after seed + first warm ASPC step |
| Production SCF iters | \(\le 5\) | SGCP partial corrector | — | full BOMD `max_iter` fails gate |
| \(n_kpts\) | \(\ge 2\) | mesh \(1\times1\times2\) unreduced | — | payload diagnostic |
| \(\lvert P_k\rvert\) | \(= n_kpts\) (`P_k.size() == n_kpts`) | multi-k ASPC seed | — | Γ-only / missing \(P_k\) fails |
| PE after seed | \(< -5\) eV | bound closed-shell H₂ | — | |
| KE after seed | \(\ge 0\), non-trivial | NVE stretch | — | |
| \(r_{\mathrm{HH}}\) after production | \((0.4, 1.5)\) Å | soft bond window | — | no dissociation / collapse |

Offline CP2K sketch (unexecuted): short NVE Quickstep RKS MD with `&KPOINTS`
MONKHORST-PACK \(1\,1\,2\), `EXTRAPOLATION ASPC` order 1, `MAX_SCF 5` — a
method-family twin for inventory and a future fair bench, not a bit-identical
density history or energy.

### Limitations

- Closed-shell multi-k RKS in this section; multi-k UKS dual ASPC
  (\(P_k^\sigma\)) is accepted at `build()` but has no dedicated e2e envelope.
- Cutoff / basis are cheap for the envelope; absolute multi-k forces vs CP2K
  are not re-pinned here.
- Not covered: SGCP + smearing, SGCP + DFT+U + \(k\), hybrid XC SGCP.

## second-gen-cpmd-ot-kpoints — Multi-k SGCP with OT corrector, H₂ NVE {#second-gen-cpmd-ot-kpoints}

Multi-k closed-shell RKS with OT as the partial-SCF corrector. Per sampled
\(k\), `ASPCPredictorCplx` extrapolates \(P_k\); the predicted \(P_k\) is
**purified to natural orbitals** (eigenvectors of \(X_k^\dagger P_k X_k\),
largest occupations, Löwdin-orthonormalized against \(S_k\)) which **seed the
per-k OT manifold** (`guess_P_k`, the complex mirror of the Γ `guess_P` OT warm
start of `second-gen-cpmd-ot`). A partial OT corrector (`corrector_max_iter` ≪
full BOMD) refines the state; analytic multi-k forces are taken at the
post-corrector density.

```cpp
cpu::gpw::GpwDft dft{{H_kind}, 100.0, cpu::gpw::XcKind::PADE};
dft.max_iter = 60;  dft.tol = 1e-5;  dft.mix = 0.3;
dft.nonperiodic = false;
dft.k_mesh[2] = 2;             // 1×1×2 Monkhorst–Pack
dft.k_time_reversal = false;   // two sampled k
dft.use_ot = true;             // OT partial-SCF corrector
cpu::gpw::GPWEvaluator gpw(std::move(dft), /*compute_forces=*/true);
cpu::electronic::SecondGenCp eval{std::move(gpw), /*aspc_k=*/1,
                                  /*corrector_max_iter=*/5};
```

### Pin

No CP2K number can exist for this method. The decisive gates:

1. **Corrector A/B identity** — full-budget multi-k OT vs multi-k
   diagonalization (`GpwDft` at `tol = 1e-7`) at the initial H₂ geometry
   (1×1×2 TR-off mesh, 100 Ry, DZV-GTH-PADE / GTH-PADE-q1): measured
   \(|\Delta E| = 3.0\times10^{-12}\) Ha (gate ≤ 1e-4 Ha, the
   [ot-kpoints](gpw-orbital-transformation.md#ot-kpoints) pin level).
2. **NVE drift envelope**: measured worst relative drift
   \(5.1\times10^{-4}\) (gate ≤ 0.10), production OT iters ≤ budget 5,
   \(n_k = 2\) with full \(P_k\) payload, soft H–H window (0.4–1.5 Å).

Scope: closed-shell multi-k RKS; multi-k UKS + OT is
[second-gen-cpmd-ot-uks-kpoints](#second-gen-cpmd-ot-uks-kpoints).

## second-gen-cpmd-ot-uks-kpoints — Multi-k UKS SGCP with OT corrector, H₂ NVE {#second-gen-cpmd-ot-uks-kpoints}

*No section of the case runs this variant.*

Multi-k collinear UKS with OT as the partial-SCF corrector. Per sampled \(k\)
and spin \(\sigma\), `ASPCPredictorCplx` extrapolates the complex Bloch spin
density \(P_k^\sigma\); each predicted \(P_k^\sigma\) is purified to natural
orbitals (eigenvectors of \(X_k^\dagger P_k^\sigma X_k\), largest occupations,
Löwdin-orthonormalized against \(S_k\)) which seed the per-(k, σ) OT manifold —
the complex dual-spin mirror of the Γ-UKS `guess_P`/`guess_P_beta` OT warm
start (`second-gen-cpmd-uks-ot`) and of the multi-k RKS `guess_P_k` warm start
(`second-gen-cpmd-ot-kpoints`). Multi-k UKS diagonalization SGCP runs inside
`GPWEvaluator::launch`.

```cpp
cpu::gpw::GpwDft dft{{H_kind}, 100.0, cpu::gpw::XcKind::PADE};
dft.max_iter = 60;  dft.tol = 1e-5;  dft.mix = 0.3;
dft.n_alpha = 1;  dft.n_beta = 1;   // collinear UKS path
dft.k_mesh[2] = 2;  dft.k_time_reversal = false;
dft.use_ot = true;                  // dual-spin per-k OT corrector
cpu::gpw::GPWEvaluator gpw(std::move(dft), /*compute_forces=*/true);
cpu::electronic::SecondGenCp eval{std::move(gpw), /*aspc_k=*/1,
                                  /*corrector_max_iter=*/5};
```

### Pin

Values recorded when this page was written (no standing regression); CP2K's
`&OT` + `&KPOINTS` refusal holds for UKS too.

1. **Corrector A/B identity** — full-budget multi-k UKS OT vs multi-k UKS
   diagonalization at the initial H₂ geometry (1×1×2 TR-off mesh, 100 Ry,
   DZV-GTH-PADE / GTH-PADE-q1, \(n_\alpha=n_\beta=1\)): measured
   \(|\Delta E| \approx 3.6\times10^{-11}\) Ha (gate ≤ 1e-4 Ha, the
   [ot-uks-kpoints](gpw-orbital-transformation.md#ot-uks-kpoints) A2 pin
   level); ⟨S²⟩ OT ≡ diag on the multi-k UKS integer seed formula
   \(s_z(s_z+1)+N_\beta\) (= 1 for this filling).
2. **NVE drift envelope**: worst relative drift ≤ 0.10, production OT iters ≤
   budget 5, \(n_k = 2\) with dual-spin \(P_k / P_k^\beta\) payloads, soft H–H
   window.

### Scope

- Singlet fillings (\(n_\alpha = n_\beta = 1\)): the dual-spin plumbing (β
  payloads, dual ASPC, per-(k, σ) OT) is exercised; open-shell k-UKS physics
  rides the same code path via
  [ot-uks-kpoints](gpw-orbital-transformation.md#ot-uks-kpoints).
- Integer-occupation warm start only; the evaluator exposes no smearing.
