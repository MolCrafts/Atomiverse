# Zero-SCF linearized XL-BOMD on the k path — periodic H₂ NVE (GPW shadow BO)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `xl-bomd-linearized-kpoints` |
| Regression | `regressions/dynamics/xl-bomd-linearized-kpoints/` |
| Public API | `cpu::md::MD` + `VelocityVerlet` + `cpu::electronic::XlBomdForce{cpu::gpw::GPWEvaluator{… k_mesh …}, k=5, cpu::electronic::XlBomdLinearized{}}` bound with `set_evaluator_ref`; one `cpu::electronic::XlBomdCplx` propagator per sampled (k, σ) |
| Roadmap / queue | [cp2k-method-queue.md](../../developer/cp2k-method-queue.md) row `xl-bomd-linearized-kpoints`; closes the multi-k residual named by `xl-bomd-linearized-forces` / `xl-bomd-linearized-uks` |
| Dual gate | Correctness unit (`test_cpu_gpw_xl_bomd`) + e2e `reg_xl-bomd-linearized-kpoints` (CP2K-anchored); speed measured ratio (see Performance) |
| Oracle | Offline **CP2K 2026.1** converged-BOMD k-point `.ener` trace (`tools/cp2k_oracle/inputs/xl-bomd-linearized-kpoints/`); never live in ctest |

## Summary

**Zero-SCF linearized-functional XL-BOMD on Monkhorst–Pack k meshes**, RKS
and collinear UKS. Nuclear forces of the Niklasson **shadow
Born–Oppenheimer** potential are evaluated with **no iterative SCF** on warm
MD steps, on the multi-k GPW path (complex Bloch densities). The outer loop
is the MD preset; the force slot binds `cpu::electronic::XlBomdForce` with
the **`XlBomdLinearized`** tag on a `GPWEvaluator` whose `k_mesh` is non-Γ.

Cold seed (first step): one full multi-k SCF seeds one complex auxiliary
density history \(n_k^\sigma\) per sampled k (dual per spin for UKS). Warm
steps: one KS build at fixed \(\{n_k\}\) (`max_iter = 0`), one generalized
diagonalization per (k, σ) → \(D_k^\sigma = q[n]\), the k-weighted shadow
energy

\[
\mathcal{E}(R, \{n_k\})
  = \sum_k w_k \sum_\sigma \mathrm{Re}\,\mathrm{Tr}\!\left[
    H_k^\sigma[n]\,\bigl(D_k^\sigma - n_k^\sigma\bigr)\right]
  + E_{\mathrm{KS}}[n],
\]

and k-summed analytic forces at the \((D_k, \varepsilon_k,
C_k^{\mathrm{occ}})\) payload — since the \(C_k\) are exact eigenvectors of
\(H_k[n]\), the Γ product's off-BO energy-weighted density
\(W = C (C^\dagger H C) C^\dagger\) reduces to the standard
\(\sum_i f \varepsilon_i C C^\dagger\) the k assembly builds. Then
`XlBomdCplx::advance(D_k^σ)` per (k, σ) with Table-I \(K = 5\).

New solver machinery for the UKS leg: `run_gpw_scf_kpoints_uks` gained the
`max_iter == 0` dual-spin fixed-density branch (the k-UKS sibling of the
multi-k RKS CPMD eval) and `GpwResult` the `H_ks_k_beta` payload.

Unlike the Γ linearized slugs (pinned against the literature NVE envelope
when no CP2K binary existed on the pin host), this regression is
**CP2K-anchored**: the per-step shadow \(E_{\mathrm{pot}}\) is gated against
the CP2K 2026.1 converged-SCF k-point BOMD `.ener` trace of the identical
twin input (the BO surface the shadow potential tracks to \(O((D-n)^2)\)),
plus an SCF-XL vs linearized A/B on the same trajectory and an exact UKS
spin-degeneracy identity.

## Method

### Per-k complex shadow step

| Stage | Action |
| ----- | ------ |
| 0. Cold seed | Full multi-k SCF → `advance(P_k^σ_scf)` per (k, σ); histories ready |
| 1. Fixed-\(n\) KS | `max_iter = 0` with `guess_P_k^σ = n_k^σ` → \(H_k^\sigma[n]\), \(S_k\), \(E_{\mathrm{KS}}[n]\) |
| 2. Diags | `KOrth(S_k)` + `diagonalize_k(H_k^σ)` per (k, σ) → \(D_k^\sigma\), \(\varepsilon_k^\sigma\) (occ 2 RKS / 1 per spin) |
| 3. Shadow \(E\) | k-weighted formula above → `kEnergy` [eV] |
| 4. Forces | `gpw_forces_from_scf` on the \((D_k, \varepsilon_k, C_k^{\mathrm{occ}})\) payload → eV/Å |
| 5. XL advance | `advance(D_k^σ)` per (k, σ); `last_result().iters = 0`, `converged = true` |

### System and observable

Identical twin of `xl-bomd-kpoints` / `gpw-bomd-kpoints` at the
`gpw-bomd-kpoints` pin settings: periodic H₂ (R = 1.0 Bohr along x, cubic
12 Bohr cell), DZV-GTH-PADE / GTH-PADE-q1, PADE, CUTOFF 300 Ry, MP 1×1×2
unreduced (2 k), NVE dt 0.5 fs, v₀ = ∓0.001 Å/fs, 1 cold seed + 4
production steps. Three legs: **A** SCF-XL RKS-k anchor, **B** linearized
RKS-k (product under test), **C** linearized UKS-k with
`n_alpha = n_beta = 1` (spin-degeneracy identity vs B).

### Exclusions (this slug)

- Not SCF-converged multi-k XL (`xl-bomd-kpoints` / `xl-bomd-uks-kpoints`)
- Not OT + XL, CPMD, or second-gen CP simultaneous with XL (mutually exclusive)
- Not `compute_stress` with linearized XL (now refused at `build()` — was a
  silent drop on the Γ path)
- Not metallic smearing on the fixed-density k eval (throws; fractional
  occupations need a diagonalization inside the eval)
- Not hybrid / meta-GGA XC; not thermostatted / multi-ps production AIMD

## Implementation

```cpp
cpu::gpw::GPWEvaluator gpw(
    /*kinds*/ {H}, /*cutoff_ry*/ 300.0, /*max_iter*/ 60, /*tol*/ 1e-7,
    /*mix*/ 0.3, /*xc*/ cpu::gpw::XcKind::PADE, /*nonperiodic*/ false,
    /*compute_forces*/ true);
gpw.k_mesh[2] = 2;
gpw.k_time_reversal = false;         // two sampled k (CP2K FULL_GRID)
// gpw.n_alpha = gpw.n_beta = 1;     // collinear UKS leg (dual P_k^σ)
cpu::electronic::XlBomdForce eval{std::move(gpw), /*k=*/5,
                                  cpu::electronic::XlBomdLinearized{}};

cpu::md::MD sim;
sim.set_integrator(cpu::md::VelocityVerlet{0.5});
sim.set_evaluator_ref(eval);
sim.build();
sim.run(frame, 1);   // cold seed: full multi-k SCF, iters > 0
sim.run(frame, 4);   // warm: last_result().iters == 0, shadow E + forces
```

### How to run the regression

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT="$OPENBLASDIR"
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
  cmake --build build -j$(nproc) --target atomiverse reg_xl-bomd-linearized-kpoints
  cmake --build build --target run_reg_xl-bomd-linearized-kpoints
'
```

Relevant sources:

| Layer | Path |
| ----- | ---- |
| Complex propagator (per k, σ) | `include/atomiverse/cpu/electronic/xl_bomd.h` (`XlBomdCplx`) |
| XL wrapper / tag | `include/atomiverse/cpu/electronic/xl_bomd_force.h` |
| Warm multi-k shadow path | `src/cpu/gpw/gpw_evaluator.cpp` |
| k-UKS fixed-density eval | `src/cpu/gpw/gpw_solver.cpp` (`run_gpw_scf_kpoints_uks`, `max_iter == 0`) |
| k forces | `src/cpu/gpw/internal/gpw_forces.cpp` (`assemble_gpw_forces_kpoints`) |
| Unit | `tests/cpu/gpw/test_gpw_xl_bomd.cpp` |
| E2E | `regressions/dynamics/xl-bomd-linearized-kpoints/` |

## Validation and performance

Source: `regressions/dynamics/xl-bomd-linearized-kpoints/REFERENCE.md` —
CP2K 2026.1 `.ener` per-step Pot trace pin (offline; EasyBuild
`2026.1-gmpflf-2026.03` on the Arrhenius epyc login node), per-step shadow
\(E_{\mathrm{pot}}\) band, A/B trace band, C/B spin-degeneracy identity,
NVE drift envelope, and the measured fair-wall ratio. The Table-I
coefficients and the shadow fixed-point identities (Γ real, dual-spin, and
k-weighted complex) are unit-pinned in `test_cpu_gpw_xl_bomd`.

## References

- Offline pin: `regressions/dynamics/xl-bomd-linearized-kpoints/REFERENCE.md`
- CP2K twin: `tools/cp2k_oracle/inputs/xl-bomd-linearized-kpoints/h2_xl_linearized_kpoints_nve.inp`
- Sibling pages: [xl-bomd-linearized-forces](xl-bomd-linearized-forces.md),
  [xl-bomd-linearized-uks](xl-bomd-linearized-uks.md),
  [xl-bomd-kpoints](xl-bomd-kpoints.md),
  [xl-bomd-uks-kpoints](xl-bomd-uks-kpoints.md),
  [gpw-bomd-kpoints](gpw-bomd-kpoints.md)
- A. M. N. Niklasson, *Phys. Rev. Lett.* **100**, 123004 (2008)
- A. M. N. Niklasson et al., *J. Chem. Phys.* **130**, 214109 (2009) — Table I
- A. M. N. Niklasson & M. J. Cawkwell, *Phys. Rev. B* **86**, 174308 (2012)
- A. M. N. Niklasson, *J. Chem. Phys.* **147**, 054103 (2017)
- J. VandeVondele et al., *Comput. Phys. Commun.* **167**, 103 (2005)
- Monkhorst & Pack, *Phys. Rev. B* **13**, 5188 (1976)
