# CUDA ES — RHF and PW device paths

Residency law: [cuda-only-no-dh.md](cuda-only-no-dh.md). GPW:
[cuda-gpw-public-shape.md](cuda-gpw-public-shape.md).

## Process (binding when writing or optimizing CUDA kernels)

Load and follow the matching CUDA skill, not ad-hoc style:

| Concern | Skill |
| --- | --- |
| Block/grid size, occupancy, tail | `choose-launch-configuration` |
| Coalescing / layout | `optimize-global-memory-access` |
| Elementwise fuse | `fuse-elementwise-ops` |
| Partial tiles / `idx < N` | `handle-boundary-conditions` |
| Numerics (thresholds, rsqrt, fp) | `write-numerically-stable-kernel` |
| Warp divergence | `avoid-warp-divergence` |
| Correctness after change | `debug-cuda-kernel-correctness` |

GH200 configure: `-DCMAKE_CUDA_ARCHITECTURES=90`; do not link x86 `libxtb`
(`-DATV_BUILD_SEMIEMPIRICAL=OFF` or empty `ATV_XTB_ROOT`).

## `cuda::scf::RHF`

| Case | Path |
| --- | --- |
| STO-3G, every atom s/p (l ≤ 1, mixed Z) | **Device** energy (`RHFFockDevice` + `launch_*_sp`) and forces (`launch_build_W_rhf` + `launch_rhf_forces_sp` → `frame.f*` eV/Å) |
| Higher L / other bases | Host fill S/Hcore/ERI → upload → device SCF; host forces |

- Device path: Löwdin S^{-1/2}, geometry and V_nuc on device; path selection
  from host Z cached before upload (`cache_host_types`); CDIIS (Pulay) mixer
  with device history (CPU `MixStage` twin; only the (m+1) Pulay solve on
  host); `level_shift` (Saunders–Hillier, |λ| ≤ 1e-4 no-op, energy from
  unshifted F so E is λ-invariant) on both paths; non-default stream.
- No scientific D2H during SCF + forces: residual via cuBLAS `nrm2` → device
  scalar with a mapped converge flag; energy via device-pointer `dot` + `d_V_nuc`.
  After SCF only bookkeeping mirrors (residual, V_nuc, `energy_ha_`).
- Never retype STO-3G numbers: `basis_data/accessors.h` over `sto3g.h`.
- Unit gates (`tests/cuda/scf/test_rhf.cu`): CPU twin |ΔF| 1e-5 (H₂) / 1e-4
  (H₂O) eV/Å, energy `ha_to_ev(1e-6)`; FD spot checks; level-shift invariance
  1e-8 Ha.

## `cuda::pw::PW`

- Public shape mirrors `cpu::pw::PW`: constructor (pseudopotential, `ecut_ry`, `k_mesh`, `xc`) + members → `build(frame)` → `run(frame)` →
  `PWResult`; Frame `forces` / `stress` members call the device
  `pw_forces_analytic` / `pw_stress`.
- Envelope (everything else throws `std::invalid_argument` naming the member):
  `XCType::LSDA` or `PBE`; closed shell or collinear fixed-occupation UKS;
  fixed occupation or metallic smear (shared μ, f ∈ [0,1]); `k_mesh ≥ 1` without
  MPI k-distribution. Spin × smear **stress** still rejects.
- Density side is device-resident (ρ(G) → V_H → V_xc → V_eff → Pulay →
  reductions); per iteration only the ≤ 8×8 Pulay Gram, its coefficients and
  five scalars cross PCIe. The coefficient solve is the shared host
  `cpu::pw::internal::solve_pulay_coefficients`.
- Band side: `build_kpoints` uploads every per-k table and the initial Davidson
  subspace once per run (sized against `cudaMemGetInfo`; k-batching not
  implemented); `bind_kpoint` is a pointer swap + one β launch. β is not cached
  across k (O(n_G·n_proj) per k).
- Subspace eigenproblem: cuSOLVER `Zheevd` in place on column-major `d_hs_`
  (upper fill = the CPU triangle). Not bit-identical to CPU Jacobi: eigenvalues
  agree to roundoff; density is invariant only if no degenerate multiplet is
  split at the n_bands cut (gapped, fixed occupations). Converged eigenvalues
  are the contract.
- Residual-expansion norms are device-pointer (`blas_dev_` handle pinned to
  `CUBLAS_POINTER_MODE_DEVICE`): one sync per band; Gram-Schmidt is sequential
  across bands, so that is the floor. Still on host per Davidson round: the
  `Zheevd` `devInfo`, the n_bands eigenvalues, the accept flag.
- Correctness: `regressions/pw/pw-cuda-lda-si --device=both` |ΔE| ~1e-14 Ry
  (bound 1e-3 Ry — do not tighten from one cell); forces/stress CPU twins at
  ~1e-12.
- Large-cell evidence (ES order step 4, Si Γ LSDA, ecut 15 Ry,
  `benchmarks/cuda/bench_pw_large_cell.cu`): 64 atoms CUDA 9.7 s vs
  same-settings CPU > 5400 s; 216 atoms CUDA 191 s. 8-atom cells are
  correctness-only. CPU PW is serial (threads enter only via OpenBLAS).
