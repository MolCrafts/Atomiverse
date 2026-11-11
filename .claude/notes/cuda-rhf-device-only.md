# CUDA ES — RHF and PW device paths

Residency law: [cuda-only-no-dh.md](cuda-only-no-dh.md). GPW:
[cuda-gpw-public-shape.md](cuda-gpw-public-shape.md).

## Process (binding when writing or optimizing CUDA kernels)

Use the installed `/mol:cuda` workflow and its SIMT design worksheet before
writing a non-trivial kernel. Record launch ownership, layout/coalescing,
reductions, reuse, tails and numerical tolerances; check suitable CUDA-X
primitives before creating local kernels. Correctness, whole-call timing and
observed resource use are separate acceptance evidence.

GH200 configure: `-DCMAKE_CUDA_ARCHITECTURES=90`; do not link x86 `libxtb`
(`-DATV_BUILD_SEMIEMPIRICAL=OFF` or empty `ATV_XTB_ROOT`).

## `cuda::scf::RHF`

| Case | Path |
| --- | --- |
| STO-3G, available element tables with s/p shells (l ≤ 1, mixed Z) | **Device** energy (`RHFFockDevice` + `launch_*_sp`) and forces (`launch_build_W_rhf` + `launch_rhf_forces_sp` → `frame.f*` eV/Å) |
| Cartesian or spherical l ≤ 3 / other supported bases | Device `launch_*_lgt1` integrals and device Cartesian-to-spherical contractions when needed; energy only, unsupported forces throw |

- Device path: Löwdin S^{-1/2}, geometry and V_nuc on device; path selection
  from host Z cached before upload (`cache_host_types`); CDIIS (Pulay) mixer
  with device history, cuBLAS history Gram/extrapolation and cuSOLVER pivoted
  LU for the (m+1) Pulay system; `level_shift` (Saunders–Hillier, |λ| ≤ 1e-4 no-op, energy from
  unshifted F so E is λ-invariant) on both paths; non-default stream.
- HF diagonalization reuses cuSOLVER batched Jacobi through order 64;
  orders 65..215 use a bounded device-controlled cyclic Jacobi solve with
  CUB norms and sorting. Both evolving matrices stay in shared memory when
  actual static + dynamic capacity permits; an odd leading dimension spreads
  column accesses. Larger orders keep the same CUDA math in global scratch.
  The installed vendor's larger-matrix paths download
  norms during sweeps, and its generic ALGO_1 option was rejected at setup.
  This dispatch needs boundary residency traces and unchanged eigenpair gates;
  it is not justified by API names alone. The generic synthetic helper's
  legacy path above 215 is outside the CUDA-only HF envelope.
- HF acceptance retains the density tolerance and adds energy change in
  Hartree plus the unmixed Fock commutator. A final raw-Fock canonical pass
  must also satisfy these checks before post-HF consumes the orbitals.
- No scientific D2H during SCF + forces: residual via cuBLAS `nrm2` → device
  scalar with a mapped converge flag; energy via device-pointer `dot` + `d_V_nuc`.
  Unused residual/V_nuc host mirrors are not retrieved. Analytic force scratch
  is reserved during setup; energy is retrieved after force finalization.
  Failed or unconverged solves throw without retrieving the candidate Frame.
- Nested RHF/UHF/MP2 leave results in the parent's device Frame. They do not
  retrieve unused host energy/MP2 payloads; `energy()` after nested or failed
  RHF/UHF throws instead of returning stale cached science.
- Never retype STO-3G numbers: `basis_data/accessors.h` over `sto3g.h`.
- Product gates (`regressions/scf/rhf`): CPU twin |ΔF| 1e-5 (H₂) / 1e-4
  (H₂O) eV/Å, energy `ha_to_ev(1e-6)`; FD spot checks; level-shift invariance
  1e-8 Ha.

UHF STO-3G s/p MINAO starts pack only element constants on the host. Cross
overlap, normalization, projection and spin density are computed on device
with existing overlap primitives, cuSOLVER Cholesky/solve and cuBLAS. Scratch
is reserved before iteration zero, and geometry changes do not require a
host projection. Missing MINAO tables retain the existing Hcore start.

Scalar and batched Boys integrals share `core/gto.h`: a positive series and
downward recurrence near zero, stable upward recurrence at large arguments.
CPU nuclear blocks and all supported CUDA angular orders use this law.
Real s/p AO ERIs exploit eightfold permutation symmetry without a second
tensor. The Hermite primitive retains only adjacent recurrence layers in
an axis-bounded box (at most 18 cells per layer) and does not clear inactive
storage. An exactly zero prefactor skips the primitive tail only within a
proven finite-arithmetic domain; this is not near-zero screening, and the
original CUDA path preserves exceptional arithmetic outside that domain.
Compiler per-thread ERI stack is 1,008 B with 128 registers; packing alone
has no clear measured latency gain. Shared Jacobi, odd shared leading and
exact-zero work avoidance reduce the target complete HF call to 568 ms
at 16 waters/112 AOs versus 918 ms CPU direct (8 threads), with unchanged gates.
Small-case regressions and latency outliers were observed in the same
2026-10-26 HF timing/residency campaign (its write-up is no longer in the tree).

## CUDA post-HF handoff

`cuda::scf::MP2{rhf}.run(frame)` borrows the RHF execution owner and consumes
its private device orbitals, eigenvalues and AO ERIs before that workspace is
released. Correlation is closed-shell canonical MP2, with frozen occupied
orbitals and independent OS/SS scales. Force requests reject explicitly.

Four cuBLAS contractions form one occupied-orbital `(ia|jb)` tile; a CUB
transform reduction produces OS/SS channels without a full MO ERI tensor or
per-term contribution array. Additional algorithm scratch is bounded by
`workspace_budget_bytes` (default 512 MiB); HF AO ERIs and library contexts
are outside that additional budget. Dense HF integral indices currently
require nAO^4 ≤ INT_MAX; setup rejects larger shapes before allocation.

Correctness targets reuse the PySCF pins in `regressions/scf/mp2` (SCS water /
STO-3G, frozen-core water / 6-31G*, and the 56-AO eight-water chain). The standalone
`bench_mp2_cpu_vs_cuda` measures complete RHF→MP2 calls, first and repeated
calls, with thread counts, iterations and resource planning. Memory snapshots
do not establish allocation peaks or residency; those need an observed trace.
New CUDA changes require GH200 validation before numerical or performance
acceptance.

2026-10-26 acceptance: GH200 job 4439121 passed all 83 selected HF/UHF/MP2
unit/physical cases, including frozen/SCS, forces and external pins. All
25 memcheck and four racecheck selections passed without errors or hazards.
The final snapshot has ten HF→MP2 and six 64/65/112/116/117/215 eigen science
windows with zero H2D/D2H, dynamic Device events, alloc/free APIs or UM
activity and complete collection. Cold immutable module statics are separate.
The unchanged UHF MINAO projection module retains its earlier six-window
residency evidence (recorded in the 2026-10-26 post-HF validation; the
write-up is no longer in the tree).

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
- Correctness: `regressions/pw/rks --device=both` |ΔE| ~1e-14 Ry
  (bound 1e-3 Ry — do not tighten from one cell); forces/stress CPU twins at
  ~1e-12.
- Large-cell evidence (ES order step 4, Si Γ LSDA, ecut 15 Ry,
  `benchmarks/perf/pw/bench_pw_large_cell.cu`): 64 atoms CUDA 9.7 s vs
  same-settings CPU > 5400 s; 216 atoms CUDA 191 s. 8-atom cells are
  correctness-only. CPU PW is serial (threads enter only via OpenBLAS).
