# Post-Hartree–Fock Correlation

Tutorial: [Electron correlation](../tutorial/ch07-correlation.md).

Correlation on a converged HF reference. Each method is its own type
(`MP2{mf}`, `MP3`, `MP4`, `CC2`, `CCSD` with `ccsd_t`, `CASCI`, `CASSCF`,
`FCI`, `NEVPT2`), not a member of `MeanFieldMethod`. CCSD and CC2 amplitude
iterations are `Driver<cc::Phase>`; CASSCF's orbital macro-iteration is
`Driver<mc::Phase>`; the others are single-shot and throw on a
non-converged reference. Legend: [Roadmap index](index.md).

## 1. Møller–Plesset

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| RMP2 | ✓ | ○ | Canonical O(N⁵) |
| UMP2 | ✓ | ○ | `MP2` on `UHF` |
| ROMP2 | ✓ | ○ | UMP2 on ROHF reference |
| Frozen-core MP2 | ✓ | ○ | `MP2::frozen = N`; RMP2 pinned (RI-MP2 combination reachable) |
| RI-MP2 | ◐ | ○ | `ri = RI::{J,JK}` + aux; RHF only. No external pin; self-pinned case removed |
| SCS-MP2 / SOS-MP2 | ✓ | ○ | `MP2::os_scale` / `ss_scale` |
| RMP2 / UMP2 analytic gradient | ✓ | ○ | Lagrangian + Z-vector |
| ROMP2 gradient | ○ | ○ | |
| MP2 Hessian | ○ | ○ | |
| RMP3 | ✓ | ○ | Closed-shell |
| RMP4(SDTQ) | ✓ | ○ | Closed-shell |
| Local / DLPNO / F12 MP2 | ○ | ○ | |

## 2. Configuration interaction

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| FCI (determinant-driven) | ✓ | ○ | Dense matrix; cap `n_mo > 14` or `N_det > 2000` — oracle tool, not production |
| Selected / heat-bath CI | ○ | ○ | |

CIS excitation energies live under
[Excited States](excited-states.md) (`TDA<RHF>::run`), not here.

## 3. Coupled cluster

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| CC2 | ✓ | ○ | `cpu::scf::CC2<Method>` : `cpu::Driver<cc::Phase>` (not `MeanFieldMethod`, no `HF` tag). Stages `update → mix → energy → check`, mounted with `on(phase).add` in the constructor. Doubles residual is PySCF `rccsd.update_amps` when `cc.cc2` (full singles, T₂ at most linear in T₂). max\|Δt\| is published on `ConvergenceMetrics::residual`; `AmplitudeConverged` decides; the method does not assign convergence. `CC2{mf, diis = true, tol = 1e-10, max_iter = 100}` — const after construction. The reference is not owned. `run(Frame&)` drives the reference SCF, overwrites `kEnergy` with the correlated total [eV], and returns `CC2Result` [Hartree]. `run(const Frame&)` writes nothing. There is no `forces(Frame&)`. `provides_energy()` and `requires_full_frame()` are true; `provides_force()` is false. Nested `on(phase).add` accumulates `kEnergy` and refuses a force request. Closed-shell RHF only. No CUDA path. `eom_cc2` stays a method of this object. Pinned in `scf/coupled-cluster`: H₂/STO-3G (T₁ = 0, CC2 = MP2) and H₂O/STO-3G (T₁ nonzero, distinct from MP2). Tutorial: [H₂](../tutorial/methods/scf-coupled-cluster.md#cc2-h2-sto3g), [H₂O](../tutorial/methods/scf-coupled-cluster.md#cc2-h2o-sto3g) |
| CCSD | ✓ | ○ | `cpu::scf::CCSD<Method>` : `cpu::Driver<cc::Phase>` (not `MeanFieldMethod`, no `HF` tag). Stages `update → mix → energy → check`, mounted with `on(phase).add` in the constructor. max\|Δt\| is published on `ConvergenceMetrics::residual`; `AmplitudeConverged` decides; the method does not assign convergence. `CCSD{mf, diis = true, frozen = 0, tol = 1e-10, max_iter = 100}` — const after construction. The reference is not owned. `run(Frame&)` drives the reference SCF, overwrites `kEnergy` with the correlated total [eV], and returns `CCSDResult` [Hartree]. `run(const Frame&)` writes nothing. `forces(Frame&)` overwrites `kAtom_f*` [eV/Å] and `kEnergy` [eV] and returns `GradientResult`. `ccsd_t(Frame&)` returns `CCSDTResult` [Hartree]. `provides_energy()`, `provides_force()`, and `requires_full_frame()` are true. Nested `on(phase).add` accumulates `kEnergy`, and `kAtom_f*` when the parent requests forces. Closed-shell RHF only. `frozen != 0` throws on `forces`. No CUDA path. Tutorial: [coupled cluster](../tutorial/methods/scf-coupled-cluster.md#call) |
| CCSD(T) | ✓ | ○ | Non-iterative triples |
| CCSD nuclear gradient | ✓ | ○ | FD of CCSD energy (not Lagrangian/Z-vector yet) |
| CCSD(T) gradient / CCSD Hessian | ○ | ○ | |
| Open-shell / DF / F12 / local CC | ○ | ○ | |

## 4. Multireference

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| CASCI | ✓ | ○ | Dense FCI in active space |
| CASSCF | ✓ | ○ | `cpu::scf::CASSCF<Method>` : `cpu::Driver<mc::Phase>` (not `MeanFieldMethod`, no `HF` tag). Stages `rotate → casci → gradient → check`, mounted with `on(phase).add` in the constructor. max \|g\| is published on `ConvergenceMetrics::residual`; `OrbitalGradientConverged` decides; the method does not assign convergence. `CASSCF{mf, ncas, nelecas}` — `orb_tol` defaults to `max(mf.tol, 1e-8)` [Hartree], `max_macro = 50`. Explicit overload `CASSCF{mf, ncas, nelecas, orb_tol, max_macro}`. `ncas`, `nelecas`, `orb_tol`, and `max_macro` are const after construction. The reference is not owned. `run(Frame&)` drives the reference SCF, overwrites `kEnergy` with the correlated total [eV], and returns `CASSCFResult` [Hartree] (`e_casscf`, `e_corr`, `ncas`, `nelecas`, `ncore`, `iterations`, `converged`). `run(const Frame&)` writes nothing. There is no `forces(Frame&)`. `provides_energy()` and `requires_full_frame()` are true; `provides_force()` is false. Nested `on(phase).add` accumulates `kEnergy` and refuses a force request. Closed-shell RHF only. No CUDA path. Pinned in `scf/multireference`: H₂/STO-3G CAS(2,2) (ncore = nvirt = 0, one CI pass) and H₂O/STO-3G CAS(2,2) (ncore = 4, nvirt = 1, orbital step). Tutorial: [multireference](../tutorial/methods/scf-multireference.md#call) |
| SC-NEVPT2 | ✓ | ○ | On closed-shell CASSCF |
| State-averaged CAS / RAS / GAS / DMRG | ○ | ○ | |
| CASPT2 / PC-NEVPT2 / QD-NEVPT2 | ○ | ○ | |
| CASSCF analytic gradient | ○ | ○ | |
