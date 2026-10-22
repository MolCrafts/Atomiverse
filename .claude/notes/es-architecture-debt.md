<!-- mol:note:topic:es-architecture-debt -->
## ES public-shape debt

Science factorization is sound (shared SCF/energy loop + concrete Hamiltonian;
PW / GPW / SE domain-split; method-as-nested-Driver for BOMD). Engineering does
**not** yet obey one public shape across families. Review verdict **BLOCK**
until the 🔴 rows close or become named exceptions in CLAUDE.md.

**Rule:** do not extend any dialect (no new Frame-bypass geometry, free-function
force entry, binding path or deeper inheritance). Close toward: concrete method
= `Driver<*::Phase>` + `run(Frame&)` / `forces(Frame&)` + one outer bind rule.

### Target

| Axis | Claim |
| --- | --- |
| Ownership | Concrete method owns Hamiltonian + parameters |
| Loop | Self-consistency is a typed-stage `Driver` |
| I/O | `Frame` (`run` / `forces` / `stress`, CLAUDE.md table) |
| Kind | `HF` / `Semiempirical` / `PlaneWave` / `GPW` are energy-source tags only |
| Depth | Inheritance ≤ 2 |

### Post-SCF and response are not mean-field members

Post-HF, response, solvent and Hessians wrap a reference method, mirroring
PySCF: `Stability<RHF>`, `TDA` / `TDHF<RHF>`, `TDA` / `TDDFT<RKS|UKS>`,
`SfTDA` / `SfTDHF<UHF>`, `CISD<RHF>` (CIS(D)), `G0W0`, `ADC`, `DirectRPA`,
`BSE`, `GF2<RHF>`, `CCSD<RHF>::eom_{ee,ip,ea}_ccsd`, `CC2<RHF>::eom_cc2`,
`CCSD<UHF>::eom_sf_ccsd`, `MP2` / `CCSD` / `CASCI(mf)`,
`Hessian<RHF|UHF|RKS|UKS|ROHF>` (KS/ROHF differentiate `forces`; nuclear CPHF
in `Hessian<RHF>`), `Polarizability<RHF>`, `WithSolvent<Method>`; FD skeleton
`central_difference_nuclear_hessian`. Density fitting stays an SCF JK selector
(`ri`).

### Live dialects (do not grow)

1. **Method-as-Driver** — CPU Gaussian mean-field (`MeanFieldMethod`).
2. **Method + `*Evaluator`** — GPW only (the wrapper owns the method and the
   CPMD/XL propagator state). Gaussian SCF, AM1/RM1 and PW mount directly as
   nested nodes on the parent Runtime.
3. **Loop inside Component** — some DFTB as bare `Semiempirical`. NDDO
   self-consistency is `NddoMethod` (`Driver<nddo::Phase>`).

### Ledger

| Sev | ID | Where | Debt |
| --- | --- | --- | --- |
| 🔴 | `es-bind-dialects` | `*Evaluator` vs direct nest vs DFTB-as-Component | Target (spec `es-unification`): mount the method itself. Done for Gaussian SCF, AM1/RM1 and PW. Open: `GPWEvaluator` (GPW nesting rejected), DFTB/xTB as bare Components. |
| 🔴 | `es-red-regressions` | `regressions/{gpw,semiempirical,architecture}` | Red independent of shape work: Γ-Si periodic post-SCF family (`periodic-{gw,mp2,ri-mp2,rpa,scs-mp2,scs-ump2,ugw,ump2,urpa}-si`, `ugw-uks-rpbe` — Γ SCF not converged in 80 iters), `periodic-{gw,ugw}-kpoints`, `periodic-gw-finite-q`, `gpw-kpoints-si` (>600 s), `gpw-fd-phonon-si`, `gpw-dftu-kpoints`, `second-gen-cpmd-ot`, GFN2 family (`gfn2-{aes-fock,camm-r,quadrupole-aes,xtb-forces,xtb-water}`, `gfn-xtb-analytic-hessian`). Re-verify against current `dev`. |
| 🟡 | `es-cuda-gpw-frame-io` | `cuda/gpw/gpw_solver.cuh` | Frame `build`/`run`/`forces` exist (`kinds` + `frame.box`); device kernels and CUDA regs still take `vector<GpwAtom>`. |
| 🟡 | `es-pw-evaluator-scf` | `cpu/planewave/pw_evaluator.cpp` | Forces + stress launch = 3 SCFs; assemble from the `run` density. |
| 🟡 | `es-cuda-pw-driver` | `cuda/planewave/pw_solver.cuh` | Frame `run`/`forces`/`stress` exist; SCF loop is inside the method, not `Driver<pw::Phase>`; free `pw_forces_analytic` / `pw_stress` are the device implementations. |
| 🟡 | `es-gpw-ot-freeloop` | `GpwDft` `use_ot` | OT still free-loop; DIIS paths are Driver stages — no third topology. |
| 🟡 | `es-dftb-qmengine` | DFTB0/3, GFN0 vs DFTB2, GFN1/2 | Uneven `QmEngine` coverage for QM/MM `set_engine`. |
| 🟡 | `es-docs-drift` | tutorials / `*Evaluator` comments | "mirrors Gaussian" without Evaluator caveat; ch09 `role()/capabilities()` prose; stale UKS / multi-k "throws" lists in `periodic-{mp2,ri-mp2,rpa,gw}-si.md`, `periodic-gw-kpoints.md`, `ot-smearing.md`; `second-gen-cpmd-uks.md` presents private flags as members. |
| 🟡 | `es-reg-internal` | `regressions/gpw/*` dispersion pins | ~25 GPW cases include `internal/{dftd3,dftd3_params,ts_dispersion,mbd_dispersion,xdm_dispersion}.h` — violates iron law 1a. Pin through the public method or move kernel pins to `tests/`. |
| 🟡 | `es-gpw-thread-determinism` | GPW Γ-UKS+U, Γ-UKS OT | Differ run to run at 8 threads: `gpw-dftu-forces`, `gpw-dftu-kpoints`, `second-gen-cpmd-uks-ot`; `test_cpu_gpw_thread_reproducibility` does not cover these paths. |
| 🟡 | `es-cuda-es-tests-e2e` | `tests/cuda/gpw/test_gpw_{full_product,kpoints,stress}.cu`, `tests/cuda/planewave/test_pw_cell_stress.cu` | Full CUDA + CPU-oracle SCFs under ctest — iron law 1. Move to `regressions/{gpw,pw}` `--device=both` cases. |
| — | `es-kind-model` | `cpu/component/hf.h` | Decided: `HF` stays a tag on the Driver node (Gaussian HF and hybrid KS); `Semiempirical` / `PlaneWave` / `GPW` stay Component kinds; the slot copies `provides_*`. |
| — | `es-ghost-atoms` | `MeanFieldMethod::ghost_atoms` | Named exception. |

### Conformance

| Method family | Driver? | Frame `run`/`forces`? | Conforms? |
| --- | --- | --- | --- |
| CPU RHF/UHF/RKS/UKS/ROHF/ROKS/GHF | Yes | Yes | Pass |
| CUDA RHF / UHF | Yes | Yes (UHF: STO-3G l ≤ 1, else throws) | Pass |
| AM1 / RM1 | Yes (`NddoMethod`, nested on parent Runtime) | Yes (both `QmEngine`) | Pass |
| CPU PW | Yes | Yes (+ `stress`) | Pass (evaluator 3-SCF aside) |
| CUDA PW | No | Yes | Fail (`es-cuda-pw-driver`) |
| CPU GpwDft | Yes (OT residual) | Yes (+ `stress`) | Pass |
| CUDA GpwDft | No | Frame shell over `GpwAtom` kernels | Fail (`es-cuda-gpw-frame-io`) |
| `*Evaluator` wrappers | Component | launch → Frame | Bind shell only |
| `CarParrinelloMethod` / `XlBomdMethod` | Composed `DriverNode` | owned method via `method()` | Pass |

### Close order

1. `es-bind-dialects`
2. `es-cuda-pw-driver` / `es-cuda-gpw-frame-io`
3. QmEngine coverage
4. Hygiene: docs, OT residual, `es-reg-internal`, `es-cuda-es-tests-e2e`
