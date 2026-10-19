<!-- mol:note:topic:es-architecture-debt -->
## [2026-09-24] ES public-shape debt

Science factorization is sound: shared SCF/energy loop + concrete
Hamiltonian / representation; PW / GPW / SE domain-split; method-as-nested
Driver for BOMD. **Engineering does not yet obey one public shape** across
method families. Review verdict **BLOCK** until the 🔴 rows below close
or are explicitly carved as named exceptions in CLAUDE.md.

**Rule**: Do not extend any dialect below (new Frame-bypass geometry,
new free-function force entry, new binding path, deeper inheritance).
Prefer closing toward: concrete method = `Driver<*::Phase>` (or one chosen
Component shape) + `run(Frame&)` / `forces(Frame&)` + one outer bind rule.
Detail lives here; CLAUDE.md keeps only a thin Remaining-debt pointer.

**Supersedes**: ad-hoc “Remaining debt” bullets that implied only CUDA RHF /
PW free-loop / GPW OT — those are rows in this ledger, not the whole story.

### Target principle (claimed)

| Axis | Claim |
| --- | --- |
| Ownership | Concrete method owns Hamiltonian + parameters |
| Loop | Self-consistency is a typed-stage `Driver` |
| I/O | Public currency is `Frame` (`run` / `forces` / `stress`, CLAUDE.md table) |
| Kind | `HF` / `Semiempirical` / `PlaneWave` / `GPW` are energy-source tags only |
| Depth | Inheritance ≤ 2 |

### Live dialects (do not grow)

Gaussian post-HF, continuum solvent, the shared nuclear FD Hessian, and
CPHF response are **not** members of `MeanFieldMethod`. `Stability<RHF>`,
`TDA<RHF>`, `TDHF<RHF>`, `CISD<RHF>`, `G0W0<RHF>`, `ADC<RHF>`,
`DirectRPA<RHF>`, `BSE<RHF>`, `GF2<RHF>`, `CCSD<RHF>::eom_ee_ccsd` /
`eom_ip_ccsd` / `eom_ea_ccsd`, `CC2<RHF>::eom_cc2`, and
`CCSD<UHF>::eom_sf_ccsd` are not members of `RHF` or `UHF`
(`pyscf.scf.stability.rhf_stability`, `pyscf.tdscf.rhf.TDA` /
`TDHF`, `pyscf.adc.ADC`, `gw.rpa.RPA`, `gw.bse.BSE`, `agf2.RAGF2`,
`gw.gw_exact.GWExact`, `cc.CCSD.eomee_ccsd_singlet` / `ipccsd` / `eaccsd`,
`cc.UCCSD.eomsf_ccsd`). CIS(D) is `CISD<RHF>` (PySCF has no CIS(D) class). Nuclear Hessians are
`Hessian<RHF|UHF|RKS|UKS|ROHF>` (`pyscf.hessian.rhf.Hessian` /
`hessian.uhf.Hessian`; KS and ROHF differentiate `forces`). KS linear
response is `TDA<RKS|UKS>` / `TDDFT<RKS|UKS>` (`pyscf.tdscf.rks` /
`uks`). Spin-flip is `SfTDA<UHF>` / `SfTDHF<UHF>`. `MP2` / `CCSD` /
`CASCI` / … take the mean-field reference (`pyscf.mp.mp2.MP2(mf)`,
`cc.CCSD.ccsd_t`). Solvent is `WithSolvent<Method>`
(`pyscf.solvent._attach_solvent.SCFWithSolvent`). The FD skeleton is
`central_difference_nuclear_hessian`. Nuclear CPHF is
`Hessian<RHF>` (`pyscf.hessian.rhf.Hessian(mf).kernel`); static
polarizability is `Polarizability<RHF>`. `pyscf.scf.cphf.solve` is not an
SCF method. Density fitting stays an SCF JK selector (`ri`);
`pyscf.df.df_jk.density_fit` returns another SCF.


1. **Method-as-Driver** — CPU Gaussian mean-field (`MeanFieldMethod`).
2. **Method + `*Evaluator`** — PW / AM1 / RM1 (Frame write often only on the
   wrapper; evaluator duplicates method params). GPW is the reference shape:
   Frame-in method + evaluator that only owns it.
3. **Loop inside Component** — some DFTB as bare `Semiempirical`. NDDO
   self-consistency is `NddoMethod` (`Driver<nddo::Phase>`).

### Debt ledger

Priority: 🔴 bind/forces/Driver alignment before any new ES public API;
🟡 are science/docs/harness hygiene.

| Sev | ID | Where | Debt |
| --- | --- | --- | --- |
| 🟡 | `es-cuda-gpw-frame-io` | `cuda/gpw/gpw_solver.cuh` | Frame `build` / `run` / `forces` exist and use `kinds` + `frame.box`. The device kernels and the current CUDA regs still take `vector<GpwAtom>`; `cell[3]` is filled from the box on the Frame path. |
| 🔴 | `es-bind-dialects` | `*Evaluator` vs direct nest vs DFTB-as-Component | `GPWEvaluator` / `PWEvaluator` / `RM1Evaluator` own one method, no parameter copy. `AM1Evaluator(int charge, …)` still re-exposes the method ctor; AM1 binds through `QmEngine::publish` while RM1/PW accumulate in the evaluator; AM1/RM1 evaluators carry byte-identical PBC unwrap helpers; DFTB family is Component-only. Pick one outer bind story; document matrix; delete redundant shells. |
| 🟡 | `es-pw-evaluator-scf` | `cpu/planewave/pw_evaluator.cpp` | Forces + stress launch = 3 SCFs (`run`, force path, stress path each converge their own density). Assemble forces/stress from the `run` density instead; needs the force/stress impls to take the converged workspace. |
|  | `es-kind-model` | `cpu/component/hf.h` | Decided. `HF` stays a tag on the Driver node. `Semiempirical` / `PlaneWave` / `GPW` stay Component kinds. The registration slot copies `provides_energy` / `provides_force`. No public adapter. |
| 🟡 | `es-cuda-pw-driver` | `cuda/planewave/pw_solver.cuh` | Frame `run` / `forces` / `stress` are members and write `kEnergy` / `kAtom_f*` / `kStress`. The SCF loop stays inside the method object; it is not `Driver<pw::Phase>`. Free `pw_forces_analytic` / `pw_stress` remain the device implementations those members call. |
| 🟡 | `es-gpw-ot-freeloop` | `GpwDft` `use_ot` | OT still free-loop; DIIS paths are Driver stages — do not add a third topology. |
|  | `es-ghost-atoms` | `MeanFieldMethod::ghost_atoms` | Named exception. Gaussian SCF ghost indices stay on the method. |
| 🟡 | `es-dftb-qmengine` | DFTB0/3 / GFN0 vs DFTB2 / GFN1/2 | Uneven `QmEngine` coverage for QM/MM `set_engine`. |
|  | `es-kind-name-hf` | `hf.h` | Documented. The tag is Gaussian-basis HF and hybrid KS. |
| 🟡 | `es-docs-drift` | tutorials / `*Evaluator` comments | “mirrors Gaussian” without Evaluator caveat; ch09 `role()/capabilities()` prose (no Capability set exists); stale UKS / multi-k “throws” guard lists in `periodic-{mp2,ri-mp2,rpa,gw}-si.md`, `periodic-gw-kpoints.md`, `ot-smearing.md`; `second-gen-cpmd-uks.md` presents private `second_gen_cp*` flags as members. |
| 🟡 | `es-reg-internal` | `regressions/gpw/*` standalone dispersion pins | ~25 GPW cases still include `internal/{dftd3,dftd3_params,ts_dispersion,mbd_dispersion,xdm_dispersion}.h` to pin the standalone kernel — out-of-library consumer law. Pin through the public method (E_disp on `GpwResult`, F(disp)−F(plain)) or move the kernel pin to `tests/`. |
| 🔴 | `es-red-regressions` | `regressions/{gpw,semiempirical,architecture}` | Red on `dev` independent of the shape work (reproduce on `a3bc83ec`): Γ-Si periodic post-SCF family (`periodic-{gw,mp2,ri-mp2,rpa,scs-mp2,scs-ump2,ugw,ump2,urpa}-si`, `ugw-uks-rpbe` — Γ SCF does not converge in 80 iters), `periodic-{gw,ugw}-kpoints`, `periodic-gw-finite-q` (QP gap), `gpw-kpoints-si` (>600 s), `gpw-fd-phonon-si` (soft modes), `gpw-dftu-kpoints`, `second-gen-cpmd-ot`, GFN2 family (`gfn2-{aes-fock,camm-r,quadrupole-aes,xtb-forces,xtb-water}`, `gfn-xtb-analytic-hessian`). |
| 🟡 | `es-gpw-thread-determinism` | GPW Γ-UKS+U SCF, Γ-UKS OT | Differ run to run at 8 threads (identical at 1): `gpw-dftu-forces`, `gpw-dftu-kpoints`, `second-gen-cpmd-uks-ot` flip gates. Fix the remaining unordered reductions; `test_cpu_gpw_thread_reproducibility` does not cover these paths. |
| 🟡 | `es-cuda-es-tests-e2e` | `tests/cuda/gpw/test_gpw_full_product.cu`, `test_gpw_kpoints.cu`, `test_gpw_stress.cu`, `tests/cuda/planewave/test_pw_cell_stress.cu` (`*CpuTwin`) | Run full CUDA and CPU-oracle SCFs under ctest — iron law 1. Move to `regressions/{gpw,pw}/cuda-*`. |

### Conformance snapshot (at note date)

| Method family | Driver? | Frame `run`/`forces`? | Conforms? |
| --- | --- | --- | --- |
| CPU RHF/UHF/RKS/UKS/ROHF/ROKS/GHF | Yes | Yes | Pass (ghost_atoms aside) |
| CUDA RHF | Yes | Yes | Pass |
| CUDA UHF | Yes | Yes (`run`/`forces`, STO-3G l ≤ 1; else throws) | Pass |
| AM1 / RM1 | Yes (`NddoMethod`) | Yes (`run`/`forces(Frame&)`; both `QmEngine`) | Pass (evaluator unwrap still `es-bind-dialects`) |
| CPU PW | Yes | Yes (`run`/`forces`/`stress(Frame&)`) | Pass (evaluator 3-SCF aside) |
| CUDA PW | No | `run` yes | Fail (`es-cuda-pw-driver`) |
| CPU GpwDft | Yes (OT residual) | Yes (`run`/`forces`/`stress(Frame&)`) | Pass (OT aside) |
| CUDA GpwDft | No | No (`internal::GpwAtom`) | Fail (`es-cuda-gpw-frame-io`) |
| `*Evaluator` wrappers | Component | launch → Frame | Pass as bind shell only |
| `CarParrinelloMethod` / `XlBomdMethod` | Composed `DriverNode` | Owned method via `method()` | Pass (no concrete-method inheritance) |

### Close order (recommended)

1. `es-bind-dialects` (one outer bind story)
2. `es-cuda-pw-driver` / `es-cuda-gpw-frame-io`
   (Driver + Frame shape on CUDA; CUDA RHF is already `cuda::Driver<scf::Phase>`)
3. `es-kind-model` + QmEngine coverage
4. Hygiene: `es-posthf-ks-gate`, docs, ghost_atoms, OT residual,
   `es-reg-internal`, `es-cuda-es-tests-e2e`
