# Hartree–Fock

Tutorial: [HF](../tutorial/ch02-rhf.md) · [open shell](../tutorial/ch03-open-shell.md) · [basis](../tutorial/ch04-basis-sets.md) · [SCF](../tutorial/ch18-scf-techniques.md).

Shares one SCF solver with [Kohn–Sham DFT](dft.md): the 2×2 product
{restricted, unrestricted} × {HF, KS}. Correlated post-HF lives in
[Post-Hartree–Fock](post-hf.md). Legend: [Roadmap index](index.md).

## 1. Molecular Hartree–Fock

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| RHF | ✓ | ✓ | `cpu::scf::RHF::run(Frame&)` / `run(const Frame&)` / `forces(Frame&)`. ADC, direct RPA, BSE, GF2, and EOM are not members. CUDA `cuda::scf::RHF` : `cuda::Driver<scf::Phase>` (fock, mix, diagonalize, density, check); `run(Frame&)` / `forces(Frame&)` only (no const overload); device Fock/density/CDIIS. STO-3G l ≤ 1: `launch_1e_integrals_sp` / `launch_eri_sp` energy and `launch_rhf_forces_sp` forces. Cartesian l ≤ 3 (6-31G, 6-31G*, 6-31+G*, ragged primitives): `launch_1e_integrals_lgt1` / `launch_eri_lgt1` energy with `with_gradient(false)`; `forces` throws `launch_rhf_forces_lgt1`. Spherical l ≤ 3 (def2-SVP, def2-TZVP, aug-cc-pVDZ) uses those Cartesian integrals and the CPU CartToSph sandwich on device — def2-TZVP's f shell is ten Cartesian GTOs, then the sandwich, not a Cartesian basis. l ≥ 4 throws `launch_eri_g`. No host ERI fill |
| UHF | ✓ | ✓ | `cpu::scf::UHF{basis, n_alpha, n_beta, charge}` : `run(Frame&)` / `run(const Frame&)` / `forces(Frame&)`. CUDA `cuda::scf::UHF{basis, n_alpha, n_beta, charge}` : `cuda::Driver<scf::Phase>` with `run(Frame&)` / `forces(Frame&)` only (no const overload). `run` writes `kEnergy` [eV]; `forces` returns `GradientResult` and writes `kAtom_f*` [eV/Å]. Device Fock, density, CDIIS, and `launch_uhf_forces_sp` for STO-3G l ≤ 1. The device start is the spin-averaged MINAO density (`UHF.init_guess_by_minao`; H, C, N, O, F), so triplet O₂ matches PySCF's default rather than the core-Hamiltonian local minimum; any other element keeps that core start. Outside STO-3G l ≤ 1 the device call throws (`launch_eri_sp`) — no host-ERI fill. `CCSD<UHF>::eom_sf_ccsd`, `SfTDA<UHF>::run`, and `SfTDHF<UHF>::run` are not members |
| ROHF | ✓ | ○ | Guest–Saunders. `ROHF::run` / `forces`. Correlation is `MP2<ROHF>::run`, not a member |
| GHF | ✓ | ○ | Real two-component. `GHF::run` / `forces` |
| Complex / spinor HF | ✓ | ○ | `cpu::scf::ComplexGHF` : `cpu::HF` (not `MeanFieldMethod`). `ComplexGHF{basis, charge = 0, tol = 1e-10, max_iter = 80}` : `run(Frame&)` / `run(const Frame&)`. Member `x2c` defaults true: spin-orbital X2C-1e (`SpinOrbitalX2CHelper.get_hcore` / `_x2c1e_get_hcore`, `GHF.x2c1e`, `xuncontract=False`); `x2c=false` is non-relativistic complex GHF (block-diagonal T+V). Two-electron integrals stay non-relativistic (`ghf.get_jk` on the dense ERI). `run(Frame&)` writes `kEnergy` and `kEnergyElectronic` [eV]; `energy()` [Ha]; `converged()`; `iterations()`. X2C shells with l > 3 throw `complex_ghf_ao_lgt3`. `provides_force()` is false; `forces(Frame&)` throws `complex_ghf_nuclear_forces` (PySCF `GHF.nuc_grad_method` and `x2c.SCF.nuc_grad_method` are unimplemented). No device path. Four-component Dirac–Coulomb is `cpu::scf::DHF` : `cpu::HF` (not `MeanFieldMethod`). `DHF{basis, charge = 0, tol = 1e-10, max_iter = 80}` : `run(Frame&)` / `run(const Frame&)` (`pyscf.scf.dhf.DHF`: j-adapted spinors, (LL|LL)+(SS|LL); (SS|SS) is not assembled; no Gaunt or Breit). Same Frame writes, `energy()`, `converged()`, and `iterations()`. l > 1 throws `dhf_ao_lgt1`. `provides_force()` is false; `forces(Frame&)` throws `dhf_nuclear_forces`. No device path. Tutorial: [open shell](../tutorial/ch03-open-shell.md) |
| ΔSCF / MOM / IMOM | ✓ | ○ | `UHF::delta_scf(Frame&, from, to, MomStyle)` |
| RHF analytic gradient | ✓ | ✓ | `RHF::forces(Frame&)` and `cuda::scf::RHF::forces`. CUDA forces are STO-3G l ≤ 1 (`launch_rhf_forces_sp`). Any other basis throws `launch_rhf_forces_lgt1` (`with_gradient(false)` keeps that energy on device) |
| UHF / ROHF / GHF analytic gradient | ✓ | UHF ✓ | `forces(Frame&)`. CUDA UHF uses `launch_uhf_forces_sp` on STO-3G l ≤ 1 (same envelope as the energy). ROHF and GHF gradients stay on the CPU |
| RHF / UHF analytic Hessian | ✓ | ○ | CPHF: `Hessian<RHF>::run(Frame&)` / `Hessian<UHF>::run(Frame&)` (`pyscf.hessian.rhf` / `uhf`) |
| ROHF Hessian | ✓ | ○ | `Hessian<ROHF>::run(Frame&)` — `central_difference_nuclear_hessian` of analytic `forces` (not CPHF; PySCF has no `hessian.ROHF`) |
| RHF static polarizability | ✓ | ○ | `Polarizability<RHF>::run()` after `RHF::run` (no Frame). Not an SCF method |
| Stability analysis | ✓ | ○ | `Stability<RHF>::run` after `RHF::run` (Seeger–Pople). Not an SCF method |

CUDA HF in `mixed` and `double` modes uses FP64 integral recurrences,
Fock/density contractions, DIIS history and eigensolver storage. The molecular
force outlet follows the Frame's `real_t` storage. Mixed-mode H₂ RHF/UHF and
H₂O RHF energy/gradient pins use the same scientific thresholds as double;
classical MD force and FFT work precision remains independent of this SCF tier.

### Gaps

- CUDA Hartree–Fock energy is **RHF** on shells with l ≤ 3 (`launch_1e_integrals_sp` / `launch_eri_sp` for uniform STO-3G s/p; `launch_1e_integrals_lgt1` / `launch_eri_lgt1` otherwise, including the CPU CartToSph sandwich on device for spherical def2-SVP, def2-TZVP, and aug-cc-pVDZ) and **UHF** on STO-3G l ≤ 1. No KS, ROHF, GHF, or complex spinor HF on device. No host ERI fill. `cuda::scf::UHF` throws outside STO-3G l ≤ 1 (`launch_eri_sp`, `launch_uhf_forces_sp`). `cuda::scf::RHF` throws `launch_eri_g` for l ≥ 4. `cuda::scf::RHF::forces` throws `launch_rhf_forces_lgt1` outside STO-3G l ≤ 1.
- ROHF: SOSCF / level shift still throw. GHF: `MP2` / `FCI` / soscf throw.
- Post-HF, continuum solvent, and CPHF response are not `MeanFieldMethod` members. SCF entries are `run(Frame&)` / `run(const Frame&)` / `forces(Frame&)`. Beside that: `ADC<RHF>::adc2` / `adc2x` / `ip_adc2` / `ea_adc2`, `DirectRPA<RHF>::run`, `BSE<RHF>::run`, `GF2<RHF>::run`, `G0W0<RHF>::run`, `Stability<RHF>::run`, `TDA<RHF>::run`, `TDHF<RHF>::run`, `CISD<RHF>::run`, `CCSD<RHF>::eom_ee_ccsd` / `eom_ip_ccsd` / `eom_ea_ccsd`, `CC2<RHF>::eom_cc2`, `CCSD<UHF>::eom_sf_ccsd`, `SfTDA<UHF>::run`, `SfTDHF<UHF>::run`, `MP2<Method>::run`, `WithSolvent<Method>::run`, `Hessian<RHF|UHF|ROHF>::run`, `Polarizability<RHF>::run`.

## 2. Periodic Hartree–Fock

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| Γ-point periodic RHF | ✓ | ○ | Truncated lattice; `RHF.periodic`; short-range erfc gate |
| k-point periodic RHF | ✓ | ✓ | `cpu::scf::KRHF` : `cpu::HF` and `cuda::scf::KRHF` : `cuda::HF` (not `MeanFieldMethod`). `KRHF{basis, k_mesh, charge = 0, tol = 1e-9, max_iter = 40}` : `run(Frame&)` / `run(const Frame&)`. Orthorhombic cell, s functions, Monkhorst–Pack mesh, FFT Coulomb/exchange with Ewald exchange divergence. `run(Frame&)` writes `kEnergy` and `kEnergyElectronic` [eV]; `energy()` [Ha]; `converged()`; `iterations()`. Device step is J/K, Fock, `zheevd`, density. l > 0 throws (`krhf_ao_lgt0`). `provides_force()` is false; `forces(Frame&)` throws (`krhf_nuclear_forces`). Tutorial: [k-point HF](../tutorial/methods/periodic-kpoint-hf.md) |
| k-point UHF periodic HF | ✓ | ○ | `cpu::scf::KUHF{basis, k_mesh, n_alpha, n_beta}.run(Frame&)`. Same orthorhombic s-function FFTDF envelope as KRHF. `n_alpha` / `n_beta` are mesh totals (`KUHF.nelec`; spin is not scaled by the number of k-points). l > 0 throws (`krhf_ao_lgt0`). Analytic forces throw (`kuhf_nuclear_forces`). No device path |

PW/GPW product paths are semi-local DFT — see [DFT § Periodic](dft.md#4-periodic-kohnsham).

## 3. SCF infrastructure

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| DIIS | ✓ | ✓ | Device CDIIS |
| Linear damping | ✓ | ○ | |
| Level shift (Saunders–Hillier) | ✓ | ✓ | |
| SOSCF | ✓ | ○ | |
| Canonical orthogonalization | ✓ | ○ | |
| Hcore / MINAO guess | ✓ | ✓ / ○ | MINAO: H, C, N, O, F only; else Hcore |
| ASPC density predictor | ✓ | ○ | AIMD warm start |
| ⟨S²⟩ analysis | ✓ | ○ | |
| Spherical (pure) AO | ✓ | RHF ✓ | RHF energy, l ≤ 3 (def2-SVP, def2-TZVP, aug-cc-pVDZ): `launch_1e_integrals_lgt1` / `launch_eri_lgt1` plus the CPU CartToSph sandwich on device. Not a native spherical ERI. `cuda::scf::UHF` stays STO-3G l ≤ 1 (`launch_eri_sp`). l ≥ 4 throws `launch_eri_g` |
| RI-J / RI-JK | ✓ | ○ | def2-universal JKFIT; SCF only |
| CFMM-J | ✓ | ○ | `Coulomb::CFMM` |
| Native Rys ERI (l ≤ 3) | ✓ | ○ | s/p/d/f |
| Device ERI | — | ✓ | STO-3G l ≤ 1: `launch_eri_sp`. Cartesian s/p/d/f, l ≤ 3: `launch_eri_lgt1`. Spherical l ≤ 3 is that tensor plus CartToSph. l ≥ 4 throws `launch_eri_g` |
| ECP | ✓ | ○ | Semi-numerical forces |
| x2c-1e | ✓ | ○ | `cpu::scf::RHF::forces(Frame&)` with `RHF::x2c = true` returns `GradientResult` and writes `kAtom_fx/fy/fz` [eV/Å] and `kEnergy` [eV] (`energy()` stays [Ha]). Molecular RHF: H2/STO-3G, H2O/6-31G, HCl/6-31G. An isolated atom is `RHF::run` only (Ar/def2-TZVP). `Hessian<RHF>::run` and `Hessian<UHF>::run` throw for `x2c=true`. No `x2c` on `cuda::scf::RHF`. Tutorial: [basis sets](../tutorial/ch04-basis-sets.md#nuclear-forces) |
| Diagonalization | ✓ | ✓ | LAPACK / cuSOLVER |

**Bases:** STO-3G, 6-31G, 6-31G\*, 6-31+G\*, def2-SVP, def2-TZVP, aug-cc-pVDZ, cc-pVDZ.
STO-3G and the Pople sets are Cartesian; def2-SVP, def2-TZVP, aug-cc-pVDZ, and
cc-pVDZ are spherical. cc-pVDZ is tabulated for hydrogen only.
