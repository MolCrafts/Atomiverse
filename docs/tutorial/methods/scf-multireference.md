# Configuration interaction and multireference methods

> The regression runner has been removed. Regression commands below are historical; case sources and oracle records are retained as archived evidence, not current executable checks.

Configuration-interaction and active-space methods built on a closed-shell RHF
reference: Head–Gordon CIS(D) excitation energies, complete active-space CI
(CASCI), complete active-space SCF (CASSCF) and strongly contracted NEVPT2 on
a CASSCF reference. Each is a concrete class template wrapping the RHF it
correlates (`post_hf.h`, `cisd.h`), mirroring the PySCF object of the same
name: `cpu::scf::CISD<RHF>{rhf}.run(n_states)` (after `rhf.run(frame)`),
`cpu::scf::CASCI<RHF>{rhf}.run(frame, ncas, nelecas)`,
`cpu::scf::CASSCF<RHF>{rhf, ncas, nelecas}` —
`cpu::Driver<mc::Phase>` (`rotate → casci → gradient → check`;
[Call](#call)),
`cpu::scf::NEVPT2<RHF>{rhf}.run(frame, ncas, nelecas)` and
`cpu::scf::FCI<RHF>{rhf}.run(frame)`. The active-space entries drive their own
SCF; CASCI, CASSCF and FCI share one dense determinant-CI engine.

The sections cover CPU only and energies only: CIS(D) on formaldehyde, the
full-valence CAS(2e,2o) family on H₂, and CASSCF(2,2) on H₂O (core and one
virtual, so the orbital step runs), all in STO-3G.

### Call

`cpu::scf::CASSCF<RHF>` is `cpu::Driver<mc::Phase>`, the same preset shape as
`cpu::md::MD`. The constructor mounts `rotate`, `casci`, `gradient`, and
`check` with `on(phase).add`. max |g| is published on
`ConvergenceMetrics::residual`. `OrbitalGradientConverged` decides. The method
does not assign convergence. `ncas`, `nelecas`, `orb_tol`, and `max_macro` are
constructor arguments and are const after construction.
`CASSCF{mf, ncas, nelecas}` sets `orb_tol` to `max(mf.tol, 1e-8)` [Hartree]
and `max_macro` to 50. The explicit overload is
`CASSCF{mf, ncas, nelecas, orb_tol, max_macro}`. The reference is not owned
and must outlive the CASSCF object. There is no `HF` tag and no CUDA path.
Closed-shell RHF only. There is no `forces(Frame&)`.

Frame in is positions [Å] and species in `kAtom_type`. `run(Frame&)` drives
the reference SCF, then overwrites `kEnergy` with the correlated total [eV].
Hartree stays on `CASSCFResult` (`e_casscf`, `e_corr`). `run(const Frame&)`
writes nothing. `provides_energy()` and `requires_full_frame()` are true;
`provides_force()` is false, so `kAtom_f*` are not written. An outer driver
that mounts CASSCF with `on(phase).add` accumulates `kEnergy` and refuses a
force request.

```cpp
cpu::Frame frame = /* positions [Å], species in kAtom_type */;
cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G, /*charge=*/0, /*tol=*/1e-12};
cpu::scf::CASSCF<cpu::scf::RHF> cas{rhf, /*ncas=*/2, /*nelecas=*/2};
// orb_tol = max(rhf.tol, 1e-8) [Hartree], max_macro = 50
const cpu::scf::CASSCFResult energy = cas.run(frame); // overwrite kEnergy [eV]
double e_ev = frame.meta.get(kEnergy);                 // correlated total [eV]
double e_ha = energy.e_casscf;                         // Hartree
bool ok = cas.converged();                             // Criterion verdict
```

Case: `ctest -R '^scf/multireference$'` — sources and oracle record in [`regressions/scf/multireference/`](../../../regressions/scf/multireference/). One section of the case per `##` heading below: `atv_regressions --case scf/multireference --section <name>`.

## Common setup

### Active-space conventions

Spatial MOs are partitioned into a doubly occupied inactive core, an active
space of `ncas` orbitals holding `nelecas` electrons, and external virtuals.
The active-space CI is dense (full determinant basis, same Slater–Condon /
FCI kernel as `FCI<RHF>`, with a frozen-core energy shift). Closed-shell RHF
reference only: `nelecas` must be even (equal α/β in the active space); no
UHF/ROHF/GHF path. Results are in Hartree.

The H₂ sections use H(0,0,0), H(0,0,0.74) Å (`pin_scf` `h2` preset; Z=1;
`Box::cubic(20)`), 2 AO, full-valence CAS(2e,2o) with `ncore=0`. On this
system CASCI, CASSCF, NEVPT2, FCI and CCSD all coincide to \(\sim 10^{-14}\) Ha
in the basis, which the sections use as cross-method identities:

```cpp
cpu::Frame frame(2);   // H(0,0,0), H(0,0,0.74) Å; Z=1; Box::cubic(20)
cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G};
rhf.tol = 1e-12;
rhf.max_iter = 100;

const cpu::scf::CASSCFResult cas =
    cpu::scf::CASSCF<cpu::scf::RHF>{rhf, /*ncas=*/2, /*nelecas=*/2}.run(frame);
// cas.e_casscf, cas.e_corr [Ha]; cas.ncas, cas.nelecas, cas.ncore
// cas.iterations, cas.converged; rhf.energy() = E_RHF [Ha]
```

Build with `-DATV_BUILD_HF=ON -DATV_BUILD_REGRESSIONS=ON` (CPU login-shell
recipe in `docs/developer/project-contract.md`, `module load buildenv-gcc/2026.03-mpich`).

### Oracle

PySCF 2.14.0, offline pins only (PySCF never runs in ctest); protocol in
`docs/tutorial/appendix-pyscf-oracle.md`. Common settings: STO-3G with
`cart=True` (engine-neutral), RHF `conv_tol=1e-12`. Shared H₂ pins:
\(E_{\mathrm{RHF}}\) = −1.116759307396426 Ha (diagnostic, \(10^{-6}\) Ha) and
the full-valence total −1.137283834488502 Ha. Repro (human / offline only):

```bash
bash <pyscf env setup>
python <pyscf pin> <casci|casscf|nevpt2> \
    --geom h2 --basis sto-3g --ncas 2 --nelecas 2 --conv-tol 1e-12
```

Fair wall: login node, `OMP_NUM_THREADS=1`, median of ≥5 runs, gate
\(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}} \le 10\) (no ⚡ optimization pass
required):

```bash
export OMP_NUM_THREADS=1
python <pyscf fair-bench> --method <method> --geom h2 --basis sto-3g --repeat 5 -- \
    <section binary>
```

The H₂ twins are startup-dominated (ATV faster than PySCF, \(r \ll 1\)) on a
2-orbital active space with no non-redundant rotations, so \(r\) is not a
reliable large-CAS / large-virtual-space predictor.

### Shared limitations

- **Energy only** on closed-shell RHF; no state averaging (SA-CASSCF), no
  public 1-/2-RDM export, no analytic gradients/Hessians.
- **Dense FCI** active-space kernel only — exponential in active size; no
  RAS/GAS, DMRG or selected-CI solvers.
- Canonical MO integrals; no DF/CD/local CAS.
- **CUDA / Kokkos:** ○ (CPU only).
- Open roadmap axes (`docs/roadmap/post-hf.md` §4,
  [Post-HF §4.1 Active-space methods](../../roadmap/post-hf.md), §4.2
  Multireference correlation): SA-CASSCF, CASPT2, PC/QD-NEVPT2, CASSCF/NEVPT2
  analytic gradients.

### References

- B. O. Roos, *The Complete Active Space Self-Consistent Field Method and
  its Applications in Electronic Structure Calculations*, Adv. Chem. Phys.
  **69**, 399 (1987). DOI
  [10.1002/9780470142943.ch7](https://doi.org/10.1002/9780470142943.ch7)

## cisd-h2co-sto3g — CIS(D), H₂CO / STO-3G {#cisd-h2co-sto3g}

CIS (RHF-TDA) is \(O(N^4)\) but misses dynamical correlation in the excited
state. Head–Gordon CIS(D) adds a second-order doubles correction \(U+V\) on
each closed-shell CIS root (Head–Gordon et al., *Chem. Phys. Lett.* **219**, 21
(1994)). `CISD<RHF>` returns `CISDResult` with `cis_energy_ev`,
`doubles_correction_ev` and `energy_ev` (CIS + correction) per root [eV]:

```cpp
cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G};
rhf.tol = 1e-12;
rhf.max_iter = 200;
rhf.run(frame);   // C(0,0,0), O(0,0,1.21), H(0,±0.943,−0.587) Å; Box::cubic(30)
const cpu::scf::CISDResult cisd = cpu::scf::CISD<cpu::scf::RHF>{rhf}.run(/*n_states=*/1);
```

### Oracle

PySCF has no built-in CIS(D); PySCF supplies RHF + TDA amplitudes and
integrals, and the pin is the same spin-orbital Head–Gordon \(U+V\) formula
evaluated on that reference (not experiment). The section also gates
\(E_{\mathrm{RHF}}\) to 1e-6 Ha.

| Quantity | Pin | Tol |
| -------- | --: | --- |
| CIS S₁ | 4.278299 eV | 1e-4 eV |
| **CIS(D) S₁** | **4.544577 eV** | **1e-4 eV** |

Strict ADC(2) S1 on the same geometry/basis is 4.535 eV
([adc2-h2co-sto3g](scf-adc.md#adc2-h2co-sto3g)).

### Performance

| Twin | \(T_{\mathrm{PySCF}}\) | \(T_{\mathrm{ATV}}\) | \(r\) | ≤10× |
| ---- | ---------------------: | -------------------: | ----: | ---- |
| `tda` (CIS twin) | 0.89 s | 0.22 s | **0.24** | ✓ |

The ATV path includes CIS(D) while the twin is TDA-only, so the ratio is an
optimistic upper bound on speed.

## casscf-h2-sto3g — CASSCF, H₂ / STO-3G {#casscf-h2-sto3g}

CASSCF optimizes the orbitals as well as the active-space CI coefficients
(unlike CASCI, which freezes them at RHF):

\[
\lvert\Psi_{\mathrm{CAS}}\rangle
  = \sum_{I \in \mathrm{CAS}} c_I \lvert\Phi_I(\mathbf{C})\rangle,\qquad
E_{\mathrm{CASSCF}}
  = \min_{\mathbf{c},\mathbf{C}}
  \frac{\mathbf{c}^\dagger H(\mathbf{C})\,\mathbf{c}}
       {\mathbf{c}^\dagger\mathbf{c}},
\quad
E_{\mathrm{corr}} = E_{\mathrm{CASSCF}} - E_{\mathrm{RHF}}.
\]

CI coefficients \(\mathbf{c}\) come from the dense active-space FCI kernel;
orbital coefficients \(\mathbf{C}\) are updated by a two-step Newton
macro-loop on the orbital gradient built from spin-summed 1- and 2-RDMs until
the generalized Fock is symmetric to tolerance (Roos 1987). State-specific,
single root. The macro iteration is `cpu::Driver<mc::Phase>` (`rotate →
casci → gradient → check`); `OrbitalGradientConverged` decides.
`CASSCF{rhf, ncas, nelecas}.run(frame)` overwrites `kEnergy` [eV] and returns
`CASSCFResult{e_casscf, e_corr, ncas, nelecas, ncore, iterations, converged}`
[Hartree] (`casscf.h`, `post_hf.h`; implementation `src/cpu/scf/casscf.cpp`). Twin:
`pyscf.mcscf.CASSCF(mf, ncas=2, nelecas=2)`, `conv_tol=1e-12`.

On full-valence CAS(2,2) there are no non-redundant orbital rotations
(`ncas = n_mo`), so CASSCF equals CASCI / FCI to \(\sim 10^{-14}\) Ha and one
CI-only macro pass reports `converged`.

### Pins and gates

| Quantity | Pin | Tol | Notes |
| -------- | --: | --- | ----- |
| \(E_{\mathrm{RHF}}\) | −1.116759307396426 Ha | \(10^{-6}\) Ha | Diagnostic |
| \(E_{\mathrm{corr}}\) (CASSCF) | **−0.020524527092077 Ha** | **\(10^{-6}\) Ha** | Product gate |
| \(E_{\mathrm{tot}}\) (CASSCF) | **−1.137283834488502 Ha** | **\(10^{-6}\) Ha** | Primary product gate |
| Bookkeeping | `ncas=2`, `nelecas=2`, `ncore=0`, `converged` | exact | Product gate |
| Consistency | \(E_{\mathrm{tot}} = E_{\mathrm{RHF}} + E_{\mathrm{corr}}\) | \(10^{-12}\) Ha | Regression check |
| CASSCF ≡ CASCI | \(\lvert E_{\mathrm{CASSCF}} - E_{\mathrm{CASCI}}\rvert\) | \(10^{-10}\) Ha | Full-valence identity |

### Performance

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) |
| ---- | -------------------------: | -----------------------: | ----: |
| `casscf` (H₂/STO-3G) | 0.2112 | 0.0092 | **0.0435** |


## casscf-h2o-sto3g — CASSCF, H₂O / STO-3G {#casscf-h2o-sto3g}

Same product as [casscf-h2-sto3g](#casscf-h2-sto3g), on water. STO-3G has 7
MOs and RHF has 5 occupied spatial orbitals, so CAS(2,2) freezes `ncore = 4`
and leaves `nvirt = 1`. Core–active, active–virtual and core–virtual pairs
are non-redundant: the macro iteration applies an orbital step, and the
CASSCF energy lies below CASCI on the RHF orbitals. `iterations >= 2`.

Twin: `pyscf.mcscf.CASSCF(mf, ncas=2, nelecas=2)`, `conv_tol=1e-12`, geometry
the `pin_scf` `water` preset.

### Pins and gates

| Quantity | Pin | Tol | Notes |
| -------- | --: | --- | ----- |
| \(E_{\mathrm{RHF}}\) | −74.96302313846292 Ha | \(10^{-6}\) Ha | Diagnostic |
| \(E_{\mathrm{corr}}\) (CASSCF) | **−0.0012536321488170188 Ha** | **\(10^{-6}\) Ha** | Product gate |
| \(E_{\mathrm{tot}}\) (CASSCF) | **−74.96427677061173 Ha** | **\(10^{-6}\) Ha** | Primary product gate |
| Bookkeeping | `ncas=2`, `nelecas=2`, `ncore=4`, `iterations >= 2`, `converged` | exact | Orbital step ran |
| Consistency | \(E_{\mathrm{tot}} = E_{\mathrm{RHF}} + E_{\mathrm{corr}}\) | \(10^{-12}\) Ha | Regression check |
| Frame | `kEnergy` [eV] equals \(E_{\mathrm{tot}}\times\) Ha→eV | \(10^{-8}\) eV | `run(Frame&)` overwrite |

## nevpt2-h2-sto3g — SC-NEVPT2, H₂ / STO-3G {#nevpt2-h2-sto3g}

`NEVPT2<RHF>::run` drives RHF → state-specific CASSCF → strongly contracted
NEVPT2 (Angeli et al. 2001, 2002): second-order *n*-electron valence
perturbation theory with a Dyall zeroth-order Hamiltonian on
\(\lvert\Psi^{(0)}\rangle = \lvert\Psi_{\mathrm{CAS}}\rangle\), with core
\(i,j\), active \(t,u\) and external \(r,s\) indices. Each of the eight
excitation classes (Sr, Si, Sij, Srs, Sir, Sijr, Srsi, Sijrs) contributes one
contracted function per class label:

\[
E_{\mathrm{corr}}^{\mathrm{SC\text{-}NEVPT2}}
  = \sum_{\mu}
  \frac{
    \lvert\langle\Psi^{(0)}\lvert H \rvert \Psi_{\mu}^{(1)}\rangle\rvert^{2}
  }{
    E^{(0)} - E_{\mu}^{(0)}
  },
\qquad
E_{\mathrm{NEVPT2}}
  = E_{\mathrm{CASSCF}} + E_{\mathrm{corr}}^{\mathrm{SC\text{-}NEVPT2}}.
\]

`e_corr` is the PT2 correction **vs CASSCF** (PySCF `mrpt.NEVPT.e_corr`), not
vs HF. Result: `NEVPT2Result{e_nevpt2, e_corr, e_casscf, ncas, nelecas,
ncore, nvirt, converged}` (`nevpt2.h`; private pack `internal/nevpt2_pack.h`;
class contributions in `src/cpu/scf/nevpt2.cpp`). Twin:
`pyscf.mrpt.NEVPT(mc).kernel()` (SC default) on
`pyscf.mcscf.CASSCF(mf, ncas=2, nelecas=2)`, `conv_tol=1e-12`. Strongly
contracted only — no PC-NEVPT2, QD-NEVPT2 or multi-state NEVPT2; no CASPT2
product path (no stock PySCF oracle; OpenMolcas / external).

### Full-valence identity

With CAS(2e,2o) on H₂/STO-3G there are no core or virtual orbitals outside the
CAS:

\[
n_{\mathrm{core}} = 0,\quad n_{\mathrm{cas}} = 2,\quad n_{\mathrm{virt}} = 0.
\]

Every SC class needs at least one core or external index, so each class
contribution is zero by construction: \(E_{\mathrm{corr}} = 0\) and
\(E_{\mathrm{NEVPT2}} = E_{\mathrm{CASSCF}} = E_{\mathrm{FCI}}\). The section
validates the product pipeline and bookkeeping, not dynamic-correlation
recovery on a large virtual space (the non-trivial Sr/Si/… class kernels are
not exercised).

### Pins and gates

| Quantity | Pin | Tol | Notes |
| -------- | --: | --- | ----- |
| \(E_{\mathrm{RHF}}\) | −1.116759307396426 Ha | \(10^{-6}\) Ha | Diagnostic |
| \(E_{\mathrm{CASSCF}}\) | −1.137283834488502 Ha | \(10^{-6}\) Ha | Zeroth-order reference |
| \(E_{\mathrm{corr}}\) (SC-NEVPT2 PT2) | **0.0** Ha | **\(10^{-6}\) Ha** | Primary product gate |
| \(E_{\mathrm{tot}}\) (NEVPT2) | **−1.137283834488502 Ha** | **\(10^{-6}\) Ha** | Product gate |
| Bookkeeping | `ncas=2`, `nelecas=2`, `ncore=0`, `nvirt=0`, `converged` | exact | Product gate |
| Consistency | \(E_{\mathrm{tot}} = E_{\mathrm{CASSCF}} + E_{\mathrm{corr}}\) | \(10^{-12}\) Ha | Regression check |
| NEVPT2 ≡ CASSCF | \(\lvert E_{\mathrm{NEVPT2}} - E_{\mathrm{CASSCF}}\rvert\) | \(10^{-10}\) Ha | Full-valence identity |

### Performance

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) |
| ---- | -------------------------: | -----------------------: | ----: |
| `nevpt2` (H₂/STO-3G) | 0.1725 | 0.0125 | **0.0723** |

\(r \approx 0.072\); the PT2 correction is identically zero here.

### References

- C. Angeli, R. Cimiraglia, S. Evangelisti, T. Leininger, J.-P. Malrieu,
  *Introduction of n-electron valence states for multireference perturbation
  theory*, J. Chem. Phys. **114**, 10252 (2001). DOI
  [10.1063/1.1361246](https://doi.org/10.1063/1.1361246)
- C. Angeli, R. Cimiraglia, J.-P. Malrieu, *n-electron valence state
  perturbation theory: A spinless formulation and an efficient implementation
  of the strongly contracted and of the partially contracted variants*,
  J. Chem. Phys. **117**, 9138 (2002). DOI
  [10.1063/1.1515317](https://doi.org/10.1063/1.1515317)

## casci-h2-sto3g — CASCI, H₂ / STO-3G {#casci-h2-sto3g}

*No section of the case runs this variant.*

CASCI expands all determinants with `nelecas` active electrons in the active
orbitals, with orbitals frozen at RHF (no orbital optimization):

\[
\lvert\Psi_{\mathrm{CASCI}}\rangle
  = \sum_{I \in \mathrm{CAS}} c_I \lvert\Phi_I\rangle,\qquad
E_{\mathrm{CASCI}} = \min_{\mathbf{c}}
  \frac{\mathbf{c}^\dagger H \mathbf{c}}{\mathbf{c}^\dagger\mathbf{c}},
\quad
E_{\mathrm{corr}} = E_{\mathrm{CASCI}} - E_{\mathrm{RHF}}.
\]

`CASCI<RHF>{rhf}.run(frame, ncas, nelecas)` returns `CASCIResult{e_casci,
e_corr, ncas, nelecas, ncore}` (`casci.h`; private pack
`internal/casci_pack.h`; kernel `src/cpu/scf/fci.cpp`). Twin:
`pyscf.mcscf.CASCI(mf, ncas=2, nelecas=2)` on the same RHF orbitals.

The PySCF CASCI(2,2) total below is gated by section
`ci-engine-equiv-h2-sto3g` of this case, which runs `FCI<RHF>` and
`CASCI<RHF>(2,2)` on H₂/STO-3G and checks both against the PySCF full-CI pin
(\(10^{-6}\) Ha), the bookkeeping `ncas=2`, `nelecas=2`, `ncore=0`, and FCI
against CASCI (\(10^{-12}\) Ha, same full space and same engine).

### Recorded values

| Quantity | Pin | Tol | Notes |
| -------- | --: | --- | ----- |
| \(E_{\mathrm{RHF}}\) | −1.116759307396426 Ha | \(10^{-6}\) Ha | Diagnostic |
| \(E_{\mathrm{corr}}\) (CASCI) | **−0.020524527092077 Ha** | **\(10^{-6}\) Ha** | Recorded |
| \(E_{\mathrm{tot}}\) (CASCI) | **−1.137283834488502 Ha** | **\(10^{-6}\) Ha** | Gated by `ci-engine-equiv-h2-sto3g` |
| Bookkeeping | `ncas=2`, `nelecas=2`, `ncore=0` | exact | Gated by `ci-engine-equiv-h2-sto3g` |
| Consistency | \(E_{\mathrm{tot}} = E_{\mathrm{RHF}} + E_{\mathrm{corr}}\) | \(10^{-12}\) Ha | Recorded |

### Performance

Recorded fair wall (`--method casci --repeat 3`):

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) |
| ---- | -------------------------: | -----------------------: | ----: |
| `casci` (H₂/STO-3G) | 0.1427 | 0.0109 | **0.076** |
