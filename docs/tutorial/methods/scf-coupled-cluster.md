# Coupled cluster (CCSD, CCSD(T), CC2)

Coupled-cluster theory writes the correlated wavefunction as an exponential
of excitation operators on a single closed-shell RHF determinant
\(\lvert\Phi_0\rangle\):

\[
\lvert\Psi_{\mathrm{CCSD}}\rangle = e^{T_1+T_2}\lvert\Phi_0\rangle,\qquad
E_{\mathrm{CCSD}} = \langle\Phi_0\rvert H e^{T}\lvert\Phi_0\rangle_{\mathrm{c}},
\quad
E_{\mathrm{corr}} = E_{\mathrm{CCSD}} - E_{\mathrm{RHF}}.
\]

The public entry points are post-HF objects that bind a concrete RHF and drive
a fresh SCF on it (`cpu/scf/post_hf.h`, mirroring `pyscf.cc.CCSD(mf)`):

- `cpu::scf::CCSD<RHF>{rhf, diis, frozen, tol, max_iter}` —
  `cpu::Driver<cc::Phase>` (`update → mix → energy → check`). See
  [Call](#call). The EOM members are documented in
  [scf-eom-cc.md](scf-eom-cc.md).
- `cpu::scf::CC2<RHF>{rhf}` — `run(frame)` → `CC2Result`; `eom_cc2` roots are
  [`eom-cc2-h2`](scf-eom-cc.md#eom-cc2-h2).

### Call

`cpu::scf::CCSD<RHF>` is `cpu::Driver<cc::Phase>`, the same preset shape as
`cpu::md::MD`. The constructor mounts `update`, `mix`, `energy`, and
`check` with `on(phase).add`. max|Δt| is published on
`ConvergenceMetrics::residual`. `AmplitudeConverged` decides. The method
does not assign convergence. `diis`, `frozen`, `tol`, and `max_iter` are
constructor arguments (`true`, `0`, `1e-10`, `100`) and are const after
construction. `diis = false` is Jacobi, the same fixed point. The reference
is not owned and must outlive the CCSD object. There is no `HF` tag and no
CUDA path. Closed-shell RHF only; `frozen != 0` throws on `forces`.

Frame in is positions [Å] and species in `kAtom_type`. `run(Frame&)` drives
the reference SCF, then overwrites `kEnergy` with the correlated total [eV].
Hartree stays on `CCSDResult`. `run(const Frame&)` writes nothing.
`forces(Frame&)` overwrites `kAtom_fx` / `kAtom_fy` / `kAtom_fz` [eV/Å] and
`kEnergy` [eV], and returns `GradientResult` (forces [eV/Å], energy [eV]).
`ccsd_t(Frame&)` returns `CCSDTResult` [Hartree]. An outer driver that
mounts CCSD with `on(phase).add` accumulates `kEnergy`, and accumulates
`kAtom_f*` when the parent requests forces.

```cpp
cpu::Frame frame = /* positions [Å], species in kAtom_type */;
cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G, /*charge=*/0, /*tol=*/1e-12};
cpu::scf::CCSD<cpu::scf::RHF> ccsd{rhf}; // diis true, frozen 0, tol 1e-10, max_iter 100
const cpu::scf::CCSDResult energy = ccsd.run(frame); // overwrite kEnergy [eV]
double e_ev = frame.meta.get(kEnergy);               // correlated total [eV]
double e_ha = energy.e_tot;                          // Hartree
bool ok = ccsd.converged();                          // Criterion verdict

const cpu::scf::GradientResult grad = ccsd.forces(frame);
// grad.forces [eV/Å], grad.energy [eV]; frame kAtom_f* and kEnergy overwritten
```

`CCSD::run` solves the closed-shell **spatial-orbital** amplitude equations
(\(n_o^2 n_v^2\) storage) with amplitude DIIS; `ccsd_t` and `CC2::run` use the
**spin-orbital** Gauss–Stanton kernel (Stanton et al. 1991; Gauss & Stanton
1995 Table III; occupied spin-orbitals α then β over spatial occupied MOs,
virtuals likewise). The `ccsd-spatial-so-parity` section of the case (H₂O /
STO-3G; oracle record in the case `REFERENCE.md`) gates the spatial path
against the spin-orbital kernel and against plain Jacobi to ≤ 1e-9 Ha, and
against the PySCF pin to ≤ 1e-6 Ha. All sections run on CPU.

Case: `ctest -R '^scf/coupled-cluster$'` — sources and oracle record in [`regressions/scf/coupled-cluster/`](../../../regressions/scf/coupled-cluster/). One section of the case per `##` heading below: `atv_regressions --case scf/coupled-cluster --section <name>`.

## Common setup and oracle

- **Oracle:** offline PySCF **2.14.0** (`docs/tutorial/appendix-pyscf-oracle.md`);
  nothing in ctest imports PySCF. SCF `conv_tol=1e-12`, Cartesian AOs
  (`cart=True`; engine-neutral for STO-3G).
- **H₂:** H(0,0,0); H(0,0,0.74) Å (`pin_scf` `h2` preset), charge 0, singlet,
  STO-3G; 2 AO, 1 occupied + 1 virtual spatial MO; \(T_1 \approx 0\) by
  symmetry.
- **H₂O:** O(0, 0, 0.1173); H(0, ±0.7572, −0.4692) Å (`pin_scf` `water`
  preset), charge 0, singlet, STO-3G; molecule in the yz-plane.
- **Frame:** Z = 1 (H) / 8 (O), `Box::cubic(20)`; RHF `tol = 1e-12`,
  `max_iter = 100` (200 for the gradient).
- **Gates:** energies \(10^{-6}\) Ha vs the pin, `converged == true`, and
  \(E_{\mathrm{tot}} = E_{\mathrm{RHF}} + E_{\mathrm{corr}}\) to
  \(10^{-12}\) Ha.
- **Build:** `-DATV_BUILD_HF=ON -DATV_BUILD_REGRESSIONS=ON` (CPU build
  environment `buildenv-gcc/2026.03-mpich`, see the repository build notes).
- **Fair wall:** \(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}} \le 10\) (login
  node; `OMP_NUM_THREADS=1`; median of ≥3;
  `python <pyscf fair-bench> --method <m> --geom <g> --basis sto-3g --repeat 3 -- <section binary>`).
  These minimal systems are startup-dominated, so \(r\) is not a reliable
  large-\(N\) predictor.
- **Common scope:** single-reference, closed-shell RHF only (open-shell
  throws — no silent fallback); canonical MO integrals; no DF/CD/F12/local CC;
  CUDA / Kokkos ○. Roadmap: [Post-HF §3 Coupled cluster](../../roadmap/post-hf.md),
  `docs/roadmap/excited-states.md`.

## References

- J. F. Stanton, J. Gauss, J. D. Watts, R. J. Bartlett, *J. Chem. Phys.*
  **94**, 4334 (1991). DOI [10.1063/1.460620](https://doi.org/10.1063/1.460620)
- J. Gauss, J. F. Stanton, *J. Chem. Phys.* **103**, 3561 (1995) (Table III
  intermediates)

## ccsd-h2-sto3g — CCSD, H₂ / STO-3G {#ccsd-h2-sto3g}

### Summary

Closed-shell **CCSD** correlation energy on H₂ / STO-3G. On this
two-electron system CCSD coincides with FCI in the basis (to
\(\sim 10^{-11}\) Ha). Fair-bench ratio \(r = 0.317\).

### Scope

- Energy only — no \(\Lambda\) equations, no 1-RDM/2-RDM; forces are
  [`ccsd-grad-h2o-sto3g`](#ccsd-grad-h2o-sto3g), perturbative triples
  [`ccsd-t-h2o-sto3g`](#ccsd-t-h2o-sto3g), EOM-EE
  [`eom-ee-ccsd`](scf-eom-cc.md#eom-ee-ccsd).
- No higher \(T_n\); full MO \(O(N^6)\) path.

### Implementation

```cpp
cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G};
rhf.tol = 1e-12;
rhf.max_iter = 100;

const cpu::scf::CCSDResult ccsd =
    cpu::scf::CCSD<cpu::scf::RHF>{rhf}.run(frame);
// ccsd.e_corr, ccsd.e_tot [Ha]; ccsd.converged; ccsd.iterations
// rhf.energy() = E_RHF [Ha] after the internal RHF
```

| Path | Role |
| ---- | ---- |
| `post_hf.h` | `CCSD<Method>` |
| `ccsd.h` | Public `CCSDResult` |
| `ccsd_pack.h` | Private `ccsd_energy` |
| `src/cpu/scf/ccsd.cpp` | spatial and spin-orbital residual / energy |

### Validation (vs PySCF)

CCSD oracle: `pyscf.cc.CCSD(mf)`, `conv_tol=1e-12`.

| Quantity | Pin | Tol | Notes |
| -------- | --: | --- | ----- |
| \(E_{\mathrm{RHF}}\) | −1.116759307396426 Ha | \(10^{-6}\) Ha | Diagnostic |
| \(E_{\mathrm{corr}}\) (CCSD) | **−0.020524527092106 Ha** | **\(10^{-6}\) Ha** | Gate |
| \(E_{\mathrm{tot}}\) (CCSD) | **−1.137283834488532 Ha** | **\(10^{-6}\) Ha** | Gate |
| Amplitudes | — | `converged == true` | Gate |
| Consistency | \(E_{\mathrm{tot}} = E_{\mathrm{RHF}} + E_{\mathrm{corr}}\) | \(10^{-12}\) Ha | Section check |

Oracle repro (offline only):

```bash
python <pyscf pin> ccsd \
    --geom h2 --basis sto-3g --conv-tol 1e-12
```

### Performance

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | -------------------------: | -----------------------: | ----: | ---- | ------ |
| `ccsd` (H₂/STO-3G) | 0.0482 | 0.0153 | **0.317** | ✓ | Startup-dominated; ATV faster than PySCF (\(r \ll 1\)); not a production-scale CCSD stress test |

### Limitations

- Energy only on closed-shell RHF; no UCCSD, no λ / density, no Hessian.
- Higher CC / local / open-shell remain ○ (`docs/roadmap/post-hf.md`).

## ccsd-grad-h2o-sto3g — CCSD nuclear gradient, H₂O / STO-3G {#ccsd-grad-h2o-sto3g}

### Summary

Closed-shell **CCSD nuclear forces** via `CCSD<RHF>::forces`, which drives a
fresh RHF, then the RHF→CCSD energy path of `CCSD::run`, and returns
`GradientResult{forces, energy}` with forces in **eV/Å** and the total CCSD
energy in **eV** (internal pin math is Ha/Bohr). \(\partial E_{\mathrm{CCSD}}/\partial R\)
is a **central finite difference** of the analytic CCSD total energy
(\(h = 10^{-4}\) Bohr), not the density-based \(\Lambda\) / Lagrangian
analytic path. The oracle is PySCF **analytic** `cc.CCSD` +
`nuc_grad_method().kernel()`. Fair-bench ratio \(r = 1.665\).

### Method

**Gradient.** Central finite difference of the total CCSD energy on nuclear
Cartesian displacements in Bohr:

\[
\frac{\partial E}{\partial R_{A,d}}
\approx
\frac{E(R_{A,d}+h) - E(R_{A,d}-h)}{2h},\qquad h = 10^{-4}\,\mathrm{Bohr}.
\]

Each displaced geometry re-runs the same RHF→`ccsd_energy` path.
`CCSD::forces` negates the energy derivatives into **forces** and converts to
eV/Å for Frame/`GradientResult`; the section converts back to Ha/Bohr for
comparison with the PySCF analytic pin.

**Oracle (PySCF).** Analytic nuclear gradient via the \(\Lambda\) equations
and Lagrangian (Gauss & Stanton 1996; PySCF `pyscf.grad.ccsd`). Element-wise
FD vs analytic agreement on this system is \(\le 4\times 10^{-8}\) Ha/Bohr
(well under the gate).

**Scope.**

- No density-based analytic gradient (\(\Lambda\), Lagrangian 2-RDM,
  Z-vector) — the follow-up for performance / large \(N\).
- No frozen core, no RI/DF-CCSD gradients (`frozen != 0` throws).
- No CCSD(T) forces, no EOM-CCSD gradients, no Hessian.
- Cartesian AO packs only; \(x\) components ~0 (yz-plane molecule).

### Implementation

```cpp
cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G};
rhf.tol = 1e-12;
rhf.max_iter = 200;

const cpu::scf::GradientResult res =
    cpu::scf::CCSD<cpu::scf::RHF>{rhf}.forces(frame);
// res.forces(A, d) [eV/Å]; res.energy [eV] (total CCSD)
// Compare vs pin: -convert(force, eV/Å → Ha/Bohr)
```

`GradientResult` fields (`gradient_result.h`): `forces`
\([n_{\mathrm{atoms}}, 3]\) eV/Å, `energy` eV.

| Path | Role |
| ---- | ---- |
| `post_hf.h` | `CCSD<Method>::forces` (RHF-only gates) |
| `gradient_result.h` | Public `GradientResult` |
| `src/cpu/scf/internal/ccsd_gradient.cpp` | FD of CCSD energy |

### Validation (vs PySCF)

| Setting | Value |
| ------- | ----- |
| CCSD | `pyscf.cc.CCSD(mf)`, `conv_tol=1e-10`; no frozen core |
| Gradient | `mycc.nuc_grad_method().kernel()` (analytic) |
| Command | `pin_scf.py ccsd --geom water --basis sto-3g --grad --conv-tol 1e-12` |

#### Energies (diagnostic / same surface)

| Quantity | Pin | Notes |
| -------- | --: | ----- |
| \(E_{\mathrm{HF}}\) | −74.963023138463 Ha | RHF reference |
| \(E_{\mathrm{corr}}\) (CCSD) | −0.049438562975 Ha | |
| \(E_{\mathrm{CCSD}}\) | −75.012461701437 Ha | \(E_{\mathrm{HF}}+E_{\mathrm{corr}}\) |

#### Analytic gradient pin \({\partial E}/{\partial R}\) [Ha/Bohr]

| Atom | \(\partial E/\partial x\) | \(\partial E/\partial y\) | \(\partial E/\partial z\) |
| ---- | ------------------------: | ------------------------: | ------------------------: |
| O | 0 | 0 | −1.106969560091×10⁻¹ |
| H | 0 | −4.698911449636×10⁻² | +5.534847800455×10⁻² |
| H | 0 | +4.698911449636×10⁻² | +5.534847800455×10⁻² |

\(\sum_A \partial E/\partial R_A \approx (0,0,0)\) to ~10⁻¹⁵ Ha/Bohr.

| Check | Tol | Notes |
| ----- | --- | ----- |
| Element-wise vs PySCF analytic pin | **1e-5 Ha/Bohr** | Gate; FD agrees \(\le 4\times 10^{-8}\) on this system |
| Translational invariance \(\lvert\sum_A g_A\rvert_{\max}\) | **1e-8 Ha/Bohr** | Gate |

Oracle repro (offline only):

```bash
python <pyscf pin> ccsd \
    --geom water --basis sto-3g --grad --conv-tol 1e-12
```

### Performance

Median of 3.

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | -------------------------: | -----------------------: | ----: | ---- | ------ |
| `ccsd_grad` (H₂O/STO-3G) | 0.241 | 0.401 | **1.665** | ✓ | ATV = FD of \(E_{\mathrm{CCSD}}\) (many energy solves); PySCF = analytic \(\Lambda\) path; not a large-\(N\) predictor |

\(r \sim 1.7\) is acceptable on this tiny basis, but the analytic \(\Lambda\)
path is the expected large-\(N\) win for both engines.

### Limitations

- FD gradient: cost scales as \(O(6N_{\mathrm{atoms}})\) full CCSD energy
  evaluations plus SCF.
- Open-shell / RI / frozen-core throw.
- No CCSD(T) forces, no EOM-CC / ADC response gradients, no Hessian.
- The analytic density-based path (Gauss & Stanton 1996; PySCF
  `pyscf.grad.ccsd`) is the follow-up.

### References

- J. Gauss, J. F. Stanton, *J. Chem. Phys.* **104**, 2574 (1996) (analytic
  CCSD gradient)
- T. Helgaker, P. Jørgensen, J. Olsen, *Molecular Electronic-Structure Theory*
  (response / gradient chapters)
- PySCF `pyscf.grad.ccsd`

## ccsd-t-h2o-sto3g — CCSD(T), H₂O / STO-3G {#ccsd-t-h2o-sto3g}

### Summary

Closed-shell **CCSD(T)** correlation energy via `CCSD<RHF>::ccsd_t`: a fresh
RHF, spin-orbital Gauss–Stanton CCSD amplitudes, then the non-iterative (T)
triples correction, returning
`CCSDTResult{e_corr, e_tot, e_corr_ccsd, e_t, iterations, converged}` in
Hartree (`e_corr = E_CCSD + E_(T)`). Oracle: PySCF `cc.CCSD` + `ccsd_t()`.
Fair-bench ratio \(r = 0.333\).

### Method

Non-iterative triples correction \(E_{(T)}\) on top of CCSD (Raghavachari
1989; Watts–Gauss–Bartlett 1993):

\[
E_{\mathrm{corr}}^{\mathrm{CCSD(T)}} = E_{\mathrm{corr}}^{\mathrm{CCSD}} + E_{(T)},
\qquad
E_{\mathrm{tot}} = E_{\mathrm{RHF}} + E_{\mathrm{corr}}^{\mathrm{CCSD(T)}}.
\]

(T) uses the spin-orbital formula of JCP **98**, 8718 (1993) (PySCF
`gccsd_t` path, energy-matched to restricted `ccsd_t()`).

**Scope.**

- Energy only — no \(\Lambda\), no densities, no analytic gradient.
- Non-iterative (T) only; no iterative CCSDT / CC3 / CCSDT-n.
- Full MO \(O(N^7)\) (T) path.
- H₂O/STO-3G: 7 AO; 5 occupied + 2 virtual spatial MO (spin-orbital
  \(n_o=10\), \(n_v=4\)).

### Implementation

```cpp
cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G};
rhf.tol = 1e-12;
rhf.max_iter = 100;

const cpu::scf::CCSDTResult ct =
    cpu::scf::CCSD<cpu::scf::RHF>{rhf}.ccsd_t(frame);
// ct.e_corr = E_CCSD + E_(T); ct.e_corr_ccsd; ct.e_t; ct.e_tot [Ha]
// ct.converged; ct.iterations; rhf.energy() = E_RHF [Ha]
```

`CCSDTResult` fields (`ccsd.h`): `e_corr`, `e_tot`, `e_corr_ccsd`, `e_t`
[Hartree], `iterations`, `converged`. Pack-level `ccsd_t_energy` is private
engine code (`ccsd_pack.h`; kernel in `src/cpu/scf/ccsd.cpp`).

### Validation (vs PySCF)

CCSD oracle `pyscf.cc.CCSD(mf)`, `conv_tol=1e-12`; (T) `mycc.ccsd_t()`.

| Quantity | Pin | Tol | Notes |
| -------- | --: | --- | ----- |
| \(E_{\mathrm{RHF}}\) | −74.963023138462930 Ha | \(10^{-6}\) Ha | Diagnostic |
| \(E_{\mathrm{corr}}\) (CCSD) | −0.049438563031963 Ha | \(10^{-6}\) Ha | Diagnostic |
| \(E_{(T)}\) | −0.000067409682944 Ha | \(10^{-6}\) Ha | Diagnostic |
| \(E_{\mathrm{corr}}\) (CCSD(T)) | **−0.049505972714907 Ha** | **\(10^{-6}\) Ha** | Gate |
| \(E_{\mathrm{tot}}\) (CCSD(T)) | **−75.012529111177841 Ha** | **\(10^{-6}\) Ha** | Gate |
| Amplitudes | — | `converged == true` | Gate |
| Consistency | \(E_{\mathrm{corr}} = E_{\mathrm{CCSD}} + E_{(T)}\) | \(10^{-12}\) Ha | Section check |
| Consistency | \(E_{\mathrm{tot}} = E_{\mathrm{RHF}} + E_{\mathrm{corr}}\) | \(10^{-12}\) Ha | Section check |

Oracle repro (offline only):

```bash
python <pyscf pin> ccsd-t \
    --geom water --basis sto-3g --conv-tol 1e-12
```

### Performance

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | -------------------------: | -----------------------: | ----: | ---- | ------ |
| `ccsd_t` (H₂O/STO-3G) | 0.0933 | 0.0311 | **0.333** | ✓ | Startup + small-\(N\) dominated (7 AO; spin-orbital \(n_o=10\), \(n_v=4\)); ATV faster than PySCF (\(r \ll 1\)); not a production-scale CCSD(T) stress test |

### Limitations

- Energy only; no UCCSD(T) / ROCCSD(T); no Hessian.
- Higher CC / local / open-shell / CCSD(T) gradient remain ○ — see
  `docs/roadmap/post-hf.md`, `docs/roadmap/excited-states.md`.

### References

- K. Raghavachari, G. W. Trucks, J. A. Pople, M. Head-Gordon, *Chem. Phys.
  Lett.* **157**, 479 (1989). DOI
  [10.1016/S0009-2614(89)87395-6](https://doi.org/10.1016/S0009-2614(89)87395-6)
- J. D. Watts, J. Gauss, R. J. Bartlett, *J. Chem. Phys.* **98**, 8718 (1993).
  DOI [10.1063/1.464480](https://doi.org/10.1063/1.464480)

## cc2-h2-sto3g — CC2, H₂ / STO-3G {#cc2-h2-sto3g}

### Summary

Closed-shell **CC2** (Christiansen–Koch–Jørgensen second-order approximate
coupled-cluster singles and doubles) correlation energy via `CC2<RHF>::run`,
returning `CC2Result{e_corr, e_tot, iterations, converged}` in Hartree. On
this minimal H₂ basis \(T_1 \approx 0\) by symmetry, so CC2 coincides with
MP2 to \(\sim 10^{-15}\) Ha and is intentionally distinct from full
CCSD/FCI. Oracle: PySCF `cc.rccsd.RCCSD` with `cc2=True` (stock PySCF has no
top-level `cc.CC2`). Fair-bench ratio \(r = 0.392\).

### Method

\[
E_{\mathrm{corr}} = E_{\mathrm{CC2}} - E_{\mathrm{RHF}},\qquad
E_{\mathrm{tot}} = E_{\mathrm{RHF}} + E_{\mathrm{corr}},
\]

with the CCSD energy expression once the amplitudes are known.

**CC2 truncation** (Christiansen, Koch & Jørgensen, CPL **243**, 409 (1995)):

- **Singles residual:** full CCSD singles equations (all terms retained).
- **Doubles residual:** at most **linear** in \(T_2\), with the Fock operator
  \(T_1\)-dressed; **no** quadratic \(T_2\) ring / ladder contributions that
  appear in full CCSD.
- Iteration: Jacobi micro-steps on \(t_1,t_2\) until max\(|\Delta t|\) falls
  below the internal tolerance; the gate also requires `converged == true`.

**Spin treatment.** Spin-orbital residual path shared with the CCSD kernel
(`ccsd_solve(..., cc2=true)`), matching PySCF `rccsd.RCCSD` with `cc2=True`
on the same RHF orbitals.

**Scope.** Energy only — no \(\Lambda\), no densities, no analytic gradient.
On H₂/STO-3G CC2 \(E_{\mathrm{corr}}\) equals RMP2; full CCSD / FCI
\(E_{\mathrm{corr}} = -0.020524527\) Ha is intentionally larger in magnitude.

### Implementation

```cpp
cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G};
rhf.tol = 1e-12;
rhf.max_iter = 100;

const cpu::scf::CC2Result cc2 =
    cpu::scf::CC2<cpu::scf::RHF>{rhf}.run(frame);
// cc2.e_corr, cc2.e_tot [Ha]; cc2.converged; cc2.iterations
```

| Path | Role |
| ---- | ---- |
| `post_hf.h` | `CC2<Method>` |
| `cc2.h` | Public `CC2Result` |
| `ccsd_pack.h` | Private `cc2_energy` |
| `src/cpu/scf/ccsd.cpp` | Shared residual / energy (`cc2` flag) |

### Validation (vs PySCF)

CC2 oracle: `pyscf.cc.rccsd.RCCSD(mf); my.cc2=True`, `conv_tol=1e-12`.

| Quantity | Pin | Tol | Notes |
| -------- | --: | --- | ----- |
| \(E_{\mathrm{RHF}}\) | −1.116759307396426 Ha | \(10^{-6}\) Ha | Diagnostic |
| \(E_{\mathrm{corr}}\) (CC2) | **−0.013138073589533 Ha** | **\(10^{-6}\) Ha** | Gate |
| \(E_{\mathrm{tot}}\) (CC2) | **−1.129897380985958 Ha** | **\(10^{-6}\) Ha** | Gate |
| Amplitudes | — | `converged == true` | Gate |
| Consistency | \(E_{\mathrm{tot}} = E_{\mathrm{RHF}} + E_{\mathrm{corr}}\) | \(10^{-12}\) Ha | Section check |

Cross-check (same geom/basis): PySCF `mp.MP2` gives the same
\(E_{\mathrm{corr}}\) (T₁ vanishes; CC2 reduces to MP2; see
[`mp3-h2-sto3g`](scf-mp2.md#mp3-h2-sto3g)). Full CCSD
\(E_{\mathrm{corr}} = -0.020524527\) Ha (FCI) is intentionally different.

Oracle repro (offline only):

```bash
python <pyscf pin> cc2 \
    --geom h2 --basis sto-3g --conv-tol 1e-12
```

### Performance

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | -------------------------: | -----------------------: | ----: | ---- | ------ |
| `cc2` (H₂/STO-3G) | 0.0350 | 0.0137 | **0.392** | ✓ | Startup-dominated; ATV faster than PySCF (\(r \ll 1\)); not a production-scale CC2 stress test |

### Limitations

- Energy only on closed-shell RHF; no UCC2, no λ / density, no analytic
  gradient or Hessian.
- On H₂/STO-3G, CC2 is **not** a numerical stress of the approximate-CC
  doubles truncation relative to MP2 (\(T_1 \approx 0\)); multi-electron /
  multi-MO systems are needed to exercise the CC2 residual difference.
- Open-shell / higher CC / local / DF remain ○ — see
  `docs/roadmap/post-hf.md`, `docs/roadmap/excited-states.md`.

### References

- O. Christiansen, H. Koch, P. Jørgensen, *The second-order approximate
  coupled cluster singles and doubles model CC2*, *Chem. Phys. Lett.*
  **243**, 409 (1995). DOI
  [10.1016/0009-2614(95)00841-Q](https://doi.org/10.1016/0009-2614(95)00841-Q)
