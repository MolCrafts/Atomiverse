# Equation-of-motion coupled cluster (EOM-CC)

Equation-of-motion coupled cluster computes excited, ionized, electron-attached
and spin-flipped states as right eigenstates of the similarity-transformed
Hamiltonian \(\bar{H} = e^{-T} H e^{T}\) of a converged ground-state coupled
cluster wavefunction. Each channel is a method on the concrete ground-state
cluster type, which drives the SCF reference itself:
`cpu::scf::CCSD<RHF>::eom_ee_ccsd` / `eom_ip_ccsd` / `eom_ea_ccsd` (closed-shell
excitation, ionization, attachment), `cpu::scf::CCSD<UHF>::eom_sf_ccsd`
(open-shell spin-flip) and `cpu::scf::CC2<RHF>::eom_cc2` (EOM-EE at CC2
amplitudes). Every call forces ERI construction, runs SCF and the ground-state
amplitudes, refuses a non-converged reference (no silent fallback), then
diagonalizes the dense EOM Jacobian.

The sections cover CPU only and pin energies only: H₂, H₂O and triplet O₂ in
STO-3G, each gating the ground-state CC energy and the lowest EOM root(s).

Case: `ctest -R '^scf/eom-cc$'` — sources and oracle record in [`regressions/scf/eom-cc/`](../../../regressions/scf/eom-cc/). One section of the case per `##` heading below: `atv_regressions --case scf/eom-cc --section <name>`.

## Common setup

### Theory shared by all channels

Ground state: cluster operator \(T = T_1 + T_2\) on a reference determinant
\(\lvert\Phi_0\rangle\):

\[
\lvert\Psi_{\mathrm{CC}}\rangle = e^{T}\lvert\Phi_0\rangle,\qquad
\bar{H} = e^{-T} H e^{T}.
\]

EOM: target states solve \(\bar{H} R_k = \omega_k R_k\) with a channel-specific
right operator \(R_k = R_1 + R_2\) in spin orbitals. ATV builds the full
non-symmetric spin-orbital Jacobian (Gauss–Stanton / PySCF `gintermediates`
intermediates) and diagonalizes it densely with `dgeev`; \(\omega\) is reported
in eV via the \(E_{\mathrm{h}}\rightarrow\mathrm{eV}\) conversion in
`core/units.h` (CODATA \(1\,\mathrm{Ha}=27.211386245988\,\mathrm{eV}\),
`units::kHartreeInEv`).

### Result types

Each result (`EOMEECCSDResult` in `eom_ee_ccsd.h`, `EOMIPCCSDResult` in
`eom_ip_ccsd.h`, `EOMEACCSDResult` in `eom_ea_ccsd.h`, `EOMSFCCSDResult` in
`eom_sf_ccsd.h`, `EOMCC2Result` in `eom_cc2.h`) carries `energy_ev` (ascending
roots [eV]), `e_corr_ha` and `e_ccsd_ha` / `e_cc2_ha` [Hartree], the
ground-state iteration count (`ccsd_iterations` / `cc2_iterations`) and
`converged`. The methods are declared in `post_hf.h`; the pack-level entries
(`eom_*_energies`, `internal/eom_*_pack.h`) are private engine code.
Implementations: `src/cpu/scf/eom_ee_ccsd.cpp` (EE, SF selection and pack, CC2
reuse), `src/cpu/scf/eom_ip_ccsd.cpp`, `src/cpu/scf/eom_ea_ccsd.cpp`.

Closed-shell sections share the same consumer shape (public headers
`atomiverse/cpu/frame/frame.h` + `atomiverse/cpu/scf/methods.h`, no `*Config`):

```cpp
cpu::Frame frame(2);   // H(0,0,0), H(0,0,0.74) Å; Z=1; Box::cubic(20)
cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G};
rhf.tol = 1e-12;
rhf.max_iter = 100;

const cpu::scf::EOMEECCSDResult eom =
    cpu::scf::CCSD<cpu::scf::RHF>{rhf}.eom_ee_ccsd(frame, /*n_states=*/1);
// eom.energy_ev[0] [eV]; eom.e_ccsd_ha, eom.e_corr_ha [Ha];
// eom.ccsd_iterations; eom.converged
```

Build with `-DATV_BUILD_HF=ON -DATV_BUILD_REGRESSIONS=ON` (CPU login-shell
recipe in `CLAUDE.md`, `module load buildenv-gcc/2026.03-mpich`).

### Oracle

PySCF 2.14.0, offline pins only (PySCF never runs in ctest); protocol in
`docs/tutorial/appendix-pyscf-oracle.md`. Common oracle settings:
`pin_scf` geometry presets, STO-3G with `cart=True` (engine-neutral for these
shells), SCF `conv_tol=1e-12`. Product gates in every section: ground-state CC
energy to \(10^{-6}\) Ha, EOM roots to \(10^{-4}\) eV, `converged == true`.
Oracle repro (human / offline only):

```bash
bash <pyscf env setup>
python <pyscf pin> <method> --geom <preset> --basis sto-3g --nstates <n> --conv-tol 1e-12
```

Fair wall: login node, `OMP_NUM_THREADS=1`, median of ≥3 runs, gate
\(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}} \le 10\) (no ⚡ optimization pass
required in any section):

```bash
export OMP_NUM_THREADS=1
python <pyscf fair-bench> --method <method> --geom <preset> --basis sto-3g \
    --nstates <n> --repeat 3 -- <section binary>
```

The perf twins are the same method family on minimal systems; wall times are
dominated by process startup and tiny MO spaces, so \(r\) is not a reliable
large-\(N\) predictor.

### Shared limitations

- **Energies only** — no left eigenvectors, transition moments / oscillator
  strengths, Dyson orbitals, \(\Lambda\), densities, or analytic EOM
  gradients/Hessians.
- **Dense Jacobian** diagonalization; no large-system iterative Davidson
  \(\sigma\) product path.
- **Canonical MO integrals** — no DF/CD/F12/local CC, CVS-EOM or STEOM; full
  MO \(O(N^6)\) ground state + dense EOM space.
- No DIP/DEA channels, no EOM-MP2.
- **CUDA / Kokkos:** ○ (CPU only).

### References

- Roadmap: `docs/roadmap/excited-states.md` (§3 Equation-of-Motion Coupled
  Cluster, [Excited states §3 EOM-CC](../../roadmap/excited-states.md));
  ground-state CC: `docs/roadmap/post-hf.md` (§3).
- J. F. Stanton & R. J. Bartlett, *The equation of motion coupled-cluster
  method…*, *J. Chem. Phys.* **98**, 7029 (1993).
  DOI [10.1063/1.464746](https://doi.org/10.1063/1.464746)
- J. Gauss & J. F. Stanton, *J. Chem. Phys.* **103**, 3561 (1995) (Table III
  intermediates; EOM section).
- Z. Wang, Z. Tu & F. Wang, *J. Chem. Theory Comput.* **10**, 5567 (2014)
  (EE \(\sigma\) equations as in PySCF `eom_gccsd`).

## eom-ee-ccsd — EOM-EE-CCSD, H₂ / STO-3G singlet S1 {#eom-ee-ccsd}

`CCSD<RHF>::eom_ee_ccsd(frame, n_states)` returns closed-shell singlet
excitation energies. Right-hand \(\sigma\) equations match Wang, Tu & Wang
Eqs. (9)–(10) as in PySCF `eom_gccsd.eeccsd_matvec`; packed vector basis
\(r_1\) plus \(\mathrm{tril}(i>j)\times\mathrm{tril}(a>b)\) \(r_2\).
**Singlet** roots are selected by \(\alpha\leftrightarrow\beta\) symmetry of
the \(R_1\) block on the closed-shell reference (PySCF twin
`eomee_ccsd_singlet`). Closed-shell RHF→CCSD only — no open-shell / UCCSD
reference on this path.

### System and oracle

H₂, charge 0, singlet; H(0,0,0), H(0,0,0.74) Å (`pin_scf` `h2` preset); 2 AO,
1 occ + 1 virt spatial MO. For two electrons EOM-EE-CCSD coincides with FCI in
the basis. Oracle: `pyscf.cc.CCSD(mf)` (`conv_tol=1e-12`), then
`mycc.eomee_ccsd_singlet(nroots=1)`.

| Quantity | Pin | Tol | Notes |
| -------- | --: | --- | ----- |
| \(E_{\mathrm{CCSD}}\) | **−1.137283834497130 Ha** | **\(10^{-6}\) Ha** | Product gate |
| \(\omega_{\mathrm{S1}}\) (singlet EOM-EE) | **26.365966612792256 eV** | **\(10^{-4}\) eV** | Product gate |
| \(\omega_{\mathrm{S1}}\) (Ha cross-check) | 0.968931401526066 Ha | — | REFERENCE only |

FCI excitation ladder on this basis (eV, REFERENCE): 16.504 (triplet),
**26.366 (singlet)**, 44.094 — EOM-EE-CCSD matches FCI for 2e.

### Performance

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) |
| ---- | -------------------------: | -----------------------: | ----: |
| `eom-ee-ccsd` (H₂/STO-3G, `--nstates 1`) | 0.0725 | 0.0140 | **0.193** |

Startup-dominated; ATV faster than PySCF (\(r \ll 1\)).

### Limitations

Singlet EE only on closed-shell RHF+CCSD; the other channels are the sibling
sections.

## eom-ip-ccsd — EOM-IP-CCSD, H₂O / STO-3G lowest IP {#eom-ip-ccsd}

`CCSD<RHF>::eom_ip_ccsd(frame, n_states)` returns unique ionization roots
\(\omega_k = E_{\mathrm{CCSD}}(N) - E_k(N-1)\) [eV]. The right operator spans
the spin-orbital 1h + 2h–1p space (Stanton & Bartlett; Nooijen & Snijders;
Tu–Wang–Li); \(\sigma\) equations follow Tu–Wang–Li Eqs. (8)–(9) as coded in
PySCF `eom_gccsd.ipccsd_matvec`. Spin-degenerate ionization partners are
de-duplicated before return, so the result matches PySCF spatial RCCSD
`ipccsd` / GCCSD `EOMIP` on the same RHF reference. Closed-shell reference
only.

### System and oracle

H₂O, charge 0, singlet; O(0,0,0.1173), H(0,±0.7572,−0.4692) Å (`pin_scf`
`water` preset; no \(d\) shells); 7 AO, 5 occ + 2 virt spatial MO. Oracle:
`pyscf.cc.CCSD(mf)` (`conv_tol=1e-12`), `mycc.ipccsd(nroots≥1)` (repro with
`--nstates 3`).

| Quantity | Pin | Tol | Notes |
| -------- | --: | --- | ----- |
| \(E_{\mathrm{CCSD}}\) | **−75.01246170149486 Ha** | **\(10^{-6}\) Ha** | Product gate |
| \(\mathrm{IP}_0\) (lowest EOM-IP) | **8.415727034061687 eV** | **\(10^{-4}\) eV** | Product gate |
| \(\mathrm{IP}_0\) (Ha cross-check) | 0.309272264117102 Ha | — | REFERENCE only |
| \(\mathrm{IP}_1\) (diagnostic) | ≈ 10.911018291684588 eV | — | Not gated |
| \(\mathrm{IP}_2\) (diagnostic) | ≈ 16.62369673430045 eV | — | Not gated |

The pin is the PySCF EOM-IP-CCSD root, not an experimental vertical IP; on this
minimal basis IP₀ ≈ 8.42 eV (valence), IP₁ ≈ 10.91 eV, IP₂ ≈ 16.62 eV.

### Performance

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) |
| ---- | -------------------------: | -----------------------: | ----: |
| `eom-ip-ccsd` (H₂O/STO-3G, `--nstates 1`) | 0.1470 | 0.1108 | **0.754** |

Startup-dominated; ATV slightly faster than PySCF (\(r < 1\)).

### References

- M. Nooijen & J. G. Snijders, *Coupled cluster approach to the single-particle
  Green's function*, *Int. J. Quantum Chem.* **S26**, 55 (1992).
- Z. Tu, F. Wang & X. Li, *Equation-of-motion coupled-cluster method for
  ionized states…*, *J. Chem. Phys.* **136**, 174102 (2012).
  DOI [10.1063/1.4704893](https://doi.org/10.1063/1.4704893)
- PySCF `eom_gccsd.ipccsd_matvec` (Tu–Wang–Li Eqs. (8)–(9) σ equations).

## eom-ea-ccsd — EOM-EA-CCSD, H₂ / STO-3G lowest attachment {#eom-ea-ccsd}

`CCSD<RHF>::eom_ea_ccsd(frame, n_states)` returns unique attachment roots
\(\omega_k = E_k(N+1) - E_{\mathrm{CCSD}}(N)\) [eV]. The right operator spans
the spin-orbital 1p + 2p–1h space (Nooijen & Bartlett); \(\sigma\) equations
follow Nooijen–Bartlett Eqs. (30)–(31) as coded in PySCF
`eom_gccsd.eaccsd_matvec`, packed as \(r_1[a]\) plus \(r_2[j,a>b]\).
Spin-degenerate partners are de-duplicated, matching PySCF spatial RCCSD
`eaccsd` / GCCSD `EOMEA`. Closed-shell reference only.

### System and oracle

H₂ as in [eom-ee-ccsd](#eom-ee-ccsd) (2 AO; 1 occ + 1 virt). Oracle:
`pyscf.cc.CCSD(mf)` (`conv_tol=1e-12`), `mycc.eaccsd(nroots=1)`.

| Quantity | Pin | Tol | Notes |
| -------- | --: | --- | ----- |
| \(E_{\mathrm{CCSD}}\) | **−1.137283834497130 Ha** | **\(10^{-6}\) Ha** | Product gate |
| \(\mathrm{EA}_0\) (lowest EOM-EA) | **18.821245619413340 eV** | **\(10^{-4}\) eV** | Product gate |
| \(\mathrm{EA}_0\) (Ha cross-check) | 0.691668019014956 Ha | — | REFERENCE only |

Attachment is **unbound** on this basis (positive EA energy relative to the
neutral CCSD; lowest root ≈ 18.82 eV). LUMO Koopmans estimate ~18.26 eV
(REFERENCE).

### Performance

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) |
| ---- | -------------------------: | -----------------------: | ----: |
| `eom-ea-ccsd` (H₂/STO-3G, `--nstates 1`) | 0.9293 | 0.0635 | **0.068** |

Startup-dominated; ATV faster than PySCF (\(r \ll 1\)).

### References

- M. Nooijen & R. J. Bartlett, *Equation of motion coupled cluster method for
  electron attachment*, *J. Chem. Phys.* **102**, 3629 (1995).
  DOI [10.1063/1.468591](https://doi.org/10.1063/1.468591)
- PySCF `eom_gccsd.eaccsd_matvec` (Nooijen–Bartlett Eqs. (30)–(31) σ equations).

## eom-sf-ccsd-o2 — EOM-SF-CCSD, triplet O₂ / STO-3G {#eom-sf-ccsd-o2}

`CCSD<UHF>::eom_sf_ccsd(frame, n_states, guess*)` runs high-spin UHF (optional
density guess), unrestricted spin-orbital CCSD, then EOM spin-flip with
\(\Delta M_S = -1\) (majority \(\alpha\to\beta\) when
\(n_\alpha\ge n_\beta\)):

\[
\lvert\Psi_{\mathrm{UCCSD}}\rangle = e^{T}\lvert\Phi_{\mathrm{UHF}}\rangle,\qquad
\bar{H} R_k = \omega_k R_k\quad\text{(singles + doubles; SF-selected)}.
\]

ATV builds the dense non-symmetric spin-orbital Jacobian, selects the
**majority SF subspace** (α-occupied → β-virtual channel weight) and reports
the lowest real roots; it targets multiplet reconstruction and SF excitations
from high-spin references (O₂ triplet → \(M_S=0\) singlets/triplets). PySCF
twin: `UCCSD.eomsf_ccsd` / `EOMEESpinFlip` (Davidson). UHF→UCCSD only, no GHF
reference; the closed-shell EE path is [eom-ee-ccsd](#eom-ee-ccsd).

### Davidson vs dense ladder

PySCF `eomsf_ccsd(nroots=5)` alone can miss intermediate SF roots (jumps from
~1.59 eV to ~8.81 eV). Requesting `nroots≥10–15` and taking the lowest five
recovers the full low-energy ladder. ATV's dense eigen-solve of the SF-selected
Jacobian matches that full lowest ladder; accuracy pins use PySCF `nroots=15` →
lowest 5, while the fair-bench twin uses `nroots=5`.

### Setup

O₂ is multi-minimum at the UHF level, so the section warm-starts UHF from an
ROHF density (same pattern as [ump2-o2-sto3g](scf-mp2.md#ump2-o2-sto3g)); bare
UHF may land on a different SCF minimum:

```cpp
// O₂, r(O–O)=1.2075 Å, Z=8; Box::cubic(20)
cpu::scf::ROHF rohf(cpu::scf::BasisSet::STO3G, /*n_alpha=*/9, /*n_beta=*/7);
rohf.tol = 1e-12;
rohf.max_iter = 200;
rohf.run(frame);

using Spin = cpu::scf::SlaterDeterminant::Spin;
cpu::scf::SCFInitialGuess guess;
guess.P_alpha = rohf.determinant().density(Spin::alpha);
guess.P_beta  = rohf.determinant().density(Spin::beta);
guess.has_alpha = guess.has_beta = true;

cpu::scf::UHF uhf(cpu::scf::BasisSet::STO3G, 9, 7);
uhf.tol = 1e-12;
uhf.max_iter = 200;
const cpu::scf::EOMSFCCSDResult eom =
    cpu::scf::CCSD<cpu::scf::UHF>{uhf}.eom_sf_ccsd(frame, /*n_states=*/5, &guess);
```

### System and oracle

O₂, charge 0, triplet (`spin=2` = 2S); O(0,0,0), O(0,0,1.2075) Å (`pin_scf`
`o2` preset); \(n_\alpha=9\), \(n_\beta=7\); 10 spatial AO/MO. Oracle: UHF
(`conv_tol=1e-12`), `pyscf.cc.UCCSD(mf)` (`conv_tol=1e-10`),
`mycc.eomsf_ccsd(nroots=15)` then the **lowest 5** real roots (repro with
`--spin 2 --nstates 15`).

| Quantity | Pin | Tol | Notes |
| -------- | --: | --- | ----- |
| \(E_{\mathrm{UHF}}\) | −147.633946785450 Ha | \(10^{-6}\) Ha | REFERENCE (SCF context) |
| \(E_{\mathrm{UCCSD}}\) | **−147.741936323434 Ha** | **\(10^{-6}\) Ha** | Product gate |
| \(E_{\mathrm{corr}}\) | −0.107989537984 Ha | \(10^{-6}\) Ha | REFERENCE |
| \(\omega_0\) (SF, near-zero multiplet) | **0.004154770937 eV** | **\(10^{-4}\) eV** | Product gate |
| \(\omega_1\) | **1.024028019354 eV** | **\(10^{-4}\) eV** | Product gate |
| \(\omega_2\) | **1.024028354230 eV** | **\(10^{-4}\) eV** | Product gate |
| \(\omega_3\) | **1.586396740693 eV** | **\(10^{-4}\) eV** | Product gate |
| \(\omega_4\) | **6.184403293079 eV** | **\(10^{-4}\) eV** | Product gate |

The near-zero \(\omega_0\) is multiplet reconstruction under mild spin
contamination.

### Performance

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) |
| ---- | -------------------------: | -----------------------: | ----: |
| `eom-sf-ccsd` (O₂/STO-3G, `--spin 2 --nstates 5`) | 0.317 | 0.988 | **3.12** |

ATV: UHF (ROHF warm-start) + UCCSD + majority-SF dense Jacobian; the PySCF wall
twin uses `eomsf_ccsd(nroots=5)` Davidson. \(r=3.12\) is within the 10× gate;
not a large-\(N\) predictor for full SF-EOM-CCSD.

### References

- Related: SF-TDA [spin-flip-tda-o2](scf-tddft.md#spin-flip-tda-o2).
- A. I. Krylov, *Chem. Phys. Lett.* **338**, 375 (2001).
  DOI [10.1016/S0009-2614(01)00287-1](https://doi.org/10.1016/S0009-2614(01)00287-1)
- S. V. Levchenko & A. I. Krylov, *J. Chem. Phys.* **120**, 175 (2004).
  DOI [10.1063/1.1630018](https://doi.org/10.1063/1.1630018)
- PySCF: `pyscf.cc.eom_uccsd.EOMEESpinFlip` / `UCCSD.eomsf_ccsd`.

## eom-cc2-h2 — EOM-CC2, H₂ / STO-3G singlet S1 {#eom-cc2-h2}

`CC2<RHF>::eom_cc2(frame, n_states)` solves Christiansen–Koch–Jørgensen CC2
amplitudes — full singles residual, doubles residual truncated to at most
linear in \(T_2\) with a \(T_1\)-dressed Fock (see
[cc2-h2-sto3g](scf-coupled-cluster.md#cc2-h2-sto3g)) — and then diagonalizes
the EOM-EE Jacobian **evaluated at those CC2 \(T_1/T_2\)**: the same
right-hand \(\sigma\) (Wang–Tu–Wang Eqs. (9)–(10), PySCF `eom_gccsd`), packing
and singlet selection as [eom-ee-ccsd](#eom-ee-ccsd), with a different \(T\).
The implementation reuses the EOM-EE path in `src/cpu/scf/eom_ee_ccsd.cpp` with
amplitudes from `ccsd_solve(..., cc2=true)`. Closed-shell RHF→CC2 only, no UCC2.

```cpp
const cpu::scf::EOMCC2Result eom =
    cpu::scf::CC2<cpu::scf::RHF>{rhf}.eom_cc2(frame, /*n_states=*/1);
// eom.energy_ev[0] [eV]; eom.e_cc2_ha, eom.e_corr_ha [Ha]; eom.cc2_iterations
```

### System and oracle

H₂ as in [eom-ee-ccsd](#eom-ee-ccsd). Stock PySCF has no top-level `EOMCC2`;
the oracle is `pyscf.cc.rccsd.RCCSD(mf); my.cc2=True` (`conv_tol=1e-12`), then
`mycc.eomee_ccsd_singlet(nroots=1)`.

| Quantity | Pin | Tol | Notes |
| -------- | --: | --- | ----- |
| \(E_{\mathrm{CC2}}\) | **−1.129897380985958 Ha** | **\(10^{-6}\) Ha** | Product gate |
| \(E_{\mathrm{corr}}\) (CC2) | −0.013138073589533 Ha | \(10^{-6}\) Ha | REFERENCE |
| \(\omega_{\mathrm{S1}}\) (singlet EOM-CC2) | **26.164970973311735 eV** | **\(10^{-4}\) eV** | Product gate |
| \(\omega_{\mathrm{S1}}\) (Ha cross-check) | 0.961544948014894 Ha | — | REFERENCE only |

Cross-check (same geometry/basis): full EOM-EE-CCSD S1 = 26.3659666 eV; CC2
amplitudes shift the root by ~0.20 eV (S1 ≈ 26.165 eV), so EOM-CC2 is not
FCI-equivalent on this two-electron system.

### Performance

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) |
| ---- | -------------------------: | -----------------------: | ----: |
| `eom-cc2` (H₂/STO-3G, `--nstates 1`) | 0.0779 | 0.0234 | **0.300** |

Startup-dominated; ATV faster than PySCF (\(r \ll 1\)).

### References

- O. Christiansen, H. Koch, P. Jørgensen, *The second-order approximate
  coupled cluster singles and doubles model CC2*, *Chem. Phys. Lett.*
  **243**, 409 (1995). DOI
  [10.1016/0009-2614(95)00841-Q](https://doi.org/10.1016/0009-2614(95)00841-Q)
