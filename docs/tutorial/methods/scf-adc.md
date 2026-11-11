# Algebraic diagrammatic construction (ADC)

> The regression runner has been removed. Regression commands below are historical; case sources and oracle records are retained as archived evidence, not current executable checks.

The algebraic diagrammatic construction expands a propagator — the
polarization propagator for excitations, the one-particle Green's function for
ionization and attachment — in perturbation order and diagonalizes its
intermediate-state representation (ISR). ATV implements ADC on a closed-shell
RHF reference through `cpu::scf::ADC<RHF>`, constructed from an `RHF` that has
already run on the Frame: `adc2(n_states)` (strict ADC(2) excitation
energies), `adc2x(n_states)` (extended ADC(2)-x), `ip_adc2(n_states)`
(ionization potentials) and `ea_adc2(n_states)` (electron attachment). The
method is chosen by the member name; parameters stay on the `RHF` (`basis`,
`tol`, `max_iter`), with no public `*Config` bag.

The sections cover CPU only and pin energies only: the lowest singlet
excitation of formaldehyde (ADC(2) and ADC(2)-x), the lowest IP of water and
the lowest attachment root of H₂, all in STO-3G.

Case: `ctest -R '^scf/adc$'` — sources and oracle record in [`regressions/scf/adc/`](../../../regressions/scf/adc/). One section of the case per `##` heading below: `atv_regressions --case scf/adc --section <name>`.

## Common setup

### Theory shared by all variants

- **Reference:** restricted HF (singlet, charge 0); canonical MOs and chemist
  MO ERIs after `RHF::run`.
- **Ground-state doubles:** first-order MP2 amplitudes on the \(N\)-electron
  reference,
  \[
  t_{ij}^{ab}
  = \frac{(ia|bj)}{\varepsilon_i + \varepsilon_j - \varepsilon_a - \varepsilon_b},
  \]
  whose MP2 correlation energy is returned as a diagnostic.
- **Units:** roots are reported in eV via `units::kHartreeInEv` (CODATA
  \(1\,\mathrm{Ha}=27.211386245988\,\mathrm{eV}\)); the regressions convert
  the Hartree pin with the same factor, so both sides of the eV gate agree.

### API

Public surface: `cpu::Frame` + concrete `cpu::scf::RHF` (headers
`atomiverse/cpu/frame/frame.h`, `atomiverse/cpu/scf/methods.h`; `ADC`,
`ADC2Result`, `IPADC2Result`, `EAADC2Result` in `adc.h`; implementation
`src/cpu/scf/adc.cpp`). Results are value-only: `energy_ev` (ascending roots
[eV]) and `e_corr_mp2_ha` (MP2 \(E_{\mathrm{corr}}\) [Ha]). `ADC2Result` is
shared by `adc2` and `adc2x`.

```cpp
cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G};
rhf.tol = 1e-12;
rhf.max_iter = 200;
rhf.run(frame);                       // required before any ADC call

const cpu::scf::ADC2Result adc =
    cpu::scf::ADC<cpu::scf::RHF>{rhf}.adc2(/*n_states=*/1);
// adc.energy_ev[0]  — lowest root [eV]
// adc.e_corr_mp2_ha — MP2 E_corr [Ha] (diagnostic)
```

Each call throws if there is no SCF result, no virtual orbitals, or `n_states`
lies outside the manifold dimension given in the section. Build with
`-DATV_BUILD_HF=ON -DATV_BUILD_REGRESSIONS=ON` (CPU login-shell recipe in
`docs/developer/project-contract.md`, `module load buildenv-gcc/2026.03-mpich`).

### Oracle

PySCF 2.14.0 `pyscf.adc.ADC(mf)` (`conv_tol=1e-12`), offline pins only — PySCF
**never** runs in ctest; protocol in `docs/tutorial/appendix-pyscf-oracle.md`.
Common settings: `pin_scf` geometry presets, STO-3G with `cart=True` (no \(d\)
shells; engine-neutral), RHF `conv_tol=1e-12`. Every section gates
\(E_{\mathrm{RHF}}\) and the MP2 \(E_{\mathrm{corr}}\) to 1e-6 Ha and the lowest
root to **1e-4 eV**. The pins are PySCF roots of the same ADC variant, not
experiment. Repro (human / offline only):

```bash
bash <pyscf env setup>   # once
python <pyscf pin> <adc2|adc2x|ip-adc2|ea-adc2> \
    --geom <preset> --basis sto-3g --nstates <n> --conv-tol 1e-12
```

Fair wall: login node, `OMP_NUM_THREADS=8`, Release, single login-node median
of ≥3 runs, gate \(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}} \le 10\):

```bash
python <pyscf fair-bench> \
  --method <method> --geom <preset> --basis sto-3g --nstates 1 --repeat 3 \
  -- -- <section binary>
```

Twin caveats shared by all sections: both sides include RHF setup on the same
geometry/basis, but host load and BLAS threading can move medians; the STO-3G
toys are not scaling probes — large-basis \(O(N^5)\) ADC matvec cost is not
characterized; comparing a section against a different ADC variant, CIS(D),
ΔSCF, G0W0 or EOM-CC twin is invalid for the 10× gate.

### Shared limitations

- Closed-shell RHF only; no UHF/ROHF, triplet, spin-flip or open-shell ADC.
- No ADC(3), CVS-ADC or core excitations.
- Energies only: no oscillator strengths, Dyson orbitals / spectroscopic
  factors, densities, or analytic gradients.
- No frozen-core option; full valence doubles on the tiny STO-3G spaces.
- Dense MO-ERI / naive matvec path — acceptable for the product pins, not
  production large-molecule ADC.
- Open roadmap axes: open-shell ADC, properties, frozen core and larger-basis
  scaling work (`docs/roadmap/excited-states.md`).

### References

- J. Schirmer, *Phys. Rev. A* **26**, 2395 (1982).
  DOI [10.1103/PhysRevA.26.2395](https://doi.org/10.1103/PhysRevA.26.2395)
- PySCF ADC module: Loos et al., *J. Chem. Phys.* **158**, 044112 (2023).
  DOI [10.1063/5.0135290](https://doi.org/10.1063/5.0135290)

## adc2-h2co-sto3g — ADC(2), H₂CO / STO-3G {#adc2-h2co-sto3g}

`ADC<RHF>::adc2(n_states)` → `ADC2Result`: closed-shell singlet excitation
energies from the **strict ADC(2)** ISR matrix (not ADC(2)-x, not ADC(3)) —
singles block \(M_{ss}\) a second-order intermediate, singles–doubles coupling
first order, doubles diagonal zeroth order only. Eigenproblem: non-symmetric
matvec Davidson on the singles+doubles space, Ritz values sorted ascending.
Twin: PySCF `radc_ee`, `method='adc(2)'`, `method_type='ee'`. `n_states` must
lie in \([1,\,n_{\mathrm{occ}} n_{\mathrm{virt}}]\).

### System and oracle

H₂CO, charge 0, singlet; C(0,0,0), O(0,0,1.21), H(0,±0.943,−0.587) Å
(`pin_scf` `h2co`; types 6, 8, 1, 1; `Box::cubic(30.0)`); 12 AO, 8 occ + 4
virt spatial. The gated root is the lowest singlet (\(n\to\pi^*\)).

| Quantity | Pin | Tol | Notes |
| -------- | --: | --- | ----- |
| \(E_{\mathrm{RHF}}\) | −112.353990369522 Ha | 1e-6 Ha | reference |
| \(E_{\mathrm{corr}}^{\mathrm{MP2}}\) | −0.117214030417 Ha | 1e-6 Ha | diagnostic |
| **ADC(2) S1 \(\omega\)** | **0.166669832218 Ha = 4.535317180049 eV** | **1e-4 eV** | **product gate** |
| ADC(2) S2 \(\omega\) | 0.371358612315 Ha ≈ 10.105 eV | — | not gated |
| ADC(2) S3 \(\omega\) | 0.508095678166 Ha ≈ 13.826 eV | — | not gated |

Repro with `--nstates 3`. Same geometry/basis CIS(D) S1 is 4.545 eV
([cisd-h2co-sto3g](scf-multireference.md#cisd-h2co-sto3g)).

### Performance

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) |
| ---- | -------------------------: | -----------------------: | ----: |
| `adc2` EE, nstates=1 | 0.247 | 0.066 | **0.267** |

Recorded `ratio_atv_over_pyscf=0.2669`, `within_10x=true`.

### References

- A. B. Trofimov, J. Schirmer, *J. Phys. B* **28**, 2299 (1995).
  DOI [10.1088/0953-4075/28/12/003](https://doi.org/10.1088/0953-4075/28/12/003)

## adc2x-h2co-sto3g — ADC(2)-x, H₂CO / STO-3G {#adc2x-h2co-sto3g}

`ADC<RHF>::adc2x(n_states)` → `ADC2Result`: the extended ADC(2)-x ISR keeps the
strict-ADC(2) singles block and first-order singles–doubles coupling, and adds
**first-order** doubles–doubles two-electron terms (vvvv, oovv/ovvo, oooo) —
the extension that defines ADC(2)-x. Same Davidson eigenproblem and `n_states`
range as [adc2-h2co-sto3g](#adc2-h2co-sto3g). Twin: PySCF
`method='adc(2)-x'`, `method_type='ee'`. ADC(2)-x is EE only on this surface;
the IP/EA channels (`ip_adc2` / `ea_adc2`) are strict ADC(2).

### System and oracle

H₂CO as in [adc2-h2co-sto3g](#adc2-h2co-sto3g).

| Quantity | Pin | Tol | Notes |
| -------- | --: | --- | ----- |
| \(E_{\mathrm{RHF}}\) | −112.353990369522 Ha | 1e-6 Ha | reference |
| \(E_{\mathrm{corr}}^{\mathrm{MP2}}\) | −0.117214030417 Ha | 1e-6 Ha | diagnostic |
| **ADC(2)-x S1 \(\omega\)** | **0.121950512434 Ha = 3.318442496747 eV** | **1e-4 eV** | **product gate** |
| ADC(2)-x S2 \(\omega\) | 0.340446186583 Ha ≈ 9.264 eV | — | not gated |
| ADC(2)-x S3 \(\omega\) | 0.418603691330 Ha ≈ 11.391 eV | — | not gated |

Repro with `--nstates 3`. Strict ADC(2) S1 on the same geometry/basis is
4.535 eV; the ≈−1.22 eV shift is the first-order doubles correction. The pin is
the PySCF ADC(2)-x root, not an experimental band origin.

### Performance

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) |
| ---- | -------------------------: | -----------------------: | ----: |
| `adc2x` EE, nstates=1 | 0.467 | 0.112 | **0.240** |

Recorded `ratio_atv_over_pyscf=0.2403`, `within_10x=true`.

### References

- A. B. Trofimov, J. Schirmer, *J. Phys. B* **28**, 2299 (1995).
  DOI [10.1088/0953-4075/28/12/003](https://doi.org/10.1088/0953-4075/28/12/003)

## ip-adc2-h2o-sto3g — IP-ADC(2), H₂O / STO-3G {#ip-adc2-h2o-sto3g}

`ADC<RHF>::ip_adc2(n_states)` → `IPADC2Result`: strict IP-ADC(2) ISR of the
one-particle Green's function — singles are the **1h** (occupied) space with
second-order intermediate \(M_{ij}\); 1h–**2h1p** coupling is first order via
\((ia|jk)\); the 2h1p diagonal is zeroth order only,
\(D_{ajk} = \varepsilon_j + \varepsilon_k - \varepsilon_a\). Eigenvalues of the
sign-flipped ISR matrix are the ionization potentials, sorted ascending. Twin:
PySCF `method='adc(2)'`, `method_type='ip'`. `n_states` must lie in
\([1,\,n_{\mathrm{occ}}]\). No IP-ADC(2)-x.

### System and oracle

H₂O, charge 0, singlet; O(0,0,0.1173), H(0,±0.7572,−0.4692) Å (`pin_scf`
`water`; types 8, 1, 1; `Box::cubic(30.0)`); 7 AO, 5 occ + 2 virt spatial.
Oracle `nroots≥1`.

| Quantity | Pin | Tol | Notes |
| -------- | --: | --- | ----- |
| \(E_{\mathrm{RHF}}\) | −74.963023138463 Ha | 1e-6 Ha | reference |
| \(E_{\mathrm{corr}}^{\mathrm{MP2}}\) | −0.035545651647 Ha | 1e-6 Ha | diagnostic |
| **IP-ADC(2) IP1** | **0.304180883162 Ha = 8.277183500364 eV** | **1e-4 eV** | **product gate** |
| IP-ADC(2) IP2 | 0.395907981066 Ha ≈ 10.773 eV | — | not gated |
| IP-ADC(2) IP3 | 0.604352918337 Ha ≈ 16.445 eV | — | not gated |
| IP-ADC(2) IP4 | 1.191953547066 Ha ≈ 32.435 eV | — | not gated |
| IP-ADC(2) IP5 | 1.403753901455 Ha ≈ 38.198 eV | — | not gated |

Repro with `--nstates 5`. On this minimal basis the lowest IP is ~8.28 eV
(valence); the gate is the PySCF strict IP-ADC(2) root, not an experimental
vertical IP.

### Performance

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) |
| ---- | -------------------------: | -----------------------: | ----: |
| `ip-adc2`, nstates=1 | 2.766 | 0.528 | **0.191** |

Recorded `ratio_atv_over_pyscf=0.1907`, `within_10x=true` (3-atom system).

### References

- J. Schirmer, L. S. Cederbaum, O. Walter, *Phys. Rev. A* **28**, 1237 (1983).
  DOI [10.1103/PhysRevA.28.1237](https://doi.org/10.1103/PhysRevA.28.1237)

## ea-adc2-h2-sto3g — EA-ADC(2), H₂ / STO-3G {#ea-adc2-h2-sto3g}

`ADC<RHF>::ea_adc2(n_states)` → `EAADC2Result`: strict EA-ADC(2) ISR of the
electron-attachment propagator — singles are the virtual (1p) space with
second-order intermediate \(M_{ab}\); doubles are 2p–1h with a
**zeroth-order** diagonal only; singles–doubles coupling is first order via
\((ia|bc)\). Eigenproblem: non-symmetric matvec Davidson on the 1p + 2p–1h
space, Ritz values ascending; positive \(\omega\) means endothermic
attachment. Twin: PySCF `radc_ea`, `method='adc(2)'`, `method_type='ea'`.
`n_states` must lie in
\([1,\,n_{\mathrm{virt}} + n_{\mathrm{occ}} n_{\mathrm{virt}}^2]\). No
EA-ADC(2)-x and no continuum / complex absorbing potential treatment.

### System and oracle

H₂, charge 0, singlet; H(0,0,0), H(0,0,0.74) Å (`pin_scf` `h2`;
`Box::cubic(20.0)`); 2 AO, 1 occ + 1 virt spatial.

| Quantity | Pin | Tol | Notes |
| -------- | --: | --- | ----- |
| \(E_{\mathrm{RHF}}\) | −1.116759307396 Ha | 1e-6 Ha | reference |
| \(E_{\mathrm{corr}}^{\mathrm{MP2}}\) | −0.013138073590 Ha | 1e-6 Ha | diagnostic |
| **EA-ADC(2) EA0 \(\omega\)** | **0.684281565504 Ha = 18.620249979933 eV** | **1e-4 eV** | **product gate** |
| EA-ADC(2) EA1 \(\omega\) | 1.920840843660 Ha ≈ 52.269 eV | — | not gated |

Repro with `--nstates 2`. On this minimal basis the LUMO is unbound relative
to the \(N\)-electron ground state (EA0 ≈ 18.62 eV endothermic); STO-3G
attachment roots for H₂ are basis-limited, so do not treat the 18.62 eV pin as
a physical electron affinity. Roadmap: EA-ADC(2) energy ✓,
properties/gradients ○.

### Performance

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) |
| ---- | -------------------------: | -----------------------: | ----: |
| `ea-adc2`, nstates=1 | 0.735 | 0.157 | **0.213** |

Recorded `ratio_atv_over_pyscf=0.2135`, `within_10x=true` (2 AO).
