# Green's-function and RPA methods

> The regression runner has been removed. Regression commands below are historical; case sources and oracle records are retained as archived evidence, not current executable checks.

Many-body Green's-function methods on a closed-shell RHF reference:
quasiparticle (QP) energies from one-shot spectral \(G_0W_0\) and from one-shot
second-order Green's function theory (G0F2), neutral excitations from the
Bethe–Salpeter equation on \(G_0W_0\) poles, and the direct-RPA correlation
energy. Each is a class template wrapping an `RHF` that has already run on the
Frame (`RHF::run(frame)`), mirroring its PySCF counterpart:
`cpu::scf::G0W0<RHF>{rhf}.run()` → `G0W0Result` (`g0w0.h`),
`cpu::scf::BSE<RHF>{rhf}.run(n_states, tda)` → `BSEResult` (`bse.h`),
`cpu::scf::DirectRPA<RHF>{rhf}.run()` → `DirectRPAResult` (`rpa.h`) and
`cpu::scf::GF2<RHF>{rhf}.run()` → `GF2Result` (`gf2.h`); implementations in
`src/cpu/scf/{g0w0,bse,rpa,gf2}.cpp`. No `*Config` bag, no factory.

The sections cover CPU only (CUDA ○) on H₂/STO-3G: the \(G_0W_0\) and GF2 QP
gap / IP / EA, the BSE singlet S1 and the dRPA correlation energy.

Case: `ctest -R '^scf/green-function$'` — sources and oracle record in [`regressions/scf/green-function/`](../../../regressions/scf/green-function/). One section of the case per `##` heading below: `atv_regressions --case scf/green-function --section <name>`.

## Common setup

### System and reference

H₂, \(r(\mathrm{H–H})=0.74\) Å, H at \((0,0,0)\), \((0,0,0.74)\), type = Z,
`Box::cubic(20.0)`; STO-3G (1 occupied, 1 virtual spatial MO). Every section
runs the same reference and gates \(E_{\mathrm{RHF}}\) = −1.116759307396 Ha to
1e-6 Ha:

```cpp
cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G};
rhf.tol = 1e-12;
rhf.max_iter = 100;
rhf.run(f);

cpu::scf::G0W0Result gw = cpu::scf::G0W0<cpu::scf::RHF>{rhf}.run();
// gw.gap_ev, gw.ip_ev, gw.ea_ev, gw.qp_energy_ha, gw.converged
```

The post-SCF entries require a prior successful `run()` on a closed-shell
restricted reference and throw otherwise (and, for G0W0 / dRPA, when
\(n_{\mathrm{occ}}=0\) or \(n_{\mathrm{vir}}=0\)). Mean-field reference values
used across sections: \(\varepsilon_{\mathrm{HOMO}}^{\mathrm{HF}}\) =
−0.578553859832 Ha, \(\varepsilon_{\mathrm{LUMO}}^{\mathrm{HF}}\) =
+0.671143491914 Ha, HF gap 34.005997328943 eV; full dRPA spectrum
\(n_{\mathrm{states}} = n_{\mathrm{occ}}\cdot n_{\mathrm{vir}} = 1\) with
\(\Omega_1\) = 1.570852128278 Ha (42.745063997913 eV).

QP observables (G0W0, GF2):

| Symbol | Definition | Unit |
| ------ | ---------- | ---- |
| \(\varepsilon_p^{\mathrm{QP}}\) | QP orbital energies | Ha |
| gap | \(\varepsilon_{\mathrm{LUMO}}^{\mathrm{QP}} - \varepsilon_{\mathrm{HOMO}}^{\mathrm{QP}}\) | eV |
| IP | \(-\varepsilon_{\mathrm{HOMO}}^{\mathrm{QP}}\) | eV |
| EA | \(\varepsilon_{\mathrm{LUMO}}^{\mathrm{QP}}\) | eV |

Build with `-DATV_BUILD_HF=ON -DATV_BUILD_REGRESSIONS=ON` (CPU login-shell
recipe in `docs/developer/project-contract.md`, `module load buildenv-gcc/2026.03-mpich`).

### Oracle

PySCF **2.14.0**, offline pins only (PySCF never imported from CMake/ctest);
protocol in `docs/tutorial/appendix-pyscf-oracle.md`. Common settings: STO-3G,
Cartesian (`cart=True`; no \(d\) shells), SCF `conv_tol = 1e-12`. Each twin is
matched to the exact product method class (spectral / exact-ERI / one-shot);
the stock PySCF analytic-continuation, density-fitted or self-consistent
variants appear only as info rows and are never the gate. Repro
(offline only):

```bash
python <pyscf pin> <g0w0|bse|direct-rpa|gf2> \
  --geom h2 --basis sto-3g --conv-tol 1e-12
```

Fair wall: median of 5, `OMP_NUM_THREADS=1`, gate
\(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}} \le 10\):

```bash
export OMP_NUM_THREADS=1
python <pyscf fair-bench> \
  --method <g0w0|bse|direct_rpa|gf2> --geom h2 --basis sto-3g --repeat 5 \
  -- <section binary>
```

Shared twin caveats: wall times include RHF SCF on both sides, so pure
post-SCF cost is not split out; the 2-AO system runs in microseconds, so the
ratios are informative for these pins, not a large-basis throughput claim.
There is no standing multi-system perf suite beyond these twins.

### Shared limitations

- **Closed-shell RHF only** — no UHF/UKS/ROHF/GHF variants.
- **CPU only** — CUDA ○.
- H₂/STO-3G is a minimal system (1 occupied, 1 virtual); larger systems and
  basis sets, denser RPA / auxiliary spectra and their numerical robustness
  are open.
- Roadmap: [excited-states.md](../../roadmap/excited-states.md)
  (`docs/roadmap/excited-states.md`: G0W0, BSE@G0W0, Direct RPA, GF2 ✓ CPU).

### References

- Hybertsen & Louie, *Phys. Rev. B* **34**, 5390 (1986) (GW quasiparticles),
  DOI 10.1103/PhysRevB.34.5390
- Zhu & Chan, *J. Chem. Theory Comput.* **17**, 727 (2021) — PySCF `gw` (AC
  G0W0 and DF + imaginary-axis `gw.rpa` companions).

## g0w0-h2-sto3g — Spectral G0W0, H₂ / STO-3G {#g0w0-h2-sto3g}

One-shot \(G_0W_0\): Green's function and screened interaction built from the
RHF solution, no self-consistency in \(G\) or \(W\). \(W\) is direct-RPA
screening (no exchange in the RPA kernel, \(\alpha_{\mathrm{hf}}=0\)) on the
full dRPA spectrum. The diagonal correlation self-energy
\(\Sigma_c(\omega)\) is the spectral (frequency-exact) sum over RPA poles with
broadening \(\eta = 10^{-8}\) Ha, and for each MO \(p\)

\[
\varepsilon_p^{\mathrm{QP}} = \varepsilon_p^{\mathrm{HF}} +
  \Sigma_{c,pp}(\varepsilon_p^{\mathrm{QP}})
\]

is solved by non-linear Newton (tolerance \(10^{-8}\) Ha) — **not** the
linearized \(Z\)-factor form. `G0W0Result`: `qp_energy_ha`, `mf_energy_ha`,
`gap_ev`, `ip_ev`, `ea_ev`, `n_occ`, `converged`. Not included:
analytic-continuation GW (`GWAC` / default `gw.GW`) in the public API, evGW /
qsGW / scGW, RPA total energies from this entry (dRPA is only the screening
step).

### Pins

Twin: `gw.gw_exact.GWExact` + `tdscf.dRPA` on the HF density (`dft.RKS` with
`xc='hf'`), not the analytic-continuation `gw.GW` path.

| Quantity | Pin | Tol | Notes |
| -------- | --: | --- | ----- |
| \(\varepsilon_{\mathrm{HOMO}}^{\mathrm{G0W0}}\) | **−0.596957673490** Ha | 1e-5 Ha | product |
| \(\varepsilon_{\mathrm{LUMO}}^{\mathrm{G0W0}}\) | **+0.689547305573** Ha | 1e-5 Ha | product |
| **QP gap** | **35.007583892680 eV** | **1e-4 eV** | **primary gate** |
| IP \(-\varepsilon_H^{\mathrm{QP}}\) | **16.244045825851 eV** | 1e-4 eV | product |
| EA \(\varepsilon_L^{\mathrm{QP}}\) | **18.763538066829 eV** | 1e-4 eV | product |
| AC twin gap (`gw.GW`) | 35.007177297967 eV | — | ~0.4 meV off spectral; not the gate |

Sanity enforced in the binary: G0W0 gap > HF gap.

### Performance

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) |
| ---- | ------------------------: | ----------------------: | ----: |
| spectral G0W0 | 0.1148 | 0.0125 | **0.109** |

ATV binary = RHF + spectral G0W0; PySCF = RKS(hf) + dRPA + GWExact. Do not A/B
against AC `gw.GW`.

### References

- van Setten et al., *J. Chem. Theory Comput.* **11**, 5665 (2015).
- PySCF spectral: Berkelbach et al., `pyscf.gw.gw_exact`.

## bse-h2-sto3g — BSE@G0W0, H₂ / STO-3G {#bse-h2-sto3g}

One-shot BSE@G0W0: spectral \(G_0W_0\) QP energies (the
[g0w0-h2-sto3g](#g0w0-h2-sto3g) path) replace mean-field orbital energies in
the Casida matrices, and the particle–hole attraction is the **static**
direct-RPA screened interaction built with QP energies and exact 4-centre MO
ERIs (no DF, no RI; PySCF BSE convention):

\[
\begin{aligned}
W &= v + v\,\Pi\,v,\\
\Pi &= 4\,(D^{-1}-4v_{\mathrm{ph}})^{-1},\\
D^{-1}_{ia} &= \varepsilon_i^{\mathrm{QP}}-\varepsilon_a^{\mathrm{QP}}.
\end{aligned}
\]

Singlet working equations (PySCF `gw.bse` / Hillenbrand–Li–Zhu 2025):

\[
\begin{aligned}
A_{ia,jb} &= (\varepsilon_a^{\mathrm{QP}}-\varepsilon_i^{\mathrm{QP}})
            \delta_{ij}\delta_{ab} + 2(ia|jb) - W_{ij,ab}, \\
B_{ia,jb} &= 2(ia|jb) - W_{ib,aj}.
\end{aligned}
\]

Full BSE solves the Casida RPA problem on \((A,B)\) by dense diagonalization
(\(n_{\mathrm{ov}}=1\) here); BSE-TDA diagonalizes \(A\) only (\(B=0\)).
`BSEResult`: `energy_ev` (ascending \(\omega_{0n}\), eV), `tda`, `channel`
(singlet), `gw` (the underlying `G0W0Result`, e.g. `bse.gw.gap_ev`). Full BSE
is the default (`tda=false`):

```cpp
cpu::scf::BSEResult bse =
    cpu::scf::BSE<cpu::scf::RHF>{rhf}.run(/*n_states=*/1, /*tda=*/false);
// bse.energy_ev[0] — S1 [eV]; bse.gw.gap_ev — G0W0 QP gap [eV]
```

Not included: DF/RI BSE in the public API, dynamic \(W(\omega)\), triplet /
spin-flip BSE. BSE is not TDDFT/TDHF (different response kernels; see
[scf-tddft.md](scf-tddft.md) for those entry points).

### Pins

Twin: PySCF exact-ERI BSE on spectral G0W0 (`GWExact` + dRPA), not the DF-only
stock `pyscf.gw.bse.BSE(gw.GW(...))` path.

| Quantity | Pin | Tol | Notes |
| -------- | --: | --- | ----- |
| \(\varepsilon_{\mathrm{HOMO}}^{\mathrm{G0W0}}\) | −0.596957673490 Ha | — | info (same as G0W0 pin) |
| \(\varepsilon_{\mathrm{LUMO}}^{\mathrm{G0W0}}\) | +0.689547305573 Ha | — | info |
| G0W0 QP gap | 35.007583892680 eV | **1e-4 eV** | sanity / secondary gate |
| **BSE S1** (full, exact ERI) | **25.956271542931 eV** | **1e-4 eV** | **primary gate** |
| BSE-TDA S1 | 26.809052342076 eV | 1e-4 eV | companion |
| DF stock twin `gw.bse.BSE` S1 | ~25.954 eV | — | info only; ~2 meV DF error |

Sanity enforced in the binary: full BSE S1 < TDA S1; BSE S1 < QP gap.

### Performance

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) |
| ---- | ------------------------: | ----------------------: | ----: |
| exact-ERI BSE@spectral-G0W0 | 0.2233 | 0.0134 | **0.060** |

ATV binary = RHF + spectral G0W0 + BSE; PySCF = RKS(hf) + dRPA + GWExact +
exact-ERI static BSE (G0W0 included on both sides). Do not A/B against DF
stock `gw.bse.BSE`.

### References

- Hillenbrand, Li & Zhu, *J. Chem. Phys.* **162**, 174117 (2025) (PySCF BSE).
- *J. Comput. Chem.* **38**, 383 (2017) (molecular BSE / static \(W\)).
- PySCF BSE module: `pyscf.gw.bse` (DF production path; exact-ERI twin in the
  case `REFERENCE.md`).

## direct-rpa-h2-sto3g — Direct RPA correlation energy, H₂ / STO-3G {#direct-rpa-h2-sto3g}

Direct RPA as a **standalone correlation energy** — not TDHF excitation RPA
(`TDHF<RHF>`), not TDDFT Casida, not RPAX/SOSEX. Singlet spin-adapted Casida
matrices with **Coulomb only** (\(\alpha_{\mathrm{hf}}=0\)):

\[
A_{ia,jb} = (\varepsilon_a - \varepsilon_i)\,\delta_{ij}\delta_{ab}
  + 2\,(ia|jb),\qquad
B_{ia,jb} = 2\,(ia|jb),
\]

and the Furche plasmon formula on the full spectrum (exact; no frequency
quadrature, no density fitting):

\[
E_{\mathrm{c}}^{\mathrm{dRPA}}
  = \tfrac12\Bigl(\sum_s \Omega_s - \mathrm{Tr}\,A\Bigr),
\]

with \(\Omega_s\) the positive eigenvalues of the Hermitian reduction
\((A-B)^{1/2}(A+B)(A-B)^{1/2}\). This equals the imaginary-frequency integral
\(\frac{1}{2\pi}\int_0^\infty\mathrm{d}\omega\,
\mathrm{Tr}[\ln(1-\chi_0 v)+\chi_0 v]\) in the non-DF limit. The same dRPA
poles feed the [g0w0-h2-sto3g](#g0w0-h2-sto3g) screening.

`DirectRPAResult`: `e_corr` (\(E_{\mathrm{c}}^{\mathrm{dRPA}}\)), `e_tot`
(\(E_{\mathrm{RHF}} + E_{\mathrm{c}}\)), `omega_ha` (\(\Omega_s\) ascending),
`tr_A` (\(\mathrm{Tr}\,A\), plasmon-identity diagnostic) — all Hartree. The
poles are diagnostics; excitations are the TDHF/TDDFT entry points. Not
included: DF-RPA or imaginary-axis Legendre quadrature (stock `gw.rpa.RPA`),
RPAX / SOSEX / ACFDT-RPA (no exchange in the kernel).

```cpp
cpu::scf::DirectRPAResult rpa = cpu::scf::DirectRPA<cpu::scf::RHF>{rhf}.run();
// rpa.e_corr, rpa.e_tot [Hartree]; rpa.omega_ha; rpa.tr_A
```

### Pins

Twin: exact MO-ERI plasmon rebuild on PySCF RHF orbitals (not stock DF +
Legendre `gw.rpa.RPA`).

| Quantity | Pin | Tol | Notes |
| -------- | --: | --- | ----- |
| dRPA \(\Omega_1\) | **1.570852128278** Ha (42.745063997913 eV) | 1e-6 Ha | same pole as G0W0 screening |
| \(\mathrm{Tr}\,A\) | **1.612118275776** Ha | 1e-6 Ha | diagnostic |
| **\(E_{\mathrm{c}}^{\mathrm{dRPA}}\)** | **−0.020633073749** Ha | **1e-6 Ha** | **primary gate** |
| \(E_{\mathrm{tot}}^{\mathrm{dRPA}}\) | **−1.137392381145** Ha | 1e-6 Ha | \(E_{\mathrm{RHF}}+E_{\mathrm{c}}\) |
| DF twin (`gw.rpa.RPA`, `nw=100`) | −0.020624247051 Ha | — | info; ~9e-6 Ha DF residual |
| RMP2 (context) | −0.013138073590 Ha | — | sanity: \(E_c^{\mathrm{dRPA}} < E_c^{\mathrm{MP2}}\) |

Additional gates: plasmon identity
\(|E_c - \tfrac12(\Omega_1-\mathrm{Tr}A)| < 10^{-12}\) Ha; \(e_{\mathrm{tot}}\)
from pieces within \(10^{-12}\) Ha.

### Performance

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) |
| ---- | ------------------------: | ----------------------: | ----: |
| exact MO-ERI plasmon dRPA | 0.0217 | 0.0165 | **0.760** |

ATV = RHF + dRPA plasmon; PySCF twin = RHF + exact MO-ERI plasmon. Do not A/B
against DF + Legendre `gw.rpa.RPA` for the accuracy gate.

### References

- Furche, *J. Chem. Phys.* **114**, 5982 (2001), DOI 10.1063/1.1353585
- Ren et al., *New J. Phys.* **14**, 053020 (2012)

## gf2-h2-sto3g — One-shot GF2 (G0F2), H₂ / STO-3G {#gf2-h2-sto3g}

One-shot second-order Green's function: the 2h1p + 2p1h second-order
self-energy \(\Sigma^{(2)}\) is built in the auxiliary (Lehmann)
representation from the HF Green's function (bare Coulomb \(v\), no screened
\(W\); construction matches PySCF AGF2 with `os_factor = ss_factor = 1` and
`weight_tol = 1e-11` to drop uncoupled poles), then the Dyson equation is
solved **once** as a single Hermitian diagonalization of the extended Fock
matrix

\[
\begin{pmatrix}
F & V \\
V^\dagger & \varepsilon_{\mathrm{aux}}
\end{pmatrix},
\]

with \(F = \mathrm{diag}(\varepsilon_{\mathrm{HF}})\) and \(V\) coupling
physical MOs to auxiliary poles; the chemical potential is found by binary
search for \(N = 2\) electrons (RHF occupancy 2). No outer density/Fock
self-consistency. The Galitskii–Migdal correlation energy from the one-shot
self-energy equals canonical MP2 on a pure HF reference.

`GF2Result`: `qp_energy_ha`, `mf_energy_ha`, `pole_energy_ha`, `gap_ev`,
`ip_ev`, `ea_ev`, `e_corr`, `e_tot`, `n_occ`, `converged`. Not G0W0 (bare
\(v\)) and not ADC(2) (no ADC propagator; see
[adc2-h2co-sto3g](scf-adc.md#adc2-h2co-sto3g)); self-consistent AGF2 is a
separate open roadmap row (GF2 ✓ CPU; self-consistent GF2 ○).

```cpp
cpu::scf::GF2Result gf = cpu::scf::GF2<cpu::scf::RHF>{rhf}.run();
// gf.gap_ev, gf.ip_ev, gf.ea_ev, gf.qp_energy_ha, gf.e_corr, gf.converged
```

### Pins

Twin: PySCF `agf2.RAGF2` **one-shot** path (SE from `init_gf()` →
`se.get_greens_function(F_HF)`), not full self-consistent AGF2.

| Quantity | Pin | Tol | Notes |
| -------- | --: | --- | ----- |
| \(\varepsilon_{\mathrm{HOMO}}^{\mathrm{GF2}}\) | **−0.591623589738** Ha | 1e-5 Ha | product |
| \(\varepsilon_{\mathrm{LUMO}}^{\mathrm{GF2}}\) | **+0.684213221821** Ha | 1e-5 Ha | product |
| **QP gap** | **34.717288266166 eV** | **1e-4 eV** | **primary gate** |
| IP \(-\varepsilon_H^{\mathrm{QP}}\) | **16.098898012595 eV** | 1e-4 eV | product |
| EA \(\varepsilon_L^{\mathrm{QP}}\) | **18.618390253572 eV** | 1e-4 eV | product |
| \(E_{\mathrm{corr}}\) (GM / MP2) | **−0.013138073590** Ha | 1e-6 Ha | Galitskii–Migdal |
| SC-AGF2 gap (info) | ≈ 34.451051 eV | — | full `RAGF2.kernel()`; **not** the gate |

Sanity enforced in the binary: gap > 0 and finite. Related:
[mp3-h2-sto3g](scf-mp2.md#mp3-h2-sto3g).

### Performance

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) |
| ---- | ------------------------: | ----------------------: | ----: |
| one-shot GF2 | 0.2449 | 0.0174 | **0.071** |

ATV binary = RHF + one-shot GF2; PySCF = RHF + RAGF2 SE + Dyson. Do not A/B
against full self-consistent `RAGF2.kernel()`.

### References

- Backhouse, Holzmann & Booth, *J. Chem. Theory Comput.* **17**, 3414 (2021)
  (PySCF auxiliary GF2).
- Galitskii & Migdal, *Sov. Phys. JETP* **7**, 96 (1958).
