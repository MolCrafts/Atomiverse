# IPEA-xTB

> The regression runner has been removed. Regression commands below are historical; case sources and oracle records are retained as archived evidence, not current executable checks.

IPEA-xTB (Ásgeirsson, Bauer, Grimme, *Chem. Sci.* **8**, 4879 (2017)) is a
special-purpose **GFN1 reparametrization for vertical ionization potentials
and electron affinities**, built for a QCEIMS mass-spectrometry context. The
Hamiltonian skeleton is GFN1 (EHT \(H^0\) from overlap, shell-resolved
Klopman–Ohno SCC, on-site third-order \(\Gamma q^3\), exponential pair
repulsion, D3(BJ) with the GFN1 damping set) but with extra \((n+1)s\)
functions on C/N/O, Klopman–Ohno \(g_{\exp}=1\), and its own
\(k_{s,p,\mathrm{sp},\mathrm{diff}}\) / onsite / repulsion numbers. It is not
GFN1, not GFN2, not DFTB3; the family oracle is xtb only.

The concrete method is `cpu::semiempirical::IPEA{charge, multiplicity}`
(defaults 0 and 1) with members `scc_tol` (1e-8), `max_scc` (250),
`mix_beta`, `with_forces`, `compute_vip`, `compute_vea`, and results
`last_vip()` / `last_vea()` [eV]; there is no public `*Config` and no
`make_*` / `build_*`. Native static H/C/N/O tables only — **no** `libxtb`.
`IPEA` is mounted on `cpu::Evaluate` at `evaluate::Phase::evaluate` (energy on
`kEnergy` [eV], forces on `kAtom_f*` [eV/Å], ΔSCC IP/EA on the result
accessors) and, with `with_forces = true`, on `cpu::vib::Hessian` at
`vib::Phase::force`. The sections cover the CPU closed-shell energy, analytic
forces, vertical IP/EA, and the FD Hessian on methane. See also
[Ch. 17 Semiempirical AM1](../ch17-semiempirical-am1.md) (SE product surface;
NDDO/AM1 is a different family with an OpenMOPAC oracle) and
[Ch. 14 Vibrations](../ch14-vibrations.md).

Case: `ctest -R '^semiempirical/ipea-xtb$'` — sources and oracle record in [`regressions/semiempirical/ipea-xtb/`](../../../regressions/semiempirical/ipea-xtb/). One section of the case per `##` heading below: `atv_regressions --case semiempirical/ipea-xtb --section <name>`.

## Method

Closed-shell total energy on the public surface (internals in Hartree / Bohr):

\[
E = E_{H^0} + E_{\gamma} + E_{\Gamma} + E_{\mathrm{rep}} + E_{\mathrm{D3}}
  = \mathrm{Tr}(P H_0)
  + \tfrac12 q\,\gamma\,q
  + \tfrac13 \Gamma q^3
  + E_{\mathrm{rep}}
  + E_{\mathrm{D3}}.
\]

| Term | Role |
| ---- | ---- |
| \(E_{H^0}=\mathrm{Tr}(P H_0)\) | Occupied density on a CN-shifted EHT Hamiltonian; extra \((n+1)s\) on C/N/O; \(S\) from Stewart STO-\(n\)G |
| \(E_\gamma=\tfrac12 q\,\gamma\,q\) | Shell-resolved Klopman–Ohno SCC (**\(g_{\exp}=1\)**, harmonic-mean hardness) |
| \(E_\Gamma=\tfrac13\sum_A\Gamma_A q_A^3\) | On-site third-order (atomic, not shell-resolved); frozen-\(q\) \(\partial E_\Gamma/\partial R=0\) |
| \(E_{\mathrm{rep}}\) | Classical pair \(Z^{\mathrm{eff}}_A Z^{\mathrm{eff}}_B\,e^{-\sqrt{\alpha_A\alpha_B}\,r^{1.5}}/r\) |
| \(E_{\mathrm{D3}}\) | Molecular D3(BJ); GFN1 set \(s_6=1\), \(s_8=2.4\), \(a_1=0.63\), \(a_2=5.0\) (xtb \(s_9=0\)) |

1. **Basis / overlap.** Per-element Stewart STO-\(n\)G primitives (tblite
   `basis/slater.f90` coefficients). H: 1s (4G, valence) + 2s (3G,
   polarization, Gram–Schmidt against 1s), i.e. hydrogen is double-ζ.
   C/N/O: 2s + 2p (6G) **plus** 3s (3G, extra \((n+1)s\), Gram–Schmidt
   against valence s). Cartesian overlap \(S_{\mu\nu}\) from Gaussian product
   integrals. CH₄: **13 AO**, 11 shells, 8 electrons (C contributes 5 AO vs
   GFN1’s 4).
2. **EHT \(H^0\).** On-site \(h_l - k_{\mathrm{CN},l}\,\mathrm{CN}_A\) with D3
   exponential coordination numbers (Pyykkö \(r_{\mathrm{cov}}\), \(k_1=16\)).
   Off-site
   \[
   H^0_{\mu\nu}
     = \tfrac12(h_\mu+h_\nu)\,k_{\mu\nu}
       \bigl(1+k_{\mathrm{poly},\mu}\sqrt{R/(r_\mu+r_\nu)}\bigr)
       \bigl(1+k_{\mathrm{poly},\nu}\sqrt{R/(r_\mu+r_\nu)}\bigr)
       S_{\mu\nu},
   \]
   \(k_{\mu\nu}\) from \(k_s=2.15\), \(k_p=2.30\), \(k_{sp}=2.25\),
   \(k_{\mathrm{diff}}=2.25\) (polarization shells), and
   \(k_{\mathrm{EN}}=-7\times10^{-3}\) on \((\Delta\chi)^2\). No GFN1 pair
   scales \(k_{\mathrm{HH}}\) / \(k_{\mathrm{NH}}\).
3. **SCC.** Shell charges \(q_i = n_i - n_i^0\) (Mulliken). Klopman–Ohno
   \(\gamma_{ij}\) with **\(g_{\exp}=1\)** (GFN1 uses 2) and harmonic-mean
   shell hardness. On-site third-order \(E_{\Gamma}=\frac13\sum_A\Gamma_A q_A^3\)
   (atomic, not shell-resolved). Anderson mixing (`mix_beta`, history 6)
   until \(\max|\Delta q_{\mathrm{shell}}| < \texttt{scc\_tol}\). Integer
   occupation (CH₄ HOMO–LUMO 7.751 eV; xtb’s 300 K Fermi fill is integer
   here).
4. **Repulsion.**
   \[
   E_{\mathrm{rep}}
     = \sum_{A<B}
       \frac{Z_A^{\mathrm{eff}} Z_B^{\mathrm{eff}}}{R}
       \exp\bigl(-\sqrt{\alpha_A\alpha_B}\,R^{1.5}\bigr).
   \]
5. **Dispersion.** Molecular D3(BJ), GFN1 set above. The halogen-bond
   correction is identically 0 on CH₄ and is not implemented.

Parameter path: static tables → D3 exponential CN → shifted atomic levels
\(\varepsilon_{A\ell}\) → \(H/S\) → generalized eigen → Mulliken shell \(q\) →
Anderson SCC → repulsion/D3 → \(E/F\), with parameters as members of `IPEA`
and tables in `basis_data/ipea_tables.h` (tblite `ipea1.f90` / xtb
`param_ipea-xtb.txt`). Unsupported \(Z\) throws.

**IPEA vs GFN1 on Td CH₄ (same geometry, different Hamiltonian):**

| Knob | IPEA-xTB | GFN1-xTB ([`gfn1-xtb-organic`](semiempirical-gfn1-xtb.md#gfn1-xtb-organic)) |
| ---- | -------- | ----------------------------- |
| Extra AO | \((n+1)s\) on C/N/O | H 2s polarization only |
| CH₄ AO / shells | 13 / 11 | 12 / 10 |
| \(k_s, k_p, k_{sp}, k_{\mathrm{diff}}\) | 2.15, 2.30, 2.25, 2.25 | 1.85, 2.25, 2.08, 2.85 |
| \(g_{\exp}\) | 1 | 2 |
| Pair \(k_{\mathrm{HH}}, k_{\mathrm{NH}}\) | none | 0.96, 1.04 |
| D3(BJ) | same GFN1 set | same |
| xtb \(E_{\mathrm{tot}}\) | **−4.670476737294 Eh** | −4.274232224875 Eh |
| Product | energy / \(G\) / FD \(H\) + ΔSCC IP/EA with `ipeashift` | energy / \(G\) / FD \(H\) only |

| Path | Role |
| ---- | ---- |
| `ipea.h` | Public `IPEA` (`scc_tol`, `max_scc`, `mix_beta`, `charge`, `multiplicity`, `with_forces`, `compute_vip`, `compute_vea`, `last_vip`, `last_vea`) |
| `ipea_tables.h` | Static IPEA H/C/N/O rows + global scales + `kIpeaShiftEv` / `kIpeaEtempK` |
| `src/cpu/semiempirical/ipea.cpp` | Stewart overlap / extra 3s / EHT+SCC / rep / D3 / analytic \(\nabla E\); ionized ΔSCC + degenerate-shell occ (private) |
| `tests/cpu/semiempirical/test_xtb_ipea.cpp` | Unit: tables, Z throw, finite H₂O \(E\), analytic-vs-FD (water) + net-force invariance, shift constant, finite CH₄ VIP/VEA, flags-off zeros |

## Common setup

Tetrahedral CH₄ (energy, VIP/VEA and Hessian sections; the same xyz as
[`gfn1-xtb-organic`](semiempirical-gfn1-xtb.md#gfn1-xtb-organic), so the two
Hamiltonians are discriminated on one molecule): C at the origin, H at
\(\pm s,\pm s,\pm s\) with \(s=1.09/\sqrt{3}\), \(r(\mathrm{C{-}H})=1.09\) Å,
charge 0, closed shell (`--uhf 0`; ATV `multiplicity = 1`), cubic
\(L = 20\) Å cell (isolated; min-image present, images not engaged). Masses
12.011 (C) / 1.008 (H) amu.

| Atom | \(Z\) | \(x\) | \(y\) | \(z\) |
| ---- | ----: | ----: | ----: | ----: |
| C | 6 | 0 | 0 | 0 |
| H | 1 | \(s\) | \(s\) | \(s\) |
| H | 1 | \(s\) | \(-s\) | \(-s\) |
| H | 1 | \(-s\) | \(s\) | \(-s\) |
| H | 1 | \(-s\) | \(-s\) | \(s\) |

```cpp
cpu::Frame frame(5);
const double s = 1.09 / std::sqrt(3.0);
const double x[5] = {0, s, s, -s, -s};
const double y[5] = {0, s, -s, s, -s};
const double z[5] = {0, s, -s, -s, s};
const int Z[5] = {6, 1, 1, 1, 1};
for (int i = 0; i < 5; ++i) {
    frame.span(kAtom_type)[i] = Z[i];
    frame.span(kAtom_x)[i] = x[i];
    frame.span(kAtom_y)[i] = y[i];
    frame.span(kAtom_z)[i] = z[i];
    frame.span(kAtom_mass)[i] = (i == 0) ? 12.011 : 1.008;
    frame.span(kAtom_tag)[i] = i + 1;
}
frame.box = cpu::Box::cubic(20.0);
frame.meta.get(kEnergy) = 0;
```

Units: \(E\) [eV] on `kEnergy`; \(\mathbf{R}\) [Å]; \(\mathbf{F}\) [eV/Å].
Conversion: \(E[\mathrm{eV}] = E[\mathrm{Ha}] \times 27.211386245988\) (ATV
CODATA, `units::kHartreeInEv`; forces via
`units::kHartreePerBohrInEvPerAngstrom`).

Build with the SE-only module closure (`module load buildenv-gcc/2026.03-mpich`;
`-DATV_BUILD_SEMIEMPIRICAL=ON -DATV_BUILD_EVALUATE=ON`, plus `-DATV_BUILD_VIB=ON`
for the Hessian section; MD/MINIMIZER/HF/PW/GPW/CUDA stay OFF).

## Oracle {#oracle}

**xtb 6.7.1** only (Arrhenius; EasyBuild `xtb/6.7.1-gcc-2025b-eb` when
present, `OMP_NUM_THREADS=1`); the family is locked to xtb — never OpenMOPAC
or DFTB+ as reference. Offline pins are hard-coded in the section binaries;
xtb is never launched from ctest. IPEA is selected on the oracle by pointing
`XTBPATH` at a directory whose `param_gfn1-xtb.txt` is a copy of
`param_ipea-xtb.txt`; xtb then prints the `I P E A - x T B` banner and cites
DOI 10.1039/c7sc00601b. Commands:

```bash
# XTBPATH dir whose param_gfn1-xtb.txt is a copy of param_ipea-xtb.txt
export XTBPATH=<twin xtbpath>
export OMP_NUM_THREADS=1
xtb mol.xyz --gfn 1 --chrg 0 --uhf 0 --sp      # energy; --acc 0.01 --grad / --vipea / --acc 0.01 --hess
```

```text
oracle: xtb
oracle_version: 6.7.1
```

Re-pin offline only (record TOTAL ENERGY [Eh], and per section
`gradient.txt`, "delta SCC IP/EA (eV)" / "empirical IP/EA shift (eV)", or
`vibspectrum.txt` ω and `hessian.txt` \(H_{C_x,C_x}\), into `REFERENCE.md`).

**Td CH₄ transcript** (default `--acc 1.0`): 13 AO, 8 electrons, HOMO–LUMO
7.751 eV, 8 SCC iterations.

| Term | Ha (`--sp`) | Ha (`--vipea`) | Ha (`--hess`) |
| ---- | --: | --: | --: |
| SCC (electronic) | −4.766828456418 | −4.766828456418 | −4.766828456418 |
| → electrostatic | 0.001132006161 | 0.001132005712 | 0.001132006171 |
| repulsion | 0.096806468732 | 0.096806468732 | 0.096806468732 |
| dispersion | −0.000454749608 | −0.000454749608 | −0.000454749608 |
| halogen | 0 | 0 | 0 |
| **Total** | **−4.670476737294** | **−4.670476737294** | **−4.670476737294** |

Public energy pin: **−127.09014645 eV**.

**Shared kernel hole vs xtb.** The CH₄ residuals (8.90 meV Td, 8.49 meV
stretched) come from D3 C6 interpolation / D3 CN vs xtb’s internal D3
(ncoord), Stewart primitive screening / contraction vs xtb `intcut`, and
Anderson vs Broyden mixing — not a missing Hamiltonian term; analytic forces
match ATV’s own energy FD. The \(5\times10^{-2}\) eV energy window covers it on
a ~127 eV total. Polar molecules carry a larger ~0.5 eV hole vs xtb
(water, polar Δq / E3 / charge); it is an energy-path hole, not a gradient or
Hessian defect, and no section pins water against xtb — the unit test
`test_cpu_xtb_ipea` gates the water gradient by ATV energy FD only.

## Limitations

- **H/C/N/O tables only.** Unsupported \(Z\) throws with a how-to; other IPEA
  elements are not in-tree.
- **Closed-shell even-electron N-electron state.** `multiplicity != 1` throws;
  VIP/VEA start from that closed-shell ground state and the ionized SCCs stay
  restricted (no UHF).
- **Halogen-bond correction** is not implemented (identically 0 on CH₄).
- No ALPB/GBSA, CPCM-X/ddCOSMO, PBC, spin, sTDA, `--etemp`; no `libxtb` /
  runtime `param_ipea-xtb.txt` parser. GFN-FF is classical and outside this
  method.
- Energy + \(G\) + FD \(H\) + VIP/VEA do not make the whole GFN family closed.

Sibling methods: GFN0 [`gfn0-xtb-forces`](semiempirical-gfn0-xtb.md#gfn0-xtb-forces) /
[`gfn0-xtb-analytic-hessian`](semiempirical-gfn0-xtb.md#gfn0-xtb-analytic-hessian);
GFN1 [`gfn1-xtb-forces`](semiempirical-gfn1-xtb.md#gfn1-xtb-forces); GFN2
[`gfn2-xtb-forces`](semiempirical-gfn2-xtb.md#gfn2-xtb-forces), AES-in-Fock
[`gfn2-aes-fock`](semiempirical-gfn2-xtb.md#gfn2-aes-fock), GFN1/2 Hessian
[`gfn-xtb-analytic-hessian`](semiempirical-gfn2-xtb.md#gfn-xtb-analytic-hessian).

## References

- V. Ásgeirsson, C. Bauer, S. Grimme, *Chem. Sci.* **8**, 4879 (2017),
  DOI [10.1039/c7sc00601b](https://doi.org/10.1039/c7sc00601b)
- C. Bannwarth, E. Caldeweyher, S. Ehlert, A. Hansen, P. Pracht, J. Seibert,
  S. Spicher, S. Grimme, *WIREs Comput. Mol. Sci.* **11**, e01493 (2021)
  (xtb)
- Offline twins under `tools/tb_oracle/inputs/xtb/`; roadmap
  `docs/roadmap/semiempirical.md` §3.

## ipea-xtb-small — Td CH₄ energy {#ipea-xtb-small}

### Summary

IPEA-xTB total energy on isolated tetrahedral methane (GFN1 −4.274232224875
Eh vs IPEA **−4.670476737294 Eh** on the same geometry). The section asserts

\[
\bigl|E_{\mathrm{ATV}} - (-127.09014645\,\mathrm{eV})\bigr| \le 0.05\,\mathrm{eV}
\]

against the offline xtb `--sp` pin. Measured ATV residual is **8.90 meV**.
Fair wall vs xtb: \(T_{\mathrm{ATV}}/T_{\mathrm{xtb}} = 2.0 \le 10\). xtb
`--vip` is a multi-step ΔSCC IP with an empirical shift and is not this pin
(see [`ipea-xtb-vip`](#ipea-xtb-vip)).

```cpp
cpu::Evaluate eval;
cpu::semiempirical::IPEA ipea;
ipea.with_forces = false;
ipea.scc_tol = 1e-8;                 // max_scc stays at the default 250
eval.on(evaluate::Phase::evaluate).add_ref(ipea);
eval.build();
eval.run(frame);
const double e_eV = frame.meta.get(kEnergy);  // [eV]
```

### Validation (vs xtb)

Command: `xtb mol.xyz --gfn 1 --chrg 0 --uhf 0 --sp` (xtb default
`--acc 1.0`; twin `mol.xyz` + `xtb.out.txt`). Provenance: xtb 6.7.1
`TOTAL ENERGY` −4.670476737294 Ha; banner `I P E A - x T B`.

| Quantity | Atomiverse / pin | xtb 6.7.1 | Tol | Notes |
| -------- | ---------------: | --------: | --- | ----- |
| \(E_{\mathrm{tot}}\) | **−127.09014645 eV** | **−4.670476737294 Ha** | **0.05 eV** | gate `\|E − pin\| ≤ 0.05` |
| Measured residual | **8.90 meV** (Release) | \(E_{\mathrm{ATV}}=-127.08124904\) eV | — | vs this xtb transcript |
| SCC residual | \(<\) `scc_tol` | 8 iter (xtb) | `1e-8` e | gated |
| Finite energy | required | — | finite, \(\lvert E\rvert>1\) eV | gated |

8.90 meV is inside the 50 meV window — not bit-identical (D3 C6, integral
cutoff, mixer), not a missing term.

### Performance

Fair 1-thread wall (Release, `OMP_NUM_THREADS=1`); recorded, **not** asserted
by this section.

| Twin | \(T_{\mathrm{xtb}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{xtb}}\) | ≤10× | Caveat |
| ---- | -----------------------: | -----------------------: | ----------------------------------------: | ---- | ------ |
| CH₄ IPEA Evaluate (forces off) | **0.005** (xtb 6.7.1 clock; `/usr/bin/time` 0.02 s) | **0.01** (Release, 1 thread) | **2.0** | ✓ | Tiny molecule; both walls are I/O-scale |

## ipea-xtb-forces — analytic nuclear forces, off-eq CH₄ {#ipea-xtb-forces}

### Summary

Closed-form nuclear gradient of IPEA-xTB on off-equilibrium methane. After
one shell-resolved Klopman–Ohno SCC (\(g_{\exp}=1\)), forces are
\(\mathbf{F}=-\nabla E\) at that density — Hellmann–Feynman + Pulay on
\(\partial S\), CN/shell-polynomial chains, \(\partial\gamma/\partial R\),
pair repulsion, and D3(BJ) — written to `frame.span(kAtom_f*)` in eV/Å. This
is the **xtb IPEA `--grad` module boundary** (not GFN1, not a \(6N\) energy
finite-difference). Fair wall \(T_{\mathrm{ATV}}/T_{\mathrm{xtb}}\approx 0.26\le 10\).

### Ionic forces

`nuclear_gradient` in `src/cpu/semiempirical/ipea.cpp` at frozen SCC density.
xtb prints \(\nabla E\) [Eh/\(a_0\)]; ATV stores \(\mathbf{F}=-\nabla E\)
[eV/Å] via `units::kHartreePerBohrInEvPerAngstrom`:

\[
\mathbf{F}_A = -\frac{\partial E}{\partial\mathbf{R}_A}
=
\mathbf{F}_A^{H_0+\mathrm{Pulay}}
+ \mathbf{F}_A^{\mathrm{CN}}
+ \mathbf{F}_A^{\gamma}
+ \mathbf{F}_A^{\mathrm{rep}}
+ \mathbf{F}_A^{\mathrm{D3}},
\]

with pairwise accumulation (antisymmetrized bond contributions) plus CN
chains:

| Term | Contraction (schematic) |
| ---- | ----------------------- |
| Electronic \(H_0\) + Pulay | \(2\,\mathrm{Tr}\!\bigl((P\odot H_0'-W)\,\partial S/\partial R + P\,\partial\Pi/\partial R\,S\bigr)\) plus SCC shift \(V_i=\sum_j\gamma_{ij}q_j\) on \(S\); \(W=\sum_i n_i\varepsilon_i\,c_ic_i^\top\) |
| CN / shell-poly chain | \(\sum_\ell(\partial E/\partial h_\ell)\,(-k_{\mathrm{CN},\ell}\,\partial\mathrm{CN}/\partial R)\) plus \(\partial\Pi/\partial R\) of the EHT polynomial |
| SCC \(\gamma\) (frozen \(q\)) | \(q_i q_j\,(\partial\gamma_{ij}/\partial R)\) from Klopman–Ohno \(\partial\gamma/\partial r\) (\(g_{\exp}=1\)) |
| Third-order \(\Gamma\) | on-site; frozen-\(q\) contribution is identically 0 |
| Repulsive | \(\partial E_{\mathrm{rep}}/\partial R\) of the \(e^{-\sqrt{\alpha_A\alpha_B}\,r^{1.5}}/r\) pair |
| D3 | Analytic two-body D3(BJ) forces (`gpw::dftd3_pbe_bj`, GFN1 damping) |

The \(6N\) central-FD path is **not** the production force. The section
compares two selected components against a **central energy-only** FD of
ATV’s own \(E\) (\(h=10^{-4}\) Å) as an identity gate.

### System and settings

The Td CH₄ with H1 scaled so \(r(\mathrm{C{-}H1})=1.20\) Å (one C–H stretched
1.09 → 1.20 Å, so \(\|F\|\neq 0\)); closed shell, `charge = 0`; SCC at class
defaults `scc_tol = 1e-8`, `max_scc = 250`; xtb `--acc 0.01` (SCF
\(10^{-8}\) Eh).

| Atom | \(Z\) | \(x\) [Å] | \(y\) [Å] | \(z\) [Å] |
| ---- | ----: | ----: | ----: | ----: |
| C | 6 | 0 | 0 | 0 |
| H1 (stretched) | 1 | 0.692820323028 | 0.692820323028 | 0.692820323028 |
| H2 | 1 | 0.629311793417 | −0.629311793417 | −0.629311793417 |
| H3 | 1 | −0.629311793417 | 0.629311793417 | −0.629311793417 |
| H4 | 1 | −0.629311793417 | −0.629311793417 | 0.629311793417 |

```cpp
const double s = 0.629311793417;          // 1.09 / sqrt(3)
const double h1 = s * (1.20 / 1.09);      // H1 at (h1, h1, h1)
cpu::semiempirical::IPEA ipea;            // charge 0, multiplicity 1
ipea.with_forces = true;
cpu::Evaluate eval;
eval.on(evaluate::Phase::evaluate).add_ref(ipea);
eval.build();
eval.run(frame);
// Energy: frame.meta.get(kEnergy) [eV]; forces: kAtom_fx / fy / fz [eV/Å]
```

### Validation (xtb)

Command: `XTBPATH=… xtb mol.xyz --gfn 1 --chrg 0 --uhf 0 --acc 0.01 --grad`;
twin `mol.xyz`, `xtb.out.txt`, `gradient.txt`. Conversion
\(F=-\nabla E\times\) `units::kHartreePerBohrInEvPerAngstrom`.

**xtb \(\nabla E\) [Eh/\(a_0\)]** (`gradient.txt`):

| Atom | \(\partial E/\partial x\) | \(\partial E/\partial y\) | \(\partial E/\partial z\) |
| ---- | ------------------------: | ------------------------: | ------------------------: |
| C | −3.5026444745746×10⁻² | −3.5026444745746×10⁻² | −3.5026444745746×10⁻² |
| H1 (stretched) | +4.0716066407528×10⁻² | +4.0716066407528×10⁻² | +4.0716066407528×10⁻² |
| H2 | −1.7357154065799×10⁻³ | −1.9769531276008×10⁻³ | −1.9769531276009×10⁻³ |
| H3 | −1.9769531276008×10⁻³ | −1.7357154065799×10⁻³ | −1.9769531276008×10⁻³ |
| H4 | −1.9769531276008×10⁻³ | −1.9769531276007×10⁻³ | −1.7357154065798×10⁻³ |

xtb also printed TOTAL ENERGY **−4.662477307157 Eh**,
\(\|\nabla E\|=0.093201061537\) Eh/\(a_0\), 13 AO / 11 shells / 8 electrons,
HOMO–LUMO 7.618 eV, 10 SCC iterations.

**xtb energy decomposition (documentation; Eh):**

| Term | Eh |
| ---- | --: |
| SCC (electronic) | −4.745157981515 |
| → electrostatic | 0.000977581394 |
| repulsion | 0.083136092009 |
| dispersion | −0.000455417651 |
| halogen | 0 |
| **Total** | **−4.662477307157** |

**Hard-coded gates:**

| Quantity | Atomiverse | xtb 6.7.1 | Tol | Notes |
| -------- | ---------: | --------: | --- | ----- |
| \(E\) | `kEnergy` | **−4.662477307157 Eh** | \(5\times10^{-2}\) eV | same floor as `ipea-xtb-small` |
| \(F_{C,x}\) | `kAtom_fx[0]` | **+1.801132205114 eV/Å** | \(5\times10^{-3}\) eV/Å | \(-\nabla E\) converted; (111) stretch axis |
| \(F_{H1,x}\) | `kAtom_fx[1]` | **−2.093704314111 eV/Å** | \(5\times10^{-3}\) eV/Å | dual-component gate |
| \(\lvert\sum_i F_{i,d}\rvert\) | identity | 0 | \(1\times10^{-9}\) eV/Å | translation invariance, each Cartesian |
| \(F_{C,x}\), \(F_{H1,x}\) vs energy FD | identity | — | \(3\times10^{-5}\) eV/Å | \(h=10^{-4}\) Å; unit test (water) same |

**Host residual (Release, `OMP_NUM_THREADS=1`, not a ctest gate):**
\(\lvert E_{\mathrm{ATV}}-E_{\mathrm{pin}}\rvert=8.49\) meV,
\(\lvert F_{C,x}-F_{\mathrm{pin}}\rvert=5.12\times10^{-4}\) eV/Å,
\(\lvert F_{H1,x}-F_{\mathrm{pin}}\rvert=1.86\times10^{-3}\) eV/Å,
analytic vs energy FD \(<2\times10^{-8}\) eV/Å.

**Tolerance rationale.** The energy floor is the shared kernel hole (8.90 meV
on Td CH₄). Force tol \(5\times10^{-3}\) eV/Å sits above the 0.51 / 1.86
meV/Å host residual on the matching (111) Cartesian components.

### Performance

Fair wall asserted in the section (`ratio > 10` fails). The analytic path is
one SCC + one gradient (replaces 30 energy solves of a \(6N\) FD on 5 atoms).

| Twin | \(T_{\mathrm{xtb}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{xtb}}\) | ≤10× | Caveat |
| ---- | -----------------------: | -----------------------: | ----------------------------------------: | ---- | ------ |
| CH₄ IPEA `--grad` | **0.004** (1 thread) | **1.035×10⁻³** | **≈ 0.26** | ✓ | xtb cold wall from archived `xtb.out.txt`; ATV tables are static |

## ipea-xtb-vip — vertical IP/EA, Td CH₄ vs xtb `--vipea` {#ipea-xtb-vip}

### Summary

Vertical ionization potential and electron affinity on the Td CH₄. This is
the **xtb `--vip` / `--vea` / `--vipea` module boundary**: a closed-shell
N-electron SCC, then two ionized ΔSCC solves at charge \(\pm 1\) with
degenerate-shell occupations (xtb 300 K Fermi on a Td-degenerate t2) minus the
empirical `ipeashift` **4.8455 eV**. Not a Koopmans orbital energy
(\(-\varepsilon_{\mathrm{HOMO}}\)), not UHF, not GFN1 (GFN1 has no
`ipeashift` VIP/VEA product). The section asserts

\[
\bigl|\mathrm{IP}_v - 13.0088\,\mathrm{eV}\bigr| \le 0.15\,\mathrm{eV},
\qquad
\bigl|\mathrm{EA}_v - (-1.4559\,\mathrm{eV})\bigr| \le 0.15\,\mathrm{eV}
\]

against the offline xtb `--vipea` pin, plus the same N-electron energy window
as [`ipea-xtb-small`](#ipea-xtb-small) (0.05 eV). Measured ionized residuals
are **0.103 eV** (IP) and **0.125 eV** (EA). Fair wall
\(T_{\mathrm{ATV}}/T_{\mathrm{xtb}} = 2.0 \le 10\).

### Method

\[
\mathrm{IP}_v = \bigl(E(N-1)-E(N)\bigr)[\mathrm{eV}] - \Delta_{\mathrm{IPEA}},
\qquad
\mathrm{EA}_v = \bigl(E(N)-E(N+1)\bigr)[\mathrm{eV}] - \Delta_{\mathrm{IPEA}}.
\]

\(\Delta_{\mathrm{IPEA}} = 4.8455\,\mathrm{eV}\) is `ipeashift` from
`param_ipea-xtb.txt`. In-tree constant

\[
\texttt{kIpeaShiftEv}
  = 1.78069 \times 0.1 \times 27.21138505
  \approx 4.8455\,\mathrm{eV}
\]

(`kIpeaEvToHa = 1/27.21138505`; tblite GAM3-style scale; the public energy
conversion stays 27.211386245988). Neutral CH₄ stays integer double
occupation. Ionized SCCs are **restricted** (not UHF), \(T_{\mathrm{el}}=300\)
K (`kIpeaEtempK`), and share remaining electrons equally across MOs within a
0.05 Eh degeneracy window — that recovers xtb’s 300 K Fermi occupations on
the Td-degenerate t2. ATV’s Stewart/intcut residual splits that t2 by
\(\gg kT\), so a literal 300 K Fermi would collapse to \(2+2+1\) and drop
\(E(N-1)\) by \(\sim 0.1\) eV; the path is the degenerate-shell share, not a
second Fermi search.

Cation (7 e): occupations \(2.0000 + 1.6667\times 3\) (t2 hole). Anion (9 e):
\(2.0000\times 4 + 1.0000\). xtb totals: \(E(N)=\) **−4.670476737294 Eh**,
\(E(N-1)=\) **−4.014341473228 Eh**, \(E(N+1)=\) **−4.795043035470 Eh**.

```cpp
cpu::Evaluate eval;
cpu::semiempirical::IPEA ipea;
ipea.with_forces = false;
ipea.compute_vip = true;
ipea.compute_vea = true;
ipea.scc_tol = 1e-8;
eval.on(evaluate::Phase::evaluate).add_ref(ipea);
eval.build();
eval.run(frame);
const double e_eV = frame.meta.get(kEnergy);  // N-electron [eV]
const double vip = ipea.last_vip();          // IP_v [eV]
const double vea = ipea.last_vea();          // EA_v [eV]
```

### Validation (vs xtb)

Command: `XTBPATH=… xtb mol.xyz --gfn 1 --chrg 0 --uhf 0 --vipea` (default
`--acc 1.0`; twin `mol.xyz` + `xtb.out.txt`). Transcript cation occupations
\(2.0000 + 1.6667\times 3\); anion \(2.0000\times 4 + 1.0000\); N-electron
decomposition in the `--vipea` column under [Oracle](#oracle).

| Quantity | Atomiverse | xtb 6.7.1 | Tol | Notes |
| -------- | ---------: | --------: | --- | ----- |
| empirical shift | `kIpeaShiftEv` | **4.8455 eV** | — | `ipeashift`; unit-test \(\lvert\Delta\rvert\le 5\times 10^{-4}\) eV |
| \(\mathrm{IP}_v\) | `IPEA::last_vip()` | **13.0088 eV** | **0.15 eV** | \(\bigl(E(N-1)-E(N)\bigr)-\Delta_{\mathrm{IPEA}}\) |
| \(\mathrm{EA}_v\) | `IPEA::last_vea()` | **−1.4559 eV** | **0.15 eV** | \(\bigl(E(N)-E(N+1)\bigr)-\Delta_{\mathrm{IPEA}}\) |
| \(E(N)\) | `kEnergy` | **−4.670476737294 Eh** (−127.09014645 eV) | **0.05 eV** | same floor as `ipea-xtb-small` |
| Finite IP/EA | identity | finite | hard | `last_vip` / `last_vea` when flags set |
| SCC residual | identity | 8 iter (xtb) | `1e-8` e | N-electron gate |

**Host residual (Release, `OMP_NUM_THREADS=1`, not a ctest gate):**
\(\lvert E_{\mathrm{ATV}}-E_{\mathrm{pin}}\rvert=8.90\) meV,
\(\lvert\mathrm{IP}_{\mathrm{ATV}}-\mathrm{IP}_{\mathrm{pin}}\rvert=0.103\) eV,
\(\lvert\mathrm{EA}_{\mathrm{ATV}}-\mathrm{EA}_{\mathrm{pin}}\rvert=0.125\) eV.
Cation t2 occupations match xtb \(2+1.6667\times 3\).

**Tolerance rationale.** IP/EA are ΔSCC of a **charged** species: the shared
kernel hole is amplified by \(\Delta q=\pm 1\). The 0.15 eV IP/EA window is
that ionized residual plus margin — **not** a missing occupation or shift
term.

### Performance

Fair 1-thread wall (Release, `OMP_NUM_THREADS=1`); recorded, **not**
asserted. Same *product* (CH₄ ΔSCC IP/EA); ATV is three SCCs (N, N−1, N+1) on
static tables; xtb `--vipea` is the same three solves plus I/O.

| Twin | \(T_{\mathrm{xtb}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{xtb}}\) | ≤10× | Caveat |
| ---- | -----------------------: | -----------------------: | ----------------------------------------: | ---- | ------ |
| CH₄ IPEA `--vipea` | **0.005** (xtb 6.7.1 cpu clock; transcript wall 0.006 s; `/usr/bin/time` 0.02–0.03 s) | **0.01** (Release, 1 thread) | **2.0** | ✓ | Tiny molecule; both walls are I/O-scale |

## ipea-xtb-analytic-hessian — FD Hessian, Td CH₄ vs xtb `--hess` {#ipea-xtb-analytic-hessian}

### Summary

Nuclear Hessian of IPEA-xTB on the Td CH₄, assembled as a **\(6N\) central
finite difference of the analytic gradient** of
[`ipea-xtb-forces`](#ipea-xtb-forces) (\(h=10^{-3}\) Å) through
`cpu::vib::Hessian` with `cpu::semiempirical::IPEA` (`with_forces = true`) on
`vib::Phase::force` — the same shape as
[`gfn-xtb-analytic-hessian`](semiempirical-gfn2-xtb.md#gfn-xtb-analytic-hessian) /
[`gfn0-xtb-analytic-hessian`](semiempirical-gfn0-xtb.md#gfn0-xtb-analytic-hessian),
**not** a closed-form \(\partial^2 E/\partial R_i\partial R_j\) kernel and
**not** `libxtb`. xtb 6.7.1 IPEA `--hess` prints an **analytical** Hessian
(log: “analytical hessian”), a different algorithm class; the gate is still
\(\omega\) and one Cartesian \(H\) element vs the xtb printout, with tols that
absorb the path mismatch. After mass-weighting (IUPAC-2013 from \(Z\)) and TR
projection the nine vibrational wavenumbers are written on
`HessianResult::frequencies` [cm⁻¹]; the Cartesian matrix is
`HessianResult::hessian` [eV/Å²]. Fair wall
\(T_{\mathrm{ATV}}/T_{\mathrm{xtb}}\approx 0.061\le 10\).

### Method

Central FD of the analytic forces, then symmetrized. With \(g=-F\) and
\(h=10^{-3}\) Å (`cpu::vib::kHessianStepAngstrom`):

\[
H_{ij}
  \approx
  \frac{g_i(+h\,e_j)-g_i(-h\,e_j)}{2h},
  \qquad
  H\leftarrow\tfrac12(H+H^\top).
\]

For \(N=5\) this is a fixed **\(6N=30\)** force budget plus one finalize
pass (restore \(R_0\), mass-weight, TR-project, diagonalize). The driver
is `cpu::vib::Hessian` (`Driver<vib::Phase>`); stages
`setup → displace → force → accumulate → diagonalize → thermochem →
end_of_step`. Masses for \(\omega\) come from IUPAC-2013 `mass_amu(Z)`
(`kAtom_type`), **not** `kAtom_mass`.

**Not included.** Closed-form \(\partial^2 E/\partial R_i\partial R_j\)
(CPHF / IPEA second-derivative kernel). IR intensities from an APT (the FD
driver does not solve CPHF).

Additional settings: xtb `--acc 0.01` (SCF \(10^{-8}\) Eh); \(H\) [eV/Å²],
\(\omega\) [cm⁻¹]; \(a_0=0.52917721092\) Å (`units::kBohrInAngstrom`);
thermo ctor \(T=298.15\) K, \(P=101325\) Pa, multiplicity 1, \(\sigma=1\),
empty recorder paths. xtb reports Td, gradient norm \(7.87\times10^{-4}\)
Eh/\(a_0\), \(n_{\mathrm{imag}}=0\); all nine vibrational eigenvalues are real.

```cpp
cpu::semiempirical::IPEA ipea;
ipea.with_forces = true;
// Empty recorder paths: do not write frequencies.tsv / mode_*.xyz.
cpu::vib::Hessian hess(298.15, 101325.0, 1, 1, "", "");
hess.on(vib::Phase::force).add(std::move(ipea));
hess.build();
const cpu::vib::HessianResult res = hess.run(frame);
// res.hessian: [15,15] dense, eV/Å²; res.frequencies: 9 modes, cm⁻¹
// Frame geometry is restored to R0; leftover kEnergy is the last displacement.
```

The section re-evaluates the energy after restore with a fresh method on
`cpu::Evaluate` (leftover `kEnergy` from the last displacement is not the
pin).

| Path | Role |
| ---- | ---- |
| `hessian.h` | `cpu::vib::Hessian` FD preset |
| `hessian_result.h` | `HessianResult` (\(H\), \(\omega\), \(n_{\mathrm{imag}}\)) |
| `hessian_workspace.h` | \(h=10^{-3}\) Å (`kHessianStepAngstrom`) |
| `tests/cpu/semiempirical/test_xtb_ipea_hessian.cpp` | Unit: `Hessian::build` accepts IPEA force kind (no `run`) |

### Validation (xtb)

Command: `XTBPATH=… xtb mol.xyz --gfn 1 --chrg 0 --uhf 0 --acc 0.01 --hess`;
twin `mol.xyz`, `xtb.out.txt`, `hessian.txt`, `vibspectrum.txt`. Conversion
\(H[\mathrm{eV}/\mathrm{Å}^2]=H[\mathrm{Eh}/a_0^2]\times 27.211386245988/(0.52917721092)^2\).

**xtb projected \(\omega\) [cm⁻¹]** (`vibspectrum.txt` modes 7–15; six TR
zeros projected out):

| Mode | Assignment | Degeneracy | xtb 6.7.1 |
| ---- | ---------- | ---------: | --------: |
| \(\omega_{\mathrm{T2}}\) | bend | 3 | **1541.68** |
| \(\omega_{\mathrm{E}}\) | bend | 2 | **1705.89** |
| \(\omega_{\mathrm{A1}}\) | stretch | 1 | **3548.78** |
| \(\omega_{\mathrm{T2}}\) | stretch | 3 | **3576.13** |

xtb also printed TOTAL ENERGY **−4.670476737294 Eh** (decomposition in the
`--hess` column under [Oracle](#oracle)), Cartesian
\(H_{C_x,C_x}=0.6813937129\) Eh/\(a_0^2\) (`hessian.txt`), 13 AO / 11 shells
/ 8 electrons, HOMO–LUMO 7.751 eV. IR intensities in the twin (T2 bend 9.62 /
E 0 / A1 0 / T2 stretch 53.36 km mol⁻¹) are **not** gated (the ATV FD
Hessian has no APT).

**Hard-coded gates:**

| Quantity | Atomiverse | xtb 6.7.1 | Tol | Notes |
| -------- | ---------: | --------: | --- | ----- |
| \(E\) | `kEnergy` (fresh Evaluate after restore) | **−4.670476737294 Eh** | \(5\times10^{-2}\) eV | same floor as `ipea-xtb-small` |
| \(\omega\) T2 bend (×3) | `res.frequencies(0..2)` | **1541.68 cm⁻¹** | 10 cm⁻¹ | projected, ascending |
| \(\omega\) E bend (×2) | `res.frequencies(3..4)` | **1705.89 cm⁻¹** | 10 cm⁻¹ | |
| \(\omega\) A1 stretch | `res.frequencies(5)` | **3548.78 cm⁻¹** | 10 cm⁻¹ | |
| \(\omega\) T2 stretch (×3) | `res.frequencies(6..8)` | **3576.13 cm⁻¹** | 10 cm⁻¹ | |
| \(H_{C_x,C_x}\) | `res.hessian(0,0)` | **66.21350 eV/Å²** | 0.2 eV/Å² | from \(0.6813937129\) Eh/\(a_0^2\) |
| Hessian shape | identity | \(15\times 15\) | hard | finite, \(\lvert H-H^\top\rvert\le 10^{-8}\) eV/Å² |
| Rotor | identity | nonlinear | hard | `res.is_linear == false`; 9 modes |
| \(n_{\mathrm{imag}}\) | identity | 0 | hard | all nine \(\omega>100\) cm⁻¹ |
| Geometry restore | identity | \(R_0\) | \(10^{-12}\) Å | `Hessian::run` restores displacements |

**Host residual (Release, `OMP_NUM_THREADS=1`, not a ctest gate):**
\(\lvert E_{\mathrm{ATV}}-E_{\mathrm{pin}}\rvert=8.90\) meV,
\(\Delta\omega = 0.36/0.64/1.30/0.26\) cm⁻¹ (T2-bend / E / A1 / T2-stretch;
Td degeneracies unsplit),
\(\lvert H_{C_x,C_x}-H_{\mathrm{pin}}\rvert=1.60\times10^{-2}\) eV/Å².

**Tolerance rationale.** Frequency / Cartesian-\(H\) tols absorb FD-of-\(G\)
vs xtb **analytical** Hessian plus IUPAC-2013 vs xtb isotope masses. The
10 cm⁻¹ / 0.2 eV/Å² gates sit well above the measured residual (max
\(\Delta\omega=1.30\) cm⁻¹; \(H\) residual 0.016 eV/Å²) and still catch a
sign, unit, or mass-weighting error on the stiff T2 stretches
(\(\sim 3576\) cm⁻¹). The 8.90 meV energy hole does not move \(\omega\) past
10 cm⁻¹.

### Performance

Fair wall asserted in the section (`ratio > 10` fails). Same *product*
(projected \(\omega\) of Td CH₄); **not** the same Hessian algorithm class
(ATV: analytic \(G\), numerical \(H\), 30 force evaluations + finalize; xtb
6.7.1: analytical Hessian).

| Twin | \(T_{\mathrm{xtb}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{xtb}}\) | ≤10× | Caveat |
| ---- | -----------------------: | -----------------------: | ----------------------------------------: | ---- | ------ |
| CH₄ IPEA `--hess` | **0.170** (1 thread, printed) | **\(1.04\times10^{-2}\)** | **≈ 0.061** | ✓ | `/usr/bin/time` process wall 0.43 s; the section pins the printed wall 0.17 s; ATV tables are static; xtb Hessian is analytical |

### Limitations

- **FD of analytic \(G\), not closed-form \(H\).** Cost is
  \(O(N_{\mathrm{atoms}})\) IPEA force evaluations; \(\omega\) tols absorb the
  path difference to xtb's analytical Hessian.
- No IR intensities; no polar-molecule (water) Hessian pin.
