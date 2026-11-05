# Plane-wave metallic smearing

> The regression runner has been removed. Regression commands below are historical; case sources and oracle records are retained as archived evidence, not current executable checks.

Plane-wave DFT for **metals** needs **fractional band occupations** and a Fermi
level \(\mu\) that conserves the valence electron count. Atomiverse exposes this
on the concrete method `cpu::pw::PW` (and `cuda::pw::PW`) as public members —
`smearing` (`Smearing::fermi_dirac` / `gaussian` / `methfessel_paxton`, order
`mp_order`), width `degauss` [Ry] and `extra_bands` — with no `*Config` bag and
no factory; `build(frame)` → `run(frame)` returns a `PWResult` whose `energy`
[Ry] is the **Mermin free energy** \(F = E_{\mathrm{KS}} - TS\), plus `energy_ha`,
`e_fermi` (\(\mu\), Ry), `n_elec_occ`, `e_entropy` (\(-TS\), Ry) and, for
collinear spin, `magnetization`. Occupations live in
`cpu/planewave/internal/smearing.h` (not a user type). The sections cover
restricted smearing (FCC Al) and collinear spin + smearing (BCC Fe) on CPU.

The case also carries CUDA sections without a tutorial page, each a same-settings
CPU↔CUDA dual gate on restricted Fermi–Dirac smear: `cuda-pw-metallic-smear`
(FCC Al free energy \(|E_\mathrm{cuda}-E_\mathrm{cpu}|\le 5\times10^{-3}\) Ry and
\(N_\mathrm{occ}=3\), the device twin of [pw-smearing-al](#pw-smearing-al)),
`cuda-pw-smear-forces` (stretched Al dimer analytic forces) and
`cuda-pw-smear-stress` (compressed FCC Al Nielsen–Martin stress). Their oracle
records live in the case `REFERENCE.md`.

Case: `ctest -R '^pw/smearing$'` — sources and oracle record in [`regressions/pw/smearing/`](../../../regressions/pw/smearing/). One section of the case per `##` heading below: `atv_regressions --case pw/smearing --section <name>`.

Roadmap: [dft.md](../../roadmap/dft.md) — Smearing for metals; spin + metallic
smearing (CPU PW). Requires `ATV_BUILD_PW=ON` (CPU build environment
`buildenv-gcc/2026.03-mpich`).

## Occupations and free energy

With reduced energy \(x = (\varepsilon-\mu)/\sigma\) and \(\sigma =\) `degauss`
(same unit as the eigenvalues inside the SCF):

| Kind | Spinless / spinor \(f(x)\) | Public enum |
| ---- | -------------------------- | ----------- |
| Fermi–Dirac | \(1/(1+e^{x})\) | `Smearing::fermi_dirac` |
| Gaussian | \(\tfrac12\mathrm{erfc}(x)\) | `Smearing::gaussian` |
| Methfessel–Paxton order \(N\) | MP series of order `mp_order` | `Smearing::methfessel_paxton` |

Restricted (closed-shell) runs double the spinless \(f(x)\in[0,1]\) to a band
occupation \(f_b\in[0,2]\); collinear runs give each KS spin-orbital
\(f_{\sigma kb}\in[0,1]\). \(\mu\) is found by Fermi bisection so that
\(\sum w_k f = Z_{\mathrm{val}}\cdot N_{\mathrm{atoms}}\); empty bands near
\(E_F\) come from `extra_bands` (Davidson needs them). With the entropy weight
\(s\) of the chosen smear,

\[
F = E_{\mathrm{KS}} - TS,\qquad
E_{\mathrm{entropy}} = -TS = -\sigma\sum w_k\, s .
\]

`Smearing::none` keeps fixed integer occupations
(\(n_{\mathrm{bands}}=\max(N_e/2,1)\)). A pseudopotential is not optional:
leaving both `upf_path` and `gth_name` empty throws at `build()` (no
bare-Coulomb path). Fermi–Dirac / Gaussian / MP-1 \(f(x)\) match the closed
forms to \(10^{-12}\) and the Fermi search recovers \(N_e\) within \(10^{-8}\) on
a synthetic spectrum (recorded; no unit test in the tree gates either). End-to-end gates use Fermi–Dirac
only.

## Oracle

No section runs CP2K or QE at ctest time. CP2K Quickstep
`&SMEAR METHOD FERMI_DIRAC` (+ `ADDED_MOS`, UKS / LSD for spin) is the
documented product counterpart, but GTH-GPW and NC-UPF PW are different
Hamiltonians, so CP2K energies are **not** an absolute-energy twin; the CP2K
inputs document SMEAR settings / algorithm only.

## Limitations (all sections)

- **No cold / Marzari–Vanderbilt** public enum; Gaussian and MP are unit-tested,
  the e2e gates are Fermi–Dirac.
- **GPW SMEAR** is not productized on this method (PW only); no hybrid /
  meta-GGA XC; no noncollinear spin / SOC.
- **Forces / stress under smear** are not gated on CPU here (restricted smear
  forces and stress are CPU↔CUDA gated in the CUDA sections; spin-polarized
  smear forces / stress are not gated).

## References

- N. D. Mermin, *Phys. Rev.* **137**, A1441 (1965)
- M. Methfessel and A. T. Paxton, *Phys. Rev. B* **40**, 3616 (1989)
- [Chapter 8 — Plane-Wave DFT](../ch08-plane-wave.md)

## pw-uks-fe-smearing — Collinear UKS + Fermi–Dirac smearing, BCC Fe {#pw-uks-fe-smearing}

### Summary

Collinear spin-polarized plane-wave Kohn–Sham with metallic Fermi–Dirac smearing
— the joint path `spin_polarized && smearing != none` that combines restricted
metal occupations ([pw-smearing-al](#pw-smearing-al)) and fixed-occupation UKS
([pw-spin-polarized-o2](pw-collinear-spin.md#pw-spin-polarized-o2) /
[pw-uks-kmesh-fe](pw-collinear-spin.md#pw-uks-kmesh-fe)). One shared chemical
potential \(\mu\) over spin channels; magnetization
\(M = N_\uparrow - N_\downarrow\) is **self-consistent** (multiplicity only seeds
the initial \(\rho_\uparrow/\rho_\downarrow\) split). System: **BCC Fe**
primitive cell, SG15 ONCV PBE Fe NC-UPF, MP \(2\times2\times2\), solved twice at
`degauss` \(\sigma = 0.02\) Ry and \(\sigma/4 = 0.005\) Ry. Every gate is an exact
statement about the Mermin free energy, the electron count or the ferro ground
state — no absolute energy is frozen.

### Method

- Separate \(\alpha/\beta\) KS orbitals and densities \(\rho_\uparrow\),
  \(\rho_\downarrow\) at every \(\mathbf{k}\) of the MP mesh (LSD channel as in
  `pw-spin-polarized-o2`).
- Hartree from total density \(\rho = \rho_\uparrow + \rho_\downarrow\).
- Spin-dependent XC \(V_{\mathrm{xc}}^\sigma(\rho_\uparrow,\rho_\downarrow)\)
  (`XCType::PBE`, true collinear spin-GGA).
- Dual Davidson diagonalizations per **k** and spin; k-weights from
  `generate_mp_mesh`.
- One shared \(\mu\) for both spin channels (Fermi bisection on the joint
  spectrum):

\[
N_{\mathrm{occ}}
  = \sum_{\sigma,k,b} w_k\, f_{\sigma kb}(\varepsilon_{\sigma kb},\mu,\sigma)
  = Z_{\mathrm{val}}\cdot N_{\mathrm{atoms}},
\qquad
M = \sum_{k,b} w_k\bigl(f_{\uparrow kb}-f_{\downarrow kb}\bigr)
\]

(for BCC Fe, \(Z_{\mathrm{val}}=16\), \(N_{\mathrm{atoms}}=1\) → \(N_e=16\)).
With `Smearing::none`, fixed integer fillings enforce \(M = m-1\) from
multiplicity \(m\); with metallic smear \(M\) is free under SCF (ferro expected
for BCC Fe with a ferro seed).

**Observables**

| Symbol | Definition | Unit |
| ------ | ---------- | ---- |
| \(F\) | Mermin free energy (`PWResult::energy`) | Ry |
| \(E_{\mathrm{Ha}}\) | \(F/2\) (`PWResult::energy_ha`) | Ha |
| \(M\) | \(N_\uparrow - N_\downarrow\) (`PWResult::magnetization`) | e⁻ |
| \(N_{\mathrm{occ}}\) | \(\sum w f\) (`PWResult::n_elec_occ`) | e⁻ |
| \(E_F\) | chemical potential (`PWResult::e_fermi`) | Ry |
| \(E_{\mathrm{ent}}\) | \(-TS\) (`PWResult::e_entropy`) | Ry |

### Implementation

```cpp
// BCC Fe primitive, cubic a = 2.87 Å, one atom at origin
cpu::Frame f(1);
f.span(kAtom_atomic_number)[0] = 26;       // Z(Fe)
f.span(kAtom_mass)[0] = 55.845;
f.span(kAtom_x)[0] = f.span(kAtom_y)[0] = f.span(kAtom_z)[0] = 0.0;
const double h = 2.87 / 2.0;
Mat3 cell;                        // primitive BCC: (-h,h,h), (h,-h,h), (h,h,-h)
cell(0, 0) = -h; cell(0, 1) =  h; cell(0, 2) =  h;
cell(1, 0) =  h; cell(1, 1) = -h; cell(1, 2) =  h;
cell(2, 0) =  h; cell(2, 1) =  h; cell(2, 2) = -h;
f.box.set_cell(cell);

PW pw{std::make_shared<UpfPseudopotential>(ATV_REG_DATA_DIR "/pw/Fe_ONCV_PBE-1.2.upf"),
      12.0,                        // algorithm pin (not production ecut)
      2,                           // 2×2×2 MP
      cpu::scf::XCType::PBE};
pw.tol = 2e-3;                     // density L2
pw.max_iter = 30;
pw.mix_beta = 0.3;
pw.pulay_depth = 5;
pw.spin_polarized = true;
pw.multiplicity = 5;               // seed M≈4; free under smear
pw.smearing = Smearing::fermi_dirac;
pw.degauss = 0.02;                 // Ry (second solve: 0.005)
pw.extra_bands = 4;                // per spin

pw.build(f);
const PWResult& r = pw.run(f);
// r.energy [Ry] free energy; r.magnetization; r.e_fermi [Ry];
// r.n_elec_occ ≈ 16; r.e_entropy [Ry]; r.converged, r.iterations
```

The section calls `PW` directly; a Driver can mount a moved `PW` as a nested
node, in which case the Frame energy is eV. Spin × smear branch:
`src/cpu/planewave/pw_solver.cpp`. Spin symmetry (no unit test in the tree
gates it): with `multiplicity = 1` both channels are seeded identically and
every operator is spin-symmetric, so \(M \equiv 0\) exactly; on a multi-k mesh
the shared \(\mu\) conserves the electron count.

### Settings

| Setting | Value |
| ------- | ----- |
| System | BCC Fe primitive, 1 atom at origin, cubic \(a = 2.87\) Å |
| PP | SG15 ONCV PBE Fe (Hamann ONCVPSP), `tests/data/pw/Fe_ONCV_PBE-1.2.upf` (\(Z_{\mathrm{val}}=16\)) |
| XC | PBE (`XCType::PBE`) |
| \(E_{\mathrm{cut}}\) | 12 Ry (algorithm pin for ≤60 s wall — **not** production ONCV completeness, which needs ≈80 Ry) |
| **k**-mesh | \(2\times 2\times 2\) (`k_mesh = 2`) |
| Smearing | Fermi–Dirac, `degauss = 0.02` Ry and `0.005` Ry |
| Extra bands | 4 (per spin) |
| Multiplicity seed | 5 (initial \(M\approx 4\); free under smear) |
| SCF | density L2 tol \(2\times10^{-3}\), `max_iter=30`, `mix_beta=0.3`, `pulay_depth=5`; ~11 SCF iters |

### Gates (both σ)

| Gate | Condition | Measured | Notes |
| ---- | --------- | -------: | ----- |
| SCF | converged at both σ | — | fails if `!converged` |
| Electron count | \(N_{\mathrm{occ}} = 16\), tol \(1\times10^{-4}\) | **16** | k-weight normalisation and ONE shared \(\mu\) |
| Entropy sign | \(-TS \le 0\) | — | \(S \ge 0\) |
| \(\partial F/\partial T = -S\) | \(F(0.02) - F(0.005) < -10^{-2}\) Ry | \(-5.13\times10^{-2}\) Ry | SCF wobble ≈ \(4\times10^{-3}\) Ry at this tol; reporting \(E\) instead of \(F\), or a wrong entropy sign, fails |
| \(TS \to 0\) | \(|{-}TS|\) shrinks with σ | \(3.71\times10^{-2}\ \to\ 5.19\times10^{-3}\) Ry | |
| Ferro ground state | \(M > 0.1\) | \(0.799\) (σ=0.02), \(0.999\) (σ=0.005) | tightly converged (`tol=1e-6`) small-σ limit \(M \to 1.00\) |
| Fermi level | finite, \(\lvert E_F\rvert < 10^3\) Ry | \(\approx 2.10\) | among bands; not NaN |

BCC Fe is ferromagnetic with an experimental moment of ~2.2 \(\mu_B\) (Kittel);
this 12 Ry / 8-k setup undershoots it by design, so the gate is the ferro sign
and its stability under smearing, not the magnitude.

**No absolute free-energy pin.** BCC Fe at 12 Ry with an 8-point mesh has no
citable external absolute energy, and a σ→0 extrapolation is not an anchor: the
discrete 8-k spectrum gives \(TS \propto \sigma^{1.3}\) rather than the
Sommerfeld \(\sigma^2\), and the σ scan crosses a magnetic transition (tightly
converged \(M = 0.026,\ 0.807,\ 0.995,\ 0.999\) at
\(\sigma = 0.04,\ 0.02,\ 0.01,\ 0.005\) Ry). The current solver gives
\(F(\sigma{=}0.02) = -205.25820008\) Ry (\(M = 0.799445\),
\(E_F = 2.095300\) Ry). The value **−205.8365363098 Ry** (± **0.05 Ry**, with
\(M \approx 0.991\), \(E_F \approx 2.16\)) is not a valid reference for this
setup: it reproduces an LSDA solve with the spin-↓ channel on an inconsistent
\(V_\mathrm{eff}\).

**CP2K note.** Quickstep UKS / LSD + `FERMI_DIRAC` + Fe GTH (if re-run
offline), SMEAR width \(T_e \approx 3158\) K matching `degauss ≈ 0.02` Ry; the
host CP2K is not loadable as a fair NC-UPF PW twin and no CP2K energy is
compared.

### Performance

**`perf_exempt`** — no fair single-rank NC-UPF PW wall twin vs CP2K GTH-GPW
(same class as [pw-uks-kmesh-fe](pw-collinear-spin.md#pw-uks-kmesh-fe));
\(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) is n/a. \(T_{\mathrm{ATV}}\):
**~20 s** per solve (OMP=8, single rank); **~34 s** for the two-solve section
(OMP=1). Wall budget: e2e ≤ 60 s single-rank CPU.

### Limitations

- **Algorithm ecut (12 Ry), not production Fe completeness.** Absolute \(F\)
  and \(M\) are not chemically converged; \(M\approx 1\) undershoots
  experimental BCC Fe (~2.2 \(\mu_B\)). Higher ecut / denser k-mesh would need
  a new external anchor, not a re-freeze.
- **No spin-polarized forces/stress under smear**; no GPW UKS + SMEAR; no CUDA
  spin + smear.
- Restricted smear without spin is [pw-smearing-al](#pw-smearing-al);
  fixed-occupation multi-k UKS is
  [pw-uks-kmesh-fe](pw-collinear-spin.md#pw-uks-kmesh-fe).

## pw-smearing-al — Restricted Fermi–Dirac smearing, FCC Al {#pw-smearing-al}

*No section of the case runs this variant.*

### Summary

Spin-unpolarized plane-wave KS SCF with Fermi–Dirac smearing on the **FCC Al**
primitive cell, NC UPF, `degauss = 0.02` Ry (reciprocal-space Poisson / FFT XC /
Davidson as in [ch08-plane-wave](../ch08-plane-wave.md)). The CPU path is
exercised as the same-settings oracle of the CUDA section
`cuda-pw-metallic-smear`; there is no CPU-only end-to-end gate. The energy
values below are frozen ATV numbers on identical ATV settings, not a GTH-GPW ↔
NC-UPF twin.

### Method

Closed-shell occupations \(f_b\in[0,2]\); for FCC Al \(N_e = 3\). Extra empty
bands above \(\lceil N_e/2\rceil\) come from `extra_bands`. FD entropy in band
form: \(s = -2[f\ln f+(1-f)\ln(1-f)]\), so
\(E_{\mathrm{entropy}} = -TS = -\sigma\sum_{k,b} w_k\, s_{kb}\).

### Implementation

```cpp
// FCC Al primitive, a = 4.05 Å, one atom at origin
cpu::Frame f(1);
f.span(kAtom_atomic_number)[0] = 13;
f.span(kAtom_mass)[0] = 26.9815385;
f.span(kAtom_x)[0] = f.span(kAtom_y)[0] = f.span(kAtom_z)[0] = 0.0;
const double h = 4.05 / 2.0;
Mat3 cell;                         // (0,h,h), (h,0,h), (h,h,0)
cell(0, 0) = 0; cell(0, 1) = h; cell(0, 2) = h;
cell(1, 0) = h; cell(1, 1) = 0; cell(1, 2) = h;
cell(2, 0) = h; cell(2, 1) = h; cell(2, 2) = 0;
f.box.set_cell(cell);

PW pw{std::make_shared<UpfPseudopotential>(ATV_REG_DATA_DIR "/pw/Al_pz_nc.upf"), 20.0, 2};  // NC, LDA/PZ, Z_val = 3
pw.tol = 1e-5;
pw.max_iter = 60;
pw.mix_beta = 0.4;
pw.pulay_depth = 5;
pw.smearing = Smearing::fermi_dirac;
pw.degauss = 0.02;   // Ry
pw.extra_bands = 4;  // → 6 bands for N_e = 3

pw.build(f);
const PWResult& r = pw.run(f);
// r.energy [Ry] free energy; r.e_fermi [Ry]; r.n_elec_occ ≈ 3;
// r.e_entropy [Ry] (−TS); r.converged, r.iterations
```

### Validation

| Setting | Value |
| ------- | ----- |
| Geometry | FCC Al primitive, \(a=4.05\) Å, 1 atom at origin |
| PP | `tests/data/pw/Al_pz_nc.upf` (NC, LDA/PZ, \(Z_{\mathrm{val}}=3\)) |
| \(E_{\mathrm{cut}}\) | 20 Ry |
| k-mesh | Monkhorst–Pack \(2\times2\times2\) |
| Smearing | Fermi–Dirac, `degauss = 0.02` Ry |
| Extra bands | 4 (\(\Rightarrow\) 6 bands for \(N_e=3\)) |
| SCF | `tol=1e-5`, `max_iter=60`, `mix_beta=0.4` |

| Quantity | Value | Tol | Notes |
| -------- | ----: | --- | ----- |
| Free energy \(F\) | **−7.54535571 Ry** | **0.05 Ry** | frozen ATV offline value (Arrhenius x86_64) |
| \(N_{\mathrm{occ}}=\sum w f\) | **3.0** | \(10^{-5}\) | valence count |
| \(E_F\) | finite, \(\lvert E_F\rvert < 10^3\) Ry | — | among bands; not NaN |

**Offline CP2K twin (algorithm documentation).**

| Item | Value |
| ---- | ----- |
| Code | **CP2K 2026.1** (`cp2k.psmp`, git:5e54ba2) |
| Method | Quickstep GPW, PADE, `GTH-PADE-q3` + `DZVP-MOLOPT-SR-GTH`, CUTOFF 200 Ry |
| Cell | same primitive FCC \(a=4.05\) Å |
| SMEAR | FERMI_DIRAC, \(T_e = 3157.75\) K (\(\approx 0.02\) Ry \(k_B T\)) |
| k-points | Monkhorst–Pack \(2\times2\times2\) |
| Example \(E_{\mathrm{tot}}\) | ≈ −2.096 Ha (grid-dependent; **not** compared to the ATV Ry value) |

### Performance

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× |
| ---- | ------------------------: | ----------------------: | ----: | ---- |
| `al_fermi_dirac.inp` vs ATV `PW` (single rank) | 211 | **1.70** | **0.008** | ✓ |

\(r = T_{\mathrm{atv}} / T_{\mathrm{cp2k}} \approx 0.008 \ll 10\). Fairness
note: ATV NC-UPF PW at 20 Ry / \(2^3\) k vs CP2K GPW GTH + MOLOPT at CUTOFF
200 Ry — different basis/PP/grid; do **not** treat \(r\) as a same-Hamiltonian
throughput claim.

### Limitations

- **Spin-unpolarized only.** Collinear spin + smear is
  [pw-uks-fe-smearing](#pw-uks-fe-smearing); fixed-occupation spin is
  `pw-spin-polarized-o2` / [pw-uks-kmesh-fe](pw-collinear-spin.md#pw-uks-kmesh-fe).
- **Absolute energy is ATV-frozen, not CP2K-matched.**
- No multi-system standing CP2K perf suite.

Design record: git `72ebb870` (`.claude/specs/pw-smearing-al/`).
