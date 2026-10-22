# Density-Functional Theory

Tutorial: [KS](../tutorial/ch05-kohn-sham-dft.md) · [hybrids](../tutorial/ch06-hybrid-dft.md) · [PW](../tutorial/ch08-plane-wave.md) · [GPW](../tutorial/ch09-gpw.md).

Molecular KS shares the Gaussian SCF solver with
[Hartree–Fock](hartree-fock.md). Periodic DFT (plane-wave + GPW) lives in
§4–6. Legend: [Roadmap index](index.md).

**Blocked on periodic engines** (no CP2K-comparable twin in tree): hybrids,
meta-GGA, GAPW/PAW, stock VV10. rVV10 ships.

## 1. Molecular Kohn–Sham

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| RKS / UKS | ✓ | ○ | No Gaussian KS on device |
| ROKS | ✓ | ○ | Energy + analytic G |
| RKS / UKS / ROKS analytic gradient | ✓ | ○ | Incl. XC + Becke grid response |
| RKS / UKS Hessian | ✓ | ○ | `Hessian<RKS>::run` / `Hessian<UKS>::run` — FD of analytic forces (not full CPKS) |
| Molecular DFT+U / ΔSCF-DFT / GKS | ○ | ○ | Periodic DFT+U is GPW (§4) |

## 2. Functionals (molecular + shared)

Hand-coded; libxc is an optional test oracle only (`ATV_BUILD_XC_ORACLE`).

| Rung / functional | CPU | CUDA | Notes |
| ----------------- | :-: | :--: | ----- |
| LSDA (Slater+VWN5) | ✓ | ✓ | CUDA = plane-wave only |
| PW92 | ✓ | ○ | LDA base of PBE |
| PBE | ✓ | ✓ | CUDA = PW closed-shell + collinear spin-GGA |
| BLYP | ✓ | ○ | Molecular pin; GPW reachable, not separately pinned |
| PBEsol / revPBE / RPBE | ◐ | ○ | Molecular + GPW reachable; libxc unit parity only, no end-to-end pin (GPW functional-swap cases removed) |
| optB88 | ◐ | ○ | GPW (with vdW-DF); no end-to-end pin (cases removed) |
| BP86 / other GGA catalog | ○ | ○ | |
| Meta-GGA (TPSS, r2SCAN, …) | ○ | ○ | No τ channel in grid engine |
| B3LYP / PBE0 | ✓ | ○ | Gaussian only |
| CAM-B3LYP | ✓ | ○ | RKS energy foothold |
| Other RSH / screened hybrids | ○ | ○ | |
| B2PLYP | ✓ | ○ | Hybrid KS + scaled RMP2 |
| Other double hybrids | ○ | ○ | |
| HF-3c / PBEh-3c / … | ○ | ○ | |

## 3. Dispersion and nonlocal correlation

| Method | Molecular | GPW | CUDA | Notes |
| ------ | :-------: | :-: | :--: | ----- |
| D3 zero | ○ | ✓ | ○ | Forces + stress; ATM |
| D3-BJ | ✓ | ✓ | ○ | Forces + stress; ATM |
| D4 | ✓ | ✓ | ○ | Forces + stress; ATM (GPW) |
| TS | ○ | ✓ | ○ | Free-atom + Hirshfeld-scaled |
| MBD@rsSCS | ○ | ✓ | ○ | Free-atom, Hirshfeld, periodic Ewald |
| XDM | ○ | ✓ | ○ | |
| vdW-DF / rVV10 | ○ | ✓ | ○ | SC nonlocal potential + F/σ |
| vdW-DF2 / optB88-vdW | ○ | ◐ | ○ | Same kernel path; no end-to-end pin (cases removed) |
| VV10 (stock) | ○ | ○ | ○ | **Blocked** — no CP2K `TYPE VV10` |

GPW dispersion runs at Γ and multi-k, RKS and UKS where productized. D3 / D4 / TS
are pinned at Γ; multi-k pins cover vdW-DF, MBD and XDM. Pin
detail: [tutorial/methods catalog](../tutorial/methods/catalog.md).

## 4. Periodic Kohn–Sham

Two engines: **plane-wave** (`cpu::pw::PW` / `cuda::pw::PW`) and **GPW**
(`cpu::gpw::GpwDft` / `GPWEvaluator`). CUDA GPW is a partial product slice
(◐), not full CPU parity.

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| Plane-wave DFT | ✓ | ✓ | NC-UPF or in-tree GTH; CUDA ⊂ CPU |
| GPW DFT | ✓ | ◐ | Public method is CPU; device kernels + first product slices |
| Γ-point RKS | ✓ | ◐ | GPW; CUDA first-slice |
| k-point RKS | ✓ | ✓ | PW on both; GPW CPU; CUDA GPW multi-k foothold |
| Γ / k-point UKS | ✓ | ◐ / ○ | PW+GPW CPU; CUDA PW collinear; CUDA GPW Γ UKS energy |
| Periodic LDA / GGA | ✓ | ✓ | No periodic hybrid / meta-GGA |
| Periodic DFT+U | ✓ | ○ | GPW Dudarev; F/σ at Γ and multi-k |
| Periodic dispersion | ✓ | ○ | See §3 |
| Periodic RI-MP2 / SCS-MP2 (Γ) | ✓ | ○ | GPW |
| Periodic UMP2 / multi-k MP2 | ◐ | ○ | GPW; no external pin, self-pinned cases removed |
| Periodic dRPA / URPA | ◐ | ○ | Γ, multi-k, finite-q reachable; no external pin, self-pinned cases removed |
| Periodic G0W0 / UGW | ◐ | ○ | Γ, multi-k, finite-q reachable; no external pin, self-pinned cases removed |
| GAPW / PAW / USPP | ○ | ○ | NC-UPF or GTH only |

### Plane-wave depth (CPU)

| Capability | Status | Notes |
| ---------- | :----: | ----- |
| Analytic ionic forces | ✓ | Local HF + Ewald + KB; FD oracle remains |
| Analytic cell stress | ✓ | Nielsen–Martin |
| True GGA PBE + collinear spin-GGA | ✓ | Incl. k-mesh UKS |
| Metallic smearing | ✓ | FD / MP; free-energy path |
| Bands / DOS | ✓ | Incl. spin + metallic |
| CELL_OPT | ✓ | |
| BOMD | ✓ | Foothold |
| Berry / MLWF | ✓ | Incl. spin |
| DFPT phonons | ✓ | Γ / finite-q / incommensurate × GGA × spin × smear |

### Plane-wave depth (CUDA)

| Capability | Status | Notes |
| ---------- | :----: | ----- |
| LSDA + PBE energy | ✓ | Device XC |
| Analytic F/σ (closed-shell) | ✓ | LSDA F/σ + PBE σ pinned; closed-shell PBE forces not separately pinned |
| Collinear spin F/σ | ✓ | Incl. spin-PBE |
| Metallic smear energy + F/σ | ◐ | Restricted path product; multi-atom stress residual; spin×smear stress ○ |
| Host Ritz + SCF control | — | Hot path (H·ψ, FFT, Poisson, V_eff) on device |

### GPW depth (CPU)

Everything below is available for **RKS and UKS**, **Γ and Monkhorst–Pack**,
via **DIIS or OT**, unless noted.

| Area | Status | Notes |
| ---- | :----: | ----- |
| Functionals | ✓ | PADE/LSD, PBE pinned; PBEsol, revPBE, RPBE, BLYP, optB88 reachable without a GPW pin (see §2) |
| DFT+U + smearing + multigrid | ✓ | Combinable with OT |
| Analytic F + analytic stress | ✓ | Fixed-P + Pulay; CELL_OPT; FD phonons |
| Dispersion / vdW | ✓ | §3 full set |
| Bands / DOS / Berry / MLWF | ✓ | Spin + metallic |
| BOMD / CPMD / SGCP / XL-BOMD | ✓ | See [MD § AIMD](molecular-dynamics.md#5-ab-initio-md) |
| Periodic correlated | ◐ | RI-MP2 / SCS-MP2 at Γ pinned; RPA, GW, UMP2, multi-k not (§4 table) |

**Public shape:** `GpwDft` owns kinds and parameters as members and is Frame
in / Frame out (`run(frame)` / `forces(frame)` / `stress(frame)`); `GPWEvaluator(std::move(dft))`
binds it into an outer driver. Keep `local_pp_on_grid` consistent between energy and gradient
calls.

### CUDA GPW

| Capability | Status | Notes |
| ---------- | :----: | ----- |
| Γ RKS DIIS energy + F/σ slice | ◐ | `cuda::gpw::GpwDft`; CPU-twin gates; no CP2K pin |
| Γ UKS energy | ◐ | Padé LSD; F/σ depth open |
| Multi-k RKS energy | ◐ | H₂ foothold + device mid-SCF pieces |
| Full method parity with CPU | ○ | Table stays ◐ until production depth + external pins |

## 5. Periodic response and analysis

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| Periodic analytic forces | ✓ | ◐ | PW + GPW CPU; CUDA = PW subset + GPW free-fn slice |
| Periodic analytic stress | ✓ | ◐ | Same split |
| DFPT | ✓ | ○ | PW only |
| FD phonons / ω(q) | ✓ | ○ | `cpu::vib::Phonon` + dispersion |
| Berry polarization | ✓ | ○ | PW + GPW; spin |
| MLWF | ✓ | ○ | PW + GPW; spin |

## 6. Periodic infrastructure

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| Monkhorst–Pack k-mesh | ✓ | ✓ | |
| Block Davidson | ✓ | ✓ | |
| NC-UPF (KB) | ✓ | ✓ | |
| GTH PP | ✓ | ○ | GPW |
| FFT Poisson | ✓ | ✓ | |
| Martyna–Tuckerman Poisson | ✓ | ○ | |
| OT minimizer | ✓ | ○ | GPW |
| Multigrid collocation | ✓ | ○ | GPW; CUDA MG energy slice exists |
| Metallic smearing | ✓ | ◐ | CUDA = restricted PW (+ spin×smear F landed; stress open) |
| Bands / DOS post-process | ✓ | ○ | |
