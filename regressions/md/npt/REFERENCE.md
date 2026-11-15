# md/npt

One case; every section below is run by `atv_regressions --case md/npt` and keeps its own oracle record.


## npt_lj — MTK NoseHooverNPT on a 64-atom LJ liquid

**One case, multi-device:** `reg_npt_lj_mtk --device=cpu|cuda|both`.
Named `npt_lj_mtk` so the binary does not collide with `mc/npt_lj`.

Public API only: `MD` + `NoseHooverNPT{T, tau_T, P, tau_P, dt}` + `PairLJCut`
+ `NeighborList`. No per-atom Verlet on CUDA.

| Field | Value |
|---|---|
| System | 6³ sc LJ (216 atoms), ρ = 0.8, ε = σ = 1, r_c = 2.5 (L/2 > r_c) |
| Ensemble | MTK NPT, T* = 1 (T = 1/k_B K with ε = 1 eV), τ_T = 0.1, P = 0, τ_P = 1, dt = 0.005 |
| Steps | 80 |
| CPU↔CUDA | \|V_cuda − V_cpu\| / V < 1e-4 |

GH200 **1336653** (T\*=1, ε=1 eV): CPU `V=269.425432`, CUDA
`V=269.425466`, `|ΔV|/V = 1.27e-7`.
LAMMPS twin deck uses a literal \(a_0\) so ρ=0.8. A digit-identical
LAMMPS volume is not claimed: ATV/LAMMPS RNGs and lj-vs-fs time units
differ. The landed gate is CPU↔CUDA.

The 1/2-step initial-force and represented-cell parity cases also run here,
with their original relative bounds 1e-8 and 1e-6. They use T=1 K and
CUDA-representable input cells/positions; they are not unit tests.
The scalar NHC half-step follows Martyna, Klein and Tuckerman, JCP 97,
2635 (1992), doi:10.1063/1.463940; MTK pressure propagation follows
Martyna, Tobias and Klein, JCP 101, 4177 (1994), doi:10.1063/1.467468.


## nph_lj — isotropic NPH enthalpy conservation (LJ)

### What is checked

Public thermostat-free `cpu::md::NphIntegrator` (isotropic NPH):

1. Extended enthalpy
   \(H = E_\mathrm{kin}+E_\mathrm{pot}+P V+\tfrac12 W\dot\varepsilon^2\)
   written to `kConserved` stays within a relative drift gate over production.
2. Volume remains positive and non-runaway.

### Physical setup

| Quantity | Value | Note |
| -------- | ----- | ---- |
| Potential | LJ argon-like | \(\varepsilon=0.0104\,\mathrm{eV}\), \(\sigma=3.4\,\mathrm{Å}\) |
| Lattice | Compressed FCC \(a=5.2\,\mathrm{Å}\), 108 atoms | Dense start |
| Ensemble | NPH (no atom thermostat) | |
| \(P\) | 0 eV/Å³ | Hydrostatic |
| \(\tau_P\) | 1000 fs | Barostat mass only |
| \(dt\) | 1 fs | warm 100 + prod 600 |

### Gates

| Check | Bound |
| ----- | ----- |
| \(\max\|\Delta H\|/\max(1,\|H_0\|)\) | \(\le 0.15\) |
| volume | \(0 < V < 10 V_0\) |

### Literature

Martyna, Tobias, Klein, *J. Chem. Phys.* **101**, 4177 (1994) — MTK barostat;
NPH is the thermostat-free subset (conserved extended enthalpy).

### Re-run (2026-11-15)

The case now runs `units metal` (times passed in ps); before, it ran the
default `lj` style with eV/Å/fs inputs and failed its volume gate. Compiled
on its own as described under *npt_pme_nacl — Re-measurement*: PASS,
max |ΔH|/max(1,|H₀|) = 5.3×10⁻⁶ (H₀ = −5.673 eV), V 3796 → 4559 Å³.


## pr_npt_lj — Parrinello–Rahman NPT on compressed LJ argon

### What is checked

Public `cpu::md::ParrinelloRahmanNPT` on compressed FCC argon (LJ):

1. Cell volume **increases** from an over-compressed start at \(P_\mathrm{target}=0\)
   (pressure relief).
2. Volume stays within a safe multiple of \(V_0\) (no runaway collapse/explosion).
3. Off-diagonal cell entries remain bounded relative to the edge length under
   hydrostatic target (isotropic liquid limit of full-cell PR).
4. Extended MTK Hamiltonian on `kConserved` is finite.

### Physical setup

| Quantity | Value | Note |
| -------- | ----- | ---- |
| Potential | LJ \(\varepsilon=0.0104\,\mathrm{eV}\), \(\sigma=3.4\,\mathrm{Å}\) | Argon-like |
| Lattice | FCC \(a=5.1\,\mathrm{Å}\), \(3^3\) cells (108 atoms) | Compressed vs Ar eq. |
| \(T\) | 100 K | Thermostat target |
| \(P\) | 0 eV/Å³ | Hydrostatic |
| \(\tau_T,\tau_P\) | 100 fs, 1000 fs | |
| \(dt\) | 1 fs | 800 steps |

### Gates

| Check | Bound |
| ----- | ----- |
| \(V_1 > V_0\) | expansion from compressed start |
| \(V_1 < 5 V_0\) | no runaway |
| shear / edge | < 0.20 |
| \(H\), \(V\) | finite, \(V>0\) |

### Literature

Parrinello & Rahman, *J. Appl. Phys.* **52**, 7182 (1981);
Martyna, Tobias, Klein, *J. Chem. Phys.* **101**, 4177 (1994).

### Re-run (2026-11-15)

The case now runs `units metal` (times passed in ps); before, it ran the
default `lj` style with eV/Å/fs inputs and failed its shear gate. Compiled
on its own as described under *npt_pme_nacl — Re-measurement*: PASS,
V 3582 → 5257 Å³.


## Reference — rock-salt NaCl under PME electrostatics at zero pressure

### System

Rock salt (B1), `4³ = 64` ions, site charges alternating `(−1)^{i+j+k}` so the
cell is neutral. The conventional cubic lattice constant is `a`, the
nearest-neighbour spacing `d = a/2`, and the periodic box edge `L = 4d = 2a`.

Experimental NaCl lattice constant at 298 K:

```text
a₀ = 5.6402 Å        (d = 2.8201 Å)
```

(Wyckoff, *Crystal Structures* Vol. 1, 2nd ed., Interscience 1963; the same
value is the room-temperature figure used to calibrate powder diffractometers.)

### Interactions

**Electrostatics.** Real-space `CoulLong` + reciprocal `KSpacePME`, sharing
`α = 3.5 / r_c` with `r_c = 5.0 Å` (`erfc(3.5) ≈ 7×10⁻⁷`). Smooth PME of
Essmann et al., *J. Chem. Phys.* **103**, 8577 (1995); grid spacing 0.5 Å,
order-6 B-splines.

`r_c` is deliberately below `L/2 = 5.64 Å` at the experimental spacing, which
is what the minimum-image convention requires; it leaves ~11 % of contraction
headroom before the convention would bite, and the run never uses it.

**Short range: Tosi–Fumi (Born–Mayer–Huggins).** Fumi & Tosi, *J. Phys. Chem.
Solids* **25**, 31 and 45 (1964) — the standard rigid-ion alkali-halide model,
fitted to the alkali-halide lattice constants and compressibilities:

```text
U_ij(r) = A_ij exp[(σ_ij − r)/ρ] − C_ij/r⁶ − D_ij/r⁸
A_ij = b·c_ij ,  b = 0.338×10⁻¹⁹ J ,  c_ij = 1 + z_i/n_i + z_j/n_j
ρ = 0.317 Å ,  σ_ij = σ_i + σ_j ,  σ_Na = 1.170 Å , σ_Cl = 1.585 Å
```

With `n = 8` for both ions, `c_{++} = 1.25`, `c_{+−} = 1.00`, `c_{−−} = 0.75`.
Converting the published SI table (`1×10⁻¹⁹ J = 1/e eV`,
`1×10⁻⁷⁹ J·m⁶ = 1×10⁻¹⁹ J·Å⁶`, `1×10⁻⁹⁹ J·m⁸ = 1×10⁻¹⁹ J·Å⁸`):

| pair | A [eV] | σ [Å] | C [eV Å⁶] | D [eV Å⁸] |
|---|---|---|---|---|
| Na⁺–Na⁺ | 0.2637038 | 2.340 | 1.0485735 | 0.4993207 |
| Na⁺–Cl⁻ | 0.2109630 | 2.755 | 6.9904902 | 8.6756976 |
| Cl⁻–Cl⁻ | 0.1582223 | 3.170 | 72.4015053 | 145.4271614 |

with the published `C = 1.68, 11.2, 116 ×10⁻⁷⁹ J·m⁶` and
`D = 0.8, 13.9, 233 ×10⁻⁹⁹ J·m⁸`.

**Erratum.** Until 2026-11-14 the case carried C and D divided once more
by 1.602177 (0.6553585, 4.3690564, 45.2509408 and 0.3114513, 5.4176299,
90.8139570): 38 % too little dispersion. The numbers below were re-measured
with the corrected table on 2026-11-15 (see *Re-measurement*). The extra
dispersion moves the static zero-pressure spacing from +0.6 % above
experiment to −0.15 % below it, so the equilibrium bracket is now two-sided.

`cpu::PairBorn` evaluates `A exp((σ−r)/ρ) − C/r⁶ **+** D/r⁸` (LAMMPS
`pair_style born`), so the case passes **negated** `D`. Energies are shifted to
zero at the cutoff (`pair_modify shift yes`); a constant per pair does not
change the pressure, only the energy zero, and no pair sits near `r_c`.

### What is checked

#### 1. Madelung constant (bare point charges, no short range)

```text
E = −½ N M k_e / d ,    M = 1.747564594633182
```

(Ziman, *Principles of the Theory of Solids*, 2nd ed., Cambridge 1972, §2.3.)
Measured `|E/E_ref − 1| = 3.8×10⁻⁸`, tolerance `5×10⁻⁴` (the PME
discretization budget at 0.5 Å / order 6).

#### 2. Uniform-scale law

Pure Coulomb is homogeneous of degree −1 in length, so scaling every
coordinate and the cell by `s` gives `E(s) = E(1)/s`. Measured
`|E(s)·s/E(1) − 1| = 1.4×10⁻⁸`. This is the cache-invalidation oracle: a
`KSpace` that reused build-time influence tables at a new volume fails it.

#### 3. The virial identity — `tr(σ) = −3V dE/dV`

**This is the decisive gate, and it needs no literature value at all.** Scale
every coordinate *and* the cell by the same λ; the fractional coordinates are
then fixed and

```text
P ≡ tr(σ)/(3V) = −dE/dV
```

exactly, for the whole Hamiltonian at once. The case evaluates the right-hand
side by central difference (`h = 10⁻⁶`) against the engine's own `kStress`.
Measured `|rel| = 1.9×10⁻⁸`, tolerance `10⁻⁵`.

A barostat integrates the pressure. If any term contributes energy and force
but no virial — or the wrong virial — the pressure is systematically wrong,
nothing about the energy or the forces looks amiss, and the cell walks to a
volume the Hamiltonian never asked for. Two shipped defects had exactly that
signature and are pinned here and in
`tests/cpu/potentials/test_pair_virial.cpp`:

- `cpu::PairBorn` wrote **no** `kStress` at all;
- the `cpu::PairLJCut` xsimd ket loop guarded its six batch reduction chains
  with a loop-invariant `if (do_virial)`, and GCC 14.3 at the project's own
  Release flags (`-O3 -march=native`, AVX-512) dropped most of the
  accumulation — energy and forces exact to `10⁻¹⁷`, virial trace 60 % short.

#### 4. Cohesive energy and mechanical equilibrium

Tosi–Fumi is *fitted* to NaCl, so both of these are experiment:

```text
E_coh(a₀) = −8.0025 eV per ion pair    experiment −7.94 eV   (0.8 %)
```

The experimental static lattice energy of NaCl is 7.94 eV per ion pair
(Born–Haber cycle with the zero-point and thermal corrections removed; Kittel,
*Introduction to Solid State Physics*, 8th ed., Ch. 3). Tolerance 2 %.

The zero-pressure lattice constant is bracketed rather than solved:

```text
P(0.985 a₀)  = +1.2095 GPa  (wants to expand)
P(a₀)        = −0.1251 GPa  (wants to contract)
P(1.015 a₀)  = −1.1854 GPa  (wants to contract)
```

so the model's mechanical equilibrium is inside `a₀ ± 1.5 %`. (A quadratic
through the three points puts its static zero-pressure spacing at
`a = 5.631 Å`, −0.15 %.)

#### 5. NPT at `P_target = 0`

`NoseHooverNPT` — Martyna–Tobias–Klein isotropic barostat with Nosé–Hoover
chains (Martyna, Tobias, Klein, *J. Chem. Phys.* **101**, 4177 (1994); Shinoda
et al., *Phys. Rev. B* **69**, 134103 (2004)). `T = 300 K`, `τ_T = 100 fs`,
`τ_P = 500 fs`, `dt = 0.5 fs` (passed in ps: the case runs `units metal`),
8000 steps = 4 ps ≈ 8 barostat times.

The cell **starts 3 % compressed**, so the barostat has to push it back out. A
stress that is too small shows up as a cell that never recovers — which a bare
"did the volume change at all?" check cannot see, and which is why the run is
not started at equilibrium.

```text
⟨a⟩ over the last 2 ps = 5.6949 Å    experiment 5.6402 Å   (+0.97 %)
max|ΔH|/max(1,|H₀|)     = 1.26×10⁻⁶  over 4 ps
```

`+0.97 %` is the model's own answer, not an error budget: the Tosi–Fumi static
zero-pressure spacing is `−0.15 %` and NaCl's linear thermal expansion
(α ≈ 40×10⁻⁶ K⁻¹) adds roughly another `+1.2 %` by 300 K. The 3 % gate leaves ~3σ of headroom on the
volume fluctuation of a 64-ion cell (`σ_V/V = √(k_B T κ_T/V) ≈ 1 %`, so
`σ_a/a ≈ 0.35 %`, further reduced by averaging 20 chunks).

### Tolerances (why these numbers)

| Check | Measured | Tolerance | Justification |
|---|---|---|---|
| Madelung `E/E_ref` | 3.8e-8 | 5e-4 | PME grid 0.5 Å, order 6, box ~11 Å |
| Scaling `E(s)·s/E(1)` | 1.4e-8 | 5e-4 | same discretization at both volumes |
| Virial identity | 1.8e-8 | 1e-5 | central difference at h = 1e-6; the identity is exact, so this is FD truncation + rounding only |
| Cohesive energy | 0.8 % | 2 % | Tosi–Fumi's own fit quality against the Born–Haber lattice energy |
| Equilibrium bracket | ±1.5 % | sign change required | the model's static zero-pressure spacing is −0.15 % |
| ⟨a⟩ under NPT | +0.97 % | 3 % | static minimum + thermal expansion, ~3σ of cell-fluctuation headroom |
| Conserved `max\|ΔH\|/max(1,\|H₀\|)` | 1.26e-6 | 5e-3 | MTK is symplectic; a secular ramp means a force/stress or cell-refresh bug |

Runtime: 3 s on four threads (Apple M-series).

### Re-measurement

2026-11-15 (the commit that wrote this section): the runner is gone, so the case was
compiled on its own against a Release CPU build (`ATV_BUILD_KSPACE=ON`) with
a one-line stand-in for `harness/registry.h` that turns `ATV_REGRESSION`
into `main`, and run once; it printed `PASS`. Madelung 3.8e-8 and scaling
1.4e-8 are unchanged (the short-range table does not enter them).

### Units

LAMMPS `units metal` (set on every driver): distance Å, energy eV, force
eV/Å, mass amu, time ps, temperature K. Every gate is evaluated in eV/Å³
(`1 eV/Å³ = 160.21766 GPa`); GPa appears only in the printed lines.

### History

The previous version of this case used a soft Lennard-Jones core
(`ε = 0.05 eV`, `σ = 1.5 Å`) as a stand-in for Born–Mayer repulsion. That core
has its minimum at `2^{1/6}σ = 1.68 Å`, far inside NaCl's 2.82 Å spacing, so
the model had **no** equilibrium anywhere near the real crystal: with the
Madelung pressure at `−8.7 GPa` and nothing to oppose it until the ions were
~1.4 Å apart, "the cell collapses" was the correct answer to the question the
case was asking. It collapsed *further* than even that model wanted, because
the `PairLJCut` virial was 60 % short. Both are fixed; the case now asks a
question about NaCl.


## cuda-npt-pme-nacl — CUDA PME under a changing cell

`reg_cuda-npt-pme-nacl` (needs `ATV_BUILD_CUDA`, `ATV_BUILD_KSPACE`,
`ATV_BUILD_EVALUATE`; one GPU).

| Gate | System | Check |
|---|---|---|
| A | 8³ rocksalt point charges, d = 2.82 Å, CoulLong + KSpacePME (α = 0.35, r_c = 9 Å, spacing 0.5 Å, order 6) | E = −N M k_e/(2d), M = 1.747564594633182; W_aa = E/3, W_ab = 0 (homogeneous degree −1 ⇒ virial = energy); relative 3e-3, off-diagonal 1e-6·\|E\| |
| B | Same lattice, Joung–Cheatham TIP3P NaCl LJ (LB mixing), PairLJCutCoulLong + KSpacePME (order 5, 24³ mesh), MTK NPT 300 K / 1 bar, τ_T 100 fs, τ_P 1000 fs, dt 1 fs, 100 steps | CPU↔CUDA step-0 virial (3e-5 double / 1e-4 mixed; the CPU twin tabulates erfc), final volume (1e-5 / 1e-4), CUDA conserved-quantity drift < 2e-4 eV/atom |
| C | Perturbed lattice scaled by s ∈ {0.95, 0.98, 1.02, 1.05}, mesh fixed at the build-time 24³ | PME vs converged Ewald (accuracy 1e-10): max force error < 5× the s = 1 error, \|ΔE/E\| < 1e-4 |

References: Madelung constant of rocksalt, M = 1.747564594633…
(Y. Sakamoto, J. Chem. Phys. 28, 164 (1958)); virial of a homogeneous
potential (Clausius); Joung & Cheatham, J. Phys. Chem. B 112, 9020 (2008),
doi:10.1021/jp8001614; MTK: Martyna, Tobias and Klein, J. Chem. Phys. 101,
4177 (1994), doi:10.1063/1.467468; smooth PME virial: Essmann et al.,
J. Chem. Phys. 103, 8577 (1995), doi:10.1063/1.470117.

Gate C documents the fixed-mesh contract of `cuda::KSpacePME`: the FFT
dimensions are chosen once at build and are not re-chosen when the cell
changes. The recorded error is not a claim that the mesh error is
maintained automatically.

### Recorded results

GH200, CUDA 13.0.3, 2026-10-21.

| Gate | double | mixed |
|---|---|---|
| A: \|E/E_M − 1\| | 1.40e-5 | 1.39e-5 |
| A: max \|W_aa/(E_M/3) − 1\| | 3.24e-4 | 3.23e-4 |
| A: max \|W_ab/E_M\| | 5.8e-19 | 2.9e-10 |
| B: step-0 \|ΔW\|/\|W\| CPU↔CUDA | 7.31e-6 | 8.44e-6 |
| B: 100-step \|ΔV\|/V CPU↔CUDA | 3.53e-7 | 2.25e-7 |
| B: CUDA \|H(100) − H(1)\|/N [eV] | 4.98e-6 | 4.97e-6 |

Gate C (fixed 24³ mesh, both modes identical to three digits):

| s | \|ΔE/E\| | max\|ΔF\|/max\|F\| |
|---|---|---|
| 0.95 | 7.11e-7 | 9.28e-5 |
| 0.98 | 1.17e-6 | 1.03e-4 |
| 1.00 | 1.60e-6 | 1.07e-4 |
| 1.02 | 2.14e-6 | 1.30e-4 |
| 1.05 | 3.22e-6 | 1.74e-4 |

The force error grows by 1.6× at +5 % linear strain (the effective mesh
spacing grows with the cell), and stays within the s = 1 accuracy class.

Re-run 2026-11-15 (GH200, CUDA 13.2, double), after the case moved to
`units metal` (times in ps, P = 1 bar; before, it ran the default `lj`
style with eV/Å/fs inputs and the 1 bar target in eV/Å³): compiled on its
own with a one-line stand-in for `harness/registry.h`, A/B/C pass. A is
unchanged (|E/E_M − 1| 1.4e-5, diagonal 3.24e-4, off-diagonal 1.6e-18); B
step-0 |ΔW|/|W| 8.8e-15, 100-step |ΔV|/V 3.3e-16 (V 11482 → 10982 Å³),
CUDA |H(100) − H(1)|/N 1.4e-6 eV; C identical to the table above to three
digits.


## surface-tension-coupling — constant surface-tension NPT (couple xy)

oracle: lammps
oracle_version: 30 Mar 2026 - Development (patch_30Mar2026-1074-g7f680de296); binary `/home/jicli594/.local/x86_64/bin/lmp`

### What is checked

Public `cpu::md::SurfaceTensionNPT` (MTK semi-isotropic barostat with
LAMMPS `fix npt … couple xy` lockstep and NPγT lateral target
\(P_\perp = P_n - n_\mathrm{surf}\,\gamma/L_z\)):

1. **Couple-xy lockstep:** \(L_x = L_y\) after MD steps.
2. **Anisotropic response:** under nonzero \(\gamma\) (equivalently
   \(P_\perp \neq P_n\)), \(L_z\) need not track \(L_x\); box expands from
   a compressed LJ FCC start without runaway volume.
3. **Surface tension diagnostic:**
   \(\gamma_\mathrm{inst} = \frac{L_z}{2}\bigl(P_{zz}-\frac12(P_{xx}+P_{yy})\bigr)\)
   is finite (LAMMPS pressure sign: positive compressive).

### Physical setup (matches offline pin input)

| Quantity | Value | Note |
| -------- | ----- | ---- |
| Potential | LJ \(\varepsilon=0.0104\,\mathrm{eV}\), \(\sigma=3.4\,\mathrm{Å}\), \(r_c=8.5\,\mathrm{Å}\) | argon-like |
| Lattice | FCC \(a=5.1\,\mathrm{Å}\), \(3^3\) cells (108 atoms) | compressed |
| \(T\) | 100 K | |
| \(P_n\) | 0 eV/Å³ | normal target |
| \(\gamma\) | 0.001 eV/Å² | \(n_\mathrm{surf}=2\) |
| \(L_z(0)\) | 15.3 Å | ⇒ \(P_\perp(0)=-1.30719\times10^{-4}\) eV/Å³ |
| \(\tau_T,\tau_P\) | 50 fs, 500 fs | metal: 0.05 ps, 0.5 ps |
| \(dt\) | 1 fs | metal timestep 0.001 ps |
| steps | 200 | e2e production |

### Offline pin (LAMMPS)

Input: `tools/metalwalls_oracle/inputs/lammps/in.surface_tension_coupling`

LAMMPS cannot take equal-style variables for `Pstart`/`Pstop` on `fix npt`,
so the twin uses **fixed** \(P_\perp = P_n - n\gamma/L_z(0)\) (community
surface-tension path). Atomiverse updates \(P_\perp(L_z)\) each step (true
NPγT); short-run gates are structural, not bit-identical trajectories.

Pin snapshot (`lmp`, OMP=1, 2026-07-24):

| step | \(L_x\) | \(L_y\) | \(L_z\) |
| ---: | ---: | ---: | ---: |
| 0 | 15.3 | 15.3 | 15.3 |
| 200 | **16.527384** | **16.527384** | 16.249165 |

- \(L_x = L_y\) exactly under `couple xy`.
- \(L_z \neq L_x\) under anisotropic targets (surface-tension path).
- Box expanded from compressed start.

Metal conversion used for the pin: \(1\,\mathrm{eV/Å^3} = 1.6021766208\times10^6\,\mathrm{bar}\),
so \(P_\perp=-209.434\,\mathrm{bar}\).

Re-run 2026-11-15, the case now in `units metal` like the pin (before, it
ran the default `lj` style with eV/Å/fs inputs), compiled on its own as
described under *npt_pme_nacl — Re-measurement*: PASS, step 200
\(L_x = L_y = 16.473106\), \(L_z = 16.315748\) Å (pin 16.527384 /
16.249165; the gap is the dynamic \(P_\perp(L_z)\) and the RNG),
\(\gamma_\mathrm{inst} = 5.76\times10^{-4}\) eV/Å².

### Dual gate (regression)

1. Structural: \(|L_x-L_y| \le 10^{-9}\max(1,|L_x|)\).
2. Surface-tension path: finite safe volume + edge motion; anisotropy /
   expansion under \(\gamma\neq 0\).

### Fair wall (perf gate)

Comparable workload: **108 atoms, 5000 MD steps**, OMP=1, same LJ + couple-xy
NPT settings as the pin (fixed Plat on LAMMPS; SurfaceTensionNPT on ATV).

| Engine | Measurement | Wall [s] |
| --- | --- | ---: |
| LAMMPS (`lmp`) | process wall, 5000 steps | **0.140** |
| Atomiverse | \(25\times\) full `reg_surface-tension-coupling` (200 steps each = 5000 atom-steps; includes 25× `build()` — conservative vs pure loop) | **0.403** |

`ratio_atv_over_oracle = T_atv / T_lammps = 0.403 / 0.140 ≈ 2.88` (≤ 10).  
within_10x: **true**. Host: same login node, 2026-07-24.

### Provenance

- LAMMPS docs: <https://docs.lammps.org/fix_nh.html> (`couple xy`)
- Martyna, Tobias, Klein, *J. Chem. Phys.* **101**, 4177 (1994)
- Queue: `docs/developer/metalwalls-method-queue.md` — `surface-tension-coupling`
