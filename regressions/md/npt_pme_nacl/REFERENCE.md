# Reference — rock-salt NaCl under PME electrostatics at zero pressure

## System

Rock salt (B1), `4³ = 64` ions, site charges alternating `(−1)^{i+j+k}` so the
cell is neutral. The conventional cubic lattice constant is `a`, the
nearest-neighbour spacing `d = a/2`, and the periodic box edge `L = 4d = 2a`.

Experimental NaCl lattice constant at 298 K:

```text
a₀ = 5.6402 Å        (d = 2.8201 Å)
```

(Wyckoff, *Crystal Structures* Vol. 1, 2nd ed., Interscience 1963; the same
value is the room-temperature figure used to calibrate powder diffractometers.)

## Interactions

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
| Na⁺–Na⁺ | 0.2637038 | 2.340 | 0.6553585 | 0.3114513 |
| Na⁺–Cl⁻ | 0.2109630 | 2.755 | 4.3690564 | 5.4176299 |
| Cl⁻–Cl⁻ | 0.1582223 | 3.170 | 45.2509408 | 90.8139570 |

`cpu::PairBorn` evaluates `A exp((σ−r)/ρ) − C/r⁶ **+** D/r⁸` (LAMMPS
`pair_style born`), so the case passes **negated** `D`. Energies are shifted to
zero at the cutoff (`pair_modify shift yes`); a constant per pair does not
change the pressure, only the energy zero, and no pair sits near `r_c`.

## What is checked

### 1. Madelung constant (bare point charges, no short range)

```text
E = −½ N M k_e / d ,    M = 1.747564594633182
```

(Ziman, *Principles of the Theory of Solids*, 2nd ed., Cambridge 1972, §2.3.)
Measured `|E/E_ref − 1| = 3.8×10⁻⁸`, tolerance `5×10⁻⁴` (the PME
discretization budget at 0.5 Å / order 6).

### 2. Uniform-scale law

Pure Coulomb is homogeneous of degree −1 in length, so scaling every
coordinate and the cell by `s` gives `E(s) = E(1)/s`. Measured
`|E(s)·s/E(1) − 1| = 1.4×10⁻⁸`. This is the cache-invalidation oracle: a
`KSpace` that reused build-time influence tables at a new volume fails it.

### 3. The virial identity — `tr(σ) = −3V dE/dV`

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

### 4. Cohesive energy and mechanical equilibrium

Tosi–Fumi is *fitted* to NaCl, so both of these are experiment:

```text
E_coh(a₀) = −7.9324 eV per ion pair    experiment −7.94 eV   (0.1 %)
```

The experimental static lattice energy of NaCl is 7.94 eV per ion pair
(Born–Haber cycle with the zero-point and thermal corrections removed; Kittel,
*Introduction to Solid State Physics*, 8th ed., Ch. 3). Tolerance 2 %.

The zero-pressure lattice constant is bracketed rather than solved:

```text
P(a₀)        = +0.499 GPa   (wants to expand)
P(1.015 a₀)  = −0.642 GPa   (wants to contract)
```

so the model's mechanical equilibrium is inside `a₀ ± 1.5 %`. (Its static
minimum is `a = 5.675 Å`, +0.6 %.)

### 5. NPT at `P_target = 0`

`NoseHooverNPT` — Martyna–Tobias–Klein isotropic barostat with Nosé–Hoover
chains (Martyna, Tobias, Klein, *J. Chem. Phys.* **101**, 4177 (1994); Shinoda
et al., *Phys. Rev. B* **69**, 134103 (2004)). `T = 300 K`, `τ_T = 100 fs`,
`τ_P = 500 fs`, `dt = 0.5 fs`, 8000 steps = 4 ps ≈ 8 barostat times.

The cell **starts 3 % compressed**, so the barostat has to push it back out. A
stress that is too small shows up as a cell that never recovers — which a bare
"did the volume change at all?" check cannot see, and which is why the run is
not started at equilibrium.

```text
⟨a⟩ over the last 2 ps = 5.7395 Å    experiment 5.6402 Å   (+1.76 %)
max|ΔH|/max(1,|H₀|)     = 1.24×10⁻⁶  over 4 ps
```

`+1.76 %` is the model's own answer, not an error budget: the Tosi–Fumi static
minimum is `+0.6 %` and NaCl's linear thermal expansion (α ≈ 40×10⁻⁶ K⁻¹) adds
roughly another `+1.2 %` by 300 K. The 3 % gate leaves ~3σ of headroom on the
volume fluctuation of a 64-ion cell (`σ_V/V = √(k_B T κ_T/V) ≈ 1 %`, so
`σ_a/a ≈ 0.35 %`, further reduced by averaging 20 chunks).

## Tolerances (why these numbers)

| Check | Measured | Tolerance | Justification |
|---|---|---|---|
| Madelung `E/E_ref` | 3.8e-8 | 5e-4 | PME grid 0.5 Å, order 6, box ~11 Å |
| Scaling `E(s)·s/E(1)` | 1.4e-8 | 5e-4 | same discretization at both volumes |
| Virial identity | 1.9e-8 | 1e-5 | central difference at h = 1e-6; the identity is exact, so this is FD truncation + rounding only |
| Cohesive energy | 0.1 % | 2 % | Tosi–Fumi's own fit quality against the Born–Haber lattice energy |
| Equilibrium bracket | ±1.5 % | sign change required | the model's static minimum is +0.6 % |
| ⟨a⟩ under NPT | +1.76 % | 3 % | static minimum + thermal expansion, ~3σ of cell-fluctuation headroom |
| Conserved `max\|ΔH\|/max(1,\|H₀\|)` | 1.24e-6 | 5e-3 | MTK is symplectic; a secular ramp means a force/stress or cell-refresh bug |

Runtime: 2.6 s single-threaded.

## Units

Distance Å, energy eV, force eV/Å, mass amu, time fs, temperature K, pressure
eV/Å³ (`1 eV/Å³ = 160.21766 GPa`). Every gate is evaluated in eV/Å³; GPa
appears only in the printed lines.

## History

The previous version of this case used a soft Lennard-Jones core
(`ε = 0.05 eV`, `σ = 1.5 Å`) as a stand-in for Born–Mayer repulsion. That core
has its minimum at `2^{1/6}σ = 1.68 Å`, far inside NaCl's 2.82 Å spacing, so
the model had **no** equilibrium anywhere near the real crystal: with the
Madelung pressure at `−8.7 GPa` and nothing to oppose it until the ions were
~1.4 Å apart, "the cell collapses" was the correct answer to the question the
case was asking. It collapsed *further* than even that model wanted, because
the `PairLJCut` virial was 60 % short. Both are fixed; the case now asks a
question about NaCl.
