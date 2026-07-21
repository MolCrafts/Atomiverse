# Reference — harmonic oscillator driver matrix

## Potential

Isotropic classical harmonic oscillator (engineering units: eV, Å, amu, fs):

```text
U(r) = ½ k |r − r₀|²
F(r) = −k (r − r₀)
```

Cartesian Hessian (single free atom): `H = k I_3` [eV/Å²].

Angular frequency from Newton’s law in Atomiverse metal coupling:

```text
a [Å/fs²] = force_to_accel · F/m
ω² = force_to_accel · k / m
T  = 2π / ω  [fs]
```

with `force_to_accel = 1/103.6427` (`core/units.h`, LAMMPS metal).

## Why this case

One user-defined `Energy|Force` evaluator type is registered into four public
presets (`Evaluate`, `MD`, `Minimize`, `Hessian`) without any private header.
A user-defined `Criterion` stops the MD path by a step budget. No library
code knows these types.

## Tolerances

| Check | Tolerance | Justification |
|---|---|---|
| Evaluate U, F | `1e-12` absolute | analytic, double arithmetic |
| Minimize \|r−r₀\| | `1e-4` Å | ForceConverged ftol / k |
| Hessian eigenvalues | `1e-3` relative to k | central FD step vs exact k |
| NVE period / energy | period within 5%; \|ΔE\|/(k A²) ≤ 0.05 | VV O(dt²) at chosen dt |

## Parameters used by `reg_harmonic_driver_matrix`

`k = 1.0` eV/Å², `m = 12.0` amu, `r₀ = (0,0,0)`, initial displacement
`A = 0.1` Å along x, `dt = 0.5` fs.
