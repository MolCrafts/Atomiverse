# Reference — CI-NEB on the Müller-Brown surface

## The surface

K. Müller, L. D. Brown, "Location of saddle points and minimum energy paths by a
constrained simplex optimization procedure", *Theor. Chim. Acta* **53**, 75 (1979).

$$
V(x,y) = \sum_{k=1}^{4} A_k \exp\!\big[a_k (x-x_k^0)^2 + b_k (x-x_k^0)(y-y_k^0)
                                        + c_k (y-y_k^0)^2\big]
$$

| k | A    | a    | b    | c    | x⁰   | y⁰  |
|---|------|------|------|------|------|-----|
| 1 | -200 | -1   | 0    | -10  | 1    | 0   |
| 2 | -100 | -1   | 0    | -10  | 0    | 0.5 |
| 3 | -170 | -6.5 | 11   | -6.5 | -0.5 | 1.5 |
| 4 | 15   | 0.7  | 0.6  | 0.7  | -1   | 1   |

It is the standard proving ground for saddle-point methods — two-dimensional, so the path
can be drawn; analytic, so the forces are exact; and possessing **three** minima and
**two** saddles, so a method that merely finds *a* stationary point is not yet right.

Dimensionless. This is a model surface, not a molecule, so the engine's eV/Å carry no
meaning here and the published numbers are used as-is.

## Stationary points (literature)

| Point         | x       | y      | V        |
|---------------|---------|--------|----------|
| Minimum A     | -0.558  | 1.442  | -146.700 |
| Minimum B     |  0.623  | 0.028  | -108.167 |
| Minimum C     | -0.050  | 0.467  |  -80.768 |
| **Saddle 1**  | **-0.822** | **0.624** | **-40.665** |
| Saddle 2      |  0.212  | 0.293  |  -72.249 |

## What this case pins

A band from minimum **A** to minimum **B** follows the minimum energy path
`A → saddle 1 → C → saddle 2 → B`. It crosses **both** saddles, and the rate-determining
one is the higher: **saddle 1**.

```
climbing image  ->  x = -0.8219   (literature -0.822)
                    y =  0.6244   (literature  0.624)
                    V = -40.6648  (literature -40.665)
barrier from A  ->  106.0352      (literature  106.035)
converged in 207 iterations, 14 ms
```

**The barrier is the assertion, and the climbing image is why it is trustworthy.** A
plain NEB band does not sit on the saddle — the spring forces hold the images at roughly
even spacing along the path, and the highest of them straddles the barrier rather than
standing on it. Reading a barrier height off a plain band therefore reports whatever the
image spacing happened to give, and it is always **too low**. The climbing image has its
parallel force *inverted*, so it walks uphill along the path while still relaxing
perpendicular to it, and it converges **onto** the saddle. The 1e-4 agreement above is
what that buys.

This case would still "pass" a loose tolerance without the climbing image, which is why
the tolerance on the saddle *position* is 0.02 and not 0.1: at 13 images the plain band's
highest image sits visibly off the saddle.

## What this case does NOT pin

The tangent estimator is exercised but not isolated. The improved tangent
(Henkelman & Jónsson, *J. Chem. Phys.* **113**, 9978 (2000)) exists to suppress kinks
that appear where the band is stiff and the energy varies rapidly along it; the
Müller-Brown path is smooth enough that the naive bisector tangent would also converge
here, just less cleanly. A case that genuinely discriminates the two would need a stiff
band — that belongs in `benchmarks/`, against an external engine.

## References

- K. Müller, L. D. Brown, *Theor. Chim. Acta* **53**, 75 (1979) — the surface.
- G. Henkelman, B. P. Uberuaga, H. Jónsson, "A climbing image nudged elastic band method
  for finding saddle points and minimum energy paths", *J. Chem. Phys.* **113**, 9901
  (2000) — the climbing image.
- G. Henkelman, H. Jónsson, "Improved tangent estimate in the nudged elastic band method",
  *J. Chem. Phys.* **113**, 9978 (2000) — the tangent.
