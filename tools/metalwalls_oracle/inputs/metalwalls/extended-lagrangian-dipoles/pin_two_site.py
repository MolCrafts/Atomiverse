#!/usr/bin/env python3
"""Closed-form two-site Thole/Applequist pin for extended-lagrangian-dipoles.

Independent (no Atomiverse code) derivation of the stationary fixed-point pin
shared with regressions/md/pim-induced-dipole-bulk: q = +1 e (alpha = 1.0 A^3)
at the origin, q = -1 e (alpha = 1.5 A^3) at (3, 0, 0) A, thole a = 0.39,
free space (L = 50 A min-image). The XL (iEL/SCF) fixed point at a stationary
geometry is exactly this SCF solution, so the same numbers pin the XL
component's cold AND warm launches.

Model: Thole exponential damping (van Duijnen/Swart lambda3/lambda5),
Applequist point-dipole SCC along the pair axis:

  u        = a r / (aA aB)^(1/6)
  lambda3  = 1 - exp(-u^3)
  lambda5  = 1 - (1 + u^3) exp(-u^3)
  E_A      = lambda3 qB (xA - xB) / r^3   [e/A^2]  (= +lambda3/9 here)
  E_B      = lambda3 qA (xB - xA) / r^3   (= +lambda3/9)
  T        = (3 lambda5 - lambda3) / r^3  (axial dipole-dipole coupling)
  mu_A     = (aA E_A + aA aB T E_B) / (1 - aA aB T^2)
  mu_B     = (aB E_B + aA aB T E_A) / (1 - aA aB T^2)
  E_pol    = -k/2 (mu_A E_A + mu_B E_B), k = 14.399645 eV*A/e^2

References: Thole, Chem. Phys. 59, 341 (1981); Applequist, Carl & Fung,
JACS 94, 2952 (1972); MetalWalls JOSS 5, 2373 (2020) bulk PIM form.
"""

import math

K_COULOMB = 14.399645  # eV*A/e^2 (Atomiverse units::coulomb)


def main() -> None:
    r = 3.0
    q_a, q_b = 1.0, -1.0
    a_a, a_b = 1.0, 1.5
    thole_a = 0.39

    u = thole_a * r / (a_a * a_b) ** (1.0 / 6.0)
    u3 = u**3
    l3 = 1.0 - math.exp(-u3)
    l5 = 1.0 - (1.0 + u3) * math.exp(-u3)

    e_a = l3 * q_b * (0.0 - r) / r**3  # field at A from B
    e_b = l3 * q_a * (r - 0.0) / r**3  # field at B from A
    t_ax = (3.0 * l5 - l3) / r**3

    den = 1.0 - a_a * a_b * t_ax**2
    mu_a = (a_a * e_a + a_a * a_b * t_ax * e_b) / den
    mu_b = (a_b * e_b + a_a * a_b * t_ax * e_a) / den

    e_pol = -0.5 * K_COULOMB * (mu_a * e_a + mu_b * e_b)

    print(f"u        = {u:.17g}")
    print(f"lambda3  = {l3:.17g}")
    print(f"lambda5  = {l5:.17g}")
    print(f"mu_A     = {mu_a:.17g}  e*A   (pin 0.082882545215708481)")
    print(f"mu_B     = {mu_b:.17g}  e*A   (pin 0.12342697961661356)")
    print(f"E_pol    = {e_pol:.17g}  eV   (pin -0.12040946414707571)")


if __name__ == "__main__":
    main()
