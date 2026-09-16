#!/usr/bin/env python3
"""Pin derivation for regressions/md/constant-d-polarization-feedback.

Literature oracle (no MetalWalls binary installed; mainline LAMMPS has no
constant-D ensemble):

- Zhang & Sprik, Phys. Rev. B 93, 144201 (2016): constant-D electric
  enthalpy F = E_PBC + V (D - 4 pi P)^2 / (8 pi), field E = D - 4 pi P.
- Cox & Sprik, J. Chem. Phys. 151, 064506 (2019): finite-field bulk
  electrolyte formalism with itinerant polarization.

Unit system: eV / Angstrom / e / fs / amu with
  k_e = 1/(4 pi eps0) = 14.399645 eV*Angstrom/e^2   (units::coulomb)
  force_to_accel = 1/103.6427                        (units::force_to_accel)
so 4*pi*k_e converts P [e/A^2] into a field [V/A] and
U_D = V * E_z^2 / (8*pi*k_e) is in eV.

Run: python3 constant_d_pins.py
"""
from math import pi, cos, sqrt, acos

K_E = 14.399645
FOUR_PI_K = 4.0 * pi * K_E
F2A = 1.0 / 103.6427

# Geometry shared by all gates: +-1 e pair on z in cubic L=10 (V=1000).
V = 1000.0
D_SEP = 2.0  # [A]

print("== Gate A: static, D_z = 0.5 V/A ==")
Pz = 1.0 * D_SEP / V
Dz = 0.5
Ez = Dz - FOUR_PI_K * Pz
U = V * Ez * Ez / (8.0 * pi * K_E)
print(f"P_z      = {Pz:.16e} e/A^2")
print(f"E_z      = {Ez:.16e} V/A")
print(f"U_D      = {U:.16e} eV")
print(f"F_z(+q)  = {Ez:.16e} eV/A")

print("== Gate B: dynamics, D_z = 0 (plasma-like oscillation) ==")
mu = 0.5  # reduced mass of two 1-amu particles [amu]
dk = FOUR_PI_K * 1.0 / V  # harmonic constant on relative coord d [eV/A^2]
omega = sqrt(dk / mu * F2A)
U0 = 2.0 * pi * K_E * D_SEP * D_SEP / V
print(f"omega    = {omega:.16e} 1/fs   (period {2*pi/omega:.6f} fs)")
print(f"U0       = {U0:.16e} eV")
dt = 0.1
for nstep in (266, 532):
    t = nstep * dt
    print(f"d({t:5.1f} fs) = {D_SEP*cos(omega*t):.16e} A  (continuous)")
# velocity-Verlet discrete twin (frequency acos(1 - w^2 dt^2/2)/dt) to
# justify the 5e-5 A regression tolerance:
wtil = acos(1.0 - omega * omega * dt * dt / 2.0) / dt
for nstep in (266, 532):
    t = nstep * dt
    d_vv = D_SEP * cos(wtil * t)
    d_ct = D_SEP * cos(omega * t)
    print(f"VV-discrete d(step {nstep}) = {d_vv:.16e}  (diff {d_vv-d_ct:+.3e})")
