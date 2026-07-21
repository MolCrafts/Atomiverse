#!/usr/bin/env python
"""Oracle pinning script for hf-capability-gaps-04-charges.

Runs ONLY in the external PySCF venv (~/work/.x86_64/bin/python) — never from
CMake/ctest. Prints the numbers that are transcribed by hand into

  regressions/scf/h2o_charges_pyscf/reg_h2o_charges_pyscf.cpp + REFERENCE.md
  regressions/scf/n2_ionization/reg_n2_ionization.cpp + REFERENCE.md

Computed here with PySCF 2.14.0 (cart=True, 6-31G*):
  (a) H2O RHF Loewdin charges  q_A = Z_A - sum_{mu in A} (S^1/2 P S^1/2)_mumu
  (b) H2O RHF ESP charges on the documented Atomiverse grid conventions
      (MK golden-angle Fibonacci shells / CHELPG cubic grid), fitted from the
      PySCF density ESP with the Lagrange total-charge constraint
  (c) N2 RHF -eps_HOMO (Koopmans IP), E(N2) RHF, E(N2+) UHF -> Delta-SCF IP

Grid conventions replicated IDENTICALLY in C++ (esp_charges.cpp):
  * grids are defined in Angstrom in both implementations (no unit drift);
  * MK: per atom, per scale s in {1.4, 1.6, 1.8, 2.0}: radius r = s*r_Bondi(A),
    N = ceil(density * 4*pi*r^2) with density = 1.0 pt/A^2, golden-angle
    spiral k = 0..N-1: z = 1 - 2*(k+0.5)/N, phi = k*pi*(3-sqrt(5)),
    rho = sqrt(1-z^2), p = R_A + r*(rho*cos(phi), rho*sin(phi), z);
    a point is DISCARDED when |p - R_B| < s*r_Bondi(B) for any B != A.
  * CHELPG: cubic grid, spacing 0.3 A, from min(R)-2.8 to max(R)+2.8 per axis
    (points at corner + i*spacing while <= max+2.8+1e-9); keep p iff
    d_A >= r_Bondi(A) for EVERY atom and d_A <= r_Bondi(A)+2.8 for SOME atom.
  * Bondi radii (Bondi 1964): H 1.20, He 1.40, C 1.70, N 1.55, O 1.52,
    F 1.47, Ne 1.54, P 1.80, S 1.80, Cl 1.75, Br 1.85.
"""

import math

import numpy as np
from pyscf import gto, scf

BOHR = 1.8897261245650618  # Angstrom -> Bohr (matches core/units.h)
HARTREE_EV = 27.211386245988

BONDI = {1: 1.20, 2: 1.40, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47,
         10: 1.54, 15: 1.80, 16: 1.80, 17: 1.75, 35: 1.85}

MK_SCALES = (1.4, 1.6, 1.8, 2.0)
MK_DENSITY = 1.0          # points per A^2
CHELPG_SPACING = 0.3      # A
CHELPG_BAND = 2.8         # A


def mk_grid(coords_ang, zs):
    """MK golden-angle Fibonacci shells; returns points in Angstrom."""
    pts = []
    natm = len(zs)
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for a in range(natm):
        for s in MK_SCALES:
            r = s * BONDI[zs[a]]
            n = int(math.ceil(MK_DENSITY * 4.0 * math.pi * r * r))
            for k in range(n):
                z = 1.0 - 2.0 * (k + 0.5) / n
                phi = k * golden
                rho = math.sqrt(max(0.0, 1.0 - z * z))
                p = coords_ang[a] + r * np.array(
                    [rho * math.cos(phi), rho * math.sin(phi), z])
                keep = True
                for b in range(natm):
                    if b == a:
                        continue
                    if np.linalg.norm(p - coords_ang[b]) < s * BONDI[zs[b]]:
                        keep = False
                        break
                if keep:
                    pts.append(p)
    return np.array(pts)


def chelpg_grid(coords_ang, zs):
    """CHELPG cubic grid; returns points in Angstrom."""
    lo = coords_ang.min(axis=0) - CHELPG_BAND
    hi = coords_ang.max(axis=0) + CHELPG_BAND
    axes = []
    for d in range(3):
        vals = []
        i = 0
        while True:
            c = lo[d] + i * CHELPG_SPACING
            if c > hi[d] + 1e-9:
                break
            vals.append(c)
            i += 1
        axes.append(vals)
    pts = []
    natm = len(zs)
    for x in axes[0]:
        for y in axes[1]:
            for z in axes[2]:
                p = np.array([x, y, z])
                outside_all = True
                near_one = False
                for a in range(natm):
                    d = np.linalg.norm(p - coords_ang[a])
                    if d < BONDI[zs[a]]:
                        outside_all = False
                        break
                    if d <= BONDI[zs[a]] + CHELPG_BAND:
                        near_one = True
                if outside_all and near_one:
                    pts.append(p)
    return np.array(pts)


def esp_at(mol, dm, pts_ang):
    """QM ESP V(r_k) [Hartree/e] at grid points given in Angstrom."""
    coords_bohr = pts_ang * BOHR
    ints = mol.intor('int1e_grids', grids=coords_bohr)  # <mu|1/|r-r_k||nu>
    v_ele = np.einsum('kij,ij->k', ints, dm)
    v_nuc = np.zeros(len(coords_bohr))
    for a in range(mol.natm):
        ra = mol.atom_coord(a)  # Bohr
        z = mol.atom_charge(a)
        v_nuc += z / np.linalg.norm(coords_bohr - ra, axis=1)
    return v_nuc - v_ele


def fit_esp(pts_ang, v, coords_ang, q_tot):
    """Lagrange-constrained LSQ: min |V - sum_A q_A/r| s.t. sum q = q_tot."""
    coords_bohr = coords_ang * BOHR
    pts_bohr = pts_ang * BOHR
    natm = len(coords_ang)
    rinv = np.zeros((len(pts_bohr), natm))
    for a in range(natm):
        rinv[:, a] = 1.0 / np.linalg.norm(pts_bohr - coords_bohr[a], axis=1)
    A = rinv.T @ rinv
    b = rinv.T @ v
    K = np.zeros((natm + 1, natm + 1))
    K[:natm, :natm] = A
    K[:natm, natm] = 1.0
    K[natm, :natm] = 1.0
    rhs = np.concatenate([b, [q_tot]])
    sol = np.linalg.solve(K, rhs)
    return sol[:natm]


def lowdin_charges(mol, mf):
    """Löwdin charges in the unit-normalized AO basis (S_ii = 1).

    Löwdin populations are not invariant under per-AO rescaling, and the
    Cartesian d normalization of PySCF's raw libcint AOs differs from
    Atomiverse's engine convention by exactly such diagonal factors.
    Normalizing each AO first (S̄ = D S D, P̄ = D⁻¹ P D⁻¹) makes the pinned
    numbers convention-independent — the same step is documented and
    performed in src/cpu/observer/lowdin.cpp.
    """
    S = mf.get_ovlp()
    P = mf.make_rdm1()
    if P.ndim == 3:
        P = P[0] + P[1]
    d = np.sqrt(np.diag(S))
    S = S / np.outer(d, d)
    P = P * np.outer(d, d)
    w, U = np.linalg.eigh(S)
    S12 = U @ np.diag(np.sqrt(w)) @ U.T
    pop = np.diag(S12 @ P @ S12)
    q = np.array(mol.atom_charges(), dtype=float)
    for mu, (a, *_rest) in enumerate(mol.ao_labels(fmt=None)):
        q[a] -= pop[mu]
    return q


def main():
    # ── H2O RHF/6-31G* (cart) ─────────────────────────────────────────
    coords = np.array([[0.0, 0.0, 0.1173],
                       [0.0, 0.7572, -0.4692],
                       [0.0, -0.7572, -0.4692]])
    zs = [8, 1, 1]
    mol = gto.M(atom=[['O', coords[0]], ['H', coords[1]], ['H', coords[2]]],
                basis='6-31g*', cart=True, unit='Angstrom')
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    e_h2o = mf.kernel()
    print(f"H2O RHF/6-31G* (cart) E = {e_h2o:.12f} Ha")

    q_low = lowdin_charges(mol, mf)
    print("Loewdin q [e]:", " ".join(f"{x:.10f}" for x in q_low))

    dm = mf.make_rdm1()
    for name, grid in (("MK", mk_grid(coords, zs)),
                       ("CHELPG", chelpg_grid(coords, zs))):
        v = esp_at(mol, dm, grid)
        q = fit_esp(grid, v, coords, 0.0)
        print(f"{name}: n_pts = {len(grid)}")
        print(f"{name} q [e]:", " ".join(f"{x:.10f}" for x in q),
              f"| sum = {q.sum():.2e}")

    # ── N2 RHF/6-31G* (cart): Koopmans + Delta-SCF ────────────────────
    r_nn = 1.0977
    mol2 = gto.M(atom=[['N', (0.0, 0.0, 0.0)], ['N', (0.0, 0.0, r_nn)]],
                 basis='6-31g*', cart=True, unit='Angstrom')
    mf2 = scf.RHF(mol2)
    mf2.conv_tol = 1e-12
    e_n2 = mf2.kernel()
    homo = mf2.mo_energy[mol2.nelectron // 2 - 1]
    print(f"N2 RHF/6-31G* (cart) E = {e_n2:.12f} Ha")
    print(f"N2 -eps_HOMO = {-homo:.12f} Ha = {-homo * HARTREE_EV:.8f} eV")

    mol2p = gto.M(atom=[['N', (0.0, 0.0, 0.0)], ['N', (0.0, 0.0, r_nn)]],
                  basis='6-31g*', cart=True, unit='Angstrom',
                  charge=1, spin=1)
    mf2p = scf.UHF(mol2p)
    mf2p.conv_tol = 1e-12
    e_n2p = mf2p.kernel()
    ip = (e_n2p - e_n2) * HARTREE_EV
    print(f"N2+ UHF/6-31G* (cart, charge=1, spin=1) E = {e_n2p:.12f} Ha")
    print(f"Delta-SCF IP = {ip:.8f} eV")


if __name__ == '__main__':
    main()
