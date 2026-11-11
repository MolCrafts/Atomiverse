"""Canonical all-electron RMP3 over PySCF RHF orbitals and integrals."""
from __future__ import annotations

import numpy as np


def rmp3_corrections(mf) -> tuple[float, float]:
    """Return E(2) and the E(3) increment for a converged PySCF RHF reference."""
    from pyscf import ao2mo

    n_occ = mf.mol.nelectron // 2
    o, v = slice(0, n_occ), slice(n_occ, None)
    eps = mf.mo_energy
    den = (
        eps[o, None, None, None]
        + eps[None, o, None, None]
        - eps[None, None, v, None]
        - eps[None, None, None, v]
    )
    if not np.all(den < 0.0):
        raise ValueError("mp3: non-negative or invalid orbital-energy denominator")

    # Reuse stored AO integrals when PySCF's SCF retained them, as in its MP2.
    ao = mf._eri if mf._eri is not None else mf.mol
    coeff = mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.full(ao, coeff), coeff.shape[1])
    mo = eri.transpose(0, 2, 1, 3)  # Chemist (pr|qs) to physicist <pq|rs>.
    g = mo[o, o, v, v]
    t = g / den
    tr = t.transpose(2, 3, 0, 1)
    e2 = np.einsum("ijab,ijab->", t, 2 * g - g.swapaxes(2, 3))

    # Spatial diagrams of Szabo & Ostlund, table 6.2, in the Psi4NumPy
    # convention used by the H2 reference pin. The doubles absorb both
    # denominators. Optimize the remaining contractions to use BLAS.
    # https://github.com/psi4/psi4numpy/blob/master/Moller-Plesset/MP3.py
    oooo, vvvv = mo[o, o, o, o], mo[v, v, v, v]
    vovo, ovvo = mo[v, o, v, o], mo[o, v, v, o]
    ovov, voov = mo[o, v, o, v], mo[v, o, o, v]
    e3 = (
        + 2 * np.einsum("abru,ruts,tsab->", t, vvvv, tr, optimize=True)
        + 2 * np.einsum("adrs,cbad,rscb->", t, oooo, tr, optimize=True)
        - 4 * np.einsum("acrt,rbsc,stab->", t, vovo, tr, optimize=True)
        - 4 * np.einsum("bcrt,rasb,stac->", t, vovo, tr, optimize=True)
        + 8 * np.einsum("acrt,btsc,rsab->", t, ovvo, tr, optimize=True)
        + 2 * np.einsum("cbrt,atsc,rsab->", t, ovvo, tr, optimize=True)
        - 1 * np.einsum("acrs,dbac,srdb->", t, oooo, tr, optimize=True)
        - 1 * np.einsum("abrt,trus,usab->", t, vvvv, tr, optimize=True)
        + 2 * np.einsum("bcrt,arbs,tsac->", t, ovov, tr, optimize=True)
        + 2 * np.einsum("cbrt,rasb,stac->", t, vovo, tr, optimize=True)
        - 4 * np.einsum("abrs,scat,rtbc->", t, voov, tr, optimize=True)
        - 4 * np.einsum("bcrt,atsc,rsab->", t, ovvo, tr, optimize=True)
    )
    return float(e2), float(e3)
