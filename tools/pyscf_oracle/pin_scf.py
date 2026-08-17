#!/usr/bin/env python3
"""Offline PySCF pin helper for Atomiverse regressions.

NEVER imported by CMake/ctest. Developers / autonomous agents run this,
then paste numbers into REFERENCE.md and hard-coded regression asserts.

Examples:
  python tools/pyscf_oracle/pin_scf.py rhf --geom water --basis sto-3g
  python tools/pyscf_oracle/pin_scf.py uhf --geom o2 --basis sto-3g --spin 2
  python tools/pyscf_oracle/pin_scf.py ump2 --geom o2 --basis sto-3g --spin 2
  python tools/pyscf_oracle/pin_scf.py romp2 --geom o2 --basis sto-3g --spin 2
  python tools/pyscf_oracle/pin_scf.py mp2-fc --geom water --basis 6-31g* --frozen 1
  python tools/pyscf_oracle/pin_scf.py scs-mp2 --geom water --basis sto-3g
  python tools/pyscf_oracle/pin_scf.py sos-mp2 --geom water --basis sto-3g
  python tools/pyscf_oracle/pin_scf.py mp2 --geom water --basis sto-3g --grad
  python tools/pyscf_oracle/pin_scf.py mp3 --geom h2 --basis sto-3g
  python tools/pyscf_oracle/pin_scf.py ccsd --geom h2 --basis sto-3g
  python tools/pyscf_oracle/pin_scf.py ccsd-t --geom water --basis sto-3g
  python tools/pyscf_oracle/pin_scf.py casci --geom h2 --basis sto-3g --ncas 2 --nelecas 2
  python tools/pyscf_oracle/pin_scf.py casscf --geom h2 --basis sto-3g --ncas 2 --nelecas 2
  python tools/pyscf_oracle/pin_scf.py nevpt2 --geom h2 --basis sto-3g --ncas 2 --nelecas 2
  python tools/pyscf_oracle/pin_scf.py eom-ee-ccsd --geom h2 --basis sto-3g --nstates 1
  python tools/pyscf_oracle/pin_scf.py eom-ip-ccsd --geom water --basis sto-3g --nstates 1
  python tools/pyscf_oracle/pin_scf.py b2plyp --geom water --basis sto-3g
  python tools/pyscf_oracle/pin_scf.py rks --geom water --basis 6-31g* --xc b3lyp
  # CAM-B3LYP v1 mix (B3LYP sl + Yanai HF): --xc '0.19*HF + 0.46*LR_HF(0.33) + 0.08*LDA_X + 0.72*GGA_X_B88 + 0.19*LDA_C_VWN_RPA + 0.81*GGA_C_LYP'
  python tools/pyscf_oracle/pin_scf.py roks --geom o2 --basis sto-3g --spin 2 --xc LDA,VWN
  python tools/pyscf_oracle/pin_scf.py tda --geom h2co --basis 6-31g* --nstates 3
  python tools/pyscf_oracle/pin_scf.py cisd --geom h2co --basis sto-3g --nstates 3
  python tools/pyscf_oracle/pin_scf.py adc2 --geom h2co --basis sto-3g --nstates 3
  python tools/pyscf_oracle/pin_scf.py rks-tda --geom h2 --basis sto-3g --xc b3lyp --nstates 3
  python tools/pyscf_oracle/pin_scf.py rks-tddft --geom h2 --basis sto-3g --xc b3lyp --nstates 3
  python tools/pyscf_oracle/pin_scf.py uks-tda --geom o2 --basis sto-3g --spin 2 --xc LDA,VWN --nstates 5
  python tools/pyscf_oracle/pin_scf.py uks-tddft --geom o2 --basis sto-3g --spin 2 --xc LDA,VWN --nstates 5
  python tools/pyscf_oracle/pin_scf.py sf-tda --geom o2 --basis sto-3g --spin 2 --nstates 5
  python tools/pyscf_oracle/pin_scf.py d3bj --geom water --basis sto-3g --xc pbe
  python tools/pyscf_oracle/pin_scf.py g0w0 --geom h2 --basis sto-3g
  python tools/pyscf_oracle/pin_scf.py delta-scf --geom n2 --basis sto-3g
  python tools/pyscf_oracle/pin_scf.py rhf --geom water --basis 6-31g* --grad --pcm
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

# CODATA Ha→eV (same constant Atomiverse units::kHartreeInEv uses for gates).
_HA_TO_EV = 27.211386245988


def _as_float(x) -> float:
    """Coerce scalar / 0-d / length-1 array to float (PySCF dispersion returns)."""
    if isinstance(x, dict):
        if "energy" in x:
            return _as_float(x["energy"])
        raise TypeError(f"dict without 'energy': {x!r}")
    arr = np.asarray(x)
    if arr.ndim == 0:
        return float(arr)
    return float(arr.ravel()[0])


def _mol(geom: str, basis: str, charge: int, spin: int, cart: bool | None):
    from pyscf import gto

    presets = {
        "water": """
O  0.000000  0.000000  0.117300
H  0.000000  0.757200 -0.469200
H  0.000000 -0.757200 -0.469200
""",
        "h2": """
H  0.000000  0.000000  0.000000
H  0.000000  0.000000  0.740000
""",
        "o2": """
O  0.000000  0.000000  0.000000
O  0.000000  0.000000  1.207500
""",
        "h2co": """
C  0.000000  0.000000  0.000000
O  0.000000  0.000000  1.210000
H  0.000000  0.943000 -0.587000
H  0.000000 -0.943000 -0.587000
""",
        "ch4": """
C  0.000000  0.000000  0.000000
H  0.629118  0.629118  0.629118
H -0.629118 -0.629118  0.629118
H  0.629118 -0.629118 -0.629118
H -0.629118  0.629118 -0.629118
""",
        "n2": """
N  0.000000  0.000000  0.000000
N  0.000000  0.000000  1.097700
""",
    }
    if geom not in presets:
        raise SystemExit(f"unknown geom preset {geom!r}; choose {sorted(presets)}")

    # Pople * bases: Cartesian d is Atomiverse default; pure for def2.
    if cart is None:
        cart = not basis.lower().startswith("def2")

    return gto.M(
        atom=presets[geom],
        basis=basis,
        charge=charge,
        spin=spin,
        unit="Angstrom",
        cart=cart,
        verbose=0,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "method",
        choices=["rhf", "uhf", "rohf", "rks", "uks", "roks", "mp2", "mp2-fc", "mp3", "scs-mp2", "sos-mp2", "ump2", "romp2", "tda", "tdhf", "cisd", "adc2", "rks-tda", "rks-tddft", "uks-tda", "sf-tda", "d3bj", "g0w0", "ccsd", "ccsd-t", "casci", "b2plyp", "delta-scf", "eom-ee-ccsd", "eom-cc2", "eom-ea-ccsd", "adc2x", "ip-adc2", "ea-adc2", "sf-tddft", "mp4", "cc2", "direct-rpa", "bse", "d4", "eom-sf-ccsd", "gf2", "eom-ip-ccsd", "uks-tddft", "casscf", "nevpt2"],
    )
    p.add_argument("--geom", default="water")
    p.add_argument("--basis", default="sto-3g")
    p.add_argument("--charge", type=int, default=0)
    p.add_argument("--spin", type=int, default=0, help="2S (PySCF spin)")
    p.add_argument("--xc", default="b3lyp")
    p.add_argument("--nstates", type=int, default=3)
    p.add_argument(
        "--ncas",
        type=int,
        default=2,
        help="Active spatial orbitals for casci/casscf/nevpt2 (default 2)",
    )
    p.add_argument(
        "--nelecas",
        type=int,
        default=2,
        help="Active electrons for casci/casscf/nevpt2 (default 2; closed-shell even)",
    )
    p.add_argument(
        "--from-mo",
        type=int,
        default=None,
        help="Occupied MO index to depopulate for delta-scf (default: HOMO)",
    )
    p.add_argument(
        "--to-mo",
        type=int,
        default=None,
        help="Virtual MO index to populate for delta-scf (default: LUMO)",
    )
    p.add_argument("--conv-tol", type=float, default=1e-12)
    p.add_argument(
        "--frozen",
        type=int,
        default=None,
        help="Number of lowest occupied MOs to freeze (mp2-fc default: 1 = O 1s on water)",
    )
    p.add_argument("--cart", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--grad", action="store_true")
    p.add_argument(
        "--pcm",
        action="store_true",
        help="Attach PySCF solvent.PCM (C-PCM) before SCF / gradient",
    )
    p.add_argument("--pcm-eps", type=float, default=78.4, help="PCM dielectric ε")
    p.add_argument(
        "--pcm-scale",
        type=float,
        default=1.2,
        help="PCM vdw radii scale (Bondi × scale)",
    )
    p.add_argument(
        "--pcm-lebedev-order",
        type=int,
        default=29,
        help="Lebedev angular order for PCM cavity",
    )
    p.add_argument(
        "--pcm-method",
        default="C-PCM",
        help="PySCF solvent method string (default C-PCM)",
    )
    args = p.parse_args()

    try:
        import pyscf
        from pyscf import scf, dft, mp, tdscf, cc as pyscf_cc
    except ImportError as e:
        print(json.dumps({"ok": False, "error": f"pyscf missing: {e}"}))
        return 2

    mol = _mol(args.geom, args.basis, args.charge, args.spin, args.cart)
    out: dict = {
        "ok": True,
        "pyscf": pyscf.__version__,
        "method": args.method,
        "geom": args.geom,
        "basis": args.basis,
        "charge": args.charge,
        "spin": args.spin,
        "cart": bool(mol.cart),
        "conv_tol": args.conv_tol,
        "natm": mol.natm,
        "nao": mol.nao,
    }

    m = args.method
    if m == "d3bj":
        # Geometry-only DFT-D3(BJ). Requires pyscf-dispersion (simple-dftd3).
        # E_disp is independent of SCF; still report optional e_tot with/without.
        try:
            from pyscf.dispersion import dftd3 as dftd3_mod
        except ImportError as e:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": f"pyscf-dispersion missing (need dftd3): {e}",
                    }
                )
            )
            return 2
        xc = args.xc if args.xc else "pbe"
        out["xc"] = xc
        model = dftd3_mod.DFTD3Dispersion(mol, xc=xc, version="d3bj")
        ed = _as_float(model.get_dispersion())
        out["e_disp_ha"] = ed
        # Optional KS totals for cross-check (same geometry / basis / xc).
        mf0 = dft.RKS(mol)
        mf0.xc = xc
        mf0.conv_tol = args.conv_tol
        e0 = float(mf0.kernel())
        mf1 = dft.RKS(mol)
        mf1.xc = xc
        mf1.conv_tol = args.conv_tol
        mf1.disp = "d3bj"
        e1 = float(mf1.kernel())
        out["e_tot_no_disp_ha"] = e0
        out["e_tot_with_disp_ha"] = e1
        out["e_tot_ha"] = e1
        out["converged"] = bool(mf1.converged)
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0

    if m == "bse":
        # Exact-ERI static-RPA BSE on spectral G0W0 poles — twin of
        # Atomiverse RHF::bse(). Stock DF pyscf.gw.bse is reported as
        # s1_df_ev for comparison only (~meV DF discrepancy on H2/STO-3G).
        import numpy as np
        from pyscf import ao2mo as ao2mo_mod
        from pyscf.gw import gw_exact
        from pyscf import tdscf as tdscf_mod

        mf = dft.RKS(mol)
        mf.xc = "hf"
        mf.conv_tol = args.conv_tol
        e = float(mf.kernel())
        C = np.asarray(mf.mo_coeff, dtype=float)
        nmo = int(C.shape[1])
        nocc = int(mol.nelectron // 2)
        nvir = nmo - nocc
        eri = ao2mo_mod.restore(1, ao2mo_mod.full(mol, C), nmo)
        td = tdscf_mod.dRPA(mf)
        td.nstates = max(1, nocc * nvir)
        td.verbose = 0
        td.kernel()
        ge = gw_exact.GWExact(mf, tdmf=td)
        ge.verbose = 0
        ge.kernel()
        eps = np.asarray(ge.mo_energy, dtype=float)
        nov = nocc * nvir
        S = np.zeros((nov, nov), dtype=float)
        for a in range(nvir):
            for i in range(nocc):
                p = a * nocc + i
                S[p, p] = eps[i] - eps[nocc + a]
                for b in range(nvir):
                    for j in range(nocc):
                        q = b * nocc + j
                        v = float(eri[i, nocc + a, j, nocc + b])
                        S[p, q] -= 4.0 * v
        Pi = 4.0 * np.linalg.inv(S)

        def W(p: int, q: int, r: int, s: int) -> float:
            acc = float(eri[p, q, r, s])
            for a in range(nvir):
                for i in range(nocc):
                    ia = a * nocc + i
                    for b in range(nvir):
                        for j in range(nocc):
                            jb = b * nocc + j
                            acc += (
                                float(eri[p, q, i, nocc + a])
                                * Pi[ia, jb]
                                * float(eri[j, nocc + b, r, s])
                            )
            return acc

        A = np.zeros((nov, nov), dtype=float)
        B = np.zeros((nov, nov), dtype=float)
        for a in range(nvir):
            for i in range(nocc):
                ia = a * nocc + i
                aa = nocc + a
                for b in range(nvir):
                    for j in range(nocc):
                        jb = b * nocc + j
                        bb = nocc + b
                        vc = float(eri[i, aa, j, bb])
                        A[ia, jb] = 2.0 * vc - W(i, j, aa, bb)
                        B[ia, jb] = 2.0 * vc - W(i, bb, aa, j)
                        if ia == jb:
                            A[ia, jb] += float(eps[aa] - eps[i])
        ApB = 0.5 * ((A + B) + (A + B).T)
        AmB = 0.5 * ((A - B) + (A - B).T)
        # Casida: M = (A-B)^{1/2} (A+B) (A-B)^{1/2}
        s_m, U_m = np.linalg.eigh(AmB)
        if s_m[0] <= 0.0:
            raise RuntimeError(f"BSE (A-B) not PD: min eig = {s_m[0]}")
        T = (U_m * np.sqrt(s_m)) @ U_m.T
        M = T @ ApB @ T
        M = 0.5 * (M + M.T)
        w2 = np.linalg.eigvalsh(M)
        w = np.sqrt(np.maximum(w2, 0.0))
        tda = np.linalg.eigvalsh(A)
        nstates = max(1, min(int(args.nstates), nov))
        out["e_tot_ha"] = e
        out["converged"] = bool(mf.converged)
        out["qp_energy_ha"] = eps.tolist()
        out["gap_ev"] = float((eps[nocc] - eps[nocc - 1]) * _HA_TO_EV)
        out["s1_ev"] = float(w[0] * _HA_TO_EV)
        out["energy_ev"] = (w[:nstates] * _HA_TO_EV).tolist()
        out["tda_s1_ev"] = float(tda[0] * _HA_TO_EV)
        out["tda_energy_ev"] = (tda[:nstates] * _HA_TO_EV).tolist()
        out["oracle"] = "exact_eri_bse_on_gw_exact_poles"
        # Optional DF stock twin (info only).
        try:
            from pyscf import gw as gw_mod
            from pyscf.gw import bse as bse_mod

            mf_rhf = scf.RHF(mol)
            mf_rhf.conv_tol = args.conv_tol
            mf_rhf.kernel()
            gdf = gw_mod.GW(mf_rhf)
            gdf.verbose = 0
            gdf.kernel()
            bb = bse_mod.BSE(gdf)
            bb.verbose = 0
            bb.TDA = False
            exci, _, _ = bb.full_diagonalization(multi="s")
            out["s1_df_ev"] = float(exci[0] * _HA_TO_EV)
        except Exception as df_err:  # pragma: no cover — optional twin
            out["df_bse_error"] = str(df_err)
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0 if out.get("converged", True) else 1

    if m == "d4":
        # Geometry-only DFT-D4. Requires pyscf-dispersion (libdftd4).
        try:
            from pyscf.dispersion import dftd4 as dftd4_mod
        except ImportError as e:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": f"pyscf-dispersion missing (need dftd4): {e}",
                    }
                )
            )
            return 2
        xc = args.xc if args.xc else "pbe"
        out["xc"] = xc
        model = dftd4_mod.DFTD4Dispersion(mol, xc=xc, version="d4", atm=True)
        ed = _as_float(model.get_dispersion())
        out["e_disp_ha"] = ed
        mf0 = dft.RKS(mol)
        mf0.xc = xc
        mf0.conv_tol = args.conv_tol
        e0 = float(mf0.kernel())
        mf1 = dft.RKS(mol)
        mf1.xc = xc
        mf1.conv_tol = args.conv_tol
        mf1.disp = "d4"
        e1 = float(mf1.kernel())
        out["e_tot_no_disp_ha"] = e0
        out["e_tot_with_disp_ha"] = e1
        out["e_tot_ha"] = e1
        out["converged"] = bool(mf1.converged)
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0

    if m == "gf2":
        # One-shot GF2 / G0F2 (AGF2 SE from HF GF + single Dyson solve) —
        # twin of Atomiverse RHF::gf2(). Not self-consistent RAGF2.kernel().
        import numpy as np
        from pyscf import agf2
        from pyscf.agf2.chempot import binsearch_chempot

        mf = scf.RHF(mol)
        mf.conv_tol = args.conv_tol
        e = float(mf.kernel())
        g1 = agf2.RAGF2(mf)
        g1.verbose = 0
        eri = g1.ao2mo()
        g0 = g1.init_gf()
        se0 = g1.build_se(eri, g0)
        fock = np.diag(np.asarray(mf.mo_energy, dtype=float))
        w, v = se0.eig(fock)
        mu, nerr = binsearch_chempot((w, v), g1.nmo, g1.nocc * 2)
        g_one = agf2.GreensFunction(w, v[: g1.nmo], chempot=mu)
        ip = float(g1.get_ip(g_one, nroots=1)[0][0])
        ea = float(g1.get_ea(g_one, nroots=1)[0][0])
        e_init = float(g1.energy_mp2(mf.mo_energy, se0))
        out["e_tot_ha"] = e
        out["converged"] = bool(mf.converged)
        out["eps_hf_ha"] = np.asarray(mf.mo_energy, dtype=float).tolist()
        out["qp_homo_ha"] = float(-ip)
        out["qp_lumo_ha"] = float(ea)
        out["gap_ha"] = float(ip + ea)
        out["gap_ev"] = float((ip + ea) * _HA_TO_EV)
        out["ip_ev"] = float(ip * _HA_TO_EV)
        out["ea_ev"] = float(ea * _HA_TO_EV)
        out["ip_ha"] = ip
        out["ea_ha"] = ea
        out["e_corr_ha"] = e_init
        out["e_tot_gf2_ha"] = float(e + e_init)
        out["chempot_ha"] = float(mu)
        out["nerr"] = float(nerr)
        out["pole_energy_ha"] = np.asarray(w, dtype=float).tolist()
        out["oracle"] = "pyscf_agf2_ragf2_oneshot"
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0 if out.get("converged", True) else 1

    if m == "delta-scf":
        # UHF ΔSCF with IMOM occupation locking (PySCF scf.addons.mom_occ_).
        # Default promotion: α HOMO → LUMO on a closed-shell RHF reference.
        import numpy as np
        from pyscf.scf import addons

        mf0 = scf.RHF(mol)
        mf0.conv_tol = args.conv_tol
        e0 = float(mf0.kernel())
        nmo = int(mf0.mo_coeff.shape[1])
        nocc = int(mol.nelectron // 2)
        from_mo = int(args.from_mo) if args.from_mo is not None else nocc - 1
        to_mo = int(args.to_mo) if args.to_mo is not None else nocc
        if from_mo < 0 or from_mo >= nocc or to_mo < nocc or to_mo >= nmo:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": f"bad promotion from={from_mo} to={to_mo} "
                        f"(nocc={nocc}, nmo={nmo})",
                    }
                )
            )
            return 2
        mo0 = np.asarray(mf0.mo_coeff)
        occ_a = np.zeros(nmo)
        occ_b = np.zeros(nmo)
        occ_a[:nocc] = 1.0
        occ_b[:nocc] = 1.0
        occ_a[from_mo] = 0.0
        occ_a[to_mo] = 1.0

        def _rdm1(C, occ):
            return np.einsum("pi,i,qi->pq", C, occ, C)

        dm = (_rdm1(mo0, occ_a), _rdm1(mo0, occ_b))
        mfu = scf.UHF(mol)
        mfu.conv_tol = args.conv_tol
        mfu.max_cycle = 200
        mfu = addons.mom_occ_(mfu, (mo0, mo0), (occ_a.copy(), occ_b.copy()))
        e1 = float(mfu.kernel(dm))
        de_ha = e1 - e0
        out["e_ground_ha"] = e0
        out["e_excited_ha"] = e1
        out["delta_e_ha"] = de_ha
        out["delta_e_ev"] = de_ha * _HA_TO_EV
        out["from_mo"] = from_mo
        out["to_mo"] = to_mo
        out["nocc"] = nocc
        out["nmo"] = nmo
        out["converged"] = bool(mfu.converged)
        if hasattr(mfu, "spin_square"):
            ss, mult = mfu.spin_square()
            out["s2"] = float(ss)
            out["multiplicity"] = float(mult)
        out["oracle"] = "pyscf_scf_addons_mom_occ_imom"
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0

    if m == "g0w0":
        # Spectral (frequency-exact) G0W0 with dRPA screening — PySCF
        # gw.gw_exact.GWExact twin of Atomiverse RHF::g0w0(). HF density via
        # RKS(xc='hf') (gw_exact asserts a DFT mean-field object).
        # Local import keeps `np` bound even though later branches also
        # `import numpy as np` (which makes np function-local in Python).
        import numpy as np
        from pyscf.gw import gw_exact
        from pyscf import tdscf as tdscf_mod

        mf = dft.RKS(mol)
        mf.xc = "hf"
        mf.conv_tol = args.conv_tol
        e = float(mf.kernel())
        nocc = int(mol.nelectron // 2)
        nvir = int(mf.mo_coeff.shape[1] - nocc)
        td = tdscf_mod.dRPA(mf)
        td.nstates = max(1, nocc * nvir)
        td.verbose = 0
        td.kernel()
        ge = gw_exact.GWExact(mf, tdmf=td)
        ge.verbose = 0
        ge.kernel()
        qp = np.asarray(ge.mo_energy, dtype=float)
        homo = nocc - 1
        lumo = nocc
        gap_ha = float(qp[lumo] - qp[homo])
        out["e_tot_ha"] = e
        out["converged"] = bool(mf.converged)
        out["eps_hf_ha"] = np.asarray(mf.mo_energy, dtype=float).tolist()
        out["qp_energy_ha"] = qp.tolist()
        out["qp_homo_ha"] = float(qp[homo])
        out["qp_lumo_ha"] = float(qp[lumo])
        out["gap_ha"] = gap_ha
        out["gap_ev"] = gap_ha * _HA_TO_EV
        out["ip_ev"] = float(-qp[homo] * _HA_TO_EV)
        out["ea_ev"] = float(qp[lumo] * _HA_TO_EV)
        out["drpa_omega_ha"] = np.asarray(td.e, dtype=float).tolist()
        # AC companion (default gw.GW on RHF) for near-twin comparison only.
        try:
            from pyscf import gw as gw_mod

            mf_rhf = scf.RHF(mol)
            mf_rhf.conv_tol = args.conv_tol
            mf_rhf.kernel()
            gac = gw_mod.GW(mf_rhf)
            gac.verbose = 0
            gac.kernel()
            qac = np.asarray(gac.mo_energy, dtype=float)
            out["qp_ac_ha"] = qac.tolist()
            out["gap_ac_ev"] = float((qac[lumo] - qac[homo]) * _HA_TO_EV)
        except Exception as ac_err:  # pragma: no cover — optional twin
            out["ac_error"] = str(ac_err)
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0 if out.get("converged", True) else 1

    if m == "rhf":
        mf = scf.RHF(mol)
    elif m in ("uhf", "ump2", "sf-tda", "sf-tddft"):
        mf = scf.UHF(mol)
    elif m in ("rohf", "romp2"):
        mf = scf.ROHF(mol)
    elif m in ("rks", "rks-tda", "rks-tddft"):
        mf = dft.RKS(mol)
        mf.xc = args.xc
        out["xc"] = args.xc
    elif m == "b2plyp":
        # Grimme B2PLYP double hybrid (JCP 124, 034108 (2006)):
        #   a_x = 0.53 HF, (1-a_x) B88, (1-a_c) LYP, a_c = 0.27 PT2/MP2.
        # Hosted libxc often lacks the named B2PLYP composite; compose the
        # hybrid-KS part explicitly and scale MP2 by a_c offline.
        mf = dft.RKS(mol)
        mf.xc = "0.53*HF + 0.47*GGA_X_B88 + 0.73*GGA_C_LYP"
        out["xc"] = mf.xc
        out["a_x"] = 0.53
        out["a_c"] = 0.27
    elif m in ("uks", "uks-tda", "uks-tddft"):
        mf = dft.UKS(mol)
        mf.xc = args.xc
        out["xc"] = args.xc
    elif m == "roks":
        mf = dft.ROKS(mol)
        mf.xc = args.xc
        out["xc"] = args.xc
    elif m == "mp4":
        # Closed-shell RMP4(SDTQ). PySCF 2.x has no mp.MP4 — exact intermediate-
        # normalization RSPT through order 4 on the full determinant basis built
        # from PySCF RHF MOs + ao2mo ERIs (H₂/STO-3G product gate; C(2 nmo, ne)
        # must stay tiny). Algebraic spin-orbital SDTQ residual form matches.
        from pyscf import ao2mo
        from itertools import combinations
        import numpy as np

        m2 = mp.MP2(mf).run()
        C = np.asarray(mf.mo_coeff)
        eps = np.asarray(mf.mo_energy)
        nmo = C.shape[1]
        ndocc = mol.nelectron // 2
        nocc_so = 2 * ndocc
        nso = 2 * nmo
        if nso > 12 or nocc_so > 6:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": "mp4 pin: determinant RSPT limited to tiny "
                        "spaces (nso<=12, nocc_so<=6); use algebraic path",
                    }
                )
            )
            return 2
        eri_chem = ao2mo.restore(1, ao2mo.full(mol, C), nmo)
        h_ao = mf.get_hcore()
        h_mo = C.T @ h_ao @ C
        h_so = np.zeros((nso, nso))
        for p in range(nmo):
            for q in range(nmo):
                h_so[2 * p, 2 * q] = h_mo[p, q]
                h_so[2 * p + 1, 2 * q + 1] = h_mo[p, q]
        # Antisym <pq||rs>
        Vaa = np.zeros((nso, nso, nso, nso))
        for p in range(nso):
            for q in range(nso):
                for r in range(nso):
                    for s in range(nso):
                        val = 0.0
                        if (p % 2) == (r % 2) and (q % 2) == (s % 2):
                            val += eri_chem[p // 2, r // 2, q // 2, s // 2]
                        if (p % 2) == (s % 2) and (q % 2) == (r % 2):
                            val -= eri_chem[p // 2, s // 2, q // 2, r // 2]
                        Vaa[p, q, r, s] = val
        eso = np.repeat(eps, 2)
        dets = list(combinations(range(nso), nocc_so))
        nd = len(dets)
        iref = dets.index(tuple(range(nocc_so)))

        def phase_exc(di, holes, particles):
            orb = list(di)
            phase = 1
            for h in holes:
                pos = orb.index(h)
                phase *= (-1) ** pos
                orb.pop(pos)
            for p_ in particles:
                n_before = sum(1 for o in orb if o < p_)
                phase *= (-1) ** n_before
                orb.append(p_)
                orb.sort()
            return phase

        def H_me(d1, d2):
            s1, s2 = set(d1), set(d2)
            only1 = sorted(s1 - s2)
            only2 = sorted(s2 - s1)
            n = len(only1)
            if n > 2:
                return 0.0
            if n == 0:
                e_ = sum(h_so[i, i] for i in d1)
                for i in d1:
                    for j in d1:
                        e_ += 0.5 * Vaa[i, j, i, j]
                return e_
            if n == 1:
                h, p_ = only1[0], only2[0]
                ph = phase_exc(d1, [h], [p_])
                common = sorted(s1 & s2)
                val = h_so[h, p_] + sum(Vaa[h, k, p_, k] for k in common)
                return ph * val
            i, j = only1
            a, b = only2
            ph = phase_exc(d1, [i, j], [a, b])
            return ph * Vaa[i, j, a, b]

        H = np.array([[H_me(d1, d2) for d2 in dets] for d1 in dets])
        H0 = np.diag([sum(eso[p_] for p_ in d) for d in dets])
        Vmat = H - H0
        E0 = H0[iref, iref]
        E1 = Vmat[iref, iref]
        Psi0 = np.zeros(nd)
        Psi0[iref] = 1.0

        def R0(vec):
            out = np.zeros_like(vec)
            for i in range(nd):
                if i != iref:
                    out[i] = vec[i] / (E0 - H0[i, i])
            return out

        Psi1 = R0(Vmat @ Psi0)
        e2 = float(Psi0 @ Vmat @ Psi1)
        Psi2 = R0(Vmat @ Psi1 - E1 * Psi1 - e2 * Psi0)
        e3 = float(Psi0 @ Vmat @ Psi2)
        Psi3 = R0(Vmat @ Psi2 - E1 * Psi2 - e2 * Psi1 - e3 * Psi0)
        e4 = float(Psi0 @ Vmat @ Psi3)
        e_corr = e2 + e3 + e4
        out["e_hf_ha"] = float(e)
        out["e_corr_mp2_ha"] = e2
        out["e_corr_mp3_ha"] = e3
        out["e_corr_mp4_ha"] = e4
        out["e_corr_ha"] = e_corr
        out["e_tot_mp4_ha"] = float(e) + e_corr
        out["e_corr_mp2_pyscf_ha"] = float(m2.e_corr)
        out["oracle"] = "rspt4_det_on_pyscf_rhf_ao2mo"
    elif m == "cc2":
        # Closed-shell CC2 (Christiansen 1995). Stock PySCF has no cc.CC2;
        # oracle is rccsd.RCCSD with cc2=True (same as rccsd_slow).
        import numpy as _np
        from pyscf.cc import rccsd as _rccsd

        mycc = _rccsd.RCCSD(mf)
        mycc.conv_tol = min(args.conv_tol, 1e-10)
        mycc.conv_tol_normt = min(args.conv_tol, 1e-10)
        mycc.cc2 = True
        e_corr, t1, t2 = mycc.kernel()
        out["e_hf_ha"] = float(e)
        out["e_corr_ha"] = float(e_corr)
        out["e_tot_cc2_ha"] = float(mycc.e_tot)
        out["e_tot_ha"] = float(mycc.e_tot)
        out["cc2_converged"] = bool(mycc.converged)
        t1a = _np.asarray(t1).ravel()
        t2a = _np.asarray(t2).ravel()
        out["t1_norm"] = float(_np.sqrt(t1a.dot(t1a)))
        out["t2_norm"] = float(_np.sqrt(t2a.dot(t2a)))
        out["oracle"] = "pyscf_cc_rccsd_cc2"
    elif m == "direct-rpa":
        # Exact dRPA correlation (Furche plasmon) on RHF orbitals + MO ERIs.
        # Stock pyscf.gw.rpa.RPA uses DF + Legendre quadrature (near twin).
        import numpy as np
        from numpy.linalg import eigh
        from scipy.linalg import sqrtm
        from pyscf import ao2mo
        from pyscf.gw import rpa as rpa_mod

        C = np.asarray(mf.mo_coeff)
        eps = np.asarray(mf.mo_energy)
        nmo = C.shape[1]
        nocc = mol.nelectron // 2
        nvir = nmo - nocc
        eri = ao2mo.restore(1, ao2mo.full(mol, C), nmo)
        n_ov = nocc * nvir
        A = np.zeros((n_ov, n_ov))
        B = np.zeros_like(A)
        for i in range(nocc):
            for a in range(nvir):
                ia = i * nvir + a
                aa = nocc + a
                A[ia, ia] = eps[aa] - eps[i]
                for j in range(nocc):
                    for b in range(nvir):
                        jb = j * nvir + b
                        bb = nocc + b
                        v = 2.0 * eri[i, aa, j, bb]
                        A[ia, jb] += v
                        B[ia, jb] += v
        Mm = A - B
        Mp = A + B
        T = np.real(sqrtm(Mm))
        M = T @ Mp @ T
        M = 0.5 * (M + M.T)
        w2, _Z = eigh(M)
        omega = np.sqrt(np.maximum(w2, 0.0))
        tr_A = float(np.trace(A))
        e_corr = 0.5 * (float(omega.sum()) - tr_A)
        rdf = rpa_mod.RPA(mf)
        rdf.kernel(nw=100)
        out["e_rhf_ha"] = float(e)
        out["e_corr_ha"] = e_corr
        out["e_tot_ha"] = float(e) + e_corr
        out["omega_ha"] = omega.tolist()
        out["tr_A_ha"] = tr_A
        out["e_corr_df_rpa_ha"] = float(rdf.e_corr)
        out["method"] = "direct-rpa"
    elif m == "eom-sf-ccsd":
        # Spin-flip EOM-CCSD on UHF → UCCSD (PySCF eomsf_ccsd).
        import numpy as _np

        mycc = pyscf_cc.UCCSD(mf)
        mycc.conv_tol = min(args.conv_tol, 1e-10)
        e_corr, _t1, _t2 = mycc.kernel()
        e_eom, _v = mycc.eomsf_ccsd(nroots=args.nstates)
        e_eom = _np.atleast_1d(_np.asarray(e_eom, dtype=float))
        out["e_hf_ha"] = float(e)
        out["e_corr_ha"] = float(e_corr)
        out["e_tot_ccsd_ha"] = float(mycc.e_tot)
        out["e_tot_ha"] = float(mycc.e_tot)
        out["ccsd_converged"] = bool(mycc.converged)
        out["omega_ha"] = e_eom.tolist()
        out["omega_ev"] = (e_eom * _HA_TO_EV).tolist()
        out["omega0_ev"] = float(e_eom[0] * _HA_TO_EV)
        out["nstates"] = int(args.nstates)
        out["sf_channel"] = "eomsf_ccsd"
    else:
        # tda / tdhf / mp2 / mp3 / ccsd ground state: RHF
        mf = scf.RHF(mol)


    if args.pcm:
        if m not in ("rhf", "uhf", "rohf", "rks", "uks", "roks"):
            print(
                json.dumps(
                    {
                        "error": f"--pcm not supported for method {m!r}",
                    },
                    indent=2,
                )
            )
            return 2
        mf = mf.PCM()
        mf.with_solvent.method = args.pcm_method
        mf.with_solvent.eps = args.pcm_eps
        mf.with_solvent.vdw_scale = args.pcm_scale
        mf.with_solvent.lebedev_order = args.pcm_lebedev_order
        out["pcm"] = {
            "method": args.pcm_method,
            "eps": args.pcm_eps,
            "vdw_scale": args.pcm_scale,
            "lebedev_order": args.pcm_lebedev_order,
        }

    mf.conv_tol = args.conv_tol
    e = mf.kernel()
    out["e_tot_ha"] = float(e)
    out["converged"] = bool(mf.converged)
    if args.pcm and hasattr(mf, "with_solvent"):
        out["e_solvent_ha"] = float(getattr(mf.with_solvent, "e", 0.0) or 0.0)

    if m in ("rohf", "roks", "uks", "uks-tda", "uks-tddft", "uhf", "ump2", "romp2", "sf-tda", "sf-tddft") and hasattr(
        mf, "spin_square"
    ):
        ss, mult = mf.spin_square()
        out["s2"] = float(ss)
        out["multiplicity"] = float(mult)

    if args.grad and m in ("rhf", "uhf", "rohf", "rks", "uks", "roks"):
        g = mf.nuc_grad_method().kernel()
        out["grad_ha_bohr"] = g.tolist()

    if m == "mp2":
        m2 = mp.MP2(mf).run()
        out["e_corr_ha"] = float(m2.e_corr)
        out["e_tot_mp2_ha"] = float(m2.e_tot)
        out["e_corr_ss_ha"] = float(m2.e_corr_ss)
        out["e_corr_os_ha"] = float(m2.e_corr_os)
        if args.grad:
            g = m2.nuc_grad_method().kernel()
            out["grad_ha_bohr"] = g.tolist()
    elif m == "mp2-fc":
        # Frozen-core RMP2: freeze lowest N occupied MOs (default 1 = O 1s on H2O).
        n_frozen = 1 if args.frozen is None else int(args.frozen)
        m2 = mp.MP2(mf, frozen=n_frozen).run()
        out["frozen"] = n_frozen
        out["e_hf_ha"] = float(e)
        out["e_corr_ha"] = float(m2.e_corr)
        out["e_tot_mp2_ha"] = float(m2.e_tot)
    elif m == "scs-mp2":
        # Spin-component-scaled MP2 (Grimme, JCP 118, 9095 (2003)):
        # E_corr^SCS = (1/3) E_SS + 1.2 E_OS on the RHF-MP2 spin channels.
        m2 = mp.MP2(mf).run()
        out["e_hf_ha"] = float(e)
        out["e_corr_mp2_ha"] = float(m2.e_corr)
        out["e_corr_ss_ha"] = float(m2.e_corr_ss)
        out["e_corr_os_ha"] = float(m2.e_corr_os)
        out["e_corr_ha"] = float(m2.emp2_scs)
        out["e_tot_mp2_ha"] = float(m2.e_tot_scs)
        out["c_os"] = 1.2
        out["c_ss"] = 1.0 / 3.0
    elif m == "sos-mp2":
        # Scaled-opposite-spin MP2 (Jung et al., JCP 121, 9793 (2004)):
        # E_corr^SOS = 1.3 E_OS + 0 * E_SS (opposite-spin only).
        # PySCF has no emp2_sos property; scale the RMP2 OS channel offline.
        m2 = mp.MP2(mf).run()
        c_os, c_ss = 1.3, 0.0
        e_corr_sos = c_os * float(m2.e_corr_os) + c_ss * float(m2.e_corr_ss)
        out["e_hf_ha"] = float(e)
        out["e_corr_mp2_ha"] = float(m2.e_corr)
        out["e_corr_ss_ha"] = float(m2.e_corr_ss)
        out["e_corr_os_ha"] = float(m2.e_corr_os)
        out["e_corr_ha"] = e_corr_sos
        out["e_tot_mp2_ha"] = float(e) + e_corr_sos
        out["c_os"] = c_os
        out["c_ss"] = c_ss
    elif m == "mp3":
        # Closed-shell RMP3. PySCF 2.x has no mp.MP3 — rebuild Szabo/Ostlund
        # (Psi4NumPy MP3.py) spatial diagrams on PySCF RHF MOs + ao2mo ERIs.
        # E_corr = E^(2) + E^(3); E^(2) must match mp.MP2.
        from pyscf import ao2mo
        import numpy as np

        m2 = mp.MP2(mf).run()
        C = np.asarray(mf.mo_coeff)
        eps = np.asarray(mf.mo_energy)
        nmo = C.shape[1]
        ndocc = mol.nelectron // 2
        eri_chem = ao2mo.restore(1, ao2mo.full(mol, C), nmo)
        # Physicist <pq|rs> = (pr|qs)
        MO = np.einsum("prqs->pqrs", eri_chem)
        o = slice(0, ndocc)
        v = slice(ndocc, nmo)
        eocc = eps[:ndocc]
        evirt = eps[ndocc:]
        den = (
            eocc.reshape(-1, 1, 1, 1)
            + eocc.reshape(-1, 1, 1)
            - evirt.reshape(-1, 1)
            - evirt
        )
        if np.any(den >= 0.0):
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": "mp3: non-negative orbital-energy denominator",
                    }
                )
            )
            return 2
        eps_inv = 1.0 / den
        e2 = 2.0 * np.einsum(
            "abrs,rsab,abrs", MO[o, o, v, v], MO[v, v, o, o], eps_inv
        )
        e2 -= np.einsum(
            "abrs,rsba,abrs", MO[o, o, v, v], MO[v, v, o, o], eps_inv
        )
        # 12 third-order diagrams (Psi4NumPy / Szabo Table 6.2).
        e3 = 2.0 * np.einsum(
            "abru,ruts,tsab,abru,abts",
            MO[o, o, v, v],
            MO[v, v, v, v],
            MO[v, v, o, o],
            eps_inv,
            eps_inv,
        )
        e3 += 2.0 * np.einsum(
            "adrs,cbad,rscb,adrs,cbrs",
            MO[o, o, v, v],
            MO[o, o, o, o],
            MO[v, v, o, o],
            eps_inv,
            eps_inv,
        )
        e3 += -4.0 * np.einsum(
            "acrt,rbsc,stab,acrt,abst",
            MO[o, o, v, v],
            MO[v, o, v, o],
            MO[v, v, o, o],
            eps_inv,
            eps_inv,
        )
        e3 += -4.0 * np.einsum(
            "bcrt,rasb,stac,bcrt,acst",
            MO[o, o, v, v],
            MO[v, o, v, o],
            MO[v, v, o, o],
            eps_inv,
            eps_inv,
        )
        e3 += 8.0 * np.einsum(
            "acrt,btsc,rsab,acrt,abrs",
            MO[o, o, v, v],
            MO[o, v, v, o],
            MO[v, v, o, o],
            eps_inv,
            eps_inv,
        )
        e3 += 2.0 * np.einsum(
            "cbrt,atsc,rsab,cbrt,abrs",
            MO[o, o, v, v],
            MO[o, v, v, o],
            MO[v, v, o, o],
            eps_inv,
            eps_inv,
        )
        e3 += -1.0 * np.einsum(
            "acrs,dbac,srdb,acrs,dbrs",
            MO[o, o, v, v],
            MO[o, o, o, o],
            MO[v, v, o, o],
            eps_inv,
            eps_inv,
        )
        e3 += -1.0 * np.einsum(
            "abrt,trus,usab,abtr,abus",
            MO[o, o, v, v],
            MO[v, v, v, v],
            MO[v, v, o, o],
            eps_inv,
            eps_inv,
        )
        e3 += 2.0 * np.einsum(
            "bcrt,arbs,tsac,cbrt,acst",
            MO[o, o, v, v],
            MO[o, v, o, v],
            MO[v, v, o, o],
            eps_inv,
            eps_inv,
        )
        e3 += 2.0 * np.einsum(
            "cbrt,rasb,stac,cbrt,acst",
            MO[o, o, v, v],
            MO[v, o, v, o],
            MO[v, v, o, o],
            eps_inv,
            eps_inv,
        )
        e3 += -4.0 * np.einsum(
            "abrs,scat,rtbc,abrs,cbrt",
            MO[o, o, v, v],
            MO[v, o, o, v],
            MO[v, v, o, o],
            eps_inv,
            eps_inv,
        )
        e3 += -4.0 * np.einsum(
            "bcrt,atsc,rsab,bctr,abrs",
            MO[o, o, v, v],
            MO[o, v, v, o],
            MO[v, v, o, o],
            eps_inv,
            eps_inv,
        )
        e_corr = float(e2 + e3)
        out["e_hf_ha"] = float(e)
        out["e_corr_mp2_ha"] = float(e2)
        out["e_corr_mp3_ha"] = float(e3)
        out["e_corr_ha"] = e_corr
        out["e_tot_mp3_ha"] = float(e) + e_corr
        out["e_corr_mp2_pyscf_ha"] = float(m2.e_corr)
        out["oracle"] = "szabo_psi4numpy_on_pyscf_rhf_ao2mo"
    elif m == "ump2":
        # Explicit UMP2 on the UHF reference (triplet O2, etc.).
        m2 = mp.UMP2(mf).run()
        out["e_hf_ha"] = float(e)
        out["e_corr_ha"] = float(m2.e_corr)
        out["e_tot_mp2_ha"] = float(m2.e_tot)
        if hasattr(mf, "spin_square"):
            ss, mult = mf.spin_square()
            out["s2"] = float(ss)
            out["multiplicity"] = float(mult)
        if args.grad:
            g = m2.nuc_grad_method().kernel()
            out["grad_ha_bohr"] = g.tolist()
    elif m == "romp2":
        # ROHF reference → PySCF converts to UHF view + UMP2 (semi-canonical
        # spin-Fock diagonal denominators). No separate mp.ROMP2 class.
        m2 = mp.UMP2(mf).run()
        out["e_hf_ha"] = float(e)
        out["e_corr_ha"] = float(m2.e_corr)
        out["e_tot_mp2_ha"] = float(m2.e_tot)
        out["e_corr_ss_ha"] = float(m2.e_corr_ss)
        out["e_corr_os_ha"] = float(m2.e_corr_os)
        if hasattr(mf, "spin_square"):
            ss, mult = mf.spin_square()
            out["s2"] = float(ss)
            out["multiplicity"] = float(mult)
    elif m in ("tda", "tdhf", "rks-tda", "rks-tddft", "uks-tda", "uks-tddft"):
        # rks-tda: hybrid/pure TDDFT-TDA on the KS reference (exact-exchange
        # kernel for hybrids via PySCF's Casida A matrix).
        # rks-tddft / uks-tddft: full RPA TDDFT (Casida A,B) — tdscf.TDDFT, not TDA.
        # uks-tda: open-shell spin-conserving TDA on a UKS reference
        # (PySCF tdscf.TDA on dft.UKS; no closed-shell singlet/triplet split).
        if m == "tdhf":
            td = tdscf.TDHF(mf)
        elif m in ("rks-tddft", "uks-tddft"):
            td = tdscf.TDDFT(mf)
        else:
            td = tdscf.TDA(mf)
        td.nstates = args.nstates
        td.kernel()
        out["excitation_ha"] = [float(x) for x in td.e]
        out["excitation_ev"] = [float(x) * _HA_TO_EV for x in td.e]
        if hasattr(td, "converged"):
            out["td_converged"] = [bool(c) for c in td.converged]
        if hasattr(td, "oscillator_strength"):
            out["oscillator_strength"] = [float(x) for x in td.oscillator_strength()]
    elif m in ("uhf", "ump2", "sf-tda", "sf-tddft"):
        mf = scf.UHF(mol)
    elif m in ("sf-tda", "sf-tddft"):
        # Collinear spin-flip response (Shao, Head-Gordon, Krylov JCP 118, 4807
        # (2003)). PySCF mainline has no product SF-TDA/SF-RPA; build the
        # Ms-decreasing singles block with PySCF UHF MOs + ao2mo.
        # A_ia,jb = δ_ij δ_ab (ε_a^min - ε_i^maj) - (ij|ab)
        # B_ia,jb = -(ib|aj)   (full SF-RPA only; TDA drops B)
        from pyscf import ao2mo
        import numpy as np

        mo_a, mo_b = mf.mo_coeff
        ea, eb = mf.mo_energy
        occa, occb = mf.mo_occ
        oa = np.where(occa > 0)[0]
        va = np.where(occa == 0)[0]
        ob = np.where(occb > 0)[0]
        vb = np.where(occb == 0)[0]
        alpha_to_beta = len(oa) >= len(ob)
        if alpha_to_beta:
            o_idx, v_idx = oa, vb
            e_o, e_v = ea, eb
            Co, Cv = mo_a[:, oa], mo_b[:, vb]
            out["sf_channel"] = "alpha_to_beta"
        else:
            o_idx, v_idx = ob, va
            e_o, e_v = eb, ea
            Co, Cv = mo_b[:, ob], mo_a[:, va]
            out["sf_channel"] = "beta_to_alpha"
        no, nv = len(o_idx), len(v_idx)
        nov = no * nv
        nstates = min(args.nstates, nov)
        ijab = ao2mo.general(mol, (Co, Co, Cv, Cv), compact=False).reshape(
            no, no, nv, nv
        )
        A = np.zeros((nov, nov))
        for i in range(no):
            for a in range(nv):
                p = i * nv + a
                A[p, p] = e_v[v_idx[a]] - e_o[o_idx[i]]
                for j in range(no):
                    for b in range(nv):
                        q = j * nv + b
                        A[p, q] -= ijab[i, j, a, b]
        A = 0.5 * (A + A.T)
        if m == "sf-tda":
            w = np.linalg.eigvalsh(A)
            out["n_ov"] = int(nov)
            out["excitation_ha"] = [float(x) for x in w[:nstates]]
            out["excitation_ev"] = [float(x) * _HA_TO_EV for x in w[:nstates]]
        else:
            # Full SF-RPA: non-Hermitian [[A,B],[-B,-A]] (positive real ω).
            ibaj = ao2mo.general(mol, (Co, Cv, Cv, Co), compact=False).reshape(
                no, nv, nv, no
            )
            B = np.zeros((nov, nov))
            for i in range(no):
                for a in range(nv):
                    p = i * nv + a
                    for j in range(no):
                        for b in range(nv):
                            q = j * nv + b
                            B[p, q] -= ibaj[i, b, a, j]
            B = 0.5 * (B + B.T)
            M = np.block([[A, B], [-B, -A]])
            w = np.linalg.eigvals(M)
            pos = sorted(
                [
                    float(wi.real)
                    for wi in w
                    if wi.real > 1e-8 and abs(wi.imag) <= 1e-8
                ]
            )
            out["n_ov"] = int(nov)
            out["a_minus_b_min"] = float(np.linalg.eigvalsh(A - B)[0])
            out["excitation_ha"] = pos[:nstates]
            out["excitation_ev"] = [x * _HA_TO_EV for x in pos[:nstates]]
    elif m == "cisd":
        # CIS(D): Head-Gordon et al., Chem. Phys. Lett. 219, 21 (1994).
        # PySCF has no product CIS(D); evaluate spin-orbital eqs. (8)/(13)/(14)
        # on pyscf.tdscf.TDA amplitudes + MO ERIs (offline pin only).
        from pyscf import ao2mo
        import numpy as np

        td = tdscf.TDA(mf)
        td.nstates = args.nstates
        td.conv_tol = min(args.conv_tol, 1e-10)
        td.kernel()
        eps = mf.mo_energy
        C = mf.mo_coeff
        nmo = C.shape[1]
        nocc = mol.nelectron // 2
        nvir = nmo - nocc
        eri_mo = ao2mo.restore(1, ao2mo.full(mf._eri, C), nmo)
        occ_so = [(i, 0) for i in range(nocc)] + [(i, 1) for i in range(nocc)]
        vir_so = [(a, 0) for a in range(nocc, nmo)] + [
            (a, 1) for a in range(nocc, nmo)
        ]
        nso_o, nso_v = len(occ_so), len(vir_so)

        def so_eri(p, q, r, s):
            (P, sp), (Q, sq), (R, sr), (S, ss) = p, q, r, s
            val = 0.0
            if sp == sr and sq == ss:
                val += eri_mo[P, R, Q, S]
            if sp == ss and sq == sr:
                val -= eri_mo[P, S, Q, R]
            return val

        a_amp = np.zeros((nso_o, nso_o, nso_v, nso_v))
        for ii, i in enumerate(occ_so):
            for jj, j in enumerate(occ_so):
                for aa, a in enumerate(vir_so):
                    for bb, b in enumerate(vir_so):
                        denom = (
                            eps[a[0]] + eps[b[0]] - eps[i[0]] - eps[j[0]]
                        )
                        a_amp[ii, jj, aa, bb] = -so_eri(a, b, i, j) / denom

        roots = []
        for s in range(args.nstates):
            X = td.xy[s][0]
            bvec = np.zeros((nso_o, nso_v))
            for i in range(nocc):
                for a in range(nvir):
                    bvec[i, a] = X[i, a]
                    bvec[nocc + i, nvir + a] = X[i, a]
            omega = float(td.e[s])
            u = np.zeros((nso_o, nso_o, nso_v, nso_v))
            for ii, i in enumerate(occ_so):
                for jj, j in enumerate(occ_so):
                    for aa, a in enumerate(vir_so):
                        for bb, bv in enumerate(vir_so):
                            ssum = 0.0
                            for cc, c in enumerate(vir_so):
                                ssum += so_eri(a, bv, c, j) * bvec[ii, cc]
                                ssum -= so_eri(a, bv, c, i) * bvec[jj, cc]
                            for kk, k in enumerate(occ_so):
                                ssum += so_eri(k, a, i, j) * bvec[kk, bb]
                                ssum -= so_eri(k, bv, i, j) * bvec[kk, aa]
                            u[ii, jj, aa, bb] = ssum
            V = 0.0
            for ii, i in enumerate(occ_so):
                for jj, j in enumerate(occ_so):
                    for aa, a in enumerate(vir_so):
                        for bb, bv in enumerate(vir_so):
                            denom = (
                                eps[a[0]]
                                + eps[bv[0]]
                                - eps[i[0]]
                                - eps[j[0]]
                                - omega
                            )
                            V += u[ii, jj, aa, bb] ** 2 / denom
            V *= -0.25
            v = np.zeros((nso_o, nso_v))
            for ii in range(nso_o):
                for aa in range(nso_v):
                    ssum = 0.0
                    for jj in range(nso_o):
                        for kk in range(nso_o):
                            for bb in range(nso_v):
                                for cc in range(nso_v):
                                    jklbc = so_eri(
                                        occ_so[jj],
                                        occ_so[kk],
                                        vir_so[bb],
                                        vir_so[cc],
                                    )
                                    ssum += jklbc * (
                                        bvec[ii, bb] * a_amp[jj, kk, cc, aa]
                                        + bvec[jj, aa] * a_amp[ii, kk, cc, bb]
                                        + 2.0
                                        * bvec[jj, bb]
                                        * a_amp[ii, kk, aa, cc]
                                    )
                    v[ii, aa] = 0.5 * ssum
            U = float(np.sum(bvec * v))
            roots.append(
                {
                    "cis_ha": omega,
                    "cis_ev": omega * _HA_TO_EV,
                    "U_ha": U,
                    "V_ha": V,
                    "corr_ha": U + V,
                    "cisd_ha": omega + U + V,
                    "cisd_ev": (omega + U + V) * _HA_TO_EV,
                }
            )
        out["roots"] = roots
        out["excitation_cis_ev"] = [r["cis_ev"] for r in roots]
        out["excitation_cisd_ev"] = [r["cisd_ev"] for r in roots]
        if hasattr(td, "converged"):
            out["td_converged"] = [bool(c) for c in td.converged]
    elif m == "adc2":
        # Strict ADC(2) electronic excitation energies (singlet, EE).
        # Schirmer PRA 26, 2395 (1982); Trofimov & Schirmer JPB 28, 2299 (1995).
        from pyscf import adc as pyscf_adc

        myadc = pyscf_adc.ADC(mf)
        myadc.method = "adc(2)"
        myadc.method_type = "ee"
        myadc.conv_tol = min(args.conv_tol, 1e-10)
        e_adc, _v, _p, _x = myadc.kernel(nroots=args.nstates)
        out["e_corr_mp2_ha"] = float(myadc.e_corr)
        out["excitation_ha"] = [float(x) for x in e_adc]
        out["excitation_ev"] = [float(x) * _HA_TO_EV for x in e_adc]
        out["s1_ha"] = float(e_adc[0])
        out["s1_ev"] = float(e_adc[0]) * _HA_TO_EV
    elif m == "ccsd":
        # Closed-shell CCSD on the RHF reference (PySCF cc.CCSD).
        # Local import: other branches also re-bind `np`, so use a fresh name.
        import numpy as _np

        mycc = pyscf_cc.CCSD(mf)
        mycc.conv_tol = min(args.conv_tol, 1e-10)
        e_corr, t1, t2 = mycc.kernel()
        out["e_hf_ha"] = float(e)
        out["e_corr_ha"] = float(e_corr)
        out["e_tot_ccsd_ha"] = float(mycc.e_tot)
        out["e_tot_ha"] = float(mycc.e_tot)
        out["ccsd_converged"] = bool(mycc.converged)
        t1a = _np.asarray(t1).ravel()
        t2a = _np.asarray(t2).ravel()
        out["t1_norm"] = float(_np.sqrt(t1a.dot(t1a)))
        out["t2_norm"] = float(_np.sqrt(t2a.dot(t2a)))
        if args.grad:
            g = mycc.nuc_grad_method().kernel()
            out["grad_ha_bohr"] = _np.asarray(g, dtype=float).tolist()
    elif m == "ccsd-t":
        # Closed-shell CCSD(T): CCSD + non-iterative (T) (PySCF cc.CCSD.ccsd_t).
        import numpy as _np

        mycc = pyscf_cc.CCSD(mf)
        mycc.conv_tol = min(args.conv_tol, 1e-10)
        e_corr_ccsd, t1, t2 = mycc.kernel()
        e_t = float(mycc.ccsd_t())
        e_corr = float(e_corr_ccsd) + e_t
        out["e_hf_ha"] = float(e)
        out["e_corr_ccsd_ha"] = float(e_corr_ccsd)
        out["e_t_ha"] = e_t
        out["e_corr_ha"] = e_corr
        out["e_tot_ccsd_ha"] = float(mycc.e_tot)
        out["e_tot_ha"] = float(e) + e_corr
        out["ccsd_converged"] = bool(mycc.converged)
        t1a = _np.asarray(t1).ravel()
        t2a = _np.asarray(t2).ravel()
        out["t1_norm"] = float(_np.sqrt(t1a.dot(t1a)))
        out["t2_norm"] = float(_np.sqrt(t2a.dot(t2a)))
    elif m == "eom-ee-ccsd":
        # Singlet EOM-EE-CCSD on closed-shell CCSD (PySCF eomee_ccsd_singlet).
        import numpy as _np

        mycc = pyscf_cc.CCSD(mf)
        mycc.conv_tol = min(args.conv_tol, 1e-10)
        e_corr, _t1, _t2 = mycc.kernel()
        e_eom, _v = mycc.eomee_ccsd_singlet(nroots=args.nstates)
        e_eom = _np.atleast_1d(_np.asarray(e_eom, dtype=float))
        out["e_hf_ha"] = float(e)
        out["e_corr_ha"] = float(e_corr)
        out["e_tot_ccsd_ha"] = float(mycc.e_tot)
        out["e_tot_ha"] = float(mycc.e_tot)
        out["ccsd_converged"] = bool(mycc.converged)
        out["omega_ha"] = e_eom.tolist()
        out["omega_ev"] = (e_eom * _HA_TO_EV).tolist()
        out["omega0_ev"] = float(e_eom[0] * _HA_TO_EV)
        out["nstates"] = int(args.nstates)
    elif m == "eom-cc2":
        # Singlet EOM-CC2: RCCSD(cc2=True) + eomee_ccsd_singlet (full EE
        # Jacobian at CC2 amplitudes — stock PySCF has no top-level EOMCC2).
        import numpy as _np
        from pyscf.cc import rccsd as _rccsd

        mycc = _rccsd.RCCSD(mf)
        mycc.conv_tol = min(args.conv_tol, 1e-10)
        mycc.conv_tol_normt = min(args.conv_tol, 1e-10)
        mycc.cc2 = True
        e_corr, _t1, _t2 = mycc.kernel()
        e_eom, _v = mycc.eomee_ccsd_singlet(nroots=args.nstates)
        e_eom = _np.atleast_1d(_np.asarray(e_eom, dtype=float))
        out["e_hf_ha"] = float(e)
        out["e_corr_ha"] = float(e_corr)
        out["e_tot_cc2_ha"] = float(mycc.e_tot)
        out["e_tot_ha"] = float(mycc.e_tot)
        out["cc2_converged"] = bool(mycc.converged)
        out["omega_ha"] = e_eom.tolist()
        out["omega_ev"] = (e_eom * _HA_TO_EV).tolist()
        out["omega0_ev"] = float(e_eom[0] * _HA_TO_EV)
        out["nstates"] = int(args.nstates)
        out["oracle"] = "pyscf_rccsd_cc2_eomee_singlet"
    elif m == "eom-ea-ccsd":
        # EOM-EA-CCSD attachment energies on closed-shell CCSD (PySCF eaccsd).
        import numpy as _np

        mycc = pyscf_cc.CCSD(mf)
        mycc.conv_tol = min(args.conv_tol, 1e-10)
        e_corr, _t1, _t2 = mycc.kernel()
        e_eom, _v = mycc.eaccsd(nroots=args.nstates)
        e_eom = _np.atleast_1d(_np.asarray(e_eom, dtype=float))
        e_sorted = _np.sort(e_eom)
        uniq = []
        for w in e_sorted:
            if not uniq or abs(uniq[-1] - float(w)) > 1e-8:
                uniq.append(float(w))
        e_uniq = _np.asarray(uniq[: int(args.nstates)], dtype=float)
        out["e_hf_ha"] = float(e)
        out["e_corr_ha"] = float(e_corr)
        out["e_tot_ccsd_ha"] = float(mycc.e_tot)
        out["e_tot_ha"] = float(mycc.e_tot)
        out["ccsd_converged"] = bool(mycc.converged)
        out["ea_ha"] = e_uniq.tolist()
        out["ea_ev"] = (e_uniq * _HA_TO_EV).tolist()
        out["ea0_ev"] = float(e_uniq[0] * _HA_TO_EV)
        out["omega_ha"] = e_uniq.tolist()
        out["omega_ev"] = (e_uniq * _HA_TO_EV).tolist()
        out["omega0_ev"] = float(e_uniq[0] * _HA_TO_EV)
        out["nstates"] = int(args.nstates)
    elif m == "eom-ip-ccsd":
        # EOM-IP-CCSD ionization potentials on closed-shell CCSD (PySCF ipccsd).
        import numpy as _np

        mycc = pyscf_cc.CCSD(mf)
        mycc.conv_tol = min(args.conv_tol, 1e-10)
        e_corr, _t1, _t2 = mycc.kernel()
        e_eom, _v = mycc.ipccsd(nroots=args.nstates)
        e_eom = _np.atleast_1d(_np.asarray(e_eom, dtype=float))
        e_sorted = _np.sort(e_eom)
        uniq = []
        for w in e_sorted:
            if not uniq or abs(uniq[-1] - float(w)) > 1e-8:
                uniq.append(float(w))
        e_uniq = _np.asarray(uniq[: int(args.nstates)], dtype=float)
        out["e_hf_ha"] = float(e)
        out["e_corr_ha"] = float(e_corr)
        out["e_tot_ccsd_ha"] = float(mycc.e_tot)
        out["e_tot_ha"] = float(mycc.e_tot)
        out["ccsd_converged"] = bool(mycc.converged)
        out["ip_ha"] = e_uniq.tolist()
        out["ip_ev"] = (e_uniq * _HA_TO_EV).tolist()
        out["ip0_ev"] = float(e_uniq[0] * _HA_TO_EV)
        out["omega_ha"] = e_uniq.tolist()
        out["omega_ev"] = (e_uniq * _HA_TO_EV).tolist()
        out["omega0_ev"] = float(e_uniq[0] * _HA_TO_EV)
        out["nstates"] = int(args.nstates)
    elif m == "adc2x":
        # Extended ADC(2)-x EE (first-order doubles–doubles).
        from pyscf import adc as pyscf_adc

        myadc = pyscf_adc.ADC(mf)
        myadc.method = "adc(2)-x"
        myadc.method_type = "ee"
        myadc.conv_tol = min(args.conv_tol, 1e-10)
        e_adc, _v, _p, _x = myadc.kernel(nroots=args.nstates)
        out["e_corr_mp2_ha"] = float(myadc.e_corr)
        out["excitation_ha"] = [float(x) for x in e_adc]
        out["excitation_ev"] = [float(x) * _HA_TO_EV for x in e_adc]
        out["s1_ha"] = float(e_adc[0])
        out["s1_ev"] = float(e_adc[0]) * _HA_TO_EV
    elif m == "ip-adc2":
        # Strict ADC(2) ionization potentials (1h / 2h1p ISR).
        from pyscf import adc as pyscf_adc

        myadc = pyscf_adc.ADC(mf)
        myadc.method = "adc(2)"
        myadc.method_type = "ip"
        myadc.conv_tol = min(args.conv_tol, 1e-10)
        e_adc, _v, _p, _x = myadc.kernel(nroots=args.nstates)
        out["e_corr_mp2_ha"] = float(myadc.e_corr)
        out["ip_ha"] = [float(x) for x in e_adc]
        out["ip_ev"] = [float(x) * _HA_TO_EV for x in e_adc]
        out["ip1_ha"] = float(e_adc[0])
        out["ip1_ev"] = float(e_adc[0]) * _HA_TO_EV
    elif m == "ea-adc2":
        # Strict EA-ADC(2) electron-attachment energies (1p / 2p–1h).
        from pyscf import adc as pyscf_adc

        myadc = pyscf_adc.ADC(mf)
        myadc.method = "adc(2)"
        myadc.method_type = "ea"
        myadc.conv_tol = min(args.conv_tol, 1e-10)
        e_adc, _v, p_spec, _x = myadc.kernel(nroots=args.nstates)
        out["e_corr_mp2_ha"] = float(myadc.e_corr)
        out["ea_ha"] = [float(x) for x in e_adc]
        out["ea_ev"] = [float(x) * _HA_TO_EV for x in e_adc]
        out["ea0_ha"] = float(e_adc[0])
        out["ea0_ev"] = float(e_adc[0]) * _HA_TO_EV
        out["spec_factor"] = [float(x) for x in p_spec]
    elif m == "casci":
        # Closed-shell CASCI on the RHF reference (PySCF mcscf.CASCI).
        from pyscf import mcscf

        ncas = int(args.ncas)
        nelecas = int(args.nelecas)
        mc = mcscf.CASCI(mf, ncas, nelecas)
        mc.kernel()
        out["e_hf_ha"] = float(e)
        out["ncas"] = ncas
        out["nelecas"] = nelecas
        out["e_casci_ha"] = float(mc.e_tot)
        out["e_tot_ha"] = float(mc.e_tot)
        out["e_corr_ha"] = float(mc.e_tot) - float(e)
        out["e_cas_ha"] = float(mc.e_cas)
    elif m == "casscf":
        # Closed-shell CASSCF on the RHF reference (PySCF mcscf.CASSCF).
        from pyscf import mcscf

        ncas = int(args.ncas)
        nelecas = int(args.nelecas)
        mc = mcscf.CASSCF(mf, ncas, nelecas)
        mc.conv_tol = float(args.conv_tol)
        e_mc = mc.kernel()[0]
        out["e_hf_ha"] = float(e)
        out["ncas"] = ncas
        out["nelecas"] = nelecas
        out["ncore"] = int(mc.ncore)
        out["e_casscf_ha"] = float(e_mc)
        out["e_tot_ha"] = float(e_mc)
        out["e_corr_ha"] = float(e_mc) - float(e)
        out["casscf_converged"] = bool(mc.converged)
        out["e_cas_ha"] = float(mc.e_cas)
    elif m == "nevpt2":
        # Strongly contracted NEVPT2 on CASSCF (PySCF mrpt.NEVPT).
        from pyscf import mcscf, mrpt

        ncas = int(args.ncas)
        nelecas = int(args.nelecas)
        mc = mcscf.CASSCF(mf, ncas, nelecas)
        mc.conv_tol = float(args.conv_tol)
        e_mc = mc.kernel()[0]
        nev = mrpt.NEVPT(mc)
        e_pt2 = float(nev.kernel())
        nmo = int(mc.mo_coeff.shape[1])
        ncore = int(mc.ncore)
        out["e_hf_ha"] = float(e)
        out["ncas"] = ncas
        out["nelecas"] = nelecas
        out["ncore"] = ncore
        out["nvirt"] = nmo - ncore - ncas
        out["e_casscf_ha"] = float(e_mc)
        out["e_corr_ha"] = e_pt2  # PT2 vs CASSCF (PySCF NEVPT.e_corr)
        out["e_tot_ha"] = float(e_mc) + e_pt2
        out["e_corr_vs_hf_ha"] = float(e_mc) + e_pt2 - float(e)
        out["casscf_converged"] = bool(mc.converged)
        out["nevpt2_converged"] = bool(nev.converged)
    elif m == "b2plyp":
        # Double-hybrid total energy: E_KS + a_c * E_MP2 on the KS orbitals.
        m2 = mp.MP2(mf).run()
        a_c = 0.27
        e_corr_pt2 = a_c * float(m2.e_corr)
        out["e_ks_ha"] = float(e)
        out["e_corr_mp2_ha"] = float(m2.e_corr)
        out["e_corr_ss_ha"] = float(m2.e_corr_ss)
        out["e_corr_os_ha"] = float(m2.e_corr_os)
        out["e_corr_pt2_ha"] = e_corr_pt2
        out["e_corr_ha"] = e_corr_pt2
        out["e_tot_ha"] = float(e) + e_corr_pt2
        out["a_c"] = a_c
        out["a_x"] = 0.53

    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if out.get("converged", True) else 1


if __name__ == "__main__":
    sys.exit(main())
