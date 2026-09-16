#!/usr/bin/env python3
"""Fair wall-time: offline PySCF vs Atomiverse binary (never a ctest dep).

Gate for the method loop (docs/developer/pyscf-method-loop.md):
  within_10x  ⇔  T_atv_median / T_pyscf_median ≤ max_ratio  (default 10)

Examples:
  # PySCF-only timing (kernel wall after import)
  python tools/pyscf_oracle/fair_bench.py --method rhf --geom water --basis 6-31g*

  # Fair A/B
  python tools/pyscf_oracle/fair_bench.py --method ump2 --geom o2 --basis sto-3g \\
      --spin 2 --repeat 3 -- -- ./build/regressions/reg_ump2-o2-sto3g

Protocol:
  - Same geometry / basis / method class on both sides when possible.
  - Report median of --repeat runs (default 3).
  - Tiny molecules may be startup-dominated; still record ratio for the gate.
  - Extend --method as new families land (mp2, tda, ump2, …).
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time


def _median(xs: list[float]) -> float:
    return float(statistics.median(xs)) if xs else float("nan")


def _mol_code(geom: str, basis: str, charge: int, spin: int, cart: bool | None) -> str:
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
        "h3": """
H  0.000000  0.000000  0.000000
H  0.000000  0.000000  0.900000
H  0.000000  0.850000  0.450000
""",
        "n2": """
N  0.000000  0.000000  0.000000
N  0.000000  0.000000  1.097700
""",
    }
    atom = presets.get(geom)
    if atom is None:
        raise SystemExit(f"unknown geom {geom!r}; choose {sorted(presets)}")
    if cart is None:
        cart = not basis.lower().startswith("def2")
    return f"""
from pyscf import gto
mol = gto.M(atom='''{atom}''', basis={basis!r}, charge={charge}, spin={spin},
            unit='Angstrom', cart={cart}, verbose=0)
"""


def pyscf_kernel_seconds(method: str, geom: str, basis: str, charge: int, spin: int,
                         cart: bool | None, xc: str, nstates: int) -> float:
    setup = _mol_code(geom, basis, charge, spin, cart)
    body = {
        "rhf": "from pyscf import scf\nmf=scf.RHF(mol); mf.conv_tol=1e-10; mf.kernel()",
        "uhf": "from pyscf import scf\nmf=scf.UHF(mol); mf.conv_tol=1e-10; mf.kernel()",
        "rohf": "from pyscf import scf\nmf=scf.ROHF(mol); mf.conv_tol=1e-10; mf.kernel()",
        "rks": f"from pyscf import dft\nmf=dft.RKS(mol); mf.xc={xc!r}; mf.conv_tol=1e-10; mf.kernel()",
        "uks": f"from pyscf import dft\nmf=dft.UKS(mol); mf.xc={xc!r}; mf.conv_tol=1e-10; mf.kernel()",
        "roks": (
            f"from pyscf import dft\nmf=dft.ROKS(mol); mf.xc={xc!r}; mf.conv_tol=1e-10; mf.kernel()"
        ),
        "roks_grad": (
            f"from pyscf import dft\nmf=dft.ROKS(mol); mf.xc={xc!r}; mf.conv_tol=1e-10; mf.kernel(); "
            "mf.nuc_grad_method().kernel()"
        ),
        "mp2": "from pyscf import scf, mp\nmf=scf.RHF(mol); mf.conv_tol=1e-10; mf.kernel(); mp.MP2(mf).run()",
        "ump2": "from pyscf import scf, mp\nmf=scf.UHF(mol); mf.conv_tol=1e-10; mf.kernel(); mp.UMP2(mf).run()",
        "ump2_grad": (
            "from pyscf import scf, mp\n"
            "mf=scf.UHF(mol); mf.conv_tol=1e-10; mf.kernel(); "
            "m2=mp.UMP2(mf).run(); m2.nuc_grad_method().kernel()"
        ),
        "romp2": "from pyscf import scf, mp\nmf=scf.ROHF(mol); mf.conv_tol=1e-10; mf.kernel(); mp.UMP2(mf).run()",
        "mp3": (
            "from pyscf import scf, ao2mo\nimport numpy as np\n"
            "mf=scf.RHF(mol); mf.conv_tol=1e-10; mf.kernel()\n"
            "C=np.asarray(mf.mo_coeff); eps=np.asarray(mf.mo_energy)\n"
            "nmo=C.shape[1]; ndocc=mol.nelectron//2\n"
            "eri=ao2mo.restore(1, ao2mo.full(mol, C), nmo)\n"
            "MO=np.einsum('prqs->pqrs', eri)\n"
            "o=slice(0,ndocc); v=slice(ndocc,nmo)\n"
            "eocc,evirt=eps[:ndocc],eps[ndocc:]\n"
            "D=1.0/(eocc.reshape(-1,1,1,1)+eocc.reshape(-1,1,1)-evirt.reshape(-1,1)-evirt)\n"
            "e2=2*np.einsum('abrs,rsab,abrs',MO[o,o,v,v],MO[v,v,o,o],D)"
            "-np.einsum('abrs,rsba,abrs',MO[o,o,v,v],MO[v,v,o,o],D)\n"
            "e3=2*np.einsum('abru,ruts,tsab,abru,abts',MO[o,o,v,v],MO[v,v,v,v],MO[v,v,o,o],D,D)\n"
            "e3+=2*np.einsum('adrs,cbad,rscb,adrs,cbrs',MO[o,o,v,v],MO[o,o,o,o],MO[v,v,o,o],D,D)\n"
            "e3+=-4*np.einsum('acrt,rbsc,stab,acrt,abst',MO[o,o,v,v],MO[v,o,v,o],MO[v,v,o,o],D,D)\n"
            "e3+=-4*np.einsum('bcrt,rasb,stac,bcrt,acst',MO[o,o,v,v],MO[v,o,v,o],MO[v,v,o,o],D,D)\n"
            "e3+=8*np.einsum('acrt,btsc,rsab,acrt,abrs',MO[o,o,v,v],MO[o,v,v,o],MO[v,v,o,o],D,D)\n"
            "e3+=2*np.einsum('cbrt,atsc,rsab,cbrt,abrs',MO[o,o,v,v],MO[o,v,v,o],MO[v,v,o,o],D,D)\n"
            "e3+=-1*np.einsum('acrs,dbac,srdb,acrs,dbrs',MO[o,o,v,v],MO[o,o,o,o],MO[v,v,o,o],D,D)\n"
            "e3+=-1*np.einsum('abrt,trus,usab,abtr,abus',MO[o,o,v,v],MO[v,v,v,v],MO[v,v,o,o],D,D)\n"
            "e3+=2*np.einsum('bcrt,arbs,tsac,cbrt,acst',MO[o,o,v,v],MO[o,v,o,v],MO[v,v,o,o],D,D)\n"
            "e3+=2*np.einsum('cbrt,rasb,stac,cbrt,acst',MO[o,o,v,v],MO[v,o,v,o],MO[v,v,o,o],D,D)\n"
            "e3+=-4*np.einsum('abrs,scat,rtbc,abrs,cbrt',MO[o,o,v,v],MO[v,o,o,v],MO[v,v,o,o],D,D)\n"
            "e3+=-4*np.einsum('bcrt,atsc,rsab,bctr,abrs',MO[o,o,v,v],MO[o,v,v,o],MO[v,v,o,o],D,D)\n"
            "float(e2+e3)"
        ),
        "ccsd": "from pyscf import scf, cc\nmf=scf.RHF(mol); mf.conv_tol=1e-10; mf.kernel(); cc.CCSD(mf).run()",
        "ccsd_grad": (
            "from pyscf import scf, cc\n"
            "mf=scf.RHF(mol); mf.conv_tol=1e-10; mf.kernel(); "
            "my=cc.CCSD(mf); my.conv_tol=1e-10; my.kernel(); my.nuc_grad_method().kernel()"
        ),
        "ccsd_t": (
            "from pyscf import scf, cc\n"
            "mf=scf.RHF(mol); mf.conv_tol=1e-10; mf.kernel(); "
            "my=cc.CCSD(mf); my.conv_tol=1e-10; my.kernel(); my.ccsd_t()"
        ),
        "casci": (
            "from pyscf import scf, mcscf\n"
            "mf=scf.RHF(mol); mf.conv_tol=1e-10; mf.kernel(); "
            "mcscf.CASCI(mf, 2, 2).kernel()"
        ),
        "casscf": (
            "from pyscf import scf, mcscf\n"
            "mf=scf.RHF(mol); mf.conv_tol=1e-10; mf.kernel(); "
            "mc=mcscf.CASSCF(mf, 2, 2); mc.conv_tol=1e-10; mc.kernel()"
        ),
        "nevpt2": (
            "from pyscf import scf, mcscf, mrpt\n"
            "mf=scf.RHF(mol); mf.conv_tol=1e-10; mf.kernel(); "
            "mc=mcscf.CASSCF(mf, 2, 2); mc.conv_tol=1e-10; mc.kernel(); "
            "mrpt.NEVPT(mc).kernel()"
        ),
        "eom-ee-ccsd": (
            "from pyscf import scf, cc\n"
            "mf=scf.RHF(mol); mf.conv_tol=1e-10; mf.kernel(); "
            f"my=cc.CCSD(mf); my.conv_tol=1e-10; my.kernel(); my.eomee_ccsd_singlet(nroots={nstates})"
        ),
        "eom-cc2": (
            "from pyscf import scf\nfrom pyscf.cc import rccsd\n"
            "mf=scf.RHF(mol); mf.conv_tol=1e-10; mf.kernel(); "
            f"my=rccsd.RCCSD(mf); my.cc2=True; my.conv_tol=1e-10; my.kernel(); "
            f"my.eomee_ccsd_singlet(nroots={nstates})"
        ),
        "b2plyp": (
            "from pyscf import dft, mp\n"
            "mf=dft.RKS(mol); mf.xc='0.53*HF + 0.47*GGA_X_B88 + 0.73*GGA_C_LYP'; "
            "mf.conv_tol=1e-10; mf.kernel(); mp.MP2(mf).run()"
        ),
        "tda": (
            "from pyscf import scf, tdscf\nmf=scf.RHF(mol); mf.conv_tol=1e-10; mf.kernel(); "
            f"td=tdscf.TDA(mf); td.nstates={nstates}; td.kernel()"
        ),
        "tdhf": (
            "from pyscf import scf, tdscf\nmf=scf.RHF(mol); mf.conv_tol=1e-10; mf.kernel(); "
            f"td=tdscf.TDHF(mf); td.nstates={nstates}; td.kernel()"
        ),
        "rks_tda": (
            f"from pyscf import dft, tdscf\nmf=dft.RKS(mol); mf.xc={xc!r}; mf.conv_tol=1e-10; mf.kernel(); "
            f"td=tdscf.TDA(mf); td.nstates={nstates}; td.kernel()"
        ),
        "rks_tddft": (
            f"from pyscf import dft, tdscf\nmf=dft.RKS(mol); mf.xc={xc!r}; mf.conv_tol=1e-10; mf.kernel(); "
            f"td=tdscf.TDDFT(mf); td.nstates={nstates}; td.kernel()"
        ),
        "uks_tddft": (
            f"from pyscf import dft, tdscf\nmf=dft.UKS(mol); mf.xc={xc!r}; mf.conv_tol=1e-10; mf.kernel(); "
            f"td=tdscf.TDDFT(mf); td.nstates={nstates}; td.kernel()"
        ),
        "sf_tda": (
            "from pyscf import scf, ao2mo\nimport numpy as np\n"
            "ro=scf.ROHF(mol); ro.conv_tol=1e-12; ro.kernel()\n"
            "mf=scf.UHF(mol); mf.conv_tol=1e-12\n"
            "dm=np.asarray(ro.make_rdm1())\n"
            "dmin=(dm[0],dm[1]) if dm.ndim==3 else (dm,dm)\n"
            "mf.kernel(dmin)\n"
            "mo_a,mo_b=mf.mo_coeff; ea,eb=mf.mo_energy; occa,occb=mf.mo_occ\n"
            "oa=np.where(occa>0)[0]; va=np.where(occa==0)[0]\n"
            "ob=np.where(occb>0)[0]; vb=np.where(occb==0)[0]\n"
            "ab=len(oa)>=len(ob)\n"
            "o_idx,v_idx=(oa,vb) if ab else (ob,va)\n"
            "e_o,e_v=(ea,eb) if ab else (eb,ea)\n"
            "Co,Cv=(mo_a[:,oa],mo_b[:,vb]) if ab else (mo_b[:,ob],mo_a[:,va])\n"
            "no,nv=len(o_idx),len(v_idx); nov=no*nv\n"
            "ijab=ao2mo.general(mol,(Co,Co,Cv,Cv),compact=False).reshape(no,no,nv,nv)\n"
            "A=np.zeros((nov,nov))\n"
            "for i in range(no):\n"
            "  for a in range(nv):\n"
            "    p=i*nv+a; A[p,p]=e_v[v_idx[a]]-e_o[o_idx[i]]\n"
            "    for j in range(no):\n"
            "      for b in range(nv):\n"
            "        A[p,j*nv+b]-=ijab[i,j,a,b]\n"
            f"np.linalg.eigvalsh(A)[:{nstates}]"
        ),
        "adc2": (
            "from pyscf import scf, adc\nmf=scf.RHF(mol); mf.conv_tol=1e-10; mf.kernel(); "
            f"my=adc.ADC(mf); my.method='adc(2)'; my.method_type='ee'; "
            f"my.conv_tol=1e-8; my.kernel(nroots={nstates})"
        ),
        "g0w0": (
            "from pyscf import dft, tdscf\n"
            "from pyscf.gw import gw_exact\n"
            "mf=dft.RKS(mol); mf.xc='hf'; mf.conv_tol=1e-12; mf.kernel()\n"
            "nocc=mol.nelectron//2; nvir=mf.mo_coeff.shape[1]-nocc\n"
            "td=tdscf.dRPA(mf); td.nstates=max(1,nocc*nvir); td.verbose=0; td.kernel()\n"
            "ge=gw_exact.GWExact(mf, tdmf=td); ge.verbose=0; ge.kernel()"
        ),
        "delta_scf": (
            "from pyscf import scf\n"
            "from pyscf.scf import addons\n"
            "import numpy as np\n"
            "mf0=scf.RHF(mol); mf0.conv_tol=1e-12; mf0.kernel()\n"
            "nmo=mf0.mo_coeff.shape[1]; nocc=mol.nelectron//2\n"
            "mo0=np.asarray(mf0.mo_coeff)\n"
            "occ_a=np.zeros(nmo); occ_b=np.zeros(nmo)\n"
            "occ_a[:nocc]=1.0; occ_b[:nocc]=1.0\n"
            "occ_a[nocc-1]=0.0; occ_a[nocc]=1.0\n"
            "dm=(np.einsum('pi,i,qi->pq',mo0,occ_a,mo0),"
            "np.einsum('pi,i,qi->pq',mo0,occ_b,mo0))\n"
            "mfu=addons.mom_occ_(scf.UHF(mol),(mo0,mo0),(occ_a.copy(),occ_b.copy()))\n"
            "mfu.conv_tol=1e-12; mfu.max_cycle=200; mfu.kernel(dm)"
        ),
        "eom-ea-ccsd": (
            "from pyscf import scf, cc\n"
            "mf=scf.RHF(mol); mf.conv_tol=1e-10; mf.kernel(); "
            f"my=cc.CCSD(mf); my.conv_tol=1e-10; my.kernel(); my.eaccsd(nroots={nstates})"
        ),
        "eom-ip-ccsd": (
            "from pyscf import scf, cc\n"
            "mf=scf.RHF(mol); mf.conv_tol=1e-10; mf.kernel(); "
            f"my=cc.CCSD(mf); my.conv_tol=1e-10; my.kernel(); my.ipccsd(nroots={nstates})"
        ),
        "adc2x": (
            "from pyscf import scf, adc\nmf=scf.RHF(mol); mf.conv_tol=1e-10; mf.kernel(); "
            f"my=adc.ADC(mf); my.method='adc(2)-x'; my.method_type='ee'; "
            f"my.conv_tol=1e-8; my.kernel(nroots={nstates})"
        ),
        "ip-adc2": (
            "from pyscf import scf, adc\nmf=scf.RHF(mol); mf.conv_tol=1e-10; mf.kernel(); "
            f"my=adc.ADC(mf); my.method='adc(2)'; my.method_type='ip'; "
            f"my.conv_tol=1e-8; my.kernel(nroots={nstates})"
        ),
        "ea-adc2": (
            "from pyscf import scf, adc\nmf=scf.RHF(mol); mf.conv_tol=1e-10; mf.kernel(); "
            f"my=adc.ADC(mf); my.method='adc(2)'; my.method_type='ea'; "
            f"my.conv_tol=1e-8; my.kernel(nroots={nstates})"
        ),
        "sf_tddft": (
            "from pyscf import scf, ao2mo\nimport numpy as np\n"
            "ro=scf.ROHF(mol); ro.conv_tol=1e-12; ro.kernel()\n"
            "mf=scf.UHF(mol); mf.conv_tol=1e-12\n"
            "dm=np.asarray(ro.make_rdm1())\n"
            "dmin=(dm[0],dm[1]) if dm.ndim==3 else (dm,dm)\n"
            "mf.kernel(dmin)\n"
            "mo_a,mo_b=mf.mo_coeff; ea,eb=mf.mo_energy; occa,occb=mf.mo_occ\n"
            "oa=np.where(occa>0)[0]; va=np.where(occa==0)[0]\n"
            "ob=np.where(occb>0)[0]; vb=np.where(occb==0)[0]\n"
            "ab=len(oa)>=len(ob)\n"
            "o_idx,v_idx=(oa,vb) if ab else (ob,va)\n"
            "e_o,e_v=(ea,eb) if ab else (eb,ea)\n"
            "Co,Cv=(mo_a[:,oa],mo_b[:,vb]) if ab else (mo_b[:,ob],mo_a[:,va])\n"
            "no,nv=len(o_idx),len(v_idx); nov=no*nv\n"
            "ijab=ao2mo.general(mol,(Co,Co,Cv,Cv),compact=False).reshape(no,no,nv,nv)\n"
            "ibaj=ao2mo.general(mol,(Co,Cv,Cv,Co),compact=False).reshape(no,nv,nv,no)\n"
            "A=np.zeros((nov,nov)); B=np.zeros((nov,nov))\n"
            "for i in range(no):\n"
            "  for a in range(nv):\n"
            "    p=i*nv+a; A[p,p]=e_v[v_idx[a]]-e_o[o_idx[i]]\n"
            "    for j in range(no):\n"
            "      for b in range(nv):\n"
            "        q=j*nv+b; A[p,q]-=ijab[i,j,a,b]; B[p,q]-=ibaj[i,b,a,j]\n"
            "A=0.5*(A+A.T); B=0.5*(B+B.T)\n"
            "M=np.block([[A,B],[-B,-A]]); w=np.linalg.eigvals(M)\n"
            f"sorted([float(z.real) for z in w if z.real>1e-8 and abs(z.imag)<=1e-8])[:{nstates}]"
        ),
        "mp4": (
            "from pyscf import scf, ao2mo\nfrom itertools import combinations\nimport numpy as np\n"
            "mf=scf.RHF(mol); mf.conv_tol=1e-10; mf.kernel()\n"
            "C=np.asarray(mf.mo_coeff); eps=np.asarray(mf.mo_energy)\n"
            "nmo=C.shape[1]; ndocc=mol.nelectron//2; nocc=2*ndocc; nso=2*nmo\n"
            "eri=ao2mo.restore(1, ao2mo.full(mol, C), nmo)\n"
            "hmo=C.T@mf.get_hcore()@C\n"
            "hso=np.zeros((nso,nso))\n"
            "for p in range(nmo):\n"
            "  for q in range(nmo):\n"
            "    hso[2*p,2*q]=hmo[p,q]; hso[2*p+1,2*q+1]=hmo[p,q]\n"
            "V=np.zeros((nso,nso,nso,nso))\n"
            "for p in range(nso):\n"
            "  for q in range(nso):\n"
            "    for r in range(nso):\n"
            "      for s in range(nso):\n"
            "        val=0.0\n"
            "        if (p%2)==(r%2) and (q%2)==(s%2): val+=eri[p//2,r//2,q//2,s//2]\n"
            "        if (p%2)==(s%2) and (q%2)==(r%2): val-=eri[p//2,s//2,q//2,r//2]\n"
            "        V[p,q,r,s]=val\n"
            "eso=np.repeat(eps,2); dets=list(combinations(range(nso),nocc)); nd=len(dets); iref=dets.index(tuple(range(nocc)))\n"
            "def phase(di,holes,parts):\n"
            "  orb=list(di); ph=1\n"
            "  for h in holes:\n"
            "    p=orb.index(h); ph*=(-1)**p; orb.pop(p)\n"
            "  for p in parts:\n"
            "    ph*=(-1)**sum(1 for o in orb if o<p); orb.append(p); orb.sort()\n"
            "  return ph\n"
            "def Hme(d1,d2):\n"
            "  s1,s2=set(d1),set(d2); o1=sorted(s1-s2); o2=sorted(s2-s1); n=len(o1)\n"
            "  if n>2: return 0.0\n"
            "  if n==0:\n"
            "    e=sum(hso[i,i] for i in d1)\n"
            "    for i in d1:\n"
            "      for j in d1: e+=0.5*V[i,j,i,j]\n"
            "    return e\n"
            "  if n==1:\n"
            "    h,p=o1[0],o2[0]; ph=phase(d1,[h],[p]); com=sorted(s1&s2)\n"
            "    return ph*(hso[h,p]+sum(V[h,k,p,k] for k in com))\n"
            "  i,j=o1; a,b=o2; return phase(d1,[i,j],[a,b])*V[i,j,a,b]\n"
            "H=np.array([[Hme(a,b) for b in dets] for a in dets])\n"
            "H0=np.diag([sum(eso[p] for p in d) for d in dets]); Vm=H-H0\n"
            "E0=H0[iref,iref]; E1=Vm[iref,iref]; P0=np.zeros(nd); P0[iref]=1.0\n"
            "def R0(v):\n"
            "  o=np.zeros_like(v)\n"
            "  for i in range(nd):\n"
            "    if i!=iref: o[i]=v[i]/(E0-H0[i,i])\n"
            "  return o\n"
            "P1=R0(Vm@P0); e2=float(P0@Vm@P1)\n"
            "P2=R0(Vm@P1-E1*P1-e2*P0); e3=float(P0@Vm@P2)\n"
            "P3=R0(Vm@P2-E1*P2-e2*P1-e3*P0); e4=float(P0@Vm@P3)\n"
            "float(e2+e3+e4)"
        ),
        "cc2": (
            "from pyscf import scf\nfrom pyscf.cc import rccsd\n"
            "mf=scf.RHF(mol); mf.conv_tol=1e-10; mf.kernel(); "
            "my=rccsd.RCCSD(mf); my.cc2=True; my.conv_tol=1e-10; my.kernel()"
        ),
        "direct_rpa": (
            "from pyscf import scf, ao2mo\nimport numpy as np\n"
            "from numpy.linalg import eigh\nfrom scipy.linalg import sqrtm\n"
            "mf=scf.RHF(mol); mf.conv_tol=1e-12; mf.kernel()\n"
            "C=np.asarray(mf.mo_coeff); eps=np.asarray(mf.mo_energy)\n"
            "nmo=C.shape[1]; nocc=mol.nelectron//2; nvir=nmo-nocc\n"
            "eri=ao2mo.restore(1, ao2mo.full(mol, C), nmo)\n"
            "nov=nocc*nvir\n"
            "A=np.zeros((nov,nov)); B=np.zeros_like(A)\n"
            "for i in range(nocc):\n"
            "  for a in range(nvir):\n"
            "    ia=i*nvir+a; aa=nocc+a; A[ia,ia]=eps[aa]-eps[i]\n"
            "    for j in range(nocc):\n"
            "      for b in range(nvir):\n"
            "        jb=j*nvir+b; bb=nocc+b; v=2*eri[i,aa,j,bb]\n"
            "        A[ia,jb]+=v; B[ia,jb]+=v\n"
            "T=np.real(sqrtm(A-B)); M=0.5*((T@(A+B)@T)+(T@(A+B)@T).T)\n"
            "w2,_=eigh(M); Om=np.sqrt(np.maximum(w2,0)); 0.5*(Om.sum()-np.trace(A))"
        ),
        "bse": (
            "from pyscf import dft, tdscf, ao2mo\n"
            "from pyscf.gw import gw_exact\n"
            "import numpy as np\n"
            "mf=dft.RKS(mol); mf.xc='hf'; mf.conv_tol=1e-12; mf.kernel()\n"
            "C=mf.mo_coeff; nmo=C.shape[1]; nocc=mol.nelectron//2; nvir=nmo-nocc\n"
            "eri=ao2mo.restore(1, ao2mo.full(mol, C), nmo)\n"
            "td=tdscf.dRPA(mf); td.nstates=max(1,nocc*nvir); td.verbose=0; td.kernel()\n"
            "ge=gw_exact.GWExact(mf, tdmf=td); ge.verbose=0; ge.kernel()\n"
            "eps=np.asarray(ge.mo_energy); nov=nocc*nvir\n"
            "S=np.zeros((nov,nov))\n"
            "for a in range(nvir):\n"
            "  for i in range(nocc):\n"
            "    p=a*nocc+i; S[p,p]=eps[i]-eps[nocc+a]\n"
            "    for b in range(nvir):\n"
            "      for j in range(nocc):\n"
            "        q=b*nocc+j; S[p,q]-=4.0*eri[i,nocc+a,j,nocc+b]\n"
            "Pi=4.0*np.linalg.inv(S)\n"
            "def W(p,q,r,s):\n"
            "  acc=eri[p,q,r,s]\n"
            "  for a in range(nvir):\n"
            "    for i in range(nocc):\n"
            "      ia=a*nocc+i\n"
            "      for b in range(nvir):\n"
            "        for j in range(nocc):\n"
            "          jb=b*nocc+j\n"
            "          acc+=eri[p,q,i,nocc+a]*Pi[ia,jb]*eri[j,nocc+b,r,s]\n"
            "  return acc\n"
            "A=np.zeros((nov,nov)); B=np.zeros((nov,nov))\n"
            "for a in range(nvir):\n"
            "  for i in range(nocc):\n"
            "    ia=a*nocc+i; aa=nocc+a\n"
            "    for b in range(nvir):\n"
            "      for j in range(nocc):\n"
            "        jb=b*nocc+j; bb=nocc+b; vc=eri[i,aa,j,bb]\n"
            "        A[ia,jb]=2*vc-W(i,j,aa,bb); B[ia,jb]=2*vc-W(i,bb,aa,j)\n"
            "        if ia==jb: A[ia,jb]+=eps[aa]-eps[i]\n"
            "ApB=0.5*((A+B)+(A+B).T); AmB=0.5*((A-B)+(A-B).T)\n"
            "sm,Um=np.linalg.eigh(AmB); T=(Um*np.sqrt(sm))@Um.T\n"
            "M=0.5*(T@ApB@T+(T@ApB@T).T); np.sqrt(np.linalg.eigvalsh(M))"
        ),
        "d4": (
            f"from pyscf.dispersion import dftd4 as d4m\n"
            f"m=d4m.DFTD4Dispersion(mol, xc={xc!r}, version='d4', atm=True)\n"
            f"for _ in range(2000): m.get_dispersion()"
        ),
        "eom-sf-ccsd": (
            "from pyscf import scf, cc\n"
            "mf=scf.UHF(mol); mf.conv_tol=1e-10; mf.kernel(); "
            f"my=cc.UCCSD(mf); my.conv_tol=1e-10; my.kernel(); my.eomsf_ccsd(nroots={nstates})"
        ),
        "gf2": (
            "from pyscf import scf, agf2\n"
            "import numpy as np\n"
            "from pyscf.agf2.chempot import binsearch_chempot\n"
            "mf=scf.RHF(mol); mf.conv_tol=1e-12; mf.kernel()\n"
            "g1=agf2.RAGF2(mf); g1.verbose=0\n"
            "eri=g1.ao2mo(); g0=g1.init_gf(); se0=g1.build_se(eri,g0)\n"
            "fock=np.diag(np.asarray(mf.mo_energy,dtype=float))\n"
            "w,v=se0.eig(fock)\n"
            "mu,_=binsearch_chempot((w,v),g1.nmo,g1.nocc*2)\n"
            "g_one=agf2.GreensFunction(w,v[:g1.nmo],chempot=mu)\n"
            "g1.get_ip(g_one,nroots=1); g1.get_ea(g_one,nroots=1)\n"
            "g1.energy_mp2(mf.mo_energy,se0)"
        ),
    }.get(method)
    if body is None:
        raise SystemExit(f"unsupported --method {method!r}")
    code = setup + "\nimport time\nt0=time.perf_counter()\n" + body + "\nprint(time.perf_counter()-t0)\n"
    r = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    return float(r.stdout.strip().splitlines()[-1])


def bench_cmd(cmd: list[str]) -> float:
    t0 = time.perf_counter()
    # Capture child I/O so PASS/FAIL banners do not pollute JSON stdout.
    subprocess.run(cmd, check=True, capture_output=True)
    return time.perf_counter() - t0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", default="rhf",
                   choices=["rhf", "uhf", "rohf", "rks", "uks", "roks", "roks_grad", "mp2", "ump2", "romp2", "tda", "tdhf", "adc2", "rks_tda", "rks_tddft", "sf_tda", "g0w0", "mp3", "ccsd", "ccsd_t", "casci", "b2plyp", "delta_scf", "eom-ee-ccsd", "eom-cc2", "eom-ea-ccsd", "adc2x", "ip-adc2", "ea-adc2", "sf_tddft", "mp4", "cc2", "direct_rpa", "bse", "d4", "eom-sf-ccsd", "gf2", "eom-ip-ccsd", "uks_tddft", "casscf", "nevpt2", "ccsd_grad", "ump2_grad"])
    p.add_argument("--geom", default="water")
    p.add_argument("--basis", default="sto-3g")
    p.add_argument("--charge", type=int, default=0)
    p.add_argument("--spin", type=int, default=0)
    p.add_argument("--xc", default="b3lyp")
    p.add_argument("--nstates", type=int, default=3)
    p.add_argument("--cart", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--max-ratio", type=float, default=10.0,
                   help="within_10x if T_atv/T_pyscf <= this (default 10)")
    p.add_argument("atv_cmd", nargs=argparse.REMAINDER,
                   help="optional: -- ./path/to/reg_binary [args…]")
    args = p.parse_args()

    out: dict = {
        "ok": True,
        "method": args.method,
        "geom": args.geom,
        "basis": args.basis,
        "max_ratio": args.max_ratio,
        "repeat": args.repeat,
    }

    try:
        pt = [pyscf_kernel_seconds(args.method, args.geom, args.basis, args.charge, args.spin,
                                   args.cart, args.xc, args.nstates)
              for _ in range(args.repeat)]
        out["pyscf_wall_s"] = pt
        out["pyscf_median_s"] = _median(pt)
    except Exception as e:
        out["ok"] = False
        out["pyscf_error"] = str(e)
        print(json.dumps(out, indent=2, sort_keys=True))
        return 2

    cmd = list(args.atv_cmd or [])
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if cmd:
        try:
            at = [bench_cmd(cmd) for _ in range(args.repeat)]
            out["atv_wall_s"] = at
            out["atv_median_s"] = _median(at)
            if out["pyscf_median_s"] > 0:
                ratio = out["atv_median_s"] / out["pyscf_median_s"]
                out["ratio_atv_over_pyscf"] = ratio
                out["within_10x"] = bool(ratio <= args.max_ratio)
        except Exception as e:
            out["ok"] = False
            out["atv_error"] = str(e)

    print(json.dumps(out, indent=2, sort_keys=True))
    if not out.get("ok"):
        return 1
    if cmd and out.get("within_10x") is False:
        return 3  # correct path may still exist; gate failed on speed
    return 0


if __name__ == "__main__":
    sys.exit(main())
