import os
BOHR=0.529177210903; HA=27.211386245988
base="/tmp/mw_oracle_pim-induced-dipole-bulk/scan"
def coul(d): return float([l for l in open(os.path.join(d,"energies_breakdown.out")) if not l.startswith('#')][0].split()[4])
def dip(d):
    ln=[l for l in open(os.path.join(d,"dipoles.out")) if not l.startswith('#')]; return [float(ln[0].split()[0]),float(ln[1].split()[0])]
def fx(d):
    ln=[l for l in open(os.path.join(d,"forces.out")) if not l.startswith('#')]; return [float(ln[0].split()[0]),float(ln[1].split()[0])]
def obs(L):
    p=f"{base}/L{L}_pol"; n=f"{base}/L{L}_nopol"; m=dip(p)
    return dict(mu0=m[0],mu1=m[1],Epol=coul(p)-coul(n),F0=fx(p)[0]-fx(n)[0])
ATV=dict(mu0=2.3523561717333238e-01,mu1=3.4109164490133193e-01,Epol=-8.9660036212596498e-03,F0=6.7922030228728921e-03)
Ls=[50,100,200,400,800,1600,3200]
D={L:obs(L) for L in Ls}
print("RAW MW output (finite box), relative deviation from ATV isolated-pair value:")
for L in Ls:
    print(f"  L={L:5d} A  mu0={D[L]['mu0']:.15e} ({(D[L]['mu0']-ATV['mu0'])/ATV['mu0']:+.3e})  "
          f"Epol={D[L]['Epol']:.15e} ({(D[L]['Epol']-ATV['Epol'])/abs(ATV['Epol']):+.3e})  "
          f"F0={D[L]['F0']:.15e} ({(D[L]['F0']-ATV['F0'])/ATV['F0']:+.3e})")
print("\nsuccessive Richardson f_inf=(8f(2L)-f(L))/7 :")
for k in ATV:
    row=[]
    for i in range(len(Ls)-1):
        a,b=D[Ls[i]][k],D[Ls[i+1]][k]; row.append((8*b-a)/7)
    print(f"  {k:>5}: "+"  ".join(f"{v:.13e}" for v in row))
    print(f"  {'':>5}  rel to ATV: "+"  ".join(f"{(v-ATV[k])/abs(ATV[k]):+.2e}" for v in row))
