import math
K = 14.399645  # ATV units::coulomb  [eV A / e^2]
BOHR = 0.529177210903
HA = 27.211386245988

def thole(u):
    if u <= 0: return 1.0,1.0,1.0
    u3=u**3; e=math.exp(-u3)
    return 1-e, 1-(1+u3)*e, 1-(1+u3+0.6*u3*u3)*e

def solve(r, q0,q1, a0,a1, thole_a):
    # sites on x axis: 0 at 0, 1 at r
    if thole_a>0 and a0>0 and a1>0:
        u = thole_a*r/ (a0*a1)**(1/6)
        l3,l5,l7 = thole(u)
    else:
        l3=l5=l7=1.0
    # E_perm_x at 0 from q1 : dx=-r, fac=l3*q1/r^3 -> Ex = fac*dx
    E0 = l3*q1/r**3 * (-r)
    E1 = l3*q0/r**3 * ( r)
    # dipole coupling constant along x (axial): K_ax
    # e_tot_x(i) += 3*(l5/r^5)*(mu_x*dx)*dx - (l3/r^3)*mu_x, dx=+-r
    Kax = 3*l5/r**3 - l3/r**3
    # mu0 = a0*(E0 + Kax*mu1);  mu1 = a1*(E1 + Kax*mu0)
    # solve 2x2
    det = 1 - a0*a1*Kax*Kax
    mu0 = a0*(E0 + Kax*a1*E1)/det
    mu1 = a1*(E1 + Kax*a0*E0)/det
    Epol = -0.5*K*(mu0*E0 + mu1*E1)
    return E0,E1,Kax,mu0,mu1,Epol

for ta,label in [(0.39,'thole a=0.39 (current pin)'), (0.0,'UNDAMPED (MW-comparable)')]:
    E0,E1,Kax,mu0,mu1,Epol = solve(3.0, 1.0,-1.0, 1.0,1.5, ta)
    print('---',label)
    print('  E0=%.16e E1=%.16e Kax=%.16e'%(E0,E1,Kax))
    print('  mu0 = %.16e e.A   = %.16e e.Bohr'%(mu0, mu0/BOHR))
    print('  mu1 = %.16e e.A   = %.16e e.Bohr'%(mu1, mu1/BOHR))
    print('  Epol= %.16e eV    = %.16e Ha'%(Epol, Epol/HA))
    # force: analytic via ATV formula, axial
    l3=l5=l7=1.0
    if ta>0:
        u = ta*3.0/ (1.0*1.5)**(1/6); l3,l5,l7 = thole(u)
    r=3.0; q0,q1=1.0,-1.0
    def force_on(i):
        if i==0: qi,mui,qj,muj,dx = q0,mu0,q1,mu1,-r
        else:    qi,mui,qj,muj,dx = q1,mu1,q0,mu0, r
        r3=r**3; r5=r**5; r7=r**7
        mjr = muj*dx; mir = mui*dx
        Em = l5*3*mjr*dx/r5 - l3*muj/r3
        f = qi*Em
        f += qj*(l3*mui/r3 - l5*3*mir*dx/r5)
        f += l5*3/r5*(mui*muj*dx + mjr*mui + mir*muj) - l7*15/r7*mir*mjr*dx
        return K*f
    print('  F0x = %.16e eV/A   F1x = %.16e eV/A'%(force_on(0), force_on(1)))
    print('        = %.16e Ha/Bohr'%(force_on(0)/HA*BOHR))
