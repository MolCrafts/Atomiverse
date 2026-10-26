import mpmath as mp
mp.mp.dps=90
D=mp.mpf

def lyp_f(ra,rb,saa,sab,sbb):
 n=ra+rb;u=2*ra/n;v=2*rb/n;g=u*v;r=n**(-D(1)/3)
 denom=1+D('.349')*r
 delta=(D('.2533')+D('.349')/denom)*r
 omega=D('.132')*mp.exp(-D('.2533')*r)/denom
 h=(2/n)**(D(8)/3)
 xa=saa*h;xb=sbb*h
 t=-(saa+2*sab+sbb)*n**(-D(8)/3)*(g*(47-7*delta)/72-D(2)/3)
 t-=D(3)/20*(3*mp.pi**2)**(D(2)/3)*g*(u**(D(8)/3)+v**(D(8)/3))
 t+=2**(D(1)/3)*g*(D(5)/2-delta/18)*(xa+xb)/32
 t+=2**(D(1)/3)*g*(delta-11)*(u*xa+v*xb)/576
 t-=2**(D(1)/3)/8*(D(2)/3*(xa+xb)-(u*u*xb+v*v*xa)/4)
 return n*D('.04918')*(omega*t-g/denom)
print('Diagnostic high-precision evaluation of the same Miehlich/LYP symbolic functional, not an external oracle')
print('mpmath',mp.__version__,'decimal precision',mp.mp.dps)
ra,rb=D('1e-6'),D('1e-8')
for s in ['.01','.1','1']:
 sigma=D(s)
 f=lyp_f(ra,rb,sigma,sigma,sigma)
 va=mp.diff(lambda x:lyp_f(x,rb,sigma,sigma,sigma),ra)
 vb=mp.diff(lambda x:lyp_f(ra,x,sigma,sigma,sigma),rb)
 print('sigma',s,'f=',mp.nstr(f,60),'vrho_a=',mp.nstr(va,60),'vrho_b=',mp.nstr(vb,60))
