# semiempirical/gfn2-xtb: one case, one section per pinned system or quantity.

atv_regression(SECTION gfn2-xtb-water
  REQUIRES SEMIEMPIRICAL)

atv_regression(SECTION gfn2-xtb-forces
  REQUIRES SEMIEMPIRICAL)

atv_regression(SECTION gfn2-aes-fock
  REQUIRES SEMIEMPIRICAL)

atv_regression(SECTION gfn2-camm-r
  REQUIRES SEMIEMPIRICAL)

atv_regression(SECTION gfn2-quadrupole-aes
  REQUIRES SEMIEMPIRICAL)

atv_regression(SECTION gfn-xtb-analytic-hessian
  REQUIRES SEMIEMPIRICAL VIB)
