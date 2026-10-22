# semiempirical/gfn0-xtb: one case, one section per pinned system or quantity.

atv_regression(SECTION gfn0-beyond-toy
  REQUIRES SEMIEMPIRICAL)

atv_regression(SECTION gfn0-xtb-forces
  REQUIRES SEMIEMPIRICAL)

atv_regression(SECTION gfn0-xtb-analytic-hessian
  REQUIRES SEMIEMPIRICAL VIB)
