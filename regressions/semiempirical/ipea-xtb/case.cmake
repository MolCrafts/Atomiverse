# semiempirical/ipea-xtb: one case, one section per pinned system or quantity.

atv_regression(SECTION ipea-xtb-small
  REQUIRES SEMIEMPIRICAL)

atv_regression(SECTION ipea-xtb-forces
  REQUIRES SEMIEMPIRICAL)

atv_regression(SECTION ipea-xtb-vip
  REQUIRES SEMIEMPIRICAL)

atv_regression(SECTION ipea-xtb-analytic-hessian
  REQUIRES SEMIEMPIRICAL VIB)
