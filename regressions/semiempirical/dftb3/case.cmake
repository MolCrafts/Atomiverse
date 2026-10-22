# semiempirical/dftb3: one case, one section per pinned system or quantity.

atv_regression(SECTION dftb3-water-mio
  REQUIRES SEMIEMPIRICAL)

atv_regression(SECTION dftb3-analytic-forces
  REQUIRES SEMIEMPIRICAL)

atv_regression(SECTION dftb3-periodic-forces
  REQUIRES SEMIEMPIRICAL)
