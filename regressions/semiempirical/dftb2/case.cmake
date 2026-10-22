# semiempirical/dftb2: one case, one section per pinned system or quantity.

atv_regression(SECTION h2o_dftb2_mio
  REQUIRES SEMIEMPIRICAL)

atv_regression(SECTION dftb2-dftbplus-energy-pin
  REQUIRES SEMIEMPIRICAL)

atv_regression(SECTION dftb2-3ob-organic
  REQUIRES SEMIEMPIRICAL)

atv_regression(SECTION dftb2-analytic-forces
  REQUIRES SEMIEMPIRICAL)

atv_regression(SECTION dftb2-analytic-stress
  REQUIRES SEMIEMPIRICAL)

atv_regression(SECTION dftb2-ewald-gamma
  REQUIRES SEMIEMPIRICAL)

atv_regression(SECTION dftb2-scc-guess
  REQUIRES SEMIEMPIRICAL)

atv_regression(SECTION dftb2-dispersion-d3
  REQUIRES SEMIEMPIRICAL)

atv_regression(SECTION dftb2-dispersion-d3-periodic
  REQUIRES SEMIEMPIRICAL)

atv_regression(SECTION spin-dftb2-o2
  REQUIRES SEMIEMPIRICAL)

atv_regression(SECTION dftb2-spin-analytic-forces
  REQUIRES SEMIEMPIRICAL)

atv_regression(SECTION dftb2-spin-analytic-stress
  REQUIRES SEMIEMPIRICAL)

atv_regression(SECTION dftb-plus-u
  REQUIRES SEMIEMPIRICAL)
