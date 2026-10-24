# semiempirical/am1: one case, one section per pinned system or quantity.

atv_regression(SECTION am1-water-mopac-energy
  REQUIRES SEMIEMPIRICAL DEVICES both)

atv_regression(SECTION am1_water_force
  REQUIRES SEMIEMPIRICAL)

atv_regression(SECTION nddo-sharp-discriminator
  REQUIRES SEMIEMPIRICAL)

atv_regression(SECTION se-analytic-hessian
  REQUIRES SEMIEMPIRICAL VIB)

# Fixed-duration NVE uses the same archived distorted-water Hamiltonian pin.
atv_regression(SECTION nddo-nve
  REQUIRES SEMIEMPIRICAL MD DEVICES both)
