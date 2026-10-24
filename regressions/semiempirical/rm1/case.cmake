# semiempirical/rm1: one case, one section per pinned system or quantity.

atv_regression(SECTION rm1-water-mopac-energy
  REQUIRES SEMIEMPIRICAL)

# Fixed-duration NVE uses the same archived distorted-water Hamiltonian pin.
atv_regression(SECTION nddo-nve
  REQUIRES SEMIEMPIRICAL MD DEVICES both)
