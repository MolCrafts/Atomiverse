# One method case; CPU and CUDA share the executed external fixtures.
atv_regression(SECTION pm3-mopac
  REQUIRES SEMIEMPIRICAL DEVICES both)

# Fixed-duration NVE uses the same archived distorted-water Hamiltonian pin.
atv_regression(SECTION nddo-nve
  REQUIRES SEMIEMPIRICAL MD DEVICES both)
