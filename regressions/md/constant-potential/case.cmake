# md/constant-potential: one case, one section per pinned system or quantity.

atv_regression(SECTION const-pot-bo-parallel-plate
  REQUIRES MD)

atv_regression(SECTION const-pot-cg-large-electrode
  REQUIRES MD)

atv_regression(SECTION const-pot-charge-halo-mpi
  REQUIRES MD
  MPIEXEC)

atv_regression(SECTION const-pot-charge-observer
  REQUIRES MD)

atv_regression(SECTION const-pot-il-edl-box
  REQUIRES KSPACE MD)

atv_regression(SECTION const-pot-md-nvt-short
  REQUIRES KSPACE MD)

atv_regression(SECTION const-pot-tip4p-edl
  REQUIRES KSPACE MD)

atv_regression(SECTION nanoporous-carbon-const-pot
  REQUIRES KSPACE MD)

atv_regression(SECTION per-electrode-eta
  REQUIRES MD)

atv_regression(SECTION gaussian-electrode-charge
  REQUIRES MD)

atv_regression(SECTION electrode-atom-forces
  REQUIRES KSPACE MD)

atv_regression(SECTION electrode-charge-predictor
  REQUIRES MD)

atv_regression(SECTION freeze-electrode-atoms
  REQUIRES MD)

atv_regression(SECTION global-electroneutrality-bo
  REQUIRES KSPACE MD)

atv_regression(SECTION piston-electrode-pz
  REQUIRES MD)

atv_regression(SECTION dipole-electrode-joint
  REQUIRES MD)
