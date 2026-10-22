# md/nve: one case, one section per pinned system or quantity.

atv_regression(SECTION lj_argon
  REQUIRES MD
  DEVICES both)

atv_regression(SECTION pe_bonded_nve
  REQUIRES MD)

atv_regression(SECTION cpu-velocity-verlet-initial-force
  REQUIRES MD POTENTIALS)

atv_regression(SECTION cuda-velocity-verlet-initial-force
  REQUIRES MD POTENTIALS NEIGHBOR CUDA)

atv_regression(SECTION cuda-multi-gpu-trajectory
  REQUIRES MD POTENTIALS NEIGHBOR
  DEVICES cuda
  GPUS 2)
