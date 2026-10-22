# md/domain-decomposition: one case, one section per pinned system or quantity.

atv_regression(SECTION comm-swaplist-halo-mpi
  REQUIRES MD
  MPIEXEC)

atv_regression(SECTION comm-topology-multihop
  REQUIRES MD
  MPIEXEC)

atv_regression(SECTION lammps-comm-brick-05-presets
  REQUIRES MD
  MPIEXEC)

atv_regression(SECTION multihop-widebox-parity
  REQUIRES MD
  MPIEXEC)
