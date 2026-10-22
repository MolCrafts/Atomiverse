# pw/rks: one case, one section per pinned system or quantity.

atv_regression(SECTION pw-lda-si
  REQUIRES PW)

atv_regression(SECTION pw-pbe-gga-si
  REQUIRES PW)

atv_regression(SECTION h_force_analytic
  REQUIRES PW)

atv_regression(SECTION h_force_fd
  REQUIRES PW)

atv_regression(SECTION h_stress_cubic
  REQUIRES PW)

atv_regression(SECTION pw-analytic-stress-si
  REQUIRES PW)

atv_regression(SECTION pw-cuda-lda-si
  REQUIRES PW
  DEVICES both)

atv_regression(SECTION cuda-pw-forces
  REQUIRES PW
  DEVICES both)

atv_regression(SECTION cuda-pw-stress
  REQUIRES PW
  DEVICES both)

atv_regression(SECTION cuda-pw-stress-pbe
  REQUIRES PW
  DEVICES both)

atv_regression(SECTION mpi-pw-scaling-4rank
  REQUIRES PW
  MPIEXEC)
