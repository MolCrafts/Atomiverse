# gpw/vdw-df: one case, one section per pinned system or quantity.

atv_regression(SECTION gpw-vdw-df-ice
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-rvv10-ice
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-vdw-scf-potential
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-vdw-stress
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-vdw-uks
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-vdw-kpoints
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-vdw-kpoint-forces
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-vdw-multigrid
  REQUIRES GPW
  THREADS 8)
