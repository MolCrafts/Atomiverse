# gpw/dft-plus-u: one case, one section per pinned system or quantity.

atv_regression(SECTION gpw-dftu-nio
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-dftu-forces
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-dftu-stress
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-dftu-kpoint-forces
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-dftu-kpoint-stress
  REQUIRES GPW
  THREADS 8)
