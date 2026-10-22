# gpw/mbd: one case, one section per pinned system or quantity.

atv_regression(SECTION gpw-mbd-ice
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-mbd-periodic
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-mbd-hirshfeld
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-mbd-forces
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-mbd-stress
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-mbd-kpoint-forces
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-mbd-kpoint-stress
  REQUIRES GPW
  THREADS 8)
