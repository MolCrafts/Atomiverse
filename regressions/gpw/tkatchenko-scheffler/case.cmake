# gpw/tkatchenko-scheffler: one case, one section per pinned system or quantity.

atv_regression(SECTION gpw-ts-ice
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-ts-forces
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-ts-stress
  REQUIRES GPW
  THREADS 8)
