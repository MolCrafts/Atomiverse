# gpw/d4: one case, one section per pinned system or quantity.

atv_regression(SECTION gpw-d4-water
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-d4-forces
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-d4-stress
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-d4-atm
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-d4-atm-forces
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-d4-atm-stress
  REQUIRES GPW
  THREADS 8)
