# gpw/xdm: one case, one section per pinned system or quantity.

atv_regression(SECTION gpw-xdm-ice
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-xdm-forces
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-xdm-stress
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-xdm-kpoint-forces
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-xdm-kpoint-stress
  REQUIRES GPW
  THREADS 8)
