# pw/collinear-spin: one case, one section per pinned system or quantity.

atv_regression(SECTION pw-spin-polarized-o2
  REQUIRES PW)

atv_regression(SECTION pw-spin-gga
  REQUIRES PW)

atv_regression(SECTION cuda-pw-collinear-spin
  REQUIRES PW
  DEVICES both)

atv_regression(SECTION cuda-pw-spin-forces
  REQUIRES PW
  DEVICES both)

atv_regression(SECTION cuda-pw-spin-pbe
  REQUIRES PW
  DEVICES both)

atv_regression(SECTION cuda-pw-spin-stress
  REQUIRES PW
  DEVICES both)
