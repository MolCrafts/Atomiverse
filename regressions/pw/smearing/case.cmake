# pw/smearing: one case, one section per pinned system or quantity.

atv_regression(SECTION pw-uks-fe-smearing
  REQUIRES PW)

atv_regression(SECTION cuda-pw-metallic-smear
  REQUIRES PW
  DEVICES both)

atv_regression(SECTION cuda-pw-smear-forces
  REQUIRES PW
  DEVICES both)

atv_regression(SECTION cuda-pw-smear-stress
  REQUIRES PW
  DEVICES both)
