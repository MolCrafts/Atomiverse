# scf/mp2: one case, one section per pinned system or quantity.

atv_regression(SECTION mp2-grad-h2o-sto3g
  REQUIRES HF)

atv_regression(SECTION frozen-core-mp2-h2o
  REQUIRES HF DEVICES both)

atv_regression(SECTION scs-mp2-h2o-sto3g
  REQUIRES HF DEVICES both)

atv_regression(SECTION mp2-water-chain
  REQUIRES HF DEVICES both)

atv_regression(SECTION ump2-o2-sto3g
  REQUIRES HF)

atv_regression(SECTION ump2-grad-o2-sto3g
  REQUIRES HF)

atv_regression(SECTION romp2-o2-sto3g
  REQUIRES HF)

atv_regression(SECTION posthf-eri-block-mp2
  REQUIRES HF)

atv_regression(SECTION mp3-h2-sto3g
  REQUIRES HF)

atv_regression(SECTION mp4-h2-sto3g
  REQUIRES HF)
