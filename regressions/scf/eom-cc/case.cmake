# scf/eom-cc: one case, one section per pinned system or quantity.

atv_regression(SECTION eom-ee-ccsd
  REQUIRES HF)

atv_regression(SECTION eom-ip-ccsd
  REQUIRES HF)

atv_regression(SECTION eom-ea-ccsd
  REQUIRES HF)

atv_regression(SECTION eom-sf-ccsd-o2
  REQUIRES HF)

atv_regression(SECTION eom-cc2-h2
  REQUIRES HF)
