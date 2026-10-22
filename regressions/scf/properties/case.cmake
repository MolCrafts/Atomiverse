# scf/properties: one case, one section per pinned system or quantity.

atv_regression(SECTION h2o_rhf_dipole
  REQUIRES HF)

atv_regression(SECTION h2o_charges_pyscf
  REQUIRES HF)

atv_regression(SECTION wannier-centers
  REQUIRES HF)

atv_regression(SECTION static-polariz-h2o
  REQUIRES HF)
