# md/electrode-observables: one case, one section per pinned system or quantity.

atv_regression(SECTION capacitance-charge-fluctuation
  REQUIRES MD)

atv_regression(SECTION charging-work-energy
  REQUIRES MD)

atv_regression(SECTION thermo-integration-capacitance
  REQUIRES KSPACE MD)

atv_regression(SECTION potential-profile-observer
  REQUIRES MD)
