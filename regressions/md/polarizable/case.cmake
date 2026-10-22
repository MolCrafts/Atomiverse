# md/polarizable: one case, one section per pinned system or quantity.

atv_regression(SECTION pim-induced-dipole-bulk
  REQUIRES MD)

atv_regression(SECTION pim-npt-polarizable-il
  REQUIRES MD)

atv_regression(SECTION extended-lagrangian-dipoles
  REQUIRES KSPACE MD)

atv_regression(SECTION qeq-fluctuating-charge
  REQUIRES MD)

atv_regression(SECTION maze-molten-salt-al
  REQUIRES KSPACE MD)

atv_regression(SECTION maze-symplectic-charge-dynamics
  REQUIRES MD)
