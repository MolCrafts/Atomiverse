/// @file semiempirical/am1/am1-water-mopac-energy.cpp
/// @brief CPU registration of the shared executed-MOPAC regression.
#include "am1-water-mopac-energy.h"

ATV_REGRESSION("semiempirical/am1", "am1-water-mopac-energy",
               atv_reg_semiempirical_am1_water_mopac_energy::run_cpu);
