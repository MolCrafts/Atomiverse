/// @brief LJ dimer minimization against the analytic 12-6 minimum (CPU).
#include "lj_dimer.h"
#include "harness/registry.h"

ATV_REGRESSION("minimizer/geometry", "lj_dimer", lj_dimer_case::run_cpu);
