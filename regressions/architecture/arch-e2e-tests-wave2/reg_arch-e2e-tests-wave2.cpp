/// @file reg_arch-e2e-tests-wave2.cpp
/// @brief Harness twin for e2e-in-tests wave-2 (iron law residual cleanup).
///
/// Public headers only. No SCF solve — physics pins stay in regressions/scf/*
/// and regressions/{pw,gpw}/mpi-*-scaling-4rank. See REFERENCE.md.

#include "atomiverse/cpu/scf/methods.h"

#ifndef ATV_HAS_HF
#error "arch-e2e-tests-wave2 requires ATV_BUILD_HF=ON"
#endif

int main() {
    // Link smoke: public MeanFieldMethod surface still constructible.
    ATV::cpu::scf::RHF rhf;
    (void)rhf.max_iter;
    return 0;
}
