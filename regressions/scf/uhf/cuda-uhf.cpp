/// @file regressions/scf/cuda-uhf/reg_cuda-uhf.cpp
/// @brief O2 and OH STO-3G UHF energy and analytic gradient vs PySCF 2.14.0.
///
/// CPU consumer. The CUDA build replaces this target with the .cu twin.
/// See REFERENCE.md. Out-of-library: public headers + link atomiverse.

#include "cuda-uhf.h"
#include "atomiverse/cpu/scf/methods.h"
#include "harness/registry.h"

namespace atv_reg_scf_cuda_uhf {

int run() {
    auto run = [](ATV::cpu::Frame& frame, int na, int nb, int charge) {
        ATV::cpu::scf::UHF uhf{
            ATV::cpu::scf::BasisSet::STO3G, na, nb, charge, 1e-12, 100};
        // PySCF default: MINAO, damp = 0, CDIIS from cycle 1. The UHF
        // constructor's Hcore guess + 0.7 damping is the other O2 basin.
        uhf.init_guess = ATV::cpu::scf::InitGuess::MINAO;
        uhf.damping_factor = 0.0;
        uhf.diis_start_iter = 1;
        uhf.diis_max_vectors = 8;
        const auto g = uhf.forces(frame);
        cuda_uhf_case::Out out;
        out.energy_ha = uhf.energy();
        out.converged = uhf.converged();
        out.forces = g.forces;
        return out;
    };
    return cuda_uhf_case::run_all(run);
}

} // namespace atv_reg_scf_cuda_uhf

ATV_REGRESSION("scf/uhf", "cuda-uhf", atv_reg_scf_cuda_uhf::run);
