/// @file regressions/scf/h2_uhf_sto3g/reg_h2_uhf_sto3g.cpp
/// @brief STO-3G UHF energy and analytic gradient vs PySCF 2.14.0.
///
/// CPU consumer. The CUDA build replaces this target with the .cu twin.
/// See REFERENCE.md. Out-of-library: public headers + link atomiverse.

#include "h2_uhf_sto3g.h"
#include "atomiverse/cpu/scf/methods.h"
#include "harness/registry.h"

namespace atv_reg_scf_h2_uhf_sto3g {

int run() {
    auto run = [](ATV::cpu::Frame& frame, int na, int nb, int charge) {
        ATV::cpu::scf::UHF uhf{
            ATV::cpu::scf::BasisSet::STO3G, na, nb, charge, 1e-12, 100};
        const auto g = uhf.forces(frame);
        uhf_case::Out out;
        out.energy_ha = uhf.energy();
        out.converged = uhf.converged();
        out.forces = g.forces;
        return out;
    };
    return uhf_case::run_all(run);
}

} // namespace atv_reg_scf_h2_uhf_sto3g

ATV_REGRESSION("scf/uhf", "h2_uhf_sto3g", atv_reg_scf_h2_uhf_sto3g::run);
