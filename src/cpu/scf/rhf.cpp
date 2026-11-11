/// @file cpu/scf/rhf.cpp
/// @brief RHF constructor defaults and occupation resolution. The shared
///        driver loop lives in mean_field_method.cpp. Seeger–Pople stability
///        is `Stability<RHF>` (`response.cpp`), not a method here.

#include "atomiverse/cpu/scf/rhf.h"

namespace ATV {
namespace cpu {
namespace scf {

RHF::RHF(BasisSet basis_in, int charge_in, double tol_in, int max_iter_in)
    : MeanFieldMethod<RHF>(basis_in, charge_in) {
    spin_ = SpinType::Restricted;
    method_ = MeanFieldType::HF;
    init_guess = InitGuess::MINAO;
    damping_factor = 0.0;
    tol = tol_in;
    max_iter = max_iter_in;
}
void RHF::resolve_occupation(int n_electrons) {
    n_alpha_ = n_electrons / 2;
    n_beta_ = n_electrons / 2;
}

} // namespace scf
} // namespace cpu
} // namespace ATV
