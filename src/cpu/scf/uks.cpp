/// @file cpu/scf/uks.cpp
/// @brief UKS constructor defaults and occupation resolution. The nuclear
///        Hessian is `Hessian<UKS>` (FD of analytic forces). The shared
///        driver loop lives in mean_field_method.cpp.

#include "atomiverse/cpu/scf/uks.h"

#include <stdexcept>
#include <string>

namespace ATV {
namespace cpu {
namespace scf {

UKS::UKS(BasisSet basis_in, int na, int nb, XCType xc_in, int charge_in)
    : MeanFieldMethod<UKS>(basis_in, charge_in), n_alpha(na), n_beta(nb),
      xc(xc_in) {
    spin_ = SpinType::Unrestricted;
    method_ = MeanFieldType::KS;
    xc_type_ = xc_in;
    init_guess = InitGuess::Hcore;
    damping_factor = 0.5;
    diis_max_vectors = 6;
    diis_start_iter = 1;
    tol = 1e-8;
}
void UKS::resolve_occupation(int) {
    xc_type_ = xc;
    n_alpha_ = n_alpha;
    n_beta_ = n_beta;
}

} // namespace scf
} // namespace cpu
} // namespace ATV
