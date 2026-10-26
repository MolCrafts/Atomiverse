/// @file cpu/scf/rks.cpp
/// @brief RKS constructor defaults and occupation resolution. The nuclear
///        Hessian is `Hessian<RKS>` (FD of analytic forces). The shared
///        SCF driver lives in mean_field_method.cpp.

#include "atomiverse/cpu/scf/rks.h"

#include <stdexcept>
#include <string>

namespace ATV {
namespace cpu {
namespace scf {

RKS::RKS(BasisSet basis_in, XCType xc_in, int charge_in)
    : RKS(basis_in, std::make_shared<BuiltinXc>(xc_in), charge_in) {}

RKS::RKS(BasisSet basis_in,
         std::shared_ptr<const ATV::electronic::XcFunctional> functional,
         int charge_in)
    : MeanFieldMethod<RKS>(basis_in, charge_in),
      xc(select_functional_(std::move(functional),
                            ATV::electronic::XcSpin::Restricted)) {
    spin_ = SpinType::Restricted;
    method_ = MeanFieldType::KS;
    xc_type_ = xc;
    init_guess = InitGuess::MINAO;
    damping_factor = 0.0;
    grid_level = GridLevel::MatchPySCF;
    diis_max_vectors = 8;
    diis_start_iter = 1;
    tol = 1e-9;
}
void RKS::resolve_occupation(int n_electrons) {
    xc_type_ = xc;
    n_alpha_ = n_electrons / 2;
    n_beta_ = n_electrons / 2;
}

} // namespace scf
} // namespace cpu
} // namespace ATV
