/// @file cpu/scf/uks.cpp
/// @brief UKS constructor defaults + occupation resolution. The shared driver
///        loop lives in mean_field_method.cpp (explicitly instantiated there).

#include "atomiverse/cpu/scf/uks.h"

namespace ATV {
namespace cpu {
namespace scf {

UKS::UKS() {
    spin_ = SpinType::Unrestricted;
    method_ = MeanFieldType::KS;
    xc_type_ = XCType::LSDA;
    xc = XCType::LSDA;
    init_guess = InitGuess::Hcore;
    damping_factor = 0.5;
    diis_max_vectors = 6;
    diis_start_iter = 1;
    tol = 1e-8;
}
UKS::UKS(int na, int nb, XCType xc_in) : UKS() {
    n_alpha = na;
    n_beta = nb;
    xc = xc_in;
    xc_type_ = xc_in;
}
void UKS::resolve_occupation(int) {
    xc_type_ = xc;
    n_alpha_ = n_alpha;
    n_beta_ = n_beta;
}

} // namespace scf
} // namespace cpu
} // namespace ATV
