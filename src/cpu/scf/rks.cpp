/// @file cpu/scf/rks.cpp
/// @brief RKS constructor defaults + occupation resolution. The shared driver
///        loop lives in mean_field_method.cpp (explicitly instantiated there).

#include "atomiverse/cpu/scf/rks.h"

namespace ATV {
namespace cpu {
namespace scf {

RKS::RKS() {
    spin_ = SpinType::Restricted;
    method_ = MeanFieldType::KS;
    xc_type_ = XCType::PBE;
    xc = XCType::PBE;
    init_guess = InitGuess::MINAO;
    damping_factor = 0.0;
    grid_level = GridLevel::MatchPySCF;
    diis_max_vectors = 8;
    diis_start_iter = 1;
    tol = 1e-9;
}
RKS::RKS(XCType xc_in) : RKS() {
    xc = xc_in;
    xc_type_ = xc_in;
}
void RKS::resolve_occupation(int n_electrons) {
    xc_type_ = xc;
    n_alpha_ = n_electrons / 2;
    n_beta_ = n_electrons / 2;
}

} // namespace scf
} // namespace cpu
} // namespace ATV
