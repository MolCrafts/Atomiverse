/// @file cpu/scf/rks.cpp
/// @brief RKS constructor defaults, occupation resolution, and nuclear
///        Hessian via central finite difference of analytic RKS forces.
///        The shared SCF driver lives in mean_field_method.cpp.

#include "atomiverse/cpu/scf/rks.h"

#include <stdexcept>
#include <string>
#include <vector>

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/scf/gradient_result.h"

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

vib::HessianResult RKS::hessian(Frame& frame) {
    if (ri != RI::None)
        throw std::runtime_error(
            "RKS::hessian: the analytic Hessian is not wired for density "
            "fitting (RI) — use ri = RI::None (no silent fallback)");

    // Reference SCF + energy (writes eV energy to frame.meta).
    run(frame);
    if (!converged())
        throw std::runtime_error(
            "RKS::hessian: SCF did not converge — no reference "
            "wavefunction to differentiate ("
            + stop_reason() + ")");

    // Central FD of analytic RKS forces (shared skeleton lives on the CRTP
    // base). A fresh RKS per ±h displacement avoids DIIS/MO state bleed. Full
    // CPKS with analytic XC second derivatives is a follow-up (the
    // ALDA/response pieces are exercised by TDDFT; the nuclear XC skeleton
    // second-derivative grid response is the remaining hard cut).
    auto forces_at = [&](Frame& f) -> GradientResult {
        RKS s(xc);
        s.basis = basis;
        s.tol = tol;
        s.max_iter = max_iter;
        s.init_guess = init_guess;
        s.damping_factor = damping_factor;
        s.diis_max_vectors = diis_max_vectors;
        s.diis_start_iter = diis_start_iter;
        s.grid_level = grid_level;
        s.n_radial = n_radial;
        s.n_angular = n_angular;
        s.conv_tol_grad = conv_tol_grad;
        const GradientResult g = s.forces(f);
        if (!s.converged())
            throw std::runtime_error(
                "RKS::hessian: SCF failed on a displaced geometry");
        return g;
    };

    vib::HessianResult out =
        central_difference_nuclear_hessian(frame, forces_at);

    // Geometry is restored to r0 by the template; rebuild the reference
    // wavefunction there.
    run(frame);
    return out;
}

} // namespace scf
} // namespace cpu
} // namespace ATV
