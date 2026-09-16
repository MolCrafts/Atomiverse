/// @file cpu/scf/uks.cpp
/// @brief UKS constructor defaults, occupation resolution, and nuclear
///        Hessian via central finite difference of analytic UKS forces.
///        The shared driver loop lives in mean_field_method.cpp (explicitly
///        instantiated there).

#include "atomiverse/cpu/scf/uks.h"

#include <stdexcept>
#include <string>
#include <vector>

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/scf/gradient_result.h"
#include "atomiverse/cpu/scf/slater_determinant.h"

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

vib::HessianResult UKS::hessian(Frame& frame, const SCFInitialGuess* guess) {
    if (ri != RI::None)
        throw std::runtime_error(
            "UKS::hessian: the analytic Hessian is not wired for density "
            "fitting (RI) — use ri = RI::None (no silent fallback)");

    // Reference SCF + energy (writes eV energy to frame.meta).
    run(frame, guess);
    if (!converged())
        throw std::runtime_error(
            "UKS::hessian: SCF did not converge — no reference "
            "wavefunction to differentiate ("
            + stop_reason() + ")");

    // Capture reference spin densities so displaced SCFs stay on the same
    // multi-minima solution (triplet O₂ needs this; cold Hcore can land high).
    using Spin = SlaterDeterminant::Spin;
    SCFInitialGuess ref_guess;
    ref_guess.P_alpha = determinant().density(Spin::alpha);
    ref_guess.P_beta = determinant().density(Spin::beta);
    ref_guess.has_alpha = true;
    ref_guess.has_beta = true;

    // Central FD of analytic UKS forces (shared skeleton lives on the CRTP
    // base). Each displaced UKS is seeded from the reference spin densities
    // (multi-minima), and a fresh UKS per ±h displacement avoids DIIS/MO
    // state bleed. Full UCPKS with analytic XC second derivatives is a
    // follow-up (the ALDA/response pieces are exercised by UKS TDA/TDDFT; the
    // nuclear XC skeleton second-derivative grid response is the hard cut).
    auto forces_at = [&](Frame& f) -> GradientResult {
        UKS s(n_alpha, n_beta, xc);
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
        const GradientResult g = s.forces(f, &ref_guess);
        if (!s.converged())
            throw std::runtime_error(
                "UKS::hessian: SCF failed on a displaced geometry");
        return g;
    };

    vib::HessianResult out =
        central_difference_nuclear_hessian(frame, forces_at);

    // Geometry is restored to r0 by the template; rebuild the reference
    // wavefunction there (with the original guess).
    run(frame, guess);
    return out;
}

} // namespace scf
} // namespace cpu
} // namespace ATV
