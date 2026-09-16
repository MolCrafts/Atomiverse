/// @file cpu/scf/rohf.cpp
/// @brief ROHF constructor defaults, occupation resolution, and nuclear
///        Hessian via central finite difference of analytic ROHF forces.
///        The shared driver loop lives in mean_field_method.cpp; the
///        Roothaan effective-Fock coupling lives in stages/rohf_couple_stage.h.

#include "atomiverse/cpu/scf/rohf.h"

#include <stdexcept>
#include <string>
#include <vector>

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/scf/gradient_result.h"

namespace ATV {
namespace cpu {
namespace scf {

ROHF::ROHF() {
    spin_ = SpinType::RestrictedOpen;
    method_ = MeanFieldType::HF;
    init_guess = InitGuess::Hcore;
    damping_factor = 0.7;
    diis_max_vectors = 6;
    diis_start_iter = 5;
    tol = 1e-8;
}
ROHF::ROHF(int na, int nb) : ROHF() {
    n_alpha = na;
    n_beta = nb;
}
void ROHF::resolve_occupation(int) {
    if (n_alpha < n_beta)
        throw std::invalid_argument(
            "ROHF::run: n_alpha (" + std::to_string(n_alpha) + ") < n_beta ("
            + std::to_string(n_beta)
            + ") — ROHF defines the open shell as excess alpha electrons; "
              "swap the counts (no fallback).");
    n_alpha_ = n_alpha;
    n_beta_ = n_beta;
}

vib::HessianResult ROHF::hessian(Frame& frame) {
    if (ri != RI::None)
        throw std::runtime_error(
            "ROHF::hessian: the analytic Hessian is not wired for density "
            "fitting (RI) — use ri = RI::None (no silent fallback)");

    // Reference SCF + energy (writes eV energy to frame.meta).
    run(frame);
    if (!converged())
        throw std::runtime_error(
            "ROHF::hessian: SCF did not converge — no reference "
            "wavefunction to differentiate ("
            + stop_reason() + ")");

    // Central FD of analytic ROHF forces (shared skeleton lives on the CRTP
    // base). A fresh ROHF per ±h displacement avoids DIIS/MO state bleed.
    // Full three-space ROHF CPHF is a follow-up (PySCF also lacks
    // hessian.ROHF).
    auto forces_at = [&](Frame& f) -> GradientResult {
        ROHF s(n_alpha, n_beta);
        s.basis = basis;
        s.tol = tol;
        s.max_iter = max_iter;
        s.init_guess = init_guess;
        s.damping_factor = damping_factor;
        s.diis_max_vectors = diis_max_vectors;
        s.diis_start_iter = diis_start_iter;
        const GradientResult g = s.forces(f);
        if (!s.converged())
            throw std::runtime_error(
                "ROHF::hessian: SCF failed on a displaced geometry");
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
