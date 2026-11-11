/// @file cpu/scf/uhf.cpp
/// @brief UHF constructor defaults and occupation resolution. The nuclear
///        Hessian is `Hessian<UHF>` (`pyscf.hessian.uhf.Hessian`). The
///        shared driver loop lives in mean_field_method.cpp.

#include "atomiverse/cpu/scf/uhf.h"

#include <stdexcept>
#include <string>

#include "atomiverse/cpu/scf/rhf.h"
#include "atomiverse/cpu/scf/scf_guess.h"
#include "atomiverse/cpu/scf/slater_determinant.h"

namespace ATV {
namespace cpu {
namespace scf {

UHF::UHF(BasisSet basis_in, int na, int nb, int charge_in, double tol_in,
         int max_iter_in)
    : MeanFieldMethod<UHF>(basis_in, charge_in), n_alpha(na), n_beta(nb) {
    spin_ = SpinType::Unrestricted;
    method_ = MeanFieldType::HF;
    init_guess = InitGuess::Hcore;
    damping_factor = 0.7;
    diis_max_vectors = 6;
    diis_start_iter = 5;
    tol = tol_in;
    max_iter = max_iter_in;
}
void UHF::resolve_occupation(int) {
    n_alpha_ = n_alpha;
    n_beta_ = n_beta;
}

DeltaScfResult UHF::delta_scf(Frame& frame, int from, int to, MomStyle style,
                              const SCFInitialGuess* guess) {
    if (style == MomStyle::Off)
        throw std::invalid_argument(
            "UHF::delta_scf: style must be MomStyle::MOM or MomStyle::IMOM "
            "(Off is plain Aufbau and collapses non-Aufbau seeds)");
    if (n_alpha <= 0 || n_beta < 0)
        throw std::invalid_argument(
            "UHF::delta_scf: set n_alpha / n_beta before calling (got n_alpha="
            + std::to_string(n_alpha) + ", n_beta=" + std::to_string(n_beta)
            + ")");

    using Spin = SlaterDeterminant::Spin;
    const MomStyle saved_mom = mom;
    mom = MomStyle::Off;

    // Closed-shell counts: warm-start UHF from a fresh RHF density so the
    // ground state lands on the RHF solution (PySCF pin path). Open-shell
    // keeps the caller guess / UHF defaults.
    SCFInitialGuess rhf_seed;
    const SCFInitialGuess* gs_guess = guess;
    if (!guess && n_alpha == n_beta) {
        RHF rhf{basis, charge, tol, max_iter};
        rhf.ghost_atoms = ghost_atoms;
        rhf.ecp = ecp;
        rhf.run(frame);
        if (!rhf.converged()) {
            mom = saved_mom;
            throw std::runtime_error(
                "UHF::delta_scf: closed-shell RHF warm-start did not converge ("
                + rhf.stop_reason() + ")");
        }
        const RealMatrix Pa = rhf.determinant().density(Spin::alpha);
        rhf_seed.P_alpha = Pa;
        rhf_seed.P_beta = Pa;
        rhf_seed.has_alpha = true;
        rhf_seed.has_beta = true;
        gs_guess = &rhf_seed;
    }

    run(frame, gs_guess);
    if (!converged()) {
        mom = saved_mom;
        throw std::runtime_error(
            "UHF::delta_scf: ground-state SCF did not converge ("
            + stop_reason() + ")");
    }
    const double e_gs = energy();
    SlaterDeterminant det_ex =
        determinant().with_occupation(Spin::alpha, from, to);

    // Non-Aufbau density seed + frozen (IMOM) / live (MOM) reference.
    SCFInitialGuess seed;
    seed.P_alpha = det_ex.density(Spin::alpha);
    seed.P_beta = det_ex.density(Spin::beta);
    seed.has_alpha = true;
    seed.has_beta = true;
    set_mom_reference(det_ex);
    mom = style;
    run(frame, &seed);
    mom = saved_mom;
    if (!converged())
        throw std::runtime_error(
            "UHF::delta_scf: excited-state MOM/IMOM SCF did not converge ("
            + stop_reason() + ")");

    DeltaScfResult out;
    out.e_ground = e_gs;
    out.e_excited = energy();
    out.delta_e_ev = (out.e_excited - out.e_ground) * units::kHartreeInEv;
    out.converged = true;
    out.iterations = iterations();
    return out;
}

} // namespace scf
} // namespace cpu
} // namespace ATV
