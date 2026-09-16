/// @file cpu/scf/uhf.cpp
/// @brief UHF constructor defaults, occupation resolution, and analytic
///        nuclear Hessian (unrestricted CPHF). The shared driver loop lives
///        in mean_field_method.cpp (explicitly instantiated there).

#include "atomiverse/cpu/scf/uhf.h"

#include <stdexcept>
#include <string>

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/scf/dense_matrix.h"
#include "atomiverse/cpu/scf/internal/integral_derivatives.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/internal/ucphf_pack.h"
#include "atomiverse/cpu/scf/rhf.h"
#include "atomiverse/cpu/scf/scf_guess.h"
#include "atomiverse/cpu/scf/slater_determinant.h"

namespace ATV {
namespace cpu {
namespace scf {

UHF::UHF() {
    spin_ = SpinType::Unrestricted;
    method_ = MeanFieldType::HF;
    init_guess = InitGuess::Hcore;
    damping_factor = 0.7;
    diis_max_vectors = 6;
    diis_start_iter = 5;
    tol = 1e-8;
}
UHF::UHF(int na, int nb) : UHF() {
    n_alpha = na;
    n_beta = nb;
}
void UHF::resolve_occupation(int) {
    n_alpha_ = n_alpha;
    n_beta_ = n_beta;
}

vib::HessianResult UHF::hessian(Frame& frame, const SCFInitialGuess* guess) {
    if (ri != RI::None)
        throw std::runtime_error(
            "UHF::hessian: the analytic Hessian is not wired for density "
            "fitting (RI) — derivative three-center integrals are a "
            "follow-up chain link; use ri = RI::None (no silent fallback)");

    // Fresh SCF at this geometry (writes the eV energy to frame.meta).
    run(frame, guess);

    const SlaterDeterminant& det = determinant();
    const internal::IntegralView pack = integral_view_();
    // Shared guard: non-convergence, spherical-harmonic pack, ECP pack.
    check_hessian_preconditions("UHF::hessian", pack);

    const xt::xtensor<double, 4> eri =
        (pack.ERI.size() > 0) ? pack.ERI : pack.eri_working();

    using Spin = SlaterDeterminant::Spin;
    const RealMatrix Pa = det.density(Spin::alpha);
    const RealMatrix Pb = det.density(Spin::beta);

    // Skeleton first derivatives: dS plus per-spin derivative Focks
    // F^{σ(x)} = dT + dV + J^{(x)}[P] − K^{(x)}[P^σ].
    const IntegralDerivPack dpack{pack};
    xt::xtensor<double, 4> dFa = dpack.dT + dpack.dV;
    xt::xtensor<double, 4> dFb = dpack.dT + dpack.dV;
    accumulate_two_electron_fock_derivative_uhf(dFa, dFb, pack, Pa, Pb);

    // Coupled α/β CPHF solve (3N right-hand sides) + Hessian assembly.
    internal::UCphfPack cphf{det, pack, eri, dpack.dS, dFa, dFb};
    cphf.solve();
    const xt::xtensor<double, 2> H_ha = cphf.hessian();

    const int natm = static_cast<int>(frame.nlocal());
    const double to_ev_ang2 =
        units::kHartreeInEv / (units::kBohrInAngstrom * units::kBohrInAngstrom);
    vib::HessianResult out;
    out.n_atoms = natm;
    out.hessian = H_ha * to_ev_ang2;
    return out;
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
        RHF rhf;
        rhf.basis = basis;
        rhf.charge = charge;
        rhf.tol = tol;
        rhf.max_iter = max_iter;
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
