/// @file cpu/scf/rpa.cpp
/// @brief Closed-shell direct RPA (dRPA) correlation energy — Furche plasmon
///        formula on the Coulomb-only Casida spectrum (α_hf = 0).

#include "atomiverse/cpu/scf/rpa.h"

#include <cstddef>
#include <stdexcept>

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/scf/dense_matrix.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/internal/orbital_hessian.h"
#include "atomiverse/cpu/scf/rhf.h"
#include "atomiverse/cpu/scf/slater_determinant.h"

namespace ATV {
namespace cpu {
namespace scf {

DirectRPAResult RHF::direct_rpa() const {
    if (!has_result())
        throw std::runtime_error(
            "RHF::direct_rpa: no SCF result — call run(frame) first");

    const SlaterDeterminant& det = determinant();
    if (!det.restricted() || det.generalized())
        throw std::runtime_error(
            "RHF::direct_rpa: closed-shell RHF reference required "
            "(no fallback)");

    const internal::IntegralView pack = integral_view_();
    const xt::xtensor<double, 4> eri_ao =
        (pack.ERI.size() > 0) ? pack.ERI : pack.eri_working();

    using Spin = SlaterDeterminant::Spin;
    const RealMatrix& C = det.coefficients(Spin::alpha);
    const RealVector& eps_raw = det.orbital_energies(Spin::alpha);
    const int n_mo = static_cast<int>(C.shape()[1]);
    if (n_mo != static_cast<int>(eps_raw.size()))
        throw std::runtime_error("RHF::direct_rpa: MO dimension mismatch");

    int n_occ = 0;
    for (int p = 0; p < n_mo; ++p) {
        const double f =
            det.occupations(Spin::alpha)(static_cast<std::size_t>(p));
        if (f == 1.0)
            ++n_occ;
        else if (f != 0.0)
            throw std::invalid_argument(
                "RHF::direct_rpa: fractional occupation — integer "
                "occupations only");
    }
    const int n_vir = n_mo - n_occ;
    if (n_occ < 1 || n_vir < 1)
        throw std::invalid_argument(
            "RHF::direct_rpa: need at least one occupied and one virtual "
            "orbital (no fallback)");

    const internal::OrbitalHessian hess{det, eri_ao};
    if (hess.n_occ() != n_occ || hess.n_virt() != n_vir)
        throw std::runtime_error(
            "RHF::direct_rpa: Hessian occupation mismatch");

    // Shared dRPA Hermitian reduction (α_hf = 0); reads Ω_s and Tr A only.
    const internal::DirectRpaResult rpa =
        hess.direct_rpa(internal::Channel::singlet, /*alpha_hf=*/0.0);

    DirectRPAResult out;
    out.tr_A = rpa.tr_A;
    out.omega_ha.resize(rpa.omega.size());
    for (std::size_t s = 0; s < rpa.omega.size(); ++s)
        out.omega_ha[s] = rpa.omega(s);
    out.e_corr = rpa.e_c;
    out.e_tot = energy() + out.e_corr;
    return out;
}

} // namespace scf
} // namespace cpu
} // namespace ATV
