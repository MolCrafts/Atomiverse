/// @file cpu/scf/g0w0.cpp
/// @brief Spectral (frequency-exact) closed-shell G0W0 on an RHF reference.
///
/// Screening is direct RPA (dRPA): Casida A/B with α_hf = 0 (Coulomb-only
/// coupling). Transition densities ρ_s,pq = Σ_ia √2 (X+Y)_ia^s (ia|pq) follow
/// the PySCF `gw.gw_exact` convention (ATV RPA norms XᵀX−YᵀY = 1 ⇒ factor √2
/// replaces PySCF's 2× on a ½-normed closed-shell amplitude). Diagonal
/// self-energy Σ_c(ω) is the spectral sum over RPA poles; the QP equation
///   ω = ε_p + Re Σ_c(ω) + v_x,p − v_mf,p
/// is solved by Newton (for pure HF, v_x − v_mf cancels to ~0).

#include "atomiverse/cpu/scf/g0w0.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/scf/dense_matrix.h"
#include "atomiverse/cpu/scf/internal/ao2mo.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/internal/orbital_hessian.h"
#include "atomiverse/cpu/scf/rhf.h"
#include "atomiverse/cpu/scf/slater_determinant.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace {

constexpr double kEta = 1e-8;    ///< Spectral broadening [Hartree].
constexpr double kQpeTol = 1e-8; ///< Newton residual on ω [Hartree].
constexpr int kQpeMaxIter = 100;

/// Diagonal correlation self-energy Re Σ_c,p(ω) [Hartree] at real ω.
double sigma_c_diag(double omega, int p, int n_occ, int n_mo,
                    const RealVector& eps, const RealVector& Om,
                    const xt::xtensor<double, 3>& tdm) {
    // tdm[s, m, p]  (n_exc × n_mo × n_mo stored as [s, first_mo, second_mo]
    // with second index fixed to the orbital p under consideration).
    const int n_exc = static_cast<int>(Om.size());
    double re = 0.0;
    for (int s = 0; s < n_exc; ++s) {
        const double Os = Om(static_cast<std::size_t>(s));
        for (int m = 0; m < n_occ; ++m) {
            const double t =
                tdm(static_cast<std::size_t>(s), static_cast<std::size_t>(m),
                    static_cast<std::size_t>(p));
            // 1/(ω + Ω_s − ε_m − iη) ; take real part
            const double den_r = omega + Os - eps(static_cast<std::size_t>(m));
            const double den2 = den_r * den_r + kEta * kEta;
            re += (t * t) * den_r / den2;
        }
        for (int m = n_occ; m < n_mo; ++m) {
            const double t =
                tdm(static_cast<std::size_t>(s), static_cast<std::size_t>(m),
                    static_cast<std::size_t>(p));
            const double den_r = omega - Os - eps(static_cast<std::size_t>(m));
            const double den2 = den_r * den_r + kEta * kEta;
            re += (t * t) * den_r / den2;
        }
    }
    return re;
}

/// Newton solve of ω − ε_p − (Σ_c(ω) + v_x − v_mf) = 0.
bool solve_qp(double eps_p, double vx_m_vmf, int p, int n_occ, int n_mo,
              const RealVector& eps, const RealVector& Om,
              const xt::xtensor<double, 3>& tdm, double& omega_out) {
    double w = eps_p;
    constexpr double kH = 1e-6;
    for (int it = 0; it < kQpeMaxIter; ++it) {
        const double sig = sigma_c_diag(w, p, n_occ, n_mo, eps, Om, tdm);
        const double f = w - eps_p - (sig + vx_m_vmf);
        if (std::abs(f) < kQpeTol) {
            omega_out = w;
            return true;
        }
        const double sig_p = sigma_c_diag(w + kH, p, n_occ, n_mo, eps, Om, tdm);
        const double dsig = (sig_p - sig) / kH;
        const double df = 1.0 - dsig;
        if (!(std::abs(df) > 1e-14))
            return false;
        w -= f / df;
    }
    omega_out = w;
    return std::abs(
               w - eps_p
               - (sigma_c_diag(w, p, n_occ, n_mo, eps, Om, tdm) + vx_m_vmf))
           < 10.0 * kQpeTol;
}

} // namespace

G0W0Result RHF::g0w0() const {
    if (!has_result())
        throw std::runtime_error(
            "RHF::g0w0: no SCF result — call run(frame) first");

    const SlaterDeterminant& det = determinant();
    if (!det.restricted() || det.generalized())
        throw std::runtime_error(
            "RHF::g0w0: closed-shell RHF reference required (no fallback)");

    const internal::IntegralView pack = integral_view_();
    const xt::xtensor<double, 4> eri_ao =
        (pack.ERI.size() > 0) ? pack.ERI : pack.eri_working();

    using Spin = SlaterDeterminant::Spin;
    const RealMatrix& C = det.coefficients(Spin::alpha);
    const RealVector& eps_raw = det.orbital_energies(Spin::alpha);
    const int n_mo = static_cast<int>(C.shape()[1]);
    if (n_mo != static_cast<int>(eps_raw.size()))
        throw std::runtime_error("RHF::g0w0: MO dimension mismatch");

    // Occupations: integer f=1 columns.
    int n_occ = 0;
    for (int p = 0; p < n_mo; ++p) {
        const double f =
            det.occupations(Spin::alpha)(static_cast<std::size_t>(p));
        if (f == 1.0)
            ++n_occ;
        else if (f != 0.0)
            throw std::invalid_argument(
                "RHF::g0w0: fractional occupation — integer occupations only");
    }
    const int n_vir = n_mo - n_occ;
    if (n_occ < 1 || n_vir < 1)
        throw std::invalid_argument(
            "RHF::g0w0: need at least one occupied and one virtual orbital "
            "(no fallback)");

    // dRPA screening of W (α_hf = 0 Casida matrices).
    const internal::OrbitalHessian hess{det, eri_ao};
    if (hess.n_occ() != n_occ || hess.n_virt() != n_vir)
        throw std::runtime_error("RHF::g0w0: Hessian occupation mismatch");

    // Shared dRPA Hermitian reduction (α_hf = 0); G0W0 consumes Ω_s + X/Y.
    const internal::DirectRpaResult rpa =
        hess.direct_rpa(internal::Channel::singlet, /*alpha_hf=*/0.0);
    const RealVector& Om = rpa.omega;
    const RealMatrix& X = rpa.X;
    const RealMatrix& Y = rpa.Y;
    const int n_exc = static_cast<int>(Om.size());

    // Full MO ERI (pq|rs) for transition densities and exchange.
    const xt::xtensor<double, 4> eri = ao2mo_full(eri_ao, C, n_mo);

    // tdm[s,p,q] = Σ_ia √2 (X+Y)_ia^s (ia|pq)
    // Composite OV index matches OrbitalHessian: a·n_occ + i.
    const double sqrt2 = std::sqrt(2.0);
    xt::xtensor<double, 3> tdm = xt::zeros<double>(
        {static_cast<std::size_t>(n_exc), static_cast<std::size_t>(n_mo),
         static_cast<std::size_t>(n_mo)});
    for (int s = 0; s < n_exc; ++s) {
        for (int a = 0; a < n_vir; ++a) {
            for (int i = 0; i < n_occ; ++i) {
                const int ia = a * n_occ + i;
                const double z = sqrt2
                                 * (X(static_cast<std::size_t>(s),
                                      static_cast<std::size_t>(ia))
                                    + Y(static_cast<std::size_t>(s),
                                        static_cast<std::size_t>(ia)));
                const int aa = n_occ + a;
                for (int p = 0; p < n_mo; ++p)
                    for (int q = 0; q < n_mo; ++q)
                        tdm(static_cast<std::size_t>(s),
                            static_cast<std::size_t>(p),
                            static_cast<std::size_t>(q)) +=
                            z
                            * eri(static_cast<std::size_t>(i),
                                  static_cast<std::size_t>(aa),
                                  static_cast<std::size_t>(p),
                                  static_cast<std::size_t>(q));
            }
        }
    }

    // PySCF QP equation: ω = ε_p + Re Σ_c(ω) + v_x,p − v_mf,p with
    // v_mf = veff − J (exchange/XC only) and v_x the HF exchange self-energy.
    // On a pure HF reference those two cancel analytically (and numerically
    // to ~1e-15), so the correction is the correlation self-energy alone.
    // Keep an explicit diagonal residual for audit / future KS G0W0:
    //   v_x,p = −Σ_i (pi|pi),  v_mf,p = −Σ_i (pi|pi)  ⇒  Δ = 0.
    RealVector eps = eps_raw;
    std::vector<double> vx_m_vmf(static_cast<std::size_t>(n_mo), 0.0);
    for (int p = 0; p < n_mo; ++p) {
        double vx = 0.0;
        for (int i = 0; i < n_occ; ++i)
            vx -= eri(static_cast<std::size_t>(p), static_cast<std::size_t>(i),
                      static_cast<std::size_t>(p), static_cast<std::size_t>(i));
        const double vmf = vx; // HF: exchange piece of veff − J
        vx_m_vmf[static_cast<std::size_t>(p)] = vx - vmf;
    }

    G0W0Result out;
    out.n_occ = n_occ;
    out.mf_energy_ha.resize(static_cast<std::size_t>(n_mo));
    out.qp_energy_ha.resize(static_cast<std::size_t>(n_mo));
    out.converged = true;
    for (int p = 0; p < n_mo; ++p) {
        out.mf_energy_ha[static_cast<std::size_t>(p)] =
            eps(static_cast<std::size_t>(p));
        double qp = eps(static_cast<std::size_t>(p));
        const bool ok = solve_qp(eps(static_cast<std::size_t>(p)),
                                 vx_m_vmf[static_cast<std::size_t>(p)], p,
                                 n_occ, n_mo, eps, Om, tdm, qp);
        if (!ok)
            out.converged = false;
        out.qp_energy_ha[static_cast<std::size_t>(p)] = qp;
    }

    const int homo = n_occ - 1;
    const int lumo = n_occ;
    const double e_h = out.qp_energy_ha[static_cast<std::size_t>(homo)];
    const double e_l = out.qp_energy_ha[static_cast<std::size_t>(lumo)];
    out.ip_ev = -e_h * units::kHartreeInEv;
    out.ea_ev = e_l * units::kHartreeInEv;
    out.gap_ev = (e_l - e_h) * units::kHartreeInEv;
    return out;
}

} // namespace scf
} // namespace cpu
} // namespace ATV
