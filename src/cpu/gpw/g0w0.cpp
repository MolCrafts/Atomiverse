/// @file src/cpu/gpw/g0w0.cpp
/// @brief GpwDft::g0w0 — Γ-point and multi-k (finite-q inter-k W)
///        spectral G0W0@KS on converged RKS orbitals, plus Γ/multi-k
///        UKS UGW. dRPA screening + MT exchange + local v_xc.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/internal/kpoints.h"
#include "atomiverse/cpu/gpw/internal/mp2_kernels.h"
#include "atomiverse/cpu/gpw/internal/xc_grid.h"
#include "atomiverse/cpu/gpw/internal/xc_grid_gga.h"

namespace ATV {
namespace cpu {
namespace gpw {
namespace {

constexpr double kEta = 1e-8;    ///< Spectral broadening [Hartree].
constexpr double kQpeTol = 1e-8; ///< Newton residual on ω [Hartree].
constexpr int kQpeMaxIter = 100;

/// Diagonal Re Σ_c,p(ω) [Hartree]. tdm row-major [s, m, p] with
/// strides (n_mo * n_mo, n_mo, 1).
double sigma_c_diag(double omega, int p, int n_occ, int n_mo,
                    const std::vector<double>& eps,
                    const std::vector<double>& Om,
                    const std::vector<double>& tdm) {
    const int n_exc = static_cast<int>(Om.size());
    const std::size_t nmo = static_cast<std::size_t>(n_mo);
    double re = 0.0;
    for (int s = 0; s < n_exc; ++s) {
        const double Os = Om[static_cast<std::size_t>(s)];
        const std::size_t s0 =
            static_cast<std::size_t>(s) * nmo * nmo
            + static_cast<std::size_t>(p);
        for (int m = 0; m < n_occ; ++m) {
            const double t = tdm[s0 + static_cast<std::size_t>(m) * nmo];
            const double den_r = omega + Os - eps[static_cast<std::size_t>(m)];
            const double den2 = den_r * den_r + kEta * kEta;
            re += (t * t) * den_r / den2;
        }
        for (int m = n_occ; m < n_mo; ++m) {
            const double t = tdm[s0 + static_cast<std::size_t>(m) * nmo];
            const double den_r = omega - Os - eps[static_cast<std::size_t>(m)];
            const double den2 = den_r * den_r + kEta * kEta;
            re += (t * t) * den_r / den2;
        }
    }
    return re;
}

bool solve_qp(double eps_p, double vx_m_vxc, int p, int n_occ, int n_mo,
              const std::vector<double>& eps, const std::vector<double>& Om,
              const std::vector<double>& tdm, double& omega_out) {
    double w = eps_p;
    constexpr double kH = 1e-6;
    for (int it = 0; it < kQpeMaxIter; ++it) {
        const double sig = sigma_c_diag(w, p, n_occ, n_mo, eps, Om, tdm);
        const double f = w - eps_p - (sig + vx_m_vxc);
        if (std::abs(f) < kQpeTol) {
            omega_out = w;
            return true;
        }
        const double sig_p =
            sigma_c_diag(w + kH, p, n_occ, n_mo, eps, Om, tdm);
        const double dsig = (sig_p - sig) / kH;
        const double df = 1.0 - dsig;
        if (!(std::abs(df) > 1e-14))
            return false;
        w -= f / df;
    }
    omega_out = w;
    return std::abs(w - eps_p
                    - (sigma_c_diag(w, p, n_occ, n_mo, eps, Om, tdm)
                       + vx_m_vxc))
           < 10.0 * kQpeTol;
}

XcOnGrid eval_xc_grid(XcKind xc, const UniformGrid& grid,
                      const std::vector<double>& rho) {
    switch (xc) {
    case XcKind::PBE:
        return xc_grid_pbe(grid, rho);
    case XcKind::PBEsol:
        return xc_grid_pbesol(grid, rho);
    case XcKind::RevPBE:
        return xc_grid_revpbe(grid, rho);
    case XcKind::RPBE:
        return xc_grid_rpbe(grid, rho);
    case XcKind::BLYP:
        return xc_grid_blyp(grid, rho);
    case XcKind::OptB88VdW:
        return xc_grid_optb88_vdw(grid, rho);
    case XcKind::PADE:
    default:
        return xc_grid_pade(grid, rho);
    }
}

XcOnGridSpin eval_xc_grid_spin(XcKind xc, const UniformGrid& grid,
                               const std::vector<double>& rho_a,
                               const std::vector<double>& rho_b) {
    switch (xc) {
    case XcKind::PBE:
        return xc_grid_pbe_spin(grid, rho_a, rho_b);
    case XcKind::PBEsol:
        return xc_grid_pbesol_spin(grid, rho_a, rho_b);
    case XcKind::RevPBE:
        return xc_grid_revpbe_spin(grid, rho_a, rho_b);
    case XcKind::BLYP:
        return xc_grid_blyp_spin(grid, rho_a, rho_b);
    case XcKind::OptB88VdW:
        return xc_grid_optb88_vdw_spin(grid, rho_a, rho_b);
    case XcKind::RPBE:
        // Product path: same xc_grid_rpbe_spin as UKS SCF / forces / stress
        // (Hammer–Hansen–Nørskov exponential Fx + PBE correlation).
        return xc_grid_rpbe_spin(grid, rho_a, rho_b);
    case XcKind::PADE:
    default:
        return xc_grid_pade_lsd(grid, rho_a, rho_b);
    }
}

int cap_virt_gw_(int n_mo, int n_occ, int mp2_n_virt,
                 const std::vector<double>& eps, const char* channel) {
    if (n_mo <= n_occ)
        throw std::invalid_argument(
            std::string("GpwDft::g0w0: no virtual orbitals in the ") + channel
            + " channel (n_mo = " + std::to_string(n_mo) + ", n_occ = "
            + std::to_string(n_occ) + ")");
    int n_virt = n_mo - n_occ;
    if (mp2_n_virt > 0)
        n_virt = std::min(n_virt, mp2_n_virt);
    if (n_occ < 1 || n_virt < 1)
        return n_virt;
    constexpr double kMinGapHa = 1e-6;
    const double gap = eps[static_cast<std::size_t>(n_occ)]
                       - eps[static_cast<std::size_t>(n_occ - 1)];
    if (gap < kMinGapHa)
        throw std::invalid_argument(
            std::string("GpwDft::g0w0: vanishing ") + channel
            + " HOMO-LUMO gap (" + std::to_string(gap)
            + " Ha < 1e-6) — dRPA (A−B) is singular");
    return n_virt;
}

} // namespace

const G0W0Result& GpwDft::g0w0(const std::vector<GpwAtom>& atoms) {
    if (!built_)
        throw std::logic_error(
            "GpwDft::g0w0 called before build() — validate the "
            "configuration first (no silent auto-build)");
    if (use_ot)
        throw std::invalid_argument(
            "GpwDft::g0w0: OT stores occupied orbitals only — canonical "
            "virtuals/eigenvalues require the diagonalization path "
            "(use_ot = false)");
    if (!result_.converged)
        throw std::logic_error(
            "GpwDft::g0w0 needs a converged run() first (no correlation "
            "on an unconverged reference)");

    const bool multi_k =
        result_.n_kpts >= 1 && !result_.C_k.empty()
        && static_cast<int>(result_.C_k.size()) == result_.n_kpts;

    if (result_.n_kpts > 0 && !multi_k)
        throw std::invalid_argument(
            "GpwDft::g0w0: multi-k SCF result is missing Bloch C_k "
            "(re-run with a current multi-k path that stores orbitals)");

    // ── Unrestricted UGW (spin-conserving spectral G0W0@UKS) ──────────
    if (is_uks()) {
        // Multi-k UKS: finite-q inter-k spectral UGW on C_k / C_k_beta.
        if (multi_k || result_.n_kpts > 0) {
            if (!multi_k)
                throw std::invalid_argument(
                    "GpwDft::g0w0: multi-k UKS SCF result is missing Bloch "
                    "C_k (re-run with a current multi-k path that stores "
                    "full α/β orbitals)");
            if (nonperiodic)
                throw std::invalid_argument(
                    "GpwDft::g0w0: multi-k UGW requires a periodic cell");
            if (!mp2_ri)
                throw std::invalid_argument(
                    "GpwDft::g0w0: multi-k UGW requires mp2_ri = true "
                    "(Poisson multi-k foothold is not offered)");
            if (static_cast<int>(result_.C_k_beta.size()) != result_.n_kpts
                || static_cast<int>(result_.eps_k_beta.size())
                       != result_.n_kpts)
                throw std::invalid_argument(
                    "GpwDft::g0w0: multi-k UKS result is missing C_k_beta / "
                    "eps_k_beta (full β Bloch MOs required for UGW)");

            const int nk = result_.n_kpts;
            const int nb_atoms = [&] {
                int nb = 0;
                for (const auto& a : atoms)
                    nb += static_cast<int>(a.basis.size());
                return nb;
            }();
            const int nb = static_cast<int>(result_.C_k[0].shape(0));
            if (nb_atoms != nb)
                throw std::invalid_argument(
                    "GpwDft::g0w0: atom basis count ("
                    + std::to_string(nb_atoms)
                    + ") does not match the converged orbitals ("
                    + std::to_string(nb)
                    + ") — same geometry/basis as the last run() required");

            int n_mo_a_min = static_cast<int>(result_.C_k[0].shape(1));
            int n_mo_b_min = static_cast<int>(result_.C_k_beta[0].shape(1));
            for (int ik = 0; ik < nk; ++ik) {
                n_mo_a_min = std::min(
                    n_mo_a_min,
                    static_cast<int>(
                        result_.C_k[static_cast<std::size_t>(ik)].shape(1)));
                n_mo_b_min = std::min(
                    n_mo_b_min,
                    static_cast<int>(
                        result_.C_k_beta[static_cast<std::size_t>(ik)]
                            .shape(1)));
                if (static_cast<int>(
                        result_.eps_k[static_cast<std::size_t>(ik)].size())
                        < n_mo_a_min
                    || static_cast<int>(
                           result_.eps_k_beta[static_cast<std::size_t>(ik)]
                               .size())
                           < n_mo_b_min)
                    throw std::logic_error(
                        "GpwDft::g0w0: multi-k UKS eps_k/C_k shape mismatch");
            }

            const int n_oa = n_alpha;
            const int n_ob = n_beta;
            std::vector<double> eps_a0(result_.eps_k[0].begin(),
                                       result_.eps_k[0].begin() + n_mo_a_min);
            std::vector<double> eps_b0(
                result_.eps_k_beta[0].begin(),
                result_.eps_k_beta[0].begin() + n_mo_b_min);
            const int n_va =
                cap_virt_gw_(n_mo_a_min, n_oa, mp2_n_virt, eps_a0, "α");
            const int n_vb =
                cap_virt_gw_(n_mo_b_min, n_ob, mp2_n_virt, eps_b0, "β");
            if (n_oa < 1 || n_va < 1)
                throw std::invalid_argument(
                    "GpwDft::g0w0: multi-k UGW needs n_α >= 1 and "
                    "n_virt_α >= 1 (primary report is the α channel)");

            auto min_eps_gap =
                [&](const std::vector<std::vector<double>>& ek, int n_occ) {
                    double g = 1e300;
                    for (int ik = 0; ik < nk; ++ik) {
                        const auto& e = ek[static_cast<std::size_t>(ik)];
                        g = std::min(
                            g, e[static_cast<std::size_t>(n_occ)]
                                   - e[static_cast<std::size_t>(n_occ - 1)]);
                    }
                    return g;
                };
            constexpr double kMinGapHa = 1e-6;
            if (min_eps_gap(result_.eps_k, n_oa) < kMinGapHa)
                throw std::invalid_argument(
                    "GpwDft::g0w0: vanishing α HOMO-LUMO gap on the k-mesh");
            if (n_ob >= 1 && n_vb >= 1
                && min_eps_gap(result_.eps_k_beta, n_ob) < kMinGapHa)
                throw std::invalid_argument(
                    "GpwDft::g0w0: vanishing β HOMO-LUMO gap on the k-mesh");

            const UniformGrid grid = make_uniform_grid(cell, cutoff_ry);
            std::vector<GridBasisFunction> basis;
            basis.reserve(static_cast<std::size_t>(nb));
            for (const auto& a : atoms)
                for (const auto& bf : a.basis)
                    basis.push_back(bf);

            std::vector<internal::KPoint> kpts(
                static_cast<std::size_t>(nk));
            for (int ik = 0; ik < nk; ++ik) {
                kpts[static_cast<std::size_t>(ik)].frac[0] =
                    result_.k_frac[static_cast<std::size_t>(ik)][0];
                kpts[static_cast<std::size_t>(ik)].frac[1] =
                    result_.k_frac[static_cast<std::size_t>(ik)][1];
                kpts[static_cast<std::size_t>(ik)].frac[2] =
                    result_.k_frac[static_cast<std::size_t>(ik)][2];
                kpts[static_cast<std::size_t>(ik)].weight =
                    result_.k_weights[static_cast<std::size_t>(ik)];
            }
            const internal::KpointBasisGrid kbg =
                internal::build_kpoint_basis_grid(grid, basis, kpts,
                                                  eps_default);

            std::vector<std::vector<internal::Cplx>> psi_a_k(
                static_cast<std::size_t>(nk));
            std::vector<std::vector<internal::Cplx>> psi_b_k(
                static_cast<std::size_t>(nk));
            std::vector<std::vector<double>> eps_a_full(
                static_cast<std::size_t>(nk));
            std::vector<std::vector<double>> eps_b_full(
                static_cast<std::size_t>(nk));
            for (int ik = 0; ik < nk; ++ik) {
                psi_a_k[static_cast<std::size_t>(ik)] =
                    internal::mo_on_grid_k(
                        kbg, ik, result_.C_k[static_cast<std::size_t>(ik)], 0,
                        n_mo_a_min);
                psi_b_k[static_cast<std::size_t>(ik)] =
                    internal::mo_on_grid_k(
                        kbg, ik,
                        result_.C_k_beta[static_cast<std::size_t>(ik)], 0,
                        n_mo_b_min);
                const auto& ea = result_.eps_k[static_cast<std::size_t>(ik)];
                const auto& eb =
                    result_.eps_k_beta[static_cast<std::size_t>(ik)];
                eps_a_full[static_cast<std::size_t>(ik)].assign(
                    ea.begin(), ea.begin() + n_mo_a_min);
                eps_b_full[static_cast<std::size_t>(ik)].assign(
                    eb.begin(), eb.begin() + n_mo_b_min);
            }

            // BZ-weighted UKS spin densities for local v_xc^σ.
            const long npts = grid.npts();
            std::vector<double> rho_a(static_cast<std::size_t>(npts), 0.0);
            std::vector<double> rho_b(static_cast<std::size_t>(npts), 0.0);
            for (int ik = 0; ik < nk; ++ik) {
                const double wk =
                    result_.k_weights[static_cast<std::size_t>(ik)];
                const auto& pa = psi_a_k[static_cast<std::size_t>(ik)];
                const auto& pb = psi_b_k[static_cast<std::size_t>(ik)];
                for (int i = 0; i < n_oa; ++i) {
                    const internal::Cplx* pi =
                        pa.data()
                        + static_cast<std::size_t>(i)
                              * static_cast<std::size_t>(npts);
                    for (long g = 0; g < npts; ++g)
                        rho_a[static_cast<std::size_t>(g)] +=
                            wk * std::norm(pi[g]);
                }
                for (int i = 0; i < n_ob; ++i) {
                    const internal::Cplx* pi =
                        pb.data()
                        + static_cast<std::size_t>(i)
                              * static_cast<std::size_t>(npts);
                    for (long g = 0; g < npts; ++g)
                        rho_b[static_cast<std::size_t>(g)] +=
                            wk * std::norm(pi[g]);
                }
            }
            const XcOnGridSpin xc_spin =
                eval_xc_grid_spin(this->xc, grid, rho_a, rho_b);

            std::vector<std::vector<double>> qp_k, sx_k, vxc_k;
            bool ok = false;
            internal::spectral_g0w0_uhf_kpoints(
                grid, psi_a_k, psi_b_k, eps_a_full, eps_b_full,
                result_.k_frac, result_.k_weights, n_oa, n_va, n_ob, n_vb,
                xc_spin.vxc_a, qp_k, sx_k, vxc_k, ok);

            const int homo = n_oa - 1;
            const int lumo = n_oa;
            double min_qp_gap = 1e300;
            double min_mf_gap = 1e300;
            int k_report = 0;
            for (int ik = 0; ik < nk; ++ik) {
                const double qp_g =
                    qp_k[static_cast<std::size_t>(ik)]
                        [static_cast<std::size_t>(lumo)]
                    - qp_k[static_cast<std::size_t>(ik)]
                          [static_cast<std::size_t>(homo)];
                const double mf_g =
                    eps_a_full[static_cast<std::size_t>(ik)]
                              [static_cast<std::size_t>(lumo)]
                    - eps_a_full[static_cast<std::size_t>(ik)]
                                [static_cast<std::size_t>(homo)];
                if (qp_g < min_qp_gap) {
                    min_qp_gap = qp_g;
                    k_report = ik;
                }
                min_mf_gap = std::min(min_mf_gap, mf_g);
            }

            g0w0_result_ = G0W0Result{};
            g0w0_result_.n_occ = n_oa;
            g0w0_result_.n_virt = n_va + n_vb;
            g0w0_result_.ri = true;
            g0w0_result_.n_kpts = nk;
            g0w0_result_.converged = ok;
            g0w0_result_.qp_energy_ha =
                qp_k[static_cast<std::size_t>(k_report)];
            g0w0_result_.mf_energy_ha =
                eps_a_full[static_cast<std::size_t>(k_report)];
            g0w0_result_.sigma_x_ha = sx_k[static_cast<std::size_t>(k_report)];
            g0w0_result_.vxc_ha = vxc_k[static_cast<std::size_t>(k_report)];

            const double e_h =
                g0w0_result_.qp_energy_ha[static_cast<std::size_t>(homo)];
            const double e_l =
                g0w0_result_.qp_energy_ha[static_cast<std::size_t>(lumo)];
            g0w0_result_.ip_ev = -e_h * units::kHartreeInEv;
            g0w0_result_.ea_ev = e_l * units::kHartreeInEv;
            g0w0_result_.gap_ev = min_qp_gap * units::kHartreeInEv;
            g0w0_result_.mf_gap_ev = min_mf_gap * units::kHartreeInEv;
            return g0w0_result_;
        }

        // Γ UKS UGW
        if (result_.C_beta.size() == 0 || result_.eps_beta.empty())
            throw std::logic_error(
                "GpwDft::g0w0: UKS result is missing β orbitals/eigenvalues");

        const int nb_atoms = [&] {
            int nb = 0;
            for (const auto& a : atoms)
                nb += static_cast<int>(a.basis.size());
            return nb;
        }();
        const int nb = static_cast<int>(result_.C.shape(0));
        const int n_mo_a = static_cast<int>(result_.C.shape(1));
        const int n_mo_b = static_cast<int>(result_.C_beta.shape(1));
        if (nb_atoms != nb)
            throw std::invalid_argument(
                "GpwDft::g0w0: atom basis count (" + std::to_string(nb_atoms)
                + ") does not match the converged orbitals ("
                + std::to_string(nb)
                + ") — same geometry/basis as the last run() required");
        if (static_cast<int>(result_.eps.size()) != n_mo_a
            || static_cast<int>(result_.eps_beta.size()) != n_mo_b)
            throw std::logic_error(
                "GpwDft::g0w0: UKS eps/C shape mismatch in the converged "
                "result");
        if (static_cast<int>(result_.C_beta.shape(0)) != nb)
            throw std::logic_error(
                "GpwDft::g0w0: UKS α/β basis dimension mismatch");

        const int n_oa = n_alpha;
        const int n_ob = n_beta;
        const int n_va =
            cap_virt_gw_(n_mo_a, n_oa, mp2_n_virt, result_.eps, "α");
        const int n_vb =
            cap_virt_gw_(n_mo_b, n_ob, mp2_n_virt, result_.eps_beta, "β");
        if (n_oa < 1 || n_va < 1)
            throw std::invalid_argument(
                "GpwDft::g0w0: UGW needs n_α >= 1 and n_virt_α >= 1 "
                "(primary report is the α channel)");

        const bool periodic = !nonperiodic;
        const UniformGrid grid = make_uniform_grid(cell, cutoff_ry);
        std::vector<GridBasisFunction> basis;
        basis.reserve(static_cast<std::size_t>(nb));
        for (const auto& a : atoms)
            for (const auto& bf : a.basis)
                basis.push_back(bf);
        const CompactBasisGrid cbg =
            build_compact_basis_grid(grid, basis, periodic, eps_default);

        const std::vector<double> psi_all_a =
            internal::mo_on_grid(cbg, result_.C, 0, n_mo_a);
        const std::vector<double> psi_all_b =
            internal::mo_on_grid(cbg, result_.C_beta, 0, n_mo_b);
        const long npts = grid.npts();
        const double gw = grid.weight();

        std::vector<double> psi_oa(
            static_cast<std::size_t>(n_oa) * static_cast<std::size_t>(npts));
        std::vector<double> psi_va(
            static_cast<std::size_t>(n_va) * static_cast<std::size_t>(npts));
        std::vector<double> psi_ob(
            static_cast<std::size_t>(n_ob) * static_cast<std::size_t>(npts));
        std::vector<double> psi_vb(
            static_cast<std::size_t>(n_vb) * static_cast<std::size_t>(npts));
        for (int i = 0; i < n_oa; ++i)
            for (long g = 0; g < npts; ++g)
                psi_oa[static_cast<std::size_t>(i)
                           * static_cast<std::size_t>(npts)
                       + static_cast<std::size_t>(g)] =
                    psi_all_a[static_cast<std::size_t>(i)
                                  * static_cast<std::size_t>(npts)
                              + static_cast<std::size_t>(g)];
        for (int a = 0; a < n_va; ++a)
            for (long g = 0; g < npts; ++g)
                psi_va[static_cast<std::size_t>(a)
                           * static_cast<std::size_t>(npts)
                       + static_cast<std::size_t>(g)] =
                    psi_all_a[static_cast<std::size_t>(n_oa + a)
                                  * static_cast<std::size_t>(npts)
                              + static_cast<std::size_t>(g)];
        for (int i = 0; i < n_ob; ++i)
            for (long g = 0; g < npts; ++g)
                psi_ob[static_cast<std::size_t>(i)
                           * static_cast<std::size_t>(npts)
                       + static_cast<std::size_t>(g)] =
                    psi_all_b[static_cast<std::size_t>(i)
                                  * static_cast<std::size_t>(npts)
                              + static_cast<std::size_t>(g)];
        for (int a = 0; a < n_vb; ++a)
            for (long g = 0; g < npts; ++g)
                psi_vb[static_cast<std::size_t>(a)
                           * static_cast<std::size_t>(npts)
                       + static_cast<std::size_t>(g)] =
                    psi_all_b[static_cast<std::size_t>(n_ob + a)
                                  * static_cast<std::size_t>(npts)
                              + static_cast<std::size_t>(g)];

        const bool use_ri = mp2_ri && periodic;
        std::vector<double> v_aa, v_bb, v_ab;
        if (n_oa > 0 && n_va > 0)
            v_aa = use_ri ? internal::mo_eri_ovov_ri(grid, psi_oa, n_oa,
                                                      psi_va, n_va)
                          : internal::mo_eri_ovov(grid, psi_oa, n_oa, psi_va,
                                                   n_va, nonperiodic);
        if (n_ob > 0 && n_vb > 0)
            v_bb = use_ri ? internal::mo_eri_ovov_ri(grid, psi_ob, n_ob,
                                                      psi_vb, n_vb)
                          : internal::mo_eri_ovov(grid, psi_ob, n_ob, psi_vb,
                                                   n_vb, nonperiodic);
        if (n_oa > 0 && n_va > 0 && n_ob > 0 && n_vb > 0)
            v_ab =
                use_ri
                    ? internal::mo_eri_ovov_ab_ri(grid, psi_oa, n_oa, psi_va,
                                                   n_va, psi_ob, n_ob, psi_vb,
                                                   n_vb)
                    : internal::mo_eri_ovov_ab(grid, psi_oa, n_oa, psi_va,
                                                n_va, psi_ob, n_ob, psi_vb,
                                                n_vb, nonperiodic);

        const std::vector<double> eps_oa(result_.eps.begin(),
                                         result_.eps.begin() + n_oa);
        const std::vector<double> eps_va(result_.eps.begin() + n_oa,
                                         result_.eps.begin() + n_oa + n_va);
        const std::vector<double> eps_ob(result_.eps_beta.begin(),
                                         result_.eps_beta.begin() + n_ob);
        const std::vector<double> eps_vb(
            result_.eps_beta.begin() + n_ob,
            result_.eps_beta.begin() + n_ob + n_vb);

        internal::DirectRpaAmplitudes rpa;
        internal::direct_rpa_uhf(v_aa, v_bb, v_ab, eps_oa, eps_va, eps_ob,
                                 eps_vb, rpa);
        const int n_exc = static_cast<int>(rpa.omega.size());
        const int nov_a = n_oa * n_va;
        const int nov_b = n_ob * n_vb;
        const int nov = nov_a + nov_b;

        // V_iaσ(r) for each spin channel (periodic Coulomb; ρ_ia neutral).
        const std::vector<double> V_ov_a =
            (n_oa > 0 && n_va > 0)
                ? internal::mo_pair_potentials_ov(grid, psi_oa, n_oa, psi_va,
                                                   n_va)
                : std::vector<double>{};
        const std::vector<double> V_ov_b =
            (n_ob > 0 && n_vb > 0)
                ? internal::mo_pair_potentials_ov(grid, psi_ob, n_ob, psi_vb,
                                                   n_vb)
                : std::vector<double>{};

        // W_s(g) = Σ_iaτ (X+Y)_iaτ^s V_iaτ(g)  (no √2 — spins explicit).
        // tdm_α[s,m,p] = w Σ_g W_s ψ_m^α ψ_p^α  (same-spin G_α W).
        std::vector<double> tdm_a(static_cast<std::size_t>(n_exc)
                                      * static_cast<std::size_t>(n_mo_a)
                                      * static_cast<std::size_t>(n_mo_a),
                                  0.0);
        std::vector<double> Ws(static_cast<std::size_t>(npts));
        for (int s = 0; s < n_exc; ++s) {
            std::fill(Ws.begin(), Ws.end(), 0.0);
            for (int ia = 0; ia < nov_a; ++ia) {
                const double z =
                    rpa.X[static_cast<std::size_t>(s)
                               * static_cast<std::size_t>(nov)
                           + static_cast<std::size_t>(ia)]
                    + rpa.Y[static_cast<std::size_t>(s)
                                 * static_cast<std::size_t>(nov)
                             + static_cast<std::size_t>(ia)];
                if (z == 0.0)
                    continue;
                const double* Via =
                    V_ov_a.data()
                    + static_cast<std::size_t>(ia)
                          * static_cast<std::size_t>(npts);
                for (long g = 0; g < npts; ++g)
                    Ws[static_cast<std::size_t>(g)] += z * Via[g];
            }
            for (int ia = 0; ia < nov_b; ++ia) {
                const double z =
                    rpa.X[static_cast<std::size_t>(s)
                               * static_cast<std::size_t>(nov)
                           + static_cast<std::size_t>(nov_a + ia)]
                    + rpa.Y[static_cast<std::size_t>(s)
                                 * static_cast<std::size_t>(nov)
                             + static_cast<std::size_t>(nov_a + ia)];
                if (z == 0.0)
                    continue;
                const double* Via =
                    V_ov_b.data()
                    + static_cast<std::size_t>(ia)
                          * static_cast<std::size_t>(npts);
                for (long g = 0; g < npts; ++g)
                    Ws[static_cast<std::size_t>(g)] += z * Via[g];
            }
            const std::size_t s0 = static_cast<std::size_t>(s)
                                   * static_cast<std::size_t>(n_mo_a)
                                   * static_cast<std::size_t>(n_mo_a);
            for (int m = 0; m < n_mo_a; ++m) {
                const double* pm =
                    psi_all_a.data()
                    + static_cast<std::size_t>(m)
                          * static_cast<std::size_t>(npts);
                for (int p = 0; p < n_mo_a; ++p) {
                    const double* pp =
                        psi_all_a.data()
                        + static_cast<std::size_t>(p)
                              * static_cast<std::size_t>(npts);
                    double acc = 0.0;
                    for (long g = 0; g < npts; ++g)
                        acc += Ws[static_cast<std::size_t>(g)] * pm[g]
                               * pp[g];
                    tdm_a[s0 + static_cast<std::size_t>(m)
                                   * static_cast<std::size_t>(n_mo_a)
                               + static_cast<std::size_t>(p)] = gw * acc;
                }
            }
        }

        // Σ_x^α (MT) and v_xc^α diagonals from spin densities.
        const std::vector<double> sigma_x_a =
            internal::exchange_diag_mt(grid, psi_all_a, n_mo_a, n_oa);

        std::vector<double> rho_a(static_cast<std::size_t>(npts), 0.0);
        std::vector<double> rho_b(static_cast<std::size_t>(npts), 0.0);
        for (int i = 0; i < n_oa; ++i) {
            const double* pi =
                psi_all_a.data()
                + static_cast<std::size_t>(i) * static_cast<std::size_t>(npts);
            for (long g = 0; g < npts; ++g)
                rho_a[static_cast<std::size_t>(g)] += pi[g] * pi[g];
        }
        for (int i = 0; i < n_ob; ++i) {
            const double* pi =
                psi_all_b.data()
                + static_cast<std::size_t>(i) * static_cast<std::size_t>(npts);
            for (long g = 0; g < npts; ++g)
                rho_b[static_cast<std::size_t>(g)] += pi[g] * pi[g];
        }
        const XcOnGridSpin xc_spin =
            eval_xc_grid_spin(this->xc, grid, rho_a, rho_b);

        std::vector<double> vxc_diag_a(static_cast<std::size_t>(n_mo_a), 0.0);
        for (int p = 0; p < n_mo_a; ++p) {
            const double* pp =
                psi_all_a.data()
                + static_cast<std::size_t>(p) * static_cast<std::size_t>(npts);
            double acc = 0.0;
            for (long g = 0; g < npts; ++g)
                acc += pp[g] * pp[g]
                       * xc_spin.vxc_a[static_cast<std::size_t>(g)];
            vxc_diag_a[static_cast<std::size_t>(p)] = gw * acc;
        }

        std::vector<double> eps_a = result_.eps;

        g0w0_result_ = G0W0Result{};
        g0w0_result_.n_occ = n_oa;
        g0w0_result_.n_virt = n_va + n_vb;
        g0w0_result_.ri = use_ri;
        g0w0_result_.n_kpts = 0;
        g0w0_result_.mf_energy_ha = eps_a;
        g0w0_result_.qp_energy_ha.resize(static_cast<std::size_t>(n_mo_a));
        g0w0_result_.sigma_x_ha = sigma_x_a;
        g0w0_result_.vxc_ha = vxc_diag_a;
        g0w0_result_.converged = true;

        for (int p = 0; p < n_mo_a; ++p) {
            const double vx_m_vxc = sigma_x_a[static_cast<std::size_t>(p)]
                                    - vxc_diag_a[static_cast<std::size_t>(p)];
            double qp = eps_a[static_cast<std::size_t>(p)];
            const bool ok =
                solve_qp(eps_a[static_cast<std::size_t>(p)], vx_m_vxc, p, n_oa,
                         n_mo_a, eps_a, rpa.omega, tdm_a, qp);
            if (!ok)
                g0w0_result_.converged = false;
            g0w0_result_.qp_energy_ha[static_cast<std::size_t>(p)] = qp;
        }

        const int homo = n_oa - 1;
        const int lumo = n_oa;
        const double e_h =
            g0w0_result_.qp_energy_ha[static_cast<std::size_t>(homo)];
        const double e_l =
            g0w0_result_.qp_energy_ha[static_cast<std::size_t>(lumo)];
        const double mf_h = eps_a[static_cast<std::size_t>(homo)];
        const double mf_l = eps_a[static_cast<std::size_t>(lumo)];
        g0w0_result_.ip_ev = -e_h * units::kHartreeInEv;
        g0w0_result_.ea_ev = e_l * units::kHartreeInEv;
        g0w0_result_.gap_ev = (e_l - e_h) * units::kHartreeInEv;
        g0w0_result_.mf_gap_ev = (mf_l - mf_h) * units::kHartreeInEv;
        return g0w0_result_;
    }

    // ── Multi-k (complex Bloch, finite-q inter-k spectral G0W0) ───────
    if (multi_k) {
        if (nonperiodic)
            throw std::invalid_argument(
                "GpwDft::g0w0: multi-k G0W0 requires a periodic cell");
        if (!mp2_ri)
            throw std::invalid_argument(
                "GpwDft::g0w0: multi-k path requires mp2_ri = true "
                "(Poisson multi-k foothold is not offered)");

        const int nk = result_.n_kpts;
        const int nb_atoms = [&] {
            int nb = 0;
            for (const auto& a : atoms)
                nb += static_cast<int>(a.basis.size());
            return nb;
        }();
        const int nb = static_cast<int>(result_.C_k[0].shape(0));
        if (nb_atoms != nb)
            throw std::invalid_argument(
                "GpwDft::g0w0: atom basis count (" + std::to_string(nb_atoms)
                + ") does not match the converged orbitals ("
                + std::to_string(nb)
                + ") — same geometry/basis as the last run() required");

        int n_mo_min = static_cast<int>(result_.C_k[0].shape(1));
        for (int ik = 0; ik < nk; ++ik) {
            n_mo_min = std::min(
                n_mo_min,
                static_cast<int>(
                    result_.C_k[static_cast<std::size_t>(ik)].shape(1)));
            if (static_cast<int>(
                    result_.eps_k[static_cast<std::size_t>(ik)].size())
                < n_mo_min)
                throw std::logic_error(
                    "GpwDft::g0w0: eps_k/C_k shape mismatch at a k-point");
        }
        if (n_mo_min <= n_occ)
            throw std::invalid_argument(
                "GpwDft::g0w0: no virtual orbitals in the multi-k "
                "result (n_mo_min = "
                + std::to_string(n_mo_min) + ", n_occ = "
                + std::to_string(n_occ) + ")");

        int n_virt = n_mo_min - n_occ;
        if (mp2_n_virt > 0)
            n_virt = std::min(n_virt, mp2_n_virt);

        constexpr double kMinGapHa = 1e-6;
        double min_gap = 1e300;
        for (int ik = 0; ik < nk; ++ik) {
            const auto& eps = result_.eps_k[static_cast<std::size_t>(ik)];
            const double gap = eps[static_cast<std::size_t>(n_occ)]
                               - eps[static_cast<std::size_t>(n_occ - 1)];
            min_gap = std::min(min_gap, gap);
        }
        if (min_gap < kMinGapHa)
            throw std::invalid_argument(
                "GpwDft::g0w0: vanishing HOMO-LUMO gap on the k-mesh ("
                + std::to_string(min_gap)
                + " Ha < 1e-6) — dRPA (A−B) is singular");

        const UniformGrid grid = make_uniform_grid(cell, cutoff_ry);
        std::vector<GridBasisFunction> basis;
        basis.reserve(static_cast<std::size_t>(nb));
        for (const auto& a : atoms)
            for (const auto& bf : a.basis)
                basis.push_back(bf);

        std::vector<internal::KPoint> kpts(static_cast<std::size_t>(nk));
        for (int ik = 0; ik < nk; ++ik) {
            kpts[static_cast<std::size_t>(ik)].frac[0] =
                result_.k_frac[static_cast<std::size_t>(ik)][0];
            kpts[static_cast<std::size_t>(ik)].frac[1] =
                result_.k_frac[static_cast<std::size_t>(ik)][1];
            kpts[static_cast<std::size_t>(ik)].frac[2] =
                result_.k_frac[static_cast<std::size_t>(ik)][2];
            kpts[static_cast<std::size_t>(ik)].weight =
                result_.k_weights[static_cast<std::size_t>(ik)];
        }
        const internal::KpointBasisGrid kbg =
            internal::build_kpoint_basis_grid(grid, basis, kpts, eps_default);

        // Full MO grids per k (screening virtuals are a prefix of the
        // virtual block; QP covers all n_mo_min columns).
        std::vector<std::vector<internal::Cplx>> psi_mo_k(
            static_cast<std::size_t>(nk));
        std::vector<std::vector<double>> eps_k_full(
            static_cast<std::size_t>(nk));
        for (int ik = 0; ik < nk; ++ik) {
            psi_mo_k[static_cast<std::size_t>(ik)] = internal::mo_on_grid_k(
                kbg, ik, result_.C_k[static_cast<std::size_t>(ik)], 0,
                n_mo_min);
            const auto& eps = result_.eps_k[static_cast<std::size_t>(ik)];
            eps_k_full[static_cast<std::size_t>(ik)].assign(
                eps.begin(), eps.begin() + n_mo_min);
        }

        // BZ-weighted closed-shell density for local v_xc.
        const long npts = grid.npts();
        std::vector<double> rho(static_cast<std::size_t>(npts), 0.0);
        for (int ik = 0; ik < nk; ++ik) {
            const double wk = result_.k_weights[static_cast<std::size_t>(ik)];
            const auto& psi = psi_mo_k[static_cast<std::size_t>(ik)];
            for (int i = 0; i < n_occ; ++i) {
                const internal::Cplx* pi =
                    psi.data()
                    + static_cast<std::size_t>(i)
                          * static_cast<std::size_t>(npts);
                for (long g = 0; g < npts; ++g)
                    rho[static_cast<std::size_t>(g)] +=
                        wk * 2.0 * std::norm(pi[g]);
            }
        }
        const XcOnGrid xc_grid = eval_xc_grid(this->xc, grid, rho);

        std::vector<std::vector<double>> qp_k, sx_k, vxc_k;
        bool ok = false;
        internal::spectral_g0w0_rhf_kpoints(
            grid, psi_mo_k, eps_k_full, result_.k_frac, result_.k_weights,
            n_occ, n_virt, xc_grid.vxc_r, qp_k, sx_k, vxc_k, ok);

        // Min direct gaps over k; report spectrum at the k of min QP gap.
        const int homo = n_occ - 1;
        const int lumo = n_occ;
        double min_qp_gap = 1e300;
        double min_mf_gap = 1e300;
        int k_report = 0;
        for (int ik = 0; ik < nk; ++ik) {
            const double qp_g =
                qp_k[static_cast<std::size_t>(ik)]
                    [static_cast<std::size_t>(lumo)]
                - qp_k[static_cast<std::size_t>(ik)]
                      [static_cast<std::size_t>(homo)];
            const double mf_g =
                eps_k_full[static_cast<std::size_t>(ik)]
                          [static_cast<std::size_t>(lumo)]
                - eps_k_full[static_cast<std::size_t>(ik)]
                            [static_cast<std::size_t>(homo)];
            if (qp_g < min_qp_gap) {
                min_qp_gap = qp_g;
                k_report = ik;
            }
            min_mf_gap = std::min(min_mf_gap, mf_g);
        }

        g0w0_result_ = G0W0Result{};
        g0w0_result_.n_occ = n_occ;
        g0w0_result_.n_virt = n_virt;
        g0w0_result_.ri = true;
        g0w0_result_.n_kpts = nk;
        g0w0_result_.converged = ok;
        g0w0_result_.qp_energy_ha = qp_k[static_cast<std::size_t>(k_report)];
        g0w0_result_.mf_energy_ha =
            eps_k_full[static_cast<std::size_t>(k_report)];
        g0w0_result_.sigma_x_ha = sx_k[static_cast<std::size_t>(k_report)];
        g0w0_result_.vxc_ha = vxc_k[static_cast<std::size_t>(k_report)];

        const double e_h =
            g0w0_result_.qp_energy_ha[static_cast<std::size_t>(homo)];
        const double e_l =
            g0w0_result_.qp_energy_ha[static_cast<std::size_t>(lumo)];
        g0w0_result_.ip_ev = -e_h * units::kHartreeInEv;
        g0w0_result_.ea_ev = e_l * units::kHartreeInEv;
        g0w0_result_.gap_ev = min_qp_gap * units::kHartreeInEv;
        g0w0_result_.mf_gap_ev = min_mf_gap * units::kHartreeInEv;
        return g0w0_result_;
    }

    // ── Γ-point real path ──────────────────────────────────────────────
    const int nb_atoms = [&] {
        int nb = 0;
        for (const auto& a : atoms)
            nb += static_cast<int>(a.basis.size());
        return nb;
    }();
    const int nb = static_cast<int>(result_.C.shape(0));
    const int n_mo = static_cast<int>(result_.C.shape(1));
    if (nb_atoms != nb)
        throw std::invalid_argument(
            "GpwDft::g0w0: atom basis count (" + std::to_string(nb_atoms)
            + ") does not match the converged orbitals (" + std::to_string(nb)
            + ") — same geometry/basis as the last run() required");
    if (static_cast<int>(result_.eps.size()) != n_mo)
        throw std::logic_error(
            "GpwDft::g0w0: eps/C shape mismatch in the converged result");
    if (n_mo <= n_occ)
        throw std::invalid_argument(
            "GpwDft::g0w0: no virtual orbitals in the converged result "
            "(n_mo = "
            + std::to_string(n_mo) + ", n_occ = " + std::to_string(n_occ)
            + ")");

    int n_virt = n_mo - n_occ;
    if (mp2_n_virt > 0)
        n_virt = std::min(n_virt, mp2_n_virt);

    constexpr double kMinGapHa = 1e-6;
    const double gap = result_.eps[static_cast<std::size_t>(n_occ)]
                       - result_.eps[static_cast<std::size_t>(n_occ - 1)];
    if (gap < kMinGapHa)
        throw std::invalid_argument(
            "GpwDft::g0w0: vanishing HOMO-LUMO gap (" + std::to_string(gap)
            + " Ha < 1e-6) — dRPA (A−B) is singular");

    const bool periodic = !nonperiodic;
    const UniformGrid grid = make_uniform_grid(cell, cutoff_ry);
    std::vector<GridBasisFunction> basis;
    basis.reserve(static_cast<std::size_t>(nb));
    for (const auto& a : atoms)
        for (const auto& bf : a.basis)
            basis.push_back(bf);
    const CompactBasisGrid cbg =
        build_compact_basis_grid(grid, basis, periodic, eps_default);

    // All MOs on the density grid (screening virtuals are a prefix of the
    // virtual block; QP levels cover the full spectrum).
    const std::vector<double> psi_all =
        internal::mo_on_grid(cbg, result_.C, 0, n_mo);
    const long npts = grid.npts();
    const double gw = grid.weight();

    std::vector<double> psi_occ(
        static_cast<std::size_t>(n_occ) * static_cast<std::size_t>(npts));
    std::vector<double> psi_virt(
        static_cast<std::size_t>(n_virt) * static_cast<std::size_t>(npts));
    for (int i = 0; i < n_occ; ++i)
        for (long g = 0; g < npts; ++g)
            psi_occ[static_cast<std::size_t>(i)
                        * static_cast<std::size_t>(npts)
                    + static_cast<std::size_t>(g)] =
                psi_all[static_cast<std::size_t>(i)
                            * static_cast<std::size_t>(npts)
                        + static_cast<std::size_t>(g)];
    for (int a = 0; a < n_virt; ++a)
        for (long g = 0; g < npts; ++g)
            psi_virt[static_cast<std::size_t>(a)
                         * static_cast<std::size_t>(npts)
                     + static_cast<std::size_t>(g)] =
                psi_all[static_cast<std::size_t>(n_occ + a)
                            * static_cast<std::size_t>(npts)
                        + static_cast<std::size_t>(g)];

    const bool use_ri = mp2_ri && periodic;
    const std::vector<double> v_ovov =
        use_ri ? internal::mo_eri_ovov_ri(grid, psi_occ, n_occ, psi_virt, n_virt)
               : internal::mo_eri_ovov(grid, psi_occ, n_occ, psi_virt, n_virt,
                                       nonperiodic);

    const std::vector<double> eps_occ(result_.eps.begin(),
                                      result_.eps.begin() + n_occ);
    const std::vector<double> eps_virt(result_.eps.begin() + n_occ,
                                       result_.eps.begin() + n_occ + n_virt);

    internal::DirectRpaAmplitudes rpa;
    internal::direct_rpa_rhf(v_ovov, eps_occ, eps_virt, rpa);
    const int n_exc = static_cast<int>(rpa.omega.size());
    const int nov = n_occ * n_virt;

    // V_ia(r) intermediates (periodic Coulomb; ρ_ia neutral).
    const std::vector<double> V_ov = internal::mo_pair_potentials_ov(
        grid, psi_occ, n_occ, psi_virt, n_virt);

    // Transition densities tdm[s, m, p] = Σ_ia √2 (X+Y)_ia^s (ia|mp)
    // Layout: [s][m][p] row-major, strides (n_mo*n_mo, n_mo, 1).
    // For each s form W_s(g) = Σ_ia √2 (X+Y)_ia^s V_ia(g), then
    // tdm[s,m,p] = w Σ_g W_s(g) ψ_m(g) ψ_p(g).
    const double sqrt2 = std::sqrt(2.0);
    std::vector<double> tdm(static_cast<std::size_t>(n_exc)
                                * static_cast<std::size_t>(n_mo)
                                * static_cast<std::size_t>(n_mo),
                            0.0);
    std::vector<double> Ws(static_cast<std::size_t>(npts));
    for (int s = 0; s < n_exc; ++s) {
        std::fill(Ws.begin(), Ws.end(), 0.0);
        for (int ia = 0; ia < nov; ++ia) {
            const double z =
                sqrt2
                * (rpa.X[static_cast<std::size_t>(s)
                             * static_cast<std::size_t>(nov)
                         + static_cast<std::size_t>(ia)]
                   + rpa.Y[static_cast<std::size_t>(s)
                               * static_cast<std::size_t>(nov)
                           + static_cast<std::size_t>(ia)]);
            if (z == 0.0)
                continue;
            const double* Via =
                V_ov.data()
                + static_cast<std::size_t>(ia) * static_cast<std::size_t>(npts);
            for (long g = 0; g < npts; ++g)
                Ws[static_cast<std::size_t>(g)] += z * Via[g];
        }
        const std::size_t s0 = static_cast<std::size_t>(s)
                               * static_cast<std::size_t>(n_mo)
                               * static_cast<std::size_t>(n_mo);
        for (int m = 0; m < n_mo; ++m) {
            const double* pm =
                psi_all.data()
                + static_cast<std::size_t>(m) * static_cast<std::size_t>(npts);
            for (int p = 0; p < n_mo; ++p) {
                const double* pp =
                    psi_all.data()
                    + static_cast<std::size_t>(p)
                          * static_cast<std::size_t>(npts);
                double acc = 0.0;
                for (long g = 0; g < npts; ++g)
                    acc += Ws[static_cast<std::size_t>(g)] * pm[g] * pp[g];
                tdm[s0 + static_cast<std::size_t>(m)
                             * static_cast<std::size_t>(n_mo)
                         + static_cast<std::size_t>(p)] = gw * acc;
            }
        }
    }

    // Exact-exchange diagonals (MT Coulomb).
    const std::vector<double> sigma_x =
        internal::exchange_diag_mt(grid, psi_all, n_mo, n_occ);

    // Local v_xc diagonals from the same functional / closed-shell density.
    std::vector<double> rho(static_cast<std::size_t>(npts), 0.0);
    for (int i = 0; i < n_occ; ++i) {
        const double* pi =
            psi_all.data()
            + static_cast<std::size_t>(i) * static_cast<std::size_t>(npts);
        for (long g = 0; g < npts; ++g)
            rho[static_cast<std::size_t>(g)] += 2.0 * pi[g] * pi[g];
    }
    const XcOnGrid xc_grid = eval_xc_grid(this->xc, grid, rho);

    std::vector<double> vxc_diag(static_cast<std::size_t>(n_mo), 0.0);
    for (int p = 0; p < n_mo; ++p) {
        const double* pp =
            psi_all.data()
            + static_cast<std::size_t>(p) * static_cast<std::size_t>(npts);
        double acc = 0.0;
        for (long g = 0; g < npts; ++g)
            acc +=
                pp[g] * pp[g] * xc_grid.vxc_r[static_cast<std::size_t>(g)];
        vxc_diag[static_cast<std::size_t>(p)] = gw * acc;
    }

    std::vector<double> eps = result_.eps; // full spectrum

    g0w0_result_ = G0W0Result{};
    g0w0_result_.n_occ = n_occ;
    g0w0_result_.n_virt = n_virt;
    g0w0_result_.ri = use_ri;
    g0w0_result_.n_kpts = 0;
    g0w0_result_.mf_energy_ha = eps;
    g0w0_result_.qp_energy_ha.resize(static_cast<std::size_t>(n_mo));
    g0w0_result_.sigma_x_ha = sigma_x;
    g0w0_result_.vxc_ha = vxc_diag;
    g0w0_result_.converged = true;

    for (int p = 0; p < n_mo; ++p) {
        const double vx_m_vxc = sigma_x[static_cast<std::size_t>(p)]
                                - vxc_diag[static_cast<std::size_t>(p)];
        double qp = eps[static_cast<std::size_t>(p)];
        const bool ok =
            solve_qp(eps[static_cast<std::size_t>(p)], vx_m_vxc, p, n_occ,
                     n_mo, eps, rpa.omega, tdm, qp);
        if (!ok)
            g0w0_result_.converged = false;
        g0w0_result_.qp_energy_ha[static_cast<std::size_t>(p)] = qp;
    }

    const int homo = n_occ - 1;
    const int lumo = n_occ;
    const double e_h =
        g0w0_result_.qp_energy_ha[static_cast<std::size_t>(homo)];
    const double e_l =
        g0w0_result_.qp_energy_ha[static_cast<std::size_t>(lumo)];
    const double mf_h = eps[static_cast<std::size_t>(homo)];
    const double mf_l = eps[static_cast<std::size_t>(lumo)];
    g0w0_result_.ip_ev = -e_h * units::kHartreeInEv;
    g0w0_result_.ea_ev = e_l * units::kHartreeInEv;
    g0w0_result_.gap_ev = (e_l - e_h) * units::kHartreeInEv;
    g0w0_result_.mf_gap_ev = (mf_l - mf_h) * units::kHartreeInEv;
    return g0w0_result_;
}

} // namespace gpw
} // namespace cpu
} // namespace ATV
