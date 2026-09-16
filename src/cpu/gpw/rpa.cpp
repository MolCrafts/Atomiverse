/// @file src/cpu/gpw/rpa.cpp
/// @brief GpwDft::direct_rpa — Γ-point (real) and multi-k finite-q (complex
///        Bloch) GPW dRPA on the converged RKS or UKS orbitals. Product:
///        RI-GPW (ia|jb); Γ foothold: Poisson tensor. Plasmon formula in
///        internal::direct_rpa_rhf_energy{,_kpoints} (RKS),
///        internal::direct_rpa_uhf (Γ URPA@UKS), and
///        internal::direct_rpa_uhf_energy_kpoints (multi-k URPA).

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/internal/kpoints.h"
#include "atomiverse/cpu/gpw/internal/mp2_kernels.h"

namespace ATV {
namespace cpu {
namespace gpw {

namespace {

// Cap virtual count and check HOMO–LUMO gap for one spin channel (URPA).
int cap_virt_rpa_(int n_mo, int n_occ, int mp2_n_virt,
                  const std::vector<double>& eps, const char* channel) {
    if (n_mo <= n_occ)
        throw std::invalid_argument(
            std::string("GpwDft::direct_rpa: no virtual orbitals in the ")
            + channel + " channel (n_mo = " + std::to_string(n_mo)
            + ", n_occ = " + std::to_string(n_occ) + ")");
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
            std::string("GpwDft::direct_rpa: vanishing ") + channel
            + " HOMO-LUMO gap (" + std::to_string(gap)
            + " Ha < 1e-6) — dRPA (A−B) is singular");
    return n_virt;
}

int count_basis_rpa_(const std::vector<GpwAtom>& atoms) {
    int nb = 0;
    for (const auto& a : atoms)
        nb += static_cast<int>(a.basis.size());
    return nb;
}

} // namespace

const RpaResult& GpwDft::direct_rpa(const std::vector<GpwAtom>& atoms) {
    if (!built_)
        throw std::logic_error(
            "GpwDft::direct_rpa called before build() — validate the "
            "configuration first (no silent auto-build)");
    if (use_ot)
        throw std::invalid_argument(
            "GpwDft::direct_rpa: OT stores occupied orbitals only — canonical "
            "virtuals/eigenvalues require the diagonalization path "
            "(use_ot = false)");
    if (!result_.converged)
        throw std::logic_error(
            "GpwDft::direct_rpa needs a converged run() first (no correlation "
            "on an unconverged reference)");

    const bool multi_k =
        result_.n_kpts >= 1 && !result_.C_k.empty()
        && static_cast<int>(result_.C_k.size()) == result_.n_kpts;

    if (result_.n_kpts > 0 && !multi_k)
        throw std::invalid_argument(
            "GpwDft::direct_rpa: multi-k SCF result is missing Bloch C_k "
            "(re-run with a current multi-k path that stores orbitals)");

    // ── Unrestricted URPA (spin-conserving dRPA@UKS) ──────────────────
    if (is_uks()) {
        // Multi-k UKS: finite-q inter-k URPA on C_k / C_k_beta.
        if (multi_k || result_.n_kpts > 0) {
            if (!multi_k)
                throw std::invalid_argument(
                    "GpwDft::direct_rpa: multi-k UKS SCF result is missing "
                    "Bloch C_k (re-run with a current multi-k path that "
                    "stores full α/β orbitals)");
            if (nonperiodic)
                throw std::invalid_argument(
                    "GpwDft::direct_rpa: multi-k URPA requires a periodic "
                    "cell");
            if (!mp2_ri)
                throw std::invalid_argument(
                    "GpwDft::direct_rpa: multi-k URPA requires mp2_ri = true "
                    "(Poisson multi-k foothold is not offered)");
            if (static_cast<int>(result_.C_k_beta.size()) != result_.n_kpts
                || static_cast<int>(result_.eps_k_beta.size())
                       != result_.n_kpts)
                throw std::invalid_argument(
                    "GpwDft::direct_rpa: multi-k UKS result is missing "
                    "C_k_beta / eps_k_beta (full β Bloch MOs required for "
                    "URPA)");

            const int nk = result_.n_kpts;
            const int nb_atoms = count_basis_rpa_(atoms);
            const int nb = static_cast<int>(result_.C_k[0].shape(0));
            if (nb_atoms != nb)
                throw std::invalid_argument(
                    "GpwDft::direct_rpa: atom basis count ("
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
                        "GpwDft::direct_rpa: multi-k UKS eps_k/C_k shape "
                        "mismatch");
            }

            const int n_oa = n_alpha;
            const int n_ob = n_beta;
            std::vector<double> eps_a0(result_.eps_k[0].begin(),
                                       result_.eps_k[0].begin() + n_mo_a_min);
            std::vector<double> eps_b0(
                result_.eps_k_beta[0].begin(),
                result_.eps_k_beta[0].begin() + n_mo_b_min);
            const int n_va =
                cap_virt_rpa_(n_mo_a_min, n_oa, mp2_n_virt, eps_a0, "α");
            const int n_vb =
                cap_virt_rpa_(n_mo_b_min, n_ob, mp2_n_virt, eps_b0, "β");
            if ((n_oa < 1 || n_va < 1) && (n_ob < 1 || n_vb < 1))
                throw std::invalid_argument(
                    "GpwDft::direct_rpa: multi-k URPA needs at least one "
                    "non-empty spin channel (n_occ and n_virt both >= 1)");

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
            if (n_oa >= 1 && n_va >= 1
                && min_eps_gap(result_.eps_k, n_oa) < kMinGapHa)
                throw std::invalid_argument(
                    "GpwDft::direct_rpa: vanishing α HOMO-LUMO gap on the "
                    "k-mesh");
            if (n_ob >= 1 && n_vb >= 1
                && min_eps_gap(result_.eps_k_beta, n_ob) < kMinGapHa)
                throw std::invalid_argument(
                    "GpwDft::direct_rpa: vanishing β HOMO-LUMO gap on the "
                    "k-mesh");

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

            std::vector<std::vector<internal::Cplx>> psi_oa_k(
                static_cast<std::size_t>(nk));
            std::vector<std::vector<internal::Cplx>> psi_va_k(
                static_cast<std::size_t>(nk));
            std::vector<std::vector<internal::Cplx>> psi_ob_k(
                static_cast<std::size_t>(nk));
            std::vector<std::vector<internal::Cplx>> psi_vb_k(
                static_cast<std::size_t>(nk));
            std::vector<std::vector<double>> eps_oa_k(
                static_cast<std::size_t>(nk));
            std::vector<std::vector<double>> eps_va_k(
                static_cast<std::size_t>(nk));
            std::vector<std::vector<double>> eps_ob_k(
                static_cast<std::size_t>(nk));
            std::vector<std::vector<double>> eps_vb_k(
                static_cast<std::size_t>(nk));
            for (int ik = 0; ik < nk; ++ik) {
                if (n_oa > 0 && n_va > 0) {
                    psi_oa_k[static_cast<std::size_t>(ik)] =
                        internal::mo_on_grid_k(
                            kbg, ik,
                            result_.C_k[static_cast<std::size_t>(ik)], 0,
                            n_oa);
                    psi_va_k[static_cast<std::size_t>(ik)] =
                        internal::mo_on_grid_k(
                            kbg, ik,
                            result_.C_k[static_cast<std::size_t>(ik)], n_oa,
                            n_va);
                    const auto& ea =
                        result_.eps_k[static_cast<std::size_t>(ik)];
                    eps_oa_k[static_cast<std::size_t>(ik)].assign(
                        ea.begin(), ea.begin() + n_oa);
                    eps_va_k[static_cast<std::size_t>(ik)].assign(
                        ea.begin() + n_oa, ea.begin() + n_oa + n_va);
                }
                if (n_ob > 0 && n_vb > 0) {
                    psi_ob_k[static_cast<std::size_t>(ik)] =
                        internal::mo_on_grid_k(
                            kbg, ik,
                            result_.C_k_beta[static_cast<std::size_t>(ik)],
                            0, n_ob);
                    psi_vb_k[static_cast<std::size_t>(ik)] =
                        internal::mo_on_grid_k(
                            kbg, ik,
                            result_.C_k_beta[static_cast<std::size_t>(ik)],
                            n_ob, n_vb);
                    const auto& eb =
                        result_.eps_k_beta[static_cast<std::size_t>(ik)];
                    eps_ob_k[static_cast<std::size_t>(ik)].assign(
                        eb.begin(), eb.begin() + n_ob);
                    eps_vb_k[static_cast<std::size_t>(ik)].assign(
                        eb.begin() + n_ob, eb.begin() + n_ob + n_vb);
                }
            }

            double e_corr = 0.0, tr_A = 0.0;
            std::vector<double> omega;
            internal::direct_rpa_uhf_energy_kpoints(
                grid, psi_oa_k, psi_va_k, psi_ob_k, psi_vb_k, eps_oa_k,
                eps_va_k, eps_ob_k, eps_vb_k, result_.k_frac,
                result_.k_weights, e_corr, tr_A, omega);

            rpa_result_ = RpaResult{};
            rpa_result_.e_corr = e_corr;
            rpa_result_.e_scf = result_.e_total;
            rpa_result_.e_total = rpa_result_.e_scf + rpa_result_.e_corr;
            rpa_result_.tr_A = tr_A;
            rpa_result_.n_occ = n_oa + n_ob;
            rpa_result_.n_virt = n_va + n_vb;
            rpa_result_.ri = true;
            rpa_result_.n_kpts = nk;
            rpa_result_.omega_ha = std::move(omega);
            return rpa_result_;
        }

        // Γ URPA
        if (result_.C_beta.size() == 0 || result_.eps_beta.empty())
            throw std::logic_error(
                "GpwDft::direct_rpa: UKS result is missing β orbitals/"
                "eigenvalues");

        const int nb_atoms = count_basis_rpa_(atoms);
        const int nb = static_cast<int>(result_.C.shape(0));
        const int n_mo_a = static_cast<int>(result_.C.shape(1));
        const int n_mo_b = static_cast<int>(result_.C_beta.shape(1));
        if (nb_atoms != nb)
            throw std::invalid_argument(
                "GpwDft::direct_rpa: atom basis count ("
                + std::to_string(nb_atoms)
                + ") does not match the converged orbitals ("
                + std::to_string(nb)
                + ") — same geometry/basis as the last run() required");
        if (static_cast<int>(result_.eps.size()) != n_mo_a
            || static_cast<int>(result_.eps_beta.size()) != n_mo_b)
            throw std::logic_error(
                "GpwDft::direct_rpa: UKS eps/C shape mismatch in the "
                "converged result");
        if (static_cast<int>(result_.C_beta.shape(0)) != nb)
            throw std::logic_error(
                "GpwDft::direct_rpa: UKS α/β basis dimension mismatch");

        const int n_oa = n_alpha;
        const int n_ob = n_beta;
        const int n_va =
            cap_virt_rpa_(n_mo_a, n_oa, mp2_n_virt, result_.eps, "α");
        const int n_vb =
            cap_virt_rpa_(n_mo_b, n_ob, mp2_n_virt, result_.eps_beta, "β");
        if ((n_oa < 1 || n_va < 1) && (n_ob < 1 || n_vb < 1))
            throw std::invalid_argument(
                "GpwDft::direct_rpa: URPA needs at least one non-empty "
                "spin channel (n_occ and n_virt both >= 1)");

        const bool periodic = !nonperiodic;
        const UniformGrid grid = make_uniform_grid(cell, cutoff_ry);
        std::vector<GridBasisFunction> basis;
        basis.reserve(static_cast<std::size_t>(nb));
        for (const auto& a : atoms)
            for (const auto& bf : a.basis)
                basis.push_back(bf);
        const CompactBasisGrid cbg =
            build_compact_basis_grid(grid, basis, periodic, eps_default);

        const std::vector<double> psi_oa =
            (n_oa > 0) ? internal::mo_on_grid(cbg, result_.C, 0, n_oa)
                       : std::vector<double>{};
        const std::vector<double> psi_va =
            (n_va > 0) ? internal::mo_on_grid(cbg, result_.C, n_oa, n_va)
                       : std::vector<double>{};
        const std::vector<double> psi_ob =
            (n_ob > 0)
                ? internal::mo_on_grid(cbg, result_.C_beta, 0, n_ob)
                : std::vector<double>{};
        const std::vector<double> psi_vb =
            (n_vb > 0)
                ? internal::mo_on_grid(cbg, result_.C_beta, n_ob, n_vb)
                : std::vector<double>{};

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

        internal::DirectRpaAmplitudes amp;
        internal::direct_rpa_uhf(v_aa, v_bb, v_ab, eps_oa, eps_va, eps_ob,
                                 eps_vb, amp);

        rpa_result_ = RpaResult{};
        rpa_result_.e_corr = amp.e_corr;
        rpa_result_.e_scf = result_.e_total;
        rpa_result_.e_total = rpa_result_.e_scf + rpa_result_.e_corr;
        rpa_result_.tr_A = amp.tr_A;
        // Spin-orbital occupied / virtual counts (UMP2 convention).
        rpa_result_.n_occ = n_oa + n_ob;
        rpa_result_.n_virt = n_va + n_vb;
        rpa_result_.ri = use_ri;
        rpa_result_.n_kpts = 0;
        rpa_result_.omega_ha = std::move(amp.omega);
        return rpa_result_;
    }

    // ── Multi-k (complex Bloch, finite-q inter-k RI-GPW dRPA) ──────────
    if (multi_k) {
        if (nonperiodic)
            throw std::invalid_argument(
                "GpwDft::direct_rpa: multi-k dRPA requires a periodic cell");
        if (!mp2_ri)
            throw std::invalid_argument(
                "GpwDft::direct_rpa: multi-k path requires mp2_ri = true "
                "(Poisson multi-k foothold is not offered)");

        const int nk = result_.n_kpts;
        const int nb_atoms = count_basis_rpa_(atoms);
        const int nb = static_cast<int>(result_.C_k[0].shape(0));
        if (nb_atoms != nb)
            throw std::invalid_argument(
                "GpwDft::direct_rpa: atom basis count ("
                + std::to_string(nb_atoms)
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
                    "GpwDft::direct_rpa: eps_k/C_k shape mismatch at a "
                    "k-point");
        }
        if (n_mo_min <= n_occ)
            throw std::invalid_argument(
                "GpwDft::direct_rpa: no virtual orbitals in the multi-k "
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
                "GpwDft::direct_rpa: vanishing HOMO-LUMO gap on the k-mesh ("
                + std::to_string(min_gap)
                + " Ha < 1e-6) — dRPA (A−B) is singular");

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
            internal::build_kpoint_basis_grid(grid, basis, kpts, eps_default);

        std::vector<std::vector<internal::Cplx>> psi_occ_k(
            static_cast<std::size_t>(nk));
        std::vector<std::vector<internal::Cplx>> psi_virt_k(
            static_cast<std::size_t>(nk));
        std::vector<std::vector<double>> eps_occ_k(
            static_cast<std::size_t>(nk));
        std::vector<std::vector<double>> eps_virt_k(
            static_cast<std::size_t>(nk));
        for (int ik = 0; ik < nk; ++ik) {
            psi_occ_k[static_cast<std::size_t>(ik)] = internal::mo_on_grid_k(
                kbg, ik, result_.C_k[static_cast<std::size_t>(ik)], 0, n_occ);
            psi_virt_k[static_cast<std::size_t>(ik)] =
                internal::mo_on_grid_k(
                    kbg, ik, result_.C_k[static_cast<std::size_t>(ik)], n_occ,
                    n_virt);
            const auto& eps = result_.eps_k[static_cast<std::size_t>(ik)];
            eps_occ_k[static_cast<std::size_t>(ik)].assign(
                eps.begin(), eps.begin() + n_occ);
            eps_virt_k[static_cast<std::size_t>(ik)].assign(
                eps.begin() + n_occ, eps.begin() + n_occ + n_virt);
        }

        double e_corr = 0.0, tr_A = 0.0;
        std::vector<double> omega;
        internal::direct_rpa_rhf_energy_kpoints(
            grid, psi_occ_k, psi_virt_k, eps_occ_k, eps_virt_k, result_.k_frac,
            result_.k_weights, e_corr, tr_A, omega);

        rpa_result_ = RpaResult{};
        rpa_result_.e_corr = e_corr;
        rpa_result_.e_scf = result_.e_total;
        rpa_result_.e_total = rpa_result_.e_scf + rpa_result_.e_corr;
        rpa_result_.tr_A = tr_A;
        rpa_result_.n_occ = n_occ;
        rpa_result_.n_virt = n_virt;
        rpa_result_.ri = true;
        rpa_result_.n_kpts = nk;
        rpa_result_.omega_ha = std::move(omega);
        return rpa_result_;
    }

    // ── Γ-point real path (product RI-GPW or Poisson foothold) ────────
    const int nb_atoms = count_basis_rpa_(atoms);
    const int nb = static_cast<int>(result_.C.shape(0));
    const int n_mo = static_cast<int>(result_.C.shape(1));
    if (nb_atoms != nb)
        throw std::invalid_argument(
            "GpwDft::direct_rpa: atom basis count (" + std::to_string(nb_atoms)
            + ") does not match the converged orbitals (" + std::to_string(nb)
            + ") — same geometry/basis as the last run() required");
    if (static_cast<int>(result_.eps.size()) != n_mo)
        throw std::logic_error(
            "GpwDft::direct_rpa: eps/C shape mismatch in the converged result");
    if (n_mo <= n_occ)
        throw std::invalid_argument(
            "GpwDft::direct_rpa: no virtual orbitals in the converged result "
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
            "GpwDft::direct_rpa: vanishing HOMO-LUMO gap ("
            + std::to_string(gap)
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

    const std::vector<double> psi_occ =
        internal::mo_on_grid(cbg, result_.C, 0, n_occ);
    const std::vector<double> psi_virt =
        internal::mo_on_grid(cbg, result_.C, n_occ, n_virt);

    const bool use_ri = mp2_ri && periodic;
    const std::vector<double> v_ovov =
        use_ri ? internal::mo_eri_ovov_ri(grid, psi_occ, n_occ, psi_virt, n_virt)
               : internal::mo_eri_ovov(grid, psi_occ, n_occ, psi_virt, n_virt,
                                       nonperiodic);

    const std::vector<double> eps_occ(result_.eps.begin(),
                                      result_.eps.begin() + n_occ);
    const std::vector<double> eps_virt(result_.eps.begin() + n_occ,
                                       result_.eps.begin() + n_occ + n_virt);
    double e_corr = 0.0, tr_A = 0.0;
    std::vector<double> omega;
    internal::direct_rpa_rhf_energy(v_ovov, eps_occ, eps_virt, e_corr, tr_A,
                                    omega);

    rpa_result_ = RpaResult{};
    rpa_result_.e_corr = e_corr;
    rpa_result_.e_scf = result_.e_total;
    rpa_result_.e_total = rpa_result_.e_scf + rpa_result_.e_corr;
    rpa_result_.tr_A = tr_A;
    rpa_result_.n_occ = n_occ;
    rpa_result_.n_virt = n_virt;
    rpa_result_.ri = use_ri;
    rpa_result_.n_kpts = 0;
    rpa_result_.omega_ha = std::move(omega);
    return rpa_result_;
}

} // namespace gpw
} // namespace cpu
} // namespace ATV
