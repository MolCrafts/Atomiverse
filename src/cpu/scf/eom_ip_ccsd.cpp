/// @file cpu/scf/eom_ip_ccsd.cpp
/// @brief Spin-orbital EOM-IP-CCSD ionization potentials.
///
/// Intermediates follow PySCF `cc.gintermediates` (Gauss & Stanton 1995 Table
/// III, EOM section (b)); the right-hand σ equations are Tu, Wang & Li,
/// J. Chem. Phys. 136, 174102 (2012) Eqs. (8)–(9) as implemented in
/// `eom_gccsd.ipccsd_matvec`. Packed vector basis: r1[i] + r2[i>j,a].

#include "atomiverse/cpu/scf/eom_ip_ccsd.h"
#include "atomiverse/cpu/scf/internal/eom_ip_ccsd_pack.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/linalg/eigensolve.h"
#include "atomiverse/cpu/scf/internal/ccsd_pack.h"
#include "atomiverse/cpu/scf/internal/eom_ccsd_common.h"
#include "atomiverse/cpu/scf/rhf.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace {

using T2 = xt::xtensor<double, 4>;
using T1 = xt::xtensor<double, 2>;
using R1 = xt::xtensor<double, 1>; // occupied singles (no)
using R2 = xt::xtensor<double, 3>; // 2h1p (no, no, nv)

/// Packed IP vector size: no + C(no, 2) * nv.
int vector_size_ip(int no, int nv) { return no + (no * (no - 1) / 2) * nv; }

void unpack_ip(const std::vector<double>& v, int no, int nv, R1& r1, R2& r2) {
    const std::size_t No = static_cast<std::size_t>(no);
    const std::size_t Nv = static_cast<std::size_t>(nv);
    r1 = xt::zeros<double>({No});
    r2 = xt::zeros<double>({No, No, Nv});
    for (int i = 0; i < no; ++i)
        r1(i) = v[static_cast<std::size_t>(i)];
    int k = no;
    // r2[i>j, a] packed as tril_indices(nocc, -1): for each a, then i, j<i
    // PySCF: r2[idx,idy] = packed.reshape(C(no,2), nv) with (idx,idy)=tril
    // then ravel order is (pairs-major, a-last) via reshape then assignment
    // amplitudes_to_vector: r2[tril].ravel() → pairs contiguous per (i,j), a last
    // vector_to_amplitudes: packed[nocc:].reshape(C(no,2), nvir) then r2[idx,idy]=
    for (int i = 0; i < no; ++i)
        for (int j = 0; j < i; ++j)
            for (int a = 0; a < nv; ++a) {
                const double x = v[static_cast<std::size_t>(k++)];
                r2(i, j, a) = x;
                r2(j, i, a) = -x;
            }
}

std::vector<double> pack_ip(const R1& r1, const R2& r2, int no, int nv) {
    std::vector<double> v(static_cast<std::size_t>(vector_size_ip(no, nv)),
                          0.0);
    for (int i = 0; i < no; ++i)
        v[static_cast<std::size_t>(i)] = r1(i);
    int k = no;
    for (int i = 0; i < no; ++i)
        for (int j = 0; j < i; ++j)
            for (int a = 0; a < nv; ++a)
                v[static_cast<std::size_t>(k++)] = r2(i, j, a);
    return v;
}

/// Antisymmetrize R2 in occupieds in place: ½ P(ij).
void antisym_r2(R2& r2, int no, int nv) {
    R2 s = xt::zeros<double>({static_cast<std::size_t>(no),
                              static_cast<std::size_t>(no),
                              static_cast<std::size_t>(nv)});
    for (int i = 0; i < no; ++i)
        for (int j = 0; j < no; ++j)
            for (int a = 0; a < nv; ++a)
                s(i, j, a) = 0.5 * (r2(i, j, a) - r2(j, i, a));
    r2 = s;
}

/// Right-hand EOM-IP-CCSD σ = H̄ r (Tu/Wang/Li + PySCF eom_gccsd.ipccsd_matvec).
std::vector<double> matvec_ip(const EomImds& im,
                              const std::vector<double>& vec) {
    const int no = im.no;
    const int nv = im.nv;
    R1 r1;
    R2 r2;
    unpack_ip(vec, no, nv, r1, r2);

    // Eq. (8) Hr1[i]
    R1 Hr1 = xt::zeros<double>({static_cast<std::size_t>(no)});
    for (int i = 0; i < no; ++i) {
        double s = 0.0;
        for (int m = 0; m < no; ++m)
            s -= im.Foo(m, i) * r1(m);
        for (int m = 0; m < no; ++m)
            for (int e = 0; e < nv; ++e)
                s += im.Fov(m, e) * r2(m, i, e);
        for (int n = 0; n < no; ++n)
            for (int m = 0; m < no; ++m)
                for (int e = 0; e < nv; ++e)
                    s -= 0.5 * im.Wooov(n, m, i, e) * r2(m, n, e);
        Hr1(i) = s;
    }

    // Eq. (9) Hr2[i,j,a]
    R2 Hr2 = xt::zeros<double>({static_cast<std::size_t>(no),
                                static_cast<std::size_t>(no),
                                static_cast<std::size_t>(nv)});
    for (int i = 0; i < no; ++i)
        for (int j = 0; j < no; ++j)
            for (int a = 0; a < nv; ++a) {
                double s = 0.0;
                for (int e = 0; e < nv; ++e)
                    s += im.Fvv(a, e) * r2(i, j, e);
                for (int m = 0; m < no; ++m)
                    s -= im.Wovoo(m, a, j, i) * r1(m);
                for (int m = 0; m < no; ++m)
                    for (int n = 0; n < no; ++n)
                        s += 0.5 * im.Woooo(m, n, i, j) * r2(m, n, a);
                double t = 0.0;
                for (int m = 0; m < no; ++m)
                    for (int n = 0; n < no; ++n)
                        for (int e = 0; e < nv; ++e)
                            for (int f = 0; f < nv; ++f)
                                t += im.Woovv(m, n, e, f) * r2(m, n, f)
                                     * im.t2(i, j, a, e);
                s += 0.5 * t;
                Hr2(i, j, a) = s;
            }

    // P(ij) Foo · r2 : tmp1[i,j,a] = Σ_m Foo[m,i] r2[m,j,a]
    {
        R2 tmp1 = xt::zeros<double>({static_cast<std::size_t>(no),
                                     static_cast<std::size_t>(no),
                                     static_cast<std::size_t>(nv)});
        for (int i = 0; i < no; ++i)
            for (int j = 0; j < no; ++j)
                for (int a = 0; a < nv; ++a) {
                    double s = 0.0;
                    for (int m = 0; m < no; ++m)
                        s += im.Foo(m, i) * r2(m, j, a);
                    tmp1(i, j, a) = s;
                }
        for (int i = 0; i < no; ++i)
            for (int j = 0; j < no; ++j)
                for (int a = 0; a < nv; ++a)
                    Hr2(i, j, a) -= tmp1(i, j, a) - tmp1(j, i, a);
    }

    // P(ij) Wovvo · r2 : tmp2[i,j,a] = Σ_me Wovvo[m,a,e,i] r2[m,j,e]
    {
        R2 tmp2 = xt::zeros<double>({static_cast<std::size_t>(no),
                                     static_cast<std::size_t>(no),
                                     static_cast<std::size_t>(nv)});
        for (int i = 0; i < no; ++i)
            for (int j = 0; j < no; ++j)
                for (int a = 0; a < nv; ++a) {
                    double s = 0.0;
                    for (int m = 0; m < no; ++m)
                        for (int e = 0; e < nv; ++e)
                            s += im.Wovvo(m, a, e, i) * r2(m, j, e);
                    tmp2(i, j, a) = s;
                }
        for (int i = 0; i < no; ++i)
            for (int j = 0; j < no; ++j)
                for (int a = 0; a < nv; ++a)
                    Hr2(i, j, a) += tmp2(i, j, a) - tmp2(j, i, a);
    }

    antisym_r2(Hr2, no, nv);
    return pack_ip(Hr1, Hr2, no, nv);
}

} // namespace

EOMIPCCSDResult eom_ip_ccsd_energies(const internal::IntegralView& ints,
                                     const SlaterDeterminant& det,
                                     double e_scf_ha, int n_states, double tol,
                                     int max_iter) {
    if (n_states < 1)
        throw std::invalid_argument(
            "eom_ip_ccsd: n_states must be ≥ 1 (dimensionless).");

    CCSDSolution sol = ccsd_solve(ints, det, e_scf_ha, tol, max_iter);
    EOMIPCCSDResult out;
    out.e_corr_ha = sol.result.e_corr;
    out.e_ccsd_ha = sol.result.e_tot;
    out.ccsd_iterations = sol.result.iterations;
    out.converged = sol.result.converged;

    if (sol.nv <= 0 || sol.no <= 0) {
        throw std::invalid_argument(
            "eom_ip_ccsd: empty occupied or virtual spin-orbital space.");
    }

    const int dim = vector_size_ip(sol.no, sol.nv);
    if (dim < 1)
        throw std::invalid_argument(
            "eom_ip_ccsd: empty IP space (need at least one occupied).");
    if (n_states > dim)
        throw std::invalid_argument(
            "eom_ip_ccsd: n_states=" + std::to_string(n_states)
            + " exceeds EOM-IP dimension " + std::to_string(dim) + ".");

    const EomImds im(sol);

    std::vector<std::vector<double>> H(
        static_cast<std::size_t>(dim),
        std::vector<double>(static_cast<std::size_t>(dim), 0.0));
    for (int j = 0; j < dim; ++j) {
        std::vector<double> e(static_cast<std::size_t>(dim), 0.0);
        e[static_cast<std::size_t>(j)] = 1.0;
        const std::vector<double> sig = matvec_ip(im, e);
        for (int i = 0; i < dim; ++i)
            H[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                sig[static_cast<std::size_t>(i)];
    }

    auto eigs = eig_dense(H, "eom_ip_ccsd");

    // Real-valued eigenvalues, ascending; de-duplicate spin degeneracy (α/β hole).
    std::vector<double> roots;
    for (const auto& ep : eigs) {
        if (std::abs(ep.wi) > 1e-8)
            continue;
        roots.push_back(ep.wr);
    }
    std::sort(roots.begin(), roots.end());
    std::vector<double> uniq;
    for (double w : roots) {
        if (!uniq.empty() && std::abs(uniq.back() - w) < 1e-8)
            continue;
        uniq.push_back(w);
    }
    if (static_cast<int>(uniq.size()) < n_states)
        throw std::runtime_error("eom_ip_ccsd: found only "
                                 + std::to_string(uniq.size())
                                 + " unique real IP roots; requested n_states="
                                 + std::to_string(n_states));

    out.energy_ev.reserve(static_cast<std::size_t>(n_states));
    for (int k = 0; k < n_states; ++k)
        out.energy_ev.push_back(uniq[static_cast<std::size_t>(k)]
                                * units::kHartreeInEv);
    out.converged = true;
    return out;
}

EOMIPCCSDResult RHF::eom_ip_ccsd(Frame& frame, int n_states) {
    const bool prev = compute_eri;
    compute_eri = true;
    run(frame);
    compute_eri = prev;
    if (!this->converged())
        throw std::invalid_argument(
            "RHF::eom_ip_ccsd: SCF reference did not converge — refusing to "
            "compute EOM-IP-CCSD on a non-stationary determinant (no "
            "fallback).");
    return eom_ip_ccsd_energies(integral_view_(), determinant(), energy(),
                                n_states);
}

} // namespace scf
} // namespace cpu
} // namespace ATV
