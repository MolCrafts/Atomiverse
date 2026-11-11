/// @file cpu/scf/eom_ea_ccsd.cpp
/// @brief Spin-orbital EOM-EA-CCSD electron-attachment energies.
///
/// Intermediates follow PySCF `cc.gintermediates` (Gauss & Stanton 1995 Table
/// III, EOM section (b)); the right-hand σ equations are Nooijen & Bartlett,
/// J. Chem. Phys. 102, 3629 (1995) Eqs. (30)–(31) as implemented in
/// `eom_gccsd.eaccsd_matvec`. Packed vector basis: r1[a] + r2[j,a>b].

#include "atomiverse/cpu/scf/eom_ea_ccsd.h"
#include "atomiverse/cpu/scf/internal/eom_ea_ccsd_pack.h"

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

namespace ATV {
namespace cpu {
namespace scf {
namespace {

using T2 = xt::xtensor<double, 4>;
using T1 = xt::xtensor<double, 2>;
using R1 = xt::xtensor<double, 1>; // virtual singles (nv)
using R2 = xt::xtensor<double, 3>; // 2p1h (no, nv, nv)

/// Packed EA vector size: nv + no * C(nv, 2).
int vector_size_ea(int no, int nv) { return nv + no * (nv * (nv - 1) / 2); }

void unpack_ea(const std::vector<double>& v, int no, int nv, R1& r1, R2& r2) {
    const std::size_t No = static_cast<std::size_t>(no);
    const std::size_t Nv = static_cast<std::size_t>(nv);
    r1 = xt::zeros<double>({Nv});
    r2 = xt::zeros<double>({No, Nv, Nv});
    for (int a = 0; a < nv; ++a)
        r1(a) = v[static_cast<std::size_t>(a)];
    int k = nv;
    // r2[:, a>b] packed as tril_indices(nvir, -1): for each j, a, b<a
    for (int j = 0; j < no; ++j)
        for (int a = 0; a < nv; ++a)
            for (int b = 0; b < a; ++b) {
                const double x = v[static_cast<std::size_t>(k++)];
                r2(j, a, b) = x;
                r2(j, b, a) = -x;
            }
}

std::vector<double> pack_ea(const R1& r1, const R2& r2, int no, int nv) {
    std::vector<double> v(static_cast<std::size_t>(vector_size_ea(no, nv)),
                          0.0);
    for (int a = 0; a < nv; ++a)
        v[static_cast<std::size_t>(a)] = r1(a);
    int k = nv;
    for (int j = 0; j < no; ++j)
        for (int a = 0; a < nv; ++a)
            for (int b = 0; b < a; ++b)
                v[static_cast<std::size_t>(k++)] = r2(j, a, b);
    return v;
}

/// Antisymmetrize R2 in virtuals in place: ½ P(ab).
void antisym_r2(R2& r2, int no, int nv) {
    R2 s = xt::zeros<double>({static_cast<std::size_t>(no),
                              static_cast<std::size_t>(nv),
                              static_cast<std::size_t>(nv)});
    for (int j = 0; j < no; ++j)
        for (int a = 0; a < nv; ++a)
            for (int b = 0; b < nv; ++b)
                s(j, a, b) = 0.5 * (r2(j, a, b) - r2(j, b, a));
    r2 = s;
}

/// Right-hand EOM-EA-CCSD σ = H̄ r (Nooijen/Bartlett + PySCF eom_gccsd).
std::vector<double> matvec_ea(const EomImds& im,
                              const std::vector<double>& vec) {
    const int no = im.no;
    const int nv = im.nv;
    R1 r1;
    R2 r2;
    unpack_ea(vec, no, nv, r1, r2);

    // Eq. (30) Hr1[a]
    R1 Hr1 = xt::zeros<double>({static_cast<std::size_t>(nv)});
    for (int a = 0; a < nv; ++a) {
        double s = 0.0;
        for (int c = 0; c < nv; ++c)
            s += im.Fvv(a, c) * r1(c);
        for (int l = 0; l < no; ++l)
            for (int d = 0; d < nv; ++d)
                s += im.Fov(l, d) * r2(l, a, d);
        for (int l = 0; l < no; ++l)
            for (int c = 0; c < nv; ++c)
                for (int d = 0; d < nv; ++d)
                    s += 0.5 * im.Wvovv(a, l, c, d) * r2(l, c, d);
        Hr1(a) = s;
    }

    // Eq. (31) Hr2[j,a,b]
    R2 Hr2 = xt::zeros<double>({static_cast<std::size_t>(no),
                                static_cast<std::size_t>(nv),
                                static_cast<std::size_t>(nv)});
    for (int j = 0; j < no; ++j)
        for (int a = 0; a < nv; ++a)
            for (int b = 0; b < nv; ++b) {
                double s = 0.0;
                for (int c = 0; c < nv; ++c)
                    s += im.Wvvvo(a, b, c, j) * r1(c);
                for (int l = 0; l < no; ++l)
                    s -= im.Foo(l, j) * r2(l, a, b);
                for (int c = 0; c < nv; ++c)
                    for (int d = 0; d < nv; ++d)
                        s += 0.5 * im.Wvvvv(a, b, c, d) * r2(j, c, d);
                double t = 0.0;
                for (int k = 0; k < no; ++k)
                    for (int l = 0; l < no; ++l)
                        for (int c = 0; c < nv; ++c)
                            for (int d = 0; d < nv; ++d)
                                t += im.Woovv(k, l, c, d) * r2(l, c, d)
                                     * im.t2(k, j, a, b);
                s -= 0.5 * t;
                Hr2(j, a, b) = s;
            }

    // P(ab) Fvv · r2 : tmp1[j,a,b] = Σ_c Fvv[a,c] r2[j,c,b]
    {
        R2 tmp1 = xt::zeros<double>({static_cast<std::size_t>(no),
                                     static_cast<std::size_t>(nv),
                                     static_cast<std::size_t>(nv)});
        for (int j = 0; j < no; ++j)
            for (int a = 0; a < nv; ++a)
                for (int b = 0; b < nv; ++b) {
                    double s = 0.0;
                    for (int c = 0; c < nv; ++c)
                        s += im.Fvv(a, c) * r2(j, c, b);
                    tmp1(j, a, b) = s;
                }
        for (int j = 0; j < no; ++j)
            for (int a = 0; a < nv; ++a)
                for (int b = 0; b < nv; ++b)
                    Hr2(j, a, b) += tmp1(j, a, b) - tmp1(j, b, a);
    }

    // P(ab) Wovvo · r2 : tmp2[j,a,b] = Σ_ld Wovvo[l,b,d,j] r2[l,a,d]
    {
        R2 tmp2 = xt::zeros<double>({static_cast<std::size_t>(no),
                                     static_cast<std::size_t>(nv),
                                     static_cast<std::size_t>(nv)});
        for (int j = 0; j < no; ++j)
            for (int a = 0; a < nv; ++a)
                for (int b = 0; b < nv; ++b) {
                    double s = 0.0;
                    for (int l = 0; l < no; ++l)
                        for (int d = 0; d < nv; ++d)
                            s += im.Wovvo(l, b, d, j) * r2(l, a, d);
                    tmp2(j, a, b) = s;
                }
        for (int j = 0; j < no; ++j)
            for (int a = 0; a < nv; ++a)
                for (int b = 0; b < nv; ++b)
                    Hr2(j, a, b) += tmp2(j, a, b) - tmp2(j, b, a);
    }

    antisym_r2(Hr2, no, nv);
    return pack_ea(Hr1, Hr2, no, nv);
}

} // namespace

EOMEACCSDResult eom_ea_ccsd_energies(const internal::IntegralView& ints,
                                     const SlaterDeterminant& det,
                                     double e_scf_ha, int n_states, double tol,
                                     int max_iter) {
    if (n_states < 1)
        throw std::invalid_argument(
            "eom_ea_ccsd: n_states must be ≥ 1 (dimensionless).");

    CCSDSolution sol = ccsd_solve(ints, det, e_scf_ha, tol, max_iter);
    EOMEACCSDResult out;
    out.e_corr_ha = sol.result.e_corr;
    out.e_ccsd_ha = sol.result.e_tot;
    out.ccsd_iterations = sol.result.iterations;
    out.converged = sol.result.converged;

    if (sol.nv <= 0 || sol.no <= 0) {
        throw std::invalid_argument(
            "eom_ea_ccsd: empty occupied or virtual spin-orbital space.");
    }

    const int dim = vector_size_ea(sol.no, sol.nv);
    if (dim < 1)
        throw std::invalid_argument(
            "eom_ea_ccsd: empty EA space (need at least one virtual).");
    if (n_states > dim)
        throw std::invalid_argument(
            "eom_ea_ccsd: n_states=" + std::to_string(n_states)
            + " exceeds EOM-EA dimension " + std::to_string(dim) + ".");

    const EomImds im(sol);

    std::vector<std::vector<double>> H(
        static_cast<std::size_t>(dim),
        std::vector<double>(static_cast<std::size_t>(dim), 0.0));
    for (int j = 0; j < dim; ++j) {
        std::vector<double> e(static_cast<std::size_t>(dim), 0.0);
        e[static_cast<std::size_t>(j)] = 1.0;
        const std::vector<double> sig = matvec_ea(im, e);
        for (int i = 0; i < dim; ++i)
            H[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                sig[static_cast<std::size_t>(i)];
    }

    auto eigs = eig_dense(H, "eom_ea_ccsd");

    // Real-valued eigenvalues, ascending; de-duplicate spin degeneracy (α/β attach).
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
        throw std::runtime_error("eom_ea_ccsd: found only "
                                 + std::to_string(uniq.size())
                                 + " unique real EA roots; requested n_states="
                                 + std::to_string(n_states));

    out.energy_ev.reserve(static_cast<std::size_t>(n_states));
    for (int k = 0; k < n_states; ++k)
        out.energy_ev.push_back(uniq[static_cast<std::size_t>(k)]
                                * units::kHartreeInEv);
    out.converged = true;
    return out;
}

} // namespace scf
} // namespace cpu
} // namespace ATV
