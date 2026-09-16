/// @file cpu/scf/mp3.cpp
/// @brief Closed-shell canonical MP3 (spin-orbital doubles residual form).

#include "atomiverse/cpu/scf/mp3.h"
#include "atomiverse/cpu/scf/internal/ao2mo.h"
#include "atomiverse/cpu/scf/internal/mp3_pack.h"
#include "atomiverse/cpu/scf/internal/spin_orbital_eri.h"

#include <cmath>
#include <stdexcept>
#include <vector>

#include <xtensor/containers/xtensor.hpp>

namespace ATV {
namespace cpu {
namespace scf {

MP3Result mp3_energy(const internal::IntegralView& ints, const SlaterDeterminant& det,
                     double e_scf_ha, int n_frozen) {
    using Spin = SlaterDeterminant::Spin;
    if (ints.ERI.size() == 0) {
        throw std::invalid_argument(
            "mp3_energy: ERI is not materialized; call internal::IntegralWorkspace{..., "
            "compute_eri=true}.");
    }
    if (det.n_electrons(Spin::alpha) != det.n_electrons(Spin::beta)) {
        throw std::invalid_argument(
            "mp3_energy: closed-shell reference only (n_alpha must equal "
            "n_beta).");
    }
    if (det.generalized()) {
        throw std::invalid_argument(
            "mp3_energy: GHF (spinor) reference is not supported.");
    }

    const xt::xtensor<double, 2> C = det.coefficients(Spin::alpha);
    const int n_occ = det.n_electrons(Spin::alpha);
    const int n_mo = static_cast<int>(C.shape(1));
    const int n_vir = n_mo - n_occ;

    if (n_frozen < 0 || n_frozen > n_occ) {
        throw std::invalid_argument(
            "mp3_energy: n_frozen must satisfy 0 <= n_frozen <= n_occ.");
    }
    if (n_vir == 0 || n_frozen == n_occ) {
        return {/*e_corr=*/0.0, /*e_tot=*/e_scf_ha, /*e_corr_mp2=*/0.0,
                /*e_corr_mp3=*/0.0, n_frozen};
    }

    // Full MO ERI in chemist notation (pq|rs).
    const xt::xtensor<double, 4> g = ao2mo_full(ints.ERI, C, n_mo);
    const xt::xtensor<double, 1>& eps = det.orbital_energies(Spin::alpha);

    // Correlated spin-orbitals: skip 2*n_frozen core SO (α+β of each frozen
    // spatial MO). SO layout: 2*spatial + spin.
    const int n_occ_so_full = 2 * n_occ;
    const int n_so = 2 * n_mo;
    const int n_fr_so = 2 * n_frozen;
    const int n_occ_so = n_occ_so_full - n_fr_so;
    const int n_vir_so = n_so - n_occ_so_full;
    if (n_occ_so < 2 || n_vir_so < 2) {
        // Need at least one opposite-spin pair for doubles; if only one
        // correlated occ SO remains, correlation is zero at this order.
        // (Still allow single-occ-spatial H2: n_occ_so=2 with α+β.)
    }

    // Map compact correlated-occ / virtual indices → absolute SO.
    std::vector<int> o_so(static_cast<std::size_t>(n_occ_so));
    std::vector<int> v_so(static_cast<std::size_t>(n_vir_so));
    for (int i = 0; i < n_occ_so; ++i)
        o_so[static_cast<std::size_t>(i)] = n_fr_so + i;
    for (int a = 0; a < n_vir_so; ++a)
        v_so[static_cast<std::size_t>(a)] = n_occ_so_full + a;

    auto eps_so = [&](int so) -> double {
        return eps(SpinOrbital::from_packed(so).spat);
    };

    // t2[i,j,a,b] = <ij||ab> / D_ijab  (compact correlated indices)
    const int no = n_occ_so;
    const int nv = n_vir_so;
    const std::size_t t_size =
        static_cast<std::size_t>(no) * static_cast<std::size_t>(no)
        * static_cast<std::size_t>(nv) * static_cast<std::size_t>(nv);
    std::vector<double> t2(t_size, 0.0);
    std::vector<double> u2(t_size, 0.0);
    auto idx = [no, nv](int i, int j, int a, int b) -> std::size_t {
        return ((static_cast<std::size_t>(i) * static_cast<std::size_t>(no)
                 + static_cast<std::size_t>(j))
                    * static_cast<std::size_t>(nv)
                + static_cast<std::size_t>(a))
                   * static_cast<std::size_t>(nv)
               + static_cast<std::size_t>(b);
    };

    for (int i = 0; i < no; ++i) {
        const int I = o_so[static_cast<std::size_t>(i)];
        for (int j = 0; j < no; ++j) {
            const int J = o_so[static_cast<std::size_t>(j)];
            for (int a = 0; a < nv; ++a) {
                const int A = v_so[static_cast<std::size_t>(a)];
                for (int b = 0; b < nv; ++b) {
                    const int B = v_so[static_cast<std::size_t>(b)];
                    const double denom = mp_denominator(eps_so(I), eps_so(J),
                                                        eps_so(A), eps_so(B));
                    t2[idx(i, j, a, b)] = antisym_eri(g, I, J, A, B) / denom;
                }
            }
        }
    }

    // u_ijab = 1/2 Σ_kl <ij||kl> t_klab
    //        + 1/2 Σ_cd <ab||cd> t_ijcd
    //        + P(ij)P(ab) Σ_kc <ak||ic> t_jkbc
    for (int i = 0; i < no; ++i) {
        const int I = o_so[static_cast<std::size_t>(i)];
        for (int j = 0; j < no; ++j) {
            const int J = o_so[static_cast<std::size_t>(j)];
            for (int a = 0; a < nv; ++a) {
                const int A = v_so[static_cast<std::size_t>(a)];
                for (int b = 0; b < nv; ++b) {
                    const int B = v_so[static_cast<std::size_t>(b)];
                    double val = 0.0;

                    for (int k = 0; k < no; ++k) {
                        const int K = o_so[static_cast<std::size_t>(k)];
                        for (int l = 0; l < no; ++l) {
                            const int L = o_so[static_cast<std::size_t>(l)];
                            val += 0.5 * antisym_eri(g, I, J, K, L)
                                   * t2[idx(k, l, a, b)];
                        }
                    }
                    for (int c = 0; c < nv; ++c) {
                        const int Cc = v_so[static_cast<std::size_t>(c)];
                        for (int d = 0; d < nv; ++d) {
                            const int Dd = v_so[static_cast<std::size_t>(d)];
                            val += 0.5 * antisym_eri(g, A, B, Cc, Dd)
                                   * t2[idx(i, j, c, d)];
                        }
                    }

                    // Ring with P(ij)P(ab): s_ijab - s_jiab - s_ijba + s_jiba
                    // s_xyzw = Σ_kc <X k || y C> t_z k w c  with X=vir, y=occ
                    auto ring_s = [&](int ii, int jj, int aa, int bb) {
                        const int II = o_so[static_cast<std::size_t>(ii)];
                        const int AA = v_so[static_cast<std::size_t>(aa)];
                        double s = 0.0;
                        for (int k = 0; k < no; ++k) {
                            const int K = o_so[static_cast<std::size_t>(k)];
                            for (int c = 0; c < nv; ++c) {
                                const int Cc =
                                    v_so[static_cast<std::size_t>(c)];
                                s += antisym_eri(g, AA, K, II, Cc)
                                     * t2[idx(jj, k, bb, c)];
                            }
                        }
                        return s;
                    };
                    val += ring_s(i, j, a, b) - ring_s(j, i, a, b)
                           - ring_s(i, j, b, a) + ring_s(j, i, b, a);

                    u2[idx(i, j, a, b)] = val;
                }
            }
        }
    }

    double e2 = 0.0;
    double e3 = 0.0;
    for (int i = 0; i < no; ++i) {
        const int I = o_so[static_cast<std::size_t>(i)];
        for (int j = 0; j < no; ++j) {
            const int J = o_so[static_cast<std::size_t>(j)];
            for (int a = 0; a < nv; ++a) {
                const int A = v_so[static_cast<std::size_t>(a)];
                for (int b = 0; b < nv; ++b) {
                    const int B = v_so[static_cast<std::size_t>(b)];
                    const double t = t2[idx(i, j, a, b)];
                    e2 += 0.25 * t * antisym_eri(g, I, J, A, B);
                    e3 += 0.25 * t * u2[idx(i, j, a, b)];
                }
            }
        }
    }

    const double e_corr = e2 + e3;
    return {e_corr, e_scf_ha + e_corr, e2, e3, n_frozen};
}

} // namespace scf
} // namespace cpu
} // namespace ATV
