/// @file cpu/scf/mp4.cpp
/// @brief Closed-shell canonical MP4(SDTQ) (spin-orbital residual form).

#include "atomiverse/cpu/scf/mp4.h"
#include "atomiverse/cpu/scf/internal/ao2mo.h"
#include "atomiverse/cpu/scf/internal/mp4_pack.h"
#include "atomiverse/cpu/scf/internal/spin_orbital_eri.h"

#include <cmath>
#include <stdexcept>
#include <vector>

#include <xtensor/containers/xtensor.hpp>

namespace ATV {
namespace cpu {
namespace scf {

MP4Result mp4_energy(const internal::IntegralView& ints,
                     const SlaterDeterminant& det, double e_scf_ha,
                     int n_frozen) {
    using Spin = SlaterDeterminant::Spin;
    if (ints.ERI.size() == 0) {
        throw std::invalid_argument("mp4_energy: ERI is not materialized; call "
                                    "internal::IntegralWorkspace{..., "
                                    "compute_eri=true}.");
    }
    if (det.n_electrons(Spin::alpha) != det.n_electrons(Spin::beta)) {
        throw std::invalid_argument(
            "mp4_energy: closed-shell reference only (n_alpha must equal "
            "n_beta).");
    }
    if (det.generalized()) {
        throw std::invalid_argument(
            "mp4_energy: GHF (spinor) reference is not supported.");
    }

    const xt::xtensor<double, 2> C = det.coefficients(Spin::alpha);
    const int n_occ = det.n_electrons(Spin::alpha);
    const int n_mo = static_cast<int>(C.shape(1));
    const int n_vir = n_mo - n_occ;

    if (n_frozen < 0 || n_frozen > n_occ) {
        throw std::invalid_argument(
            "mp4_energy: n_frozen must satisfy 0 <= n_frozen <= n_occ.");
    }
    if (n_vir == 0 || n_frozen == n_occ) {
        return {/*e_corr=*/0.0,
                /*e_tot=*/e_scf_ha,
                /*e_corr_mp2=*/0.0,
                /*e_corr_mp3=*/0.0,
                /*e_corr_mp4=*/0.0,
                /*e_corr_s=*/0.0,
                /*e_corr_d=*/0.0,
                /*e_corr_t=*/0.0,
                /*e_corr_q=*/0.0,   n_frozen};
    }

    // Full MO ERI in chemist notation (pq|rs).
    const xt::xtensor<double, 4> g = ao2mo_full(ints.ERI, C, n_mo);
    const xt::xtensor<double, 1>& eps = det.orbital_energies(Spin::alpha);

    // Correlated spin-orbitals: skip 2*n_frozen core SO.
    const int n_occ_so_full = 2 * n_occ;
    const int n_so = 2 * n_mo;
    const int n_fr_so = 2 * n_frozen;
    const int n_occ_so = n_occ_so_full - n_fr_so;
    const int n_vir_so = n_so - n_occ_so_full;

    std::vector<int> o_so(static_cast<std::size_t>(n_occ_so));
    std::vector<int> v_so(static_cast<std::size_t>(n_vir_so));
    for (int i = 0; i < n_occ_so; ++i)
        o_so[static_cast<std::size_t>(i)] = n_fr_so + i;
    for (int a = 0; a < n_vir_so; ++a)
        v_so[static_cast<std::size_t>(a)] = n_occ_so_full + a;

    auto eps_so = [&](int so) -> double {
        return eps(SpinOrbital::from_packed(so).spat);
    };

    const int no = n_occ_so;
    const int nv = n_vir_so;
    const std::size_t t_size =
        static_cast<std::size_t>(no) * static_cast<std::size_t>(no)
        * static_cast<std::size_t>(nv) * static_cast<std::size_t>(nv);
    std::vector<double> t2(t_size, 0.0);
    std::vector<double> u2(t_size, 0.0);
    std::vector<double> D2(t_size, 0.0);
    auto idx = [no, nv](int i, int j, int a, int b) -> std::size_t {
        return ((static_cast<std::size_t>(i) * static_cast<std::size_t>(no)
                 + static_cast<std::size_t>(j))
                    * static_cast<std::size_t>(nv)
                + static_cast<std::size_t>(a))
                   * static_cast<std::size_t>(nv)
               + static_cast<std::size_t>(b);
    };

    // First-order doubles t2 and denominators D2.
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
                    const std::size_t k = idx(i, j, a, b);
                    D2[k] = denom;
                    t2[k] = antisym_eri(g, I, J, A, B) / denom;
                }
            }
        }
    }

    // u_ijab = ladder + ring residual (same as MP3).
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
    double t2_norm = 0.0; // 1/4 sum t2^2 = ||Ψ^(1)||^2
    double e4_d_conn = 0.0;
    for (int i = 0; i < no; ++i) {
        const int I = o_so[static_cast<std::size_t>(i)];
        for (int j = 0; j < no; ++j) {
            const int J = o_so[static_cast<std::size_t>(j)];
            for (int a = 0; a < nv; ++a) {
                const int A = v_so[static_cast<std::size_t>(a)];
                for (int b = 0; b < nv; ++b) {
                    const int B = v_so[static_cast<std::size_t>(b)];
                    const std::size_t k = idx(i, j, a, b);
                    const double t = t2[k];
                    const double u = u2[k];
                    const double d = D2[k];
                    e2 += 0.25 * t * antisym_eri(g, I, J, A, B);
                    e3 += 0.25 * t * u;
                    t2_norm += 0.25 * t * t;
                    // Second-order doubles t2^(2) = u/D; connected DD piece
                    // of E^(4): (1/4) sum D (t2^(2))^2 = (1/4) sum u^2 / D
                    e4_d_conn += 0.25 * (u * u) / d;
                }
            }
        }
    }

    // E_Q^(4) renormalization (disconnected): -E^(2) ||Ψ^(1)||^2
    const double e4_q = -e2 * t2_norm;
    // E_D^(4) connected residual channel
    const double e4_d = e4_d_conn;

    // ── E_S^(4): second-order singles ────────────────────────────────
    // X_ia = 1/2 Σ_jbc <ja||bc> t_ijbc − 1/2 Σ_jkb <jk||ib> t_jkab
    // E_S^(4) = Σ_ia X_ia^2 / D_ia ,  D_ia = ε_i − ε_a
    double e4_s = 0.0;
    for (int i = 0; i < no; ++i) {
        const int I = o_so[static_cast<std::size_t>(i)];
        for (int a = 0; a < nv; ++a) {
            const int A = v_so[static_cast<std::size_t>(a)];
            double X = 0.0;
            for (int j = 0; j < no; ++j) {
                const int J = o_so[static_cast<std::size_t>(j)];
                for (int b = 0; b < nv; ++b) {
                    const int Bb = v_so[static_cast<std::size_t>(b)];
                    for (int c = 0; c < nv; ++c) {
                        const int Cc = v_so[static_cast<std::size_t>(c)];
                        X += 0.5 * antisym_eri(g, J, A, Bb, Cc)
                             * t2[idx(i, j, b, c)];
                    }
                }
            }
            for (int j = 0; j < no; ++j) {
                const int J = o_so[static_cast<std::size_t>(j)];
                for (int k = 0; k < no; ++k) {
                    const int K = o_so[static_cast<std::size_t>(k)];
                    for (int b = 0; b < nv; ++b) {
                        const int Bb = v_so[static_cast<std::size_t>(b)];
                        X -= 0.5 * antisym_eri(g, J, K, I, Bb)
                             * t2[idx(j, k, a, b)];
                    }
                }
            }
            const double d1 = eps_so(I) - eps_so(A);
            if (d1 >= 0.0) {
                throw std::invalid_argument(
                    "mp4_energy: non-negative singles orbital-energy "
                    "denominator; reference is degenerate or non-aufbau.");
            }
            e4_s += (X * X) / d1;
        }
    }

    // ── E_T^(4): second-order triples ────────────────────────────────
    // W3_ijkabc = P(i/jk) P(a/bc) [ Σ_e <bc||ek> t_ij^ae
    //                             − Σ_m <mc||jk> t_im^ab ]
    // E_T^(4) = (1/36) Σ W3^2 / D_ijkabc
    //
    // With antisym storage over all index orderings the 1/36 factor accounts
    // for overcounting (3! × 3!). For no < 3 or nv < 3 the sum is identically
    // zero (H₂/STO-3G).
    double e4_t = 0.0;
    if (no >= 3 && nv >= 3) {
        for (int i = 0; i < no; ++i) {
            const int I = o_so[static_cast<std::size_t>(i)];
            for (int j = 0; j < no; ++j) {
                const int J = o_so[static_cast<std::size_t>(j)];
                for (int k = 0; k < no; ++k) {
                    const int K = o_so[static_cast<std::size_t>(k)];
                    for (int a = 0; a < nv; ++a) {
                        const int A = v_so[static_cast<std::size_t>(a)];
                        for (int b = 0; b < nv; ++b) {
                            const int B = v_so[static_cast<std::size_t>(b)];
                            for (int c = 0; c < nv; ++c) {
                                const int C = v_so[static_cast<std::size_t>(c)];
                                // Build W3 with explicit P(i/jk)P(a/bc):
                                // P(i/jk) X_ijk = X_ijk - X_jik - X_kji
                                // (three terms; full antisym uses all 6 but
                                // the standard compact form uses this).
                                auto base = [&](int ii, int jj, int kk, int aa,
                                                int bb, int cc) -> double {
                                    const int II =
                                        o_so[static_cast<std::size_t>(ii)];
                                    const int JJ =
                                        o_so[static_cast<std::size_t>(jj)];
                                    const int KK =
                                        o_so[static_cast<std::size_t>(kk)];
                                    const int AA =
                                        v_so[static_cast<std::size_t>(aa)];
                                    const int BB =
                                        v_so[static_cast<std::size_t>(bb)];
                                    const int CC =
                                        v_so[static_cast<std::size_t>(cc)];
                                    double s = 0.0;
                                    for (int e = 0; e < nv; ++e) {
                                        const int Ee =
                                            v_so[static_cast<std::size_t>(e)];
                                        s += antisym_eri(g, BB, CC, Ee, KK)
                                             * t2[idx(ii, jj, aa, e)];
                                    }
                                    for (int m = 0; m < no; ++m) {
                                        const int Mm =
                                            o_so[static_cast<std::size_t>(m)];
                                        s -= antisym_eri(g, Mm, CC, JJ, KK)
                                             * t2[idx(ii, m, aa, bb)];
                                    }
                                    (void)II;
                                    (void)AA;
                                    return s;
                                };
                                // P(i/jk) P(a/bc) base:
                                // P(p/qr) f = f(pqr) - f(qpr) - f(rqp)
                                auto Pocc = [&](int ii, int jj, int kk, int aa,
                                                int bb, int cc) {
                                    return base(ii, jj, kk, aa, bb, cc)
                                           - base(jj, ii, kk, aa, bb, cc)
                                           - base(kk, jj, ii, aa, bb, cc);
                                };
                                const double W = Pocc(i, j, k, a, b, c)
                                                 - Pocc(i, j, k, b, a, c)
                                                 - Pocc(i, j, k, c, b, a);
                                const double denom = eps_so(I) + eps_so(J)
                                                     + eps_so(K) - eps_so(A)
                                                     - eps_so(B) - eps_so(C);
                                if (std::fabs(W) < 1e-18)
                                    continue;
                                if (denom >= 0.0) {
                                    throw std::invalid_argument(
                                        "mp4_energy: non-negative triples "
                                        "orbital-energy denominator.");
                                }
                                e4_t += W * W / denom;
                            }
                        }
                    }
                }
            }
        }
        e4_t /= 36.0;
    }

    const double e4 = e4_s + e4_d + e4_t + e4_q;
    const double e_corr = e2 + e3 + e4;
    return {e_corr, e_scf_ha + e_corr, e2, e3, e4, e4_s, e4_d, e4_t,
            e4_q,   n_frozen};
}

} // namespace scf
} // namespace cpu
} // namespace ATV
