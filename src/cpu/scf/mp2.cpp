/// @file cpu/scf/mp2.cpp
/// @brief Implementation of ao2mo_ovov + closed-shell / UMP2 / RI MP2.

#include "atomiverse/cpu/scf/mp2.h"
#include "atomiverse/core/log.h"
#include "atomiverse/cpu/linalg/internal/blas.h"
#include "atomiverse/cpu/scf/internal/ao2mo.h"
#include "atomiverse/cpu/scf/internal/density_fitting.h"
#include "atomiverse/cpu/scf/internal/exact_jk.h"
#include "atomiverse/cpu/scf/internal/mp2_pack.h"
#include "atomiverse/cpu/scf/internal/spin_orbital_eri.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/views/xview.hpp>

namespace ATV {
namespace cpu {
namespace scf {

namespace {

template <typename LoadRow, typename StoreRow = std::nullptr_t>
void transform_pair_rows(std::size_t rows, const xt::xtensor<double, 2>& Co,
                         const xt::xtensor<double, 2>& Cv, double* out,
                         LoadRow load_row, StoreRow store_row = nullptr) {
    const int n = static_cast<int>(Co.shape(0));
    const int no = static_cast<int>(Co.shape(1));
    const int nv = static_cast<int>(Cv.shape(1));
    const auto stride = static_cast<std::size_t>(no) * nv;
    constexpr bool pack_output = !std::is_same_v<StoreRow, std::nullptr_t>;
    using linalg::internal::gemm;
#ifdef _OPENMP
#pragma omp parallel if (rows >= 64)
#endif
    {
        auto ao =
            xt::xtensor<double, 2>::from_shape({Co.shape(0), Co.shape(0)});
        auto av =
            xt::xtensor<double, 2>::from_shape({Co.shape(0), Cv.shape(1)});
        auto product =
            xt::xtensor<double, 1>::from_shape({pack_output ? stride : 0});
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (std::size_t row = 0; row < rows; ++row) {
            load_row(row, ao.data());
            // The row's AO matrix is symmetric. In row-major storage the
            // two GEMMs form AO * Cv and then Co^T * (AO * Cv).
            gemm('N', 'N', nv, n, n, 1.0, Cv.data(), nv, ao.data(), n, 0.0,
                 av.data(), nv);
            double* dest = pack_output ? product.data() : out + row * stride;
            gemm('N', 'T', nv, no, n, 1.0, av.data(), nv, Co.data(), no, 0.0,
                 dest, nv);
            if constexpr (pack_output)
                store_row(row, dest);
        }
    }
}

void restore_pair_row(std::size_t n, const double* packed, double* ao) {
    for (std::size_t a = 0; a < n; ++a)
        for (std::size_t b = 0; b <= a; ++b)
            ao[a * n + b] = ao[b * n + a] = *packed++;
}

xt::xtensor<double, 4> transform_cached_ovov(const internal::ExactJK& cache,
                                             const xt::xtensor<double, 2>& Co,
                                             const xt::xtensor<double, 2>& Cv) {
    const auto n = Co.shape(0), no = Co.shape(1), nv = Cv.shape(1);
    const auto np = n * (n + 1) / 2;
    const auto nov = no * nv;
    const xt::xtensor<double, 2> half = [&] {
        auto pair_ov = xt::xtensor<double, 2>::from_shape({np, nov});
        transform_pair_rows(np, Co, Cv, pair_ov.data(),
                            [&](std::size_t pair, double* ao) {
                                cache.unpack_coulomb_cache_row(pair, ao);
                            });
        return xt::xtensor<double, 2>(xt::transpose(pair_ov));
    }();
    auto ovov = xt::xtensor<double, 4>::from_shape({no, nv, no, nv});
    transform_pair_rows(nov, Co, Cv, ovov.data(),
                        [&](std::size_t ov, double* ao) {
                            restore_pair_row(n, half.data() + ov * np, ao);
                        });
    return ovov;
}

/// @brief Four-index AO→MO transform to chemist (p q | r s) with independent
///        coefficient blocks per index (OrbitalHessian::transform_block_ pattern).
xt::xtensor<double, 4> ao2mo_mixed_(const xt::xtensor<double, 4>& eri_ao,
                                    const xt::xtensor<double, 2>& C0,
                                    const xt::xtensor<double, 2>& C1,
                                    const xt::xtensor<double, 2>& C2,
                                    const xt::xtensor<double, 2>& C3) {
    const xt::xtensor<double, 2>* Cs[4] = {&C3, &C2, &C1, &C0};
    // Round 0 reads eri_ao directly and writes a fresh work tensor; later
    // rounds iterate on `cur`. Dropping the old `cur = eri_ao` entry copy
    // saves one full N⁴ materialization per call (contraction order unchanged).
    xt::xtensor<double, 4> cur;
    for (int round = 0; round < 4; ++round) {
        const xt::xtensor<double, 4>& src = (round == 0) ? eri_ao : cur;
        const auto sh = src.shape();
        auto flat = xt::reshape_view(src, {sh[0] * sh[1] * sh[2], sh[3]});
        xt::xtensor<double, 2> h = xt::linalg::dot(flat, *Cs[round]);
        xt::xtensor<double, 4> next =
            xt::reshape_view(h, {sh[0], sh[1], sh[2], Cs[round]->shape()[1]});
        cur = xt::transpose(next, {3, 0, 1, 2});
    }
    return cur; // (C0, C1, C2, C3) → (i, a, j, b) when C0=CoL, C1=CvL, …
}

/// @brief Same-spin UMP2 channel: ¼ Σ | (ia|jb) − (ib|ja) |² / Δ.
double same_spin_corr_(const xt::xtensor<double, 4>& ovov,
                       const xt::xtensor<double, 1>& eps, int n_occ,
                       int n_vir) {
    if (n_occ < 2 || n_vir < 2)
        return 0.0;
    double e = 0.0;
    for (int i = 0; i < n_occ; ++i) {
        for (int j = 0; j < n_occ; ++j) {
            for (int a = 0; a < n_vir; ++a) {
                for (int b = 0; b < n_vir; ++b) {
                    const double iajb = ovov(i, a, j, b);
                    const double ibja = ovov(i, b, j, a);
                    const double t = iajb - ibja;
                    const double denom = mp_denominator(
                        eps(i), eps(j), eps(n_occ + a), eps(n_occ + b));
                    e += t * t / denom;
                }
            }
        }
    }
    return 0.25 * e;
}

// Transform with alternating coefficient blocks (A,B,A,B). Equal blocks
// give the full MO tensor, while occupied/virtual blocks give (ia|jb).
xt::xtensor<double, 4> ao2mo_pairs(const xt::xtensor<double, 4>& ERI,
                                   const xt::xtensor<double, 2>& first,
                                   const xt::xtensor<double, 2>& second) {
    static_assert(xt::xtensor<double, 4>::static_layout
                  == xt::layout_type::row_major);
    static_assert(xt::xtensor<double, 2>::static_layout
                  == xt::layout_type::row_major);
    const std::size_t n = ERI.shape(0), no = first.shape(1),
                      nv = second.shape(1);
    if (first.shape(0) != n || second.shape(0) != n
        || ERI.shape() != std::array{n, n, n, n})
        throw std::invalid_argument(
            "AO to MO transform: inconsistent AO dimensions");
    if (n == 0 || no == 0 || nv == 0)
        return xt::zeros<double>({no, nv, no, nv});

    constexpr auto blas_max =
        static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (n > blas_max / n / n || no > blas_max / n / n || no > blas_max / nv / n
        || no > blas_max / nv / no)
        throw std::overflow_error(
            "AO to MO transform: dimensions exceed the BLAS integer range");
    const int nao = static_cast<int>(n);
    const int first_mos = static_cast<int>(no);
    const int second_mos = static_cast<int>(nv);

    // BLAS sees a row-major buffer as its column-major transpose. Thus
    // (C^T X)^T = X^T C contracts a leading index without repacking X.
    using linalg::internal::gemm;
    xt::xtensor<double, 4> work;
    {
        // mu -> i, then bring nu to the front for the next contraction.
        const int width = static_cast<int>(n * n * n);
        auto quarter = xt::xtensor<double, 4>::from_shape({no, n, n, n});
        gemm('N', 'T', width, first_mos, nao, 1.0, ERI.data(), width,
             first.data(), first_mos, 0.0, quarter.data(), width);
        work = xt::transpose(quarter, {1, 0, 2, 3}); // (nu,i,lam,sig)
    }
    {
        // nu -> a, then bring lambda to the front in (lam,i,a,sig) order.
        const int width = static_cast<int>(no * n * n);
        auto quarter = xt::xtensor<double, 4>::from_shape({nv, no, n, n});
        gemm('N', 'T', width, second_mos, nao, 1.0, work.data(), width,
             second.data(), second_mos, 0.0, quarter.data(), width);
        work = xt::transpose(quarter, {2, 1, 0, 3}); // (lam,i,a,sig)
    }
    {
        // lambda -> j, then leave sigma as the trailing index.
        const int width = static_cast<int>(no * nv * n);
        auto quarter = xt::xtensor<double, 4>::from_shape({no, no, nv, n});
        gemm('N', 'T', width, first_mos, nao, 1.0, work.data(), width,
             first.data(), first_mos, 0.0, quarter.data(), width);
        work = xt::transpose(quarter, {1, 2, 0, 3}); // (i,a,j,sig)
    }

    // sigma -> b. The transposed product is Cv^T X^T.
    const int rows = static_cast<int>(no * nv * no);
    auto result = xt::xtensor<double, 4>::from_shape({no, nv, no, nv});
    gemm('N', 'N', second_mos, rows, nao, 1.0, second.data(), second_mos,
         work.data(), nao, 0.0, result.data(), second_mos);
    return result;
}

} // namespace

xt::xtensor<double, 4> ao2mo_ovov(const xt::xtensor<double, 4>& ERI,
                                  const xt::xtensor<double, 2>& C, int n_occ,
                                  int n_mo) {
    if (n_occ < 0 || n_mo < n_occ
        || static_cast<std::size_t>(n_mo) > C.shape(1))
        throw std::invalid_argument("ao2mo_ovov: inconsistent MO dimensions");
    const xt::xtensor<double, 2> Co =
        xt::view(C, xt::all(), xt::range(0, n_occ));
    const xt::xtensor<double, 2> Cv =
        xt::view(C, xt::all(), xt::range(n_occ, n_mo));
    return ao2mo_pairs(ERI, Co, Cv);
}

xt::xtensor<double, 4> ao2mo_full(const xt::xtensor<double, 4>& ERI,
                                  const xt::xtensor<double, 2>& C, int n_mo) {
    if (n_mo < 0 || static_cast<std::size_t>(n_mo) > C.shape(1))
        throw std::invalid_argument("ao2mo_full: inconsistent MO dimensions");
    if (static_cast<std::size_t>(n_mo) == C.shape(1))
        return ao2mo_pairs(ERI, C, C);
    const xt::xtensor<double, 2> Cm =
        xt::view(C, xt::all(), xt::range(0, n_mo));
    return ao2mo_pairs(ERI, Cm, Cm);
}

xt::xtensor<double, 4> ao2mo_full(const internal::ExactJK& cache,
                                  const xt::xtensor<double, 2>& C) {
    const auto n = C.shape(0), m = C.shape(1);
    const auto nao_pairs = n * (n + 1) / 2;
    std::vector<std::array<std::size_t, 2>> mo_pairs;
    mo_pairs.reserve(m * (m + 1) / 2);
    for (std::size_t p = 0; p < m; ++p)
        for (std::size_t q = 0; q <= p; ++q)
            mo_pairs.push_back({p, q});

    // Transform one symmetric pair at a time and keep only canonical pairs
    // in the half transform. Expand the second pair into the final MO tensor.
    const auto half = [&] {
        auto pair_mo =
            xt::xtensor<double, 2>::from_shape({nao_pairs, mo_pairs.size()});
        transform_pair_rows(
            nao_pairs, C, C, nullptr,
            [&](std::size_t row, double* ao) {
                cache.unpack_coulomb_cache_row(row, ao);
            },
            [&](std::size_t row, const double* mo) {
                auto* dest = pair_mo.data() + row * mo_pairs.size();
                for (const auto& [p, q] : mo_pairs)
                    *dest++ = mo[p * m + q];
            });
        return xt::xtensor<double, 2>(xt::transpose(pair_mo));
    }();
    auto result = xt::xtensor<double, 4>::from_shape({m, m, m, m});
    const auto stride = m * m;
    transform_pair_rows(
        mo_pairs.size(), C, C, nullptr,
        [&](std::size_t row, double* ao) {
            restore_pair_row(n, half.data() + row * nao_pairs, ao);
        },
        [&](std::size_t row, const double* mo) {
            const auto [p, q] = mo_pairs[row];
            std::copy_n(mo, stride, result.data() + (p * m + q) * stride);
            if (p != q)
                std::copy_n(mo, stride, result.data() + (q * m + p) * stride);
        });
    return result;
}

double coefficient_norm_squared(const xt::xtensor<double, 2>& C) {
    // The largest absolute row sum of C^T C bounds its spectral radius.
    const auto gram = xt::linalg::dot(xt::transpose(C), C);
    return xt::amax(xt::sum(xt::abs(gram), {1}))();
}

xt::xtensor<double, 2> mo_hcore(const xt::xtensor<double, 2>& Hcore,
                                const xt::xtensor<double, 2>& C, int n_mo) {
    // MO coefficient block, materialized [n_ao x n_mo].
    xt::xtensor<double, 2> Cm = xt::view(C, xt::all(), xt::range(0, n_mo));
    xt::xtensor<double, 2> tmp = xt::linalg::dot(Hcore, Cm); // [n_ao x n_mo]
    return xt::linalg::dot(xt::transpose(Cm), tmp);          // [n_mo x n_mo]
}

MP2Result mp2_energy(const internal::IntegralView& ints,
                     const SlaterDeterminant& det, double e_scf_ha,
                     int n_frozen, double os_scale, double ss_scale,
                     const internal::ExactJK* cached_jk) {
    using Spin = SlaterDeterminant::Spin;
    // No-fallback guards -- throw, never return a sentinel. (Convergence of
    // the reference is guarded by the caller, not here.)
    if (ints.ERI.size() == 0 && !cached_jk) {
        throw std::invalid_argument("mp2_energy: ERI is not materialized; call "
                                    "internal::IntegralWorkspace{..., "
                                    "compute_eri=true}.");
    }
    if (det.n_electrons(Spin::alpha) != det.n_electrons(Spin::beta)) {
        throw std::invalid_argument(
            "mp2_energy: closed-shell reference only (n_alpha must equal "
            "n_beta).");
    }

    // Row-major copy of the alpha MO coefficients (hoisted once).
    xt::xtensor<double, 2> C = det.coefficients(Spin::alpha);
    const int n_occ = det.n_electrons(Spin::alpha);
    const int n_mo = static_cast<int>(C.shape(1));
    const int n_vir = n_mo - n_occ;

    if (n_frozen < 0 || n_frozen > n_occ) {
        throw std::invalid_argument(
            "mp2_energy: n_frozen must satisfy 0 <= n_frozen <= n_occ.");
    }

    // No virtuals (e.g. He/STO-3G) or all occupied frozen: correlation is
    // exactly zero. Return a finite 0.0 rather than empty-denominator loops.
    if (n_vir == 0 || n_frozen == n_occ) {
        return {0.0, e_scf_ha, /*ri=*/false, n_frozen, 0.0, 0.0};
    }

    const int n_active = n_occ - n_frozen;
    if (n_frozen > 0)
        C = xt::eval(xt::view(C, xt::all(), xt::range(n_frozen, n_mo)));
    const xt::xtensor<double, 1>& eps = det.orbital_energies(Spin::alpha);

    // Spin-component split (matches PySCF mp2.kernel):
    //   E_OS = Σ (ia|jb)² / Δ
    //   E_SS = Σ (ia|jb)[(ia|jb) − (ib|ja)] / Δ
    //   E_MP2 = E_OS + E_SS = Σ (ia|jb)[2(ia|jb) − (ib|ja)] / Δ
    // SCS-MP2: c_OS * E_OS + c_SS * E_SS (Grimme JCP 118, 9095 (2003)).
    const auto evaluate = [&](const xt::xtensor<double, 4>& ovov) {
        double e_os = 0.0;
        double e_ss = 0.0;
        for (int i = 0; i < n_active; ++i) {
            for (int j = 0; j < n_active; ++j) {
                for (int a = 0; a < n_vir; ++a) {
                    for (int b = 0; b < n_vir; ++b) {
                        const double iajb = ovov(i, a, j, b);
                        const double ibja = ovov(i, b, j, a);
                        const double denom =
                            mp_denominator(eps(i + n_frozen), eps(j + n_frozen),
                                           eps(n_occ + a), eps(n_occ + b));
                        const double coul = iajb * iajb / denom;
                        e_os += coul;
                        e_ss += coul - iajb * ibja / denom;
                    }
                }
            }
        }
        const double e_corr = os_scale * e_os + ss_scale * e_ss;
        return MP2Result{
            e_corr, e_scf_ha + e_corr, /*ri=*/false, n_frozen, e_os, e_ss};
    };

    if (cached_jk && ints.ERI.size() == 0) {
        if (const auto schwarz = cached_jk->coulomb_cache_screening_bound()) {
            const xt::xtensor<double, 2> Co =
                xt::view(C, xt::all(), xt::range(0, n_active));
            const xt::xtensor<double, 2> Cv =
                xt::view(C, xt::all(), xt::range(n_active, n_active + n_vir));
            const auto start = LogClock::now();
            const auto result =
                evaluate(transform_cached_ovov(*cached_jk, Co, Cv));

            // Bound the Frobenius norm of the omitted Cartesian AO tensor by
            // n_cart² * schwarz, and propagate through each coefficient block.
            // For spherical bases the effective coefficients are c2s * C.
            const double occ_norm =
                ints.spherical()
                    ? coefficient_norm_squared(xt::linalg::dot(ints.c2s, Co))
                    : coefficient_norm_squared(Co);
            const double vir_norm =
                ints.spherical()
                    ? coefficient_norm_squared(xt::linalg::dot(ints.c2s, Cv))
                    : coefficient_norm_squared(Cv);
            const double n_cart = ints.n_ao_cart();
            const double e_occ =
                *std::max_element(eps.data() + n_frozen, eps.data() + n_occ);
            const double e_vir =
                *std::min_element(eps.data() + n_occ, eps.data() + n_mo);
            const double min_denom =
                -mp_denominator(e_occ, e_occ, e_vir, e_vir);
            const double delta = n_cart * n_cart * *schwarz * occ_norm
                                 * vir_norm / std::sqrt(min_denom);
            // In the denominator-weighted norm, OS = -||ovov||² and
            // SS = -2||antisymmetrize(ovov)||². Antisymmetrization is an
            // orthogonal projection, so its perturbation is no larger.
            const double os_error =
                delta * (2.0 * std::sqrt(-result.e_corr_os) + delta);
            const double error =
                os_error
                * std::max(2.0, std::abs(os_scale) + 2.0 * std::abs(ss_scale));
            // Reserve most of the 1e-6 Ha oracle tolerance for other errors.
            constexpr double screening_budget_ha = 1e-7;
            const bool reuse = error <= screening_budget_ha
                               && std::isfinite(result.e_corr)
                               && std::isfinite(result.e_corr_ss);
            ATV_DEBUG("[rmp2] rhf_cache={} screening_bound_ha={:.3e} ms={:.3f}",
                      reuse, error, elapsed_ms(start));
            if (reuse)
                return result;
        }
    }

    if (ints.ERI.size() != 0)
        return evaluate(ao2mo_ovov(ints.ERI, C, n_active, n_mo - n_frozen));
    return evaluate(
        ao2mo_ovov(ints.eri_working(), C, n_active, n_mo - n_frozen));
}

MP2Result ump2_energy(const internal::IntegralView& ints,
                      const SlaterDeterminant& det, double e_scf_ha,
                      double os_scale, double ss_scale) {
    using Spin = SlaterDeterminant::Spin;
    if (ints.ERI.size() == 0) {
        throw std::invalid_argument("ump2_energy: ERI is not materialized; "
                                    "call internal::IntegralWorkspace{..., "
                                    "compute_eri=true}.");
    }
    if (det.generalized()) {
        throw std::invalid_argument(
            "ump2_energy: GHF (spinor) reference is not supported.");
    }
    if (det.restricted()) {
        throw std::invalid_argument(
            "ump2_energy: restricted reference — use mp2_energy (RMP2).");
    }

    const xt::xtensor<double, 2>& Ca = det.coefficients(Spin::alpha);
    const xt::xtensor<double, 2>& Cb = det.coefficients(Spin::beta);
    const xt::xtensor<double, 1>& eps_a = det.orbital_energies(Spin::alpha);
    const xt::xtensor<double, 1>& eps_b = det.orbital_energies(Spin::beta);
    const int n_occ_a = det.n_electrons(Spin::alpha);
    const int n_occ_b = det.n_electrons(Spin::beta);
    const int n_mo_a = static_cast<int>(Ca.shape(1));
    const int n_mo_b = static_cast<int>(Cb.shape(1));
    const int n_vir_a = n_mo_a - n_occ_a;
    const int n_vir_b = n_mo_b - n_occ_b;

    if (n_vir_a < 0 || n_vir_b < 0) {
        throw std::invalid_argument(
            "ump2_energy: n_electrons exceeds MO count on a spin channel.");
    }
    // No doubles possible when either both same-spin channels lack space or
    // the opposite-spin product of virtuals/occs vanishes — still return finite 0.
    if ((n_occ_a < 2 || n_vir_a < 2) && (n_occ_b < 2 || n_vir_b < 2)
        && (n_occ_a == 0 || n_occ_b == 0 || n_vir_a == 0 || n_vir_b == 0)) {
        return {0.0, e_scf_ha, /*ri=*/false, /*frozen=*/0, 0.0, 0.0};
    }

    double e_ss = 0.0;
    double e_os = 0.0;

    // Same-spin αα / ββ.
    if (n_occ_a >= 2 && n_vir_a >= 2) {
        xt::xtensor<double, 2> Co =
            xt::view(Ca, xt::all(), xt::range(0, n_occ_a));
        xt::xtensor<double, 2> Cv =
            xt::view(Ca, xt::all(), xt::range(n_occ_a, n_mo_a));
        const xt::xtensor<double, 4> ovov =
            ao2mo_mixed_(ints.ERI, Co, Cv, Co, Cv);
        e_ss += same_spin_corr_(ovov, eps_a, n_occ_a, n_vir_a);
    }
    if (n_occ_b >= 2 && n_vir_b >= 2) {
        xt::xtensor<double, 2> Co =
            xt::view(Cb, xt::all(), xt::range(0, n_occ_b));
        xt::xtensor<double, 2> Cv =
            xt::view(Cb, xt::all(), xt::range(n_occ_b, n_mo_b));
        const xt::xtensor<double, 4> ovov =
            ao2mo_mixed_(ints.ERI, Co, Cv, Co, Cv);
        e_ss += same_spin_corr_(ovov, eps_b, n_occ_b, n_vir_b);
    }

    // Opposite-spin αβ: Σ (i_a a_a | j_b b_b)² / Δ  (no exchange).
    if (n_occ_a > 0 && n_occ_b > 0 && n_vir_a > 0 && n_vir_b > 0) {
        xt::xtensor<double, 2> Coa =
            xt::view(Ca, xt::all(), xt::range(0, n_occ_a));
        xt::xtensor<double, 2> Cva =
            xt::view(Ca, xt::all(), xt::range(n_occ_a, n_mo_a));
        xt::xtensor<double, 2> Cob =
            xt::view(Cb, xt::all(), xt::range(0, n_occ_b));
        xt::xtensor<double, 2> Cvb =
            xt::view(Cb, xt::all(), xt::range(n_occ_b, n_mo_b));
        const xt::xtensor<double, 4> ovov_ab =
            ao2mo_mixed_(ints.ERI, Coa, Cva, Cob, Cvb);
        for (int i = 0; i < n_occ_a; ++i) {
            for (int j = 0; j < n_occ_b; ++j) {
                for (int a = 0; a < n_vir_a; ++a) {
                    for (int b = 0; b < n_vir_b; ++b) {
                        const double v = ovov_ab(i, a, j, b);
                        const double denom = eps_a(i) + eps_b(j)
                                             - eps_a(n_occ_a + a)
                                             - eps_b(n_occ_b + b);
                        if (denom >= 0.0) {
                            throw std::invalid_argument(
                                "ump2_energy: non-negative opposite-spin "
                                "orbital-energy denominator; reference is "
                                "degenerate or non-aufbau.");
                        }
                        e_os += v * v / denom;
                    }
                }
            }
        }
    }

    const double e_corr = os_scale * e_os + ss_scale * e_ss;
    return {e_corr, e_scf_ha + e_corr, /*ri=*/false, /*frozen=*/0, e_os, e_ss};
}

MP2Result romp2_energy(const internal::IntegralView& ints,
                       const SlaterDeterminant& det, const RealMatrix& F_alpha,
                       const RealMatrix& F_beta, double e_scf_ha,
                       double os_scale, double ss_scale) {
    using Spin = SlaterDeterminant::Spin;
    if (ints.ERI.size() == 0) {
        throw std::invalid_argument("romp2_energy: ERI is not materialized; "
                                    "call internal::IntegralWorkspace{..., "
                                    "compute_eri=true}.");
    }
    if (det.generalized()) {
        throw std::invalid_argument(
            "romp2_energy: GHF (spinor) reference is not supported.");
    }

    // Shared spatial MOs (ROHF). Either restricted or unrestricted-form det
    // is accepted; occupations come from the per-spin electron counts.
    const RealMatrix& Cref = det.coefficients(Spin::alpha);
    const int n_ao = static_cast<int>(Cref.shape(0));
    const int n_mo = static_cast<int>(Cref.shape(1));
    const int n_occ_a = det.n_electrons(Spin::alpha);
    const int n_occ_b = det.n_electrons(Spin::beta);

    if (static_cast<int>(F_alpha.shape(0)) != n_ao
        || static_cast<int>(F_alpha.shape(1)) != n_ao
        || static_cast<int>(F_beta.shape(0)) != n_ao
        || static_cast<int>(F_beta.shape(1)) != n_ao) {
        throw std::invalid_argument(
            "romp2_energy: spin Fock matrices must be n_ao × n_ao matching the "
            "MO coefficient row dimension.");
    }
    if (n_occ_a < 0 || n_occ_b < 0 || n_occ_a > n_mo || n_occ_b > n_mo) {
        throw std::invalid_argument(
            "romp2_energy: electron counts out of range for the MO space.");
    }

    // Semi-canonical denominators (PySCF ROHF.canonicalize / to_uhf):
    //   ε_i^σ = (C† F_σ C)_ii
    // Roothaan F_eff eigenvalues on the det are intentionally NOT used.
    const RealMatrix Fmo_a =
        xt::linalg::dot(xt::transpose(Cref), xt::linalg::dot(F_alpha, Cref));
    const RealMatrix Fmo_b =
        xt::linalg::dot(xt::transpose(Cref), xt::linalg::dot(F_beta, Cref));
    RealVector eps_a = xt::zeros<double>({static_cast<std::size_t>(n_mo)});
    RealVector eps_b = xt::zeros<double>({static_cast<std::size_t>(n_mo)});
    for (int i = 0; i < n_mo; ++i) {
        eps_a(i) = Fmo_a(i, i);
        eps_b(i) = Fmo_b(i, i);
    }

    RealVector occ_a = xt::zeros<double>({static_cast<std::size_t>(n_mo)});
    RealVector occ_b = xt::zeros<double>({static_cast<std::size_t>(n_mo)});
    for (int i = 0; i < n_occ_a; ++i)
        occ_a(i) = 1.0;
    for (int i = 0; i < n_occ_b; ++i)
        occ_b(i) = 1.0;

    // Independent α/β channels with identical spatial C (UHF-shaped view of
    // the ROHF solution) for the shared ump2_energy kernel.
    SlaterDeterminant udet(Cref, std::move(eps_a), std::move(occ_a), n_occ_a,
                           Cref, std::move(eps_b), std::move(occ_b), n_occ_b);
    return ump2_energy(ints, udet, e_scf_ha, os_scale, ss_scale);
}

MP2Result ri_mp2_energy(const internal::IntegralView& ints,
                        const SlaterDeterminant& det, double e_scf_ha,
                        AuxBasis aux, int n_frozen, double os_scale,
                        double ss_scale) {
    using Spin = SlaterDeterminant::Spin;
    if (det.n_electrons(Spin::alpha) != det.n_electrons(Spin::beta))
        throw std::invalid_argument(
            "ri_mp2_energy: closed-shell reference only (n_alpha must equal "
            "n_beta).");
    if (ints.shells.empty())
        throw std::invalid_argument(
            "ri_mp2_energy: internal::IntegralWorkspace has no shells for "
            "density fitting.");

    const RealMatrix& C_rm = det.coefficients(Spin::alpha);
    // Working AO coefficients may be spherical; DensityFitting lives in
    // Cartesian engine AO. Bridge with pack.c2s when present.
    const int n_occ = det.n_electrons(Spin::alpha);
    const int n_mo = static_cast<int>(C_rm.shape()[1]);
    const int n_vir = n_mo - n_occ;
    if (n_frozen < 0 || n_frozen > n_occ)
        throw std::invalid_argument(
            "ri_mp2_energy: n_frozen must satisfy 0 <= n_frozen <= n_occ.");
    if (n_vir == 0 || n_frozen == n_occ)
        return {0.0, e_scf_ha, /*ri=*/true, n_frozen, 0.0, 0.0};

    internal::DensityFitting df(ints.shells, ints.atoms, aux);
    const int n_cart = df.n_ao();
    const int n_aux = df.n_aux();
    const RealMatrix& B = df.b_tensor(); // (n_cart² × n_aux), column-major

    RealMatrix C_cart;
    if (ints.spherical()) {
        C_cart = xt::linalg::dot(ints.c2s, C_rm); // [n_cart × n_mo]
    } else {
        if (static_cast<int>(C_rm.shape()[0]) != n_cart)
            throw std::invalid_argument(
                "ri_mp2_energy: AO dimension mismatch with DensityFitting");
        C_cart = C_rm;
    }

    // B_ov(ia, P) = Σ_μν C_μi C_νa B_μν^P, assembled BLAS-shaped from the
    // frozen df.b_tensor() output. B is column-major (n_cart² × n_aux) with the
    // (μ·n_cart + ν) pair index on the rows, so each auxiliary column is one
    // contiguous n_cart² slice: the M_P matrix in row-major (μ,ν). Adapt that
    // slice as a view (no per-P element-wise M reconstruction) and half-
    // transform it with Co^T M_P Cv. The fitting metric (Coulomb V=(P|Q),
    // Cholesky-whitened W=L⁻¹ ⇒ V⁻¹) is frozen — B is consumed unchanged.
    const int nov = n_occ * n_vir;
    const RealMatrix Co =
        xt::view(C_cart, xt::all(), xt::range(0, n_occ)); // [n_cart × n_occ]
    const RealMatrix Cv =
        xt::view(C_cart, xt::all(), xt::range(n_occ, n_mo)); // [n_cart × n_vir]
    const std::size_t n_cart2 =
        static_cast<std::size_t>(n_cart) * static_cast<std::size_t>(n_cart);
    xt::xtensor<double, 2> B_ov = xt::zeros<double>(
        {static_cast<std::size_t>(nov), static_cast<std::size_t>(n_aux)});
    for (int P = 0; P < n_aux; ++P) {
        // Row-major (μ,ν) view of column P's contiguous n_cart² slice.
        auto M_P = xt::adapt(
            B.data() + static_cast<std::size_t>(P) * n_cart2, n_cart2,
            xt::no_ownership(),
            std::array<std::size_t, 2>{static_cast<std::size_t>(n_cart),
                                       static_cast<std::size_t>(n_cart)});
        const RealMatrix ia = xt::linalg::dot(
            xt::transpose(Co), xt::linalg::dot(M_P, Cv)); // [n_occ × n_vir]
        for (int i = 0; i < n_occ; ++i)
            for (int a = 0; a < n_vir; ++a)
                B_ov(static_cast<std::size_t>(i * n_vir + a),
                     static_cast<std::size_t>(P)) = ia(i, a);
    }

    // (ia|jb) ≈ Σ_P B_ov(ia,P) B_ov(jb,P) — one dgemm over B_ov replaces the
    // scalar inner-P accumulation; OS/SS split + frozen-core logic unchanged.
    const xt::xtensor<double, 2> iajb_table =
        xt::linalg::dot(B_ov, xt::transpose(B_ov)); // [nov × nov]

    const RealVector& eps = det.orbital_energies(Spin::alpha);
    double e_os = 0.0;
    double e_ss = 0.0;
    for (int i = n_frozen; i < n_occ; ++i) {
        for (int j = n_frozen; j < n_occ; ++j) {
            for (int a = 0; a < n_vir; ++a) {
                for (int b = 0; b < n_vir; ++b) {
                    const int ia = i * n_vir + a;
                    const int jb = j * n_vir + b;
                    const int ib = i * n_vir + b;
                    const int ja = j * n_vir + a;
                    const double iajb = iajb_table(ia, jb);
                    const double ibja = iajb_table(ib, ja);
                    const double denom = mp_denominator(
                        eps(i), eps(j), eps(n_occ + a), eps(n_occ + b));
                    const double coul = iajb * iajb / denom;
                    e_os += coul;
                    e_ss += coul - iajb * ibja / denom;
                }
            }
        }
    }
    const double e_corr = os_scale * e_os + ss_scale * e_ss;
    return {e_corr, e_scf_ha + e_corr, /*ri=*/true, n_frozen, e_os, e_ss};
}

} // namespace scf
} // namespace cpu
} // namespace ATV
