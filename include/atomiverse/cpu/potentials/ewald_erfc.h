/// @file cpu/potentials/ewald_erfc.h
/// @brief Real-space Ewald pair factors erfc(αr)/r and its force factor,
///        evaluated analytically with portable xsimd batches.
///
/// erfc is a vectorized port of FreeBSD msun `s_erf.c` (fdlibm) `erfc`,
/// via the Rust `libm` 0.2.16 transcription `src/math/erf.rs`; coefficients
/// and interval boundaries are unchanged, |error| ≲ 1 ulp:
///
///     Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
///     Developed at SunPro, a Sun Microsystems, Inc. business.
///     Permission to use, copy, modify, and distribute this
///     software is freely granted, provided that this notice
///     is preserved.
///
/// Changes from fdlibm: only x ≥ 0 is handled (x = αr); the asymptotic
/// branches multiply by an e^{−x²} that is evaluated once and shared with
/// the force factor, exp(−x²) = exp(−z²)·exp((z−x)(z+x)) with z = x
/// truncated to its high word (fdlibm's own split, so z² is exact and the
/// argument carries no rounding error); the four intervals are selected
/// per lane, a branch skipped when no lane of the batch needs it.
///
/// This replaces the linear-in-r² `ErfcTable` (relative error ≈ 4e-6 on a
/// partially cancelling real-space sum) and matches the CUDA kernels'
/// erfcx(x)·e^{−x²} to rounding.
#pragma once

#include <bit>
#include <cstdint>

#include <xsimd/xsimd.hpp>

namespace ATV {
namespace cpu {
namespace md {

/// @brief erfc(x) and e^{−x²} of one batch, x ≥ 0 (dimensionless).
/// @param x     Arguments, ≥ 0, finite.
/// @param inv_x 1/x (any value where x < 1.25); the caller usually has it.
/// @param[out] erfc_x erfc(x).
/// @param[out] exp_m  e^{−x²}.
template <class Batch>
inline void ewald_erfc_exp(const Batch& x, const Batch& inv_x, Batch& erfc_x,
                           Batch& exp_m) {
    using B = Batch;
    // fdlibm coefficients (s_erf.c).
    constexpr double erx = 8.45062911510467529297e-01;
    constexpr double pp0 = 1.28379167095512558561e-01,
                     pp1 = -3.25042107247001499370e-01,
                     pp2 = -2.84817495755985104766e-02,
                     pp3 = -5.77027029648944159157e-03,
                     pp4 = -2.37630166566501626084e-05;
    constexpr double qq1 = 3.97917223959155352819e-01,
                     qq2 = 6.50222499887672944485e-02,
                     qq3 = 5.08130628187576562776e-03,
                     qq4 = 1.32494738004321644526e-04,
                     qq5 = -3.96022827877536812320e-06;
    constexpr double pa0 = -2.36211856075265944077e-03,
                     pa1 = 4.14856118683748331666e-01,
                     pa2 = -3.72207876035701323847e-01,
                     pa3 = 3.18346619901161753674e-01,
                     pa4 = -1.10894694282396677476e-01,
                     pa5 = 3.54783043256182359371e-02,
                     pa6 = -2.16637559486879084300e-03;
    constexpr double qa1 = 1.06420880400844228286e-01,
                     qa2 = 5.40397917702171048937e-01,
                     qa3 = 7.18286544141962662868e-02,
                     qa4 = 1.26171219808761642112e-01,
                     qa5 = 1.36370839120290507362e-02,
                     qa6 = 1.19844998467991074170e-02;
    constexpr double ra0 = -9.86494403484714822705e-03,
                     ra1 = -6.93858572707181764372e-01,
                     ra2 = -1.05586262253232909814e+01,
                     ra3 = -6.23753324503260060396e+01,
                     ra4 = -1.62396669462573470355e+02,
                     ra5 = -1.84605092906711035994e+02,
                     ra6 = -8.12874355063065934246e+01,
                     ra7 = -9.81432934416914548592e+00;
    constexpr double sa1 = 1.96512716674392571292e+01,
                     sa2 = 1.37657754143519042600e+02,
                     sa3 = 4.34565877475229228821e+02,
                     sa4 = 6.45387271733267880336e+02,
                     sa5 = 4.29008140027567833386e+02,
                     sa6 = 1.08635005541779435134e+02,
                     sa7 = 6.57024977031928170135e+00,
                     sa8 = -6.04244152148580987438e-02;
    constexpr double rb0 = -9.86494292470009928597e-03,
                     rb1 = -7.99283237680523006574e-01,
                     rb2 = -1.77579549177547519889e+01,
                     rb3 = -1.60636384855821916062e+02,
                     rb4 = -6.37566443368389627722e+02,
                     rb5 = -1.02509513161107724954e+03,
                     rb6 = -4.83519191608651397019e+02;
    constexpr double sb1 = 3.03380607434824582924e+01,
                     sb2 = 3.25792512996573918826e+02,
                     sb3 = 1.53672958608443695994e+03,
                     sb4 = 3.19985821950859553908e+03,
                     sb5 = 2.55305040643316442583e+03,
                     sb6 = 4.74528541206955367215e+02,
                     sb7 = -2.24409524465858183362e+01;
    // fdlibm compares the high word: 0x3feb0000 (0.84375), 0x3fd00000
    // (0.25), 0x3ff40000 (1.25) and 0x4006db6d (≈ 1/0.35).
    const double lim_near = 0.84375;
    const double lim_quarter = 0.25;
    const double lim_one = 1.25;
    const double lim_far = std::bit_cast<double>(0x4006db6d00000000ULL);
    const B high_word_mask(std::bit_cast<double>(0xffffffff00000000ULL));

    // e^{−x²} with an exact z² (z = x with the low word cleared) and the
    // small remainder d = (z − x)(z + x), |d| < 2^-19 x², by its series.
    const B z = xsimd::bitwise_and(x, high_word_mask);
    const B d = (z - x) * (z + x);
    const B ez = xsimd::exp(-(z * z));
    exp_m = ez * (B(1.0) + d * (B(1.0) + d * (B(0.5) + d * B(1.0 / 6.0))));

    B out(0.0);
    const auto in_near = x < B(lim_near);
    const auto in_mid = (x >= B(lim_near)) && (x < B(lim_one));
    const auto in_far = x >= B(lim_one);
    if (xsimd::any(in_near)) {
        const B zz = x * x;
        const B r = pp0 + zz * (pp1 + zz * (pp2 + zz * (pp3 + zz * pp4)));
        const B s =
            1.0 + zz * (qq1 + zz * (qq2 + zz * (qq3 + zz * (qq4 + zz * qq5))));
        const B y = r / s;
        const B small = B(1.0) - (x + x * y);
        const B large = B(0.5) - (x - B(0.5) + x * y);
        out = xsimd::select(
            in_near, xsimd::select(x < B(lim_quarter), small, large), out);
    }
    if (xsimd::any(in_mid)) {
        const B s = x - B(1.0);
        const B p =
            pa0
            + s
                  * (pa1
                     + s * (pa2 + s * (pa3 + s * (pa4 + s * (pa5 + s * pa6)))));
        const B q =
            1.0
            + s
                  * (qa1
                     + s * (qa2 + s * (qa3 + s * (qa4 + s * (qa5 + s * qa6)))));
        out = xsimd::select(in_mid, B(1.0 - erx) - p / q, out);
    }
    if (xsimd::any(in_far)) {
        // fdlibm s = 1/x²; unused lanes (x < 1.25, possibly 0) read 1/2.
        const B inv_xs = xsimd::select(in_far, inv_x, B(0.5));
        const B s = inv_xs * inv_xs;
        const auto inner = x < B(lim_far);
        B r(0.0), big_s(1.0);
        if (xsimd::any(inner && in_far)) {
            const B ra =
                ra0
                + s
                      * (ra1
                         + s
                               * (ra2
                                  + s
                                        * (ra3
                                           + s
                                                 * (ra4
                                                    + s
                                                          * (ra5
                                                             + s
                                                                   * (ra6
                                                                      + s * ra7))))));
            const B sa =
                1.0
                + s
                      * (sa1
                         + s
                               * (sa2
                                  + s
                                        * (sa3
                                           + s
                                                 * (sa4
                                                    + s
                                                          * (sa5
                                                             + s
                                                                   * (sa6
                                                                      + s
                                                                            * (sa7
                                                                               + s * sa8)))))));
            r = xsimd::select(inner, ra, r);
            big_s = xsimd::select(inner, sa, big_s);
        }
        if (xsimd::any(!inner && in_far)) {
            const B rb =
                rb0
                + s
                      * (rb1
                         + s
                               * (rb2
                                  + s
                                        * (rb3
                                           + s * (rb4 + s * (rb5 + s * rb6)))));
            const B sb =
                1.0
                + s
                      * (sb1
                         + s
                               * (sb2
                                  + s
                                        * (sb3
                                           + s
                                                 * (sb4
                                                    + s
                                                          * (sb5
                                                             + s
                                                                   * (sb6
                                                                      + s * sb7))))));
            r = xsimd::select(inner, r, rb);
            big_s = xsimd::select(inner, big_s, sb);
        }
        // fdlibm: exp(−x² − 0.5625 + R/S) / x.
        // fdlibm g = R/S − 0.5625 = log(x·erfc(x)) + x² lies in
        // [−0.777, −0.572] for every x ≥ 1.25, so exp(g) is a degree-11
        // Taylor series about t0 = −691/1024 (|g − t0| < 0.103, truncation
        // < 2e-21 relative); coefficients e^{t0}/k!.
        const B u = (r / big_s - B(0.5625)) - B(-0.6748046875);
        const B exp_g =
            0.5092558749329694
            + u
                  * (0.5092558749329694
                     + u
                           * (0.2546279374664847
                              + u
                                    * (0.0848759791554949
                                       + u
                                             * (0.021218994788873725
                                                + u
                                                      * (0.004243798957774745
                                                         + u
                                                               * (0.0007072998262957908
                                                                  + u
                                                                        * (0.00010104283232797011
                                                                           + u
                                                                                 * (1.2630354040996264e-05
                                                                                    + u
                                                                                          * (1.403372671221807e-06
                                                                                             + u
                                                                                                   * (1.4033726712218073e-07
                                                                                                      + u * 1.2757933374743702e-08))))))))));
        const B tail = exp_m * exp_g * inv_xs;
        out = xsimd::select(in_far, tail, out);
    }
    erfc_x = out;
}

/// @brief Real-space Ewald factors of n pair distances.
///
/// e[k] = erfc(α r_k)/r_k [1/Å] and f[k] = (erfc(α r_k) + (2/√π) α r_k
/// e^{−α² r_k²}) / r_k³ [1/Å³], so that, with qq = k_e q_i q_j, the pair
/// energy is qq·e[k] and the force on i is qq·f[k]·(r_i − r_j) (LAMMPS
/// `fpair` convention).
///
/// @param r2    Squared distances [Å²], > 0, readable up to the next
///              multiple of the batch width (`ewald_batch_width()`).
/// @param n     Pair count.
/// @param alpha Ewald α [1/Å].
/// @param[out] e,f Same padded length as @p r2; padding lanes are written.
inline void ewald_pair_factors(const double* r2, int n, double alpha, double* e,
                               double* f) {
    using B = xsimd::batch<double>;
    constexpr int W = static_cast<int>(B::size);
    constexpr double ewald_f = 1.1283791670955126; // 2/√π
    const B a(alpha), inv_a(1.0 / alpha);
    for (int k = 0; k < n; k += W) {
        const B rsq = B::load_unaligned(r2 + k);
        const B r = xsimd::sqrt(rsq);
        const B inv_r = B(1.0) / r;
        const B x = a * r;
        B erfc_x, exp_m;
        ewald_erfc_exp(x, inv_r * inv_a, erfc_x, exp_m);
        const B ev = erfc_x * inv_r;
        const B fv =
            (erfc_x + B(ewald_f) * x * exp_m) * (inv_r * inv_r * inv_r);
        ev.store_unaligned(e + k);
        fv.store_unaligned(f + k);
    }
}

/// @brief Lanes per batch of ewald_pair_factors (pad arrays to a multiple).
/// @return The xsimd double batch width of this build (count).
constexpr int ewald_batch_width() {
    return static_cast<int>(xsimd::batch<double>::size);
}

} // namespace md
} // namespace cpu
} // namespace ATV
