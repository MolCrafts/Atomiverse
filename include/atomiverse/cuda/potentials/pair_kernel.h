/// @file cuda/potentials/pair_kernel.h
/// @brief Backend-agnostic, potential-neutral pairwise force/energy kernels.
///
/// force-infra-03: the per-pair force *math* lives here, ONCE, decoupled from
/// the neighbor traversal/reduction (which stays per-backend — CPU xsimd
/// ket-loop, CUDA Verlet kernel). A `PairKernel` maps a squared distance (and
/// the pair's charges) to a `PairContribution` — the scalar `fpair` such that
/// the force on atom *i* is `fpair * (r_i - r_j)`, plus the pair potential
/// energy. LJ is just one instance; Coulomb real-space is a co-equal instance;
/// a combined style is `SumKernel<LJKernel, CoulKernel>` (one neighbor walk,
/// two contributions — LAMMPS `lj/cut/coul/long`).
///
/// Every kernel is templated on the scalar type `T` so ONE definition
/// instantiates for `double` (CPU scalar tail), `xsimd::batch<double>` (CPU
/// SIMD ket-loop), and the CUDA device real/work type. Each term masks its own
/// cutoff branch-free (multiply by a 0/1 `T`), so combined styles with
/// different LJ/Coulomb cutoffs compose correctly and vectorize.
///
/// The header is CUDA-safe: in a `.cu` translation unit only the scalar/device
/// math is compiled; the `xsimd` batch overloads are guarded out (a `.cu` only
/// ever instantiates the device scalar type).
#pragma once

#include <cmath>

#if defined(__CUDACC__)
#define ATV_PAIR_HD __host__ __device__
#else
#define ATV_PAIR_HD
#include <xsimd/xsimd.hpp>
#endif

namespace ATV {
namespace potentials {

/// @brief One pair's contribution: `force_on_i = fpair * (r_i - r_j)`, `energy`.
template <class T>
struct PairContribution {
    T fpair;
    T energy;
};

// ── Width-agnostic math helpers ──────────────────────────────────────────────
// Scalar overloads (host: std::; device: CUDA global namespace, selected by
// __CUDA_ARCH__). Batch overloads are added below, only in non-CUDA TUs.

ATV_PAIR_HD inline double av_sqrt(double x) {
#if defined(__CUDA_ARCH__)
    return ::sqrt(x);
#else
    return std::sqrt(x);
#endif
}
ATV_PAIR_HD inline float av_sqrt(float x) {
#if defined(__CUDA_ARCH__)
    return ::sqrtf(x);
#else
    return std::sqrt(x);
#endif
}
ATV_PAIR_HD inline double av_exp(double x) {
#if defined(__CUDA_ARCH__)
    return ::exp(x);
#else
    return std::exp(x);
#endif
}
ATV_PAIR_HD inline float av_exp(float x) {
#if defined(__CUDA_ARCH__)
    return ::expf(x);
#else
    return std::exp(x);
#endif
}
ATV_PAIR_HD inline double av_erfc(double x) {
#if defined(__CUDA_ARCH__)
    return ::erfc(x);
#else
    return std::erfc(x);
#endif
}
ATV_PAIR_HD inline float av_erfc(float x) {
#if defined(__CUDA_ARCH__)
    return ::erfcf(x);
#else
    return std::erfc(x);
#endif
}

/// @brief Cutoff indicator as a 0/1 value of type `T`: 1 where `r2 < rc2`.
ATV_PAIR_HD inline double pair_inside(double r2, double rc2) {
    return r2 < rc2 ? 1.0 : 0.0;
}
ATV_PAIR_HD inline float pair_inside(float r2, float rc2) {
    return r2 < rc2 ? 1.0f : 0.0f;
}

#if !defined(__CUDACC__)
// SIMD batch overloads (CPU only). erfc/exp/sqrt/select are xsimd primitives.
inline xsimd::batch<double> av_sqrt(xsimd::batch<double> x) {
    return xsimd::sqrt(x);
}
inline xsimd::batch<double> av_exp(xsimd::batch<double> x) {
    return xsimd::exp(x);
}
inline xsimd::batch<double> av_erfc(xsimd::batch<double> x) {
    return xsimd::erfc(x);
}
inline xsimd::batch<double> pair_inside(xsimd::batch<double> r2,
                                        xsimd::batch<double> rc2) {
    return xsimd::select(r2 < rc2, xsimd::batch<double>(1.0),
                         xsimd::batch<double>(0.0));
}
#endif

// ── Kernels ──────────────────────────────────────────────────────────────────

/// @brief Lennard-Jones 12-6 with shifted potential. Parameters held as
/// `double`; the math runs in `T`. Single-type; per-type tables can layer on
/// later without changing the seam.
struct LJKernel {
    double eps = 0.0;
    double sigma_sq = 0.0;
    double cutoff_sq = 0.0;
    double v_shift = 0.0;

    /// LJ is charge-independent — lets the launcher skip per-pair charge reads.
    static constexpr bool needs_charge = false;

    /// Outer cutoff² for the launcher's pre-skip.
    ATV_PAIR_HD double cutoff() const { return cutoff_sq; }

    /// Unmasked evaluation — the caller (single-kernel launcher) guarantees
    /// `r2 < cutoff()`, so no per-pair mask is computed (perf-parity with the
    /// hand-written kernel).
    template <class T>
    ATV_PAIR_HD PairContribution<T> raw(T r2, T /*qi*/, T /*qj*/) const {
        const T inv_r2 = T(1.0) / r2;
        const T sr2 = T(sigma_sq) * inv_r2;
        const T sr6 = sr2 * sr2 * sr2;
        const T sr12 = sr6 * sr6;
        const T fpair = T(24.0) * T(eps) * (T(2.0) * sr12 - sr6) * inv_r2;
        const T energy = T(4.0) * T(eps) * (sr12 - sr6) - T(v_shift);
        return {fpair, energy};
    }

    /// Masked evaluation — zeroes the contribution beyond the LJ cutoff. Used
    /// when composed (SumKernel) where `r2` may exceed this term's cutoff.
    template <class T>
    ATV_PAIR_HD PairContribution<T> operator()(T r2, T qi, T qj) const {
        const PairContribution<T> c = raw(r2, qi, qj);
        const T inside = pair_inside(r2, T(cutoff_sq));
        return {c.fpair * inside, c.energy * inside};
    }
};

/// @brief Real-space (screened) Coulomb — the short-range half of Ewald,
/// `E = qqr2e·q_i q_j·erfc(α r)/r`. Mirrors `cpu::CoulLong` exactly. `qqr2e` is
/// the Coulomb constant in the active unit system (ATV::units::coulomb).
struct CoulKernel {
    double qqr2e = 0.0;
    double alpha = 0.0;
    double cutoff_sq = 0.0;
    static constexpr double kEwaldF = 1.1283791670955126; // 2/sqrt(pi)

    static constexpr bool needs_charge = true;

    ATV_PAIR_HD double cutoff() const { return cutoff_sq; }

    template <class T>
    ATV_PAIR_HD PairContribution<T> raw(T r2, T qi, T qj) const {
        const T inv_r2 = T(1.0) / r2;
        const T r = av_sqrt(r2);
        const T grij = T(alpha) * r;
        const T expm2 = av_exp(-grij * grij);
        const T erfcv = av_erfc(grij);
        const T prefactor = T(qqr2e) * qi * qj / r;
        const T forcecoul = prefactor * (erfcv + T(kEwaldF) * grij * expm2);
        return {forcecoul * inv_r2, prefactor * erfcv};
    }

    template <class T>
    ATV_PAIR_HD PairContribution<T> operator()(T r2, T qi, T qj) const {
        const PairContribution<T> c = raw(r2, qi, qj);
        const T inside = pair_inside(r2, T(cutoff_sq));
        return {c.fpair * inside, c.energy * inside};
    }
};

/// @brief Compose two kernels into one neighbor-walk (e.g. lj/cut/coul/long):
/// sums their `fpair` and `energy` contributions per pair.
template <class A, class B>
struct SumKernel {
    A a;
    B b;
    static constexpr bool needs_charge = A::needs_charge || B::needs_charge;

    /// Outer cutoff² = the larger of the two terms' cutoffs (the launcher
    /// pre-skips beyond it; each term still masks its own cutoff internally).
    ATV_PAIR_HD double cutoff() const {
        const double ca = a.cutoff(), cb = b.cutoff();
        return ca > cb ? ca : cb;
    }

    /// Within the outer cutoff a pair may still exceed one term's cutoff, so
    /// raw() sums the per-term MASKED contributions (a()/b()).
    template <class T>
    ATV_PAIR_HD PairContribution<T> raw(T r2, T qi, T qj) const {
        const PairContribution<T> ca = a(r2, qi, qj);
        const PairContribution<T> cb = b(r2, qi, qj);
        return {ca.fpair + cb.fpair, ca.energy + cb.energy};
    }
    template <class T>
    ATV_PAIR_HD PairContribution<T> operator()(T r2, T qi, T qj) const {
        return raw(r2, qi, qj);
    }
};

} // namespace potentials
} // namespace ATV
