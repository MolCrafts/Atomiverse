/// @file cuda/semiempirical/nddo_pair.cuh
/// @brief Device-only scalar pair laws shared by NDDO ERI and Hcore kernels.
/// Radial multipoles follow cpu/semiempirical/nddo_two_electron.cpp.
#pragma once
#include "atomiverse/cuda/semiempirical/internal/nddo_fock.cuh"
#include "nddo_derivative.cuh"
#include <cmath>

namespace ATV::cuda::semiempirical::internal::detail {
constexpr double kA0 = 0.529177210903;
constexpr double kEv = 27.211386245988;

template <bool Full, class Scalar>
__device__ inline void radial_integrals(const NDDOAtom& pi, const NDDOAtom& pj,
                                        Scalar r_angstrom, Scalar* ri) {
    const double ev1 = kEv / 2.0;
    const double ev2 = ev1 / 2.0;
    const double ev3 = ev2 / 2.0;
    const double ev4 = ev3 / 2.0;
    constexpr double td = 2.0;
    constexpr double pp = 0.5;

    Scalar arg[Full ? 72 : 13]{};
    Scalar* sqr = arg; // In-place square roots: one 72-value radial scratch.
    const Scalar r = r_angstrom / kA0;
    const bool si = pi.n_orbitals >= 3;
    const bool sj = pj.n_orbitals >= 3;

    double aee = pp / pi.am + pp / pj.am;
    aee = aee * aee;
    if (!si && !sj) {
        ri[0] = kEv / sqrt(r * r + aee);
    } else if (si && !sj) {
        const double da = pi.dd;
        double qa = pi.qq * td;
        double ade = pp / pi.ad + pp / pj.am;
        ade *= ade;
        double aqe = pp / pi.aq + pp / pj.am;
        aqe *= aqe;
        const Scalar rsq = r * r;
        arg[0] = rsq + aee;
        arg[1] = (r + da) * (r + da) + ade;
        arg[2] = (r - da) * (r - da) + ade;
        arg[3] = (r + qa) * (r + qa) + aqe;
        arg[4] = (r - qa) * (r - qa) + aqe;
        arg[5] = rsq + aqe;
        arg[6] = arg[5] + qa * qa;
        for (int i = 0; i < 7; ++i) {
            sqr[i] = sqrt(arg[i]);
        }
        const Scalar ee = kEv / sqr[0];
        ri[0] = ee;
        ri[1] = ev1 / sqr[1] - ev1 / sqr[2];
        ri[2] = ee + ev2 / sqr[3] + ev2 / sqr[4] - ev1 / sqr[5];
        ri[3] = ee + ev1 / sqr[6] - ev1 / sqr[5];
    } else if (!si && sj) {
        const double db = pj.dd;
        double qb = pj.qq * td;
        double aed = pp / pi.am + pp / pj.ad;
        aed *= aed;
        double aeq = pp / pi.am + pp / pj.aq;
        aeq *= aeq;
        const Scalar rsq = r * r;
        arg[0] = rsq + aee;
        arg[1] = (r - db) * (r - db) + aed;
        arg[2] = (r + db) * (r + db) + aed;
        arg[3] = (r - qb) * (r - qb) + aeq;
        arg[4] = (r + qb) * (r + qb) + aeq;
        arg[5] = rsq + aeq;
        arg[6] = arg[5] + qb * qb;
        for (int i = 0; i < 7; ++i) {
            sqr[i] = sqrt(arg[i]);
        }
        const Scalar ee = kEv / sqr[0];
        ri[0] = ee;
        ri[4] = ev1 / sqr[1] - ev1 / sqr[2];
        ri[10] = ee + ev2 / sqr[3] + ev2 / sqr[4] - ev1 / sqr[5];
        ri[11] = ee + ev1 / sqr[6] - ev1 / sqr[5];
    } else {
        const double da = pi.dd;
        const double db = pj.dd;
        double qa = pi.qq * td;
        double qb = pj.qq * td;
        double ade = pp / pi.ad + pp / pj.am;
        ade *= ade;
        double aqe = pp / pi.aq + pp / pj.am;
        aqe *= aqe;
        double aed = pp / pi.am + pp / pj.ad;
        aed *= aed;
        double aeq = pp / pi.am + pp / pj.aq;
        aeq *= aeq;
        double axx = pp / pi.ad + pp / pj.ad;
        axx *= axx;
        double adq = pp / pi.ad + pp / pj.aq;
        adq *= adq;
        double aqd = pp / pi.aq + pp / pj.ad;
        aqd *= aqd;
        double aqq = pp / pi.aq + pp / pj.aq;
        aqq *= aqq;
        const Scalar rsq = r * r;
        arg[0] = rsq + aee;
        arg[1] = (r + da) * (r + da) + ade;
        arg[2] = (r - da) * (r - da) + ade;
        arg[3] = (r - qa) * (r - qa) + aqe;
        arg[4] = (r + qa) * (r + qa) + aqe;
        arg[5] = rsq + aqe;
        arg[6] = arg[5] + qa * qa;
        arg[7] = (r - db) * (r - db) + aed;
        arg[8] = (r + db) * (r + db) + aed;
        arg[9] = (r - qb) * (r - qb) + aeq;
        arg[10] = (r + qb) * (r + qb) + aeq;
        arg[11] = rsq + aeq;
        arg[12] = arg[11] + qb * qb;
        for (int i = 0; i < 13; ++i)
            sqr[i] = sqrt(arg[i]);
        const Scalar ee = kEv / sqr[0];
        ri[0] = ee;
        ri[1] = ev1 / sqr[1] - ev1 / sqr[2];
        ri[2] = ee + ev2 / sqr[3] + ev2 / sqr[4] - ev1 / sqr[5];
        ri[3] = ee + ev1 / sqr[6] - ev1 / sqr[5];
        ri[4] = ev1 / sqr[7] - ev1 / sqr[8];
        ri[10] = ee + ev2 / sqr[9] + ev2 / sqr[10] - ev1 / sqr[11];
        ri[11] = ee + ev1 / sqr[12] - ev1 / sqr[11];
        if constexpr (Full) {
            arg[13] = rsq + axx + (da - db) * (da - db);
            arg[14] = rsq + axx + (da + db) * (da + db);
            arg[15] = (r + da - db) * (r + da - db) + axx;
            arg[16] = (r - da + db) * (r - da + db) + axx;
            arg[17] = (r - da - db) * (r - da - db) + axx;
            arg[18] = (r + da + db) * (r + da + db) + axx;
            arg[19] = (r + da) * (r + da) + adq;
            arg[20] = arg[19] + qb * qb;
            arg[21] = (r - da) * (r - da) + adq;
            arg[22] = arg[21] + qb * qb;
            arg[23] = (r - db) * (r - db) + aqd;
            arg[24] = arg[23] + qa * qa;
            arg[25] = (r + db) * (r + db) + aqd;
            arg[26] = arg[25] + qa * qa;
            arg[27] = (r + da - qb) * (r + da - qb) + adq;
            arg[28] = (r - da - qb) * (r - da - qb) + adq;
            arg[29] = (r + da + qb) * (r + da + qb) + adq;
            arg[30] = (r - da + qb) * (r - da + qb) + adq;
            arg[31] = (r + qa - db) * (r + qa - db) + aqd;
            arg[32] = (r + qa + db) * (r + qa + db) + aqd;
            arg[33] = (r - qa - db) * (r - qa - db) + aqd;
            arg[34] = (r - qa + db) * (r - qa + db) + aqd;
            arg[35] = rsq + aqq;
            arg[36] = arg[35] + (qa - qb) * (qa - qb);
            arg[37] = arg[35] + (qa + qb) * (qa + qb);
            arg[38] = arg[35] + qa * qa;
            arg[39] = arg[35] + qb * qb;
            arg[40] = arg[38] + qb * qb;
            arg[41] = (r - qb) * (r - qb) + aqq;
            arg[42] = arg[41] + qa * qa;
            arg[43] = (r + qb) * (r + qb) + aqq;
            arg[44] = arg[43] + qa * qa;
            arg[45] = (r + qa) * (r + qa) + aqq;
            arg[46] = arg[45] + qb * qb;
            arg[47] = (r - qa) * (r - qa) + aqq;
            arg[48] = arg[47] + qb * qb;
            arg[49] = (r + qa - qb) * (r + qa - qb) + aqq;
            arg[50] = (r + qa + qb) * (r + qa + qb) + aqq;
            arg[51] = (r - qa - qb) * (r - qa - qb) + aqq;
            arg[52] = (r - qa + qb) * (r - qa + qb) + aqq;
            qa = pi.qq;
            qb = pj.qq;
            arg[53] = (da - qb) * (da - qb) + (r - qb) * (r - qb) + adq;
            arg[54] = (da - qb) * (da - qb) + (r + qb) * (r + qb) + adq;
            arg[55] = (da + qb) * (da + qb) + (r - qb) * (r - qb) + adq;
            arg[56] = (da + qb) * (da + qb) + (r + qb) * (r + qb) + adq;
            arg[57] = (r + qa) * (r + qa) + (qa - db) * (qa - db) + aqd;
            arg[58] = (r - qa) * (r - qa) + (qa - db) * (qa - db) + aqd;
            arg[59] = (r + qa) * (r + qa) + (qa + db) * (qa + db) + aqd;
            arg[60] = (r - qa) * (r - qa) + (qa + db) * (qa + db) + aqd;
            arg[61] = arg[35] + td * (qa - qb) * (qa - qb);
            arg[62] = arg[35] + td * (qa + qb) * (qa + qb);
            arg[63] = arg[35] + td * (qa * qa + qb * qb);
            arg[64] =
                (r + qa - qb) * (r + qa - qb) + (qa - qb) * (qa - qb) + aqq;
            arg[65] =
                (r + qa - qb) * (r + qa - qb) + (qa + qb) * (qa + qb) + aqq;
            arg[66] =
                (r + qa + qb) * (r + qa + qb) + (qa - qb) * (qa - qb) + aqq;
            arg[67] =
                (r + qa + qb) * (r + qa + qb) + (qa + qb) * (qa + qb) + aqq;
            arg[68] =
                (r - qa - qb) * (r - qa - qb) + (qa - qb) * (qa - qb) + aqq;
            arg[69] =
                (r - qa - qb) * (r - qa - qb) + (qa + qb) * (qa + qb) + aqq;
            arg[70] =
                (r - qa + qb) * (r - qa + qb) + (qa - qb) * (qa - qb) + aqq;
            arg[71] =
                (r - qa + qb) * (r - qa + qb) + (qa + qb) * (qa + qb) + aqq;
            for (int i = 13; i < 72; ++i) {
                sqr[i] = sqrt(arg[i]);
            }
            const Scalar dze = -ri[1], edz = -ri[4];
            const Scalar qzze = ev2 / sqr[3] + ev2 / sqr[4] - ev1 / sqr[5];
            const Scalar qxxe = ev1 / sqr[6] - ev1 / sqr[5];
            const Scalar eqzz = ev2 / sqr[9] + ev2 / sqr[10] - ev1 / sqr[11];
            const Scalar eqxx = ev1 / sqr[12] - ev1 / sqr[11];
            const Scalar dxdx = ev1 / sqr[13] - ev1 / sqr[14];
            const Scalar dzdz =
                ev2 / sqr[15] + ev2 / sqr[16] - ev2 / sqr[17] - ev2 / sqr[18];
            const Scalar dzqxx =
                ev2 / sqr[19] - ev2 / sqr[20] - ev2 / sqr[21] + ev2 / sqr[22];
            const Scalar qxxdz =
                ev2 / sqr[23] - ev2 / sqr[24] - ev2 / sqr[25] + ev2 / sqr[26];
            const Scalar dzqzz = (-ev3 / sqr[27]) + ev3 / sqr[28]
                                 - ev3 / sqr[29] + ev3 / sqr[30] - ev2 / sqr[21]
                                 + ev2 / sqr[19];
            const Scalar qzzdz = (-ev3 / sqr[31]) + ev3 / sqr[32]
                                 - ev3 / sqr[33] + ev3 / sqr[34] + ev2 / sqr[23]
                                 - ev2 / sqr[25];
            const Scalar qxxqxx = ev3 / sqr[36] + ev3 / sqr[37] - ev2 / sqr[38]
                                  - ev2 / sqr[39] + ev2 / sqr[35];
            const Scalar qxxqyy =
                ev2 / sqr[40] - ev2 / sqr[38] - ev2 / sqr[39] + ev2 / sqr[35];
            const Scalar qxxqzz = ev3 / sqr[42] + ev3 / sqr[44] - ev3 / sqr[41]
                                  - ev3 / sqr[43] - ev2 / sqr[38]
                                  + ev2 / sqr[35];
            const Scalar qzzqxx = ev3 / sqr[46] + ev3 / sqr[48] - ev3 / sqr[45]
                                  - ev3 / sqr[47] - ev2 / sqr[39]
                                  + ev2 / sqr[35];
            const Scalar qzzqzz = ev4 / sqr[49] + ev4 / sqr[50] + ev4 / sqr[51]
                                  + ev4 / sqr[52] - ev3 / sqr[47]
                                  - ev3 / sqr[45] - ev3 / sqr[41]
                                  - ev3 / sqr[43] + ev2 / sqr[35];
            const Scalar dxqxz = (-ev2 / sqr[53]) + ev2 / sqr[54]
                                 + ev2 / sqr[55] - ev2 / sqr[56];
            const Scalar qxzdx = (-ev2 / sqr[57]) + ev2 / sqr[58]
                                 + ev2 / sqr[59] - ev2 / sqr[60];
            const Scalar qxzqxz =
                ev3 / sqr[64] - ev3 / sqr[66] - ev3 / sqr[68] + ev3 / sqr[70]
                - ev3 / sqr[65] + ev3 / sqr[67] + ev3 / sqr[69] - ev3 / sqr[71];
            ri[5] = dzdz;
            ri[6] = dxdx;
            ri[7] = (-edz) - qzzdz;
            ri[8] = (-edz) - qxxdz;
            ri[9] = -qxzdx;
            ri[12] = (-dze) - dzqzz;
            ri[13] = (-dze) - dzqxx;
            ri[14] = -dxqxz;
            ri[15] = ee + eqzz + qzze + qzzqzz;
            ri[16] = ee + eqzz + qxxe + qxxqzz;
            ri[17] = ee + eqxx + qzze + qzzqxx;
            ri[18] = ee + eqxx + qxxe + qxxqxx;
            ri[19] = qxzqxz;
            ri[20] = ee + eqxx + qxxe + qxxqyy;
            ri[21] = pp * (qxxqxx - qxxqyy);
        }
    }

    constexpr double kNri[22] = {1.0,  -1.0, 1.0, 1.0, -1.0, 1.0,  1.0,  -1.0,
                                 -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0,
                                 1.0,  1.0,  1.0, 1.0, 1.0,  1.0};
    for (int i = 0; i < 22; ++i) {
        ri[i] *= kNri[i];
    }
}

template <class Scalar>
__device__ inline void pair_axes(Scalar dx, Scalar dy, Scalar dz, Scalar r,
                                 Scalar* axes) {
    axes[0] = dx / r;
    axes[1] = dy / r;
    axes[2] = dz / r;
    axes[6] = fabs(axes[2]) < 0.8 ? -axes[1] : axes[2];
    axes[7] = fabs(axes[2]) < 0.8 ? axes[0] : Scalar(0.0);
    axes[8] = fabs(axes[2]) < 0.8 ? Scalar(0.0) : -axes[0];
    const Scalar t =
        sqrt(axes[6] * axes[6] + axes[7] * axes[7] + axes[8] * axes[8]);
    for (int d = 0; d < 3; ++d)
        axes[6 + d] /= t;
    for (int d = 0; d < 3; ++d)
        axes[3 + d] = axes[6 + (d + 1) % 3] * axes[(d + 2) % 3]
                      - axes[6 + (d + 2) % 3] * axes[(d + 1) % 3];
}
// The 34 nonzero molecular-axis pair integrals and their 22 radial values.
// Packed local pair order is ss, xs, xx, ys, yx, yy, zs, zx, zy, zz.
static __device__ __constant__ int pair_row[34] = {
    0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3,
    4, 4, 5, 5, 5, 5, 5, 6, 6, 7, 7, 8, 9, 9, 9, 9, 9};
static __device__ __constant__ int pair_col[34] = {
    0, 1, 2, 5, 9, 0, 1, 2, 5, 9, 0, 1, 2, 5, 9, 3, 4,
    3, 4, 0, 1, 2, 5, 9, 6, 7, 6, 7, 8, 0, 1, 2, 5, 9};
static __device__ __constant__ int radial_index[34] = {
    0, 4,  10, 11, 11, 1,  5,  12, 13, 13, 2,  7,  15, 17, 17, 6,  14,
    9, 19, 3,  8,  16, 18, 20, 6,  14, 9,  19, 21, 3,  8,  16, 20, 18};
static __device__ __constant__ int local_row[10] = {0, 1, 1, 2, 2,
                                                    2, 3, 3, 3, 3};
static __device__ __constant__ int local_col[10] = {0, 0, 1, 0, 1,
                                                    2, 0, 1, 2, 3};

template <class Scalar>
__device__ Scalar rotation_coefficient(const Scalar* axes, int global_pair,
                                       int molecular_pair) {
    const int a = local_row[global_pair], b = local_col[global_pair];
    const int k = local_row[molecular_pair], l = local_col[molecular_pair];
    if (a == 0)
        return k == 0 ? 1.0 : 0.0;
    if (b == 0)
        return k > 0 && l == 0 ? axes[(k - 1) * 3 + a - 1] : Scalar(0.0);
    if (l == 0)
        return 0.0;
    const Scalar first = axes[(k - 1) * 3 + a - 1] * axes[(l - 1) * 3 + b - 1];
    return k == l
               ? first
               : first + axes[(l - 1) * 3 + a - 1] * axes[(k - 1) * 3 + b - 1];
}

} // namespace ATV::cuda::semiempirical::internal::detail
