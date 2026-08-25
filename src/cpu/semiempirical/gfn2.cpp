/// @file src/cpu/semiempirical/gfn2.cpp
/// GFN2-xTB energy (H/C/N/O): native static tables, no libxtb.
#include "atomiverse/cpu/semiempirical/gfn2.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/linalg/eigensolve.h"
#include "atomiverse/cpu/linalg/matrix.h"
#include "atomiverse/cpu/scf/dftd4.h"
#include "atomiverse/cpu/semiempirical/basis_data/gfn2_xtb_tables.h"
#include "atomiverse/cpu/semiempirical/internal/scc_mixer.h"

namespace ATV {
namespace cpu {
namespace semiempirical {
namespace {

using basis_data::Gfn2ElementRow;
using basis_data::kGfn2AesDmp3;
using basis_data::kGfn2AesDmp5;
using basis_data::kGfn2AesKexp;
using basis_data::kGfn2AesRmax;
using basis_data::kGfn2AesShift;
using basis_data::kGfn2CnK;
using basis_data::kGfn2D4A1;
using basis_data::kGfn2D4A2;
using basis_data::kGfn2D4S6;
using basis_data::kGfn2D4S8;
using basis_data::kGfn2D4S9;
using basis_data::kGfn2Elements;
using basis_data::kGfn2Enscale;
using basis_data::kGfn2EvToEh;
using basis_data::kGfn2Gexp;
using basis_data::kGfn2Kexp;
using basis_data::kGfn2KexpLight;
using basis_data::kGfn2Kp;
using basis_data::kGfn2Ks;
using basis_data::kGfn2NElements;
using basis_data::kGfn2Rexp;
using basis_data::kGfn2Wexp;

constexpr double kPi = 3.14159265358979323846;
constexpr double kBohr = units::kAngstromInBohr;

const Gfn2ElementRow& element_row(int z) {
    for (std::size_t i = 0; i < kGfn2NElements; ++i)
        if (kGfn2Elements[i].z == z)
            return kGfn2Elements[i];
    throw std::runtime_error(
        "GFN2: unsupported element Z=" + std::to_string(z)
        + " (static tables cover H/C/N/O; other Z is a later deepen)");
}

int n_ao_of(const Gfn2ElementRow& e) {
    int n = 0;
    for (int s = 0; s < e.n_shell; ++s)
        n += 2 * e.l[s] + 1;
    return n;
}

// Stewart STO-nG (tblite basis/slater.f90). 1s n=3; 2s/2p n=4.
void sto_ng(int n_prim, int pqn, int l, double zeta, double* alpha,
            double* coeff) {
    const double z2 = zeta * zeta;
    if (n_prim == 3 && pqn == 1 && l == 0) {
        alpha[0] = 2.227660584 * z2;
        alpha[1] = 0.4057711562 * z2;
        alpha[2] = 0.1098175104 * z2;
        coeff[0] = 0.1543289673;
        coeff[1] = 0.5353281423;
        coeff[2] = 0.4446345422;
    } else if (n_prim == 4 && pqn == 2 && l == 0) {
        alpha[0] = 11.61525551 * z2;
        alpha[1] = 2.000243111 * z2;
        alpha[2] = 0.1607280687 * z2;
        alpha[3] = 0.06125744532 * z2;
        coeff[0] = -0.01198411747;
        coeff[1] = -0.05472052539;
        coeff[2] = 0.5805587176;
        coeff[3] = 0.4770079976;
    } else if (n_prim == 4 && pqn == 2 && l == 1) {
        alpha[0] = 1.798260992 * z2;
        alpha[1] = 0.4662622228 * z2;
        alpha[2] = 0.1643718620 * z2;
        alpha[3] = 0.06543927065 * z2;
        coeff[0] = 0.05713170255;
        coeff[1] = 0.2857455515;
        coeff[2] = 0.5517873105;
        coeff[3] = 0.2632314924;
    } else {
        throw std::runtime_error("GFN2: STO-nG expansion not tabulated for "
                                 "this (n_prim,pqn,l)");
    }
    const double top = 2.0 / kPi;
    const double dfact = (l == 0) ? 1.0 : 1.0; // (2l-1)!! ; l=0,1 → 1
    for (int p = 0; p < n_prim; ++p) {
        const double a = alpha[p];
        coeff[p] *= std::pow(top * a, 0.75) * std::pow(std::sqrt(4.0 * a), l)
                    / std::sqrt(dfact);
    }
}

struct Shell {
    int atom = 0;
    int l = 0;
    int n_prim = 0;
    int ao0 = 0;
    double zeta = 0;
    double alpha[4]{};
    double coeff[4]{};
};

void primitive_ss(double a, double b, const double A[3], const double B[3],
                  double& S00, double P[3]) {
    const double g = a + b;
    const double mu = a * b / g;
    double r2 = 0;
    for (int k = 0; k < 3; ++k) {
        const double d = A[k] - B[k];
        r2 += d * d;
        P[k] = (a * A[k] + b * B[k]) / g;
    }
    S00 = std::pow(kPi / g, 1.5) * std::exp(-mu * r2);
}

// Cartesian second moment <r_a r_b> at origin 0 (s/p primitives).
double cartesian_rr(int a, int b, const double P[3], double dAi0, double dAj0,
                    int li, int lj, int ci, int cj, double inv2g) {
    double v = P[a] * P[b] * dAi0 * dAj0;
    if (a == b)
        v += inv2g * dAi0 * dAj0;
    if (li == 1) {
        if (ci == a)
            v += inv2g * P[b] * dAj0;
        if (ci == b)
            v += inv2g * P[a] * dAj0;
    }
    if (lj == 1) {
        if (cj == a)
            v += inv2g * P[b] * dAi0;
        if (cj == b)
            v += inv2g * P[a] * dAi0;
    }
    if (li == 1 && lj == 1) {
        if (ci == cj) {
            v += inv2g * P[a] * P[b];
            if (a == b)
                v += inv2g * inv2g;
        }
        if (ci == a && cj == b)
            v += inv2g * inv2g;
        if (ci == b && cj == a)
            v += inv2g * inv2g;
    }
    return v;
}

void traceless_quad(const double Q[6], double T[6]) {
    const double tr = 0.5 * (Q[0] + Q[2] + Q[5]);
    T[0] = 1.5 * Q[0] - tr;
    T[1] = 1.5 * Q[1];
    T[2] = 1.5 * Q[2] - tr;
    T[3] = 1.5 * Q[3];
    T[4] = 1.5 * Q[4];
    T[5] = 1.5 * Q[5] - tr;
}

void origin_shift_quad(const double Q[6], const double D[3], const double* R,
                       double s, double tq[6]) {
    // tblite shift_operator / physical ⟨(r−R)(r−R)⟩. Minus-RRS (xtb
    // mmompop with opab4-signed Q plugged into unsigned ATV Q) breaks
    // SCC (residual stuck 7e-6, ΔE ~ 0.67 eV).
    double qs[6];
    qs[0] = Q[0] - 2.0 * R[0] * D[0] + R[0] * R[0] * s;
    qs[1] = Q[1] - R[0] * D[1] - R[1] * D[0] + R[0] * R[1] * s;
    qs[2] = Q[2] - 2.0 * R[1] * D[1] + R[1] * R[1] * s;
    qs[3] = Q[3] - R[0] * D[2] - R[2] * D[0] + R[0] * R[2] * s;
    qs[4] = Q[4] - R[1] * D[2] - R[2] * D[1] + R[1] * R[2] * s;
    qs[5] = Q[5] - 2.0 * R[2] * D[2] + R[2] * R[2] * s;
    traceless_quad(qs, tq);
}

// Overlap, dipole, and Cartesian quadrupole (origin 0) between two
// contracted Cartesian s/p shells. Packed Q: xx, xy, yy, xz, yz, zz.
void shell_ints(const Shell& si, const double Ai[3], const Shell& sj,
                const double Aj[3], int iao, int jao, double& S, double D[3],
                double Q[6]) {
    S = 0;
    D[0] = D[1] = D[2] = 0;
    for (int k = 0; k < 6; ++k)
        Q[k] = 0;
    const int li = si.l;
    const int lj = sj.l;
    const int ci = (li == 0) ? 0 : iao;
    const int cj = (lj == 0) ? 0 : jao;
    static constexpr int kPack[6][2] = {{0, 0}, {0, 1}, {1, 1},
                                        {0, 2}, {1, 2}, {2, 2}};
    for (int p = 0; p < si.n_prim; ++p) {
        for (int q = 0; q < sj.n_prim; ++q) {
            const double a = si.alpha[p];
            const double b = sj.alpha[q];
            const double cab = si.coeff[p] * sj.coeff[q];
            double P[3];
            double S00;
            primitive_ss(a, b, Ai, Aj, S00, P);
            const double g = a + b;
            const double inv2g = 0.5 / g;
            const double dAi0 = (li == 1) ? (P[ci] - Ai[ci]) : 1.0;
            const double dAj0 = (lj == 1) ? (P[cj] - Aj[cj]) : 1.0;
            double s = S00 * dAi0 * dAj0;
            if (li == 1 && lj == 1 && ci == cj)
                s += S00 * inv2g;
            S += cab * s;
            for (int k = 0; k < 3; ++k) {
                double t = P[k] * dAi0 * dAj0;
                if (li == 1 && lj == 1 && ci == cj)
                    t += P[k] * inv2g;
                if (li == 1 && ci == k)
                    t += inv2g * dAj0;
                if (lj == 1 && cj == k)
                    t += inv2g * dAi0;
                D[k] += cab * S00 * t;
            }
            for (int k = 0; k < 6; ++k) {
                const double t = cartesian_rr(kPack[k][0], kPack[k][1], P, dAi0,
                                              dAj0, li, lj, ci, cj, inv2g);
                Q[k] += cab * S00 * t;
            }
        }
    }
}

double k_shell(int li, int lj) {
    if (li > lj)
        std::swap(li, lj);
    if (li == 0 && lj == 0)
        return kGfn2Ks;
    if (li == 1 && lj == 1)
        return kGfn2Kp;
    return 0.5 * (kGfn2Ks + kGfn2Kp);
}

double cn_d3(double r, double r0) {
    if (r < 1e-12)
        return 0;
    // DFT-D3 counting (xtb covCN); r0 = rcov_i+rcov_j [Bohr], k=16, 4/3.
    const double rco = (4.0 / 3.0) * r0;
    return 1.0 / (1.0 + std::exp(-kGfn2CnK * (rco / r - 1.0)));
}

double dcn_d3_dr(double r, double r0) {
    if (r < 1e-12)
        return 0;
    const double rco = (4.0 / 3.0) * r0;
    const double e = std::exp(-kGfn2CnK * (rco / r - 1.0));
    const double den = 1.0 + e;
    return -e / (den * den) * kGfn2CnK * rco / (r * r);
}

using Grad3 = std::array<double, 3>;

// dS/dA of a Cartesian s/p primitive pair (GFN0 kernel). dS/dB = -dS/dA.
void primitive_overlap_grad(double a, const double A[3], int la, double b,
                            const double B[3], int lb, double& s,
                            double ds_dA[3]) {
    const double gamma = a + b;
    const double mu = a * b / gamma;
    double R[3];
    double R2 = 0;
    for (int k = 0; k < 3; ++k) {
        R[k] = A[k] - B[k];
        R2 += R[k] * R[k];
    }
    const double S00 = std::pow(kPi / gamma, 1.5) * std::exp(-mu * R2);
    double dS00[3];
    for (int k = 0; k < 3; ++k)
        dS00[k] = -2.0 * mu * R[k] * S00;
    double P[3];
    for (int k = 0; k < 3; ++k)
        P[k] = (a * A[k] + b * B[k]) / gamma;

    if (la == 0 && lb == 0) {
        s = S00;
        for (int k = 0; k < 3; ++k)
            ds_dA[k] = dS00[k];
        return;
    }
    if (la == 0) {
        const int ib = lb - 1;
        const double dlt = P[ib] - B[ib];
        s = dlt * S00;
        for (int k = 0; k < 3; ++k) {
            const double ddlt = (k == ib) ? (a / gamma) : 0.0;
            ds_dA[k] = ddlt * S00 + dlt * dS00[k];
        }
        return;
    }
    if (lb == 0) {
        const int ia = la - 1;
        const double dlt = P[ia] - A[ia];
        s = dlt * S00;
        for (int k = 0; k < 3; ++k) {
            const double ddlt = (k == ia) ? (-b / gamma) : 0.0;
            ds_dA[k] = ddlt * S00 + dlt * dS00[k];
        }
        return;
    }
    const int ia = la - 1;
    const int ib = lb - 1;
    const double dA = P[ia] - A[ia];
    const double dB = P[ib] - B[ib];
    s = dA * dB * S00;
    if (ia == ib)
        s += S00 / (2.0 * gamma);
    for (int k = 0; k < 3; ++k) {
        const double ddA = (k == ia) ? (-b / gamma) : 0.0;
        const double ddB = (k == ib) ? (a / gamma) : 0.0;
        ds_dA[k] = (ddA * dB + dA * ddB) * S00 + dA * dB * dS00[k];
        if (ia == ib)
            ds_dA[k] += dS00[k] / (2.0 * gamma);
    }
}

void primitive_sd_grad(double a, const double A[3], int la, double b,
                       const double B[3], int lb, double& s, double ds_dA[3],
                       double D[3], double dD_dA[3][3]) {
    const double gamma = a + b;
    const double mu = a * b / gamma;
    const double inv2g = 0.5 / gamma;
    double R[3];
    double R2 = 0;
    for (int k = 0; k < 3; ++k) {
        R[k] = A[k] - B[k];
        R2 += R[k] * R[k];
    }
    const double S00 = std::pow(kPi / gamma, 1.5) * std::exp(-mu * R2);
    double dS00[3];
    for (int k = 0; k < 3; ++k)
        dS00[k] = -2.0 * mu * R[k] * S00;
    double P[3];
    for (int k = 0; k < 3; ++k)
        P[k] = (a * A[k] + b * B[k]) / gamma;
    auto dP = [a, gamma](int k, int m) {
        return (k == m) ? (a / gamma) : 0.0;
    };

    primitive_overlap_grad(a, A, la, b, B, lb, s, ds_dA);

    if (la == 0 && lb == 0) {
        for (int k = 0; k < 3; ++k) {
            D[k] = P[k] * S00;
            for (int m = 0; m < 3; ++m)
                dD_dA[k][m] = dP(k, m) * S00 + P[k] * dS00[m];
        }
        return;
    }
    const int ci = (la == 0) ? 0 : (la - 1);
    const int cj = (lb == 0) ? 0 : (lb - 1);
    const double dAi0 = (la == 1) ? (P[ci] - A[ci]) : 1.0;
    const double dAj0 = (lb == 1) ? (P[cj] - B[cj]) : 1.0;
    for (int k = 0; k < 3; ++k) {
        double t = P[k] * dAi0 * dAj0;
        if (la == 1 && lb == 1 && ci == cj)
            t += P[k] * inv2g;
        if (la == 1 && ci == k)
            t += inv2g * dAj0;
        if (lb == 1 && cj == k)
            t += inv2g * dAi0;
        D[k] = S00 * t;
        for (int m = 0; m < 3; ++m) {
            const double ddAi =
                (la == 1) ? (dP(ci, m) - ((m == ci) ? 1.0 : 0.0)) : 0.0;
            const double ddAj = (lb == 1) ? dP(cj, m) : 0.0;
            double dt = dP(k, m) * dAi0 * dAj0 + P[k] * ddAi * dAj0
                        + P[k] * dAi0 * ddAj;
            if (la == 1 && lb == 1 && ci == cj)
                dt += dP(k, m) * inv2g;
            if (la == 1 && ci == k)
                dt += inv2g * ddAj;
            if (lb == 1 && cj == k)
                dt += inv2g * ddAi;
            dD_dA[k][m] = dS00[m] * t + S00 * dt;
        }
    }
}

void shell_sd_grad(const Shell& si, const double Ai[3], int ui, const Shell& sj,
                   const double Aj[3], int vj, double& s, double ds_dAi[3],
                   double D[3], double dD_dAi[3][3]) {
    s = 0;
    ds_dAi[0] = ds_dAi[1] = ds_dAi[2] = 0;
    D[0] = D[1] = D[2] = 0;
    for (int k = 0; k < 3; ++k)
        for (int m = 0; m < 3; ++m)
            dD_dAi[k][m] = 0;
    const int Li = (si.l == 0) ? 0 : (ui + 1);
    const int Lj = (sj.l == 0) ? 0 : (vj + 1);
    for (int p = 0; p < si.n_prim; ++p) {
        for (int q = 0; q < sj.n_prim; ++q) {
            double sp = 0;
            double dsp[3];
            double Dp[3];
            double dDp[3][3];
            primitive_sd_grad(si.alpha[p], Ai, Li, sj.alpha[q], Aj, Lj, sp, dsp,
                              Dp, dDp);
            const double c = si.coeff[p] * sj.coeff[q];
            s += c * sp;
            for (int k = 0; k < 3; ++k) {
                ds_dAi[k] += c * dsp[k];
                D[k] += c * Dp[k];
                for (int m = 0; m < 3; ++m)
                    dD_dAi[k][m] += c * dDp[k][m];
            }
        }
    }
}

// Cartesian quadrupole <r_a r_b> at origin 0 and d/dA (s/p primitives).
void cartesian_rr_grad(int ia, int ib, const double P[3], const double dS00[3],
                       double S00, double dAi0, double dAj0, double ddAi0[3],
                       double ddAj0[3], int li, int lj, int ci, int cj,
                       double inv2g, double a_over_g, double& v,
                       double dv_dA[3]) {
    auto dP = [a_over_g](int k, int m) {
        return (k == m) ? a_over_g : 0.0;
    };
    v = P[ia] * P[ib] * dAi0 * dAj0;
    if (ia == ib)
        v += inv2g * dAi0 * dAj0;
    if (li == 1) {
        if (ci == ia)
            v += inv2g * P[ib] * dAj0;
        if (ci == ib)
            v += inv2g * P[ia] * dAj0;
    }
    if (lj == 1) {
        if (cj == ia)
            v += inv2g * P[ib] * dAi0;
        if (cj == ib)
            v += inv2g * P[ia] * dAi0;
    }
    if (li == 1 && lj == 1) {
        if (ci == cj) {
            v += inv2g * P[ia] * P[ib];
            if (ia == ib)
                v += inv2g * inv2g;
        }
        if (ci == ia && cj == ib)
            v += inv2g * inv2g;
        if (ci == ib && cj == ia)
            v += inv2g * inv2g;
    }
    for (int m = 0; m < 3; ++m) {
        double dv = dP(ia, m) * P[ib] * dAi0 * dAj0
                    + P[ia] * dP(ib, m) * dAi0 * dAj0
                    + P[ia] * P[ib] * ddAi0[m] * dAj0
                    + P[ia] * P[ib] * dAi0 * ddAj0[m];
        if (ia == ib)
            dv += inv2g * (ddAi0[m] * dAj0 + dAi0 * ddAj0[m]);
        if (li == 1) {
            if (ci == ia)
                dv += inv2g * (dP(ib, m) * dAj0 + P[ib] * ddAj0[m]);
            if (ci == ib)
                dv += inv2g * (dP(ia, m) * dAj0 + P[ia] * ddAj0[m]);
        }
        if (lj == 1) {
            if (cj == ia)
                dv += inv2g * (dP(ib, m) * dAi0 + P[ib] * ddAi0[m]);
            if (cj == ib)
                dv += inv2g * (dP(ia, m) * dAi0 + P[ia] * ddAi0[m]);
        }
        if (li == 1 && lj == 1 && ci == cj)
            dv += inv2g * (dP(ia, m) * P[ib] + P[ia] * dP(ib, m));
        dv_dA[m] = dS00[m] * v + S00 * dv;
    }
}

void primitive_sdq_grad(double a, const double A[3], int la, double b,
                        const double B[3], int lb, double& s, double ds_dA[3],
                        double D[3], double dD_dA[3][3], double Q[6],
                        double dQ_dA[6][3]) {
    primitive_sd_grad(a, A, la, b, B, lb, s, ds_dA, D, dD_dA);
    const double gamma = a + b;
    const double mu = a * b / gamma;
    const double inv2g = 0.5 / gamma;
    const double a_over_g = a / gamma;
    double R[3];
    double R2 = 0;
    for (int k = 0; k < 3; ++k) {
        R[k] = A[k] - B[k];
        R2 += R[k] * R[k];
    }
    const double S00 = std::pow(kPi / gamma, 1.5) * std::exp(-mu * R2);
    double dS00[3];
    for (int k = 0; k < 3; ++k)
        dS00[k] = -2.0 * mu * R[k] * S00;
    double P[3];
    for (int k = 0; k < 3; ++k)
        P[k] = (a * A[k] + b * B[k]) / gamma;
    const int ci = (la == 0) ? 0 : (la - 1);
    const int cj = (lb == 0) ? 0 : (lb - 1);
    const double dAi0 = (la == 1) ? (P[ci] - A[ci]) : 1.0;
    const double dAj0 = (lb == 1) ? (P[cj] - B[cj]) : 1.0;
    double ddAi0[3] = {0, 0, 0};
    double ddAj0[3] = {0, 0, 0};
    for (int m = 0; m < 3; ++m) {
        if (la == 1)
            ddAi0[m] = ((m == ci) ? a_over_g : 0.0) - ((m == ci) ? 1.0 : 0.0);
        if (lb == 1)
            ddAj0[m] = (m == cj) ? a_over_g : 0.0;
    }
    static constexpr int kPack[6][2] = {{0, 0}, {0, 1}, {1, 1},
                                        {0, 2}, {1, 2}, {2, 2}};
    const int li = (la == 0) ? 0 : 1;
    const int lj = (lb == 0) ? 0 : 1;
    for (int k = 0; k < 6; ++k) {
        double v = 0;
        double dQk[3];
        cartesian_rr_grad(kPack[k][0], kPack[k][1], P, dS00, S00, dAi0, dAj0,
                          ddAi0, ddAj0, li, lj, ci, cj, inv2g, a_over_g, v,
                          dQk);
        Q[k] = S00 * v;
        for (int m = 0; m < 3; ++m)
            dQ_dA[k][m] = dQk[m];
    }
}

void shell_sdq_grad(const Shell& si, const double Ai[3], int ui,
                    const Shell& sj, const double Aj[3], int vj, double& s,
                    double ds_dAi[3], double D[3], double dD_dAi[3][3],
                    double Q[6], double dQ_dAi[6][3]) {
    s = 0;
    ds_dAi[0] = ds_dAi[1] = ds_dAi[2] = 0;
    D[0] = D[1] = D[2] = 0;
    for (int k = 0; k < 3; ++k)
        for (int m = 0; m < 3; ++m)
            dD_dAi[k][m] = 0;
    for (int k = 0; k < 6; ++k) {
        Q[k] = 0;
        for (int m = 0; m < 3; ++m)
            dQ_dAi[k][m] = 0;
    }
    const int Li = (si.l == 0) ? 0 : (ui + 1);
    const int Lj = (sj.l == 0) ? 0 : (vj + 1);
    for (int p = 0; p < si.n_prim; ++p) {
        for (int q = 0; q < sj.n_prim; ++q) {
            double sp = 0;
            double dsp[3];
            double Dp[3];
            double dDp[3][3];
            double Qp[6];
            double dQp[6][3];
            primitive_sdq_grad(si.alpha[p], Ai, Li, sj.alpha[q], Aj, Lj, sp,
                               dsp, Dp, dDp, Qp, dQp);
            const double c = si.coeff[p] * sj.coeff[q];
            s += c * sp;
            for (int k = 0; k < 3; ++k) {
                ds_dAi[k] += c * dsp[k];
                D[k] += c * Dp[k];
                for (int m = 0; m < 3; ++m)
                    dD_dAi[k][m] += c * dDp[k][m];
            }
            for (int k = 0; k < 6; ++k) {
                Q[k] += c * Qp[k];
                for (int m = 0; m < 3; ++m)
                    dQ_dAi[k][m] += c * dQp[k][m];
            }
        }
    }
}

// Origin-shifted traceless quadrupole and d/dA. dR[α][m] = ∂R_α/∂A_m
// (identity if the CAMM owner is A, zero if it is B).
void origin_shift_quad_grad(const double Q[6], const double dQ[6][3],
                            const double D[3], const double dD[3][3],
                            const double* R, const double dR[3][3], double s,
                            const double ds[3], double tq[6],
                            double dtq[6][3]) {
    double qs[6];
    double dqs[6][3];
    qs[0] = Q[0] - 2.0 * R[0] * D[0] + R[0] * R[0] * s;
    qs[1] = Q[1] - R[0] * D[1] - R[1] * D[0] + R[0] * R[1] * s;
    qs[2] = Q[2] - 2.0 * R[1] * D[1] + R[1] * R[1] * s;
    qs[3] = Q[3] - R[0] * D[2] - R[2] * D[0] + R[0] * R[2] * s;
    qs[4] = Q[4] - R[1] * D[2] - R[2] * D[1] + R[1] * R[2] * s;
    qs[5] = Q[5] - 2.0 * R[2] * D[2] + R[2] * R[2] * s;
    for (int m = 0; m < 3; ++m) {
        dqs[0][m] = dQ[0][m] - 2.0 * (dR[0][m] * D[0] + R[0] * dD[0][m])
                    + 2.0 * R[0] * dR[0][m] * s + R[0] * R[0] * ds[m];
        dqs[1][m] = dQ[1][m] - dR[0][m] * D[1] - R[0] * dD[1][m]
                    - dR[1][m] * D[0] - R[1] * dD[0][m]
                    + dR[0][m] * R[1] * s + R[0] * dR[1][m] * s
                    + R[0] * R[1] * ds[m];
        dqs[2][m] = dQ[2][m] - 2.0 * (dR[1][m] * D[1] + R[1] * dD[1][m])
                    + 2.0 * R[1] * dR[1][m] * s + R[1] * R[1] * ds[m];
        dqs[3][m] = dQ[3][m] - dR[0][m] * D[2] - R[0] * dD[2][m]
                    - dR[2][m] * D[0] - R[2] * dD[0][m]
                    + dR[0][m] * R[2] * s + R[0] * dR[2][m] * s
                    + R[0] * R[2] * ds[m];
        dqs[4][m] = dQ[4][m] - dR[1][m] * D[2] - R[1] * dD[2][m]
                    - dR[2][m] * D[1] - R[2] * dD[1][m]
                    + dR[1][m] * R[2] * s + R[1] * dR[2][m] * s
                    + R[1] * R[2] * ds[m];
        dqs[5][m] = dQ[5][m] - 2.0 * (dR[2][m] * D[2] + R[2] * dD[2][m])
                    + 2.0 * R[2] * dR[2][m] * s + R[2] * R[2] * ds[m];
    }
    traceless_quad(qs, tq);
    for (int m = 0; m < 3; ++m) {
        const double dtr = 0.5 * (dqs[0][m] + dqs[2][m] + dqs[5][m]);
        dtq[0][m] = 1.5 * dqs[0][m] - dtr;
        dtq[1][m] = 1.5 * dqs[1][m];
        dtq[2][m] = 1.5 * dqs[2][m] - dtr;
        dtq[3][m] = 1.5 * dqs[3][m];
        dtq[4][m] = 1.5 * dqs[4][m];
        dtq[5][m] = 1.5 * dqs[5][m] - dtr;
    }
}

double dgamma_dr(double r, double gi, double gj) {
    if (r < 1e-12)
        return 0;
    const double gam = 0.5 * (gi + gj);
    const double gexp = kGfn2Gexp;
    const double t = std::pow(r, gexp) + std::pow(gam, -gexp);
    return -std::pow(r, gexp - 1.0) / std::pow(t, 1.0 / gexp + 1.0);
}

struct SolveOut {
    double energy_eh = 0;
    double residual = 0;
    int iterations = 0;
    std::vector<double> dq;
    std::vector<Grad3> grad; // ∇E [Eh/Bohr]; empty unless want_grad
};

SolveOut solve_gfn2(const Frame& f, int charge, double scc_tol, int max_scc,
                    double mix_beta, bool want_grad) {
    const int n_at = f.nlocal();
    const int* type = f.span(kAtom_type).data();
    const StorageReal* xA = f.span(kAtom_x).data();
    const StorageReal* yA = f.span(kAtom_y).data();
    const StorageReal* zA = f.span(kAtom_z).data();

    std::vector<const Gfn2ElementRow*> el(static_cast<std::size_t>(n_at));
    std::vector<int> ao0(static_cast<std::size_t>(n_at) + 1, 0);
    std::vector<int> sh0(static_cast<std::size_t>(n_at) + 1, 0);
    double n_elec = -static_cast<double>(charge);
    for (int a = 0; a < n_at; ++a) {
        el[static_cast<std::size_t>(a)] = &element_row(type[a]);
        const auto& e = *el[static_cast<std::size_t>(a)];
        for (int s = 0; s < e.n_shell; ++s)
            n_elec += e.refocc[s];
        ao0[static_cast<std::size_t>(a) + 1] =
            ao0[static_cast<std::size_t>(a)] + n_ao_of(e);
        sh0[static_cast<std::size_t>(a) + 1] =
            sh0[static_cast<std::size_t>(a)] + e.n_shell;
    }
    const int n_ao = ao0[static_cast<std::size_t>(n_at)];
    const int n_sh = sh0[static_cast<std::size_t>(n_at)];
    if (n_ao < 1)
        throw std::runtime_error("GFN2: no basis functions");
    if (n_elec < 0 || std::fabs(n_elec - std::round(n_elec)) > 1e-9
        || static_cast<int>(std::lround(n_elec)) % 2 != 0)
        throw std::runtime_error(
            "GFN2: odd/non-integer valence count not supported (spin-gfn-xtb)");
    const int n_occ = static_cast<int>(std::lround(n_elec)) / 2;

    std::vector<std::array<double, 3>> xyz(static_cast<std::size_t>(n_at));
    for (int a = 0; a < n_at; ++a) {
        xyz[static_cast<std::size_t>(a)] = {static_cast<double>(xA[a]) * kBohr,
                                            static_cast<double>(yA[a]) * kBohr,
                                            static_cast<double>(zA[a]) * kBohr};
    }

    std::vector<Shell> shells(static_cast<std::size_t>(n_sh));
    std::vector<int> ao_shell(static_cast<std::size_t>(n_ao));
    std::vector<int> ao_atom(static_cast<std::size_t>(n_ao));
    for (int a = 0; a < n_at; ++a) {
        const auto& e = *el[static_cast<std::size_t>(a)];
        int ish = sh0[static_cast<std::size_t>(a)];
        int iao = ao0[static_cast<std::size_t>(a)];
        for (int s = 0; s < e.n_shell; ++s) {
            Shell sh;
            sh.atom = a;
            sh.l = e.l[s];
            sh.n_prim = e.n_prim[s];
            sh.ao0 = iao;
            sh.zeta = e.zeta[s];
            sto_ng(sh.n_prim, e.pqn[s], sh.l, sh.zeta, sh.alpha, sh.coeff);
            shells[static_cast<std::size_t>(ish)] = sh;
            const int nao = 2 * sh.l + 1;
            for (int u = 0; u < nao; ++u) {
                ao_shell[static_cast<std::size_t>(iao + u)] = ish;
                ao_atom[static_cast<std::size_t>(iao + u)] = a;
            }
            iao += nao;
            ++ish;
        }
    }

    std::vector<double> cn(static_cast<std::size_t>(n_at), 0.0);
    for (int a = 0; a < n_at; ++a) {
        const double rca = el[static_cast<std::size_t>(a)]->rcov * kBohr;
        for (int b = 0; b < n_at; ++b) {
            if (a == b)
                continue;
            double dx = static_cast<double>(xA[a] - xA[b]);
            double dy = static_cast<double>(yA[a] - yA[b]);
            double dz = static_cast<double>(zA[a] - zA[b]);
            f.box.min_image(dx, dy, dz);
            const double r = std::sqrt(dx * dx + dy * dy + dz * dz) * kBohr;
            const double r0 =
                rca + el[static_cast<std::size_t>(b)]->rcov * kBohr;
            cn[static_cast<std::size_t>(a)] += cn_d3(r, r0);
        }
    }

    std::vector<double> selfe(static_cast<std::size_t>(n_sh));
    std::vector<double> hardness(static_cast<std::size_t>(n_sh));
    std::vector<double> gam3(static_cast<std::size_t>(n_sh));
    std::vector<double> n0sh(static_cast<std::size_t>(n_sh));
    for (int a = 0; a < n_at; ++a) {
        const auto& e = *el[static_cast<std::size_t>(a)];
        const int is0 = sh0[static_cast<std::size_t>(a)];
        for (int s = 0; s < e.n_shell; ++s) {
            const int ish = is0 + s;
            selfe[static_cast<std::size_t>(ish)] =
                e.level[s] * kGfn2EvToEh
                - e.kcn[s] * kGfn2EvToEh * cn[static_cast<std::size_t>(a)];
            hardness[static_cast<std::size_t>(ish)] = e.hubbard * e.lgam[s];
            const double k3 = (e.l[s] == 0) ? 1.0 : 0.5;
            gam3[static_cast<std::size_t>(ish)] = e.gam3 * k3;
            n0sh[static_cast<std::size_t>(ish)] = e.refocc[s];
        }
    }

    linalg::RealMatrix S = xt::zeros<double>({n_ao, n_ao});
    linalg::RealMatrix H0 = xt::zeros<double>({n_ao, n_ao});
    std::vector<double> dip(static_cast<std::size_t>(3 * n_ao * n_ao), 0.0);
    std::vector<double> quad(static_cast<std::size_t>(6 * n_ao * n_ao), 0.0);
    auto D = [&](int k, int mu, int nu) -> double& {
        return dip[static_cast<std::size_t>((k * n_ao + mu) * n_ao + nu)];
    };
    auto Q = [&](int k, int mu, int nu) -> double& {
        return quad[static_cast<std::size_t>((k * n_ao + mu) * n_ao + nu)];
    };

    for (int ish = 0; ish < n_sh; ++ish) {
        const Shell& si = shells[static_cast<std::size_t>(ish)];
        const int a = si.atom;
        const double* Ai = xyz[static_cast<std::size_t>(a)].data();
        const int nai = 2 * si.l + 1;
        for (int jsh = 0; jsh < n_sh; ++jsh) {
            const Shell& sj = shells[static_cast<std::size_t>(jsh)];
            const int b = sj.atom;
            const double* Aj = xyz[static_cast<std::size_t>(b)].data();
            const int naj = 2 * sj.l + 1;
            double r2 = 0;
            for (int k = 0; k < 3; ++k) {
                const double d = Ai[k] - Aj[k];
                r2 += d * d;
            }
            const double r = std::sqrt(r2);
            const double radi = el[static_cast<std::size_t>(a)]->rcov * kBohr;
            const double radj = el[static_cast<std::size_t>(b)]->rcov * kBohr;
            const double rr =
                (r > 1e-12) ? std::sqrt(std::sqrt(r2 / (radi + radj))) : 0.0;
            const double shpi =
                1.0
                + el[static_cast<std::size_t>(a)]->shpoly[si.l == 0 ? 0 : 1]
                      * rr;
            const double shpj =
                1.0
                + el[static_cast<std::size_t>(b)]->shpoly[sj.l == 0 ? 0 : 1]
                      * rr;
            const double shpoly = shpi * shpj;
            const double zi = si.zeta;
            const double zj = sj.zeta;
            const double zij =
                std::pow(2.0 * std::sqrt(zi * zj) / (zi + zj), kGfn2Wexp);
            const double den = el[static_cast<std::size_t>(a)]->en
                               - el[static_cast<std::size_t>(b)]->en;
            const double enp = 1.0 + kGfn2Enscale * den * den;
            const double hsc = zij * k_shell(si.l, sj.l) * enp;
            // On-site: ½(εi+εj) S, no k-scale (tblite get_hamiltonian).
            const double hij = 0.5
                               * (selfe[static_cast<std::size_t>(ish)]
                                  + selfe[static_cast<std::size_t>(jsh)])
                               * ((a == b) ? 1.0 : hsc * shpoly);
            for (int u = 0; u < nai; ++u) {
                for (int v = 0; v < naj; ++v) {
                    double s = 0, d[3], q6[6];
                    shell_ints(si, Ai, sj, Aj, u, v, s, d, q6);
                    const int mu = si.ao0 + u;
                    const int nu = sj.ao0 + v;
                    S(mu, nu) = s;
                    H0(mu, nu) = s * hij;
                    for (int k = 0; k < 3; ++k)
                        D(k, mu, nu) = d[k];
                    for (int k = 0; k < 6; ++k)
                        Q(k, mu, nu) = q6[k];
                }
            }
        }
    }
    std::vector<double> gamma(static_cast<std::size_t>(n_sh * n_sh), 0.0);
    for (int ish = 0; ish < n_sh; ++ish) {
        const int a = shells[static_cast<std::size_t>(ish)].atom;
        for (int jsh = 0; jsh < n_sh; ++jsh) {
            const int b = shells[static_cast<std::size_t>(jsh)].atom;
            const double gi = hardness[static_cast<std::size_t>(ish)];
            const double gj = hardness[static_cast<std::size_t>(jsh)];
            const double gam = 0.5 * (gi + gj);
            double r = 0;
            if (a != b) {
                double dx = static_cast<double>(xA[a] - xA[b]);
                double dy = static_cast<double>(yA[a] - yA[b]);
                double dz = static_cast<double>(zA[a] - zA[b]);
                f.box.min_image(dx, dy, dz);
                r = std::sqrt(dx * dx + dy * dy + dz * dz) * kBohr;
            }
            const double gexp = kGfn2Gexp;
            double gij;
            if (r < 1e-12)
                gij = gam;
            else
                gij = 1.0
                      / std::pow(std::pow(r, gexp) + std::pow(gam, -gexp),
                                 1.0 / gexp);
            gamma[static_cast<std::size_t>(ish) * n_sh
                  + static_cast<std::size_t>(jsh)] = gij;
        }
    }

    auto repulsion = [&]() {
        double e = 0;
        for (int a = 0; a < n_at; ++a) {
            const auto& ea = *el[static_cast<std::size_t>(a)];
            for (int b = a + 1; b < n_at; ++b) {
                const auto& eb = *el[static_cast<std::size_t>(b)];
                double dx = static_cast<double>(xA[a] - xA[b]);
                double dy = static_cast<double>(yA[a] - yA[b]);
                double dz = static_cast<double>(zA[a] - zA[b]);
                f.box.min_image(dx, dy, dz);
                const double r = std::sqrt(dx * dx + dy * dy + dz * dz) * kBohr;
                if (r < 1e-12)
                    continue;
                // tblite: kexp unless *both* partners have Z≤2 (then kexplight).
                const double kexp =
                    (ea.z > 2 || eb.z > 2) ? kGfn2Kexp : kGfn2KexpLight;
                const double al = std::sqrt(ea.alpha_rep * eb.alpha_rep);
                const double zf = ea.zeff * eb.zeff;
                e += zf * std::exp(-al * std::pow(r, kexp))
                     / std::pow(r, kGfn2Rexp);
            }
        }
        return e;
    };

    auto d4_energy = [&]() {
        std::vector<int> Z(static_cast<std::size_t>(n_at));
        std::vector<std::array<double, 3>> pos(static_cast<std::size_t>(n_at));
        for (int a = 0; a < n_at; ++a) {
            Z[static_cast<std::size_t>(a)] = type[a];
            pos[static_cast<std::size_t>(a)] = xyz[static_cast<std::size_t>(a)];
        }
        const double cell[3] = {20.0 * kBohr, 20.0 * kBohr, 20.0 * kBohr};
        return scf::dftd4_energy(Z, pos, cell, /*periodic=*/false, kGfn2D4S6,
                                 kGfn2D4S8, kGfn2D4A1, kGfn2D4A2, 3.0, 2.0, 6.0,
                                 static_cast<double>(charge), 30.0, 60.0,
                                 kGfn2D4S9, 40.0);
    };

    std::vector<double> qsh(static_cast<std::size_t>(n_sh), 0.0);
    std::vector<double> qsh_out(static_cast<std::size_t>(n_sh), 0.0);
    std::vector<double> qat(static_cast<std::size_t>(n_at), 0.0);
    std::vector<std::array<double, 3>> dipat(static_cast<std::size_t>(n_at));
    std::vector<std::array<double, 6>> quadat(static_cast<std::size_t>(n_at));
    internal::ChargeMixer mixer(mix_beta, 6);
    linalg::RealMatrix P = xt::zeros<double>({n_ao, n_ao});
    linalg::RealMatrix Cmat = xt::zeros<double>({n_ao, n_ao});
    linalg::RealVector eps_mo = xt::zeros<double>({n_ao});
    std::vector<double> vsh(static_cast<std::size_t>(n_sh), 0.0);
    SolveOut out;
    bool converged = false;

    auto aes_energy = [&]() {
        double e = 0;
        for (int a = 0; a < n_at; ++a) {
            const auto& ea = *el[static_cast<std::size_t>(a)];
            const double* ra = xyz[static_cast<std::size_t>(a)].data();
            // AXC (dipole + packed quadrupole; off-diag scale 2, tblite)
            const auto& mu = dipat[static_cast<std::size_t>(a)];
            const auto& th = quadat[static_cast<std::size_t>(a)];
            e += ea.dkernel * (mu[0] * mu[0] + mu[1] * mu[1] + mu[2] * mu[2]);
            e += ea.qkernel
                 * (th[0] * th[0] + 2.0 * th[1] * th[1] + th[2] * th[2]
                    + 2.0 * th[3] * th[3] + 2.0 * th[4] * th[4]
                    + th[5] * th[5]);
            const double arg =
                cn[static_cast<std::size_t>(a)] - ea.mpvcn - kGfn2AesShift;
            const double t1 = std::exp(-kGfn2AesKexp * arg);
            // p_rad / rmax in the GFN2 set are Bohr.
            const double radA =
                ea.mprad + (kGfn2AesRmax - ea.mprad) / (1.0 + t1);
            for (int b = 0; b < n_at; ++b) {
                if (a == b)
                    continue;
                const auto& eb = *el[static_cast<std::size_t>(b)];
                const double* rb = xyz[static_cast<std::size_t>(b)].data();
                double vec[3] = {ra[0] - rb[0], ra[1] - rb[1], ra[2] - rb[2]};
                const double r1 = std::sqrt(vec[0] * vec[0] + vec[1] * vec[1]
                                            + vec[2] * vec[2]);
                if (r1 < 1e-12)
                    continue;
                const double argb =
                    cn[static_cast<std::size_t>(b)] - eb.mpvcn - kGfn2AesShift;
                const double t1b = std::exp(-kGfn2AesKexp * argb);
                const double radB =
                    eb.mprad + (kGfn2AesRmax - eb.mprad) / (1.0 + t1b);
                const double g1 = 1.0 / r1;
                const double g3 = g1 * g1 * g1;
                const double g5 = g3 * g1 * g1;
                const double rr = 0.5 * (radA + radB) * g1;
                const double fdmp3 =
                    1.0 / (1.0 + 6.0 * std::pow(rr, kGfn2AesDmp3));
                const double fdmp5 =
                    1.0 / (1.0 + 6.0 * std::pow(rr, kGfn2AesDmp5));
                const auto& mua = dipat[static_cast<std::size_t>(a)];
                const auto& mub = dipat[static_cast<std::size_t>(b)];
                const auto& tha = quadat[static_cast<std::size_t>(a)];
                const double qb = -qat[static_cast<std::size_t>(b)];
                // tblite get_energy: vd = amat_sd q; E += μ·vd  (ordered i≠j)
                const double sd =
                    (mua[0] * vec[0] + mua[1] * vec[1] + mua[2] * vec[2]) * g3
                    * fdmp3;
                e += qb * sd;
                const double mdot =
                    mua[0] * mub[0] + mua[1] * mub[1] + mua[2] * mub[2];
                const double mrv =
                    (mua[0] * vec[0] + mua[1] * vec[1] + mua[2] * vec[2])
                    * (mub[0] * vec[0] + mub[1] * vec[1] + mub[2] * vec[2]);
                // ½ μ T μ with T filled for every ordered pair
                e += 0.5 * (mdot * g3 * fdmp5 - 3.0 * mrv * g5 * fdmp5);
                // Charge–quadrupole. Dipole CAMM is +½P(D−RS); that operator
                // is odd in R so the foothold sign already matches xtb AES.
                // Quadrupole is even in R, so the contraction follows xtb
                // mmompop (−P·Q). /3 converts the 3/2-traceless packed Θ:rr
                // into the GFN2 0d kernel (xtb aniso_electro gab5).
                const double sq =
                    -(tha[0] * vec[0] * vec[0]
                      + 2.0 * tha[1] * vec[0] * vec[1]
                      + tha[2] * vec[1] * vec[1]
                      + 2.0 * tha[3] * vec[0] * vec[2]
                      + 2.0 * tha[4] * vec[1] * vec[2]
                      + tha[5] * vec[2] * vec[2])
                    * g5 * fdmp5 / 3.0;
                e += qb * sq;
            }
        }
        return e;
    };

    auto fill_aes_pot = [&](std::vector<double>& vat, std::vector<Grad3>& vdp,
                            std::vector<std::array<double, 6>>& vqp) {
        vat.assign(static_cast<std::size_t>(n_at), 0.0);
        vdp.assign(static_cast<std::size_t>(n_at), Grad3{0, 0, 0});
        vqp.assign(static_cast<std::size_t>(n_at),
                   std::array<double, 6>{0, 0, 0, 0, 0, 0});
        for (int a = 0; a < n_at; ++a) {
            const auto& ea = *el[static_cast<std::size_t>(a)];
            const auto& mua = dipat[static_cast<std::size_t>(a)];
            const auto& tha = quadat[static_cast<std::size_t>(a)];
            vdp[static_cast<std::size_t>(a)] = {2.0 * ea.dkernel * mua[0],
                                                2.0 * ea.dkernel * mua[1],
                                                2.0 * ea.dkernel * mua[2]};
            vqp[static_cast<std::size_t>(a)] = {
                2.0 * ea.qkernel * tha[0], 4.0 * ea.qkernel * tha[1],
                2.0 * ea.qkernel * tha[2], 4.0 * ea.qkernel * tha[3],
                4.0 * ea.qkernel * tha[4], 2.0 * ea.qkernel * tha[5]};
            const double* ra = xyz[static_cast<std::size_t>(a)].data();
            const double arg =
                cn[static_cast<std::size_t>(a)] - ea.mpvcn - kGfn2AesShift;
            const double t1 = std::exp(-kGfn2AesKexp * arg);
            const double radA =
                ea.mprad + (kGfn2AesRmax - ea.mprad) / (1.0 + t1);
            for (int b = 0; b < n_at; ++b) {
                if (a == b)
                    continue;
                const auto& eb = *el[static_cast<std::size_t>(b)];
                const double* rb = xyz[static_cast<std::size_t>(b)].data();
                const auto& mub = dipat[static_cast<std::size_t>(b)];
                const double qb = -qat[static_cast<std::size_t>(b)];
                double vec[3] = {ra[0] - rb[0], ra[1] - rb[1], ra[2] - rb[2]};
                const double r1 = std::sqrt(vec[0] * vec[0] + vec[1] * vec[1]
                                            + vec[2] * vec[2]);
                if (r1 < 1e-12)
                    continue;
                const double argb =
                    cn[static_cast<std::size_t>(b)] - eb.mpvcn - kGfn2AesShift;
                const double t1b = std::exp(-kGfn2AesKexp * argb);
                const double radB =
                    eb.mprad + (kGfn2AesRmax - eb.mprad) / (1.0 + t1b);
                const double g1 = 1.0 / r1;
                const double g3 = g1 * g1 * g1;
                const double g5 = g3 * g1 * g1;
                const double rrd = 0.5 * (radA + radB) * g1;
                const double fdmp3 =
                    1.0 / (1.0 + 6.0 * std::pow(rrd, kGfn2AesDmp3));
                const double fdmp5 =
                    1.0 / (1.0 + 6.0 * std::pow(rrd, kGfn2AesDmp5));
                const double sd =
                    (mua[0] * vec[0] + mua[1] * vec[1] + mua[2] * vec[2]) * g3
                    * fdmp3;
                for (int k = 0; k < 3; ++k)
                    vdp[static_cast<std::size_t>(a)][static_cast<std::size_t>(k)] +=
                        qb * vec[k] * g3 * fdmp3;
                vat[static_cast<std::size_t>(b)] += -sd;
                const double mub_r =
                    mub[0] * vec[0] + mub[1] * vec[1] + mub[2] * vec[2];
                for (int k = 0; k < 3; ++k)
                    vdp[static_cast<std::size_t>(a)][static_cast<std::size_t>(k)] +=
                        mub[k] * g3 * fdmp5
                        - 3.0 * vec[k] * mub_r * g5 * fdmp5;
                const double rrq =
                    tha[0] * vec[0] * vec[0] + 2.0 * tha[1] * vec[0] * vec[1]
                    + tha[2] * vec[1] * vec[1] + 2.0 * tha[3] * vec[0] * vec[2]
                    + 2.0 * tha[4] * vec[1] * vec[2]
                    + tha[5] * vec[2] * vec[2];
                const double sq = -rrq * g5 * fdmp5 / 3.0;
                const double facq = qb * (-g5 * fdmp5 / 3.0);
                auto& vqa = vqp[static_cast<std::size_t>(a)];
                vqa[0] += facq * vec[0] * vec[0];
                vqa[1] += facq * 2.0 * vec[0] * vec[1];
                vqa[2] += facq * vec[1] * vec[1];
                vqa[3] += facq * 2.0 * vec[0] * vec[2];
                vqa[4] += facq * 2.0 * vec[1] * vec[2];
                vqa[5] += facq * vec[2] * vec[2];
                vat[static_cast<std::size_t>(b)] += -sq;
            }
        }
    };

    std::vector<double> vat(static_cast<std::size_t>(n_at), 0.0);
    std::vector<Grad3> vdp(static_cast<std::size_t>(n_at), Grad3{0, 0, 0});
    std::vector<std::array<double, 6>> vqp(
        static_cast<std::size_t>(n_at), std::array<double, 6>{0, 0, 0, 0, 0, 0});

    for (int it = 0; it < max_scc; ++it) {
        fill_aes_pot(vat, vdp, vqp);
        for (int ish = 0; ish < n_sh; ++ish) {
            double s = 0;
            for (int jsh = 0; jsh < n_sh; ++jsh)
                s += gamma[static_cast<std::size_t>(ish) * n_sh
                           + static_cast<std::size_t>(jsh)]
                     * qsh[static_cast<std::size_t>(jsh)];
            s += gam3[static_cast<std::size_t>(ish)]
                 * qsh[static_cast<std::size_t>(ish)]
                 * qsh[static_cast<std::size_t>(ish)];
            const int a = shells[static_cast<std::size_t>(ish)].atom;
            vsh[static_cast<std::size_t>(ish)] =
                s + vat[static_cast<std::size_t>(a)];
        }
        linalg::RealMatrix H = H0;
        for (int mu = 0; mu < n_ao; ++mu) {
            const int ish = ao_shell[static_cast<std::size_t>(mu)];
            const int a = ao_atom[static_cast<std::size_t>(mu)];
            const double* Ra = xyz[static_cast<std::size_t>(a)].data();
            for (int nu = 0; nu < n_ao; ++nu) {
                const int jsh = ao_shell[static_cast<std::size_t>(nu)];
                const int b = ao_atom[static_cast<std::size_t>(nu)];
                const double* Rb = xyz[static_cast<std::size_t>(b)].data();
                const double sij = S(mu, nu);
                H(mu, nu) += 0.5 * sij
                             * (vsh[static_cast<std::size_t>(ish)]
                                + vsh[static_cast<std::size_t>(jsh)]);
                for (int k = 0; k < 3; ++k) {
                    const double d = D(k, mu, nu);
                    H(mu, nu) +=
                        0.5
                        * (vdp[static_cast<std::size_t>(a)]
                              [static_cast<std::size_t>(k)]
                               * (d - Ra[k] * sij)
                           + vdp[static_cast<std::size_t>(b)]
                                [static_cast<std::size_t>(k)]
                                 * (d - Rb[k] * sij));
                }
                {
                    const double q0[6] = {Q(0, mu, nu), Q(1, mu, nu),
                                          Q(2, mu, nu), Q(3, mu, nu),
                                          Q(4, mu, nu), Q(5, mu, nu)};
                    const double d0[3] = {D(0, mu, nu), D(1, mu, nu),
                                          D(2, mu, nu)};
                    double tqA[6];
                    origin_shift_quad(q0, d0, Ra, sij, tqA);
                    const auto& va = vqp[static_cast<std::size_t>(a)];
                    for (int k = 0; k < 6; ++k)
                        H(mu, nu) += 0.5 * va[static_cast<std::size_t>(k)]
                                     * tqA[k];
                    double tqB[6];
                    origin_shift_quad(q0, d0, Rb, sij, tqB);
                    const auto& vb = vqp[static_cast<std::size_t>(b)];
                    for (int k = 0; k < 6; ++k)
                        H(mu, nu) += 0.5 * vb[static_cast<std::size_t>(k)]
                                     * tqB[k];
                }
            }
        }
        auto sol = linalg::diagonalize_fock_generalized(H, S);
        eps_mo = sol.first;
        Cmat = sol.second;
        const auto& C = Cmat;
        P = xt::zeros<double>({n_ao, n_ao});
        for (int o = 0; o < n_occ; ++o)
            for (int mu = 0; mu < n_ao; ++mu)
                for (int nu = 0; nu < n_ao; ++nu)
                    P(mu, nu) += 2.0 * C(mu, o) * C(nu, o);

        const linalg::RealMatrix PS = xt::linalg::dot(P, S);
        for (int a = 0; a < n_at; ++a) {
            qat[static_cast<std::size_t>(a)] = 0;
            dipat[static_cast<std::size_t>(a)] = {0, 0, 0};
            quadat[static_cast<std::size_t>(a)] = {0, 0, 0, 0, 0, 0};
            const auto& e = *el[static_cast<std::size_t>(a)];
            const int is0 = sh0[static_cast<std::size_t>(a)];
            for (int s = 0; s < e.n_shell; ++s) {
                const int ish = is0 + s;
                const Shell& sh = shells[static_cast<std::size_t>(ish)];
                const int nao = 2 * sh.l + 1;
                double pop = 0;
                for (int u = 0; u < nao; ++u)
                    pop += PS(sh.ao0 + u, sh.ao0 + u);
                // Electron excess Δn = pop − n0 (DFTB/xTB SCC). Mulliken
                // charge written to Frame is −Δn.
                qsh_out[static_cast<std::size_t>(ish)] =
                    pop - n0sh[static_cast<std::size_t>(ish)];
                qat[static_cast<std::size_t>(a)] +=
                    qsh_out[static_cast<std::size_t>(ish)];
            }
            const double* Ra = xyz[static_cast<std::size_t>(a)].data();
            const int ia = ao0[static_cast<std::size_t>(a)];
            const int na = n_ao_of(e);
            // xtb mmompop: bra on A (same Mulliken split as the dipole).
            // Two-centre origin-shifted Θ is kept on s-only H.
            for (int u = 0; u < na; ++u) {
                for (int b2 = 0; b2 < n_at; ++b2) {
                    const int ib = ao0[static_cast<std::size_t>(b2)];
                    const int nb = n_ao_of(*el[static_cast<std::size_t>(b2)]);
                    for (int v = 0; v < nb; ++v) {
                        const double p = P(ia + u, ib + v);
                        const double s = S(ia + u, ib + v);
                        for (int k = 0; k < 3; ++k)
                            dipat[static_cast<std::size_t>(a)]
                                 [static_cast<std::size_t>(k)] +=
                                0.5 * p
                                * (D(k, ia + u, ib + v) - Ra[k] * s);
                        const double q0[6] = {
                            Q(0, ia + u, ib + v), Q(1, ia + u, ib + v),
                            Q(2, ia + u, ib + v), Q(3, ia + u, ib + v),
                            Q(4, ia + u, ib + v), Q(5, ia + u, ib + v)};
                        const double d0[3] = {D(0, ia + u, ib + v),
                                              D(1, ia + u, ib + v),
                                              D(2, ia + u, ib + v)};
                        double tq[6];
                        origin_shift_quad(q0, d0, Ra, s, tq);
                        for (int k = 0; k < 6; ++k)
                            quadat[static_cast<std::size_t>(a)]
                                  [static_cast<std::size_t>(k)] +=
                                0.5 * p * tq[k];
                    }
                }
            }
        }
        double residual = 0;
        for (int ish = 0; ish < n_sh; ++ish)
            residual = std::max(residual,
                                std::abs(qsh_out[static_cast<std::size_t>(ish)]
                                         - qsh[static_cast<std::size_t>(ish)]));
        out.residual = residual;
        out.iterations = it + 1;
        if (residual < scc_tol) {
            qsh = qsh_out;
            converged = true;
            break;
        }
        qsh = mixer.next(qsh, qsh_out);
    }
    if (!converged)
        throw std::runtime_error("GFN2: SCC did not converge in max_scc="
                                 + std::to_string(max_scc)
                                 + " (residual=" + std::to_string(out.residual)
                                 + "); increase max_scc or lower mix_beta");

    double e_h0 = 0;
    for (int mu = 0; mu < n_ao; ++mu)
        for (int nu = 0; nu < n_ao; ++nu)
            e_h0 += P(mu, nu) * H0(mu, nu);
    double e_ies = 0;
    for (int ish = 0; ish < n_sh; ++ish)
        for (int jsh = 0; jsh < n_sh; ++jsh)
            e_ies += 0.5 * qsh[static_cast<std::size_t>(ish)]
                     * gamma[static_cast<std::size_t>(ish) * n_sh
                             + static_cast<std::size_t>(jsh)]
                     * qsh[static_cast<std::size_t>(jsh)];
    double e3 = 0;
    for (int ish = 0; ish < n_sh; ++ish) {
        const double q = qsh[static_cast<std::size_t>(ish)];
        e3 += q * q * q * gam3[static_cast<std::size_t>(ish)] / 3.0;
    }
    const double e_rep = repulsion();
    const double e_aes = aes_energy();

    std::vector<int> Z(static_cast<std::size_t>(n_at));
    for (int a = 0; a < n_at; ++a)
        Z[static_cast<std::size_t>(a)] = type[a];
    const double cell[3] = {20.0 * kBohr, 20.0 * kBohr, 20.0 * kBohr};
    scf::D4Result d4;
    double e_d4 = 0;
    if (want_grad) {
        d4 = scf::dftd4(Z, xyz, cell, /*periodic=*/false, kGfn2D4S6, kGfn2D4S8,
                        kGfn2D4A1, kGfn2D4A2, 3.0, 2.0, 6.0,
                        static_cast<double>(charge), 30.0, 60.0, kGfn2D4S9,
                        40.0);
        e_d4 = d4.energy;
    } else {
        e_d4 = d4_energy();
    }

    out.energy_eh = e_h0 + e_ies + e3 + e_rep + e_d4 + e_aes;
    out.dq.assign(static_cast<std::size_t>(n_at), 0.0);
    for (int a = 0; a < n_at; ++a)
        out.dq[static_cast<std::size_t>(a)] = -qat[static_cast<std::size_t>(a)];
    // Energy pieces stay on the existing kernels; gradient is additive.

    if (!want_grad)
        return out;

    // Converged CAMM → AES potential for quadrupole (and dipole) chains.
    fill_aes_pot(vat, vdp, vqp);

    linalg::RealMatrix W = xt::zeros<double>({n_ao, n_ao});
    for (int o = 0; o < n_occ; ++o) {
        const double wfac = 2.0 * eps_mo(o);
        for (int mu = 0; mu < n_ao; ++mu) {
            const double cmu = Cmat(mu, o);
            for (int nu = 0; nu < n_ao; ++nu)
                W(mu, nu) += wfac * cmu * Cmat(nu, o);
        }
    }

    // ∂E_AES/∂μ including AXC (frozen kernel radii). Used for CAMM chains.
    std::vector<Grad3> vd(static_cast<std::size_t>(n_at), Grad3{0, 0, 0});
    for (int a = 0; a < n_at; ++a) {
        const auto& ea = *el[static_cast<std::size_t>(a)];
        const auto& mua = dipat[static_cast<std::size_t>(a)];
        vd[static_cast<std::size_t>(a)] = {2.0 * ea.dkernel * mua[0],
                                           2.0 * ea.dkernel * mua[1],
                                           2.0 * ea.dkernel * mua[2]};
        const double* ra = xyz[static_cast<std::size_t>(a)].data();
        const double arg =
            cn[static_cast<std::size_t>(a)] - ea.mpvcn - kGfn2AesShift;
        const double t1 = std::exp(-kGfn2AesKexp * arg);
        const double radA =
            ea.mprad + (kGfn2AesRmax - ea.mprad) / (1.0 + t1);
        for (int b = 0; b < n_at; ++b) {
            if (a == b)
                continue;
            const auto& eb = *el[static_cast<std::size_t>(b)];
            const double* rb = xyz[static_cast<std::size_t>(b)].data();
            const auto& mub = dipat[static_cast<std::size_t>(b)];
            const double qb = -qat[static_cast<std::size_t>(b)];
            double vec[3] = {ra[0] - rb[0], ra[1] - rb[1], ra[2] - rb[2]};
            const double r1 = std::sqrt(vec[0] * vec[0] + vec[1] * vec[1]
                                        + vec[2] * vec[2]);
            if (r1 < 1e-12)
                continue;
            const double argb =
                cn[static_cast<std::size_t>(b)] - eb.mpvcn - kGfn2AesShift;
            const double t1b = std::exp(-kGfn2AesKexp * argb);
            const double radB =
                eb.mprad + (kGfn2AesRmax - eb.mprad) / (1.0 + t1b);
            const double g1 = 1.0 / r1;
            const double g3 = g1 * g1 * g1;
            const double g5 = g3 * g1 * g1;
            const double rrd = 0.5 * (radA + radB) * g1;
            const double fdmp3 =
                1.0 / (1.0 + 6.0 * std::pow(rrd, kGfn2AesDmp3));
            const double fdmp5 =
                1.0 / (1.0 + 6.0 * std::pow(rrd, kGfn2AesDmp5));
            for (int k = 0; k < 3; ++k)
                vd[static_cast<std::size_t>(a)][static_cast<std::size_t>(k)] +=
                    qb * vec[k] * g3 * fdmp3;
            const double mrv =
                (mua[0] * vec[0] + mua[1] * vec[1] + mua[2] * vec[2])
                * (mub[0] * vec[0] + mub[1] * vec[1] + mub[2] * vec[2]);
            (void)mrv;
            // Ordered a≠b: E_dd = ½ (μa·μb g3 − 3 (μa·r)(μb·r) g5) fdmp5
            // both directions ⇒ vd_a += μb g3 fdmp5 − 3 r (μb·r) g5 fdmp5.
            const double mub_r =
                mub[0] * vec[0] + mub[1] * vec[1] + mub[2] * vec[2];
            for (int k = 0; k < 3; ++k)
                vd[static_cast<std::size_t>(a)][static_cast<std::size_t>(k)] +=
                    mub[k] * g3 * fdmp5 - 3.0 * vec[k] * mub_r * g5 * fdmp5;
        }
    }

    std::vector<Grad3> g(static_cast<std::size_t>(n_at), Grad3{0, 0, 0});
    std::vector<double> dEdcn(static_cast<std::size_t>(n_at), 0.0);
    std::vector<Grad3> dcndr(
        static_cast<std::size_t>(n_at) * static_cast<std::size_t>(n_at),
        Grad3{0, 0, 0});

    for (int a = 0; a < n_at; ++a) {
        const double rca = el[static_cast<std::size_t>(a)]->rcov * kBohr;
        for (int b = 0; b < a; ++b) {
            const double rcb = el[static_cast<std::size_t>(b)]->rcov * kBohr;
            double dx = static_cast<double>(xA[a] - xA[b]);
            double dy = static_cast<double>(yA[a] - yA[b]);
            double dz = static_cast<double>(zA[a] - zA[b]);
            f.box.min_image(dx, dy, dz);
            const double dxb = dx * kBohr;
            const double dyb = dy * kBohr;
            const double dzb = dz * kBohr;
            const double r = std::sqrt(dxb * dxb + dyb * dyb + dzb * dzb);
            if (r < 1e-12)
                continue;
            const double r0 = rca + rcb;
            const double dc = dcn_d3_dr(r, r0);
            const double invr = 1.0 / r;
            const double dvec[3] = {dxb * invr * dc, dyb * invr * dc,
                                    dzb * invr * dc};
            // CN_a and CN_b each pick up this pair.
            for (int k = 0; k < 3; ++k) {
                dcndr[static_cast<std::size_t>(a * n_at + a)]
                     [static_cast<std::size_t>(k)] += dvec[k];
                dcndr[static_cast<std::size_t>(b * n_at + a)]
                     [static_cast<std::size_t>(k)] -= dvec[k];
                dcndr[static_cast<std::size_t>(a * n_at + b)]
                     [static_cast<std::size_t>(k)] += dvec[k];
                dcndr[static_cast<std::size_t>(b * n_at + b)]
                     [static_cast<std::size_t>(k)] -= dvec[k];
            }
        }
    }

    // On-site CN: diagonal AO populations of each shell (tblite).
    for (int a = 0; a < n_at; ++a) {
        const auto& e = *el[static_cast<std::size_t>(a)];
        const int is0 = sh0[static_cast<std::size_t>(a)];
        for (int s = 0; s < e.n_shell; ++s) {
            const int ish = is0 + s;
            const Shell& sh = shells[static_cast<std::size_t>(ish)];
            const double dsedcn = -e.kcn[s] * kGfn2EvToEh;
            const int nao = 2 * sh.l + 1;
            double pop = 0;
            for (int u = 0; u < nao; ++u)
                pop += P(sh.ao0 + u, sh.ao0 + u);
            dEdcn[static_cast<std::size_t>(a)] += dsedcn * pop;
        }
    }

    for (int a = 0; a < n_at; ++a) {
        const auto& ea = *el[static_cast<std::size_t>(a)];
        const double* Ai = xyz[static_cast<std::size_t>(a)].data();
        const int is0 = sh0[static_cast<std::size_t>(a)];
        for (int b = 0; b < a; ++b) {
            const auto& eb = *el[static_cast<std::size_t>(b)];
            const double* Aj = xyz[static_cast<std::size_t>(b)].data();
            const int js0 = sh0[static_cast<std::size_t>(b)];
            double vec[3] = {Ai[0] - Aj[0], Ai[1] - Aj[1], Ai[2] - Aj[2]};
            const double r2 =
                vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2];
            if (r2 < 1e-16)
                continue;
            const double r = std::sqrt(r2);
            const double radi = ea.rcov * kBohr;
            const double radj = eb.rcov * kBohr;
            const double rr = std::sqrt(std::sqrt(r2 / (radi + radj)));
            const double den = ea.en - eb.en;
            const double enp = 1.0 + kGfn2Enscale * den * den;

            Grad3 dG{0, 0, 0};
            double dcni = 0;
            double dcnj = 0;

            const int nsh_a = ea.n_shell;
            const int nsh_b = eb.n_shell;
            for (int sa = 0; sa < nsh_a; ++sa) {
                const int ish = is0 + sa;
                const Shell& si = shells[static_cast<std::size_t>(ish)];
                const double shpi = 1.0 + ea.shpoly[si.l == 0 ? 0 : 1] * rr;
                const int nai = 2 * si.l + 1;
                for (int sb = 0; sb < nsh_b; ++sb) {
                    const int jsh = js0 + sb;
                    const Shell& sj = shells[static_cast<std::size_t>(jsh)];
                    const double shpj =
                        1.0 + eb.shpoly[sj.l == 0 ? 0 : 1] * rr;
                    const double shpoly = shpi * shpj;
                    const double zi = si.zeta;
                    const double zj = sj.zeta;
                    const double zij =
                        std::pow(2.0 * std::sqrt(zi * zj) / (zi + zj),
                                 kGfn2Wexp);
                    const double hsc = zij * k_shell(si.l, sj.l) * enp;
                    const double hs = hsc * shpoly;
                    const double hij =
                        0.5
                        * (selfe[static_cast<std::size_t>(ish)]
                           + selfe[static_cast<std::size_t>(jsh)])
                        * hs;
                    const double dshpoly =
                        (shpi * eb.shpoly[sj.l == 0 ? 0 : 1]
                         + shpj * ea.shpoly[si.l == 0 ? 0 : 1])
                        * 0.5 * rr / r2;
                    const double dsv[3] = {dshpoly / shpoly * vec[0],
                                           dshpoly / shpoly * vec[1],
                                           dshpoly / shpoly * vec[2]};
                    const double dsed_i =
                        -ea.kcn[sa] * kGfn2EvToEh * hs;
                    const double dsed_j =
                        -eb.kcn[sb] * kGfn2EvToEh * hs;
                    const double vscale =
                        0.5
                        * (vsh[static_cast<std::size_t>(ish)]
                           + vsh[static_cast<std::size_t>(jsh)]);
                    const double fscale = hij + vscale;
                    const int naj = 2 * sj.l + 1;
                    for (int u = 0; u < nai; ++u) {
                        const int mu = si.ao0 + u;
                        for (int v = 0; v < naj; ++v) {
                            const int nu = sj.ao0 + v;
                            double sij = 0;
                            double ds[3];
                            double Dip[3];
                            double dDip[3][3];
                            double Qint[6];
                            double dQint[6][3];
                            shell_sdq_grad(si, Ai, u, sj, Aj, v, sij, ds, Dip,
                                           dDip, Qint, dQint);
                            const double pij = P(mu, nu);
                            const double hpij = pij * hij;
                            const double sval =
                                2.0 * (pij * fscale - W(mu, nu));
                            double dRa[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
                            double dRb[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
                            double tqA[6];
                            double dtqA[6][3] = {};
                            double tqB[6];
                            double dtqB[6][3] = {};
                            origin_shift_quad_grad(Qint, dQint, Dip, dDip, Ai,
                                                   dRa, sij, ds, tqA, dtqA);
                            origin_shift_quad_grad(Qint, dQint, Dip, dDip, Aj,
                                                   dRb, sij, ds, tqB, dtqB);
                            (void)tqA;
                            (void)tqB;
                            const auto& vqa = vqp[static_cast<std::size_t>(a)];
                            const auto& vqb = vqp[static_cast<std::size_t>(b)];
                            for (int k = 0; k < 3; ++k) {
                                dG[static_cast<std::size_t>(k)] +=
                                    sval * ds[k] + 2.0 * hpij * sij * dsv[k];
                                // Frozen-P CAMM dipole: μ = ½ P (D − R S).
                                double dmu_a = 0;
                                double dmu_b = 0;
                                for (int al = 0; al < 3; ++al) {
                                    const double dD = dDip[al][k];
                                    const double dmuA =
                                        0.5 * pij
                                        * (dD - ((al == k) ? sij : 0.0)
                                           - Ai[al] * ds[k]);
                                    const double dmuB =
                                        0.5 * pij * (dD - Aj[al] * ds[k]);
                                    dmu_a += vd[static_cast<std::size_t>(a)]
                                               [static_cast<std::size_t>(al)]
                                             * dmuA;
                                    dmu_b += vd[static_cast<std::size_t>(b)]
                                               [static_cast<std::size_t>(al)]
                                             * dmuB;
                                }
                                dG[static_cast<std::size_t>(k)] += dmu_a + dmu_b;
                                // Frozen-P CAMM quadrupole (xtb mmompop),
                                // including two-centre Θ on H.
                                double dth = 0;
                                for (int c = 0; c < 6; ++c)
                                    dth += 0.5 * pij
                                           * (vqa[static_cast<std::size_t>(c)]
                                                  * dtqA[c][k]
                                              + vqb[static_cast<std::size_t>(c)]
                                                    * dtqB[c][k]);
                                dG[static_cast<std::size_t>(k)] += dth;
                            }
                            dcni += dsed_i * pij * sij;
                            dcnj += dsed_j * pij * sij;
                        }
                    }
                }
            }

            // IES: Σ_{ish on a, jsh on b} q_i q_j dγ/dR_a
            double dEies_dr = 0;
            for (int sa = 0; sa < nsh_a; ++sa) {
                const int ish = is0 + sa;
                for (int sb = 0; sb < nsh_b; ++sb) {
                    const int jsh = js0 + sb;
                    dEies_dr += qsh[static_cast<std::size_t>(ish)]
                                * qsh[static_cast<std::size_t>(jsh)]
                                * dgamma_dr(r, hardness[static_cast<std::size_t>(ish)],
                                            hardness[static_cast<std::size_t>(jsh)]);
                }
            }
            const double invr = 1.0 / r;
            for (int k = 0; k < 3; ++k)
                dG[static_cast<std::size_t>(k)] += dEies_dr * vec[k] * invr;

            // Repulsion
            const double kexp =
                (ea.z > 2 || eb.z > 2) ? kGfn2Kexp : kGfn2KexpLight;
            const double al = std::sqrt(ea.alpha_rep * eb.alpha_rep);
            const double zf = ea.zeff * eb.zeff;
            const double erep =
                zf * std::exp(-al * std::pow(r, kexp)) / std::pow(r, kGfn2Rexp);
            const double dErep_dr =
                erep * (-al * kexp * std::pow(r, kexp - 1.0) - kGfn2Rexp / r);
            for (int k = 0; k < 3; ++k)
                dG[static_cast<std::size_t>(k)] += dErep_dr * vec[k] * invr;

            dEdcn[static_cast<std::size_t>(a)] += dcni;
            dEdcn[static_cast<std::size_t>(b)] += dcnj;
            for (int k = 0; k < 3; ++k) {
                g[static_cast<std::size_t>(a)][static_cast<std::size_t>(k)] +=
                    dG[static_cast<std::size_t>(k)];
                g[static_cast<std::size_t>(b)][static_cast<std::size_t>(k)] -=
                    dG[static_cast<std::size_t>(k)];
            }
        }
    }

    for (int moved = 0; moved < n_at; ++moved) {
        for (int a = 0; a < n_at; ++a) {
            const auto& dcn =
                dcndr[static_cast<std::size_t>(moved * n_at + a)];
            const double dcnE = dEdcn[static_cast<std::size_t>(a)];
            for (int k = 0; k < 3; ++k)
                g[static_cast<std::size_t>(moved)][static_cast<std::size_t>(k)] +=
                    dcn[static_cast<std::size_t>(k)] * dcnE;
        }
    }

    // AES kernel at frozen CAMM (tblite get_multipole_gradient_0d).
    std::vector<double> mrad(static_cast<std::size_t>(n_at));
    std::vector<double> dmrdcn(static_cast<std::size_t>(n_at));
    std::vector<double> dEdrad(static_cast<std::size_t>(n_at), 0.0);
    for (int a = 0; a < n_at; ++a) {
        const auto& ea = *el[static_cast<std::size_t>(a)];
        const double arg =
            cn[static_cast<std::size_t>(a)] - ea.mpvcn - kGfn2AesShift;
        const double t1 = std::exp(-kGfn2AesKexp * arg);
        const double t2 = (kGfn2AesRmax - ea.mprad) / (1.0 + t1);
        mrad[static_cast<std::size_t>(a)] = ea.mprad + t2;
        dmrdcn[static_cast<std::size_t>(a)] =
            -t2 * kGfn2AesKexp * t1 / (1.0 + t1);
    }
    for (int iat = 0; iat < n_at; ++iat) {
        const double* ri = xyz[static_cast<std::size_t>(iat)].data();
        const auto& mui = dipat[static_cast<std::size_t>(iat)];
        const auto& thi = quadat[static_cast<std::size_t>(iat)];
        const double qi = -qat[static_cast<std::size_t>(iat)]; // Mulliken
        for (int jat = 0; jat < iat; ++jat) {
            const double* rj = xyz[static_cast<std::size_t>(jat)].data();
            const auto& muj = dipat[static_cast<std::size_t>(jat)];
            const auto& thj = quadat[static_cast<std::size_t>(jat)];
            const double qj = -qat[static_cast<std::size_t>(jat)];
            double vec[3] = {rj[0] - ri[0], rj[1] - ri[1], rj[2] - ri[2]};
            const double r1 = std::sqrt(vec[0] * vec[0] + vec[1] * vec[1]
                                        + vec[2] * vec[2]);
            if (r1 < 1e-12)
                continue;
            const double r2 = r1 * r1;
            const double g1 = 1.0 / r1;
            const double g3 = g1 * g1 * g1;
            const double g5 = g3 * g1 * g1;
            const double g7 = g5 * g1 * g1;
            const double rr =
                0.5
                * (mrad[static_cast<std::size_t>(iat)]
                   + mrad[static_cast<std::size_t>(jat)]);
            const double fdmp3 =
                1.0 / (1.0 + 6.0 * std::pow(rr * g1, kGfn2AesDmp3));
            const double ddmp3 =
                -3.0 * g5 * fdmp3
                - kGfn2AesDmp3 * fdmp3 * (fdmp3 - 1.0) * g5;
            const double fdmp5 =
                1.0 / (1.0 + 6.0 * std::pow(rr * g1, kGfn2AesDmp5));
            const double ddmp5 =
                -5.0 * fdmp5 - kGfn2AesDmp5 * (fdmp5 * fdmp5 - fdmp5);

            const double dpiqj =
                (mui[0] * vec[0] + mui[1] * vec[1] + mui[2] * vec[2]) * qj;
            const double qidpj =
                (muj[0] * vec[0] + muj[1] * vec[1] + muj[2] * vec[2]) * qi;
            const double fddr_cd =
                3.0 * (dpiqj - qidpj) * kGfn2AesDmp3 * fdmp3 * g3
                * (fdmp3 / rr) * std::pow(rr * g1, kGfn2AesDmp3);
            Grad3 dG{0, 0, 0};
            for (int k = 0; k < 3; ++k)
                dG[static_cast<std::size_t>(k)] =
                    -ddmp3 * vec[k] * (dpiqj - qidpj)
                    + fdmp3 * g3 * (qi * muj[k] - qj * mui[k]);

            const double dpidpj =
                mui[0] * muj[0] + mui[1] * muj[1] + mui[2] * muj[2];
            const double dpiv =
                mui[0] * vec[0] + mui[1] * vec[1] + mui[2] * vec[2];
            const double dpjv =
                muj[0] * vec[0] + muj[1] * vec[1] + muj[2] * vec[2];
            const double edd = dpidpj * r2 - 3.0 * dpjv * dpiv;
            const double fddr_dd =
                3.0 * edd * kGfn2AesDmp5 * fdmp5 * g5 * (fdmp5 / rr)
                * std::pow(rr * g1, kGfn2AesDmp5);
            for (int k = 0; k < 3; ++k)
                dG[static_cast<std::size_t>(k)] +=
                    -2.0 * fdmp5 * g5 * dpidpj * vec[k]
                    + 3.0 * fdmp5 * g5 * (dpiv * muj[k] + dpjv * mui[k])
                    - edd * ddmp5 * g7 * vec[k];

            // Charge–quadrupole: ATV energy uses −Θ:rr g5 fdmp5 / 3.
            const double eq =
                2.0
                    * (qj * thi[1] + thj[1] * qi) * vec[0] * vec[1]
                + 2.0
                    * (qj * thi[3] + thj[3] * qi) * vec[0] * vec[2]
                + 2.0
                    * (qj * thi[4] + thj[4] * qi) * vec[1] * vec[2]
                + (qj * thi[0] + thj[0] * qi) * vec[0] * vec[0]
                + (qj * thi[2] + thj[2] * qi) * vec[1] * vec[1]
                + (qj * thi[5] + thj[5] * qi) * vec[2] * vec[2];
            const double eq_atv = -eq / 3.0;
            const double fddr_cq =
                eq_atv * 3.0 * kGfn2AesDmp5 * fdmp5 * g5 * fdmp5 / rr
                * std::pow(rr * g1, kGfn2AesDmp5);
            auto qvec = [](const std::array<double, 6>& th, const double v[3],
                           double out[3]) {
                out[0] = v[0] * th[0] + v[1] * th[1] + v[2] * th[3];
                out[1] = v[0] * th[1] + v[1] * th[2] + v[2] * th[4];
                out[2] = v[0] * th[3] + v[1] * th[4] + v[2] * th[5];
            };
            double tvi[3], tvj[3];
            qvec(thi, vec, tvi);
            qvec(thj, vec, tvj);
            for (int k = 0; k < 3; ++k)
                dG[static_cast<std::size_t>(k)] +=
                    (-eq_atv * ddmp5 * g7 * vec[k]
                     - 2.0 * fdmp5 * g5 * qi * tvj[k] / 3.0
                     - 2.0 * fdmp5 * g5 * qj * tvi[k] / 3.0);

            const double fddr = fddr_cd + fddr_dd + fddr_cq;
            dEdrad[static_cast<std::size_t>(iat)] += fddr;
            dEdrad[static_cast<std::size_t>(jat)] += fddr;
            for (int k = 0; k < 3; ++k) {
                g[static_cast<std::size_t>(iat)][static_cast<std::size_t>(k)] +=
                    dG[static_cast<std::size_t>(k)];
                g[static_cast<std::size_t>(jat)][static_cast<std::size_t>(k)] -=
                    dG[static_cast<std::size_t>(k)];
            }
        }
    }
    std::vector<double> dEdcn_aes(static_cast<std::size_t>(n_at), 0.0);
    for (int a = 0; a < n_at; ++a)
        dEdcn_aes[static_cast<std::size_t>(a)] =
            dEdrad[static_cast<std::size_t>(a)]
            * dmrdcn[static_cast<std::size_t>(a)];
    for (int moved = 0; moved < n_at; ++moved) {
        for (int a = 0; a < n_at; ++a) {
            const auto& dcn =
                dcndr[static_cast<std::size_t>(moved * n_at + a)];
            const double dcnE = dEdcn_aes[static_cast<std::size_t>(a)];
            for (int k = 0; k < 3; ++k)
                g[static_cast<std::size_t>(moved)][static_cast<std::size_t>(k)] +=
                    dcn[static_cast<std::size_t>(k)] * dcnE;
        }
    }

    for (int a = 0; a < n_at; ++a) {
        g[static_cast<std::size_t>(a)][0] -=
            d4.forces[static_cast<std::size_t>(a)][0];
        g[static_cast<std::size_t>(a)][1] -=
            d4.forces[static_cast<std::size_t>(a)][1];
        g[static_cast<std::size_t>(a)][2] -=
            d4.forces[static_cast<std::size_t>(a)][2];
    }

    out.grad = std::move(g);
    return out;
}

} // namespace

void GFN2::build(const BuildContext& /*ctx*/) {}

void GFN2::launch(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    if (f.nlocal() < 1)
        throw std::invalid_argument("GFN2: empty frame");
    if (!(mix_beta > 0.0) || mix_beta > 1.0)
        throw std::invalid_argument("GFN2: mix_beta must be in (0, 1]");

    const bool do_f = with_forces && ctx.forces_requested;
    const auto out =
        solve_gfn2(f, charge, scc_tol, max_scc, mix_beta, do_f);
    energy_ = out.energy_eh * units::kHartreeInEv;
    scc_residual_ = out.residual;
    scc_iterations_ = out.iterations;
    last_dq_ = out.dq;
    f.meta.get(kEnergy) += energy_;
    f.meta.get(kEnergyElectronic) += energy_;
    StorageReal* qcol = f.span(kAtom_charge);
    const int n = f.nlocal();
    for (int a = 0; a < n; ++a)
        qcol[a] =
            static_cast<StorageReal>(last_dq_[static_cast<std::size_t>(a)]);
    if (do_f) {
        AccReal* fx = f.span(kAtom_fx);
        AccReal* fy = f.span(kAtom_fy);
        AccReal* fz = f.span(kAtom_fz);
        const double fac = units::kHartreePerBohrInEvPerAngstrom;
        for (int a = 0; a < n; ++a) {
            fx[a] += static_cast<AccReal>(
                -out.grad[static_cast<std::size_t>(a)][0] * fac);
            fy[a] += static_cast<AccReal>(
                -out.grad[static_cast<std::size_t>(a)][1] * fac);
            fz[a] += static_cast<AccReal>(
                -out.grad[static_cast<std::size_t>(a)][2] * fac);
        }
    }
}

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
