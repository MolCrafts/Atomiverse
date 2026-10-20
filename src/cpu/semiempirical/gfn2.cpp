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
using basis_data::kGfn2CnKa;
using basis_data::kGfn2CnKb;
using basis_data::kGfn2CnShift;
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

// Radius of the H⁰ distance polynomial Π(R) [Bohr].
double rpoly_of(const Gfn2ElementRow& e) { return e.rpoly * kBohr; }

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
    // Traceless 3/2-form of ⟨(r−R)⊗(r−R)⟩ from origin-0 integrals.
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

// GFN2 coordination number CN' (Bannwarth et al. 2019): double-exponential
// counting f(R) = [1 + e^{-k_a(R0/R - 1)}]^{-1} [1 + e^{-k_b((R0+R_s)/R - 1)}]^{-1}
// with R0 = 4/3 (R_A^cov + R_B^cov) [Bohr], k_a = 10, k_b = 20, R_s = 2 Bohr.
double cn_count(double r, double r0) {
    if (r < 1e-12)
        return 0;
    const double rco = (4.0 / 3.0) * r0;
    const double fa = 1.0 / (1.0 + std::exp(-kGfn2CnKa * (rco / r - 1.0)));
    const double fb =
        1.0 / (1.0 + std::exp(-kGfn2CnKb * ((rco + kGfn2CnShift) / r - 1.0)));
    return fa * fb;
}

double dcn_count_dr(double r, double r0) {
    if (r < 1e-12)
        return 0;
    const double rco = (4.0 / 3.0) * r0;
    const double ea = std::exp(-kGfn2CnKa * (rco / r - 1.0));
    const double eb = std::exp(-kGfn2CnKb * ((rco + kGfn2CnShift) / r - 1.0));
    const double fa = 1.0 / (1.0 + ea);
    const double fb = 1.0 / (1.0 + eb);
    // df/dr = -e k R0' / r^2 / (1+e)^2 for each factor.
    const double dfa = -fa * fa * ea * kGfn2CnKa * rco / (r * r);
    const double dfb =
        -fb * fb * eb * kGfn2CnKb * (rco + kGfn2CnShift) / (r * r);
    return dfa * fb + fa * dfb;
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
    auto dP = [a, gamma](int k, int m) { return (k == m) ? (a / gamma) : 0.0; };

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
    const double dAi0 = (la > 0) ? (P[ci] - A[ci]) : 1.0;
    const double dAj0 = (lb > 0) ? (P[cj] - B[cj]) : 1.0;
    for (int k = 0; k < 3; ++k) {
        double t = P[k] * dAi0 * dAj0;
        if (la > 0 && lb > 0 && ci == cj)
            t += P[k] * inv2g;
        if (la > 0 && ci == k)
            t += inv2g * dAj0;
        if (lb > 0 && cj == k)
            t += inv2g * dAi0;
        D[k] = S00 * t;
        for (int m = 0; m < 3; ++m) {
            const double ddAi =
                (la > 0) ? (dP(ci, m) - ((m == ci) ? 1.0 : 0.0)) : 0.0;
            const double ddAj = (lb > 0) ? dP(cj, m) : 0.0;
            double dt = dP(k, m) * dAi0 * dAj0 + P[k] * ddAi * dAj0
                        + P[k] * dAi0 * ddAj;
            if (la > 0 && lb > 0 && ci == cj)
                dt += dP(k, m) * inv2g;
            if (la > 0 && ci == k)
                dt += inv2g * ddAj;
            if (lb > 0 && cj == k)
                dt += inv2g * ddAi;
            dD_dA[k][m] = dS00[m] * t + S00 * dt;
        }
    }
}

// Cartesian quadrupole <r_a r_b> at origin 0 and d/dA (s/p primitives).
void cartesian_rr_grad(int ia, int ib, const double P[3], const double dS00[3],
                       double S00, double dAi0, double dAj0, double ddAi0[3],
                       double ddAj0[3], int li, int lj, int ci, int cj,
                       double inv2g, double a_over_g, double& v,
                       double dv_dA[3]) {
    auto dP = [a_over_g](int k, int m) { return (k == m) ? a_over_g : 0.0; };
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
        double dv =
            dP(ia, m) * P[ib] * dAi0 * dAj0 + P[ia] * dP(ib, m) * dAi0 * dAj0
            + P[ia] * P[ib] * ddAi0[m] * dAj0 + P[ia] * P[ib] * dAi0 * ddAj0[m];
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
    const double dAi0 = (la > 0) ? (P[ci] - A[ci]) : 1.0;
    const double dAj0 = (lb > 0) ? (P[cj] - B[cj]) : 1.0;
    double ddAi0[3] = {0, 0, 0};
    double ddAj0[3] = {0, 0, 0};
    for (int m = 0; m < 3; ++m) {
        if (la > 0)
            ddAi0[m] = ((m == ci) ? a_over_g : 0.0) - ((m == ci) ? 1.0 : 0.0);
        if (lb > 0)
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
                    - dR[1][m] * D[0] - R[1] * dD[0][m] + dR[0][m] * R[1] * s
                    + R[0] * dR[1][m] * s + R[0] * R[1] * ds[m];
        dqs[2][m] = dQ[2][m] - 2.0 * (dR[1][m] * D[1] + R[1] * dD[1][m])
                    + 2.0 * R[1] * dR[1][m] * s + R[1] * R[1] * ds[m];
        dqs[3][m] = dQ[3][m] - dR[0][m] * D[2] - R[0] * dD[2][m]
                    - dR[2][m] * D[0] - R[2] * dD[0][m] + dR[0][m] * R[2] * s
                    + R[0] * dR[2][m] * s + R[0] * R[2] * ds[m];
        dqs[4][m] = dQ[4][m] - dR[1][m] * D[2] - R[1] * dD[2][m]
                    - dR[2][m] * D[1] - R[2] * dD[1][m] + dR[1][m] * R[2] * s
                    + R[1] * dR[2][m] * s + R[1] * R[2] * ds[m];
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
            cn[static_cast<std::size_t>(a)] += cn_count(r, r0);
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
            // Π(R) = Π_l (1 + k_poly,l (R / (R_A + R_B))^{1/2}).
            const double rr =
                (r > 1e-12)
                    ? std::sqrt(
                          r
                          / (rpoly_of(*el[static_cast<std::size_t>(a)])
                             + rpoly_of(*el[static_cast<std::size_t>(b)])))
                    : 0.0;
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

    // -----------------------------------------------------------------
    // Anisotropic electrostatics (AES) and anisotropic XC (AXC) of
    // GFN2-xTB (Bannwarth, Ehlert, Grimme, JCTC 15, 1652 (2019)).
    //
    // Cumulative atomic multipoles of the density P (Mulliken split; the
    // operator is centred on the bra atom A of κ):
    //   q_A = −Σ_{l∈A} Δn_l,   Δn_l = Σ_{κ∈l} (PS)_κκ − n⁰_l
    //   μ_A = −Σ_{κ∈A} Σ_ν P_κν ⟨κ|r − R_A|ν⟩
    //   Θ_A = −Σ_{κ∈A} Σ_ν P_κν ⟨κ|3/2 (r−R_A)⊗(r−R_A) − ½|r−R_A|²|ν⟩
    // Pair energy, R = R_A − R_B, g_n = f_n(R)/R^n:
    //   E_AB = g3 (q_A μ_B − q_B μ_A)·R
    //        + g5 [q_A R·Θ_B·R + q_B R·Θ_A·R + (μ_A·μ_B) R² − 3(μ_A·R)(μ_B·R)]
    //   f_n = 1 / (1 + 6 (R0_AB/R)^{a_n}),  R0_AB = ½ (R0_A + R0_B),
    //   R0_A = R_A^mp + (R_max − R_A^mp) / (1 + exp(−k (CN_A − N_A^v − Δ)))
    // AXC: Σ_A f^μ_A |μ_A|² + f^Θ_A Θ_A:Θ_A.
    // Θ is stored packed (xx, xy, yy, xz, yz, zz); derivatives with
    // respect to the packed components carry the factor 2 of the
    // symmetric off-diagonal pair.
    // -----------------------------------------------------------------
    const std::size_t nat = static_cast<std::size_t>(n_at);
    auto U = [](int i) { return static_cast<std::size_t>(i); };

    std::vector<double> r0(nat), dr0dcn(nat);
    for (int a = 0; a < n_at; ++a) {
        const auto& ea = *el[U(a)];
        const double ex =
            std::exp(-kGfn2AesKexp * (cn[U(a)] - ea.mpvcn - kGfn2AesShift));
        r0[U(a)] = ea.mprad + (kGfn2AesRmax - ea.mprad) / (1.0 + ex);
        dr0dcn[U(a)] = (kGfn2AesRmax - ea.mprad) * kGfn2AesKexp * ex
                       / ((1.0 + ex) * (1.0 + ex));
    }

    struct Moments {
        std::vector<double> q;                 // physical charge [e]
        std::vector<std::array<double, 3>> mu; // [e·Bohr]
        std::vector<std::array<double, 6>> th; // packed [e·Bohr²]
    };
    auto rthr = [](const std::array<double, 6>& t, const double v[3]) {
        return t[0] * v[0] * v[0] + 2.0 * t[1] * v[0] * v[1]
               + t[2] * v[1] * v[1] + 2.0 * t[3] * v[0] * v[2]
               + 2.0 * t[4] * v[1] * v[2] + t[5] * v[2] * v[2];
    };
    auto thv = [](const std::array<double, 6>& t, const double v[3],
                  double o[3]) {
        o[0] = t[0] * v[0] + t[1] * v[1] + t[3] * v[2];
        o[1] = t[1] * v[0] + t[2] * v[1] + t[4] * v[2];
        o[2] = t[3] * v[0] + t[4] * v[1] + t[5] * v[2];
    };
    auto dot3 = [](const double* x, const double* y) {
        return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
    };
    // f_n(R) and its partial derivatives in R and R0.
    auto damp = [](double r, double rr0, double an, double& fn, double& dfdr,
                   double& dfdr0) {
        const double xa = std::pow(rr0 / r, an);
        fn = 1.0 / (1.0 + 6.0 * xa);
        dfdr = 6.0 * an * fn * fn * xa / r;
        dfdr0 = -6.0 * an * fn * fn * xa / rr0;
    };

    // E_AES + E_AXC and its partial derivatives in the multipoles.
    auto aes_eval = [&](const Moments& m, std::vector<double>& vq,
                        std::vector<std::array<double, 3>>& vmu,
                        std::vector<std::array<double, 6>>& vth) {
        vq.assign(nat, 0.0);
        vmu.assign(nat, std::array<double, 3>{0, 0, 0});
        vth.assign(nat, std::array<double, 6>{0, 0, 0, 0, 0, 0});
        double e = 0;
        for (int a = 0; a < n_at; ++a) {
            const auto& ea = *el[U(a)];
            const auto& mu = m.mu[U(a)];
            const auto& t = m.th[U(a)];
            e += ea.dkernel * dot3(mu.data(), mu.data());
            e += ea.qkernel
                 * (t[0] * t[0] + 2.0 * t[1] * t[1] + t[2] * t[2]
                    + 2.0 * t[3] * t[3] + 2.0 * t[4] * t[4] + t[5] * t[5]);
            for (int k = 0; k < 3; ++k)
                vmu[U(a)][U(k)] += 2.0 * ea.dkernel * mu[U(k)];
            const double tq[6] = {2.0, 4.0, 2.0, 4.0, 4.0, 2.0};
            for (int k = 0; k < 6; ++k)
                vth[U(a)][U(k)] += tq[k] * ea.qkernel * t[U(k)];
        }
        for (int a = 0; a < n_at; ++a) {
            for (int b = 0; b < a; ++b) {
                const double R[3] = {xyz[U(a)][0] - xyz[U(b)][0],
                                     xyz[U(a)][1] - xyz[U(b)][1],
                                     xyz[U(a)][2] - xyz[U(b)][2]};
                const double r2 = dot3(R, R);
                const double r = std::sqrt(r2);
                const double rr0 = 0.5 * (r0[U(a)] + r0[U(b)]);
                double f3, f5, d1, d2;
                damp(r, rr0, kGfn2AesDmp3, f3, d1, d2);
                damp(r, rr0, kGfn2AesDmp5, f5, d1, d2);
                const double g3 = f3 / (r2 * r);
                const double g5 = f5 / (r2 * r2 * r);
                const auto& ma = m.mu[U(a)];
                const auto& mb = m.mu[U(b)];
                const double qa = m.q[U(a)];
                const double qb = m.q[U(b)];
                const double mar = dot3(ma.data(), R);
                const double mbr = dot3(mb.data(), R);
                const double ta = rthr(m.th[U(a)], R);
                const double tb = rthr(m.th[U(b)], R);
                const double mab = dot3(ma.data(), mb.data());
                e +=
                    g3 * (qa * mbr - qb * mar)
                    + g5 * ((qa * tb + qb * ta) + (mab * r2 - 3.0 * mar * mbr));
                vq[U(a)] += g3 * mbr + g5 * tb;
                vq[U(b)] += -g3 * mar + g5 * ta;
                for (int k = 0; k < 3; ++k) {
                    vmu[U(a)][U(k)] +=
                        -g3 * qb * R[k]
                        + g5 * (mb[U(k)] * r2 - 3.0 * R[k] * mbr);
                    vmu[U(b)][U(k)] +=
                        g3 * qa * R[k]
                        + g5 * (ma[U(k)] * r2 - 3.0 * R[k] * mar);
                }
                const double rr[6] = {R[0] * R[0],       2.0 * R[0] * R[1],
                                      R[1] * R[1],       2.0 * R[0] * R[2],
                                      2.0 * R[1] * R[2], R[2] * R[2]};
                for (int k = 0; k < 6; ++k) {
                    vth[U(a)][U(k)] += g5 * qb * rr[k];
                    vth[U(b)][U(k)] += g5 * qa * rr[k];
                }
            }
        }
        return e;
    };

    // Integrals with the operator origin on the bra atom:
    //   dA(k,κ,ν) = ⟨κ|r_k − R_A|ν⟩,  tA(k,κ,ν) = traceless Θ about R_A.
    std::vector<double> dA(static_cast<std::size_t>(3 * n_ao * n_ao), 0.0);
    std::vector<double> tA(static_cast<std::size_t>(6 * n_ao * n_ao), 0.0);
    auto DA = [&](int k, int mu, int nu) -> double& {
        return dA[static_cast<std::size_t>((k * n_ao + mu) * n_ao + nu)];
    };
    auto TA = [&](int k, int mu, int nu) -> double& {
        return tA[static_cast<std::size_t>((k * n_ao + mu) * n_ao + nu)];
    };
    for (int mu = 0; mu < n_ao; ++mu) {
        const double* Ra = xyz[U(ao_atom[U(mu)])].data();
        for (int nu = 0; nu < n_ao; ++nu) {
            const double s = S(mu, nu);
            const double d0[3] = {D(0, mu, nu), D(1, mu, nu), D(2, mu, nu)};
            const double q0[6] = {Q(0, mu, nu), Q(1, mu, nu), Q(2, mu, nu),
                                  Q(3, mu, nu), Q(4, mu, nu), Q(5, mu, nu)};
            for (int k = 0; k < 3; ++k)
                DA(k, mu, nu) = d0[k] - Ra[k] * s;
            double tq[6];
            origin_shift_quad(q0, d0, Ra, s, tq);
            for (int k = 0; k < 6; ++k)
                TA(k, mu, nu) = tq[k];
        }
    }

    // Shell populations Δn and multipoles of a density.
    auto moments_of = [&](const linalg::RealMatrix& Pm, std::vector<double>& dn,
                          Moments& m) {
        dn.assign(U(n_sh), 0.0);
        m.q.assign(nat, 0.0);
        m.mu.assign(nat, std::array<double, 3>{0, 0, 0});
        m.th.assign(nat, std::array<double, 6>{0, 0, 0, 0, 0, 0});
        for (int mu = 0; mu < n_ao; ++mu) {
            const int a = ao_atom[U(mu)];
            double pop = 0;
            for (int nu = 0; nu < n_ao; ++nu) {
                const double p = Pm(mu, nu);
                pop += p * S(mu, nu);
                for (int k = 0; k < 3; ++k)
                    m.mu[U(a)][U(k)] -= p * DA(k, mu, nu);
                for (int k = 0; k < 6; ++k)
                    m.th[U(a)][U(k)] -= p * TA(k, mu, nu);
            }
            dn[U(ao_shell[U(mu)])] += pop;
        }
        for (int ish = 0; ish < n_sh; ++ish) {
            dn[U(ish)] -= n0sh[U(ish)];
            m.q[U(shells[U(ish)].atom)] -= dn[U(ish)];
        }
    };

    // Self-consistent D4 (GFN2): C6 scaled by the SCC charges q_A.
    std::vector<int> Z(nat);
    for (int a = 0; a < n_at; ++a)
        Z[U(a)] = type[a];
    auto d4_of = [&](const Moments& m, std::vector<double>& dEdq) {
        return scf::dftd4_fixed_charges(Z, xyz, m.q, kGfn2D4S6, kGfn2D4S8,
                                        kGfn2D4A1, kGfn2D4A2, kGfn2D4S9, dEdq);
    };

    // Shell potential ∂E/∂Δn_l of the isotropic terms.
    auto shell_potential = [&](const std::vector<double>& dn,
                               const std::vector<double>& vq,
                               std::vector<double>& v) {
        v.assign(U(n_sh), 0.0);
        for (int ish = 0; ish < n_sh; ++ish) {
            double s = 0;
            for (int jsh = 0; jsh < n_sh; ++jsh)
                s += gamma[U(ish) * U(n_sh) + U(jsh)] * dn[U(jsh)];
            // E3 = ⅓ Σ Γ_l q_l³ in the shell charge q_l = −Δn_l.
            s -= gam3[U(ish)] * dn[U(ish)] * dn[U(ish)];
            // q_A = −Σ Δn_l ⇒ ∂E/∂Δn_l = −∂E/∂q_A (AES + D4).
            s -= vq[U(shells[U(ish)].atom)];
            v[U(ish)] = s;
        }
    };

    const std::size_t nmix = U(n_sh) + 9 * nat;
    auto pack = [&](const std::vector<double>& dn, const Moments& m) {
        std::vector<double> x(nmix);
        for (int i = 0; i < n_sh; ++i)
            x[U(i)] = dn[U(i)];
        for (int a = 0; a < n_at; ++a) {
            for (int k = 0; k < 3; ++k)
                x[U(n_sh) + 3 * U(a) + U(k)] = m.mu[U(a)][U(k)];
            for (int k = 0; k < 6; ++k)
                x[U(n_sh) + 3 * nat + 6 * U(a) + U(k)] = m.th[U(a)][U(k)];
        }
        return x;
    };
    auto unpack = [&](const std::vector<double>& x, std::vector<double>& dn,
                      Moments& m) {
        dn.assign(x.begin(), x.begin() + n_sh);
        m.q.assign(nat, 0.0);
        m.mu.assign(nat, std::array<double, 3>{0, 0, 0});
        m.th.assign(nat, std::array<double, 6>{0, 0, 0, 0, 0, 0});
        for (int ish = 0; ish < n_sh; ++ish)
            m.q[U(shells[U(ish)].atom)] -= dn[U(ish)];
        for (int a = 0; a < n_at; ++a) {
            for (int k = 0; k < 3; ++k)
                m.mu[U(a)][U(k)] = x[U(n_sh) + 3 * U(a) + U(k)];
            for (int k = 0; k < 6; ++k)
                m.th[U(a)][U(k)] = x[U(n_sh) + 3 * nat + 6 * U(a) + U(k)];
        }
    };

    // Fock matrix F = ∂E/∂P (symmetrised) for given potentials.
    auto build_fock = [&](const std::vector<double>& vsh,
                          const std::vector<std::array<double, 3>>& vmu,
                          const std::vector<std::array<double, 6>>& vth) {
        linalg::RealMatrix H = H0;
        for (int mu = 0; mu < n_ao; ++mu) {
            const int a = ao_atom[U(mu)];
            const double va = vsh[U(ao_shell[U(mu)])];
            for (int nu = 0; nu < n_ao; ++nu) {
                const int b = ao_atom[U(nu)];
                double h = 0.5 * S(mu, nu) * (va + vsh[U(ao_shell[U(nu)])]);
                double m = 0;
                for (int k = 0; k < 3; ++k)
                    m += vmu[U(a)][U(k)] * DA(k, mu, nu)
                         + vmu[U(b)][U(k)] * DA(k, nu, mu);
                for (int k = 0; k < 6; ++k)
                    m += vth[U(a)][U(k)] * TA(k, mu, nu)
                         + vth[U(b)][U(k)] * TA(k, nu, mu);
                H(mu, nu) += h - 0.5 * m;
            }
        }
        return H;
    };

    internal::ChargeMixer mixer(mix_beta, 6);
    linalg::RealMatrix P = xt::zeros<double>({n_ao, n_ao});
    linalg::RealMatrix Cmat = xt::zeros<double>({n_ao, n_ao});
    linalg::RealVector eps_mo = xt::zeros<double>({n_ao});
    std::vector<double> dn_in(U(n_sh), 0.0), dn(U(n_sh), 0.0), vsh;
    Moments mom_in, mom;
    std::vector<double> vq;
    std::vector<std::array<double, 3>> vmu;
    std::vector<std::array<double, 6>> vth;
    unpack(std::vector<double>(nmix, 0.0), dn_in, mom_in);
    std::vector<double> x_in(nmix, 0.0);
    SolveOut out;
    bool converged = false;
    for (int it = 0; it < max_scc; ++it) {
        unpack(x_in, dn_in, mom_in);
        aes_eval(mom_in, vq, vmu, vth);
        std::vector<double> dEdq_d4;
        d4_of(mom_in, dEdq_d4);
        for (int a = 0; a < n_at; ++a)
            vq[U(a)] += dEdq_d4[U(a)];
        shell_potential(dn_in, vq, vsh);
        const linalg::RealMatrix H = build_fock(vsh, vmu, vth);
        auto sol = linalg::diagonalize_fock_generalized(H, S);
        eps_mo = sol.first;
        Cmat = sol.second;
        P = xt::zeros<double>({n_ao, n_ao});
        for (int o = 0; o < n_occ; ++o)
            for (int mu = 0; mu < n_ao; ++mu)
                for (int nu = 0; nu < n_ao; ++nu)
                    P(mu, nu) += 2.0 * Cmat(mu, o) * Cmat(nu, o);
        moments_of(P, dn, mom);
        const std::vector<double> x_out = pack(dn, mom);
        double residual = 0;
        for (std::size_t i = 0; i < nmix; ++i)
            residual = std::max(residual, std::abs(x_out[i] - x_in[i]));
        out.residual = residual;
        out.iterations = it + 1;
        if (residual < scc_tol) {
            converged = true;
            break;
        }
        x_in = mixer.next(x_in, x_out);
    }
    if (!converged)
        throw std::runtime_error("GFN2: SCC did not converge in max_scc="
                                 + std::to_string(max_scc)
                                 + " (residual=" + std::to_string(out.residual)
                                 + "); increase max_scc or lower mix_beta");

    // Energy and potentials of the converged density.
    const double e_aes = aes_eval(mom, vq, vmu, vth);
    std::vector<double> dEdq_d4;
    const scf::D4Result d4 = d4_of(mom, dEdq_d4);
    for (int a = 0; a < n_at; ++a)
        vq[U(a)] += dEdq_d4[U(a)];
    shell_potential(dn, vq, vsh);
    double e_h0 = 0;
    for (int mu = 0; mu < n_ao; ++mu)
        for (int nu = 0; nu < n_ao; ++nu)
            e_h0 += P(mu, nu) * H0(mu, nu);
    double e_ies = 0;
    double e3 = 0;
    for (int ish = 0; ish < n_sh; ++ish) {
        for (int jsh = 0; jsh < n_sh; ++jsh)
            e_ies += 0.5 * dn[U(ish)] * gamma[U(ish) * U(n_sh) + U(jsh)]
                     * dn[U(jsh)];
        e3 -= dn[U(ish)] * dn[U(ish)] * dn[U(ish)] * gam3[U(ish)] / 3.0;
    }
    const double e_rep = repulsion();

    const double e_d4 = d4.energy;

    out.energy_eh = e_h0 + e_ies + e3 + e_rep + e_d4 + e_aes;
    out.dq = mom.q;

    if (!want_grad)
        return out;

    // ---------------------------------------------------------------
    // Nuclear gradient. The SCC energy is stationary in P (F = ∂E/∂P),
    // so dE/dR = ∂E/∂R|_P − Tr(W ∂S/∂R), W = Σ_occ 2 ε_i c_i c_iᵀ.
    // ---------------------------------------------------------------
    linalg::RealMatrix W = xt::zeros<double>({n_ao, n_ao});
    for (int o = 0; o < n_occ; ++o) {
        const double wfac = 2.0 * eps_mo(o);
        for (int mu = 0; mu < n_ao; ++mu)
            for (int nu = 0; nu < n_ao; ++nu)
                W(mu, nu) += wfac * Cmat(mu, o) * Cmat(nu, o);
    }

    std::vector<Grad3> g(nat, Grad3{0, 0, 0});
    std::vector<double> dEdcn(nat, 0.0);
    std::vector<Grad3> dcndr(nat * nat, Grad3{0, 0, 0});
    for (int a = 0; a < n_at; ++a) {
        const double rca = el[U(a)]->rcov * kBohr;
        for (int b = 0; b < a; ++b) {
            const double rcb = el[U(b)]->rcov * kBohr;
            double dx = static_cast<double>(xA[a] - xA[b]);
            double dy = static_cast<double>(yA[a] - yA[b]);
            double dz = static_cast<double>(zA[a] - zA[b]);
            f.box.min_image(dx, dy, dz);
            const double dv[3] = {dx * kBohr, dy * kBohr, dz * kBohr};
            const double r = std::sqrt(dot3(dv, dv));
            if (r < 1e-12)
                continue;
            const double dc = dcn_count_dr(r, rca + rcb) / r;
            // ∂CN_c/∂R_moved stored at dcndr[moved * n_at + c].
            for (int k = 0; k < 3; ++k) {
                dcndr[U(a) * nat + U(a)][U(k)] += dv[k] * dc;
                dcndr[U(b) * nat + U(a)][U(k)] -= dv[k] * dc;
                dcndr[U(a) * nat + U(b)][U(k)] += dv[k] * dc;
                dcndr[U(b) * nat + U(b)][U(k)] -= dv[k] * dc;
            }
        }
    }

    // On-site self-energy CN shift: Σ_{κ∈l} P_κκ ∂h_l/∂CN.
    for (int ish = 0; ish < n_sh; ++ish) {
        const Shell& sh = shells[U(ish)];
        const auto& e = *el[U(sh.atom)];
        const int s = ish - sh0[U(sh.atom)];
        double pop = 0;
        for (int u = 0; u < 2 * sh.l + 1; ++u)
            pop += P(sh.ao0 + u, sh.ao0 + u);
        dEdcn[U(sh.atom)] += -e.kcn[s] * kGfn2EvToEh * pop;
    }

    for (int a = 0; a < n_at; ++a) {
        const auto& ea = *el[U(a)];
        const double* Ai = xyz[U(a)].data();
        const int is0 = sh0[U(a)];
        for (int b = 0; b < a; ++b) {
            const auto& eb = *el[U(b)];
            const double* Aj = xyz[U(b)].data();
            const int js0 = sh0[U(b)];
            const double vec[3] = {Ai[0] - Aj[0], Ai[1] - Aj[1], Ai[2] - Aj[2]};
            const double r2 = dot3(vec, vec);
            if (r2 < 1e-16)
                continue;
            const double r = std::sqrt(r2);
            const double rr = std::sqrt(r / (rpoly_of(ea) + rpoly_of(eb)));
            const double den = ea.en - eb.en;
            const double enp = 1.0 + kGfn2Enscale * den * den;

            Grad3 dG{0, 0, 0};
            double dcni = 0;
            double dcnj = 0;
            for (int sa = 0; sa < ea.n_shell; ++sa) {
                const int ish = is0 + sa;
                const Shell& si = shells[U(ish)];
                const double kpi = ea.shpoly[si.l == 0 ? 0 : 1];
                const double shpi = 1.0 + kpi * rr;
                for (int sb = 0; sb < eb.n_shell; ++sb) {
                    const int jsh = js0 + sb;
                    const Shell& sj = shells[U(jsh)];
                    const double kpj = eb.shpoly[sj.l == 0 ? 0 : 1];
                    const double shpj = 1.0 + kpj * rr;
                    const double shpoly = shpi * shpj;
                    const double zij =
                        std::pow(2.0 * std::sqrt(si.zeta * sj.zeta)
                                     / (si.zeta + sj.zeta),
                                 kGfn2Wexp);
                    const double hs = zij * k_shell(si.l, sj.l) * enp * shpoly;
                    const double hij =
                        0.5 * (selfe[U(ish)] + selfe[U(jsh)]) * hs;
                    // ∇_a ln Π = (k_i shpj + k_j shpi) (½ rr / r) / Π · R/r.
                    const double dlnpi =
                        (kpi * shpj + kpj * shpi) * 0.5 * rr / (r * r * shpoly);
                    const double dsed_i = -ea.kcn[sa] * kGfn2EvToEh * hs;
                    const double dsed_j = -eb.kcn[sb] * kGfn2EvToEh * hs;
                    const double vscale = 0.5 * (vsh[U(ish)] + vsh[U(jsh)]);
                    for (int u = 0; u < 2 * si.l + 1; ++u) {
                        const int mu = si.ao0 + u;
                        for (int v = 0; v < 2 * sj.l + 1; ++v) {
                            const int nu = sj.ao0 + v;
                            double sij = 0, ds[3], Dip[3], dDip[3][3], Qint[6],
                                   dQint[6][3];
                            shell_sdq_grad(si, Ai, u, sj, Aj, v, sij, ds, Dip,
                                           dDip, Qint, dQint);
                            const double pij = P(mu, nu);
                            const double sval =
                                2.0 * (pij * (hij + vscale) - W(mu, nu));
                            const double dRa[3][3] = {
                                {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
                            const double dRb[3][3] = {
                                {0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
                            double tq[6], dtqA[6][3], dtqB[6][3];
                            origin_shift_quad_grad(Qint, dQint, Dip, dDip, Ai,
                                                   dRa, sij, ds, tq, dtqA);
                            origin_shift_quad_grad(Qint, dQint, Dip, dDip, Aj,
                                                   dRb, sij, ds, tq, dtqB);
                            for (int k = 0; k < 3; ++k) {
                                double gk =
                                    sval * ds[k]
                                    + 2.0 * pij * hij * sij * dlnpi * vec[k];
                                // μ_a ∋ −P d(a), μ_b ∋ −P d(b); d(X)=D−R_X S.
                                double dm = 0;
                                for (int al = 0; al < 3; ++al) {
                                    const double dDa = dDip[al][k]
                                                       - ((al == k) ? sij : 0.0)
                                                       - Ai[al] * ds[k];
                                    const double dDb =
                                        dDip[al][k] - Aj[al] * ds[k];
                                    dm += vmu[U(a)][U(al)] * dDa
                                          + vmu[U(b)][U(al)] * dDb;
                                }
                                for (int c = 0; c < 6; ++c)
                                    dm += vth[U(a)][U(c)] * dtqA[c][k]
                                          + vth[U(b)][U(c)] * dtqB[c][k];
                                gk -= pij * dm;
                                dG[U(k)] += gk;
                            }
                            dcni += dsed_i * pij * sij;
                            dcnj += dsed_j * pij * sij;
                        }
                    }
                }
            }

            double dEies_dr = 0;
            for (int sa = 0; sa < ea.n_shell; ++sa)
                for (int sb = 0; sb < eb.n_shell; ++sb)
                    dEies_dr += dn[U(is0 + sa)] * dn[U(js0 + sb)]
                                * dgamma_dr(r, hardness[U(is0 + sa)],
                                            hardness[U(js0 + sb)]);
            const double kexp =
                (ea.z > 2 || eb.z > 2) ? kGfn2Kexp : kGfn2KexpLight;
            const double al = std::sqrt(ea.alpha_rep * eb.alpha_rep);
            const double erep = ea.zeff * eb.zeff
                                * std::exp(-al * std::pow(r, kexp))
                                / std::pow(r, kGfn2Rexp);
            const double dErep_dr =
                erep * (-al * kexp * std::pow(r, kexp - 1.0) - kGfn2Rexp / r);
            for (int k = 0; k < 3; ++k)
                dG[U(k)] += (dEies_dr + dErep_dr) * vec[k] / r;

            dEdcn[U(a)] += dcni;
            dEdcn[U(b)] += dcnj;
            for (int k = 0; k < 3; ++k) {
                g[U(a)][U(k)] += dG[U(k)];
                g[U(b)][U(k)] -= dG[U(k)];
            }
        }
    }

    // Explicit AES geometry dependence at frozen multipoles.
    {
        for (int a = 0; a < n_at; ++a) {
            for (int b = 0; b < a; ++b) {
                const double R[3] = {xyz[U(a)][0] - xyz[U(b)][0],
                                     xyz[U(a)][1] - xyz[U(b)][1],
                                     xyz[U(a)][2] - xyz[U(b)][2]};
                const double r2 = dot3(R, R);
                const double r = std::sqrt(r2);
                const double rr0 = 0.5 * (r0[U(a)] + r0[U(b)]);
                double f3, df3, df3r0, f5, df5, df5r0;
                damp(r, rr0, kGfn2AesDmp3, f3, df3, df3r0);
                damp(r, rr0, kGfn2AesDmp5, f5, df5, df5r0);
                const double r3 = r2 * r;
                const double r5 = r3 * r2;
                const double g3 = f3 / r3;
                const double g5 = f5 / r5;
                const double dg3 = (df3 - 3.0 * f3 / r) / r3; // ∂g3/∂R
                const double dg5 = (df5 - 5.0 * f5 / r) / r5;
                const auto& ma = mom.mu[U(a)];
                const auto& mb = mom.mu[U(b)];
                const double qa = mom.q[U(a)];
                const double qb = mom.q[U(b)];
                double w[3];
                for (int k = 0; k < 3; ++k)
                    w[k] = qa * mb[U(k)] - qb * ma[U(k)];
                const double e3p = dot3(w, R);
                double ta[3], tb[3];
                thv(mom.th[U(a)], R, ta);
                thv(mom.th[U(b)], R, tb);
                const double mar = dot3(ma.data(), R);
                const double mbr = dot3(mb.data(), R);
                const double mab = dot3(ma.data(), mb.data());
                const double e5p = (qa * dot3(tb, R) + qb * dot3(ta, R))
                                   + (mab * r2 - 3.0 * mar * mbr);
                for (int k = 0; k < 3; ++k) {
                    const double gk =
                        e3p * dg3 * R[k] / r + g3 * w[k] + e5p * dg5 * R[k] / r
                        + g5
                              * (2.0 * (qa * tb[k] + qb * ta[k])
                                 + (2.0 * mab * R[k] - 3.0 * ma[U(k)] * mbr
                                    - 3.0 * mb[U(k)] * mar));
                    g[U(a)][U(k)] += gk;
                    g[U(b)][U(k)] -= gk;
                }
                const double dEdr0 =
                    0.5 * (e3p * df3r0 / r3 + e5p * df5r0 / r5);
                dEdcn[U(a)] += dEdr0 * dr0dcn[U(a)];
                dEdcn[U(b)] += dEdr0 * dr0dcn[U(b)];
            }
        }
    }

    for (int moved = 0; moved < n_at; ++moved)
        for (int c = 0; c < n_at; ++c)
            for (int k = 0; k < 3; ++k)
                g[U(moved)][U(k)] +=
                    dcndr[U(moved) * nat + U(c)][U(k)] * dEdcn[U(c)];

    for (int a = 0; a < n_at; ++a)
        for (int k = 0; k < 3; ++k)
            g[U(a)][U(k)] -= d4.forces[U(a)][U(k)];

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
    const auto out = solve_gfn2(f, charge, scc_tol, max_scc, mix_beta, do_f);
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
