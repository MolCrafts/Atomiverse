/// @file src/cpu/semiempirical/gfn0.cpp
/// GFN0-xTB energy + analytic nuclear gradient (H/C/N/O): Pracht 2019.
#include "atomiverse/cpu/semiempirical/gfn0.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <xtensor-blas/xlinalg.hpp>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/linalg/eigensolve.h"
#include "atomiverse/cpu/linalg/matrix.h"
#include "atomiverse/cpu/scf/dftd4.h"
#include "atomiverse/cpu/semiempirical/basis_data/gfn0_tables.h"

namespace ATV {
namespace cpu {
namespace semiempirical {
namespace {

using basis_data::Gfn0Element;
using basis_data::kGfn0AutoAA;
using basis_data::kGfn0D4a1;
using basis_data::kGfn0D4a2;
using basis_data::kGfn0D4s8;
using basis_data::kGfn0Elements;
using basis_data::kGfn0EnScale4;
using basis_data::kGfn0EnScaleP;
using basis_data::kGfn0EnScaleS;
using basis_data::kGfn0EvToAu;
using basis_data::kGfn0Kdiff;
using basis_data::kGfn0Kpp;
using basis_data::kGfn0Kss;
using basis_data::kGfn0NElements;
using basis_data::kGfn0RepEnScale;
using basis_data::kGfn0SrbEnScale;
using basis_data::kGfn0SrbPrefactor;
using basis_data::kGfn0SrbShift;
using basis_data::kGfn0SrbSteepness;

constexpr double kPi = 3.14159265358979323846;
constexpr double kSqrtPi = 1.7724538509055160273;
constexpr double kAatoau = 1.0 / kGfn0AutoAA;

using Grad3 = std::array<double, 3>;

const Gfn0Element& element_row(int z) {
    for (std::size_t i = 0; i < kGfn0NElements; ++i)
        if (kGfn0Elements[i].z == z)
            return kGfn0Elements[i];
    throw std::runtime_error(
        "GFN0: unsupported element Z=" + std::to_string(z)
        + " (static tables cover H/C/N/O only; add a transcribed row to "
          "basis_data/gfn0_tables.h)");
}

int ncore(int z) {
    if (z <= 2)
        return 0;
    if (z <= 10)
        return 2;
    throw std::runtime_error("GFN0: ncore only tabulated for Z<=10");
}

double k_scale(int li, int lj) {
    const double ki = (li == 0) ? kGfn0Kss : kGfn0Kpp;
    const double kj = (lj == 0) ? kGfn0Kss : kGfn0Kpp;
    return 0.5 * (ki + kj);
}

double en_scale(int li, int lj) {
    const double ei = (li == 0) ? kGfn0EnScaleS : kGfn0EnScaleP;
    const double ej = (lj == 0) ? kGfn0EnScaleS : kGfn0EnScaleP;
    return 0.005 * (ei + ej);
}

double pauling_en(int z) {
    switch (z) {
    case 1:
        return 2.20;
    case 6:
        return 2.55;
    case 7:
        return 3.04;
    case 8:
        return 3.44;
    default:
        throw std::runtime_error("GFN0: Pauling EN missing for Z="
                                 + std::to_string(z));
    }
}

// Stewart 1970 STO-nG (zeta=1); alpha is then * zeta^2.
void slater_to_gauss(int ng, int n, int l, double zeta, double* alpha,
                     double* coeff) {
    // ityp: 1s=1, 2s=2, 2p=6
    if (ng == 3 && n == 1 && l == 0) {
        alpha[0] = 2.227660584;
        alpha[1] = 4.057711562e-1;
        alpha[2] = 1.098175104e-1;
        coeff[0] = 1.543289673e-1;
        coeff[1] = 5.353281423e-1;
        coeff[2] = 4.446345422e-1;
    } else if (ng == 2 && n == 2 && l == 0) {
        alpha[0] = 1.292278611e-1;
        alpha[1] = 4.908584205e-2;
        coeff[0] = 7.470867124e-1;
        coeff[1] = 2.855980556e-1;
    } else if (ng == 4 && n == 2 && l == 0) {
        alpha[0] = 1.161525551e+1;
        alpha[1] = 2.000243111e+0;
        alpha[2] = 1.607280687e-1;
        alpha[3] = 6.125744532e-2;
        coeff[0] = -1.198411747e-2;
        coeff[1] = -5.472052539e-2;
        coeff[2] = 5.805587176e-1;
        coeff[3] = 4.770079976e-1;
    } else if (ng == 3 && n == 2 && l == 1) {
        alpha[0] = 9.192379002e-1;
        alpha[1] = 2.359194503e-1;
        alpha[2] = 8.009805746e-2;
        coeff[0] = 1.623948553e-1;
        coeff[1] = 5.661708862e-1;
        coeff[2] = 4.223071752e-1;
    } else {
        throw std::runtime_error("GFN0: STO-nG expansion not tabulated");
    }
    const double z2 = zeta * zeta;
    const double df = 1.0; // (2l-1)!! for l=0,1
    for (int i = 0; i < ng; ++i) {
        alpha[i] *= z2;
        const double a = alpha[i];
        coeff[i] *= std::pow(2.0 * a / kPi, 0.75)
                    * std::pow(std::sqrt(4.0 * a), l) / std::sqrt(df);
    }
}

double primitive_overlap(double a, const double A[3], int la, double b,
                         const double B[3], int lb) {
    const double gamma = a + b;
    const double mu = a * b / gamma;
    double R2 = 0;
    for (int k = 0; k < 3; ++k) {
        const double d = A[k] - B[k];
        R2 += d * d;
    }
    const double S00 = std::pow(kPi / gamma, 1.5) * std::exp(-mu * R2);
    double P[3];
    for (int k = 0; k < 3; ++k)
        P[k] = (a * A[k] + b * B[k]) / gamma;
    if (la == 0 && lb == 0)
        return S00;
    if (la == 0) {
        const int ib = lb - 1;
        return (P[ib] - B[ib]) * S00;
    }
    if (lb == 0) {
        const int ia = la - 1;
        return (P[ia] - A[ia]) * S00;
    }
    const int ia = la - 1;
    const int ib = lb - 1;
    double s = (P[ia] - A[ia]) * (P[ib] - B[ib]) * S00;
    if (ia == ib)
        s += S00 / (2.0 * gamma);
    return s;
}

// dS/dA of a Cartesian s/p primitive pair. dS/dB = -dS/dA by translation.
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

double contracted_s_overlap(const double* aa, const double* ca, int na,
                            const double* ab, const double* cb, int nb) {
    double s = 0;
    const double Z[3] = {0, 0, 0};
    for (int i = 0; i < na; ++i)
        for (int j = 0; j < nb; ++j)
            s += ca[i] * cb[j]
                 * primitive_overlap(aa[i], Z, 0, ab[j], Z, 0);
    return s;
}

struct AO {
    int atom = 0;
    int shell = 0;
    int L = 0; // 0=s, 1=px, 2=py, 3=pz
    int ang = 0;
    int valence = 1;
    double zeta = 0;
    std::vector<double> alp;
    std::vector<double> cont;
};

void build_basis(const std::vector<const Gfn0Element*>& el, int n_at,
                 std::vector<AO>& aos, std::vector<int>& ao0) {
    aos.clear();
    ao0.assign(static_cast<std::size_t>(n_at) + 1, 0);
    for (int a = 0; a < n_at; ++a) {
        const auto& e = *el[static_cast<std::size_t>(a)];
        double val_s_a[8] = {};
        double val_s_c[8] = {};
        int val_s_n = 0;
        for (int sh = 0; sh < e.n_shell; ++sh) {
            const int ng = e.nprim[sh];
            double al[8] = {};
            double co[8] = {};
            slater_to_gauss(ng, e.pqn[sh], e.ang[sh], e.zeta[sh], al, co);
            if (e.ang[sh] == 0 && e.valence[sh] != 0) {
                val_s_n = ng;
                for (int i = 0; i < ng; ++i) {
                    val_s_a[i] = al[i];
                    val_s_c[i] = co[i];
                }
            }
            std::vector<double> alp(al, al + ng);
            std::vector<double> cont(co, co + ng);
            if (e.ang[sh] == 0 && e.valence[sh] == 0) {
                // Polarisation s: orthogonalise against valence s.
                const double ss =
                    contracted_s_overlap(al, co, ng, val_s_a, val_s_c, val_s_n);
                alp.resize(static_cast<std::size_t>(ng + val_s_n));
                cont.resize(static_cast<std::size_t>(ng + val_s_n));
                for (int i = 0; i < ng; ++i) {
                    alp[static_cast<std::size_t>(i)] = al[i];
                    cont[static_cast<std::size_t>(i)] = co[i];
                }
                for (int i = 0; i < val_s_n; ++i) {
                    alp[static_cast<std::size_t>(ng + i)] = val_s_a[i];
                    cont[static_cast<std::size_t>(ng + i)] = -ss * val_s_c[i];
                }
                const double nrm = contracted_s_overlap(
                    alp.data(), cont.data(), ng + val_s_n, alp.data(),
                    cont.data(), ng + val_s_n);
                const double inv = 1.0 / std::sqrt(nrm);
                for (double& c : cont)
                    c *= inv;
            }
            const int ncart = (e.ang[sh] == 0) ? 1 : 3;
            for (int c = 0; c < ncart; ++c) {
                AO ao;
                ao.atom = a;
                ao.shell = sh;
                ao.ang = e.ang[sh];
                ao.L = (e.ang[sh] == 0) ? 0 : (c + 1);
                ao.valence = e.valence[sh];
                ao.zeta = e.zeta[sh];
                ao.alp = alp;
                ao.cont = cont;
                aos.push_back(std::move(ao));
            }
        }
        ao0[static_cast<std::size_t>(a) + 1] =
            static_cast<int>(aos.size());
    }
}

double ao_overlap(const AO& u, const double Ru[3], const AO& v,
                  const double Rv[3]) {
    double s = 0;
    for (std::size_t i = 0; i < u.alp.size(); ++i)
        for (std::size_t j = 0; j < v.alp.size(); ++j)
            s += u.cont[i] * v.cont[j]
                 * primitive_overlap(u.alp[i], Ru, u.L, v.alp[j], Rv, v.L);
    return s;
}

void ao_overlap_grad(const AO& u, const double Ru[3], const AO& v,
                     const double Rv[3], double& s, double ds_dRu[3]) {
    s = 0;
    ds_dRu[0] = ds_dRu[1] = ds_dRu[2] = 0;
    for (std::size_t i = 0; i < u.alp.size(); ++i) {
        for (std::size_t j = 0; j < v.alp.size(); ++j) {
            double sp = 0;
            double dsp[3];
            primitive_overlap_grad(u.alp[i], Ru, u.L, v.alp[j], Rv, v.L, sp,
                                   dsp);
            const double c = u.cont[i] * v.cont[j];
            s += c * sp;
            ds_dRu[0] += c * dsp[0];
            ds_dRu[1] += c * dsp[1];
            ds_dRu[2] += c * dsp[2];
        }
    }
}

// dcndr[moved * n_at + cn_atom][xyz] = d CN_{cn_atom} / d R_{moved,xyz}
void coordination_numbers(const std::vector<const Gfn0Element*>& el,
                          const std::vector<Grad3>& xyz, int n_at,
                          std::vector<double>& cn, bool want_grad,
                          std::vector<Grad3>& dcndr) {
    constexpr double kcn = 7.5;
    constexpr double cutoff = 40.0;
    constexpr double cnmax = 8.0;
    cn.assign(static_cast<std::size_t>(n_at), 0.0);
    if (want_grad)
        dcndr.assign(static_cast<std::size_t>(n_at) * static_cast<std::size_t>(n_at),
                     Grad3{0, 0, 0});
    const double cut2 = cutoff * cutoff;
    for (int i = 0; i < n_at; ++i) {
        const double ri =
            el[static_cast<std::size_t>(i)]->rcov_d3_ang * kAatoau * 4.0 / 3.0;
        for (int j = 0; j <= i; ++j) {
            double rij[3];
            double r2 = 0;
            for (int k = 0; k < 3; ++k) {
                rij[k] = xyz[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)]
                         - xyz[static_cast<std::size_t>(j)][static_cast<std::size_t>(k)];
                r2 += rij[k] * rij[k];
            }
            if (r2 > cut2 || r2 < 1.0e-12)
                continue;
            const double r = std::sqrt(r2);
            const double rj =
                el[static_cast<std::size_t>(j)]->rcov_d3_ang * kAatoau * 4.0
                / 3.0;
            const double r0 = ri + rj;
            const double c = 0.5 * (1.0 + std::erf(-kcn * (r - r0) / r0));
            cn[static_cast<std::size_t>(i)] += c;
            if (i != j)
                cn[static_cast<std::size_t>(j)] += c;
            if (want_grad) {
                const double derf =
                    -kcn / (kSqrtPi * r0)
                    * std::exp(-kcn * kcn * (r - r0) * (r - r0) / (r0 * r0));
                const double invr = 1.0 / r;
                const int n = n_at;
                for (int k = 0; k < 3; ++k) {
                    const double d = derf * rij[k] * invr;
                    dcndr[static_cast<std::size_t>(i * n + i)][static_cast<std::size_t>(k)] +=
                        d;
                    dcndr[static_cast<std::size_t>(j * n + j)][static_cast<std::size_t>(k)] -=
                        d;
                    dcndr[static_cast<std::size_t>(i * n + j)][static_cast<std::size_t>(k)] +=
                        d;
                    dcndr[static_cast<std::size_t>(j * n + i)][static_cast<std::size_t>(k)] -=
                        d;
                }
            }
        }
    }
    const double logcut = std::log(1.0 + std::exp(cnmax));
    for (int i = 0; i < n_at; ++i) {
        const double cni = cn[static_cast<std::size_t>(i)];
        if (want_grad) {
            const double dcut = std::exp(cnmax) / (std::exp(cnmax) + std::exp(cni));
            for (int m = 0; m < n_at; ++m) {
                auto& g =
                    dcndr[static_cast<std::size_t>(m * n_at + i)];
                g[0] *= dcut;
                g[1] *= dcut;
                g[2] *= dcut;
            }
        }
        cn[static_cast<std::size_t>(i)] =
            logcut - std::log(1.0 + std::exp(cnmax - cni));
    }
}

void eeq_charges(const std::vector<const Gfn0Element*>& el,
                 const std::vector<Grad3>& xyz, int n_at, int chrg,
                 const std::vector<double>& cn, std::vector<double>& qat,
                 double& e_ies, bool want_grad, const std::vector<Grad3>& dcndr,
                 std::vector<Grad3>& g, std::vector<Grad3>& dqdr) {
    const int ndim = n_at + 1;
    linalg::RealMatrix J = xt::zeros<double>({ndim, ndim});
    linalg::RealVector x = xt::zeros<double>({ndim});
    std::vector<double> dxdcn(static_cast<std::size_t>(n_at), 0.0);
    for (int i = 0; i < n_at; ++i) {
        const auto& ei = *el[static_cast<std::size_t>(i)];
        const double tmp =
            ei.eeq_kcn / (std::sqrt(cn[static_cast<std::size_t>(i)]) + 1.0e-14);
        x(i) = -ei.eeq_chi + tmp * cn[static_cast<std::size_t>(i)];
        dxdcn[static_cast<std::size_t>(i)] = 0.5 * tmp;
        const double gii = std::sqrt(0.5) / ei.eeq_alp;
        J(i, i) = 2.0 / kSqrtPi * gii + ei.eeq_gam;
        J(i, n_at) = 1.0;
        J(n_at, i) = 1.0;
        for (int j = 0; j < i; ++j) {
            const auto& ej = *el[static_cast<std::size_t>(j)];
            double r2 = 0;
            for (int k = 0; k < 3; ++k) {
                const double d =
                    xyz[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)]
                    - xyz[static_cast<std::size_t>(j)][static_cast<std::size_t>(k)];
                r2 += d * d;
            }
            const double r = std::sqrt(r2);
            const double gij =
                1.0 / std::sqrt(ei.eeq_alp * ei.eeq_alp + ej.eeq_alp * ej.eeq_alp);
            const double t = std::erf(gij * r) / r;
            J(i, j) = t;
            J(j, i) = t;
        }
    }
    x(n_at) = static_cast<double>(chrg);
    J(n_at, n_at) = 0.0;
    linalg::RealMatrix Jwork = J;
    linalg::RealVector q = xt::linalg::solve(Jwork, x);
    qat.resize(static_cast<std::size_t>(n_at));
    for (int i = 0; i < n_at; ++i)
        qat[static_cast<std::size_t>(i)] = q(i);
    linalg::RealVector Jq = xt::linalg::dot(J, q);
    e_ies = 0;
    for (int i = 0; i < ndim; ++i)
        e_ies += q(i) * (0.5 * Jq(i) - x(i));
    if (!want_grad)
        return;

    // djdr[moved][col][xyz]; col in 0..ndim-1 (Lagrange unused).
    std::vector<std::vector<Grad3>> djdr(
        static_cast<std::size_t>(n_at),
        std::vector<Grad3>(static_cast<std::size_t>(ndim), Grad3{0, 0, 0}));
    std::vector<Grad3> djdtr(static_cast<std::size_t>(n_at), Grad3{0, 0, 0});
    for (int i = 0; i < n_at; ++i) {
        const auto& ei = *el[static_cast<std::size_t>(i)];
        for (int j = 0; j < i; ++j) {
            const auto& ej = *el[static_cast<std::size_t>(j)];
            double vec[3];
            double r2 = 0;
            for (int k = 0; k < 3; ++k) {
                // vec = R_j - R_i (pprcht)
                vec[k] = xyz[static_cast<std::size_t>(j)][static_cast<std::size_t>(k)]
                         - xyz[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)];
                r2 += vec[k] * vec[k];
            }
            const double gij2 =
                1.0 / (ei.eeq_alp * ei.eeq_alp + ej.eeq_alp * ej.eeq_alp);
            const double g1 = std::erf(std::sqrt(gij2 * r2)) / std::sqrt(r2);
            const double pref =
                (2.0 * std::sqrt(gij2) * std::exp(-gij2 * r2) / kSqrtPi - g1) / r2;
            double dG[3] = {pref * vec[0], pref * vec[1], pref * vec[2]};
            const double qi = q(i);
            const double qj = q(j);
            for (int k = 0; k < 3; ++k) {
                djdr[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]
                    [static_cast<std::size_t>(k)] -= dG[k] * qi;
                djdr[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)]
                    [static_cast<std::size_t>(k)] += dG[k] * qj;
                djdtr[static_cast<std::size_t>(j)][static_cast<std::size_t>(k)] +=
                    dG[k] * qi;
                djdtr[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)] -=
                    dG[k] * qj;
            }
        }
    }
    for (int iat = 0; iat < n_at; ++iat) {
        const double dx = dxdcn[static_cast<std::size_t>(iat)];
        for (int moved = 0; moved < n_at; ++moved) {
            const auto& dcn =
                dcndr[static_cast<std::size_t>(moved * n_at + iat)];
            for (int k = 0; k < 3; ++k)
                djdr[static_cast<std::size_t>(moved)][static_cast<std::size_t>(iat)]
                    [static_cast<std::size_t>(k)] -= dx * dcn[static_cast<std::size_t>(k)];
        }
    }
    for (int moved = 0; moved < n_at; ++moved) {
        for (int col = 0; col < ndim; ++col) {
            const double qc = q(col);
            for (int k = 0; k < 3; ++k)
                g[static_cast<std::size_t>(moved)][static_cast<std::size_t>(k)] +=
                    djdr[static_cast<std::size_t>(moved)][static_cast<std::size_t>(col)]
                        [static_cast<std::size_t>(k)]
                    * qc;
        }
    }
    for (int iat = 0; iat < n_at; ++iat) {
        for (int k = 0; k < 3; ++k)
            djdr[static_cast<std::size_t>(iat)][static_cast<std::size_t>(iat)]
                [static_cast<std::size_t>(k)] +=
                djdtr[static_cast<std::size_t>(iat)][static_cast<std::size_t>(k)];
    }
    linalg::RealMatrix Jinv = xt::linalg::inv(J);
    dqdr.assign(static_cast<std::size_t>(n_at) * static_cast<std::size_t>(n_at),
                Grad3{0, 0, 0});
    for (int moved = 0; moved < n_at; ++moved) {
        for (int a = 0; a < n_at; ++a) {
            Grad3 acc{0, 0, 0};
            for (int col = 0; col < ndim; ++col) {
                const double invca = Jinv(col, a);
                const auto& djc =
                    djdr[static_cast<std::size_t>(moved)][static_cast<std::size_t>(col)];
                acc[0] -= djc[0] * invca;
                acc[1] -= djc[1] * invca;
                acc[2] -= djc[2] * invca;
            }
            dqdr[static_cast<std::size_t>(moved * n_at + a)] = acc;
        }
    }
}

double repulsion_energy(const std::vector<const Gfn0Element*>& el,
                        const std::vector<Grad3>& xyz, int n_at, bool want_grad,
                        std::vector<Grad3>& g) {
    double erep = 0;
    constexpr double rthr = 1600.0;
    for (int i = 0; i < n_at; ++i) {
        const auto& ei = *el[static_cast<std::size_t>(i)];
        for (int j = 0; j < i; ++j) {
            const auto& ej = *el[static_cast<std::size_t>(j)];
            double rij[3];
            double r2 = 0;
            for (int k = 0; k < 3; ++k) {
                rij[k] =
                    xyz[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)]
                    - xyz[static_cast<std::size_t>(j)][static_cast<std::size_t>(k)];
                r2 += rij[k] * rij[k];
            }
            if (r2 > rthr || r2 < 1.0e-6)
                continue;
            const double r = std::sqrt(r2);
            const double den2 = (ei.en_ham - ej.en_ham) * (ei.en_ham - ej.en_ham);
            const double den4 = den2 * den2;
            const double alpha =
                std::sqrt(ei.rep_alpha * ej.rep_alpha)
                * (1.0 + (0.01 * den2 + 0.01 * den4) * kGfn0RepEnScale);
            const double repab = ei.zeff * ej.zeff;
            const double r2top34 = std::pow(r2, 0.75);
            const double expterm = std::exp(-alpha * r2top34) * repab;
            erep += expterm / r;
            if (want_grad) {
                const double r2top34top2 = r2top34 * r2top34;
                const double dtmp =
                    expterm * (1.5 * alpha * r2top34 + 1.0) / r2top34top2;
                for (int k = 0; k < 3; ++k) {
                    g[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)] -=
                        dtmp * rij[k];
                    g[static_cast<std::size_t>(j)][static_cast<std::size_t>(k)] +=
                        dtmp * rij[k];
                }
            }
        }
    }
    return erep;
}

int srb_row(int z) {
    if (z >= 1 && z <= 2)
        return 0;
    if (z <= 10)
        return 1;
    if (z <= 18)
        return 2;
    return 3;
}

double srb_energy(const std::vector<const Gfn0Element*>& el,
                  const std::vector<Grad3>& xyz, int n_at,
                  const std::vector<double>& cn, bool want_grad,
                  const std::vector<Grad3>& dcndr, std::vector<Grad3>& g) {
    constexpr double p[4][2] = {
        {29.84522887, -8.87843763},
        {-1.70549806, 2.10878369},
        {6.54013762, 0.08009374},
        {6.39169003, -0.85808076},
    };
    double esrb = 0;
    for (int i = 0; i < n_at; ++i) {
        const int zi = el[static_cast<std::size_t>(i)]->z;
        if (zi < 5 || zi > 9)
            continue;
        for (int j = i + 1; j < n_at; ++j) {
            const int zj = el[static_cast<std::size_t>(j)]->z;
            if (zj < 5 || zj > 9)
                continue;
            if (zi == zj)
                continue;
            double rij[3];
            double r2 = 0;
            for (int k = 0; k < 3; ++k) {
                rij[k] =
                    xyz[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)]
                    - xyz[static_cast<std::size_t>(j)][static_cast<std::size_t>(k)];
                r2 += rij[k] * rij[k];
            }
            if (r2 > 200.0)
                continue;
            const auto& ei = *el[static_cast<std::size_t>(i)];
            const auto& ej = *el[static_cast<std::size_t>(j)];
            const int ir = srb_row(zi);
            const int jr = srb_row(zj);
            const double ra =
                ei.srb_r0 + ei.srb_cnfak * cn[static_cast<std::size_t>(i)]
                + kGfn0SrbShift;
            const double rb =
                ej.srb_r0 + ej.srb_cnfak * cn[static_cast<std::size_t>(j)]
                + kGfn0SrbShift;
            const double den_srb = std::fabs(ei.srb_en - ej.srb_en);
            const double k1 = 0.005 * (p[ir][0] + p[jr][0]);
            const double k2 = 0.005 * (p[ir][1] + p[jr][1]);
            const double ff = 1.0 - k1 * den_srb - k2 * den_srb * den_srb;
            const double rab0 = (ra + rb) * ff;
            const double rab = std::sqrt(r2);
            const double dr = rab - rab0;
            const double den = pauling_en(zi) - pauling_en(zj);
            const double pre =
                kGfn0SrbSteepness * (1.0 + kGfn0SrbEnScale * den * den);
            const double expterm = kGfn0SrbPrefactor * std::exp(-pre * dr * dr);
            esrb += expterm;
            if (want_grad) {
                const double dtmp = 2.0 * pre * dr * expterm;
                const double invrab = 1.0 / rab;
                for (int k = 0; k < 3; ++k) {
                    g[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)] -=
                        dtmp * rij[k] * invrab;
                    g[static_cast<std::size_t>(j)][static_cast<std::size_t>(k)] +=
                        dtmp * rij[k] * invrab;
                }
                // dE/d rab0 = dtmp; rab0 = ff*(r0i(CN_i)+r0j(CN_j))
                for (int moved = 0; moved < n_at; ++moved) {
                    const auto& dci =
                        dcndr[static_cast<std::size_t>(moved * n_at + i)];
                    const auto& dcj =
                        dcndr[static_cast<std::size_t>(moved * n_at + j)];
                    for (int k = 0; k < 3; ++k) {
                        const double drab0 =
                            ff
                            * (ei.srb_cnfak * dci[static_cast<std::size_t>(k)]
                               + ej.srb_cnfak * dcj[static_cast<std::size_t>(k)]);
                        g[static_cast<std::size_t>(moved)][static_cast<std::size_t>(k)] +=
                            dtmp * drab0;
                    }
                }
            }
        }
    }
    return esrb;
}

void self_energies(const std::vector<const Gfn0Element*>& el, int n_at,
                   const std::vector<double>& cn, const std::vector<double>& qat,
                   std::vector<std::array<double, 2>>& se,
                   std::vector<std::array<double, 2>>& dSEdcn,
                   std::vector<std::array<double, 2>>& dSEdq, bool want_grad) {
    se.resize(static_cast<std::size_t>(n_at));
    if (want_grad) {
        dSEdcn.resize(static_cast<std::size_t>(n_at));
        dSEdq.resize(static_cast<std::size_t>(n_at));
    }
    for (int a = 0; a < n_at; ++a) {
        const auto& e = *el[static_cast<std::size_t>(a)];
        const double cna = cn[static_cast<std::size_t>(a)];
        const double qa = qat[static_cast<std::size_t>(a)];
        for (int sh = 0; sh < e.n_shell; ++sh) {
            se[static_cast<std::size_t>(a)][static_cast<std::size_t>(sh)] =
                e.self_e[sh] - e.k_cn[sh] * cna - e.k_q_shell[sh] * qa
                - e.k_q_atom * qa * qa;
            if (want_grad) {
                dSEdcn[static_cast<std::size_t>(a)][static_cast<std::size_t>(sh)] =
                    -e.k_cn[sh];
                dSEdq[static_cast<std::size_t>(a)][static_cast<std::size_t>(sh)] =
                    -e.k_q_shell[sh] - 2.0 * e.k_q_atom * qa;
            }
        }
    }
}

void dshell_poly(double iPoly, double jPoly, double iRad, double jRad,
                 const double ri[3], const double rj[3], double& shpoly,
                 double dxyz[3]) {
    constexpr double aexp = 0.5;
    double rij[3];
    double rab2 = 0;
    for (int k = 0; k < 3; ++k) {
        rij[k] = ri[k] - rj[k];
        rab2 += rij[k] * rij[k];
    }
    const double rab = std::sqrt(rab2);
    const double rr = iRad + jRad;
    const double t14 = std::pow(rab / rr, aexp);
    const double k1 = iPoly * 0.01;
    const double k2 = jPoly * 0.01;
    const double t15 = k1 * t14;
    const double t23 = k2 * t14;
    shpoly = (1.0 + t15) * (1.0 + t23);
    const double fac =
        (t15 * (1.0 + t23) + (1.0 + t15) * t23) * aexp / rab2;
    dxyz[0] = fac * rij[0];
    dxyz[1] = fac * rij[1];
    dxyz[2] = fac * rij[2];
}

double eht_energy(const std::vector<const Gfn0Element*>& el,
                  const std::vector<Grad3>& xyz, int n_at,
                  const std::vector<AO>& aos,
                  const std::vector<std::array<double, 2>>& se, int n_occ,
                  bool want_grad,
                  const std::vector<std::array<double, 2>>& dSEdcn,
                  const std::vector<std::array<double, 2>>& dSEdq,
                  const std::vector<Grad3>& dcndr,
                  const std::vector<Grad3>& dqdr, std::vector<Grad3>& g) {
    const int nao = static_cast<int>(aos.size());
    linalg::RealMatrix H = xt::zeros<double>({nao, nao});
    linalg::RealMatrix S = xt::zeros<double>({nao, nao});
    std::vector<Grad3> R(static_cast<std::size_t>(n_at));
    for (int a = 0; a < n_at; ++a)
        R[static_cast<std::size_t>(a)] = xyz[static_cast<std::size_t>(a)];

    for (int mu = 0; mu < nao; ++mu) {
        const auto& u = aos[static_cast<std::size_t>(mu)];
        const double hii =
            se[static_cast<std::size_t>(u.atom)][static_cast<std::size_t>(u.shell)];
        S(mu, mu) = 1.0;
        H(mu, mu) = hii;
        for (int nu = 0; nu < mu; ++nu) {
            const auto& v = aos[static_cast<std::size_t>(nu)];
            if (u.atom == v.atom)
                continue;
            const auto& ei = *el[static_cast<std::size_t>(u.atom)];
            const auto& ej = *el[static_cast<std::size_t>(v.atom)];
            const double* Ri = R[static_cast<std::size_t>(u.atom)].data();
            const double* Rj = R[static_cast<std::size_t>(v.atom)].data();
            double r2 = 0;
            for (int k = 0; k < 3; ++k) {
                const double d = Ri[k] - Rj[k];
                r2 += d * d;
            }
            const double r = std::sqrt(r2);
            const double sij = ao_overlap(u, Ri, v, Rj);
            const double den = ei.en_ham - ej.en_ham;
            const double den2 = den * den;
            const double den4 = den2 * den2;
            const double ens = en_scale(v.ang, u.ang);
            const double enpoly = 1.0 + ens * den2 + kGfn0EnScale4 * ens * den4;
            const double zetaij =
                2.0 * std::sqrt(u.zeta * v.zeta) / (u.zeta + v.zeta);
            double km = k_scale(v.ang, u.ang) * zetaij * enpoly;
            const bool vali = u.valence != 0;
            const bool valj = v.valence != 0;
            if (!vali && !valj)
                km = 0.0;
            else if (!vali || !valj)
                km *= kGfn0Kdiff;
            const double hjj =
                se[static_cast<std::size_t>(v.atom)][static_cast<std::size_t>(
                    v.shell)];
            const double hav = 0.5 * km * (hii + hjj);
            const double rr = ei.rad_ang * kAatoau + ej.rad_ang * kAatoau;
            const double rrab = r / rr;
            const double rf1 =
                1.0 + 0.01 * ei.shell_poly[u.ang] * std::sqrt(rrab);
            const double rf2 =
                1.0 + 0.01 * ej.shell_poly[v.ang] * std::sqrt(rrab);
            const double shpoly = rf1 * rf2;
            S(mu, nu) = S(nu, mu) = sij;
            H(mu, nu) = H(nu, mu) = hav * shpoly * sij;
        }
    }
    auto sol = linalg::diagonalize_fock_generalized(H, S);
    const auto& eps = sol.first;
    const auto& C = sol.second;
    double e_ev = 0;
    for (int o = 0; o < n_occ; ++o)
        e_ev += 2.0 * eps(o);
    const double e_ha = e_ev * kGfn0EvToAu;
    if (!want_grad)
        return e_ha;

    linalg::RealMatrix P = xt::zeros<double>({nao, nao});
    linalg::RealMatrix W = xt::zeros<double>({nao, nao});
    for (int o = 0; o < n_occ; ++o) {
        const double wfac = 2.0 * eps(o) * kGfn0EvToAu;
        for (int mu = 0; mu < nao; ++mu) {
            const double cmu = C(mu, o);
            for (int nu = 0; nu < nao; ++nu) {
                const double cc = cmu * C(nu, o);
                P(mu, nu) += 2.0 * cc;
                W(mu, nu) += wfac * cc;
            }
        }
    }

    std::vector<double> dHdcn(static_cast<std::size_t>(n_at), 0.0);
    std::vector<double> dHdq(static_cast<std::size_t>(n_at), 0.0);

    for (int mu = 0; mu < nao; ++mu) {
        const auto& u = aos[static_cast<std::size_t>(mu)];
        const int iat = u.atom;
        const double Pii = P(mu, mu) * kGfn0EvToAu;
        dHdcn[static_cast<std::size_t>(iat)] +=
            Pii
            * dSEdcn[static_cast<std::size_t>(iat)][static_cast<std::size_t>(
                u.shell)];
        dHdq[static_cast<std::size_t>(iat)] +=
            Pii
            * dSEdq[static_cast<std::size_t>(iat)][static_cast<std::size_t>(
                u.shell)];
        const double hii =
            se[static_cast<std::size_t>(iat)][static_cast<std::size_t>(u.shell)];
        for (int nu = 0; nu < nao; ++nu) {
            const auto& v = aos[static_cast<std::size_t>(nu)];
            const int jat = v.atom;
            if (jat >= iat)
                continue;
            const auto& ei = *el[static_cast<std::size_t>(iat)];
            const auto& ej = *el[static_cast<std::size_t>(jat)];
            const double* Ri = R[static_cast<std::size_t>(iat)].data();
            const double* Rj = R[static_cast<std::size_t>(jat)].data();
            const double den = ei.en_ham - ej.en_ham;
            const double den2 = den * den;
            const double den4 = den2 * den2;
            const double ens = en_scale(v.ang, u.ang);
            const double enpoly = 1.0 + ens * den2 + kGfn0EnScale4 * ens * den4;
            const double zetaij =
                2.0 * std::sqrt(u.zeta * v.zeta) / (u.zeta + v.zeta);
            double km = k_scale(v.ang, u.ang) * zetaij * enpoly;
            const bool vali = u.valence != 0;
            const bool valj = v.valence != 0;
            if (!vali && !valj)
                km = 0.0;
            else if (!vali || !valj)
                km *= kGfn0Kdiff;
            const double hjj =
                se[static_cast<std::size_t>(jat)][static_cast<std::size_t>(
                    v.shell)];
            const double hav = 0.5 * km * (hii + hjj);
            double shpoly = 0;
            double dsh[3];
            dshell_poly(ei.shell_poly[u.ang], ej.shell_poly[v.ang],
                        ei.rad_ang * kAatoau, ej.rad_ang * kAatoau, Ri, Rj,
                        shpoly, dsh);
            double sij = 0;
            double ds[3];
            ao_overlap_grad(u, Ri, v, Rj, sij, ds);
            const double Pij = P(nu, mu) * kGfn0EvToAu;
            const double Hij = hav * shpoly;
            const double HPij = Hij * Pij;
            const double stmp = 2.0 * (HPij - W(nu, mu));
            const double H0sr = 2.0 * HPij * sij;
            double gxyz[3];
            for (int k = 0; k < 3; ++k)
                gxyz[k] = stmp * ds[k] + H0sr * dsh[k] / shpoly;
            for (int k = 0; k < 3; ++k) {
                g[static_cast<std::size_t>(iat)][static_cast<std::size_t>(k)] +=
                    gxyz[k];
                g[static_cast<std::size_t>(jat)][static_cast<std::size_t>(k)] -=
                    gxyz[k];
            }
            const double HPkm = km * shpoly * Pij * sij;
            dHdcn[static_cast<std::size_t>(iat)] +=
                HPkm
                * dSEdcn[static_cast<std::size_t>(iat)]
                        [static_cast<std::size_t>(u.shell)];
            dHdcn[static_cast<std::size_t>(jat)] +=
                HPkm
                * dSEdcn[static_cast<std::size_t>(jat)]
                        [static_cast<std::size_t>(v.shell)];
            dHdq[static_cast<std::size_t>(iat)] +=
                HPkm
                * dSEdq[static_cast<std::size_t>(iat)]
                       [static_cast<std::size_t>(u.shell)];
            dHdq[static_cast<std::size_t>(jat)] +=
                HPkm
                * dSEdq[static_cast<std::size_t>(jat)]
                       [static_cast<std::size_t>(v.shell)];
        }
    }
    for (int moved = 0; moved < n_at; ++moved) {
        for (int a = 0; a < n_at; ++a) {
            const auto& dcn =
                dcndr[static_cast<std::size_t>(moved * n_at + a)];
            const auto& dq =
                dqdr[static_cast<std::size_t>(moved * n_at + a)];
            const double dcnE = dHdcn[static_cast<std::size_t>(a)];
            const double dqE = dHdq[static_cast<std::size_t>(a)];
            for (int k = 0; k < 3; ++k) {
                g[static_cast<std::size_t>(moved)][static_cast<std::size_t>(k)] +=
                    dcn[static_cast<std::size_t>(k)] * dcnE
                    + dq[static_cast<std::size_t>(k)] * dqE;
            }
        }
    }
    return e_ha;
}

double energy_gfn0(const Frame& f, const StorageReal* x, const StorageReal* y,
                   const StorageReal* z, const int* type, int n_at, int chrg,
                   bool want_grad, std::vector<Grad3>* grad_ha_bohr) {
    std::vector<const Gfn0Element*> el(static_cast<std::size_t>(n_at));
    double n_elec = -static_cast<double>(chrg);
    for (int a = 0; a < n_at; ++a) {
        el[static_cast<std::size_t>(a)] = &element_row(type[a]);
        n_elec += static_cast<double>(type[a] - ncore(type[a]));
    }
    if (n_elec < 0.5 || std::fabs(n_elec - std::round(n_elec)) > 1e-9
        || static_cast<int>(std::lround(n_elec)) % 2 != 0)
        throw std::runtime_error(
            "GFN0: odd/non-integer valence electron count not supported");
    const int n_occ = static_cast<int>(std::lround(n_elec)) / 2;

    std::vector<Grad3> xyz(static_cast<std::size_t>(n_at));
    for (int a = 0; a < n_at; ++a) {
        xyz[static_cast<std::size_t>(a)] = {
            static_cast<double>(x[a]) * kAatoau,
            static_cast<double>(y[a]) * kAatoau,
            static_cast<double>(z[a]) * kAatoau};
    }

    std::vector<Grad3> g(static_cast<std::size_t>(n_at), Grad3{0, 0, 0});
    std::vector<double> cn;
    std::vector<Grad3> dcndr;
    coordination_numbers(el, xyz, n_at, cn, want_grad, dcndr);
    std::vector<double> qat;
    std::vector<Grad3> dqdr;
    double e_ies = 0;
    eeq_charges(el, xyz, n_at, chrg, cn, qat, e_ies, want_grad, dcndr, g,
                dqdr);
    const double e_rep = repulsion_energy(el, xyz, n_at, want_grad, g);
    const double e_srb =
        srb_energy(el, xyz, n_at, cn, want_grad, dcndr, g);

    std::vector<int> Z(static_cast<std::size_t>(n_at));
    for (int a = 0; a < n_at; ++a)
        Z[static_cast<std::size_t>(a)] = type[a];
    const double cell[3] = {1.0, 1.0, 1.0};
    double e_d4 = 0;
    if (want_grad) {
        const auto d4 = cpu::scf::dftd4(
            Z, xyz, cell, /*periodic=*/false, /*s6=*/1.0, kGfn0D4s8, kGfn0D4a1,
            kGfn0D4a2, /*ga=*/3.0, /*gc=*/2.0, /*wf=*/6.0,
            static_cast<double>(chrg), /*cn_cut=*/30.0, /*disp2_cut=*/60.0,
            /*s9=*/0.0);
        e_d4 = d4.energy;
        for (int a = 0; a < n_at; ++a) {
            g[static_cast<std::size_t>(a)][0] -= d4.forces[static_cast<std::size_t>(a)][0];
            g[static_cast<std::size_t>(a)][1] -= d4.forces[static_cast<std::size_t>(a)][1];
            g[static_cast<std::size_t>(a)][2] -= d4.forces[static_cast<std::size_t>(a)][2];
        }
    } else {
        e_d4 = cpu::scf::dftd4_energy(
            Z, xyz, cell, /*periodic=*/false, /*s6=*/1.0, kGfn0D4s8, kGfn0D4a1,
            kGfn0D4a2, /*ga=*/3.0, /*gc=*/2.0, /*wf=*/6.0,
            static_cast<double>(chrg), /*cn_cut=*/30.0, /*disp2_cut=*/60.0,
            /*s9=*/0.0);
    }

    std::vector<AO> aos;
    std::vector<int> ao0;
    build_basis(el, n_at, aos, ao0);
    std::vector<std::array<double, 2>> se, dSEdcn, dSEdq;
    self_energies(el, n_at, cn, qat, se, dSEdcn, dSEdq, want_grad);
    const double e_h0 = eht_energy(el, xyz, n_at, aos, se, n_occ, want_grad,
                                   dSEdcn, dSEdq, dcndr, dqdr, g);
    (void)f;
    if (want_grad && grad_ha_bohr)
        *grad_ha_bohr = std::move(g);
    return e_h0 + e_rep + e_ies + e_d4 + e_srb;
}

} // namespace

void GFN0::build(const BuildContext& /*ctx*/) {}

void GFN0::launch(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    const int n_at = f.nlocal();
    if (n_at < 1)
        throw std::invalid_argument("GFN0: empty frame");

    const auto type = f.span(kAtom_type);
    StorageReal* x = f.span(kAtom_x);
    StorageReal* y = f.span(kAtom_y);
    StorageReal* z = f.span(kAtom_z);
    AccReal* fx = f.span(kAtom_fx);
    AccReal* fy = f.span(kAtom_fy);
    AccReal* fz = f.span(kAtom_fz);
    double& energy_meta = f.meta.get(kEnergy);

    const bool do_f = with_forces && ctx.forces_requested;
    std::vector<Grad3> grad;
    const double e_ha =
        energy_gfn0(f, x, y, z, type, n_at, charge, do_f, do_f ? &grad : nullptr);
    energy_ = e_ha * units::kHartreeInEv;
    energy_meta += energy_;

    if (do_f) {
        // F[eV/Å] = −(dE_Ha/dR_Bohr) * (Hartree/eV) / a0[Å]
        const double fac = units::kHartreeInEv * kAatoau;
        for (int a = 0; a < n_at; ++a) {
            fx[a] += static_cast<AccReal>(-grad[static_cast<std::size_t>(a)][0] * fac);
            fy[a] += static_cast<AccReal>(-grad[static_cast<std::size_t>(a)][1] * fac);
            fz[a] += static_cast<AccReal>(-grad[static_cast<std::size_t>(a)][2] * fac);
        }
    }
}

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
