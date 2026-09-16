/// @file src/cpu/semiempirical/ipea.cpp
/// IPEA-xTB energy, analytic nuclear forces, and ΔSCC VIP/VEA (organic H/C/N/O).
#include "atomiverse/cpu/semiempirical/ipea.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/gpw/internal/dftd3.h"
#include "atomiverse/cpu/linalg/eigensolve.h"
#include "atomiverse/cpu/linalg/matrix.h"
#include "atomiverse/cpu/semiempirical/basis_data/ipea_tables.h"
#include "atomiverse/cpu/semiempirical/internal/scc_mixer.h"

namespace ATV {
namespace cpu {
namespace semiempirical {
namespace {

using basis_data::IpeaElement;
using basis_data::kIpeaD3a1;
using basis_data::kIpeaD3a2;
using basis_data::kIpeaD3k1;
using basis_data::kIpeaD3s6;
using basis_data::kIpeaD3s8;
using basis_data::kIpeaElements;
using basis_data::kIpeaEtempK;
using basis_data::kIpeaGExp;
using basis_data::kIpeaKdiff;
using basis_data::kIpeaKEn;
using basis_data::kIpeaKp;
using basis_data::kIpeaKs;
using basis_data::kIpeaKsp;
using basis_data::kIpeaNElements;
using basis_data::kIpeaRepKExp;
using basis_data::kIpeaShiftEv;

constexpr double kPi = 3.14159265358979323846;
constexpr double kA2Bohr = units::kAngstromInBohr;
constexpr double kHa2eV = units::kHartreeInEv;

const IpeaElement& element_row(int z) {
    for (std::size_t i = 0; i < kIpeaNElements; ++i)
        if (kIpeaElements[i].z == z)
            return kIpeaElements[i];
    throw std::runtime_error(
        "IPEA: unsupported element Z=" + std::to_string(z)
        + " (static organic tables cover H/C/N/O only; add a row to "
          "basis_data/ipea_tables.h)");
}

double pyykko_rcov_A(int z) {
    switch (z) {
    case 1:
        return 0.32;
    case 6:
        return 0.75;
    case 7:
        return 0.71;
    case 8:
        return 0.63;
    default:
        throw std::runtime_error("IPEA: no D3 rcov for Z=" + std::to_string(z));
    }
}

double d3_rcov_bohr(int z) { return (4.0 / 3.0) * pyykko_rcov_A(z) * kA2Bohr; }

double kshell(int la, int lb) {
    if ((la == 0 && lb == 1) || (la == 1 && lb == 0))
        return kIpeaKsp;
    const double ka = (la == 0) ? kIpeaKs : kIpeaKp;
    const double kb = (lb == 0) ? kIpeaKs : kIpeaKp;
    return 0.5 * (ka + kb);
}

double hscale_pair(const IpeaElement& ea, int ish, const IpeaElement& eb,
                   int jsh) {
    const bool va = ea.valence[ish];
    const bool vb = eb.valence[jsh];
    const int la = ea.ang[ish];
    const int lb = eb.ang[jsh];
    if (va && vb) {
        const double den = (ea.en - eb.en) * (ea.en - eb.en);
        return kshell(la, lb) * (1.0 + kIpeaKEn * den);
    }
    if (va && !vb)
        return 0.5 * (kshell(la, la) + kIpeaKdiff);
    if (!va && vb)
        return 0.5 * (kshell(lb, lb) + kIpeaKdiff);
    return kIpeaKdiff;
}

// Stewart STO-nG (JCP 52, 431 (1970)) as used by tblite basis/slater.f90.
void sto_ng(int ng, int n, int l, double zeta, std::vector<double>& alpha,
            std::vector<double>& coeff) {
    alpha.assign(static_cast<std::size_t>(ng), 0.0);
    coeff.assign(static_cast<std::size_t>(ng), 0.0);
    if (ng == 4 && n == 1 && l == 0) {
        alpha = {5.216844534, 9.546182760e-1, 2.652034102e-1, 8.801862774e-2};
        coeff = {5.675242080e-2, 2.601413550e-1, 5.328461143e-1,
                 2.916254405e-1};
    } else if (ng == 3 && n == 2 && l == 0) {
        alpha = {2.581578398, 1.567622104e-1, 6.018332272e-2};
        coeff = {-5.994474934e-2, 5.960385398e-1, 4.581786291e-1};
    } else if (ng == 3 && n == 3 && l == 0) {
        alpha = {5.641487709e-1, 6.924421391e-2, 3.269529097e-2};
        coeff = {-1.782577972e-1, 8.612761663e-1, 2.261841969e-1};
    } else if (ng == 6 && n == 2 && l == 0) {
        alpha = {2.768496241e+1, 5.077140627,    1.426786050,
                 2.040335729e-1, 9.260298399e-2, 4.416183978e-2};
        coeff = {-4.151277819e-3, -2.067024148e-2, -5.150303337e-2,
                 3.346271174e-1,  5.621061301e-1,  1.712994697e-1};
    } else if (ng == 6 && n == 2 && l == 1) {
        alpha = {5.868285913,    1.530329631,    5.475665231e-1,
                 2.288932733e-1, 1.046655969e-1, 4.948220127e-2};
        coeff = {7.924233646e-3, 5.144104825e-2, 1.898400060e-1,
                 4.049863191e-1, 4.012362861e-1, 1.051855189e-1};
    } else {
        throw std::runtime_error("IPEA: no Stewart STO-nG for n="
                                 + std::to_string(n) + " l=" + std::to_string(l)
                                 + " ng=" + std::to_string(ng));
    }
    const double z2 = zeta * zeta;
    for (int i = 0; i < ng; ++i) {
        alpha[static_cast<std::size_t>(i)] *= z2;
        const double a = alpha[static_cast<std::size_t>(i)];
        const double nrm =
            std::pow(2.0 * a / kPi, 0.75) * std::pow(4.0 * a, 0.5 * l);
        coeff[static_cast<std::size_t>(i)] *= nrm;
    }
}

struct Prim {
    double a = 0;
    double c = 0;
};

struct Shell {
    int atom = 0;
    int l = 0;
    int ao0 = 0;
    int nao = 1;
    bool valence = true;
    double h = 0;        ///< CN-shifted onsite [Hartree].
    double kcn = 0;      ///< dh/dCN = −kcn [Hartree].
    double shpoly = 0;   ///< k_poly.
    double hardness = 0; ///< η_l [Hartree].
    double n0 = 0;
    double gam3 = 0; ///< Atomic Γ (same on every shell of the atom).
    double rad = 0;
    std::vector<Prim> prim;
};

double ov1d(int la, int lb, double alpha, double beta, double Aa, double Ba) {
    const double gamma = alpha + beta;
    const double P = (alpha * Aa + beta * Ba) / gamma;
    const double mu = alpha * beta / gamma;
    const double dx = Aa - Ba;
    const double I00 = std::sqrt(kPi / gamma) * std::exp(-mu * dx * dx);
    if (la == 0 && lb == 0)
        return I00;
    const double I10 = (P - Aa) * I00;
    const double I01 = (P - Ba) * I00;
    if (la == 1 && lb == 0)
        return I10;
    if (la == 0 && lb == 1)
        return I01;
    return (P - Aa) * I01 + 0.5 / gamma * I00;
}

double ov3d(int lxa, int lya, int lza, double alpha, const double A[3], int lxb,
            int lyb, int lzb, double beta, const double B[3]) {
    return ov1d(lxa, lxb, alpha, beta, A[0], B[0])
           * ov1d(lya, lyb, alpha, beta, A[1], B[1])
           * ov1d(lza, lzb, alpha, beta, A[2], B[2]);
}

void ov1d_grad(int la, int lb, double alpha, double beta, double Aa, double Ba,
               double& I, double& dI_dA) {
    const double gamma = alpha + beta;
    const double mu = alpha * beta / gamma;
    const double dx = Aa - Ba;
    const double I00 = std::sqrt(kPi / gamma) * std::exp(-mu * dx * dx);
    const double dI00 = -2.0 * mu * dx * I00;
    const double P = (alpha * Aa + beta * Ba) / gamma;
    if (la == 0 && lb == 0) {
        I = I00;
        dI_dA = dI00;
        return;
    }
    const double I10 = (P - Aa) * I00;
    const double I01 = (P - Ba) * I00;
    const double dP = alpha / gamma;
    const double dI10 = (dP - 1.0) * I00 + (P - Aa) * dI00;
    const double dI01 = dP * I00 + (P - Ba) * dI00;
    if (la == 1 && lb == 0) {
        I = I10;
        dI_dA = dI10;
        return;
    }
    if (la == 0 && lb == 1) {
        I = I01;
        dI_dA = dI01;
        return;
    }
    I = (P - Aa) * I01 + 0.5 / gamma * I00;
    dI_dA = (dP - 1.0) * I01 + (P - Aa) * dI01 + 0.5 / gamma * dI00;
}

void ov3d_grad(int lxa, int lya, int lza, double alpha, const double A[3],
               int lxb, int lyb, int lzb, double beta, const double B[3],
               double& s, double ds_dA[3]) {
    double Ix = 0, Iy = 0, Iz = 0, dIx = 0, dIy = 0, dIz = 0;
    ov1d_grad(lxa, lxb, alpha, beta, A[0], B[0], Ix, dIx);
    ov1d_grad(lya, lyb, alpha, beta, A[1], B[1], Iy, dIy);
    ov1d_grad(lza, lzb, alpha, beta, A[2], B[2], Iz, dIz);
    s = Ix * Iy * Iz;
    ds_dA[0] = dIx * Iy * Iz;
    ds_dA[1] = Ix * dIy * Iz;
    ds_dA[2] = Ix * Iy * dIz;
}

void cart_pow(int l, int m, int pows[3][3]) {
    pows[m][0] = pows[m][1] = pows[m][2] = 0;
    if (l == 1)
        pows[m][m] = 1;
}

double self_overlap(const std::vector<Prim>& prim, int l) {
    double s = 0;
    const double A[3] = {0, 0, 0};
    const int nao = (l == 0) ? 1 : 3;
    for (int ia = 0; ia < nao; ++ia) {
        int pa[3][3] = {};
        cart_pow(l, ia, pa);
        for (const auto& p : prim)
            for (const auto& q : prim)
                s += p.c * q.c
                     * ov3d(pa[ia][0], pa[ia][1], pa[ia][2], p.a, A, pa[ia][0],
                            pa[ia][1], pa[ia][2], q.a, A);
    }
    return s / static_cast<double>(nao);
}

void renormalize(std::vector<Prim>& prim, int l) {
    const double s = self_overlap(prim, l);
    if (s <= 0)
        throw std::runtime_error("IPEA: non-positive shell self-overlap");
    const double nrm = 1.0 / std::sqrt(s);
    for (auto& p : prim)
        p.c *= nrm;
}

double ss_overlap(const std::vector<Prim>& a, const std::vector<Prim>& b) {
    double s = 0;
    const double A[3] = {0, 0, 0};
    for (const auto& p : a)
        for (const auto& q : b)
            s += p.c * q.c * ov3d(0, 0, 0, p.a, A, 0, 0, 0, q.a, A);
    return s;
}

void orthogonalize_s(std::vector<Prim>& target, const std::vector<Prim>& ref) {
    const double s = ss_overlap(ref, target);
    for (const auto& p : ref)
        target.push_back({p.a, -s * p.c});
    renormalize(target, 0);
}

void fill_shell_prims(Shell& sh, const IpeaElement& el, int ish,
                      const Shell* ortho_against) {
    std::vector<double> alpha, coeff;
    sto_ng(el.nprim[ish], el.pqn[ish], el.ang[ish], el.zeta[ish], alpha, coeff);
    sh.prim.resize(alpha.size());
    for (std::size_t i = 0; i < alpha.size(); ++i)
        sh.prim[i] = {alpha[i], coeff[i]};
    if (ortho_against)
        orthogonalize_s(sh.prim, ortho_against->prim);
}

void add_shell_block(const Shell& a, const double A[3], const Shell& b,
                     const double B[3], linalg::RealMatrix& S, double scale,
                     linalg::RealMatrix& H, double hij) {
    int pa[3][3] = {}, pb[3][3] = {};
    for (int ia = 0; ia < a.nao; ++ia)
        cart_pow(a.l, ia, pa);
    for (int ib = 0; ib < b.nao; ++ib)
        cart_pow(b.l, ib, pb);
    for (int ia = 0; ia < a.nao; ++ia) {
        for (int ib = 0; ib < b.nao; ++ib) {
            double sij = 0;
            for (const auto& p : a.prim)
                for (const auto& q : b.prim)
                    sij += p.c * q.c
                           * ov3d(pa[ia][0], pa[ia][1], pa[ia][2], p.a, A,
                                  pb[ib][0], pb[ib][1], pb[ib][2], q.a, B);
            const int mu = a.ao0 + ia;
            const int nu = b.ao0 + ib;
            S(mu, nu) += scale * sij;
            H(mu, nu) += scale * sij * hij;
        }
    }
}

double gamma_ko(double r, double eta_a, double eta_b) {
    // Harmonic-mean hardness; Klopman–Ohno with gexp = 1 (IPEA alphaj).
    const double eta = 2.0 / (1.0 / eta_a + 1.0 / eta_b);
    const double g = kIpeaGExp;
    return 1.0 / std::pow(std::pow(r, g) + std::pow(eta, -g), 1.0 / g);
}

double d3_cn(double r, double rcovab) {
    if (r < 1e-14)
        return 0;
    const double ee = std::exp(-kIpeaD3k1 * (rcovab / r - 1.0));
    return 1.0 / (1.0 + ee);
}

struct IpeaSystem {
    int n_at = 0;
    int n_ao = 0;
    int n_sh = 0;
    int n_elec = 0;
    int n_occ = 0;
    std::vector<const IpeaElement*> el;
    std::vector<std::array<double, 3>> xyz; // Bohr
    std::vector<double> cn;
    std::vector<Shell> sh;
    std::vector<int> atom_sh0;
    std::vector<int> atom_nsh;
    linalg::RealMatrix S;
    linalg::RealMatrix H0;
    std::vector<double> gamma;
};

IpeaSystem build_system(const StorageReal* x, const StorageReal* y,
                        const StorageReal* z, const int* type, int n_at,
                        int charge) {
    IpeaSystem sys;
    sys.n_at = n_at;
    sys.el.resize(static_cast<std::size_t>(n_at));
    sys.xyz.resize(static_cast<std::size_t>(n_at));
    double n_elec = -static_cast<double>(charge);
    for (int a = 0; a < n_at; ++a) {
        sys.el[static_cast<std::size_t>(a)] = &element_row(type[a]);
        n_elec += sys.el[static_cast<std::size_t>(a)]->n_val;
        sys.xyz[static_cast<std::size_t>(a)] = {
            static_cast<double>(x[a]) * kA2Bohr,
            static_cast<double>(y[a]) * kA2Bohr,
            static_cast<double>(z[a]) * kA2Bohr};
    }
    if (n_elec < 0.5 || std::fabs(n_elec - std::round(n_elec)) > 1e-9)
        throw std::runtime_error(
            "IPEA: non-integer valence electron count not supported");
    sys.n_elec = static_cast<int>(std::lround(n_elec));
    sys.n_occ = sys.n_elec / 2;

    sys.cn.assign(static_cast<std::size_t>(n_at), 0.0);
    for (int a = 0; a < n_at; ++a) {
        const double ra = d3_rcov_bohr(type[a]);
        for (int b = 0; b < n_at; ++b) {
            if (a == b)
                continue;
            const double dx = sys.xyz[static_cast<std::size_t>(a)][0]
                              - sys.xyz[static_cast<std::size_t>(b)][0];
            const double dy = sys.xyz[static_cast<std::size_t>(a)][1]
                              - sys.xyz[static_cast<std::size_t>(b)][1];
            const double dz = sys.xyz[static_cast<std::size_t>(a)][2]
                              - sys.xyz[static_cast<std::size_t>(b)][2];
            const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
            const double rb = d3_rcov_bohr(type[b]);
            sys.cn[static_cast<std::size_t>(a)] += d3_cn(r, ra + rb);
        }
    }

    sys.atom_sh0.resize(static_cast<std::size_t>(n_at) + 1, 0);
    sys.atom_nsh.resize(static_cast<std::size_t>(n_at), 0);
    int ao = 0;
    for (int a = 0; a < n_at; ++a) {
        const auto& ea = *sys.el[static_cast<std::size_t>(a)];
        sys.atom_sh0[static_cast<std::size_t>(a)] =
            static_cast<int>(sys.sh.size());
        for (int ish = 0; ish < ea.n_shell; ++ish) {
            Shell sh;
            sh.atom = a;
            sh.l = ea.ang[ish];
            sh.nao = (sh.l == 0) ? 1 : 3;
            sh.ao0 = ao;
            sh.valence = ea.valence[ish];
            sh.h = ea.level[ish]
                   - ea.kcn[ish] * sys.cn[static_cast<std::size_t>(a)];
            sh.kcn = ea.kcn[ish];
            sh.shpoly = ea.shpoly[ish];
            sh.hardness = ea.gam * ea.lgam[ish];
            sh.n0 = ea.refocc[ish];
            sh.gam3 = ea.gam3;
            sh.rad = ea.rad;
            fill_shell_prims(sh, ea, ish, nullptr);
            ao += sh.nao;
            sys.sh.push_back(std::move(sh));
            ++sys.atom_nsh[static_cast<std::size_t>(a)];
        }
    }
    sys.atom_sh0[static_cast<std::size_t>(n_at)] =
        static_cast<int>(sys.sh.size());
    sys.n_sh = static_cast<int>(sys.sh.size());
    sys.n_ao = ao;

    // Extra (n+1)s is Gram–Schmidt against the atom's valence s (tblite
    // orthogonalize). Redo after the vector is stable.
    for (int a = 0; a < n_at; ++a) {
        const int s0 = sys.atom_sh0[static_cast<std::size_t>(a)];
        const int ns = sys.atom_nsh[static_cast<std::size_t>(a)];
        const Shell* val_s = nullptr;
        for (int i = 0; i < ns; ++i) {
            Shell& sh = sys.sh[static_cast<std::size_t>(s0 + i)];
            if (sh.l == 0 && sh.valence)
                val_s = &sh;
        }
        for (int i = 0; i < ns; ++i) {
            Shell& sh = sys.sh[static_cast<std::size_t>(s0 + i)];
            if (sh.l == 0 && !sh.valence && val_s) {
                const auto& ea = *sys.el[static_cast<std::size_t>(a)];
                fill_shell_prims(sh, ea, i, val_s);
            }
        }
    }

    sys.S = xt::zeros<double>({sys.n_ao, sys.n_ao});
    sys.H0 = xt::zeros<double>({sys.n_ao, sys.n_ao});
    for (int i = 0; i < sys.n_sh; ++i) {
        const Shell& si = sys.sh[static_cast<std::size_t>(i)];
        const double* Ai = sys.xyz[static_cast<std::size_t>(si.atom)].data();
        for (int j = 0; j < sys.n_sh; ++j) {
            const Shell& sj = sys.sh[static_cast<std::size_t>(j)];
            const double* Bj =
                sys.xyz[static_cast<std::size_t>(sj.atom)].data();
            double hij = 0.5 * (si.h + sj.h);
            if (si.atom != sj.atom) {
                const double dx = Ai[0] - Bj[0];
                const double dy = Ai[1] - Bj[1];
                const double dz = Ai[2] - Bj[2];
                const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
                const double rr = std::sqrt(r / ((si.rad + sj.rad) * kA2Bohr));
                const double pi = 1.0 + si.shpoly * rr;
                const double pj = 1.0 + sj.shpoly * rr;
                const auto& ea = *sys.el[static_cast<std::size_t>(si.atom)];
                const auto& eb = *sys.el[static_cast<std::size_t>(sj.atom)];
                const int ish =
                    i - sys.atom_sh0[static_cast<std::size_t>(si.atom)];
                const int jsh =
                    j - sys.atom_sh0[static_cast<std::size_t>(sj.atom)];
                hij *= hscale_pair(ea, ish, eb, jsh) * pi * pj;
            }
            add_shell_block(si, Ai, sj, Bj, sys.S, 1.0, sys.H0, hij);
        }
    }

    sys.gamma.assign(static_cast<std::size_t>(sys.n_sh * sys.n_sh), 0.0);
    for (int i = 0; i < sys.n_sh; ++i) {
        const Shell& si = sys.sh[static_cast<std::size_t>(i)];
        const auto& Ai = sys.xyz[static_cast<std::size_t>(si.atom)];
        for (int j = 0; j < sys.n_sh; ++j) {
            const Shell& sj = sys.sh[static_cast<std::size_t>(j)];
            const auto& Bj = sys.xyz[static_cast<std::size_t>(sj.atom)];
            const double dx = Ai[0] - Bj[0];
            const double dy = Ai[1] - Bj[1];
            const double dz = Ai[2] - Bj[2];
            const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
            sys.gamma[static_cast<std::size_t>(i * sys.n_sh + j)] =
                gamma_ko(r, si.hardness, sj.hardness);
        }
    }
    return sys;
}

double repulsive_e(const IpeaSystem& sys) {
    double e = 0;
    for (int a = 0; a < sys.n_at; ++a) {
        const auto& ea = *sys.el[static_cast<std::size_t>(a)];
        const auto& A = sys.xyz[static_cast<std::size_t>(a)];
        for (int b = a + 1; b < sys.n_at; ++b) {
            const auto& eb = *sys.el[static_cast<std::size_t>(b)];
            const auto& B = sys.xyz[static_cast<std::size_t>(b)];
            const double dx = A[0] - B[0];
            const double dy = A[1] - B[1];
            const double dz = A[2] - B[2];
            const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (r < 1e-12)
                continue;
            const double kexp =
                std::sqrt(ea.alpha * eb.alpha) * std::pow(r, kIpeaRepKExp);
            e += ea.zeff * eb.zeff / r * std::exp(-kexp);
        }
    }
    return e;
}

double dispersion_e(const IpeaSystem& sys, const int* type) {
    std::vector<int> Z(static_cast<std::size_t>(sys.n_at));
    std::vector<std::array<double, 3>> pos(static_cast<std::size_t>(sys.n_at));
    for (int a = 0; a < sys.n_at; ++a) {
        Z[static_cast<std::size_t>(a)] = type[a];
        pos[static_cast<std::size_t>(a)] = sys.xyz[static_cast<std::size_t>(a)];
    }
    const double cell[3] = {0, 0, 0};
    const auto d3 =
        gpw::dftd3_pbe_bj(Z, pos, cell, false, gpw::D3Params{}, 94.2, kIpeaD3s6,
                          kIpeaD3s8, kIpeaD3a1, kIpeaD3a2);
    return d3.energy;
}

/// Aufbau (kT=0) or degenerate-shell sharing (ionized ΔSCC).
///
/// xtb `--vip` Fermi-smears the CH4 cation t2 hole to 5/3 each at 300 K
/// because its integrals keep Td degeneracy. ATV's Stewart/intcut residual
/// splits that t2 by ≫ kT, so 300 K Fermi collapses to 2+2+1 and drops
/// E(N-1) by ~0.1 eV. Sharing remaining electrons equally across MOs
/// within `deg_tol` (0.05 Eh, between intra-t2 noise and the ~0.23 Eh
/// a1–t2 gap) recovers xtb's 1.6667 occupations.
std::vector<double> occupations(const linalg::RealVector& eps, int n_ao,
                                int n_elec, double kT_ha) {
    std::vector<double> n(static_cast<std::size_t>(n_ao), 0.0);
    if (n_elec < 0 || n_elec > 2 * n_ao)
        throw std::runtime_error(
            "IPEA: n_elec out of range for the AO basis (n_elec="
            + std::to_string(n_elec) + ", n_ao=" + std::to_string(n_ao) + ")");
    if (!(kT_ha > 1e-12)) {
        const int nd = n_elec / 2;
        const int rem = n_elec % 2;
        for (int i = 0; i < nd; ++i)
            n[static_cast<std::size_t>(i)] = 2.0;
        if (rem)
            n[static_cast<std::size_t>(nd)] = 1.0;
        return n;
    }
    constexpr double deg_tol = 0.05; // Hartree
    int remaining = n_elec;
    int i = 0;
    while (remaining > 0 && i < n_ao) {
        int j = i + 1;
        while (j < n_ao && std::fabs(eps(j) - eps(i)) < deg_tol)
            ++j;
        const int nlev = j - i;
        const int cap = 2 * nlev;
        const int take = std::min(remaining, cap);
        const double per =
            static_cast<double>(take) / static_cast<double>(nlev);
        for (int k = i; k < j; ++k)
            n[static_cast<std::size_t>(k)] = per;
        remaining -= take;
        i = j;
    }
    return n;
}

void accumulate_density(const linalg::RealVector& eps,
                        const linalg::RealMatrix& C, int n_ao, int n_elec,
                        double kT_ha, linalg::RealMatrix& P,
                        linalg::RealMatrix* W) {
    const std::vector<double> nocc = occupations(eps, n_ao, n_elec, kT_ha);
    P = xt::zeros<double>({n_ao, n_ao});
    if (W)
        *W = xt::zeros<double>({n_ao, n_ao});
    for (int o = 0; o < n_ao; ++o) {
        const double ni = nocc[static_cast<std::size_t>(o)];
        if (ni < 1e-16)
            continue;
        for (int mu = 0; mu < n_ao; ++mu)
            for (int nu = 0; nu < n_ao; ++nu) {
                const double cc = C(mu, o) * C(nu, o);
                P(mu, nu) += ni * cc;
                if (W)
                    (*W)(mu, nu) += ni * eps(o) * cc;
            }
    }
}

struct SccOut {
    double energy_eh = 0;
    double residual = 0;
    int iterations = 0;
    linalg::RealMatrix P;
    linalg::RealMatrix W;
    std::vector<double> q;
    std::vector<double> V;
};

SccOut scc_energy(const IpeaSystem& sys, double scc_tol, int max_scc,
                  double mix_beta, double kT_ha) {
    std::vector<double> q(static_cast<std::size_t>(sys.n_sh), 0.0);
    internal::ChargeMixer mixer(mix_beta, 6);
    bool converged = false;
    SccOut out;
    linalg::RealMatrix H = sys.H0;
    std::pair<linalg::RealVector, linalg::RealMatrix> sol;
    for (int it = 0; it < max_scc; ++it) {
        std::vector<double> q_at(static_cast<std::size_t>(sys.n_at), 0.0);
        for (int i = 0; i < sys.n_sh; ++i)
            q_at[static_cast<std::size_t>(
                sys.sh[static_cast<std::size_t>(i)].atom)] +=
                q[static_cast<std::size_t>(i)];
        std::vector<double> V(static_cast<std::size_t>(sys.n_sh), 0.0);
        for (int i = 0; i < sys.n_sh; ++i) {
            double v = 0;
            for (int j = 0; j < sys.n_sh; ++j)
                v += sys.gamma[static_cast<std::size_t>(i * sys.n_sh + j)]
                     * q[static_cast<std::size_t>(j)];
            const int ia = sys.sh[static_cast<std::size_t>(i)].atom;
            v += sys.sh[static_cast<std::size_t>(i)].gam3
                 * q_at[static_cast<std::size_t>(ia)]
                 * q_at[static_cast<std::size_t>(ia)];
            V[static_cast<std::size_t>(i)] = v;
        }
        H = sys.H0;
        for (int i = 0; i < sys.n_sh; ++i) {
            const Shell& si = sys.sh[static_cast<std::size_t>(i)];
            for (int j = 0; j < sys.n_sh; ++j) {
                const Shell& sj = sys.sh[static_cast<std::size_t>(j)];
                const double vij = 0.5
                                   * (V[static_cast<std::size_t>(i)]
                                      + V[static_cast<std::size_t>(j)]);
                for (int ia = 0; ia < si.nao; ++ia)
                    for (int ja = 0; ja < sj.nao; ++ja)
                        H(si.ao0 + ia, sj.ao0 + ja) +=
                            vij * sys.S(si.ao0 + ia, sj.ao0 + ja);
            }
        }
        sol = linalg::diagonalize_fock_generalized(H, sys.S);
        const auto& C = sol.second;
        linalg::RealMatrix P;
        accumulate_density(sol.first, C, sys.n_ao, sys.n_elec, kT_ha, P,
                           nullptr);
        std::vector<double> q_out(static_cast<std::size_t>(sys.n_sh), 0.0);
        for (int i = 0; i < sys.n_sh; ++i) {
            const Shell& si = sys.sh[static_cast<std::size_t>(i)];
            double pop = 0;
            for (int ia = 0; ia < si.nao; ++ia)
                for (int nu = 0; nu < sys.n_ao; ++nu)
                    pop += P(si.ao0 + ia, nu) * sys.S(si.ao0 + ia, nu);
            q_out[static_cast<std::size_t>(i)] = pop - si.n0;
        }
        out.residual = 0;
        for (int i = 0; i < sys.n_sh; ++i)
            out.residual = std::max(
                out.residual, std::fabs(q_out[static_cast<std::size_t>(i)]
                                        - q[static_cast<std::size_t>(i)]));
        out.iterations = it + 1;
        if (out.residual < scc_tol) {
            q = q_out;
            converged = true;
            break;
        }
        q = mixer.next(q, q_out);
    }
    if (!converged)
        throw std::runtime_error(
            "IPEA: SCC did not converge in " + std::to_string(max_scc)
            + " iterations (residual " + std::to_string(out.residual)
            + " e > scc_tol " + std::to_string(scc_tol)
            + "); raise max_scc or lower mix_beta");

    const auto& C = sol.second;
    accumulate_density(sol.first, C, sys.n_ao, sys.n_elec, kT_ha, out.P,
                       &out.W);

    double e_h0 = 0;
    for (int mu = 0; mu < sys.n_ao; ++mu)
        for (int nu = 0; nu < sys.n_ao; ++nu)
            e_h0 += out.P(mu, nu) * sys.H0(mu, nu);
    double e2 = 0;
    for (int i = 0; i < sys.n_sh; ++i)
        for (int j = 0; j < sys.n_sh; ++j)
            e2 += 0.5 * q[static_cast<std::size_t>(i)]
                  * sys.gamma[static_cast<std::size_t>(i * sys.n_sh + j)]
                  * q[static_cast<std::size_t>(j)];
    std::vector<double> q_at(static_cast<std::size_t>(sys.n_at), 0.0);
    for (int i = 0; i < sys.n_sh; ++i)
        q_at[static_cast<std::size_t>(
            sys.sh[static_cast<std::size_t>(i)].atom)] +=
            q[static_cast<std::size_t>(i)];
    double e3 = 0;
    for (int a = 0; a < sys.n_at; ++a) {
        const int s0 = sys.atom_sh0[static_cast<std::size_t>(a)];
        const double qa = q_at[static_cast<std::size_t>(a)];
        e3 += (1.0 / 3.0) * sys.sh[static_cast<std::size_t>(s0)].gam3 * qa * qa
              * qa;
    }
    out.energy_eh = e_h0 + e2 + e3;
    out.q = q;
    out.V.assign(static_cast<std::size_t>(sys.n_sh), 0.0);
    for (int i = 0; i < sys.n_sh; ++i) {
        double v = 0;
        for (int j = 0; j < sys.n_sh; ++j)
            v += sys.gamma[static_cast<std::size_t>(i * sys.n_sh + j)]
                 * q[static_cast<std::size_t>(j)];
        const int ia = sys.sh[static_cast<std::size_t>(i)].atom;
        v += sys.sh[static_cast<std::size_t>(i)].gam3
             * q_at[static_cast<std::size_t>(ia)]
             * q_at[static_cast<std::size_t>(ia)];
        out.V[static_cast<std::size_t>(i)] = v;
    }
    return out;
}

double pair_hscale(const IpeaSystem& sys, int i, int j) {
    const Shell& si = sys.sh[static_cast<std::size_t>(i)];
    const Shell& sj = sys.sh[static_cast<std::size_t>(j)];
    if (si.atom == sj.atom)
        return 1.0;
    const auto& Ai = sys.xyz[static_cast<std::size_t>(si.atom)];
    const auto& Bj = sys.xyz[static_cast<std::size_t>(sj.atom)];
    const double dx = Ai[0] - Bj[0];
    const double dy = Ai[1] - Bj[1];
    const double dz = Ai[2] - Bj[2];
    const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
    const double rr = std::sqrt(r / ((si.rad + sj.rad) * kA2Bohr));
    const double pi = 1.0 + si.shpoly * rr;
    const double pj = 1.0 + sj.shpoly * rr;
    const auto& ea = *sys.el[static_cast<std::size_t>(si.atom)];
    const auto& eb = *sys.el[static_cast<std::size_t>(sj.atom)];
    const int ish = i - sys.atom_sh0[static_cast<std::size_t>(si.atom)];
    const int jsh = j - sys.atom_sh0[static_cast<std::size_t>(sj.atom)];
    return hscale_pair(ea, ish, eb, jsh) * pi * pj;
}

double dgamma_ko_dr(double r, double eta_a, double eta_b) {
    if (r < 1e-14)
        return 0;
    const double g = kIpeaGExp;
    const double eta = 2.0 / (1.0 / eta_a + 1.0 / eta_b);
    const double u = std::pow(r, g) + std::pow(eta, -g);
    const double gam = 1.0 / std::pow(u, 1.0 / g);
    return -std::pow(r, g - 1.0) * std::pow(gam, g + 1.0);
}

// dE/dR [Hartree/Bohr] of the IPEA energy at frozen SCC density.
void nuclear_gradient(const IpeaSystem& sys, const SccOut& scc, const int* type,
                      std::vector<std::array<double, 3>>& g) {
    const int n_at = sys.n_at;
    g.assign(static_cast<std::size_t>(n_at), {0.0, 0.0, 0.0});
    const auto& P = scc.P;
    const auto& W = scc.W;
    const auto& q = scc.q;

    std::vector<double> dEdh(static_cast<std::size_t>(sys.n_sh), 0.0);
    for (int i = 0; i < sys.n_sh; ++i) {
        const Shell& si = sys.sh[static_cast<std::size_t>(i)];
        for (int j = 0; j < sys.n_sh; ++j) {
            const Shell& sj = sys.sh[static_cast<std::size_t>(j)];
            const double scale = pair_hscale(sys, i, j);
            for (int ia = 0; ia < si.nao; ++ia)
                for (int ja = 0; ja < sj.nao; ++ja)
                    dEdh[static_cast<std::size_t>(i)] +=
                        P(si.ao0 + ia, sj.ao0 + ja) * scale
                        * sys.S(si.ao0 + ia, sj.ao0 + ja);
        }
    }
    std::vector<double> dEdcn(static_cast<std::size_t>(n_at), 0.0);
    for (int i = 0; i < sys.n_sh; ++i) {
        const Shell& si = sys.sh[static_cast<std::size_t>(i)];
        dEdcn[static_cast<std::size_t>(si.atom)] +=
            dEdh[static_cast<std::size_t>(i)] * (-si.kcn);
    }

    for (int i = 0; i < sys.n_sh; ++i) {
        const Shell& si = sys.sh[static_cast<std::size_t>(i)];
        const int a = si.atom;
        const double* A = sys.xyz[static_cast<std::size_t>(a)].data();
        int pa[3][3] = {};
        for (int ia = 0; ia < si.nao; ++ia)
            cart_pow(si.l, ia, pa);
        for (int j = 0; j < sys.n_sh; ++j) {
            const Shell& sj = sys.sh[static_cast<std::size_t>(j)];
            const int b = sj.atom;
            if (a >= b)
                continue;
            const double* B = sys.xyz[static_cast<std::size_t>(b)].data();
            const double dx = A[0] - B[0];
            const double dy = A[1] - B[1];
            const double dz = A[2] - B[2];
            const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (r < 1e-14)
                continue;
            const double dir[3] = {dx / r, dy / r, dz / r};
            const double radsum = (si.rad + sj.rad) * kA2Bohr;
            const double rr = std::sqrt(r / radsum);
            const double pi = 1.0 + si.shpoly * rr;
            const double pj = 1.0 + sj.shpoly * rr;
            const auto& ea = *sys.el[static_cast<std::size_t>(a)];
            const auto& eb = *sys.el[static_cast<std::size_t>(b)];
            const int ish = i - sys.atom_sh0[static_cast<std::size_t>(a)];
            const int jsh = j - sys.atom_sh0[static_cast<std::size_t>(b)];
            const double hs = hscale_pair(ea, ish, eb, jsh);
            const double scale = hs * pi * pj;
            const double drr_dr = 0.5 * rr / r;
            const double dscale_dr =
                hs * (si.shpoly * pj + sj.shpoly * pi) * drr_dr;
            const double havg = 0.5 * (si.h + sj.h);
            const double vij = 0.5
                               * (scc.V[static_cast<std::size_t>(i)]
                                  + scc.V[static_cast<std::size_t>(j)]);
            const double dgdR = dgamma_ko_dr(r, si.hardness, sj.hardness);
            const double qg = q[static_cast<std::size_t>(i)]
                              * q[static_cast<std::size_t>(j)] * dgdR;

            int pb[3][3] = {};
            for (int ib = 0; ib < sj.nao; ++ib)
                cart_pow(sj.l, ib, pb);
            for (int ia = 0; ia < si.nao; ++ia) {
                for (int ib = 0; ib < sj.nao; ++ib) {
                    double sij = 0;
                    double ds[3] = {0, 0, 0};
                    for (const auto& p : si.prim) {
                        for (const auto& qq : sj.prim) {
                            double sp = 0;
                            double dsp[3];
                            ov3d_grad(pa[ia][0], pa[ia][1], pa[ia][2], p.a, A,
                                      pb[ib][0], pb[ib][1], pb[ib][2], qq.a, B,
                                      sp, dsp);
                            const double c = p.c * qq.c;
                            sij += c * sp;
                            ds[0] += c * dsp[0];
                            ds[1] += c * dsp[1];
                            ds[2] += c * dsp[2];
                        }
                    }
                    const int mu = si.ao0 + ia;
                    const int nu = sj.ao0 + ib;
                    const double Pmn = P(mu, nu);
                    const double Wmn = W(mu, nu);
                    for (int k = 0; k < 3; ++k) {
                        const double dH0 =
                            havg * (dscale_dr * dir[k] * sij + scale * ds[k]);
                        const double dE =
                            2.0 * (Pmn * dH0 + (vij * Pmn - Wmn) * ds[k]);
                        g[static_cast<std::size_t>(a)]
                         [static_cast<std::size_t>(k)] += dE;
                        g[static_cast<std::size_t>(b)]
                         [static_cast<std::size_t>(k)] -= dE;
                    }
                }
            }
            for (int k = 0; k < 3; ++k) {
                const double t = qg * dir[k];
                g[static_cast<std::size_t>(a)][static_cast<std::size_t>(k)] +=
                    t;
                g[static_cast<std::size_t>(b)][static_cast<std::size_t>(k)] -=
                    t;
            }
        }
    }

    for (int a = 0; a < n_at; ++a) {
        const double ra = d3_rcov_bohr(type[a]);
        const auto& A = sys.xyz[static_cast<std::size_t>(a)];
        for (int b = a + 1; b < n_at; ++b) {
            const double rb = d3_rcov_bohr(type[b]);
            const auto& B = sys.xyz[static_cast<std::size_t>(b)];
            const double dx = A[0] - B[0];
            const double dy = A[1] - B[1];
            const double dz = A[2] - B[2];
            const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (r < 1e-14)
                continue;
            const double dir[3] = {dx / r, dy / r, dz / r};
            const double rcov = ra + rb;
            const double f = d3_cn(r, rcov);
            const double dfdr = f * (1.0 - f) * (-kIpeaD3k1 * rcov / (r * r));
            const double chain = (dEdcn[static_cast<std::size_t>(a)]
                                  + dEdcn[static_cast<std::size_t>(b)])
                                 * dfdr;
            for (int k = 0; k < 3; ++k) {
                const double t = chain * dir[k];
                g[static_cast<std::size_t>(a)][static_cast<std::size_t>(k)] +=
                    t;
                g[static_cast<std::size_t>(b)][static_cast<std::size_t>(k)] -=
                    t;
            }

            const auto& ea = *sys.el[static_cast<std::size_t>(a)];
            const auto& eb = *sys.el[static_cast<std::size_t>(b)];
            const double kpre = std::sqrt(ea.alpha * eb.alpha);
            const double kexp = kpre * std::pow(r, kIpeaRepKExp);
            const double epair = ea.zeff * eb.zeff / r * std::exp(-kexp);
            const double dEdr =
                epair
                * (-1.0 / r
                   - kIpeaRepKExp * kpre * std::pow(r, kIpeaRepKExp - 1.0));
            for (int k = 0; k < 3; ++k) {
                const double t = dEdr * dir[k];
                g[static_cast<std::size_t>(a)][static_cast<std::size_t>(k)] +=
                    t;
                g[static_cast<std::size_t>(b)][static_cast<std::size_t>(k)] -=
                    t;
            }
        }
    }

    std::vector<int> Z(static_cast<std::size_t>(n_at));
    std::vector<std::array<double, 3>> pos(static_cast<std::size_t>(n_at));
    for (int a = 0; a < n_at; ++a) {
        Z[static_cast<std::size_t>(a)] = type[a];
        pos[static_cast<std::size_t>(a)] = sys.xyz[static_cast<std::size_t>(a)];
    }
    const double cell[3] = {0, 0, 0};
    const auto d3 =
        gpw::dftd3_pbe_bj(Z, pos, cell, false, gpw::D3Params{}, 94.2, kIpeaD3s6,
                          kIpeaD3s8, kIpeaD3a1, kIpeaD3a2);
    for (int a = 0; a < n_at; ++a) {
        g[static_cast<std::size_t>(a)][0] -=
            d3.forces[static_cast<std::size_t>(a)][0];
        g[static_cast<std::size_t>(a)][1] -=
            d3.forces[static_cast<std::size_t>(a)][1];
        g[static_cast<std::size_t>(a)][2] -=
            d3.forces[static_cast<std::size_t>(a)][2];
    }
}

struct TotalOut {
    double energy_ev = 0;
    double residual = 0;
    int iterations = 0;
    std::vector<std::array<double, 3>> grad_ha_bohr;
};

TotalOut total_energy_eV(const StorageReal* x, const StorageReal* y,
                         const StorageReal* z, const int* type, int n_at,
                         int charge, double scc_tol, int max_scc,
                         double mix_beta, bool want_grad, double kT_ha) {
    const IpeaSystem sys = build_system(x, y, z, type, n_at, charge);
    const SccOut scc = scc_energy(sys, scc_tol, max_scc, mix_beta, kT_ha);
    const double e_rep = repulsive_e(sys);
    const double e_disp = dispersion_e(sys, type);
    TotalOut out;
    out.energy_ev = (scc.energy_eh + e_rep + e_disp) * kHa2eV;
    out.residual = scc.residual;
    out.iterations = scc.iterations;
    if (want_grad)
        nuclear_gradient(sys, scc, type, out.grad_ha_bohr);
    return out;
}

} // namespace

void IPEA::build(const BuildContext&) {}

void IPEA::launch(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    const int n_at = f.nlocal();
    if (n_at < 1)
        throw std::invalid_argument("IPEA: empty frame");
    if (!(mix_beta > 0.0) || mix_beta > 1.0)
        throw std::invalid_argument("IPEA: mix_beta must be in (0, 1]");
    if (multiplicity != 1)
        throw std::runtime_error(
            "IPEA: open shell (multiplicity!=1) is not a product path; "
            "closed-shell only (see queue spin-gfn-xtb). VIP/VEA start from "
            "a closed-shell N-electron state (set compute_vip/compute_vea)");
    const auto type = f.span(kAtom_type);
    const StorageReal* x = f.span(kAtom_x);
    const StorageReal* y = f.span(kAtom_y);
    const StorageReal* z = f.span(kAtom_z);
    AccReal* fx = f.span(kAtom_fx);
    AccReal* fy = f.span(kAtom_fy);
    AccReal* fz = f.span(kAtom_fz);

    const bool do_f = with_forces && ctx.forces_requested;
    const TotalOut out = total_energy_eV(x, y, z, type, n_at, charge, scc_tol,
                                         max_scc, mix_beta, do_f, 0.0);
    energy_ = out.energy_ev;
    scc_residual_ = out.residual;
    scc_iterations_ = out.iterations;
    vip_ = 0;
    vea_ = 0;
    f.meta.get(kEnergy) += energy_;
    f.meta.get(kEnergyElectronic) += energy_;

    if (compute_vip || compute_vea) {
        // N-electron must be even; ionized ΔSCC shares degenerate shells.
        const IpeaSystem sys0 = build_system(x, y, z, type, n_at, charge);
        if (sys0.n_elec % 2 != 0)
            throw std::runtime_error(
                "IPEA: VIP/VEA require a closed-shell N-electron state "
                "(even valence count); odd ground state is spin-gfn-xtb");
        const double kT_ha =
            units::kB_ev_per_k * kIpeaEtempK / units::kHartreeInEv;
        if (compute_vip) {
            const TotalOut cat =
                total_energy_eV(x, y, z, type, n_at, charge + 1, scc_tol,
                                max_scc, mix_beta, false, kT_ha);
            vip_ = (cat.energy_ev - energy_) - kIpeaShiftEv;
        }
        if (compute_vea) {
            const TotalOut an =
                total_energy_eV(x, y, z, type, n_at, charge - 1, scc_tol,
                                max_scc, mix_beta, false, kT_ha);
            vea_ = (energy_ - an.energy_ev) - kIpeaShiftEv;
        }
    }

    if (do_f) {
        const double fac = units::kHartreePerBohrInEvPerAngstrom;
        for (int a = 0; a < n_at; ++a) {
            fx[a] += static_cast<AccReal>(
                -out.grad_ha_bohr[static_cast<std::size_t>(a)][0] * fac);
            fy[a] += static_cast<AccReal>(
                -out.grad_ha_bohr[static_cast<std::size_t>(a)][1] * fac);
            fz[a] += static_cast<AccReal>(
                -out.grad_ha_bohr[static_cast<std::size_t>(a)][2] * fac);
        }
    }
}

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
