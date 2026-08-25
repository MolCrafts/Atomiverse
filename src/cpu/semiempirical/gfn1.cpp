/// @file src/cpu/semiempirical/gfn1.cpp
/// GFN1-xTB energy (organic H/C/N/O): Stewart overlap, EHT+SCC, rep, D3(BJ).
#include "atomiverse/cpu/semiempirical/gfn1.h"

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
#include "atomiverse/cpu/semiempirical/basis_data/gfn1_tables.h"
#include "atomiverse/cpu/semiempirical/internal/scc_mixer.h"

namespace ATV {
namespace cpu {
namespace semiempirical {
namespace {

using basis_data::Gfn1Element;
using basis_data::kGfn1D3a1;
using basis_data::kGfn1D3a2;
using basis_data::kGfn1D3k1;
using basis_data::kGfn1D3s6;
using basis_data::kGfn1D3s8;
using basis_data::kGfn1Elements;
using basis_data::kGfn1GExp;
using basis_data::kGfn1Kdiff;
using basis_data::kGfn1KEn;
using basis_data::kGfn1KHH;
using basis_data::kGfn1KNH;
using basis_data::kGfn1Kp;
using basis_data::kGfn1Ks;
using basis_data::kGfn1Ksp;
using basis_data::kGfn1NElements;
using basis_data::kGfn1RepKExp;

constexpr double kPi = 3.14159265358979323846;
constexpr double kA2Bohr = units::kAngstromInBohr;
constexpr double kHa2eV = units::kHartreeInEv;

const Gfn1Element& element_row(int z) {
    for (std::size_t i = 0; i < kGfn1NElements; ++i)
        if (kGfn1Elements[i].z == z)
            return kGfn1Elements[i];
    throw std::runtime_error(
        "GFN1: unsupported element Z=" + std::to_string(z)
        + " (static organic tables cover H/C/N/O only; add a row to "
          "basis_data/gfn1_tables.h)");
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
        throw std::runtime_error("GFN1: no D3 rcov for Z=" + std::to_string(z));
    }
}

// tblite exp CN uses mctc get_covalent_rad = (4/3) Pyykkö, k=16.
double d3_rcov_bohr(int z) { return (4.0 / 3.0) * pyykko_rcov_A(z) * kA2Bohr; }

double kshell(int la, int lb) {
    if ((la == 0 && lb == 1) || (la == 1 && lb == 0))
        return kGfn1Ksp;
    const double ka = (la == 0) ? kGfn1Ks : kGfn1Kp;
    const double kb = (lb == 0) ? kGfn1Ks : kGfn1Kp;
    return 0.5 * (ka + kb);
}

double hscale_pair(const Gfn1Element& ea, int ish, const Gfn1Element& eb,
                   int jsh) {
    const bool va = ea.valence[ish];
    const bool vb = eb.valence[jsh];
    const int la = ea.ang[ish];
    const int lb = eb.ang[jsh];
    if (va && vb) {
        const double den = (ea.en - eb.en) * (ea.en - eb.en);
        double kpair = 1.0;
        if (ea.z == 1 && eb.z == 1)
            kpair = kGfn1KHH;
        else if ((ea.z == 1 && eb.z == 7) || (ea.z == 7 && eb.z == 1))
            kpair = kGfn1KNH;
        const double km = kpair * kshell(la, lb);
        return km * (1.0 + kGfn1KEn * den);
    }
    if (va && !vb)
        return 0.5 * (kshell(la, la) + kGfn1Kdiff);
    if (!va && vb)
        return 0.5 * (kshell(lb, lb) + kGfn1Kdiff);
    return kGfn1Kdiff;
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
    } else if (ng == 6 && n == 2 && l == 0) {
        alpha = {2.768496241e+1, 5.077140627, 1.426786050, 2.040335729e-1,
                 9.260298399e-2, 4.416183978e-2};
        coeff = {-4.151277819e-3, -2.067024148e-2, -5.150303337e-2,
                 3.346271174e-1, 5.621061301e-1, 1.712994697e-1};
    } else if (ng == 6 && n == 2 && l == 1) {
        alpha = {5.868285913, 1.530329631, 5.475665231e-1, 2.288932733e-1,
                 1.046655969e-1, 4.948220127e-2};
        coeff = {7.924233646e-3, 5.144104825e-2, 1.898400060e-1, 4.049863191e-1,
                 4.012362861e-1, 1.051855189e-1};
    } else {
        throw std::runtime_error("GFN1: no Stewart STO-nG for n="
                                 + std::to_string(n) + " l=" + std::to_string(l)
                                 + " ng=" + std::to_string(ng));
    }
    const double z2 = zeta * zeta;
    for (int i = 0; i < ng; ++i) {
        alpha[static_cast<std::size_t>(i)] *= z2;
        const double a = alpha[static_cast<std::size_t>(i)];
        // Primitive Cartesian norm: (2α/π)^{3/4} (4α)^{l/2} / sqrt((2l-1)!!).
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
    double h = 0;       ///< CN-shifted onsite [Hartree].
    double shpoly = 0;  ///< k_poly.
    double hardness = 0;///< η_l [Hartree].
    double n0 = 0;
    double gam3 = 0;    ///< Atomic Γ (same on every shell of the atom).
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

void cart_pow(int l, int m, int pows[3][3]) {
    // m = 0,1,2 → px, py, pz for l=1; ignored for l=0.
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
        throw std::runtime_error("GFN1: non-positive shell self-overlap");
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

void fill_shell_prims(Shell& sh, const Gfn1Element& el, int ish,
                      const Shell* ortho_against) {
    std::vector<double> alpha, coeff;
    sto_ng(el.nprim[ish], el.pqn[ish], el.ang[ish], el.zeta[ish], alpha, coeff);
    sh.prim.resize(alpha.size());
    for (std::size_t i = 0; i < alpha.size(); ++i)
        sh.prim[i] = {alpha[i], coeff[i]};
    // tblite leaves Stewart+primitive-norm contractions as-is; only the
    // Gram–Schmidt 2s (H polarization) is re-normalized.
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
    // Harmonic-mean hardness; Klopman–Ohno with gexp = 2.
    const double eta = 2.0 / (1.0 / eta_a + 1.0 / eta_b);
    const double g = kGfn1GExp;
    return 1.0 / std::pow(std::pow(r, g) + std::pow(eta, -g), 1.0 / g);
}

double d3_cn(double r, double rcovab) {
    if (r < 1e-14)
        return 0;
    const double ee = std::exp(-kGfn1D3k1 * (rcovab / r - 1.0));
    return 1.0 / (1.0 + ee);
}

struct Gfn1System {
    int n_at = 0;
    int n_ao = 0;
    int n_sh = 0;
    int n_occ = 0;
    std::vector<const Gfn1Element*> el;
    std::vector<std::array<double, 3>> xyz; // Bohr
    std::vector<double> cn;
    std::vector<Shell> sh;
    std::vector<int> atom_sh0; // first shell of atom
    std::vector<int> atom_nsh;
    linalg::RealMatrix S;
    linalg::RealMatrix H0;
    std::vector<double> gamma; // n_sh * n_sh
};

Gfn1System build_system(const StorageReal* x, const StorageReal* y,
                        const StorageReal* z, const int* type, int n_at,
                        int charge) {
    Gfn1System sys;
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
    if (n_elec < 0.5 || std::fabs(n_elec - std::round(n_elec)) > 1e-9
        || static_cast<int>(std::lround(n_elec)) % 2 != 0)
        throw std::runtime_error(
            "GFN1: odd/non-integer valence electron count not supported "
            "(closed-shell energy foothold; open shell is spin-gfn-xtb)");
    sys.n_occ = static_cast<int>(std::lround(n_elec)) / 2;

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
        sys.atom_sh0[static_cast<std::size_t>(a)] = static_cast<int>(sys.sh.size());
        const Shell* s_val = nullptr;
        for (int ish = 0; ish < ea.n_shell; ++ish) {
            Shell sh;
            sh.atom = a;
            sh.l = ea.ang[ish];
            sh.nao = (sh.l == 0) ? 1 : 3;
            sh.ao0 = ao;
            sh.valence = ea.valence[ish];
            sh.h = ea.level[ish]
                   - ea.kcn[ish] * sys.cn[static_cast<std::size_t>(a)];
            sh.shpoly = ea.shpoly[ish];
            sh.hardness = ea.gam * ea.lgam[ish];
            sh.n0 = ea.refocc[ish];
            sh.gam3 = ea.gam3;
            sh.rad = ea.rad;
            fill_shell_prims(sh, ea, ish, (sh.l == 0 && !sh.valence) ? s_val
                                                                    : nullptr);
            if (sh.l == 0 && sh.valence)
                s_val = nullptr; // set after push
            ao += sh.nao;
            sys.sh.push_back(std::move(sh));
            if (sys.sh.back().l == 0 && sys.sh.back().valence)
                s_val = &sys.sh.back();
            ++sys.atom_nsh[static_cast<std::size_t>(a)];
        }
        (void)s_val;
    }
    sys.atom_sh0[static_cast<std::size_t>(n_at)] = static_cast<int>(sys.sh.size());
    sys.n_sh = static_cast<int>(sys.sh.size());
    sys.n_ao = ao;

    // Polarization 2s must be orthogonalized against the atom's valence s.
    // fill_shell_prims saw a pointer into a vector that may reallocate — redo.
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
                int ish = i;
                fill_shell_prims(sh, ea, ish, val_s);
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
            const double* Bj = sys.xyz[static_cast<std::size_t>(sj.atom)].data();
            double hij = 0.5 * (si.h + sj.h);
            if (si.atom != sj.atom) {
                const double dx = Ai[0] - Bj[0];
                const double dy = Ai[1] - Bj[1];
                const double dz = Ai[2] - Bj[2];
                const double r2 = dx * dx + dy * dy + dz * dz;
                const double r = std::sqrt(r2);
                // tblite get_hamiltonian: rr = sqrt(rab / (rad_i+rad_j))
                // with Pyykkö radii in Bohr (get_atomic_rad).
                const double rr =
                    std::sqrt(r / ((si.rad + sj.rad) * kA2Bohr));
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

double repulsive_e(const Gfn1System& sys) {
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
            const double kexp = std::sqrt(ea.alpha * eb.alpha)
                                * std::pow(r, kGfn1RepKExp);
            e += ea.zeff * eb.zeff / r * std::exp(-kexp);
        }
    }
    return e;
}

double dispersion_e(const Gfn1System& sys, const int* type) {
    std::vector<int> Z(static_cast<std::size_t>(sys.n_at));
    std::vector<std::array<double, 3>> pos(static_cast<std::size_t>(sys.n_at));
    for (int a = 0; a < sys.n_at; ++a) {
        Z[static_cast<std::size_t>(a)] = type[a];
        pos[static_cast<std::size_t>(a)] = sys.xyz[static_cast<std::size_t>(a)];
    }
    const double cell[3] = {0, 0, 0};
    const auto d3 =
        gpw::dftd3_pbe_bj(Z, pos, cell, false, gpw::D3Params{}, 94.2,
                          kGfn1D3s6, kGfn1D3s8, kGfn1D3a1, kGfn1D3a2);
    return d3.energy;
}

struct SccOut {
    double energy_eh = 0;
    double residual = 0;
    int iterations = 0;
};

SccOut scc_energy(const Gfn1System& sys, double scc_tol, int max_scc,
                  double mix_beta) {
    std::vector<double> q(static_cast<std::size_t>(sys.n_sh), 0.0);
    internal::ChargeMixer mixer(mix_beta, 6);
    bool converged = false;
    SccOut out;
    linalg::RealMatrix H = sys.H0;
    std::pair<linalg::RealVector, linalg::RealMatrix> sol;
    for (int it = 0; it < max_scc; ++it) {
        std::vector<double> q_at(static_cast<std::size_t>(sys.n_at), 0.0);
        for (int i = 0; i < sys.n_sh; ++i)
            q_at[static_cast<std::size_t>(sys.sh[static_cast<std::size_t>(i)].atom)] +=
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
                const double vij =
                    0.5
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
        linalg::RealMatrix P = xt::zeros<double>({sys.n_ao, sys.n_ao});
        for (int o = 0; o < sys.n_occ; ++o)
            for (int mu = 0; mu < sys.n_ao; ++mu)
                for (int nu = 0; nu < sys.n_ao; ++nu)
                    P(mu, nu) += 2.0 * C(mu, o) * C(nu, o);
        std::vector<double> q_out(static_cast<std::size_t>(sys.n_sh), 0.0);
        for (int i = 0; i < sys.n_sh; ++i) {
            const Shell& si = sys.sh[static_cast<std::size_t>(i)];
            double pop = 0;
            for (int ia = 0; ia < si.nao; ++ia)
                for (int nu = 0; nu < sys.n_ao; ++nu)
                    pop += P(si.ao0 + ia, nu) * sys.S(si.ao0 + ia, nu);
            // Electron excess Δn = pop − n0 (DFTB2 / GFN2 / tblite SCC).
            q_out[static_cast<std::size_t>(i)] = pop - si.n0;
        }
        out.residual = 0;
        for (int i = 0; i < sys.n_sh; ++i)
            out.residual = std::max(
                out.residual,
                std::fabs(q_out[static_cast<std::size_t>(i)]
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
            "GFN1: SCC did not converge in " + std::to_string(max_scc)
            + " iterations (residual " + std::to_string(out.residual)
            + " e > scc_tol " + std::to_string(scc_tol)
            + "); raise max_scc or lower mix_beta");

    // Rebuild P from the last diagonalization (q already mixed to q_out
    // on the converged iteration).
    const auto& C = sol.second;
    linalg::RealMatrix P = xt::zeros<double>({sys.n_ao, sys.n_ao});
    for (int o = 0; o < sys.n_occ; ++o)
        for (int mu = 0; mu < sys.n_ao; ++mu)
            for (int nu = 0; nu < sys.n_ao; ++nu)
                P(mu, nu) += 2.0 * C(mu, o) * C(nu, o);

    double e_h0 = 0;
    for (int mu = 0; mu < sys.n_ao; ++mu)
        for (int nu = 0; nu < sys.n_ao; ++nu)
            e_h0 += P(mu, nu) * sys.H0(mu, nu);
    double e2 = 0;
    for (int i = 0; i < sys.n_sh; ++i)
        for (int j = 0; j < sys.n_sh; ++j)
            e2 += 0.5 * q[static_cast<std::size_t>(i)]
                  * sys.gamma[static_cast<std::size_t>(i * sys.n_sh + j)]
                  * q[static_cast<std::size_t>(j)];
    std::vector<double> q_at(static_cast<std::size_t>(sys.n_at), 0.0);
    for (int i = 0; i < sys.n_sh; ++i)
        q_at[static_cast<std::size_t>(sys.sh[static_cast<std::size_t>(i)].atom)] +=
            q[static_cast<std::size_t>(i)];
    double e3 = 0;
    for (int a = 0; a < sys.n_at; ++a) {
        const int s0 = sys.atom_sh0[static_cast<std::size_t>(a)];
        const double qa = q_at[static_cast<std::size_t>(a)];
        e3 += (1.0 / 3.0) * sys.sh[static_cast<std::size_t>(s0)].gam3 * qa * qa
              * qa;
    }
    out.energy_eh = e_h0 + e2 + e3;
    return out;
}

struct TotalOut {
    double energy_ev = 0;
    double residual = 0;
    int iterations = 0;
};

TotalOut total_energy_eV(const StorageReal* x, const StorageReal* y,
                         const StorageReal* z, const int* type, int n_at,
                         int charge, double scc_tol, int max_scc,
                         double mix_beta) {
    const Gfn1System sys = build_system(x, y, z, type, n_at, charge);
    const SccOut scc = scc_energy(sys, scc_tol, max_scc, mix_beta);
    const double e_rep = repulsive_e(sys);
    const double e_disp = dispersion_e(sys, type);
    TotalOut out;
    out.energy_ev = (scc.energy_eh + e_rep + e_disp) * kHa2eV;
    out.residual = scc.residual;
    out.iterations = scc.iterations;
    return out;
}

} // namespace

void GFN1::build(const BuildContext&) {}

void GFN1::launch(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    const int n_at = f.nlocal();
    if (n_at < 1)
        throw std::invalid_argument("GFN1: empty frame");
    if (!(mix_beta > 0.0) || mix_beta > 1.0)
        throw std::invalid_argument("GFN1: mix_beta must be in (0, 1]");
    if (multiplicity != 1)
        throw std::runtime_error(
            "GFN1: open shell (multiplicity!=1) is not a product path; "
            "closed-shell energy only (see queue spin-gfn-xtb)");
    if (with_forces)
        throw std::runtime_error(
            "GFN1: nuclear forces are not implemented on this energy "
            "foothold; set with_forces=false (later slug: GFN-xTB analytic "
            "gradient)");

    const auto type = f.span(kAtom_type);
    const StorageReal* x = f.span(kAtom_x);
    const StorageReal* y = f.span(kAtom_y);
    const StorageReal* z = f.span(kAtom_z);

    const TotalOut out = total_energy_eV(x, y, z, type, n_at, charge, scc_tol,
                                         max_scc, mix_beta);
    energy_ = out.energy_ev;
    scc_residual_ = out.residual;
    scc_iterations_ = out.iterations;
    f.meta.get(kEnergy) += energy_;
    f.meta.get(kEnergyElectronic) += energy_;
}

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
