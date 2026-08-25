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

// Overlap and dipole (origin 0) between two contracted Cartesian shells.
void shell_ints(const Shell& si, const double Ai[3], const Shell& sj,
                const double Aj[3], int iao, int jao, double& S, double D[3]) {
    S = 0;
    D[0] = D[1] = D[2] = 0;
    const int li = si.l;
    const int lj = sj.l;
    const int ci = (li == 0) ? 0 : iao;
    const int cj = (lj == 0) ? 0 : jao;
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

struct SolveOut {
    double energy_eh = 0;
    double residual = 0;
    int iterations = 0;
    std::vector<double> dq;
};

SolveOut solve_gfn2(const Frame& f, int charge, double scc_tol, int max_scc,
                    double mix_beta) {
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
    auto D = [&](int k, int mu, int nu) -> double& {
        return dip[static_cast<std::size_t>((k * n_ao + mu) * n_ao + nu)];
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
                    double s = 0, d[3];
                    shell_ints(si, Ai, sj, Aj, u, v, s, d);
                    const int mu = si.ao0 + u;
                    const int nu = sj.ao0 + v;
                    S(mu, nu) = s;
                    H0(mu, nu) = s * hij;
                    for (int k = 0; k < 3; ++k)
                        D(k, mu, nu) = d[k];
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
    internal::ChargeMixer mixer(mix_beta, 6);
    linalg::RealMatrix P = xt::zeros<double>({n_ao, n_ao});
    SolveOut out;
    bool converged = false;

    auto aes_energy = [&]() {
        double e = 0;
        for (int a = 0; a < n_at; ++a) {
            const auto& ea = *el[static_cast<std::size_t>(a)];
            const double* ra = xyz[static_cast<std::size_t>(a)].data();
            // AXC
            const auto& mu = dipat[static_cast<std::size_t>(a)];
            e += ea.dkernel * (mu[0] * mu[0] + mu[1] * mu[1] + mu[2] * mu[2]);
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
                const double qa = -qat[static_cast<std::size_t>(a)];
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
            }
        }
        return e;
    };

    for (int it = 0; it < max_scc; ++it) {
        std::vector<double> vsh(static_cast<std::size_t>(n_sh), 0.0);
        for (int ish = 0; ish < n_sh; ++ish) {
            double s = 0;
            for (int jsh = 0; jsh < n_sh; ++jsh)
                s += gamma[static_cast<std::size_t>(ish) * n_sh
                           + static_cast<std::size_t>(jsh)]
                     * qsh[static_cast<std::size_t>(jsh)];
            s += gam3[static_cast<std::size_t>(ish)]
                 * qsh[static_cast<std::size_t>(ish)]
                 * qsh[static_cast<std::size_t>(ish)];
            vsh[static_cast<std::size_t>(ish)] = s;
        }
        linalg::RealMatrix H = H0;
        for (int mu = 0; mu < n_ao; ++mu) {
            const int ish = ao_shell[static_cast<std::size_t>(mu)];
            for (int nu = 0; nu < n_ao; ++nu) {
                const int jsh = ao_shell[static_cast<std::size_t>(nu)];
                H(mu, nu) += 0.5 * S(mu, nu)
                             * (vsh[static_cast<std::size_t>(ish)]
                                + vsh[static_cast<std::size_t>(jsh)]);
            }
        }
        auto sol = linalg::diagonalize_fock_generalized(H, S);
        const auto& C = sol.second;
        P = xt::zeros<double>({n_ao, n_ao});
        for (int o = 0; o < n_occ; ++o)
            for (int mu = 0; mu < n_ao; ++mu)
                for (int nu = 0; nu < n_ao; ++nu)
                    P(mu, nu) += 2.0 * C(mu, o) * C(nu, o);

        const linalg::RealMatrix PS = xt::linalg::dot(P, S);
        for (int a = 0; a < n_at; ++a) {
            qat[static_cast<std::size_t>(a)] = 0;
            dipat[static_cast<std::size_t>(a)] = {0, 0, 0};
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
            // Mulliken: bra on A, ket on any centre (tblite dpint contraction).
            for (int u = 0; u < na; ++u) {
                for (int b2 = 0; b2 < n_at; ++b2) {
                    const int ib = ao0[static_cast<std::size_t>(b2)];
                    const int nb = n_ao_of(*el[static_cast<std::size_t>(b2)]);
                    for (int v = 0; v < nb; ++v) {
                        const double p = P(ia + u, ib + v);
                        for (int k = 0; k < 3; ++k)
                            dipat[static_cast<std::size_t>(a)]
                                 [static_cast<std::size_t>(k)] +=
                                0.5 * p
                                * (D(k, ia + u, ib + v)
                                   - Ra[k] * S(ia + u, ib + v));
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
    const double e_d4 = d4_energy();
    const double e_aes = aes_energy();

    out.energy_eh = e_h0 + e_ies + e3 + e_rep + e_d4 + e_aes;
    out.dq.assign(static_cast<std::size_t>(n_at), 0.0);
    for (int a = 0; a < n_at; ++a)
        out.dq[static_cast<std::size_t>(a)] = -qat[static_cast<std::size_t>(a)];
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
    if (with_forces && ctx.forces_requested)
        throw std::runtime_error(
            "GFN2: nuclear forces are not this slug — see queue "
            "gfn2-xtb-forces (energy-only foothold)");

    const auto out = solve_gfn2(f, charge, scc_tol, max_scc, mix_beta);
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
}

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
