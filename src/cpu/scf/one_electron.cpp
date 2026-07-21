/// @file cpu/scf/one_electron.cpp
/// @brief Native-engine implementations of the one-electron integral kernels
///        (IntegralPack private static members) + nuclear repulsion energy.

#include "atomiverse/cpu/scf/internal/one_electron.h"

#include <cmath>

#include "atomiverse/cpu/scf/cart.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/internal/shell_pair.h"
#include "atomiverse/cpu/scf/native_integrals.h"

namespace ATV {
namespace cpu {
namespace scf {

namespace {

/// Max buffer needed for a Cartesian (AB) block of 1e integrals: the
/// largest case is d×d = 6×6 = 36 doubles.
constexpr int kBlockBuf1e = kMaxNCart * kMaxNCart;

/// @brief Scatter a block's result into a symmetric matrix.
///
/// `buf` is the dense A.n_cart × B.n_cart output from a block kernel.
/// For i != j block pairs we also populate the transposed entries; for
/// diagonal block pairs the kernel already produces the whole block.
inline void scatter_block(xt::xtensor<double, 2>& M,
                          const native::ShellBlock& A,
                          const native::ShellBlock& B, const double* buf,
                          bool diag) {
    const int na = A.n_cart, nb = B.n_cart;
    if (diag) {
        for (int ia = 0; ia < na; ++ia) {
            for (int ib = 0; ib < nb; ++ib) {
                M(A.ao_start + ia, B.ao_start + ib) = buf[ia * nb + ib];
            }
        }
    } else {
        for (int ia = 0; ia < na; ++ia) {
            const int i = A.ao_start + ia;
            for (int ib = 0; ib < nb; ++ib) {
                const int j = B.ao_start + ib;
                const double v = buf[ia * nb + ib];
                M(i, j) = v;
                M(j, i) = v;
            }
        }
    }
}

/// @brief Canonical block-pair loop: apply `kernel(A, B, buf)` to every
///        lower-triangular (i ≥ j) block pair and scatter into `M`.
template <typename Kernel>
void for_each_canonical_block_pair(
    xt::xtensor<double, 2>& M, const std::vector<native::ShellBlock>& blocks,
    Kernel&& kernel) {
    const int nb = static_cast<int>(blocks.size());
    double buf[kBlockBuf1e];
    for (int i = 0; i < nb; ++i) {
        for (int j = 0; j <= i; ++j) {
            const auto& A = blocks[static_cast<std::size_t>(i)];
            const auto& B = blocks[static_cast<std::size_t>(j)];
            const int nab = A.n_cart * B.n_cart;
            for (int k = 0; k < nab; ++k)
                buf[k] = 0.0;
            kernel(A, B, buf);
            scatter_block(M, A, B, buf, /*diag=*/(i == j));
        }
    }
}

} // anonymous namespace

xt::xtensor<double, 2> IntegralPack::overlap(const ShellList& shells) {
    const int n = shells.n_basis;
    xt::xtensor<double, 2> S = xt::zeros<double>({n, n});
    const auto blocks = blocks_from_shells(shells.shells);
    for_each_canonical_block_pair(
        S, blocks,
        [](const native::ShellBlock& A, const native::ShellBlock& B,
           double* buf) { native::overlap_block(A, B, buf); });
    return S;
}

xt::xtensor<double, 2> IntegralPack::kinetic(const ShellList& shells) {
    const int n = shells.n_basis;
    xt::xtensor<double, 2> T = xt::zeros<double>({n, n});
    const auto blocks = blocks_from_shells(shells.shells);
    for_each_canonical_block_pair(
        T, blocks,
        [](const native::ShellBlock& A, const native::ShellBlock& B,
           double* buf) { native::kinetic_block(A, B, buf); });
    return T;
}

xt::xtensor<double, 2>
IntegralPack::nuclear_attraction(const ShellList& shells,
                                 const AtomList& atoms) {
    const int n = shells.n_basis;
    const int natm = static_cast<int>(atoms.atoms.size());
    xt::xtensor<double, 2> V = xt::zeros<double>({n, n});
    const auto blocks = blocks_from_shells(shells.shells);
    for_each_canonical_block_pair(
        V, blocks,
        [&](const native::ShellBlock& A, const native::ShellBlock& B,
            double* buf) {
            native::nuclear_block(A, B, atoms.atoms.data(), natm, buf);
        });
    return V;
}

namespace {

/// @brief ⟨∂g_a/∂x_d | 1/r_C | ∂g_b/∂x_d⟩ for one Cartesian direction via
///        the Gaussian derivative identity (without −Z).
///
/// ∂_d g(a,α) = 2α g(a+e_d) − a_d g(a−e_d). Expanding the product yields
/// four nuclear-attraction primitives of raised/lowered angular momenta.
inline double pvp_prim_dir(const int a_ang[3], const int b_ang[3], double aa,
                           double ab, double gamma, const double PA[3],
                           const double PB[3], const double PC[3], double K_AB,
                           int d) {
    int a_plus[3] = {a_ang[0], a_ang[1], a_ang[2]};
    int b_plus[3] = {b_ang[0], b_ang[1], b_ang[2]};
    a_plus[d] += 1;
    b_plus[d] += 1;

    // (2α_a)(2α_b) ⟨a+e|1/r|b+e⟩
    double acc = 4.0 * aa * ab
                 * native::nuclear_prim(a_plus, b_plus, gamma, PA, PB, PC, K_AB);

    if (b_ang[d] > 0) {
        int b_minus[3] = {b_ang[0], b_ang[1], b_ang[2]};
        b_minus[d] -= 1;
        // − (2α_a) b_d ⟨a+e|1/r|b−e⟩
        acc -= 2.0 * aa * static_cast<double>(b_ang[d])
               * native::nuclear_prim(a_plus, b_minus, gamma, PA, PB, PC, K_AB);
    }
    if (a_ang[d] > 0) {
        int a_minus[3] = {a_ang[0], a_ang[1], a_ang[2]};
        a_minus[d] -= 1;
        // − a_d (2α_b) ⟨a−e|1/r|b+e⟩
        acc -= static_cast<double>(a_ang[d]) * 2.0 * ab
               * native::nuclear_prim(a_minus, b_plus, gamma, PA, PB, PC, K_AB);
        if (b_ang[d] > 0) {
            int b_minus[3] = {b_ang[0], b_ang[1], b_ang[2]};
            b_minus[d] -= 1;
            // + a_d b_d ⟨a−e|1/r|b−e⟩
            acc += static_cast<double>(a_ang[d] * b_ang[d])
                   * native::nuclear_prim(a_minus, b_minus, gamma, PA, PB, PC,
                                          K_AB);
        }
    }
    return acc;
}

/// @brief Cartesian pVp shell-block: Σ_d ⟨∂A|V|∂B⟩ with V = −Σ Z/r.
void pvp_block(const native::ShellBlock& A, const native::ShellBlock& B,
               const native::Atom* atoms, int natm, double* out) {
    int a_ang[kMaxNCart][3], b_ang[kMaxNCart][3];
    fill_cart_comps(A.l, a_ang);
    fill_cart_comps(B.l, b_ang);

    for (int p = 0; p < A.n_prim; ++p) {
        for (int q = 0; q < B.n_prim; ++q) {
            const double aa = A.alpha[p];
            const double ab = B.alpha[q];
            const double gamma = aa + ab;
            const double mu = aa * ab / gamma;
            const double R2_AB = (A.center[0] - B.center[0])
                                     * (A.center[0] - B.center[0])
                                 + (A.center[1] - B.center[1])
                                       * (A.center[1] - B.center[1])
                                 + (A.center[2] - B.center[2])
                                       * (A.center[2] - B.center[2]);
            const double K_AB = std::exp(-mu * R2_AB);
            const double cpq = A.coeff[p] * B.coeff[q];

            double P_c[3], PA[3], PB[3];
            for (int d = 0; d < 3; ++d) {
                P_c[d] = (aa * A.center[d] + ab * B.center[d]) / gamma;
                PA[d] = P_c[d] - A.center[d];
                PB[d] = P_c[d] - B.center[d];
            }

            for (int c = 0; c < natm; ++c) {
                if (atoms[c].Z == 0)
                    continue;
                double PC[3];
                for (int d = 0; d < 3; ++d)
                    PC[d] = P_c[d] - atoms[c].pos[d];
                const double mz = -static_cast<double>(atoms[c].Z);

                for (int ia = 0; ia < A.n_cart; ++ia) {
                    for (int ib = 0; ib < B.n_cart; ++ib) {
                        double w_ab = 0.0;
                        for (int d = 0; d < 3; ++d)
                            w_ab += pvp_prim_dir(a_ang[ia], b_ang[ib], aa, ab,
                                                 gamma, PA, PB, PC, K_AB, d);
                        out[ia * B.n_cart + ib] += cpq * mz * w_ab;
                    }
                }
            }
        }
    }
}

} // namespace

xt::xtensor<double, 2> IntegralPack::pvp(const ShellList& shells,
                                         const AtomList& atoms) {
    const int n = shells.n_basis;
    const int natm = static_cast<int>(atoms.atoms.size());
    xt::xtensor<double, 2> Wmat = xt::zeros<double>({n, n});
    const auto blocks = blocks_from_shells(shells.shells);
    for_each_canonical_block_pair(
        Wmat, blocks,
        [&](const native::ShellBlock& A, const native::ShellBlock& B,
            double* buf) { pvp_block(A, B, atoms.atoms.data(), natm, buf); });
    return Wmat;
}

double nuclear_repulsion_energy(const AtomList& atoms) {
    const int natm = static_cast<int>(atoms.atoms.size());
    double e = 0.0;
    for (int a = 0; a < natm; ++a) {
        for (int b = a + 1; b < natm; ++b) {
            const double dx = atoms.atoms[a].pos[0] - atoms.atoms[b].pos[0];
            const double dy = atoms.atoms[a].pos[1] - atoms.atoms[b].pos[1];
            const double dz = atoms.atoms[a].pos[2] - atoms.atoms[b].pos[2];
            const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
            e += static_cast<double>(atoms.atoms[a].Z * atoms.atoms[b].Z) / r;
        }
    }
    return e;
}

} // namespace scf
} // namespace cpu
} // namespace ATV

namespace ATV {
namespace cpu {
namespace scf {

namespace {

/// 1D integral I(la, lb, e) = ∫ (x−A)^la (x−B)^lb (x−C)^e exp(−p(x−P)²) dx
/// via binomial expansion about the Gaussian product centre P and the
/// even-moment closed form ∫ x^n e^{−px²} dx = (n−1)!! / (2p)^{n/2} √(π/p).
double moment_1d(int la, int lb, int e, double pa, double pb, double pc,
                 double p) {
    auto binom = [](int n, int k) {
        double r = 1.0;
        for (int i = 1; i <= k; ++i)
            r = r * (n - k + i) / i;
        return r;
    };
    const double spi = std::sqrt(M_PI / p);
    double acc = 0.0;
    for (int i = 0; i <= la; ++i) {
        for (int j = 0; j <= lb; ++j) {
            for (int k = 0; k <= e; ++k) {
                const int n = i + j + k;
                if (n % 2)
                    continue;
                double dfac = 1.0; // (n−1)!!
                for (int m = n - 1; m > 1; m -= 2)
                    dfac *= m;
                acc += binom(la, i) * binom(lb, j) * binom(e, k)
                       * std::pow(pa, la - i) * std::pow(pb, lb - j)
                       * std::pow(pc, e - k) * dfac
                       / std::pow(2.0 * p, n / 2.0) * spi;
            }
        }
    }
    return acc;
}

} // namespace

MultipoleIntegrals::MultipoleIntegrals(const native::Shell* shells, int n_ao,
                                       const double* origin_bohr) {
    static const int kPow[9][3] = {// m1: x,y,z; m2: xx,xy,xz,yy,yz,zz
                                   {1, 0, 0}, {0, 1, 0}, {0, 0, 1},
                                   {2, 0, 0}, {1, 1, 0}, {1, 0, 1},
                                   {0, 2, 0}, {0, 1, 1}, {0, 0, 2}};
    xt::xtensor<double, 2>* out[9];
    for (int a = 0; a < 3; ++a) {
        m1_[a] = xt::zeros<double>({std::size_t(n_ao), std::size_t(n_ao)});
        out[a] = &m1_[a];
    }
    for (int k = 0; k < 6; ++k) {
        m2_[k] = xt::zeros<double>({std::size_t(n_ao), std::size_t(n_ao)});
        out[3 + k] = &m2_[k];
    }

    int ca[3], cb[3];
    for (int i = 0; i < n_ao; ++i) {
        const auto& A = shells[i];
        int compsA[kMaxNCart][3];
        fill_cart_comps(A.l, compsA);
        for (int c = 0; c < 3; ++c)
            ca[c] = compsA[A.cart][c];
        for (int j = 0; j <= i; ++j) {
            const auto& B = shells[j];
            int compsB[kMaxNCart][3];
            fill_cart_comps(B.l, compsB);
            for (int c = 0; c < 3; ++c)
                cb[c] = compsB[B.cart][c];

            double acc[9] = {};
            for (int pa = 0; pa < A.n_prim; ++pa) {
                for (int pb = 0; pb < B.n_prim; ++pb) {
                    const double a = A.alpha[pa];
                    const double b = B.alpha[pb];
                    const double p = a + b;
                    const double cc = A.coeff[pa] * B.coeff[pb];
                    double pre = cc;
                    double P[3], PA[3], PB[3], PC[3];
                    double r2 = 0.0;
                    for (int c = 0; c < 3; ++c) {
                        const double d = A.center[c] - B.center[c];
                        r2 += d * d;
                        P[c] = (a * A.center[c] + b * B.center[c]) / p;
                        PA[c] = P[c] - A.center[c];
                        PB[c] = P[c] - B.center[c];
                        PC[c] = P[c] - origin_bohr[c];
                    }
                    pre *= std::exp(-a * b / p * r2);
                    // Per-axis 1D moments up to e = 2 (reused across the 9
                    // operators for this primitive pair).
                    double I[3][3]; // [axis][e]
                    for (int c = 0; c < 3; ++c)
                        for (int e = 0; e < 3; ++e)
                            I[c][e] = moment_1d(ca[c], cb[c], e, PA[c],
                                                PB[c], PC[c], p);
                    for (int op = 0; op < 9; ++op) {
                        double v = pre;
                        for (int c = 0; c < 3; ++c)
                            v *= I[c][kPow[op][c]];
                        acc[op] += v;
                    }
                }
            }
            for (int op = 0; op < 9; ++op) {
                (*out[op])(i, j) = acc[op];
                (*out[op])(j, i) = acc[op];
            }
        }
    }
}

} // namespace scf
} // namespace cpu
} // namespace ATV
