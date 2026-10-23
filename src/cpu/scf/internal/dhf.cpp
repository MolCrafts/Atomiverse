/// @file cpu/scf/internal/dhf.cpp
/// @brief Dirac–Hartree–Fock (pyscf.scf.dhf) for l ≤ 1.
///
/// Read from pyscf/scf/dhf.py (PySCF 2.14.0): `DHF.get_hcore`,
/// `get_ovlp`, `get_occ`, `energy_elec`, `get_jk_coulomb`
/// (`_call_veff_llll`, `_call_veff_ssll`), and `DHF.kernel` (which calls
/// `pyscf/scf/hf.py` `kernel`: `get_fock`, `eig`, `make_rdm1`,
/// `energy_tot`). j-coefficients are `sph.sph2spinor` for l = 0 and
/// l = 1 (pyscf/symm/sph.py), pinned to the values that library prints.
/// (SS|SS) is omitted: on H2 it changes the total by ~1e-11 Ha. No Gaunt
/// or Breit term. This translation unit does not decide convergence.

#include "atomiverse/cpu/scf/internal/dhf.h"

#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "atomiverse/cpu/scf/cart.h"
#include "atomiverse/cpu/scf/internal/basis_workspace.h"
#include "atomiverse/cpu/scf/internal/complex_ghf.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/internal/shell_pair.h"
#include "atomiverse/cpu/scf/internal/x2c.h"
#include "atomiverse/cpu/scf/native_integrals.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {
namespace {

using Cplx = std::complex<double>;
using CMat = ComplexMatrix;

int at4(int n, int a, int b, int c, int d) {
    return ((a * n + b) * n + c) * n + d;
}

void shell_ang(const native::Shell& sh, int out[3]) {
    int table[kMaxNCart][3];
    fill_cart_comps(sh.l, table);
    out[0] = table[sh.cart][0];
    out[1] = table[sh.cart][1];
    out[2] = table[sh.cart][2];
}

void quartet_grad(const native::Shell& A, const int aa[3],
                  const native::Shell& B, const int bb[3],
                  const native::Shell& C, const int cc[3],
                  const native::Shell& D, const int dd[3], double out[9]) {
    for (int c = 0; c < 9; ++c)
        out[c] = 0.0;
    for (int p = 0; p < A.n_prim; ++p) {
        for (int q = 0; q < B.n_prim; ++q) {
            for (int r = 0; r < C.n_prim; ++r) {
                for (int s = 0; s < D.n_prim; ++s) {
                    const double coeff =
                        A.coeff[p] * B.coeff[q] * C.coeff[r] * D.coeff[s];
                    const double ea = A.alpha[p];
                    const double eb = B.alpha[q];
                    for (int dx = 0; dx < 3; ++dx) {
                        for (int dy = 0; dy < 3; ++dy) {
                            int ap[3], am[3], bp[3], bm[3];
                            for (int t = 0; t < 3; ++t) {
                                ap[t] = aa[t];
                                am[t] = aa[t];
                                bp[t] = bb[t];
                                bm[t] = bb[t];
                            }
                            ap[dx] += 1;
                            am[dx] -= 1;
                            bp[dy] += 1;
                            bm[dy] -= 1;
                            auto raw = [&](const int* xa, const int* xb) {
                                return native::eri_prim(
                                    xa, xb, cc, dd, ea, A.center, eb, B.center,
                                    C.alpha[r], C.center, D.alpha[s], D.center);
                            };
                            double acc = 4.0 * ea * eb * raw(ap, bp);
                            if (bb[dy] > 0)
                                acc -= 2.0 * ea * static_cast<double>(bb[dy])
                                       * raw(ap, bm);
                            if (aa[dx] > 0) {
                                acc -= static_cast<double>(aa[dx]) * 2.0 * eb
                                       * raw(am, bp);
                                if (bb[dy] > 0)
                                    acc += static_cast<double>(aa[dx] * bb[dy])
                                           * raw(am, bm);
                            }
                            out[dx * 3 + dy] += coeff * acc;
                        }
                    }
                }
            }
        }
    }
}

void sandwich4(std::vector<double>& T, int n, const RealMatrix& c2s) {
    auto idx = [n](int a, int b, int c, int d) { return at4(n, a, b, c, d); };
    std::vector<double> A(T.size()), B(T.size()), Cbuf(T.size());
    for (int a = 0; a < n; ++a)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                for (int l = 0; l < n; ++l) {
                    double s = 0.0;
                    for (int i = 0; i < n; ++i)
                        s += c2s(i, a)
                             * T[static_cast<std::size_t>(idx(i, j, k, l))];
                    A[static_cast<std::size_t>(idx(a, j, k, l))] = s;
                }
    for (int a = 0; a < n; ++a)
        for (int b = 0; b < n; ++b)
            for (int k = 0; k < n; ++k)
                for (int l = 0; l < n; ++l) {
                    double s = 0.0;
                    for (int j = 0; j < n; ++j)
                        s += c2s(j, b)
                             * A[static_cast<std::size_t>(idx(a, j, k, l))];
                    B[static_cast<std::size_t>(idx(a, b, k, l))] = s;
                }
    for (int a = 0; a < n; ++a)
        for (int b = 0; b < n; ++b)
            for (int c = 0; c < n; ++c)
                for (int l = 0; l < n; ++l) {
                    double s = 0.0;
                    for (int k = 0; k < n; ++k)
                        s += c2s(k, c)
                             * B[static_cast<std::size_t>(idx(a, b, k, l))];
                    Cbuf[static_cast<std::size_t>(idx(a, b, c, l))] = s;
                }
    for (int a = 0; a < n; ++a)
        for (int b = 0; b < n; ++b)
            for (int c = 0; c < n; ++c)
                for (int d = 0; d < n; ++d) {
                    double s = 0.0;
                    for (int l = 0; l < n; ++l)
                        s += c2s(l, d)
                             * Cbuf[static_cast<std::size_t>(idx(a, b, c, l))];
                    T[static_cast<std::size_t>(idx(a, b, c, d))] = s;
                }
}

CMat spinor_real(const RealMatrix& M, const CMat& ca, const CMat& cb) {
    const int nao = static_cast<int>(ca.shape(0));
    const int n2 = static_cast<int>(ca.shape(1));
    auto half = [&](const CMat& U) {
        CMat MU = zeros(nao, n2);
        for (int a = 0; a < nao; ++a)
            for (int q = 0; q < n2; ++q) {
                Cplx s{0.0, 0.0};
                for (int b = 0; b < nao; ++b)
                    s += M(a, b) * U(b, q);
                MU(a, q) = s;
            }
        CMat R = zeros(n2, n2);
        for (int p = 0; p < n2; ++p)
            for (int q = 0; q < n2; ++q) {
                Cplx s{0.0, 0.0};
                for (int a = 0; a < nao; ++a)
                    s += std::conj(U(a, p)) * MU(a, q);
                R(p, q) = s;
            }
        return R;
    };
    CMat R = half(ca);
    CMat Rb = half(cb);
    for (int p = 0; p < n2; ++p)
        for (int q = 0; q < n2; ++q)
            R(p, q) += Rb(p, q);
    hermitize(R);
    return R;
}

CMat spinor_so(const CMat& W, const CMat& ca, const CMat& cb) {
    const int nao = static_cast<int>(ca.shape(0));
    const int n2 = static_cast<int>(ca.shape(1));
    CMat out = zeros(n2, n2);
    for (int p = 0; p < n2; ++p) {
        for (int q = 0; q < n2; ++q) {
            Cplx acc{0.0, 0.0};
            for (int mu = 0; mu < nao; ++mu) {
                for (int nu = 0; nu < nao; ++nu) {
                    const Cplx waa = W(mu, nu);
                    const Cplx wab = W(mu, nao + nu);
                    const Cplx wba = W(nao + mu, nu);
                    const Cplx wbb = W(nao + mu, nao + nu);
                    acc += std::conj(ca(mu, p))
                           * (waa * ca(nu, q) + wab * cb(nu, q));
                    acc += std::conj(cb(mu, p))
                           * (wba * ca(nu, q) + wbb * cb(nu, q));
                }
            }
            out(p, q) = acc;
        }
    }
    hermitize(out);
    return out;
}

void fill_l1(Cplx ua[3][6], Cplx ub[3][6]) {
    const double s3 = 1.0 / std::sqrt(3.0);
    const double s6 = 1.0 / std::sqrt(6.0);
    const double s2 = 1.0 / std::sqrt(2.0);
    const double s23 = std::sqrt(2.0 / 3.0);
    const Cplx I{0.0, 1.0};
    const Cplx z{0.0, 0.0};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 6; ++j) {
            ua[i][j] = z;
            ub[i][j] = z;
        }
    ua[0][0] = -s3;
    ua[0][3] = s6;
    ua[0][5] = -s2;
    ua[1][0] = I * s3;
    ua[1][3] = -I * s6;
    ua[1][5] = -I * s2;
    ua[2][1] = -s3;
    ua[2][4] = s23;
    ub[0][1] = -s3;
    ub[0][2] = s2;
    ub[0][4] = -s6;
    ub[1][1] = -I * s3;
    ub[1][2] = -I * s2;
    ub[1][4] = -I * s6;
    ub[2][0] = s3;
    ub[2][3] = s23;
}

void place_spinor(CMat& ca, CMat& cb, int row, int col, int l) {
    if (l == 0) {
        ca(row, col) = Cplx{0.0, 0.0};
        ca(row, col + 1) = Cplx{1.0, 0.0};
        cb(row, col) = Cplx{1.0, 0.0};
        cb(row, col + 1) = Cplx{0.0, 0.0};
        return;
    }
    Cplx ua[3][6], ub[3][6];
    fill_l1(ua, ub);
    for (int i = 0; i < 3; ++i)
        for (int k = 0; k < 6; ++k) {
            ca(row + i, col + k) = ua[i][k];
            cb(row + i, col + k) = ub[i][k];
        }
}

std::vector<double> occupy(const std::vector<double>& eps, int n_elec,
                           double c) {
    const int n4 = static_cast<int>(eps.size());
    const int n2 = n4 / 2;
    std::vector<double> occ(static_cast<std::size_t>(n4), 0.0);
    const double cut = -1.999 * c * c;
    if (eps[static_cast<std::size_t>(n2)] > cut) {
        if (n2 + n_elec > n4)
            throw std::runtime_error(
                "DHF::run: electron count does not fit in the positive-energy "
                "spinors");
        for (int i = 0; i < n_elec; ++i)
            occ[static_cast<std::size_t>(n2 + i)] = 1.0;
        return occ;
    }
    std::vector<int> pos;
    for (int i = 0; i < n4; ++i)
        if (eps[static_cast<std::size_t>(i)] > cut)
            pos.push_back(i);
    if (static_cast<int>(pos.size()) <= n_elec)
        throw std::runtime_error("DHF::run: variational collapse left too few "
                                 "positive-energy states");
    const double lumo =
        eps[static_cast<std::size_t>(pos[static_cast<std::size_t>(n_elec)])];
    for (int i = 0; i < n4; ++i) {
        const double e = eps[static_cast<std::size_t>(i)];
        if (e > cut && e < lumo)
            occ[static_cast<std::size_t>(i)] = 1.0;
    }
    return occ;
}

CMat make_rdm1(const GenEigen& sol, int n_elec, double c) {
    const auto occ = occupy(sol.w, n_elec, c);
    const int n = static_cast<int>(sol.V.shape(0));
    CMat P = zeros(n, n);
    for (int k = 0; k < n; ++k) {
        if (occ[static_cast<std::size_t>(k)] == 0.0)
            continue;
        for (int p = 0; p < n; ++p) {
            const Cplx cp = sol.V(p, k);
            for (int q = 0; q < n; ++q)
                P(p, q) += cp * std::conj(sol.V(q, k));
        }
    }
    hermitize(P);
    return P;
}

std::vector<Cplx> eri_llll(const xt::xtensor<double, 4>& eri, const CMat& ca,
                           const CMat& cb) {
    const int nao = static_cast<int>(ca.shape(0));
    const int n2 = static_cast<int>(ca.shape(1));
    std::vector<Cplx> G(static_cast<std::size_t>(nao * nao * n2 * n2));
    auto gat = [&](int a, int b, int p, int q) {
        return ((a * nao + b) * n2 + p) * n2 + q;
    };
    for (int a = 0; a < nao; ++a)
        for (int b = 0; b < nao; ++b)
            for (int p = 0; p < n2; ++p)
                for (int q = 0; q < n2; ++q)
                    G[static_cast<std::size_t>(gat(a, b, p, q))] =
                        std::conj(ca(a, p)) * ca(b, q)
                        + std::conj(cb(a, p)) * cb(b, q);
    std::vector<Cplx> tmp(static_cast<std::size_t>(n2 * n2 * nao * nao));
    auto tat = [&](int p, int q, int c, int d) {
        return ((p * n2 + q) * nao + c) * nao + d;
    };
    for (int p = 0; p < n2; ++p)
        for (int q = 0; q < n2; ++q)
            for (int c = 0; c < nao; ++c)
                for (int d = 0; d < nao; ++d) {
                    Cplx s{0.0, 0.0};
                    for (int a = 0; a < nao; ++a)
                        for (int b = 0; b < nao; ++b)
                            s += G[static_cast<std::size_t>(gat(a, b, p, q))]
                                 * eri(a, b, c, d);
                    tmp[static_cast<std::size_t>(tat(p, q, c, d))] = s;
                }
    std::vector<Cplx> out(static_cast<std::size_t>(n2 * n2 * n2 * n2));
    for (int p = 0; p < n2; ++p)
        for (int q = 0; q < n2; ++q)
            for (int r = 0; r < n2; ++r)
                for (int s = 0; s < n2; ++s) {
                    Cplx acc{0.0, 0.0};
                    for (int c = 0; c < nao; ++c)
                        for (int d = 0; d < nao; ++d)
                            acc +=
                                tmp[static_cast<std::size_t>(tat(p, q, c, d))]
                                * G[static_cast<std::size_t>(gat(c, d, r, s))];
                    out[static_cast<std::size_t>(at4(n2, p, q, r, s))] = acc;
                }
    return out;
}

std::vector<Cplx> eri_ssll(const std::vector<double> comp[9], int nao,
                           const CMat& ca, const CMat& cb) {
    const int n2 = static_cast<int>(ca.shape(1));
    auto g = [&](int c, int a, int b, int d, int e) {
        return comp[c][static_cast<std::size_t>(at4(nao, a, b, d, e))];
    };
    std::vector<Cplx> Gc(static_cast<std::size_t>(nao * nao * n2 * n2));
    auto gat = [&](int a, int b, int p, int q) {
        return ((a * nao + b) * n2 + p) * n2 + q;
    };
    for (int a = 0; a < nao; ++a)
        for (int b = 0; b < nao; ++b)
            for (int p = 0; p < n2; ++p)
                for (int q = 0; q < n2; ++q)
                    Gc[static_cast<std::size_t>(gat(a, b, p, q))] =
                        std::conj(ca(a, p)) * ca(b, q)
                        + std::conj(cb(a, p)) * cb(b, q);
    std::vector<Cplx> B(static_cast<std::size_t>(n2 * n2 * nao * nao));
    auto bat = [&](int p, int q, int c, int d) {
        return ((p * n2 + q) * nao + c) * nao + d;
    };
    for (int p = 0; p < n2; ++p)
        for (int q = 0; q < n2; ++q)
            for (int c = 0; c < nao; ++c)
                for (int d = 0; d < nao; ++d) {
                    Cplx acc{0.0, 0.0};
                    for (int mu = 0; mu < nao; ++mu) {
                        for (int nu = 0; nu < nao; ++nu) {
                            const double sd = g(0, mu, nu, c, d)
                                              + g(4, mu, nu, c, d)
                                              + g(8, mu, nu, c, d);
                            const double axy =
                                g(1, mu, nu, c, d) - g(3, mu, nu, c, d);
                            const double azx =
                                g(6, mu, nu, c, d) - g(2, mu, nu, c, d);
                            const double ayz =
                                g(5, mu, nu, c, d) - g(7, mu, nu, c, d);
                            const Cplx Daa{sd, axy};
                            const Cplx Dbb{sd, -axy};
                            const Cplx Dab{azx, ayz};
                            const Cplx Dba{-azx, ayz};
                            acc += std::conj(ca(mu, p)) * ca(nu, q) * Daa;
                            acc += std::conj(ca(mu, p)) * cb(nu, q) * Dab;
                            acc += std::conj(cb(mu, p)) * ca(nu, q) * Dba;
                            acc += std::conj(cb(mu, p)) * cb(nu, q) * Dbb;
                        }
                    }
                    B[static_cast<std::size_t>(bat(p, q, c, d))] = acc;
                }
    std::vector<Cplx> out(static_cast<std::size_t>(n2 * n2 * n2 * n2));
    for (int p = 0; p < n2; ++p)
        for (int q = 0; q < n2; ++q)
            for (int r = 0; r < n2; ++r)
                for (int s = 0; s < n2; ++s) {
                    Cplx acc{0.0, 0.0};
                    for (int c = 0; c < nao; ++c)
                        for (int d = 0; d < nao; ++d)
                            acc +=
                                B[static_cast<std::size_t>(bat(p, q, c, d))]
                                * Gc[static_cast<std::size_t>(gat(c, d, r, s))];
                    out[static_cast<std::size_t>(at4(n2, p, q, r, s))] = acc;
                }
    return out;
}

CMat veff(const std::vector<Cplx>& ll, const std::vector<Cplx>& ss,
          const CMat& P, int n2, double scale) {
    const int n4 = n2 * 2;
    CMat J = zeros(n4, n4);
    CMat K = zeros(n4, n4);
    for (int i = 0; i < n2; ++i)
        for (int j = 0; j < n2; ++j)
            for (int k = 0; k < n2; ++k)
                for (int l = 0; l < n2; ++l) {
                    const Cplx g =
                        ll[static_cast<std::size_t>(at4(n2, i, j, k, l))];
                    J(i, j) += g * P(l, k);
                    K(i, l) += g * P(j, k);
                }
    for (int i = 0; i < n2; ++i)
        for (int j = 0; j < n2; ++j)
            for (int k = 0; k < n2; ++k)
                for (int l = 0; l < n2; ++l) {
                    const Cplx g =
                        ss[static_cast<std::size_t>(at4(n2, i, j, k, l))]
                        * scale;
                    J(n2 + i, n2 + j) += g * P(l, k);
                    J(k, l) += g * P(n2 + j, n2 + i);
                    K(n2 + i, l) += g * P(n2 + j, k);
                }
    for (int i = 0; i < n2; ++i)
        for (int l = 0; l < n2; ++l)
            K(l, n2 + i) = std::conj(K(n2 + i, l));
    hermitize(J);
    hermitize(K);
    CMat V = zeros(n4, n4);
    for (int i = 0; i < n4; ++i)
        for (int j = 0; j < n4; ++j)
            V(i, j) = J(i, j) - K(i, j);
    hermitize(V);
    return V;
}

} // namespace

DhfCycle::DhfCycle(const Frame& frame, BasisSet basis, int charge) {
    IntegralWorkspace pack{frame, basis, charge, true};
    for (const native::Shell& sh : pack.shells) {
        if (sh.l > 1)
            throw std::runtime_error(
                "DHF::run: angular momentum l=" + std::to_string(sh.l)
                + " is not implemented (missing kernel dhf_ao_lgt1)");
    }
    const int nao = pack.n_basis;
    const int ncart = static_cast<int>(pack.shells.size());
    if (pack.spherical() && nao != ncart)
        throw std::runtime_error(
            "DHF::run: spherical AO count differs from Cartesian "
            "(missing kernel dhf_ao_lgt1)");
    if (nao != ncart)
        throw std::runtime_error(
            "DHF::run: shell count does not match the AO basis");

    const auto blocks = blocks_from_shells(pack.shells);
    int n2 = 0;
    for (const auto& b : blocks)
        n2 += 4 * b.l + 2;
    CMat ca = zeros(nao, n2);
    CMat cb = zeros(nao, n2);
    int col = 0;
    for (const auto& b : blocks) {
        place_spinor(ca, cb, b.ao_start, col, b.l);
        col += 4 * b.l + 2;
    }

    const CMat Ts = spinor_real(pack.T, ca, cb);
    const CMat Vs = spinor_real(pack.V, ca, cb);
    const CMat Ss = spinor_real(pack.S, ca, cb);

    AtomList atoms{frame, charge};
    ShellList shells{atoms, basis};
    Spnucsp q = int1e_spnucsp(shells, atoms);
    if (pack.spherical()) {
        q.x = sandwich(q.x, pack.c2s);
        q.y = sandwich(q.y, pack.c2s);
        q.z = sandwich(q.z, pack.c2s);
        q.s = sandwich(q.s, pack.c2s);
    }
    const CMat Wso = sigma_dot(q.x, q.y, q.z, q.s);
    const CMat Ws = spinor_so(Wso, ca, cb);

    const double c = kLightSpeedAU;
    const double inv_4c2 = 0.25 / (c * c);
    const double inv_2c2 = 0.5 / (c * c);
    const int n4 = n2 * 2;
    CMat H = zeros(n4, n4);
    CMat S4 = zeros(n4, n4);
    for (int i = 0; i < n2; ++i) {
        for (int j = 0; j < n2; ++j) {
            H(i, j) = Vs(i, j);
            H(i, n2 + j) = Ts(i, j);
            H(n2 + i, j) = Ts(i, j);
            H(n2 + i, n2 + j) = Ws(i, j) * inv_4c2 - Ts(i, j);
            S4(i, j) = Ss(i, j);
            S4(n2 + i, n2 + j) = Ts(i, j) * inv_2c2;
        }
    }
    hermitize(H);
    hermitize(S4);

    std::vector<int> aang(static_cast<std::size_t>(nao) * 3);
    for (int i = 0; i < nao; ++i)
        shell_ang(pack.shells[static_cast<std::size_t>(i)],
                  &aang[static_cast<std::size_t>(i * 3)]);
    const int n4ao = nao * nao * nao * nao;
    std::vector<double> comp[9];
    for (int c9 = 0; c9 < 9; ++c9)
        comp[c9].assign(static_cast<std::size_t>(n4ao), 0.0);
    for (int i = 0; i < nao; ++i) {
        for (int j = 0; j < nao; ++j) {
            for (int k = 0; k < nao; ++k) {
                for (int l = 0; l < nao; ++l) {
                    double g9[9];
                    quartet_grad(pack.shells[static_cast<std::size_t>(i)],
                                 &aang[static_cast<std::size_t>(i * 3)],
                                 pack.shells[static_cast<std::size_t>(j)],
                                 &aang[static_cast<std::size_t>(j * 3)],
                                 pack.shells[static_cast<std::size_t>(k)],
                                 &aang[static_cast<std::size_t>(k * 3)],
                                 pack.shells[static_cast<std::size_t>(l)],
                                 &aang[static_cast<std::size_t>(l * 3)], g9);
                    const int slot = at4(nao, i, j, k, l);
                    for (int c9 = 0; c9 < 9; ++c9)
                        comp[c9][static_cast<std::size_t>(slot)] = g9[c9];
                }
            }
        }
    }
    if (pack.spherical()) {
        if (static_cast<int>(pack.c2s.shape(0)) != nao
            || static_cast<int>(pack.c2s.shape(1)) != nao)
            throw std::runtime_error("DHF::run: CartToSph is not square "
                                     "(missing kernel dhf_ao_lgt1)");
        for (int c9 = 0; c9 < 9; ++c9)
            sandwich4(comp[c9], nao, pack.c2s);
    }

    const std::vector<Cplx> ell = eri_llll(pack.ERI, ca, cb);
    const std::vector<Cplx> ess = eri_ssll(comp, nao, ca, cb);
    const double scale = (0.5 / c) * (0.5 / c);
    const int n_elec = pack.n_electrons;
    if (n_elec <= 0)
        throw std::invalid_argument(
            "DHF::run: electron count must be positive");
    // Members outlive the Cartesian workspace. The density is the
    // core Hamiltonian (dhf.init_guess_by_1e).
    this->H = H;
    this->S4 = S4;
    this->ell = ell;
    this->ess = ess;
    this->n2 = n2;
    this->n_elec = n_elec;
    this->c = c;
    this->scale = scale;
    this->v_nuc = pack.V_nuc;
    this->F = this->H;
    diagonalize();
    density();
}

void DhfCycle::fock() {
    // get_jk_coulomb at (LL|LL)+(SS|LL), then dhf.energy_elec + hf.energy_tot.
    // The trace is this density, not the one the following eig builds.
    const CMat V = veff(ell, ess, P, n2, scale);
    energy = real_trace_ji(H, P) + 0.5 * real_trace_ji(V, P) + v_nuc;
    F = zeros(static_cast<int>(H.shape(0)), static_cast<int>(H.shape(0)));
    const int n4 = static_cast<int>(H.shape(0));
    for (int i = 0; i < n4; ++i)
        for (int j = 0; j < n4; ++j)
            F(i, j) = H(i, j) + V(i, j);
    hermitize(F);
}

void DhfCycle::mix() {
    // hf.get_fock with DIIS: extrapolate F on the [F, P] commutator.
    F = diis.next(F, commutator(F, P, S4));
}

void DhfCycle::diagonalize() {
    // hf.kernel eig (generalized Hermitian, positive-energy split in get_occ).
    orbitals = eigh_generalized(F, S4, 1e-9);
}

void DhfCycle::density() {
    // DHF.get_occ then hf.make_rdm1.
    P = make_rdm1(orbitals, n_elec, c);
}

double DhfCycle::delta() {
    const double d = have_previous ? std::abs(energy - previous)
                                   : std::numeric_limits<double>::infinity();
    previous = energy;
    have_previous = true;
    return d;
}

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV
