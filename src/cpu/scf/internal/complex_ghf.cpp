/// @file cpu/scf/internal/complex_ghf.cpp
/// @brief Complex spin-orbital GHF and X2C-1e core.
///
/// PySCF functions transplanted (contracted basis, xuncontract=False):
///   ghf.get_jk, ghf.get_hcore, ghf.get_ovlp, ghf.get_occ, ghf.get_veff
///   hf.energy_elec, hf.make_rdm1
///   x2c.SpinOrbitalX2CHelper.get_hcore
///   x2c._block_diag, x2c._sigma_dot, x2c._x2c1e_get_hcore
/// dhf.get_hcore supplies the same modified-Dirac block layout in the
/// j-adapted basis; the spin-orbital form is _x2c1e_get_hcore.
/// hessian.rhf.hess_elec and df.df_jk.density_fit are not this product.

#include "atomiverse/cpu/scf/internal/complex_ghf.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "atomiverse/cpu/linalg/eigensolve.h"
#include "atomiverse/cpu/scf/cart.h"
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

bool solve_dense(std::vector<double>& A, std::vector<double>& b, int n) {
    for (int k = 0; k < n; ++k) {
        int piv = k;
        double best = std::abs(A[static_cast<std::size_t>(k * n + k)]);
        for (int i = k + 1; i < n; ++i) {
            const double v = std::abs(A[static_cast<std::size_t>(i * n + k)]);
            if (v > best) {
                best = v;
                piv = i;
            }
        }
        if (!(best > 1e-14))
            return false;
        if (piv != k) {
            for (int j = k; j < n; ++j)
                std::swap(A[static_cast<std::size_t>(k * n + j)],
                          A[static_cast<std::size_t>(piv * n + j)]);
            std::swap(b[static_cast<std::size_t>(k)],
                      b[static_cast<std::size_t>(piv)]);
        }
        const double diag = A[static_cast<std::size_t>(k * n + k)];
        for (int i = k + 1; i < n; ++i) {
            const double f = A[static_cast<std::size_t>(i * n + k)] / diag;
            for (int j = k + 1; j < n; ++j)
                A[static_cast<std::size_t>(i * n + j)] -=
                    f * A[static_cast<std::size_t>(k * n + j)];
            b[static_cast<std::size_t>(i)] -=
                f * b[static_cast<std::size_t>(k)];
        }
    }
    for (int i = n - 1; i >= 0; --i) {
        double s = b[static_cast<std::size_t>(i)];
        for (int j = i + 1; j < n; ++j)
            s -= A[static_cast<std::size_t>(i * n + j)]
                 * b[static_cast<std::size_t>(j)];
        b[static_cast<std::size_t>(i)] =
            s / A[static_cast<std::size_t>(i * n + i)];
    }
    return true;
}

double frobenius_re(const CMat& A, const CMat& B) {
    const int n = static_cast<int>(A.shape(0));
    const int m = static_cast<int>(A.shape(1));
    double acc = 0.0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            acc += (std::conj(A(i, j)) * B(i, j)).real();
    return acc;
}

/// ⟨∂_d g_a | 1/r | ∂_e g_b⟩ without −Z. Same four-term identity as
/// the spin-free pVp kernel, with independent Cartesian axes.
double pvp_prim_pair(const int a_ang[3], const int b_ang[3], double aa,
                     double ab, double gamma, const double PA[3],
                     const double PB[3], const double PC[3], double K_AB, int d,
                     int e) {
    int a_plus[3] = {a_ang[0], a_ang[1], a_ang[2]};
    int b_plus[3] = {b_ang[0], b_ang[1], b_ang[2]};
    a_plus[d] += 1;
    b_plus[e] += 1;
    double acc =
        4.0 * aa * ab
        * native::nuclear_prim(a_plus, b_plus, gamma, PA, PB, PC, K_AB);
    if (b_ang[e] > 0) {
        int b_minus[3] = {b_ang[0], b_ang[1], b_ang[2]};
        b_minus[e] -= 1;
        acc -= 2.0 * aa * static_cast<double>(b_ang[e])
               * native::nuclear_prim(a_plus, b_minus, gamma, PA, PB, PC, K_AB);
    }
    if (a_ang[d] > 0) {
        int a_minus[3] = {a_ang[0], a_ang[1], a_ang[2]};
        a_minus[d] -= 1;
        acc -= static_cast<double>(a_ang[d]) * 2.0 * ab
               * native::nuclear_prim(a_minus, b_plus, gamma, PA, PB, PC, K_AB);
        if (b_ang[e] > 0) {
            int b_minus[3] = {b_ang[0], b_ang[1], b_ang[2]};
            b_minus[e] -= 1;
            acc += static_cast<double>(a_ang[d] * b_ang[e])
                   * native::nuclear_prim(a_minus, b_minus, gamma, PA, PB, PC,
                                          K_AB);
        }
    }
    return acc;
}

void spnucsp_block(const native::ShellBlock& A, const native::ShellBlock& B,
                   const native::Atom* atoms, int natm, double* out) {
    int a_ang[kMaxNCart][3];
    int b_ang[kMaxNCart][3];
    fill_cart_comps(A.l, a_ang);
    fill_cart_comps(B.l, b_ang);
    const int na = A.n_cart;
    const int nb = B.n_cart;
    const int nab = na * nb;

    for (int p = 0; p < A.n_prim; ++p) {
        for (int q = 0; q < B.n_prim; ++q) {
            const double aa = A.alpha[p];
            const double ab = B.alpha[q];
            const double gamma = aa + ab;
            const double mu = aa * ab / gamma;
            const double R2 =
                (A.center[0] - B.center[0]) * (A.center[0] - B.center[0])
                + (A.center[1] - B.center[1]) * (A.center[1] - B.center[1])
                + (A.center[2] - B.center[2]) * (A.center[2] - B.center[2]);
            const double K_AB = std::exp(-mu * R2);
            const double cpq = A.coeff[p] * B.coeff[q];
            double PA[3], PB[3], P[3];
            for (int d = 0; d < 3; ++d) {
                P[d] = (aa * A.center[d] + ab * B.center[d]) / gamma;
                PA[d] = P[d] - A.center[d];
                PB[d] = P[d] - B.center[d];
            }
            for (int c = 0; c < natm; ++c) {
                if (atoms[c].Z == 0)
                    continue;
                double PC[3];
                for (int d = 0; d < 3; ++d)
                    PC[d] = P[d] - atoms[c].pos[d];
                const double mz = -static_cast<double>(atoms[c].Z);
                for (int ia = 0; ia < na; ++ia) {
                    for (int ib = 0; ib < nb; ++ib) {
                        for (int d = 0; d < 3; ++d) {
                            for (int e = 0; e < 3; ++e) {
                                const double val = pvp_prim_pair(
                                    a_ang[ia], b_ang[ib], aa, ab, gamma, PA, PB,
                                    PC, K_AB, d, e);
                                out[(d * 3 + e) * nab + ia * nb + ib] +=
                                    cpq * mz * val;
                            }
                        }
                    }
                }
            }
        }
    }
}

CMat block_of(const CMat& M, int row, int col, int n) {
    CMat B = zeros(n, n);
    const int r0 = row * n;
    const int c0 = col * n;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            B(i, j) = M(r0 + i, c0 + j);
    return B;
}

void write_block(CMat& M, int row, int col, int n, const CMat& B) {
    const int r0 = row * n;
    const int c0 = col * n;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            M(r0 + i, c0 + j) = B(i, j);
}

/// ghf.get_jk: J sees Paa+Pbb; K acts on each spin block.
/// (ij|kl) chemist: J_kl = Σ_ij (ij|kl) D_ji, K_il = Σ_jk (ij|kl) D_jk.
CMat ghf_fock(const CMat& h, const xt::xtensor<double, 4>& eri, const CMat& P) {
    const int nso = static_cast<int>(h.shape(0));
    if (nso % 2 != 0)
        throw std::invalid_argument("ghf_fock: spin-orbital dimension");
    const int n = nso / 2;
    const CMat Paa = block_of(P, 0, 0, n);
    const CMat Pab = block_of(P, 0, 1, n);
    const CMat Pba = block_of(P, 1, 0, n);
    const CMat Pbb = block_of(P, 1, 1, n);
    CMat Pc = zeros(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            Pc(i, j) = Paa(i, j) + Pbb(i, j);

    CMat J = zeros(n, n);
    CMat Kaa = zeros(n, n);
    CMat Kbb = zeros(n, n);
    CMat Kab = zeros(n, n);
    CMat Kba = zeros(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                for (int l = 0; l < n; ++l) {
                    const Cplx g{eri(i, j, k, l), 0.0};
                    J(k, l) += g * Pc(j, i);
                    Kaa(i, l) += g * Paa(j, k);
                    Kbb(i, l) += g * Pbb(j, k);
                    Kab(i, l) += g * Pab(j, k);
                    Kba(i, l) += g * Pba(j, k);
                }
            }
        }
    }
    CMat F = zeros(nso, nso);
    const CMat haa = block_of(h, 0, 0, n);
    const CMat hab = block_of(h, 0, 1, n);
    const CMat hba = block_of(h, 1, 0, n);
    const CMat hbb = block_of(h, 1, 1, n);
    CMat Faa = zeros(n, n);
    CMat Fab = zeros(n, n);
    CMat Fba = zeros(n, n);
    CMat Fbb = zeros(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            Faa(i, j) = haa(i, j) + J(i, j) - Kaa(i, j);
            Fab(i, j) = hab(i, j) - Kab(i, j);
            Fba(i, j) = hba(i, j) - Kba(i, j);
            Fbb(i, j) = hbb(i, j) + J(i, j) - Kbb(i, j);
        }
    }
    write_block(F, 0, 0, n, Faa);
    write_block(F, 0, 1, n, Fab);
    write_block(F, 1, 0, n, Fba);
    write_block(F, 1, 1, n, Fbb);
    hermitize(F);
    return F;
}

/// hf.make_rdm1 with ghf.get_occ: lowest N spinors, occupation 1,
/// energies rounded to 9 decimals before the stable sort.
CMat make_rdm1(const GenEigen& sol, int n_elec) {
    const int n = static_cast<int>(sol.V.shape(0));
    if (n_elec < 0 || n_elec > n)
        throw std::invalid_argument(
            "complex_ghf: n_electrons " + std::to_string(n_elec)
            + " is outside the spinor count " + std::to_string(n));
    std::vector<int> order(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
        order[static_cast<std::size_t>(i)] = i;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        const double ea =
            std::round(sol.w[static_cast<std::size_t>(a)] * 1e9) / 1e9;
        const double eb =
            std::round(sol.w[static_cast<std::size_t>(b)] * 1e9) / 1e9;
        return ea < eb;
    });
    CMat P = zeros(n, n);
    for (int a = 0; a < n_elec; ++a) {
        const int k = order[static_cast<std::size_t>(a)];
        for (int p = 0; p < n; ++p) {
            const Cplx cp = sol.V(p, k);
            for (int q = 0; q < n; ++q)
                P(p, q) += cp * std::conj(sol.V(q, k));
        }
    }
    hermitize(P);
    return P;
}

} // namespace

CMat zeros(int n, int m) { return xt::zeros<Cplx>({n, m}); }

CMat adjoint(const CMat& A) {
    const int n = static_cast<int>(A.shape(0));
    const int m = static_cast<int>(A.shape(1));
    CMat B = zeros(m, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            B(j, i) = std::conj(A(i, j));
    return B;
}

CMat mul(const CMat& A, const CMat& B) {
    const int n = static_cast<int>(A.shape(0));
    const int k = static_cast<int>(A.shape(1));
    const int m = static_cast<int>(B.shape(1));
    if (static_cast<int>(B.shape(0)) != k)
        throw std::invalid_argument("complex_ghf: matrix product shape");
    CMat C = zeros(n, m);
    for (int i = 0; i < n; ++i) {
        for (int p = 0; p < k; ++p) {
            const Cplx a = A(i, p);
            if (a == Cplx{0, 0})
                continue;
            for (int j = 0; j < m; ++j)
                C(i, j) += a * B(p, j);
        }
    }
    return C;
}

void hermitize(CMat& A) {
    const int n = static_cast<int>(A.shape(0));
    for (int i = 0; i < n; ++i) {
        A(i, i) = Cplx{A(i, i).real(), 0.0};
        for (int j = i + 1; j < n; ++j) {
            const Cplx v = 0.5 * (A(i, j) + std::conj(A(j, i)));
            A(i, j) = v;
            A(j, i) = std::conj(v);
        }
    }
}

double real_trace_ji(const CMat& H, const CMat& P) {
    const int n = static_cast<int>(H.shape(0));
    Cplx acc{0.0, 0.0};
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            acc += H(i, j) * P(j, i);
    return acc.real();
}

GenEigen eigh_generalized(const CMat& H_in, const CMat& M_in, double tol) {
    CMat H = H_in;
    CMat M = M_in;
    hermitize(H);
    hermitize(M);
    const int n = static_cast<int>(H.shape(0));
    std::vector<double> mw;
    CMat MV;
    linalg::eigh_hermitian(M, mw, MV);
    int nkeep = 0;
    for (double ev : mw)
        if (ev > tol)
            ++nkeep;
    if (nkeep != n) {
        double emin = mw.empty() ? 0.0 : mw.front();
        throw std::runtime_error(
            "complex_ghf: metric eigenvalue " + std::to_string(emin)
            + " is at or below " + std::to_string(tol)
            + " — refusing a rank-deficient complex spinor metric");
    }
    CMat X = zeros(n, n);
    for (int k = 0; k < n; ++k) {
        const double inv = 1.0 / std::sqrt(mw[static_cast<std::size_t>(k)]);
        for (int i = 0; i < n; ++i)
            X(i, k) = MV(i, k) * inv;
    }
    CMat Hp = mul(adjoint(X), mul(H, X));
    hermitize(Hp);
    std::vector<double> ew;
    CMat Cp;
    linalg::eigh_hermitian(Hp, ew, Cp);
    GenEigen out;
    out.w = std::move(ew);
    out.V = mul(X, Cp);
    return out;
}

CMat commutator(const CMat& F, const CMat& P, const CMat& S) {
    return mul(mul(F, P), S) - mul(mul(S, P), F);
}

CMat pulay(const std::vector<CMat>& F, const std::vector<CMat>& E) {
    const int m = static_cast<int>(F.size());
    const int n = m + 1;
    std::vector<double> A(static_cast<std::size_t>(n * n), 0.0);
    std::vector<double> rhs(static_cast<std::size_t>(n), 0.0);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j)
            A[static_cast<std::size_t>(i * n + j)] = frobenius_re(E[i], E[j]);
        A[static_cast<std::size_t>(i * n + m)] = 1.0;
        A[static_cast<std::size_t>(m * n + i)] = 1.0;
    }
    rhs[static_cast<std::size_t>(m)] = 1.0;
    if (!solve_dense(A, rhs, n))
        return F.back();
    const int nr = static_cast<int>(F[0].shape(0));
    CMat out = zeros(nr, nr);
    for (int i = 0; i < m; ++i) {
        const double c = rhs[static_cast<std::size_t>(i)];
        for (int p = 0; p < nr; ++p)
            for (int q = 0; q < nr; ++q)
                out(p, q) += c * F[static_cast<std::size_t>(i)](p, q);
    }
    hermitize(out);
    return out;
}

RealMatrix sandwich(const RealMatrix& M, const RealMatrix& c2s) {
    const int nc = static_cast<int>(c2s.shape(0));
    const int ns = static_cast<int>(c2s.shape(1));
    RealMatrix MC = xt::zeros<double>({nc, ns});
    for (int i = 0; i < nc; ++i)
        for (int a = 0; a < ns; ++a) {
            double s = 0.0;
            for (int p = 0; p < nc; ++p)
                s += M(i, p) * c2s(p, a);
            MC(i, a) = s;
        }
    RealMatrix out = xt::zeros<double>({ns, ns});
    for (int a = 0; a < ns; ++a)
        for (int b = 0; b < ns; ++b) {
            double s = 0.0;
            for (int i = 0; i < nc; ++i)
                s += c2s(i, a) * MC(i, b);
            out(a, b) = s;
        }
    return out;
}

ComplexMatrix HermitianDiis::next(const ComplexMatrix& fock,
                                  const ComplexMatrix& error) {
    fock_.push_back(fock);
    error_.push_back(error);
    if (static_cast<int>(fock_.size()) > 8) {
        fock_.erase(fock_.begin());
        error_.erase(error_.begin());
    }
    if (fock_.size() < 2)
        return fock;
    return pulay(fock_, error_);
}

bool TwoStepEnergy::converged(double energy, double tol) {
    const bool close = seen_ && std::abs(energy - previous_) < tol;
    const bool done = close && was_close_;
    was_close_ = close;
    previous_ = energy;
    seen_ = true;
    return done;
}

Spnucsp int1e_spnucsp(const ShellList& shells, const AtomList& atoms) {
    for (const native::Shell& sh : shells.shells) {
        if (sh.l > kMaxLOrbital)
            throw std::runtime_error(
                "complex_ghf: shell l=" + std::to_string(sh.l)
                + " exceeds the pVp buffer (missing kernel "
                  "complex_ghf_ao_lgt3)");
    }
    const int n = shells.n_basis;
    const int natm = static_cast<int>(atoms.atoms.size());
    std::vector<RealMatrix> M(9);
    for (int c = 0; c < 9; ++c)
        M[static_cast<std::size_t>(c)] = xt::zeros<double>({n, n});
    const auto blocks = blocks_from_shells(shells.shells);
    const int nb = static_cast<int>(blocks.size());
    for (int i = 0; i < nb; ++i) {
        for (int j = 0; j < nb; ++j) {
            const auto& A = blocks[static_cast<std::size_t>(i)];
            const auto& B = blocks[static_cast<std::size_t>(j)];
            const int nab = A.n_cart * B.n_cart;
            std::vector<double> buf(static_cast<std::size_t>(9 * nab), 0.0);
            spnucsp_block(A, B, atoms.atoms.data(), natm, buf.data());
            for (int comp = 0; comp < 9; ++comp) {
                for (int ia = 0; ia < A.n_cart; ++ia) {
                    for (int ib = 0; ib < B.n_cart; ++ib) {
                        M[static_cast<std::size_t>(comp)](A.ao_start + ia,
                                                          B.ao_start + ib) =
                            buf[static_cast<std::size_t>(comp * nab
                                                         + ia * B.n_cart + ib)];
                    }
                }
            }
        }
    }
    Spnucsp q;
    // PySCF int1e_spnucsp: (yz−zy, zx−xz, xy−yx, xx+yy+zz).
    q.x = M[1 * 3 + 2] - M[2 * 3 + 1];
    q.y = M[2 * 3 + 0] - M[0 * 3 + 2];
    q.z = M[0 * 3 + 1] - M[1 * 3 + 0];
    q.s = M[0] + M[4] + M[8];
    return q;
}

ComplexMatrix block_diag_real(const RealMatrix& a) {
    const int n = static_cast<int>(a.shape(0));
    CMat B = zeros(2 * n, 2 * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            const Cplx v{a(i, j), 0.0};
            B(i, j) = v;
            B(n + i, n + j) = v;
        }
    }
    return B;
}

ComplexMatrix sigma_dot(const RealMatrix& wx, const RealMatrix& wy,
                        const RealMatrix& wz, const RealMatrix& ws) {
    const int n = static_cast<int>(ws.shape(0));
    if (static_cast<int>(wx.shape(0)) != n || static_cast<int>(wy.shape(0)) != n
        || static_cast<int>(wz.shape(0)) != n)
        throw std::invalid_argument("sigma_dot: quaternion blocks differ");
    CMat W = zeros(2 * n, 2 * n);
    const Cplx I{0.0, 1.0};
    for (int p = 0; p < n; ++p) {
        for (int q = 0; q < n; ++q) {
            const Cplx x{wx(p, q), 0.0};
            const Cplx y{wy(p, q), 0.0};
            const Cplx z{wz(p, q), 0.0};
            const Cplx s{ws(p, q), 0.0};
            // i σx, i σy, i σz, I — pyscf.x2c.x2c._sigma_dot.
            W(p, q) = I * z + s;
            W(p, n + q) = I * x + y;
            W(n + p, q) = I * x - y;
            W(n + p, n + q) = -I * z + s;
        }
    }
    hermitize(W);
    return W;
}

ComplexMatrix x2c1e_get_hcore(const ComplexMatrix& t, const ComplexMatrix& v,
                              const ComplexMatrix& w, const ComplexMatrix& s,
                              double c) {
    const int nso = static_cast<int>(s.shape(0));
    if (c <= 0.0 || !std::isfinite(c))
        throw std::invalid_argument(
            "x2c1e_get_hcore: speed of light must be finite and positive");
    if (static_cast<int>(t.shape(0)) != nso
        || static_cast<int>(v.shape(0)) != nso
        || static_cast<int>(w.shape(0)) != nso)
        throw std::invalid_argument("x2c1e_get_hcore: block shapes differ");
    const int n4 = nso * 2;
    const double inv_4c2 = 0.25 / (c * c);
    const double inv_2c2 = 0.5 / (c * c);
    CMat H = zeros(n4, n4);
    CMat M = zeros(n4, n4);
    for (int i = 0; i < nso; ++i) {
        for (int j = 0; j < nso; ++j) {
            H(i, j) = v(i, j);
            H(i, nso + j) = t(i, j);
            H(nso + i, j) = t(i, j);
            H(nso + i, nso + j) = w(i, j) * inv_4c2 - t(i, j);
            M(i, j) = s(i, j);
            M(nso + i, nso + j) = t(i, j) * inv_2c2;
        }
    }
    // Full-rank metric: the PySCF success path of _x2c1e_get_hcore
    // (eigh of the modified Dirac pair, positive-energy columns).
    const GenEigen dirac = eigh_generalized(H, M, 1e-12);
    if (static_cast<int>(dirac.w.size()) != n4)
        throw std::runtime_error(
            "x2c1e_get_hcore: Dirac spectrum length is not 2 n_so");
    const double split = -c * c;
    if (!(dirac.w[static_cast<std::size_t>(nso - 1)] < split
          && dirac.w[static_cast<std::size_t>(nso)] > split))
        throw std::runtime_error(
            "x2c1e_get_hcore: Dirac spectrum does not split into n_so "
            "negative and n_so positive states about -c^2");
    CMat cl = zeros(nso, nso);
    std::vector<double> epos(static_cast<std::size_t>(nso));
    for (int k = 0; k < nso; ++k) {
        epos[static_cast<std::size_t>(k)] =
            dirac.w[static_cast<std::size_t>(nso + k)];
        for (int i = 0; i < nso; ++i)
            cl(i, k) = dirac.V(i, nso + k);
    }
    // r = (u_kept / sqrt(w)) u_kept† cl† s ; h = r† diag(e) r.
    CMat G = mul(adjoint(cl), mul(s, cl));
    hermitize(G);
    std::vector<double> gw;
    CMat gu;
    linalg::eigh_hermitian(G, gw, gu);
    constexpr double kW = 1e-14;
    std::vector<int> kept;
    kept.reserve(static_cast<std::size_t>(nso));
    for (int k = 0; k < nso; ++k)
        if (gw[static_cast<std::size_t>(k)] > kW)
            kept.push_back(k);
    if (kept.empty())
        throw std::runtime_error(
            "x2c1e_get_hcore: cl† S cl has no eigenvalue above 1e-14");
    const int nk = static_cast<int>(kept.size());
    CMat Uscaled = zeros(nso, nk);
    CMat Udag = zeros(nk, nso);
    for (int col = 0; col < nk; ++col) {
        const int k = kept[static_cast<std::size_t>(col)];
        const double inv = 1.0 / std::sqrt(gw[static_cast<std::size_t>(k)]);
        for (int i = 0; i < nso; ++i) {
            Uscaled(i, col) = gu(i, k) * inv;
            Udag(col, i) = std::conj(gu(i, k));
        }
    }
    const CMat r = mul(mul(Uscaled, Udag), mul(adjoint(cl), s));
    CMat scaled = zeros(nso, nso);
    for (int i = 0; i < nso; ++i)
        for (int j = 0; j < nso; ++j)
            scaled(i, j) = epos[static_cast<std::size_t>(i)] * r(i, j);
    CMat h1 = mul(adjoint(r), scaled);
    hermitize(h1);
    return h1;
}

struct ComplexGhfCycle {
    IntegralWorkspace pack;
    CMat h;
    CMat S;
    CMat P;
    CMat F;
    GenEigen orbitals;
    HermitianDiis diis;
    int n_elec = 0;
    double energy = 0.0;
    double previous = 0.0;
    bool have_previous = false;

    ComplexGhfCycle(const Frame& frame, BasisSet basis, int charge, bool x2c)
        : pack(frame, basis, charge, /*compute_eri=*/true, {}, {},
               /*compute_pvp=*/x2c) {
        if (pack.ERI.size() == 0)
            throw std::runtime_error(
                "complex_ghf: stored dense ERI is required (ghf.get_jk)");
        n_elec = pack.n_electrons;
        const int nao = pack.n_basis;
        if (n_elec <= 0 || n_elec > 2 * nao)
            throw std::invalid_argument(
                "complex_ghf: electron count " + std::to_string(n_elec)
                + " does not fit in " + std::to_string(2 * nao) + " spinors");
        if (x2c) {
            // GHF.x2c1e → SpinOrbitalX2CHelper.get_hcore / _x2c1e_get_hcore.
            AtomList atoms{frame, charge};
            ShellList shells{atoms, basis};
            Spnucsp q = int1e_spnucsp(shells, atoms);
            if (pack.spherical()) {
                q.x = sandwich(q.x, pack.c2s);
                q.y = sandwich(q.y, pack.c2s);
                q.z = sandwich(q.z, pack.c2s);
                q.s = sandwich(q.s, pack.c2s);
            }
            const CMat t = block_diag_real(pack.T);
            const CMat v = block_diag_real(pack.V);
            const CMat s = block_diag_real(pack.S);
            const CMat w = sigma_dot(q.x, q.y, q.z, q.s);
            h = x2c1e_get_hcore(t, v, w, s, kLightSpeedAU);
        } else {
            // GHF.get_hcore: block_diag(T+V, T+V).
            h = block_diag_real(pack.Hcore);
        }
        // GHF.get_ovlp: block_diag(S, S).
        S = block_diag_real(pack.S);
        // init_guess_by_1e: occupy the core Hamiltonian (hf.kernel dm0).
        F = h;
    }
};

void complex_ghf_cycle_delete(ComplexGhfCycle* cycle) { delete cycle; }

void complex_ghf_cycle_fock(ComplexGhfCycle& c) {
    // GHF.get_jk / get_veff: F = h + J − K. energy_elec traces this
    // density, not the one the following eig builds. energy_tot adds Enn.
    c.F = ghf_fock(c.h, c.pack.ERI, c.P);
    CMat vhf = c.F;
    const int n = static_cast<int>(c.h.shape(0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            vhf(i, j) -= c.h(i, j);
    const double e_elec =
        real_trace_ji(c.h, c.P) + 0.5 * real_trace_ji(vhf, c.P);
    c.energy = e_elec + c.pack.V_nuc;
}

void complex_ghf_cycle_mix(ComplexGhfCycle& c) {
    // hf.get_fock with DIIS: extrapolate F on the [F, P] commutator.
    c.F = c.diis.next(c.F, commutator(c.F, c.P, c.S));
}

void complex_ghf_cycle_diagonalize(ComplexGhfCycle& c) {
    // hf.kernel eig.
    c.orbitals = eigh_generalized(c.F, c.S, 1e-8);
}

void complex_ghf_cycle_density(ComplexGhfCycle& c) {
    // GHF.get_occ (occupation 1, energies rounded to 9 decimals) then
    // hf.make_rdm1.
    c.P = make_rdm1(c.orbitals, c.n_elec);
}

ComplexGhfCycleOwner complex_ghf_cycle_open(const Frame& frame, BasisSet basis,
                                            int charge, bool x2c) {
    ComplexGhfCycleOwner own(new ComplexGhfCycle(frame, basis, charge, x2c),
                             &complex_ghf_cycle_delete);
    complex_ghf_cycle_diagonalize(*own);
    complex_ghf_cycle_density(*own);
    return own;
}

double complex_ghf_cycle_energy(const ComplexGhfCycle& c) { return c.energy; }

double complex_ghf_cycle_delta(ComplexGhfCycle& c) {
    const double delta = c.have_previous
                             ? std::abs(c.energy - c.previous)
                             : std::numeric_limits<double>::infinity();
    c.previous = c.energy;
    c.have_previous = true;
    return delta;
}

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV
