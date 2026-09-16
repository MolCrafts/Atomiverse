/// @file cpu/scf/bse.cpp
/// @brief Closed-shell BSE@G0W0 (static dRPA-screened W) on an RHF reference.
///
/// Builds the singlet Casida A/B matrices with G0W0 quasiparticle energy
/// differences and the static direct-RPA screened interaction W from the
/// exact MO ERI tensor (no density fitting). Full BSE uses the Hermitian
/// RPA reduction of tdhf.cpp / g0w0.cpp; TDA diagonalizes A alone.
/// Working equations: PySCF `pyscf.gw.bse` (Hillenbrand, Li & Zhu 2025).

#include "atomiverse/cpu/scf/bse.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/linalg/eigensolve.h"
#include "atomiverse/cpu/scf/dense_matrix.h"
#include "atomiverse/cpu/scf/internal/ao2mo.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/rhf.h"
#include "atomiverse/cpu/scf/slater_determinant.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace {

constexpr double kAmBFloor = 1e-10;
constexpr double kOmega2Tol = 1e-10;

/// Composite OV index matching OrbitalHessian / g0w0: `a * n_occ + i`.
inline int ov(int a, int i, int n_occ) { return a * n_occ + i; }

/// (A−B)^{p} via the shared `linalg::sym_power`, guarding positive-
/// definiteness first: the pre-refactor local sym_power THREW when
/// λ_min(A−B) ≤ kAmBFloor, whereas linalg::sym_power CLAMPS — keep the throw
/// so behavior is unchanged (the power math is bit-identical on a PD basis).
RealMatrix checked_sym_power(const RealMatrix& M, double p, const char* what) {
    auto [s, U] = xt::linalg::eigh(M);
    (void)U;
    if (s(0) <= kAmBFloor)
        throw std::runtime_error(
            std::string(what)
            + ": (A - B) is not positive definite "
              "(lowest eigenvalue = "
            + std::to_string(s(0))
            + " Ha) — BSE RPA reduction undefined (no fallback)");
    return linalg::sym_power(M, p, kAmBFloor, what);
}

/// Symmetrize in place (numerical noise from W assembly).
void symmetrize(RealMatrix& M) {
    const std::size_t n = M.shape()[0];
    for (std::size_t r = 0; r < n; ++r)
        for (std::size_t c = r + 1; c < n; ++c) {
            const double v = 0.5 * (M(r, c) + M(c, r));
            M(r, c) = v;
            M(c, r) = v;
        }
}

} // namespace

BSEResult RHF::bse(int n_states, bool tda) const {
    if (!has_result())
        throw std::runtime_error(
            "RHF::bse: no SCF result — call run(frame) first");

    const SlaterDeterminant& det = determinant();
    if (!det.restricted() || det.generalized())
        throw std::runtime_error(
            "RHF::bse: closed-shell RHF reference required (no fallback)");

    // Spectral G0W0 poles (and HF audit fields) — same path as RHF::g0w0().
    const G0W0Result gw = g0w0();
    if (!gw.converged)
        throw std::runtime_error(
            "RHF::bse: G0W0 quasiparticle Newton did not converge");

    const internal::IntegralView pack = integral_view_();
    const xt::xtensor<double, 4> eri_ao =
        (pack.ERI.size() > 0) ? pack.ERI : pack.eri_working();

    using Spin = SlaterDeterminant::Spin;
    const RealMatrix& C = det.coefficients(Spin::alpha);
    const int n_mo = static_cast<int>(C.shape()[1]);
    const int n_occ = gw.n_occ;
    const int n_vir = n_mo - n_occ;
    if (n_occ < 1 || n_vir < 1)
        throw std::invalid_argument(
            "RHF::bse: need at least one occupied and one virtual orbital "
            "(no fallback)");
    if (static_cast<int>(gw.qp_energy_ha.size()) != n_mo)
        throw std::runtime_error("RHF::bse: QP energy dimension mismatch");

    const int n_ov = n_occ * n_vir;
    if (n_states < 1 || n_states > n_ov)
        throw std::invalid_argument(
            "RHF::bse: n_states = " + std::to_string(n_states)
            + " outside [1, n_ov = " + std::to_string(n_ov) + "]");

    const xt::xtensor<double, 4> eri = ao2mo_full(eri_ao, C, n_mo);

    // QP energies [Hartree].
    RealVector eps = xt::zeros<double>({static_cast<std::size_t>(n_mo)});
    for (int p = 0; p < n_mo; ++p)
        eps(static_cast<std::size_t>(p)) =
            gw.qp_energy_ha[static_cast<std::size_t>(p)];

    // Static dRPA polarizability in the OV space:
    //   Π = 4 (D^{-1} − 4 v)^{-1},  D^{-1}_ia = ε_i − ε_a,  v_ia,jb = (ia|jb)
    RealMatrix S = xt::zeros<double>(
        {static_cast<std::size_t>(n_ov), static_cast<std::size_t>(n_ov)});
    for (int a = 0; a < n_vir; ++a) {
        for (int i = 0; i < n_occ; ++i) {
            const int ia = ov(a, i, n_occ);
            const int aa = n_occ + a;
            S(static_cast<std::size_t>(ia), static_cast<std::size_t>(ia)) =
                eps(static_cast<std::size_t>(i))
                - eps(static_cast<std::size_t>(aa));
            for (int b = 0; b < n_vir; ++b) {
                for (int j = 0; j < n_occ; ++j) {
                    const int jb = ov(b, j, n_occ);
                    const int bb = n_occ + b;
                    const double v = eri(static_cast<std::size_t>(i),
                                         static_cast<std::size_t>(aa),
                                         static_cast<std::size_t>(j),
                                         static_cast<std::size_t>(bb));
                    S(static_cast<std::size_t>(ia),
                      static_cast<std::size_t>(jb)) -= 4.0 * v;
                }
            }
        }
    }
    // Π = 4 S^{-1}
    RealMatrix Pi = 4.0 * xt::linalg::inv(S);

    // W_pqrs = (pq|rs) + Σ_ia,jb (pq|ia) Π_ia,jb (jb|rs)
    auto W_mo = [&](int p, int q, int r, int s) -> double {
        double acc =
            eri(static_cast<std::size_t>(p), static_cast<std::size_t>(q),
                static_cast<std::size_t>(r), static_cast<std::size_t>(s));
        for (int a = 0; a < n_vir; ++a) {
            for (int i = 0; i < n_occ; ++i) {
                const int ia = ov(a, i, n_occ);
                const int aa = n_occ + a;
                const double v_pq_ia = eri(
                    static_cast<std::size_t>(p), static_cast<std::size_t>(q),
                    static_cast<std::size_t>(i), static_cast<std::size_t>(aa));
                for (int b = 0; b < n_vir; ++b) {
                    for (int j = 0; j < n_occ; ++j) {
                        const int jb = ov(b, j, n_occ);
                        const int bb = n_occ + b;
                        const double v_jb_rs = eri(static_cast<std::size_t>(j),
                                                   static_cast<std::size_t>(bb),
                                                   static_cast<std::size_t>(r),
                                                   static_cast<std::size_t>(s));
                        acc += v_pq_ia
                               * Pi(static_cast<std::size_t>(ia),
                                    static_cast<std::size_t>(jb))
                               * v_jb_rs;
                    }
                }
            }
        }
        return acc;
    };

    // Singlet BSE A/B.
    RealMatrix A = xt::zeros<double>(
        {static_cast<std::size_t>(n_ov), static_cast<std::size_t>(n_ov)});
    RealMatrix B = xt::zeros<double>(
        {static_cast<std::size_t>(n_ov), static_cast<std::size_t>(n_ov)});
    for (int a = 0; a < n_vir; ++a) {
        for (int i = 0; i < n_occ; ++i) {
            const int ia = ov(a, i, n_occ);
            const int aa = n_occ + a;
            for (int b = 0; b < n_vir; ++b) {
                for (int j = 0; j < n_occ; ++j) {
                    const int jb = ov(b, j, n_occ);
                    const int bb = n_occ + b;
                    const double v_coul = eri(static_cast<std::size_t>(i),
                                              static_cast<std::size_t>(aa),
                                              static_cast<std::size_t>(j),
                                              static_cast<std::size_t>(bb));
                    // W_ij,ab and W_ib,aj
                    const double W_ijab = W_mo(i, j, aa, bb);
                    const double W_ibaj = W_mo(i, bb, aa, j);
                    double aij = 2.0 * v_coul - W_ijab;
                    if (ia == jb)
                        aij += eps(static_cast<std::size_t>(aa))
                               - eps(static_cast<std::size_t>(i));
                    A(static_cast<std::size_t>(ia),
                      static_cast<std::size_t>(jb)) = aij;
                    B(static_cast<std::size_t>(ia),
                      static_cast<std::size_t>(jb)) = 2.0 * v_coul - W_ibaj;
                }
            }
        }
    }
    symmetrize(A);
    symmetrize(B);

    BSEResult out;
    out.tda = tda;
    out.channel = ExcitationChannel::singlet;
    out.gw = gw;
    out.energy_ev.resize(static_cast<std::size_t>(n_states));

    if (tda) {
        auto [w, U] = xt::linalg::eigh(A);
        for (int s = 0; s < n_states; ++s) {
            if (!(w(static_cast<std::size_t>(s)) > 0.0))
                throw std::runtime_error(
                    "RHF::bse: TDA root " + std::to_string(s)
                    + " is non-positive ("
                    + std::to_string(w(static_cast<std::size_t>(s)))
                    + " Ha) — no fallback");
            out.energy_ev[static_cast<std::size_t>(s)] =
                w(static_cast<std::size_t>(s)) * units::kHartreeInEv;
        }
        return out;
    }

    // Full BSE: Hermitian Casida reduction (same as TDHF/RPA).
    RealMatrix Mp = A + B;
    RealMatrix Mm = A - B;
    symmetrize(Mp);
    symmetrize(Mm);
    const RealMatrix T = checked_sym_power(Mm, 0.5, "RHF::bse");
    RealMatrix M = xt::linalg::dot(T, RealMatrix(xt::linalg::dot(Mp, T)));
    symmetrize(M);
    auto [w2, Z] = xt::linalg::eigh(M);
    (void)Z; // eigenvectors reserved for transition properties later
    for (int s = 0; s < n_states; ++s) {
        const double w2s = w2(static_cast<std::size_t>(s));
        if (w2s < -kOmega2Tol)
            throw std::runtime_error(
                "RHF::bse: omega^2 = " + std::to_string(w2s) + " Ha^2 (root "
                + std::to_string(s)
                + ") is negative — imaginary excitation (no fallback)");
        if (w2s <= kOmega2Tol)
            throw std::runtime_error(
                "RHF::bse: omega^2 = " + std::to_string(w2s) + " Ha^2 (root "
                + std::to_string(s) + ") is numerically zero (no fallback)");
        out.energy_ev[static_cast<std::size_t>(s)] =
            std::sqrt(w2s) * units::kHartreeInEv;
    }
    return out;
}

} // namespace scf
} // namespace cpu
} // namespace ATV
