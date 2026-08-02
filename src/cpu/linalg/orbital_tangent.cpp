/// @file src/cpu/linalg/orbital_tangent.cpp
/// @brief Implementation of the S-metric orbital primitives (see header).

#include "atomiverse/cpu/linalg/orbital_tangent.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/core/xmath.hpp>

extern "C" {
// Direct Fortran BLAS binding for the complex matrix product, used by the
// Bloch (complex) twins below. The COMPLEX*16 arrays are spelled `double*`
// (interleaved re/im doubles — Fortran COMPLEX*16 storage, layout-compatible
// with std::complex<double>), the same vocabulary as the zheevd_/dsygvd_
// bindings in cpu/linalg/eigensolve.h; xtensor-blas's bundled cxxblas does not
// declare `zgemm_` at all, so no conflicting declaration can arise.
void zgemm_(const char* transa, const char* transb, const int* m, const int* n,
            const int* k, const double* alpha, const double* a, const int* lda,
            const double* b, const int* ldb, const double* beta, double* c,
            const int* ldc);
}

namespace ATV {
namespace cpu {
namespace linalg {

namespace {

using xt::linalg::dot;
using Cplx = std::complex<double>;

} // namespace

RealMatrix s_tangent(const RealMatrix& C, const RealMatrix& S,
                     const RealMatrix& d) {
    // (I - C C^T S) d  =>  C^T S (result) = 0  (uses C^T S C = I).
    return d - dot(C, dot(xt::transpose(C), dot(S, d)));
}

RealMatrix band_gradient(const RealMatrix& H, const RealMatrix& S,
                         const RealMatrix& C, double occ) {
    // Factor 2*occ: closed-shell occ=2 → 4 H C (P=2CCᵀ); UKS occ=1 → 2 H C.
    const RealMatrix W = (2.0 * occ) * dot(H, C);
    // (I - S C C^T) W
    return W - dot(S, dot(C, dot(xt::transpose(C), W)));
}

RealMatrix lowdin_orthonormalize(const RealMatrix& C, const RealMatrix& S) {
    const RealMatrix G = dot(xt::transpose(C), dot(S, C)); // no x no, SPD
    const RealMatrix Ginvhalf =
        sym_function(G, [](double t) { return 1.0 / std::sqrt(t); });
    return dot(C, Ginvhalf);
}

// ── Complex (Bloch) twins ──────────────────────────────────────────────────

// Row-major complex products via BLAS zgemm. A row-major matrix X [r x c] is
// bitwise the column-major X^T [c x r], so both products are evaluated
// transposed (C^T = op(B)^T op(A)^T) with no copies and no layout flags.
CplxMatrix c_mul(const CplxMatrix& A, const CplxMatrix& B) {
    const int n = static_cast<int>(A.shape(0));
    const int m = static_cast<int>(A.shape(1));
    const int p = static_cast<int>(B.shape(1));
    CplxMatrix C = xt::zeros<Cplx>({n, p});
    if (n == 0 || m == 0 || p == 0)
        return C;
    const Cplx alpha(1.0, 0.0);
    const Cplx beta(0.0, 0.0);
    // Column-major view: C^T [p x n] = B^T [p x m] · A^T [m x n].
    zgemm_("N", "N", &p, &n, &m, reinterpret_cast<const double*>(&alpha),
           reinterpret_cast<const double*>(B.data()), &p,
           reinterpret_cast<const double*>(A.data()), &m,
           reinterpret_cast<const double*>(&beta),
           reinterpret_cast<double*>(C.data()), &p);
    return C;
}

// A† B
CplxMatrix c_dagger_mul(const CplxMatrix& A, const CplxMatrix& B) {
    const int n = static_cast<int>(A.shape(0));
    const int m = static_cast<int>(A.shape(1));
    const int p = static_cast<int>(B.shape(1));
    CplxMatrix C = xt::zeros<Cplx>({m, p});
    if (n == 0 || m == 0 || p == 0)
        return C;
    const Cplx alpha(1.0, 0.0);
    const Cplx beta(0.0, 0.0);
    // Column-major view: C^T [p x m] = B^T [p x n] · conj(A) [n x m], where
    // conj(A) = (A^T)^H is exactly op('C') on the column-major image of A.
    zgemm_("N", "C", &p, &m, &n, reinterpret_cast<const double*>(&alpha),
           reinterpret_cast<const double*>(B.data()), &p,
           reinterpret_cast<const double*>(A.data()), &m,
           reinterpret_cast<const double*>(&beta),
           reinterpret_cast<double*>(C.data()), &p);
    return C;
}

CplxMatrix s_tangent_k(const CplxMatrix& C, const CplxMatrix& S,
                       const CplxMatrix& d) {
    // (I − C C† S) d
    return d - c_mul(C, c_dagger_mul(C, c_mul(S, d)));
}

CplxMatrix band_gradient_k(const CplxMatrix& H, const CplxMatrix& S,
                           const CplxMatrix& C, double occ) {
    const CplxMatrix W = (2.0 * occ) * c_mul(H, C);
    // (I − S C C†) W
    return W - c_mul(S, c_mul(C, c_dagger_mul(C, W)));
}

CplxMatrix band_gradient_k(const CplxMatrix& H, const CplxMatrix& S,
                           const CplxMatrix& C, const std::vector<double>& f) {
    const int nb = static_cast<int>(C.shape(0));
    const int no = static_cast<int>(C.shape(1));
    CplxMatrix HC = c_mul(H, C); // [nb x no]
    // W[:,i] = 2 f_i (H C)[:,i]  — fractional band occupations in [0, 2].
    for (int o = 0; o < no; ++o) {
        const double fo = (o < static_cast<int>(f.size()))
                              ? f[static_cast<std::size_t>(o)]
                              : 0.0;
        const double s = 2.0 * fo;
        for (int mu = 0; mu < nb; ++mu)
            HC(mu, o) *= s;
    }
    return HC - c_mul(S, c_mul(C, c_dagger_mul(C, HC)));
}

CplxMatrix lowdin_orthonormalize_k(const CplxMatrix& C, const CplxMatrix& S) {
    const CplxMatrix G = c_dagger_mul(C, c_mul(S, C));
    const CplxMatrix Ginvhalf = herm_function(G, [](double t) {
        if (!(t > 1e-16))
            throw std::runtime_error(
                "lowdin_orthonormalize_k: non-positive metric eigenvalue");
        return 1.0 / std::sqrt(t);
    });
    return c_mul(C, Ginvhalf);
}

} // namespace linalg
} // namespace cpu
} // namespace ATV
