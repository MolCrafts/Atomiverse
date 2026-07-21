/// @file cpu/pw/internal/pw_subspace.h
/// @brief Host-side Davidson subspace utilities (real spherical harmonics,
///        small dense Jacobi eigensolver, subspace initialization) shared
///        by the CPU and CUDA plane-wave band solvers.
#pragma once

#include "atomiverse/core/precision.h"
#include "atomiverse/cpu/pw/internal/pw_workspace.h"
#include <complex>
#include <vector>

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {

/// @brief Number of m components for angular momentum l.
inline int n_m_for_l(int l) { return 2 * l + 1; }

/// @brief Real spherical harmonics Y_lm(Ĝ) for l ≤ 2.
/// @param out Array of size 2l+1 (m = -l..l).
void compute_ylm(AccReal x, AccReal y, AccReal z, AccReal r, int l,
                 AccReal* out);

/// @brief Cyclic Jacobi eigensolver for small dense symmetric matrices.
///
/// Destroys `A`; eigenpairs are returned sorted ascending — both band
/// solvers take the first n_bands columns as the occupied bands.
void jacobi_eigh(int n, AccReal* A, AccReal* eig, AccReal* vecs,
                 int max_sweep = 20);

/// @brief Build the initial Davidson subspace for the current k-point.
///
/// Uses atomic wavefunctions (QE startingwfc='atomic+random') when the
/// workspace provides enough of them, otherwise the lowest-T(G) plane
/// waves. `sub_vectors` is resized to n_sub × ws.n_g (row-major).
///
/// @return n_sub, the number of initial subspace vectors.
int init_subspace(const PWWorkspace& ws, int n_bands,
                  std::vector<std::complex<AccReal>>& sub_vectors);

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV
