/// @file cpu/linalg/pulay_coefficients.h
/// @brief Row-major Pulay coefficient solve, without an xtensor argument.
///
/// Same bordered system as `pulay_coefficients`. Callers that keep their own
/// history (plane-wave density mixing, GPW AO-DIIS) use this entry so a
/// singular subspace is a failed solve rather than a second elimination.
#pragma once

namespace ATV::cpu::linalg {

/// @brief Solve the unit-sum Pulay system for a row-major m-by-m Gram matrix.
/// @param gram         Row-major residual inner products, m*m entries.
/// @param m            History length, m >= 1.
/// @param coefficients Output, m entries. Untouched when this returns false.
/// @return False when the Gram is empty, singular, or the coefficients
///         are nonfinite.
bool try_pulay_coefficients(const double* gram, int m, double* coefficients);

} // namespace ATV::cpu::linalg
