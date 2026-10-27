/// @file cpu/linalg/pulay.cpp
/// @brief Row-major adapter over the public Pulay coefficient solve.

#include "atomiverse/cpu/linalg/pulay_coefficients.h"

#include "atomiverse/cpu/linalg/pulay.h"

#include <exception>

namespace ATV::cpu::linalg {

bool try_pulay_coefficients(const double* gram, int m, double* coefficients) {
    if (m < 1 || gram == nullptr || coefficients == nullptr)
        return false;
    try {
        const auto n = static_cast<std::size_t>(m);
        RealMatrix matrix = xt::zeros<double>({n, n});
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < m; ++j)
                matrix(static_cast<std::size_t>(i),
                       static_cast<std::size_t>(j)) =
                    gram[static_cast<std::size_t>(i * m + j)];
        const RealVector solved = pulay_coefficients(matrix);
        for (int i = 0; i < m; ++i)
            coefficients[i] = solved(static_cast<std::size_t>(i));
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

} // namespace ATV::cpu::linalg
