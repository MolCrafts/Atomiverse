/// @file cpu/semiempirical/internal/scc_mixer.h
/// @brief **PRIVATE WORKSPACE** for the SCC methods — Anderson/Pulay charge
///        mixing. Users never construct this; Frame is the only I/O currency.
///
/// Undamped Picard iteration of the SCC charge map is a period-2 limit cycle
/// for polar systems: the residual never falls below `scc_tol`, the loop runs
/// to `max_scc`, and the "converged" energy is selected by the parity of the
/// iteration budget. Plain linear damping fixes that but needs 50–400
/// iterations, so the shared mixer is Anderson (Pulay/DIIS on the charge
/// residual) with linear damping as its relaxation.
#pragma once

#include <cstddef>
#include <vector>

namespace ATV {
namespace cpu {
namespace semiempirical {
namespace internal {

/// @brief Anderson-mixed fixed-point iteration for an SCC population vector.
///
/// Drives \f$x_{k+1} = \sum_j c_j (x_j + \beta f_j)\f$ where
/// \f$f_j = g(x_j) - x_j\f$ is the charge residual and the coefficients
/// minimize \f$\lVert\sum_j c_j f_j\rVert\f$ under \f$\sum_j c_j = 1\f$
/// (Pulay, *Chem. Phys. Lett.* **73**, 393 (1980)). With one vector in
/// history this is exactly linear damping \f$x + \beta f\f$.
///
/// Reset at each SCC solve so history never leaks between geometries
/// (finite-difference displacements especially). Storage can be retained.
/// The vector is whatever the loop iterates on: Δq alone, or Δq concatenated
/// with the spin population m.
///
/// Units: the mixed vector is Mulliken populations [e]; β is dimensionless.
class ChargeMixer {
  public:
    /// @brief Fix the damping and history depth of one SCC solve.
    /// @param beta Linear relaxation β ∈ (0,1] (dimensionless).
    /// @param depth Maximum number of retained iterates (≥ 1).
    /// @throws std::invalid_argument on β outside (0,1] or depth < 1.
    ChargeMixer(double beta, int depth);

    /// @brief Start an independent solve, retaining compatible storage.
    /// @param size Number of mixed populations; zero leaves the mixer unbound.
    /// @param beta Linear relaxation β in (0,1].
    void reset(std::size_t size, double beta);

    /// @brief Mix one iteration and return the next input vector.
    /// @param x_in Input population vector of this iteration [e].
    /// @param x_out Population vector the SCC map produced from it [e].
    /// @return Retained next-input vector [e], valid until next/reset.
    /// @throws std::invalid_argument on empty or mismatched sizes.
    const std::vector<double>& next(const std::vector<double>& x_in,
                                    const std::vector<double>& x_out);

  private:
    /// @brief Pulay coefficients for the retained residuals.
    /// @return False when the DIIS system is numerically singular.
    bool pulay_coefficients();

    double beta_;
    std::size_t depth_;
    std::vector<std::vector<double>> x_;
    std::vector<std::vector<double>> f_;
    std::size_t first_ = 0, count_ = 0;
    std::vector<double> gram_, rhs_, coefficients_, mixed_;
};

} // namespace internal
} // namespace semiempirical
} // namespace cpu
} // namespace ATV
