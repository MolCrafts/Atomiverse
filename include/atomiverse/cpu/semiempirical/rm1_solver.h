/// @file cpu/semiempirical/rm1_solver.h
/// @brief RM1 concrete method — public members + Frame build/run.
#pragma once

#include <memory>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/internal/nddo_solver.h"

namespace ATV {
namespace cpu {
namespace semiempirical {

using RM1Result = NDDOResult;

/// @brief RM1 concrete semiempirical method (members + run).
class RM1 {
  public:
    int charge = 0;
    int multiplicity = 1;
    int max_iter = 200;
    double density_tol = 1.0e-8;
    double energy_tol = 1.0e-10;

    /// @brief Default parameters (charge 0, singlet, max_iter 200, …).
    RM1();
    /// @brief Construct with SCF parameters assigned immediately.
    /// @param charge Total molecular charge [e].
    /// @param multiplicity Spin multiplicity 2S+1 (dimensionless).
    /// @param max_iter SCF iteration cap (dimensionless).
    /// @param density_tol Density residual tolerance (dimensionless).
    /// @param energy_tol Energy residual tolerance [eV].
    RM1(int charge, int multiplicity, int max_iter = 200,
        double density_tol = 1.0e-8, double energy_tol = 1.0e-10);
    ~RM1();
    RM1(RM1&&) noexcept;
    RM1& operator=(RM1&&) noexcept;
    RM1(const RM1&) = delete;
    RM1& operator=(const RM1&) = delete;

    void build(const cpu::Frame& reference);
    RM1Result& run(const cpu::Frame& frame);
    RM1Result& compute_forces(cpu::Frame& frame, double step_angstrom = 1.0e-4);

    const RM1Result& last_result() const;
    bool built() const noexcept;
    bool has_density() const noexcept;
    const NDDOSystem& system() const;
    const NDDOScfState& state() const;

  private:
    void sync_engine_();
    std::unique_ptr<NDDOSolver> engine_;
};

using RM1Solver = RM1;

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
