/// @file cpu/path/dimer.h
/// @brief Henkelman–Jónsson dimer method skeleton (CPU).
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "atomiverse/cpu/frame/frame.h"

namespace ATV {
namespace cpu {
namespace path {

/// @brief Dimer saddle search over one Frame (midpoint = saddle estimate).
/// No `DimerConfig` bag — parameters are constructor args / public members.
class Dimer {
  public:
    /// @brief Half-separation δ [Å] (dimer length = 2δ).
    double delta = 0.01;
    /// @brief Translation step [Å] along modified force.
    double dR = 0.05;
    /// @brief Rotation iterations per translation step.
    int max_rotations = 8;
    /// @brief Stop rotating when |F_perp| < rot_tol [eV/Å].
    double rot_tol = 1e-3;
    /// @brief Converge when |F_mod| < f_tol [eV/Å].
    double f_tol = 0.05;
    /// @brief RNG seed for initial mode N.
    std::uint64_t seed = 1;

    /// @param delta Half-separation δ [Å] (> 0).
    /// @param dR Translation step [Å] (> 0).
    /// @param max_rotations Rotation iterations per translation (≥ 1).
    /// @param rot_tol Rotation force tolerance [eV/Å].
    /// @param f_tol Modified-force convergence [eV/Å].
    /// @param seed Initial mode RNG seed.
    explicit Dimer(double delta = 0.01, double dR = 0.05, int max_rotations = 8,
                   double rot_tol = 1e-3, double f_tol = 0.05,
                   std::uint64_t seed = 1);

    using ForceFn = std::function<void(cpu::Frame&)>;
    void set_force_fn(ForceFn fn);
    void use_inverted_harmonic(double kx = 1.0, double ky = 1.0);
    int run(cpu::Frame& frame, int max_iter);
    bool converged() const { return converged_; }
    double max_force() const { return f_mod_; }

  private:
    void eval_force_(cpu::Frame& f);
    void normalize3_(double& x, double& y, double& z) const;
    void random_mode_(std::uint64_t seed);

    ForceFn force_fn_;
    bool use_test_pot_ = false;
    double kx_ = 1.0, ky_ = 1.0;
    double nx_ = 1, ny_ = 0, nz_ = 0;
    double f_mod_ = 0;
    bool converged_ = false;
};

} // namespace path
} // namespace cpu
} // namespace ATV
