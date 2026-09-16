/// @file cpu/path/string_method.h
/// @brief Simplified string method (E–Ren–Vanden-Eijnden) skeleton (CPU).
#pragma once

#include <functional>
#include <vector>

#include "atomiverse/cpu/frame/frame.h"

namespace ATV {
namespace cpu {
namespace path {

/// @brief Simplified string method over a chain of 1-particle Frames (2D/3D).
/// No `StringConfig` bag.
class StringMethod {
  public:
    /// @brief Interior + endpoints count (≥ 3).
    int n_images = 12;
    /// @brief Steepest-descent step [coord units].
    double step = 0.01;
    /// @brief Outer iterations.
    int max_iter = 200;
    /// @brief Stop when max |F_perp| < f_tol.
    double f_tol = 0.05;

    /// @param n_images Image count (≥ 3).
    /// @param step Steepest-descent step [coord units].
    /// @param max_iter Outer iterations.
    /// @param f_tol Perp-force stop threshold.
    explicit StringMethod(int n_images = 12, double step = 0.01,
                          int max_iter = 200, double f_tol = 0.05);

    using ForceFn = std::function<void(cpu::Frame&)>;
    void set_force_fn(ForceFn fn);
    void use_muller_brown();
    int run(const cpu::Frame& start, const cpu::Frame& end);

    double max_energy() const { return max_energy_; }
    double energy(int i) const { return energies_.at(static_cast<size_t>(i)); }
    void image(int i, cpu::Frame& out) const;
    bool converged() const { return converged_; }

  private:
    void reparameterize_();
    void evolve_once_();

    ForceFn force_;
    std::vector<cpu::Frame> images_;
    std::vector<double> energies_;
    double max_energy_ = 0;
    bool converged_ = false;
};

} // namespace path
} // namespace cpu
} // namespace ATV
