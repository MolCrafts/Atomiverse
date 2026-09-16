/// @file cpu/path/irc.h
/// @brief Intrinsic reaction coordinate (Fukui) skeleton — CPU.
#pragma once

#include <functional>
#include <vector>

#include "atomiverse/cpu/frame/frame.h"

namespace ATV {
namespace cpu {
namespace path {

/// @brief Fukui IRC from a first-order saddle (1-particle / few-atom Frames).
/// No `IrcConfig` bag.
class Irc {
  public:
    /// @brief SD step [coord units].
    double step = 0.01;
    /// @brief Max steps per branch.
    int max_steps = 500;
    /// @brief Stop when |F| < f_tol.
    double f_tol = 1e-3;
    /// @brief Initial displacement along mode [coord].
    double kick = 0.05;

    /// @param step SD step [coord units].
    /// @param max_steps Max steps per branch.
    /// @param f_tol Force magnitude stop threshold.
    /// @param kick Initial displacement along mode [coord].
    explicit Irc(double step = 0.01, int max_steps = 500, double f_tol = 1e-3,
                 double kick = 0.05);

    using ForceFn = std::function<void(cpu::Frame&)>;
    void set_force_fn(ForceFn fn);
    void use_muller_brown();
    void run(const cpu::Frame& saddle, double mode_x = 0.0,
             double mode_y = 1.0);

    double forward_final_energy() const { return E_fwd_; }
    double backward_final_energy() const { return E_bwd_; }
    const std::vector<double>& forward_energies() const { return path_E_fwd_; }
    const std::vector<double>& backward_energies() const { return path_E_bwd_; }

  private:
    void branch_(cpu::Frame f, double mx, double my, bool forward);

    ForceFn force_;
    std::vector<double> path_E_fwd_, path_E_bwd_;
    double E_fwd_ = 0, E_bwd_ = 0;
};

} // namespace path
} // namespace cpu
} // namespace ATV
