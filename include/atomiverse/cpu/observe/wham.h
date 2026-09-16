/// @file cpu/observe/wham.h
/// @brief Thin C++ façade over molrs multi-window WHAM (analysis math in molrs).
///
/// The engine streams CV samples / window histograms; molrs owns the
/// self-consistent free-energy solve (Kumar et al. 1992). No WHAM iteration
/// lives in C++.
#pragma once

#include <memory>
#include <vector>

namespace ATV {
namespace cpu {
namespace compute {

/// @brief Multi-window 1D WHAM free-energy reconstruction (molrs-backed).
///
/// Units: caller chooses β = 1/(k_B T). With β in 1/eV, free energy F(s) is in
/// eV (min shifted to 0). Harmonic window \(w=\frac12 k(s-c)^2\) uses the same
/// energy unit as \(1/\beta\).
class Wham {
  public:
    /// @param beta Inverse temperature β = 1/(k_B T) [1/energy].
    explicit Wham(double beta);
    ~Wham();

    Wham(Wham&&) noexcept;
    Wham& operator=(Wham&&) noexcept;
    Wham(const Wham&) = delete;
    Wham& operator=(const Wham&) = delete;

    /// @brief Solve WHAM from pre-binned window histograms.
    /// @param hist Row-major counts [n_windows * n_bins].
    /// @param n_samples Total samples N_i per window [n_windows].
    /// @param k Force constants [energy / CV²] per window.
    /// @param centers Window centres [CV unit].
    /// @param s_min Histogram lower edge [CV].
    /// @param s_max Histogram upper edge [CV].
    /// @param n_bins Number of bins.
    /// @param max_iter Self-consistency iteration budget.
    /// @param tol Convergence on max|Δf| [energy].
    /// @return Free energy F(s) per bin [energy]; empty on failure.
    std::vector<double> free_energy(const std::vector<double>& hist,
                                    const std::vector<double>& n_samples,
                                    const std::vector<double>& k,
                                    const std::vector<double>& centers,
                                    double s_min, double s_max, int n_bins,
                                    int max_iter = 500, double tol = 1e-10);

    bool last_converged() const;
    int last_iterations() const;
    const std::vector<double>& last_f_windows() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace compute
} // namespace cpu
} // namespace ATV
