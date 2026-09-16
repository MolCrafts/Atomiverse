/// @file cpu/observe/potential_profile.h
/// @brief PotentialProfile — planar slab electrostatic potential \(\Psi(z)\).
///
/// E10 map: post-process / on-the-fly electrostatic potential profile for
/// classical electrode / EDL slabs. Frame-immutable Observer that bins
/// `kAtom_charge` along \(z\) into \(\rho(z)\) and integrates the planar
/// Poisson equation in MD metal units:
///
/// \[
///   \frac{\mathrm{d} E_z}{\mathrm{d} z}
///     = 4\pi\,k_{\mathrm{e}}\,\rho(z),\qquad
///   E_z = -\frac{\mathrm{d}\Psi}{\mathrm{d} z},
/// \]
///
/// with \(k_{\mathrm{e}}=\texttt{units::coulomb}\) (eV·Å/e²), \(\rho\) [e/Å³],
/// \(E_z\) [V/Å], \(\Psi\) [V]. Gauge: \(\Psi(z_{\min})=0\),
/// \(E_z(z_{\min})=0\) at the left edge of the bin window (ideal neutralized
/// capacitor / vacuum left of the left wall).
///
/// Does **not** solve constant-potential charges — it measures whatever is
/// already on the Frame. Axis is \(z\) only in this foothold.
///
/// Refs: Jackson Classical Electrodynamics (planar Poisson); MetalWalls JOSS
/// 5, 2373 (2020) potential-profile analysis tools; Scalfi et al. PCCP 22,
/// 10480 (2020) EDL profiles.
#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/observer.h"

namespace ATV {
namespace cpu {

/// @brief Planar electrostatic potential profile observer \(\Psi(z)\).
///
/// Construct with bin count and \(z\) window [Å]. Parameters are ctor args —
/// no public Config bag.
class PotentialProfile : public Observer {
  public:
    /// @brief Construct slab potential-profile observer.
    /// @param n_bins Number of bins along \(z\) (must be \(> 0\)).
    /// @param z_min Left edge of the binning window [Å].
    /// @param z_max Right edge of the binning window [Å] (\(> z_{\min}\)).
    /// @throws std::invalid_argument if \(n_{\mathrm{bins}}\le 0\) or
    ///         \(z_{\max}\le z_{\min}\).
    PotentialProfile(int n_bins, double z_min, double z_max)
        : n_bins_(n_bins), z_min_(z_min), z_max_(z_max) {
        if (n_bins_ <= 0)
            throw std::invalid_argument("PotentialProfile: n_bins must be > 0");
        if (!(z_max_ > z_min_))
            throw std::invalid_argument(
                "PotentialProfile: z_max must be > z_min");
        dz_ = (z_max_ - z_min_) / static_cast<double>(n_bins_);
        q_bin_.assign(static_cast<std::size_t>(n_bins_), 0.0);
        rho_.assign(static_cast<std::size_t>(n_bins_), 0.0);
        field_.assign(static_cast<std::size_t>(n_bins_), 0.0);
        psi_.assign(static_cast<std::size_t>(n_bins_), 0.0);
    }

    /// @brief Component identifier.
    /// @return `"potential_profile"`.
    std::string name() const override { return "potential_profile"; }

    /// @brief Default phase: end of MD step (after charge update).
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::end_of_step, +45)};
    }

    /// @brief Bin charges, integrate planar Poisson, publish profiles.
    /// @param ctx Launch context (frame + optional domain decomposition).
    void launch(const LaunchContext& ctx) override;

    /// @brief Number of bins.
    /// @return \(n_{\mathrm{bins}}\).
    int n_bins() const { return n_bins_; }

    /// @brief Left edge of the bin window [Å].
    /// @return \(z_{\min}\).
    double z_min() const { return z_min_; }

    /// @brief Right edge of the bin window [Å].
    /// @return \(z_{\max}\).
    double z_max() const { return z_max_; }

    /// @brief Bin width \(\Delta z = (z_{\max}-z_{\min})/n_{\mathrm{bins}}\) [Å].
    /// @return \(\Delta z\).
    double dz() const { return dz_; }

    /// @brief Bin-center coordinate [Å].
    /// @param i Bin index in \([0, n_{\mathrm{bins}})\).
    /// @return \(z_i = z_{\min} + (i+\tfrac12)\Delta z\).
    double z_center(int i) const {
        return z_min_ + (static_cast<double>(i) + 0.5) * dz_;
    }

    /// @brief Charge density profile \(\rho(z)\) from last launch [e/Å³].
    /// @return Length-\(n_{\mathrm{bins}}\) vector.
    const std::vector<double>& rho() const { return rho_; }

    /// @brief Electric field \(E_z\) at bin centers from last launch [V/Å].
    /// @return Length-\(n_{\mathrm{bins}}\) vector.
    const std::vector<double>& field() const { return field_; }

    /// @brief Electrostatic potential \(\Psi\) at bin centers [V].
    /// @return Length-\(n_{\mathrm{bins}}\) vector.
    const std::vector<double>& psi() const { return psi_; }

    /// @brief Potential drop \(\Psi(z_{\max})-\Psi(z_{\min})\) [V] (right edge
    ///        minus left edge; left edge is the zero gauge).
    /// @return \(\Delta\Psi\) [V] from last launch.
    double potential_drop() const { return potential_drop_; }

  private:
    int n_bins_ = 0;
    double z_min_ = 0.0;
    double z_max_ = 0.0;
    double dz_ = 0.0;
    double potential_drop_ = 0.0;

    std::vector<double> q_bin_;
    std::vector<double> rho_;
    std::vector<double> field_;
    std::vector<double> psi_;
};

} // namespace cpu
} // namespace ATV
