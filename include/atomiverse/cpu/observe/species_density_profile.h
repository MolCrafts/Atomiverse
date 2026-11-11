/// @file cpu/observe/species_density_profile.h
/// @brief SpeciesDensityProfile — per-species number-density profile ρ_s(z).
///
/// EDL production observable (map §8/§9): bins owned atoms of each requested
/// `kAtom_type` along \(z\) into instantaneous and run-mean number densities
/// [Å⁻³]. Companion to `PotentialProfile` (charge-only): the pair resolves an
/// electric double layer into its Ψ(z) response and its per-species layering
/// (cation crowding at the negative electrode, anion at the positive one).
///
/// Frame-immutable Observer; does **not** move atoms or touch charges. Bin
/// volume is `Lx · Ly · dz` from the Frame box at each launch.
///
/// Units: number density [Å⁻³]; window and bin width [Å].
///
/// Refs: MetalWalls EDL diagnostics (JOSS 10.21105/joss.02373); Merlet et
/// al., JPCC 115, 16613 (2011) (graphite + CG IL interfacial profiles).
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

/// @brief Per-species planar number-density profile observer.
///
/// Construct with the species type ids and the \(z\) bin window. Parameters
/// are ctor args — no public Config bag. Bind at MD `end_of_step` (default);
/// each launch overwrites the instantaneous profile and accumulates the
/// run mean.
class SpeciesDensityProfile : public Observer {
  public:
    /// @brief Construct a planar per-species density-profile observer.
    /// @param species_types Atom type ids, one entry per species; must be
    ///        non-empty (e.g. `{cation_type, anion_type}`).
    /// @param n_bins Number of bins along \(z\) (must be \(> 0\)).
    /// @param z_min Left edge of the binning window [Å].
    /// @param z_max Right edge of the binning window [Å] (\(> z_{\min}\)).
    /// @throws std::invalid_argument on empty type list, non-positive bin
    ///         count, or an inverted/empty window.
    SpeciesDensityProfile(std::vector<int> species_types, int n_bins,
                          double z_min, double z_max)
        : types_(std::move(species_types)), n_bins_(n_bins), z_min_(z_min),
          z_max_(z_max) {
        if (types_.empty())
            throw std::invalid_argument(
                "SpeciesDensityProfile: need at least one species type");
        if (n_bins_ <= 0)
            throw std::invalid_argument(
                "SpeciesDensityProfile: n_bins must be > 0");
        if (!(z_max_ > z_min_))
            throw std::invalid_argument(
                "SpeciesDensityProfile: z_max must be > z_min");
        dz_ = (z_max_ - z_min_) / static_cast<double>(n_bins_);
        const std::size_t cells =
            types_.size() * static_cast<std::size_t>(n_bins_);
        density_.assign(cells, 0.0);
        mean_sum_.assign(cells, 0.0);
    }

    /// @brief Component identifier.
    /// @return `"species_density_profile"`.
    std::string name() const override { return "species_density_profile"; }

    /// @brief Default phase: end of MD step (after integration).
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::end_of_step, +46)};
    }

    /// @brief Bin owned atoms per species; refresh instantaneous + mean.
    /// @param ctx Launch context (frame + optional domain decomposition).
    void launch(const LaunchContext& ctx) override;

    /// @brief Number of species (size of the type list).
    std::size_t n_species() const { return types_.size(); }

    /// @brief Atom type id for species `s`.
    /// @param s Species index in \([0, n_{\mathrm{species}})\).
    /// @return Type id.
    int species_type(std::size_t s) const { return types_.at(s); }

    /// @brief Number of bins.
    /// @return \(n_{\mathrm{bins}}\).
    int n_bins() const { return n_bins_; }

    /// @brief Bin width \(\Delta z = (z_{\max}-z_{\min})/n_{\mathrm{bins}}\) [Å].
    /// @return \(\Delta z\).
    double dz() const { return dz_; }

    /// @brief Bin-center coordinate [Å].
    /// @param i Bin index in \([0, n_{\mathrm{bins}})\).
    /// @return \(z_i = z_{\min} + (i+\tfrac12)\Delta z\).
    double z_center(int i) const {
        return z_min_ + (static_cast<double>(i) + 0.5) * dz_;
    }

    /// @brief Instantaneous density profile of species `s` from the last
    ///        launch [Å⁻³], one value per bin.
    /// @param s Species index.
    /// @return Pointer to `n_bins()` contiguous values.
    const double* density(std::size_t s) const {
        return density_.data() + s * static_cast<std::size_t>(n_bins_);
    }

    /// @brief Run-mean density profile of species `s` over all launches
    ///        since construction (or the last `reset()`) [Å⁻³].
    /// @param s Species index.
    /// @param i Bin index.
    /// @return \(\langle\rho_s(z_i)\rangle\); 0 before the first launch.
    double mean_density(std::size_t s, int i) const {
        if (n_samples_ == 0)
            return 0.0;
        return mean_sum_[s * static_cast<std::size_t>(n_bins_)
                         + static_cast<std::size_t>(i)]
               / static_cast<double>(n_samples_);
    }

    /// @brief Number of accumulated launches in the run mean.
    /// @return Sample count.
    long n_samples() const { return n_samples_; }

    /// @brief Atoms of species `s` binned in the last launch (inside window).
    /// @param s Species index.
    /// @return Count of binned atoms.
    long binned_count(std::size_t s) const { return counts_.at(s); }

    /// @brief Clear the run-mean accumulation (instantaneous kept).
    void reset() {
        std::fill(mean_sum_.begin(), mean_sum_.end(), 0.0);
        n_samples_ = 0;
    }

  private:
    std::vector<int> types_;
    int n_bins_;
    double z_min_, z_max_, dz_ = 0.0;

    std::vector<double> density_;  ///< instantaneous [n_species × n_bins]
    std::vector<double> mean_sum_; ///< Σ over launches [n_species × n_bins]
    std::vector<long> counts_ = {};
    long n_samples_ = 0;
};

} // namespace cpu
} // namespace ATV
