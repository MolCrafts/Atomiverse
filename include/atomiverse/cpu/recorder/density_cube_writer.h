/// @file cpu/recorder/density_cube_writer.h
/// @brief DensityCubeWriter -- Gaussian-cube electron-density dump.
///
/// Reads the latest density matrix and shell descriptors from a sibling
/// mean-field method via `MeanFieldOutputs`. Evaluates
///   rho(r) = sum_{mu,nu} [P_alpha + P_beta]_{mu,nu} * phi_mu(r) * phi_nu(r)
/// on a uniform Cartesian grid and writes a Gaussian ".cube" file.
///
/// @note Current basis-evaluation kernel supports l <= 2 (s/p/d). Bases
///       with f or higher angular momentum throw from `basis_eval.h` at
///       the first grid point.
///
/// Usage:
/// @code
///   cpu::scf::RHF scf{};
///   auto handle = scf.result_handle();
///   opt.on(minimizer::Phase::force).add(std::move(scf));
///   opt.on(minimizer::Phase::end_of_step)
///      .add(cpu::recorder::DensityCubeWriter{handle, "density.cube"});
/// @endcode
#pragma once

#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/cpu/component/recorder.h"
#include "atomiverse/cpu/recorder/cube.h"
#include "atomiverse/cpu/scf/mean_field_method.h"

namespace ATV {
namespace cpu {
namespace recorder {

/// @brief Electron-density cube writer.
///
/// Default stride = 0 (every step). Set stride > 0 to thin output.
class DensityCubeWriter : public cpu::Component {
  public:
    /// @brief Construct with automatic bounding-box grid.
    /// @param handle     Shared view of the method's state.
    /// @param filename   Output cube path. Overwritten on each launch.
    /// @param padding    Bounding-box padding [Bohr].
    /// @param spacing    Isotropic grid spacing [Bohr].
    /// @param stride     Write every N steps (0 or 1 = every step).
    DensityCubeWriter(scf::MeanFieldOutputs handle, std::string filename,
                      double padding = 4.0, double spacing = 0.25,
                      int stride = 1);

    /// @brief Construct with an explicit grid.
    DensityCubeWriter(scf::MeanFieldOutputs handle, std::string filename,
                      CubeGrid grid, int stride = 1);

    std::string name() const override { return "density_cube_writer"; }

    void launch(const cpu::LaunchContext& ctx) override;

    /// @brief Number of cube files written.
    int n_writes() const { return n_writes_; }

    /// @brief Integral of the last-written density (Bohr^3-weighted sum).
    /// Useful for verifying normalization against n_electrons.
    double last_integrated_density() const { return last_integral_; }

  private:
    scf::MeanFieldOutputs handle_;
    std::string filename_;
    bool auto_grid_;
    CubeGrid grid_{};
    double padding_;
    double spacing_;
    int stride_;
    int n_writes_ = 0;
    double last_integral_ = 0.0;
};

} // namespace recorder
} // namespace cpu
} // namespace ATV
