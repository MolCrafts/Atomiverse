/// @file cpu/recorder/orbital_cube_writer.h
/// @brief OrbitalCubeWriter -- Gaussian-cube molecular-orbital dump.
///
/// Reads the latest MO coefficients and shells from a sibling mean-field
/// method via `MeanFieldOutputs`. Evaluates
///   psi_i(r) = sum_mu C_alpha(mu, i) * phi_mu(r)
/// on a uniform Cartesian grid and writes a Gaussian ".cube" file.
///
/// @note Basis-evaluation kernel supports l <= 2 (s/p/d).
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

/// @brief Channel selector for orbital evaluation.
enum class OrbitalSpin { Alpha, Beta };

/// @brief Molecular-orbital cube writer.
class OrbitalCubeWriter : public cpu::Component {
  public:
    /// @brief Construct with automatic bounding-box grid.
    /// @param handle     Shared view of the method's state.
    /// @param filename   Output cube path.
    /// @param mo_index   Zero-based MO index (HOMO = n_occ - 1, LUMO = n_occ).
    /// @param spin       Which spin channel to read from.
    /// @param padding    Bounding-box padding [Bohr].
    /// @param spacing    Isotropic grid spacing [Bohr].
    /// @param stride     Write every N steps.
    OrbitalCubeWriter(scf::MeanFieldOutputs handle, std::string filename,
                      int mo_index, OrbitalSpin spin = OrbitalSpin::Alpha,
                      double padding = 4.0, double spacing = 0.25,
                      int stride = 1);

    /// @brief Construct with an explicit grid.
    OrbitalCubeWriter(scf::MeanFieldOutputs handle, std::string filename,
                      int mo_index, OrbitalSpin spin, CubeGrid grid,
                      int stride = 1);

    std::string name() const override { return "orbital_cube_writer"; }

    void launch(const cpu::LaunchContext& ctx) override;

    /// @brief Number of cube files written.
    int n_writes() const { return n_writes_; }

  private:
    scf::MeanFieldOutputs handle_;
    std::string filename_;
    int mo_index_;
    OrbitalSpin spin_;
    bool auto_grid_;
    CubeGrid grid_{};
    double padding_;
    double spacing_;
    int stride_;
    int n_writes_ = 0;
};

} // namespace recorder
} // namespace cpu
} // namespace ATV
