/// @file cpu/record/density_cube_writer.h
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
#include "atomiverse/cpu/record/cube.h"
#include "atomiverse/cpu/scf/mean_field_method.h"

namespace ATV {
namespace cpu {
namespace recorder {

/// @brief Electron-density cube writer.
///
/// Default stride = 1 (every iteration); 0 is clamped to 1. Bound in a
/// minimization it follows the LAMMPS `minimize` + `dump` step convention:
/// the initial state at setup (step 0), the state after every `stride`-th
/// completed iteration, and the final state when a criterion stops the run
/// early — the file is overwritten each time, so it ends as the final
/// state's. `last_step_written()` names the state in the file.
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

    /// @brief Start of the run (LAMMPS `Output::setup`): the initial state
    ///        as step 0.
    void setup_output(const cpu::LaunchContext& ctx) override;
    /// @brief End of iteration `ctx.step`: step `ctx.step + 1` when on
    ///        stride.
    void launch(const cpu::LaunchContext& ctx) override;
    /// @brief Early stop (LAMMPS `Min::run`): the final state unless that
    ///        step is already written.
    void final_output(const cpu::LaunchContext& ctx) override;

    /// @brief Completed iterations of the state in the file now [count];
    ///        -1 before the first write.
    BigInt last_step_written() const { return last_step_; }

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
    BigInt last_step_ = -1; ///< LAMMPS `last_dump` (rank 0)

    /// @brief Overwrite the cube with the state labelled `step`; rank 0.
    void write_(const cpu::LaunchContext& ctx, BigInt step);
    double last_integral_ = 0.0;
};

} // namespace recorder
} // namespace cpu
} // namespace ATV
