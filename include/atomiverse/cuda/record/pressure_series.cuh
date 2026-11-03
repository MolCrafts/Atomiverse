/// @file cuda/record/pressure_series.cuh
/// @brief End-of-step −P* series for one SLLOD run.
///
/// Bound at `end_of_step` after `NvtSllod`; an asynchronous consumer of the
/// driver's shared capture buffer. The kinetic tensor is reduced from the
/// captured velocities and masses (lab frame, after that step's remap); the
/// virial is the force-phase tensor captured in the step's
/// `SystemScalars::stress`. Python `run()` cannot sample this by chunks: a
/// new build restarts the strain clock.
///
/// Step convention (LAMMPS `dump`, @see AsyncConsumer): the `step` column is
/// the completed-step count and the row is the state after that many steps.
/// A run writes the initial state (step 0, strain 0) at setup and then a row
/// after completed steps every, 2·every, ….
#pragma once

#include "atomiverse/cuda/asynchronous/async_consumer.cuh"

#include <fstream>
#include <string>

namespace ATV {
namespace cuda {
namespace md {

class PressureSeries : public AsyncRecorder {
  public:
    PressureSeries(std::string path, int every, double epsilon_ev,
                   int direction);
    PressureSeries(PressureSeries&&) noexcept = default;
    PressureSeries& operator=(PressureSeries&&) noexcept = default;

    std::string name() const override { return "pressure_series"; }
    std::string capture_name() const override {
        return "cuda::md::PressureSeries: writing '" + path_ + "'";
    }
    // Rows are written after every `every_` completed steps, i.e. at step
    // indices every_-1, 2 every_-1, …; the virial is demanded exactly there.
    int frequency() const override { return every_; }
    EvaluationDemand demand() const override {
        return {false, true, every_, every_ - 1};
    }
    /// @brief Velocities and masses (the lab kinetic tensor); the cell and
    ///        the virial come with every capture.
    CaptureColumns capture_columns() const override {
        CaptureColumns c;
        c.velocities = true;
        c.masses = true;
        return c;
    }

  protected:
    void on_build(const BuildContext& ctx) override;
    void consume(const CaptureView& view) override;
    void on_destroy(const DestroyContext& ctx) override;

  private:
    std::string path_;
    int every_ = 10;
    double epsilon_ev_ = 0;
    int direction_ = 0;
    double cell0_[9]{};
    std::ofstream file_;
};

} // namespace md
} // namespace cuda
} // namespace ATV
