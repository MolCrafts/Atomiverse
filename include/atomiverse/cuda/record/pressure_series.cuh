/// @file cuda/record/pressure_series.cuh
/// @brief End-of-step −P* series for one SLLOD run.
///
/// Bound at `end_of_step` after `NvtSllod`. The kinetic tensor is reduced
/// from the velocities after that step's remap. The virial is the force-phase
/// tensor still sitting in `SystemScalars::stress`. Python `run()` cannot
/// sample this by chunks: a new build restarts the strain clock.
#pragma once

#include "atomiverse/cuda/asynchronous/async_bridge.cuh"

#include <fstream>
#include <string>

namespace ATV {
namespace cuda {
namespace md {

class PressureSeries : public AsyncRecorder {
  public:
    PressureSeries(std::string path, int every, double epsilon_ev,
                   int direction);

    std::string name() const override { return "pressure_series"; }
    int frequency() const override { return every_; }

  protected:
    void on_build(const BuildContext& ctx) override;
    bool can_run(const LaunchContext& ctx) const override;
    void freeze_slot(const LaunchContext& ctx, int slot) override;
    void d2h_slot(const LaunchContext& ctx, int slot) override;
    void on_data_ready(const LaunchContext& ctx, int slot) override;
    void on_destroy(const DestroyContext& ctx) override;

  private:
    struct Payload {
        double stress[6];
        double kinetic[6];
        double cell[9];
    };

    std::string path_;
    int every_ = 10;
    double epsilon_ev_ = 0;
    int direction_ = 0;
    double cell0_[9]{};
    Payload* device_[3]{};
    Payload* host_[3]{};
    std::ofstream file_;
};

} // namespace md
} // namespace cuda
} // namespace ATV
