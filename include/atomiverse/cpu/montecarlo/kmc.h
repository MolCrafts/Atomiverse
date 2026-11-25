/// @file cpu/montecarlo/kmc.h
/// @brief Continuous-time kinetic Monte Carlo driver.
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "atomiverse/cpu/montecarlo/kinetic.h"
#include "atomiverse/cpu/montecarlo/mc.h"

namespace ATV::cpu::mc {

/// Shares the MC run lifecycle. Each step applies a kinetic event and advances
/// physical time. A zero total event rate ends the run before observation.
class KMC : public MC {
  public:
    explicit KMC(std::uint64_t seed = 0);
    ~KMC() override;

    KMC& set_event(std::shared_ptr<Event> event);
    KMC& set_solver(std::shared_ptr<KineticSolver> solver);
    AccReal current_energy() const override;

  protected:
    void require_trial_statistics_() const override;
    void install_defaults() override {}
    void validate_components() const override;
    void prepare_sampling_(cpu::Runtime& runtime, cpu::Frame& frame) override;
    bool advance_step_(cpu::Runtime& runtime, cpu::Frame& frame,
                       BigInt step) override;

  private:
    std::shared_ptr<Event> event_;
    std::shared_ptr<KineticSolver> solver_;
    std::vector<double> event_rates_;
};

} // namespace ATV::cpu::mc
