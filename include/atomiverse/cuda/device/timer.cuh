/// @file cuda/device/timer.cuh
/// @brief cuda::Timer::Impl -- CUDA-event-based GPU profiler implementation.
///
/// Defines `Timer::Impl` (the CUDA-dependent profiler state declared by
/// the CUDA-free `cuda/timer.h` PIMPL facade) and the out-of-line
/// `Timer` special members / methods. CUDA-dependent, internal header --
/// only included by `.cu` translation units (currently `src/cuda/driver/driver.cu`).
///
/// Attach to a `cuda::Backend` via `backend.set_timer(&timer)`. Enabled
/// by `Driver::enable_timing()`, results via `Driver::summary()`.
#pragma once

#include <cuda_runtime.h>

#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "atomiverse/cuda/device/timer.h"

namespace ATV {
namespace cuda {

/// @brief CUDA-dependent profiler state behind the `Timer` PIMPL facade.
///
/// Records start/stop events around component launches. Accumulates
/// elapsed times after each slot sync (zero extra sync overhead).
/// Produces a LAMMPS-style summary table.
class Timer::Impl {
  public:
    /// @brief Allocate CUDA events.
    void init() {
        for (int i = 0; i < kMaxOps; ++i) {
            cudaEventCreate(&starts_[i]);
            cudaEventCreate(&stops_[i]);
        }
    }

    /// @brief Record a start event before a component launch.
    /// @param stream CUDA stream to record the event on.
    void begin(cudaStream_t stream) {
        cudaEventRecord(starts_[pending_], stream);
    }

    /// @brief Record a stop event after a component launch.
    /// @param stream CUDA stream to record the event on.
    /// @param phase Phase index for grouping in the summary.
    /// @param name Component name for display in the summary.
    void end(cudaStream_t stream, int phase, const std::string& name) {
        cudaEventRecord(stops_[pending_], stream);
        pending_phase_[pending_] = phase;
        pending_name_[pending_] = name;
        ++pending_;
    }

    /// @brief Accumulate elapsed times (call after stream sync).
    void flush() {
        for (int k = 0; k < pending_; ++k) {
            float ms;
            cudaEventElapsedTime(&ms, starts_[k], stops_[k]);
            auto key = std::make_pair(pending_phase_[k], pending_name_[k]);
            auto it = index_.find(key);
            if (it == index_.end()) {
                index_[key] = static_cast<int>(entries_.size());
                entries_.push_back({key.first, key.second, 0, 0});
                it = index_.find(key);
            }
            entries_[it->second].total_ms += ms;
            entries_[it->second].calls++;
        }
        pending_ = 0;
    }

    /// @brief Mark one step completed.
    void step_done() { ++total_steps_; }

    /// @brief Format a LAMMPS-style timing summary.
    /// @param phase_names Phase name strings indexed by phase number.
    /// @return Formatted timing table string [no units — text].
    std::string summary(const std::vector<std::string>& phase_names) const {
        if (total_steps_ == 0)
            return "No timing data.\n";

        // Per-phase totals and max phase index
        std::map<int, double> phase_ms;
        int max_phase = -1;
        for (auto& e : entries_) {
            phase_ms[e.phase] += e.total_ms;
            if (e.phase > max_phase)
                max_phase = e.phase;
        }

        double total = 0;
        for (auto& [p, ms] : phase_ms)
            total += ms;

        std::ostringstream os;
        os << std::fixed;

        double sps = (total > 0) ? total_steps_ / (total * 0.001) : 0;
        os << "Performance: " << total_steps_ << " steps, "
           << std::setprecision(1) << total << " ms (" << std::setprecision(0)
           << sps << " steps/s)\n\n";

        os << std::left << std::setw(22) << "Phase" << std::setw(22)
           << "Component" << std::right << std::setw(10) << "Time(ms)"
           << std::setw(8) << "Calls" << std::setw(8) << "%" << "\n";
        os << std::string(70, '-') << "\n";

        // Iterate by phase index — guarantees each phase appears once
        for (int p = 0; p <= max_phase; ++p) {
            auto pit = phase_ms.find(p);
            if (pit == phase_ms.end())
                continue;

            double pms = pit->second;
            double ppct = total > 0 ? 100.0 * pms / total : 0;
            std::string pname = (p < (int)phase_names.size())
                                    ? phase_names[p]
                                    : std::to_string(p);

            os << std::left << std::setw(22) << pname << std::setw(22) << ""
               << std::right << std::setprecision(2) << std::setw(10) << pms
               << std::setw(8) << "" << std::setprecision(1) << std::setw(8)
               << ppct << "\n";

            // Component rows for this phase (insertion order preserved)
            for (auto& e : entries_) {
                if (e.phase != p)
                    continue;
                double cpct = total > 0 ? 100.0 * e.total_ms / total : 0;
                os << std::left << std::setw(22) << "" << std::setw(22)
                   << e.name << std::right << std::setprecision(2)
                   << std::setw(10) << e.total_ms << std::setw(8) << e.calls
                   << std::setprecision(1) << std::setw(8) << cpct << "\n";
            }
        }

        os << std::string(70, '-') << "\n";
        os << std::left << std::setw(22) << "Total" << std::setw(22) << ""
           << std::right << std::setprecision(2) << std::setw(10) << total
           << std::setw(8) << "" << std::setprecision(1) << std::setw(8)
           << 100.0 << "\n";

        return os.str();
    }

    /// @brief Destroy CUDA events and reset state.
    void shutdown() {
        for (int i = 0; i < kMaxOps; ++i) {
            if (starts_[i])
                cudaEventDestroy(starts_[i]);
            if (stops_[i])
                cudaEventDestroy(stops_[i]);
            starts_[i] = nullptr;
            stops_[i] = nullptr;
        }
        entries_.clear();
        index_.clear();
        total_steps_ = 0;
        pending_ = 0;
    }

  private:
    static constexpr int kMaxOps = 16;

    // Per-slot scratch (reset by flush)
    cudaEvent_t starts_[kMaxOps] = {};
    cudaEvent_t stops_[kMaxOps] = {};
    int pending_phase_[kMaxOps] = {};
    std::string pending_name_[kMaxOps];
    int pending_ = 0;

    // Accumulated data (insertion-ordered)
    struct Entry {
        int phase;
        std::string name;
        double total_ms = 0;
        int calls = 0;
    };
    std::vector<Entry> entries_;
    std::map<std::pair<int, std::string>, int> index_;
    int total_steps_ = 0;
};

} // namespace cuda
} // namespace ATV
