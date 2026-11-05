/// @file cpu/observe/rdf.h
/// @brief CPU async radial-distribution-function observer (molpy-backed,
/// streaming).
///
/// The engine *measures* (each captured `FrameSnapshot`'s positions + cell are
/// packed into a reusable scratch on the analysis worker thread); molpy
/// *analyzes* (`compute::Rdf` through its `RdfAccumulator`: per-frame periodic
/// cell-list self-query + ideal-gas shell-volume normalization, folded
/// frame-by-frame). Memory is O(n_bins) — never
/// O(trajectory) — so million-step runs stream through at constant footprint.
/// No analysis math and no record type in C++ — both backends' RDF observers
/// stream through the molpy accumulator; the g(r) file format has one emitter
/// (`rdf_write_curve` in observer/rdf_curve.h).
///
/// @see cpu::VACF (observer/vacf.h) for the same streaming async pattern
/// @see .claude/specs/parallel-md-engine.md  § B.4 (async observer pattern)
#pragma once

#include <string>
#include <vector>

#include "atomiverse/cpu/asynchronous/async_observer.h"
#include "atomiverse/cpu/asynchronous/frame_snapshot.h"
#include "atomiverse/cpu/observe/constant_count_gate.h"
#include "atomiverse/cpu/observe/trajectory_metrics.h"

namespace ATV {

/// @brief MD async radial-distribution-function observer (molpy-backed).
///
/// Streams positions + per-frame cell from each captured `FrameSnapshot` into
/// the molpy `RdfAccumulator`; `finalize()` reads the normalized g(r) with its
/// bin centers and writes it to `output_file`.
///
/// Usage:
/// @code{.cpp}
/// cpu::md::MD sim;
/// sim.set_integrator(cpu::md::VelocityVerlet{0.001});
/// sim.on(md::Phase::force).add(cpu::md::NeighborList{6.0}, -10);
/// sim.on(md::Phase::force).add(cpu::PairLJCut{0.0104, 3.4, 6.8});
///
/// RDF rdf{5.5, 200, 100, "rdf.dat"};  // r_max, bins, freq, file
/// sim.set_observer_ref(rdf);                    // explicit borrow; read gr() after
/// sim.build();
/// sim.run(frame, 1'000'000);
/// @endcode
class RDF : public cpu::md::AsyncObserver {
  public:
    /// @param r_max upper edge of the last bin [Å]. @param n_bins radial bin
    /// count. @param freq capture every `freq` steps (0 = every snapshot).
    /// @param output_file g(r) output path.
    /// @throws std::invalid_argument on n_bins < 1 or r_max <= 0 (refused by
    /// molpy `Rdf::new`).
    RDF(double r_max, int n_bins, int freq = 0,
        std::string output_file = "rdf.dat");

    /// @brief This analysis consumes a complete system, not a spatial shard.
    bool requires_full_frame() const override { return true; }
    std::string name() const override { return "rdf_async"; }
    /* role via Observer base */
    int frequency() const override { return freq_; }

    cpu::CaptureMask capture_mask() const override {
        cpu::CaptureMask m;
        m.positions = true;
        return m;
    }

    void on_snapshot(const cpu::FrameSnapshot& snap) override;
    void finalize() override;

    /// @brief Normalized g(r) curve; empty until `finalize()` runs.
    const std::vector<double>& gr() const { return gr_; }
    /// @brief Number of snapshots accumulated.
    int n_samples() const { return n_frames_; }
    const std::string& output_file() const { return output_file_; }

  private:
    int freq_;
    std::string output_file_;

    cpu::ConstantCountGate gate_;      ///< constant-atom-count admission gate
    cpu::RejectedFrames rejected_;     ///< frames the molpy accumulator refused
    std::vector<double> scratch_;      ///< reused blocked x|y|z packing buffer
    cpu::compute::RDFAccumulator acc_; ///< molpy streaming g(r) state
    std::vector<double> gr_; ///< normalized g(r), filled at finalize()
    int n_frames_ = 0;
};

} // namespace ATV
