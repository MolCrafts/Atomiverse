/// @file cpu/observe/trajectory_metrics.cpp
/// @brief Trajectory-analysis compute wrappers — thin marshaling to the molrs
/// FFI compute objects and streaming accumulators.
///
/// Bridge always uses f64 (molrs-core hardcodes F = f64). Each batch `compute`
/// instantiates the matching molrs compute object (`molrs::*_compute_new`),
/// calls its `compute` method over the flat C++ buffers (as `rust::Slice`), and
/// copies the returned curve back. The streaming accumulators hold the molrs
/// object behind a PIMPL (`rust::Box`) and feed it one frame per call. No
/// analysis math here.

#include "atomiverse/cpu/observe/trajectory_metrics.h"
#include "atomiverse/molrs.hpp"

#include <cstdint>
#include <stdexcept>

namespace ATV {
namespace cpu {
namespace compute {

std::vector<double> MSD::compute(const std::vector<double>& positions,
                                 int n_frames, int n_dof) const {
    auto obj = molrs::msd_compute_new();
    rust::Vec<double> out = obj->compute({positions.data(), positions.size()},
                                         static_cast<std::int64_t>(n_frames),
                                         static_cast<std::int64_t>(n_dof));
    return std::vector<double>(out.begin(), out.end());
}

EinsteinDiffusion::EinsteinDiffusion(double dt, int dims, double fit_lo,
                                     double fit_hi)
    : dt_(dt), dims_(dims), fit_lo_(fit_lo), fit_hi_(fit_hi) {}

double EinsteinDiffusion::compute(const std::vector<double>& positions,
                                  int n_frames, int n_dof) const {
    auto obj = molrs::diffusion_compute_new(dt_, dims_, fit_lo_, fit_hi_);
    return obj->compute({positions.data(), positions.size()},
                        static_cast<std::int64_t>(n_frames),
                        static_cast<std::int64_t>(n_dof));
}

VACF::VACF(double dt, int resolution) : dt_(dt), resolution_(resolution) {}

std::vector<double> VACF::compute(const std::vector<double>& velocities,
                                  int n_frames, int n_dof) const {
    auto obj =
        molrs::vacf_compute_new(dt_, static_cast<std::int64_t>(resolution_));
    rust::Vec<double> out = obj->compute({velocities.data(), velocities.size()},
                                         static_cast<std::int64_t>(n_frames),
                                         static_cast<std::int64_t>(n_dof));
    return std::vector<double>(out.begin(), out.end());
}

RDF::RDF(int n_bins, double r_max, double r_min)
    : n_bins_(n_bins), r_max_(r_max), r_min_(r_min) {}

std::vector<double> RDF::compute(const std::vector<double>& positions,
                                 const std::vector<double>& boxes, int n_frames,
                                 int n_atoms) const {
    auto obj = molrs::rdf_compute_new(static_cast<std::int64_t>(n_bins_),
                                      r_max_, r_min_);
    rust::Vec<double> out = obj->compute({positions.data(), positions.size()},
                                         {boxes.data(), boxes.size()},
                                         static_cast<std::int64_t>(n_frames),
                                         static_cast<std::int64_t>(n_atoms));
    return std::vector<double>(out.begin(), out.end());
}

// ── Streaming accumulators (PIMPL over molrs rust::Box) ─────────────────────

struct RDFAccumulator::Impl {
    rust::Box<molrs::RdfAccumulator> obj;
};

RDFAccumulator::RDFAccumulator(int n_bins, double r_max, double r_min) {
    if (n_bins < 1)
        throw std::invalid_argument("RDFAccumulator: n_bins must be >= 1");
    if (!(r_max > r_min) || r_min < 0.0)
        throw std::invalid_argument(
            "RDFAccumulator: require 0 <= r_min < r_max");
    impl_ = std::make_unique<Impl>(Impl{molrs::rdf_accumulator_new(
        static_cast<std::int64_t>(n_bins), r_max, r_min)});
}

RDFAccumulator::~RDFAccumulator() = default;
RDFAccumulator::RDFAccumulator(RDFAccumulator&&) noexcept = default;
RDFAccumulator& RDFAccumulator::operator=(RDFAccumulator&&) noexcept = default;

bool RDFAccumulator::accumulate(const std::vector<double>& positions,
                                const std::array<double, 9>& cell) {
    return impl_->obj->accumulate({positions.data(), positions.size()},
                                  {cell.data(), cell.size()});
}

int RDFAccumulator::n_frames() const {
    return static_cast<int>(impl_->obj->n_frames());
}

std::vector<double> RDFAccumulator::finalize() const {
    rust::Vec<double> out = impl_->obj->finalize();
    return std::vector<double>(out.begin(), out.end());
}

struct MSDAccumulator::Impl {
    rust::Box<molrs::MsdAccumulator> obj;
};

MSDAccumulator::MSDAccumulator(int window) {
    if (window < 0)
        throw std::invalid_argument("MSDAccumulator: window must be >= 0");
    impl_ = std::make_unique<Impl>(
        Impl{molrs::msd_accumulator_new(static_cast<std::int64_t>(window))});
}

MSDAccumulator::~MSDAccumulator() = default;
MSDAccumulator::MSDAccumulator(MSDAccumulator&&) noexcept = default;
MSDAccumulator& MSDAccumulator::operator=(MSDAccumulator&&) noexcept = default;

bool MSDAccumulator::accumulate(const std::vector<double>& positions) {
    return impl_->obj->accumulate({positions.data(), positions.size()});
}

int MSDAccumulator::n_frames() const {
    return static_cast<int>(impl_->obj->n_frames());
}

std::vector<double> MSDAccumulator::direct_curve() const {
    rust::Vec<double> out = impl_->obj->direct_curve();
    return std::vector<double>(out.begin(), out.end());
}

double MSDAccumulator::diffusion(double dt, int dims, double fit_lo,
                                 double fit_hi) const {
    return impl_->obj->diffusion(dt, dims, fit_lo, fit_hi);
}

struct VACFAccumulator::Impl {
    rust::Box<molrs::VacfAccumulator> obj;
};

VACFAccumulator::VACFAccumulator(int resolution) {
    if (resolution < 1)
        throw std::invalid_argument("VACFAccumulator: resolution must be >= 1");
    impl_ = std::make_unique<Impl>(Impl{
        molrs::vacf_accumulator_new(static_cast<std::int64_t>(resolution))});
}

VACFAccumulator::~VACFAccumulator() = default;
VACFAccumulator::VACFAccumulator(VACFAccumulator&&) noexcept = default;
VACFAccumulator&
VACFAccumulator::operator=(VACFAccumulator&&) noexcept = default;

bool VACFAccumulator::accumulate(const std::vector<double>& velocities) {
    return impl_->obj->accumulate({velocities.data(), velocities.size()});
}

int VACFAccumulator::n_frames() const {
    return static_cast<int>(impl_->obj->n_frames());
}

std::vector<double> VACFAccumulator::finalize() const {
    rust::Vec<double> out = impl_->obj->finalize();
    return std::vector<double>(out.begin(), out.end());
}

} // namespace compute
} // namespace cpu
} // namespace ATV
