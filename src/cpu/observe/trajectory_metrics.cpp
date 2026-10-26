/// @file cpu/observe/trajectory_metrics.cpp
/// @brief Trajectory-analysis wrappers — thin marshaling to the molrs FFI
/// analyses and their streaming accumulators.
///
/// Bridge always uses f64 (molrs-core hardcodes F = f64). Each analysis keeps
/// its configuration and builds the matching molrs object (`molrs::msd_new`,
/// `vacf_new`, `linear_fit_new`, `rdf_new`) on demand; the constructor builds
/// one once so a configuration molrs refuses fails at construction. Results and
/// accumulators hold the molrs object behind a PIMPL (`rust::Box`). Every
/// `rust::Error` is re-raised as a std exception naming the ATV operation —
/// the public header is cxx-free, so callers never see `rust::Error`. No
/// analysis math here.

#include "atomiverse/cpu/observe/trajectory_metrics.h"
#include "atomiverse/molpy.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#include "rust/cxx.h"

namespace ATV {
namespace cpu {
namespace compute {

namespace {

/// @brief Run one molrs call, re-raising its `rust::Error` as `E` prefixed
/// with the ATV operation `who`.
template <typename E, typename F>
auto molrs_call(const char* who, F&& f) -> decltype(f()) {
    try {
        return f();
    } catch (const rust::Error& e) {
        throw E(std::string(who) + ": " + e.what());
    }
}

/// @brief A non-negative `int` count as the bridge's `u64`.
std::uint64_t as_u64(int value, const char* who, const char* what) {
    if (value < 0)
        throw std::invalid_argument(std::string(who) + ": " + what
                                    + " must be >= 0, got "
                                    + std::to_string(value));
    return static_cast<std::uint64_t>(value);
}

rust::Slice<const double> slice(const std::vector<double>& v) {
    return {v.data(), v.size()};
}

rust::Slice<const double> slice(const std::array<double, 9>& a) {
    return {a.data(), a.size()};
}

std::vector<double> to_std(const rust::Vec<double>& v) {
    return std::vector<double>(v.begin(), v.end());
}

/// @brief The molrs Msd of a configuration (molrs `Msd::new` + `with_*`).
rust::Box<molrs::Msd> molrs_msd(int max_lag, MsdOrigins origins, int stride) {
    constexpr const char* who = "cpu::compute::MSD";
    if (max_lag < 1)
        throw std::invalid_argument(std::string(who)
                                    + ": max_lag must be >= 1, got "
                                    + std::to_string(max_lag));
    // molrs refuses this pairing only at compute; name it at construction.
    if (origins == MsdOrigins::First && stride != 1)
        throw std::invalid_argument(std::string(who)
                                    + ": a single first origin takes no "
                                      "origin stride (stride must be 1)");
    auto msd = molrs::msd_new(static_cast<std::uint64_t>(max_lag));
    molrs_call<std::invalid_argument>(who, [&] {
        msd->set_origins(origins == MsdOrigins::All ? "all" : "first");
        msd->set_stride(as_u64(stride, who, "stride"));
    });
    return msd;
}

/// @brief The molrs Vacf of a configuration (molrs `Vacf::new` + stride).
rust::Box<molrs::Vacf> molrs_vacf(int max_lag, int stride) {
    constexpr const char* who = "cpu::compute::VACF";
    if (max_lag < 1)
        throw std::invalid_argument(std::string(who)
                                    + ": max_lag must be >= 1, got "
                                    + std::to_string(max_lag));
    auto vacf = molrs::vacf_new(static_cast<std::uint64_t>(max_lag));
    molrs_call<std::invalid_argument>(
        who, [&] { vacf->set_stride(as_u64(stride, who, "stride")); });
    return vacf;
}

rust::Box<molrs::Rdf> molrs_rdf(int n_bins, double r_max, double r_min) {
    constexpr const char* who = "cpu::compute::RDF";
    const std::uint64_t bins = as_u64(n_bins, who, "n_bins");
    return molrs_call<std::invalid_argument>(
        who, [&] { return molrs::rdf_new(bins, r_max, r_min); });
}

RDFResult to_std(const molrs::RdfResult& r) {
    RDFResult out;
    out.bin_edges = to_std(r.bin_edges);
    out.bin_centers = to_std(r.bin_centers);
    out.rdf = to_std(r.rdf);
    out.n_r = to_std(r.n_r);
    out.volume = r.volume;
    out.n_frames = static_cast<int>(r.n_frames);
    return out;
}

} // namespace

// ── TimeCorrelation ─────────────────────────────────────────────────────────

struct TimeCorrelation::Impl {
    rust::Box<molrs::TimeCorrelation> obj;
};

TimeCorrelation::TimeCorrelation(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
TimeCorrelation::~TimeCorrelation() = default;
TimeCorrelation::TimeCorrelation(TimeCorrelation&&) noexcept = default;
TimeCorrelation&
TimeCorrelation::operator=(TimeCorrelation&&) noexcept = default;

std::vector<double> TimeCorrelation::lag_times() const {
    return to_std(impl_->obj->lag_times());
}

std::vector<double> TimeCorrelation::values() const {
    return to_std(
        molrs_call<std::runtime_error>("cpu::compute::TimeCorrelation::values",
                                       [&] { return impl_->obj->channel(0); }));
}

// ── CorrelatorAccumulator ───────────────────────────────────────────────────

struct CorrelatorAccumulator::Impl {
    rust::Box<molrs::CorrelatorAccumulator> obj;
};

CorrelatorAccumulator::CorrelatorAccumulator(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
CorrelatorAccumulator::~CorrelatorAccumulator() = default;
CorrelatorAccumulator::CorrelatorAccumulator(CorrelatorAccumulator&&) noexcept =
    default;
CorrelatorAccumulator&
CorrelatorAccumulator::operator=(CorrelatorAccumulator&&) noexcept = default;

void CorrelatorAccumulator::push(const std::vector<double>& frame, double dt) {
    molrs_call<std::runtime_error>(
        "cpu::compute::CorrelatorAccumulator::push", [&] {
            impl_->obj->push(slice(frame), dt, rust::Slice<const double>{});
        });
}

void CorrelatorAccumulator::push(const std::vector<double>& frame, double dt,
                                 const std::array<double, 9>& cell) {
    molrs_call<std::runtime_error>(
        "cpu::compute::CorrelatorAccumulator::push",
        [&] { impl_->obj->push(slice(frame), dt, slice(cell)); });
}

int CorrelatorAccumulator::n_frames() const {
    return static_cast<int>(impl_->obj->n_frames());
}

TimeCorrelation CorrelatorAccumulator::finish() const {
    auto curve = molrs_call<std::runtime_error>(
        "cpu::compute::CorrelatorAccumulator::finish",
        [&] { return impl_->obj->finish(); });
    return TimeCorrelation(std::make_unique<TimeCorrelation::Impl>(
        TimeCorrelation::Impl{std::move(curve)}));
}

// ── MSD ─────────────────────────────────────────────────────────────────────

MSD::MSD(int max_lag, MsdOrigins origins, int stride)
    : max_lag_(max_lag), origins_(origins), stride_(stride) {
    (void)molrs_msd(max_lag_, origins_, stride_); // validate eagerly
}

CorrelatorAccumulator MSD::accumulator() const {
    auto msd = molrs_msd(max_lag_, origins_, stride_);
    return CorrelatorAccumulator(std::make_unique<CorrelatorAccumulator::Impl>(
        CorrelatorAccumulator::Impl{msd->accumulator()}));
}

// ── VACF ────────────────────────────────────────────────────────────────────

VACF::VACF(int max_lag, int stride) : max_lag_(max_lag), stride_(stride) {
    (void)molrs_vacf(max_lag_, stride_); // validate eagerly
}

CorrelatorAccumulator VACF::accumulator() const {
    auto vacf = molrs_vacf(max_lag_, stride_);
    return CorrelatorAccumulator(std::make_unique<CorrelatorAccumulator::Impl>(
        CorrelatorAccumulator::Impl{vacf->accumulator()}));
}

// ── LinearFit ───────────────────────────────────────────────────────────────

LinearFit::LinearFit(double start, double end) : start_(start), end_(end) {
    (void)molrs_call<std::invalid_argument>("cpu::compute::LinearFit", [&] {
        return molrs::linear_fit_new(start_, end_);
    });
}

LinearFitResult LinearFit::compute(const TimeCorrelation& correlation) const {
    const molrs::LinearFitResult fit =
        molrs_call<std::runtime_error>("cpu::compute::LinearFit::compute", [&] {
            return molrs::linear_fit_new(start_, end_)
                ->compute(*correlation.impl_->obj, /*channel=*/0);
        });
    return LinearFitResult{fit.slope, fit.intercept, fit.r2,
                           static_cast<int>(fit.fit_start),
                           static_cast<int>(fit.fit_end)};
}

// ── RDF ─────────────────────────────────────────────────────────────────────

RDF::RDF(int n_bins, double r_max, double r_min)
    : n_bins_(n_bins), r_max_(r_max), r_min_(r_min) {
    (void)molrs_rdf(n_bins_, r_max_, r_min_); // validate eagerly
}

struct RDFAccumulator::Impl {
    rust::Box<molrs::RdfAccumulator> obj;
};

RDFAccumulator RDF::accumulator() const {
    auto rdf = molrs_rdf(n_bins_, r_max_, r_min_);
    return RDFAccumulator(std::make_unique<RDFAccumulator::Impl>(
        RDFAccumulator::Impl{rdf->accumulator()}));
}

RDFAccumulator::RDFAccumulator(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
RDFAccumulator::~RDFAccumulator() = default;
RDFAccumulator::RDFAccumulator(RDFAccumulator&&) noexcept = default;
RDFAccumulator& RDFAccumulator::operator=(RDFAccumulator&&) noexcept = default;

void RDFAccumulator::push(const std::vector<double>& positions,
                          const std::array<double, 9>& cell) {
    molrs_call<std::runtime_error>("cpu::compute::RDFAccumulator::push", [&] {
        const std::array<bool, 3> periodic{true, true, true};
        impl_->obj->push(slice(positions), slice(cell),
                         {periodic.data(), periodic.size()});
    });
}

int RDFAccumulator::n_frames() const {
    return static_cast<int>(impl_->obj->n_frames());
}

RDFResult RDFAccumulator::finish() const {
    return to_std(
        molrs_call<std::runtime_error>("cpu::compute::RDFAccumulator::finish",
                                       [&] { return impl_->obj->finish(); }));
}

} // namespace compute
} // namespace cpu
} // namespace ATV
