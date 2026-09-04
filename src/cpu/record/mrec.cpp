/// @file cpu/record/mrec.cpp
/// @brief MrecWriter -- cpu::Frame → molrs exchange frame → streaming
///        `TrajectoryWriterRef`, one frame per append.

#include "atomiverse/cpu/record/mrec.h"

#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "../io/readable_path.h"
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/log.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/frame/molrs_exchange.h"
#include "atomiverse/molrs.hpp"

namespace ATV {
namespace cpu {

/// @brief The molrs side: one persistent exchange frame (so the static
///        columns stay in place between appends) and the opaque writer handle,
///        absent until the first append mints the store and again after close.
struct MrecWriter::Impl {
    ::rust::Box<molrs::FrameRef> frame;
    std::optional<::rust::Box<molrs::TrajectoryWriterRef>> writer;
    std::vector<std::uint64_t> z_scratch; ///< per-append `atomic_number`
    std::vector<double> mass_scratch;     ///< per-append `mass`

    Impl() : frame(molrs::frame_new()) {}

    /// @brief Push `src` into the exchange frame.
    ///
    /// `cpu::to_molrs` carries the write-back state (x/y/z, v, f, charge,
    /// image, tag, box, serializable meta). `atomic_number` and `mass` are
    /// per-run immutable *inputs* it deliberately leaves alone; a trajectory
    /// store needs them anyway, and molrs pins sparsity per block, not per
    /// column, so they are re-sent with every atoms block rather than once.
    void export_from(const Frame& src) {
        molrs::FrameRef& dst = *frame;
        to_molrs(src, dst);

        const int n = src.nlocal();
        z_scratch.resize(static_cast<std::size_t>(n));
        mass_scratch.resize(static_cast<std::size_t>(n));
        // Hoisted once before the loop -- the 13.7x rule.
        const int* __restrict sp_type = src.span(kAtom_type);
        const StorageReal* __restrict sp_mass = src.span(kAtom_mass);
        for (int i = 0; i < n; ++i) {
            z_scratch[static_cast<std::size_t>(i)] =
                static_cast<std::uint64_t>(sp_type[i]);
            mass_scratch[static_cast<std::size_t>(i)] =
                static_cast<double>(sp_mass[i]);
        }
        molrs::frame_set_column_u32(dst, "atoms", "atomic_number",
                                    ::rust::Slice<const std::uint64_t>(
                                        z_scratch.data(), z_scratch.size()));
        molrs::frame_set_column_f64(
            dst, "atoms", "mass",
            ::rust::Slice<const double>(mass_scratch.data(),
                                        mass_scratch.size()));
    }
};

namespace {

bool ends_with_mrec(const std::string& path) {
    static constexpr std::string_view kSuffix{".mrec"};
    return path.size() > kSuffix.size()
           && path.compare(path.size() - kSuffix.size(), kSuffix.size(),
                           kSuffix)
                  == 0;
}

} // namespace

MrecWriter::MrecWriter(std::string path, std::uint64_t flush_every,
                       bool durable)
    : path_(std::move(path)), flush_every_(flush_every), durable_(durable),
      impl_(std::make_unique<Impl>()) {
    // Loud rather than substituting a suffix: molrs brands a scientific
    // record by its `*.mrec` path (and refuses the retired `.zarr`), and the
    // first append below replaces whatever stands at the path — a guard that
    // only ever deletes a `.mrec` tree is the point of insisting on it here.
    if (!ends_with_mrec(path_))
        throw std::invalid_argument(
            "MrecWriter: output path '" + path_
            + "' must end in .mrec (molrs scientific record store; "
              "e.g. traj.mrec)");
}

MrecWriter::MrecWriter(MrecWriter&&) noexcept = default;

MrecWriter::~MrecWriter() {
    try {
        close();
    } catch (const std::exception& e) {
        ATV_ERROR("cpu::MrecWriter: closing '{}' failed in the destructor "
                  "(frames after the last checkpoint may be lost): {}",
                  path_, e.what());
    }
}

void MrecWriter::append(const Frame& frame, std::int64_t step, double time,
                        bool has_time) {
    if (closed_ || !impl_)
        throw std::logic_error(
            "MrecWriter: '" + path_
            + "' is closed (or moved-from) — one writer records one run; "
              "construct a new MrecWriter (or MrecRecorder) for another");

    internal::with_molrs_context("cpu::MrecWriter::append", path_,
                                 [&] { impl_->export_from(frame); });

    if (!impl_->writer) {
        // A fresh writer starts its own trajectory. molrs refuses to mint
        // over a store that already holds a sequence (never a silent
        // overwrite on its side), so the replacement is explicit here — and
        // bounded to a `*.mrec` path by the constructor check.
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
        if (ec)
            throw std::runtime_error("cpu::MrecWriter::append('" + path_
                                     + "'): cannot replace the existing "
                                       "store: "
                                     + ec.message());
        impl_->writer =
            internal::with_molrs_context("cpu::MrecWriter::append", path_, [&] {
                return molrs::trajectory_writer_create(
                    {path_.data(), path_.size()}, *impl_->frame, flush_every_,
                    durable_);
            });
    }

    internal::with_molrs_context("cpu::MrecWriter::append", path_, [&] {
        molrs::trajectory_writer_append(**impl_->writer, *impl_->frame, step,
                                        time, has_time);
    });
    ++appended_;
}

void MrecWriter::flush() {
    if (!impl_ || !impl_->writer)
        return;
    internal::with_molrs_context("cpu::MrecWriter::flush", path_, [&] {
        molrs::trajectory_writer_flush(**impl_->writer);
    });
}

void MrecWriter::close() {
    if (closed_ || !impl_)
        return;
    closed_ = true;
    if (!impl_->writer)
        return;
    // Take the handle out first: `trajectory_writer_close` consumes the Box,
    // and the writer must read as released even when the commit throws.
    ::rust::Box<molrs::TrajectoryWriterRef> handle = std::move(*impl_->writer);
    impl_->writer.reset();
    internal::with_molrs_context("cpu::MrecWriter::close", path_, [&] {
        molrs::trajectory_writer_close(std::move(handle));
    });
    committed_at_close_ = appended_;
}

std::uint64_t MrecWriter::committed() const {
    if (impl_ && impl_->writer)
        return molrs::trajectory_writer_committed(**impl_->writer);
    return committed_at_close_;
}

} // namespace cpu
} // namespace ATV
