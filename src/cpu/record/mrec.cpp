/// @file cpu/record/mrec.cpp
/// @brief MrecWriter -- cpu::Frame → molpy exchange frame → streaming
///        `MrecWriterRef`, one frame per append.

#include "atomiverse/cpu/record/mrec.h"

#include <chrono>
#include <cstdio>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "../io/molpy_species.h"
#include "../io/readable_path.h"
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/log.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/frame/molpy_exchange.h"
#include "atomiverse/molpy.hpp"

#include <fcntl.h>
#include <unistd.h>

namespace ATV {
namespace cpu {

/// @brief The molpy side: one persistent exchange frame (so the static
///        columns stay in place between appends) and the opaque writer handle,
///        absent until the first append mints the store and again after close.
struct MrecWriter::Impl {
    ::rust::Box<molrs::FrameRef> frame;
    std::optional<::rust::Box<molrs::MrecWriterRef>> writer;
    std::vector<std::uint64_t> z_scratch;    ///< per-append `atomic_number`
    std::vector<std::uint64_t> type_scratch; ///< per-append `type_id`
    std::vector<double> mass_scratch;        ///< per-append `mass`

    Impl() : frame(molrs::frame_new()) {}

    /// @brief Push `src` into the exchange frame.
    ///
    /// `cpu::to_molpy` carries the write-back state (x/y/z, v, f, charge,
    /// image, tag, box, serializable meta). `atomic_number`, `type_id` and
    /// `mass` are per-run immutable *inputs* it deliberately leaves alone; a
    /// trajectory store needs them anyway, and molpy pins sparsity per block,
    /// not per column, so they are re-sent with every atoms block rather than
    /// once.
    void export_from(const Frame& src) {
        molrs::FrameRef& dst = *frame;
        to_molpy(src, dst);

        const int n = src.nlocal();
        // The two identity columns travel when the Frame carries them:
        // species Z as `atomic_number`, the force-field type as `type_id`
        // (a coarse-grained frame has the second only, a QM geometry the
        // first). A negative value is refused, not wrapped.
        if (src.has("atoms", kAtom_atomic_number.key)) {
            internal::molpy_atomic_numbers(src.span(kAtom_atomic_number).data(),
                                           static_cast<std::size_t>(n),
                                           z_scratch, "cpu::MrecWriter");
            molrs::frame_set_column_u64(
                dst, "atoms", "atomic_number",
                ::rust::Slice<const std::uint64_t>(z_scratch.data(),
                                                   z_scratch.size()));
        }
        if (src.has("atoms", kAtom_type.key)) {
            const int* __restrict sp_type = src.span(kAtom_type);
            type_scratch.resize(static_cast<std::size_t>(n));
            for (int i = 0; i < n; ++i) {
                if (sp_type[i] < 0)
                    throw std::invalid_argument(
                        "cpu::MrecWriter: atom " + std::to_string(i)
                        + " has force-field type " + std::to_string(sp_type[i])
                        + "; types are >= 0");
                type_scratch[static_cast<std::size_t>(i)] =
                    static_cast<std::uint64_t>(sp_type[i]);
            }
            molrs::frame_set_column_u64(
                dst, "atoms", "type_id",
                ::rust::Slice<const std::uint64_t>(type_scratch.data(),
                                                   type_scratch.size()));
        }
        mass_scratch.resize(static_cast<std::size_t>(n));
        // Hoisted once before the loop -- the 13.7x rule.
        const StorageReal* __restrict sp_mass = src.span(kAtom_mass);
        for (int i = 0; i < n; ++i)
            mass_scratch[static_cast<std::size_t>(i)] =
                static_cast<double>(sp_mass[i]);
        molrs::frame_set_column_f64(
            dst, "atoms", "mass",
            ::rust::Slice<const double>(mass_scratch.data(),
                                        mass_scratch.size()));
    }
};

namespace {

/// @brief RFC 3339 UTC timestamp of now (`YYYY-MM-DDTHH:MM:SSZ`).
std::string utc_now_rfc3339() {
    const std::time_t t =
        std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm tm{};
    gmtime_r(&t, &tm);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}

/// @brief fsync one path (file or directory); a failure is an error.
void sync_path(const std::filesystem::path& p) {
    const int fd = ::open(p.c_str(), O_RDONLY);
    if (fd < 0 || ::fsync(fd) != 0) {
        if (fd >= 0)
            ::close(fd);
        throw std::runtime_error("cannot sync '" + p.string() + "'");
    }
    ::close(fd);
}

/// @brief @p text as the body of a JSON string (RFC 8259 escapes).
std::string json_escape(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    for (const char ch : text) {
        const auto c = static_cast<unsigned char>(ch);
        switch (ch) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (c < 0x20) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                out += buf;
            } else {
                out += ch;
            }
        }
    }
    return out;
}

/// @brief Write the record's molrec `status` section: a Zarr v3 group at
///        `<store>/status` whose attributes are the status document
///        (molrec spec status.md, reference binding).
void write_status_group(const std::string& store, std::uint64_t written,
                        const std::string& failure,
                        const MrecBufferStatus* buffer, bool durable) {
    namespace fs = std::filesystem;
    const fs::path dir = fs::path(store) / "status";
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec)
        throw std::runtime_error("cannot create '" + dir.string()
                                 + "': " + ec.message());
    std::string attrs = R"({"state":")";
    attrs += failure.empty() ? "succeeded" : "failed";
    attrs += R"(","stage":"simulate","frames_written":)"
             + std::to_string(written) + R"(,"finished_at":")"
             + utc_now_rfc3339() + R"(")";
    if (buffer != nullptr) {
        char seconds[32];
        std::snprintf(seconds, sizeof(seconds), "%.9g",
                      buffer->blocked_seconds);
        attrs += R"(,"buffer":{"budget_bytes":)"
                 + std::to_string(buffer->budget_bytes) + R"(,"frame_bytes":)"
                 + std::to_string(buffer->frame_bytes) + R"(,"peak_bytes":)"
                 + std::to_string(buffer->peak_bytes)
                 + R"(,"blocked_episodes":)"
                 + std::to_string(buffer->blocked_episodes)
                 + R"(,"blocked_seconds":)" + seconds + "}";
    }
    if (!failure.empty()) {
        const std::string msg = json_escape(failure);
        attrs += R"(,"message":")" + msg
                 + R"(","error":{"type":"WriteError","message":")" + msg
                 + R"("})";
    }
    attrs += "}";
    const std::string doc =
        R"({"zarr_format":3,"node_type":"group","attributes":)" + attrs + "}\n";
    const fs::path file = dir / "zarr.json";
    {
        std::ofstream out(file, std::ios::binary | std::ios::trunc);
        out << doc;
        out.flush();
        if (!out)
            throw std::runtime_error("cannot write '" + file.string() + "'");
    }
    if (durable) {
        sync_path(file);
        sync_path(dir);
        sync_path(fs::path(store));
    }
}

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
    // Loud rather than substituting a suffix: molpy brands a scientific
    // record by its `*.mrec` path (and refuses the retired `.zarr`), and the
    // first append below replaces whatever stands at the path — a guard that
    // only ever deletes a `.mrec` tree is the point of insisting on it here.
    if (!ends_with_mrec(path_))
        throw std::invalid_argument(
            "MrecWriter: output path '" + path_
            + "' must end in .mrec (molpy scientific record store; "
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
    try {
        append_(frame, step, time, has_time);
    } catch (const std::exception& e) {
        set_failed(e.what());
        throw;
    }
    ++appended_;
}

void MrecWriter::append_(const Frame& frame, std::int64_t step, double time,
                         bool has_time) {
    internal::with_molpy_context("cpu::MrecWriter::append", path_,
                                 [&] { impl_->export_from(frame); });

    if (!impl_->writer) {
        // A fresh writer starts its own trajectory. molpy refuses to mint
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
            internal::with_molpy_context("cpu::MrecWriter::append", path_, [&] {
                return molrs::mrec_writer_create({path_.data(), path_.size()},
                                                 *impl_->frame, flush_every_,
                                                 durable_);
            });
    }

    internal::with_molpy_context("cpu::MrecWriter::append", path_, [&] {
        molrs::mrec_writer_append(**impl_->writer, *impl_->frame, step, time,
                                  has_time);
    });
}

void MrecWriter::set_failed(std::string message) {
    if (failure_.empty())
        failure_ = message.empty() ? "unspecified failure" : std::move(message);
}

void MrecWriter::flush() {
    if (!impl_ || !impl_->writer)
        return;
    try {
        internal::with_molpy_context("cpu::MrecWriter::flush", path_, [&] {
            molrs::mrec_writer_flush(**impl_->writer);
        });
    } catch (const std::exception& e) {
        set_failed(e.what());
        throw;
    }
}

void MrecWriter::close() {
    if (closed_ || !impl_)
        return;
    closed_ = true;
    if (!impl_->writer)
        return;
    // Take the handle out first: `mrec_writer_close` consumes the Box,
    // and the writer must read as released even when the commit throws.
    ::rust::Box<molrs::MrecWriterRef> handle = std::move(*impl_->writer);
    impl_->writer.reset();
    std::exception_ptr commit_error;
    try {
        internal::with_molpy_context("cpu::MrecWriter::close", path_, [&] {
            molrs::mrec_writer_close(std::move(handle));
        });
        committed_at_close_ = appended_;
    } catch (const std::exception& e) {
        set_failed(e.what());
        commit_error = std::current_exception();
    }
    // The status is written even after a failed final commit, recording it:
    // a store whose run failed says so.
    try {
        write_status_group(path_, appended_, failure_,
                           has_buffer_ ? &buffer_ : nullptr, durable_);
    } catch (const std::exception& e) {
        if (commit_error)
            std::rethrow_exception(commit_error);
        throw std::runtime_error("cpu::MrecWriter::close('" + path_
                                 + "'): the frames are committed but the "
                                   "status section was not written: "
                                 + e.what());
    }
    if (commit_error)
        std::rethrow_exception(commit_error);
}

std::uint64_t MrecWriter::committed() const {
    if (impl_ && impl_->writer)
        return molrs::mrec_writer_committed(**impl_->writer);
    return committed_at_close_;
}

} // namespace cpu
} // namespace ATV
