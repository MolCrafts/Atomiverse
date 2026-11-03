/// @file cuda/record/mrec_recorder.cu
/// @brief cuda::MrecRecorder -- a `cuda::AsyncConsumer` of the driver's shared
///        capture buffer that converts each captured slot to a `cpu::Frame`
///        (rows in tag order) and appends it through `cpu::MrecWriter` on its
///        worker thread. Capture, admission, backpressure and failure
///        propagation are the buffer's (cuda/asynchronous/capture_buffer.cu).

#include "atomiverse/cuda/record/mrec_recorder.cuh"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <exception>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/log.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/core/system_scalars.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/record/mrec.h"
#include "atomiverse/cuda/device/error.cuh"

namespace ATV {
namespace cuda {

struct MrecRecorder::Impl {
    Impl(const std::string& path_, std::uint64_t flush_every, bool durable)
        : writer(path_, flush_every, durable), path(path_) {}

    cpu::MrecWriter writer;
    mutable std::mutex writer_mtx; ///< guards `writer` (worker vs user)
    std::string path;
    std::atomic<std::uint64_t> written{0};
    std::atomic<bool> closed{false}; ///< no capture is admitted any more
    cudaStream_t compute = nullptr;  ///< borrowed, for flush()/close()

    // Worker-side conversion state (the frame, laid out at build).
    int n = 0;
    bool has_type = false;
    bool has_z = false;
    cpu::Frame host_frame{0};
    std::vector<int> order;

    /// @brief Convert one captured slot into `host_frame` (rows in tag
    ///        order, as the CPU recorder writes them).
    void convert(const CaptureView& v) {
        const int rows = v.n_atoms;
        if (rows != n)
            throw std::logic_error(
                "cuda::MrecRecorder: a capture of " + std::to_string(rows)
                + " atoms reached a recorder built for " + std::to_string(n));
        const std::int64_t* tag = v.tag;
        order.resize(static_cast<std::size_t>(n));
        std::iota(order.begin(), order.end(), 0);
        if (!std::is_sorted(tag, tag + n))
            std::stable_sort(order.begin(), order.end(),
                             [&](int a, int b) { return tag[a] < tag[b]; });

        cpu::Frame& f = host_frame;
        StorageReal* x = f.span(kAtom_x).data();
        StorageReal* y = f.span(kAtom_y).data();
        StorageReal* z = f.span(kAtom_z).data();
        StorageReal* vx = f.span(kAtom_vx).data();
        StorageReal* vy = f.span(kAtom_vy).data();
        StorageReal* vz = f.span(kAtom_vz).data();
        AccReal* fx = f.span(kAtom_fx).data();
        AccReal* fy = f.span(kAtom_fy).data();
        AccReal* fz = f.span(kAtom_fz).data();
        StorageReal* q = f.span(kAtom_charge).data();
        StorageReal* m = f.span(kAtom_mass).data();
        int* ty = has_type ? f.span(kAtom_type).data() : nullptr;
        int* zz = has_z ? f.span(kAtom_atomic_number).data() : nullptr;
        TagInt* tg = f.span(kAtom_tag).data();
        auto image = f.span(kAtom_image);
        for (int row = 0; row < n; ++row) {
            const auto s = static_cast<std::size_t>(order[row]);
            // The identity copied when the buffer was allocated belongs to
            // device row s only if the row still holds the same atom.
            if (tag[s] != v.tag_at_build[s])
                throw std::logic_error(
                    "cuda::MrecRecorder: device row " + std::to_string(s)
                    + " now holds tag " + std::to_string(tag[s])
                    + " but held tag " + std::to_string(v.tag_at_build[s])
                    + " at build; rows must not move under the recorder");
            if (ty)
                ty[row] = v.type[s];
            if (zz)
                zz[row] = v.atomic_number[s];
            x[row] = static_cast<StorageReal>(v.x[s]);
            y[row] = static_cast<StorageReal>(v.y[s]);
            z[row] = static_cast<StorageReal>(v.z[s]);
            vx[row] = static_cast<StorageReal>(v.vx[s]);
            vy[row] = static_cast<StorageReal>(v.vy[s]);
            vz[row] = static_cast<StorageReal>(v.vz[s]);
            fx[row] = static_cast<AccReal>(v.fx[s]);
            fy[row] = static_cast<AccReal>(v.fy[s]);
            fz[row] = static_cast<AccReal>(v.fz[s]);
            q[row] = static_cast<StorageReal>(v.charge[s]);
            m[row] = static_cast<StorageReal>(v.mass[s]);
            tg[row] = static_cast<TagInt>(tag[s]);
            for (int d = 0; d < 3; ++d)
                image(row, d) = v.image[3 * s + static_cast<std::size_t>(d)];
        }

        Mat3 cell;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                cell(i, j) = static_cast<AccReal>(v.cell[i * 3 + j]);
        f.box.set_cell(cell);

        const SystemScalars& sc = *v.scalars;
        f.meta.set(kEnergy, sc.energy);
        f.meta.set(kEnergyPair, sc.energy_pair);
        f.meta.set(kEnergyBond, sc.energy_bond);
        f.meta.set(kEnergyAngle, sc.energy_angle);
        f.meta.set(kEnergyDihedral, sc.energy_dihedral);
        f.meta.set(kEnergyImproper, sc.energy_improper);
        f.meta.set(kKinetic, sc.kinetic);
        f.meta.set(kConserved, sc.conserved);
        MetaF64x6 w{}, k{};
        for (int c = 0; c < 6; ++c) {
            w[static_cast<std::size_t>(c)] = sc.stress[c];
            k[static_cast<std::size_t>(c)] = sc.stress_kinetic[c];
        }
        f.meta.set(kStress, w);
        f.meta.set(kStressKinetic, k);
    }
};

// ── MrecRecorder ───────────────────────────────────────────────────────────

MrecRecorder::MrecRecorder(std::string filename, int freq,
                           std::uint64_t flush_every, bool durable)
    : filename_(std::move(filename)), freq_(freq) {
    if (filename_.empty())
        throw std::invalid_argument(
            "MrecRecorder: filename is empty — pass the output *.mrec path");
    if (freq_ < 1)
        throw std::invalid_argument("MrecRecorder: freq must be >= 1 step, got "
                                    + std::to_string(freq_));
    // The writer checks the `.mrec` suffix (std::invalid_argument).
    impl_ = std::make_unique<Impl>(filename_, flush_every, durable);
}

MrecRecorder::MrecRecorder(MrecRecorder&& other) noexcept = default;

MrecRecorder::~MrecRecorder() = default;

std::string MrecRecorder::capture_name() const {
    return "cuda::MrecRecorder: writing '" + filename_ + "'";
}

CaptureColumns MrecRecorder::capture_columns() const {
    CaptureColumns c;
    c.positions = c.velocities = c.forces = c.charges = c.masses = true;
    c.tags = c.images = true;
    return c;
}

void MrecRecorder::set_timestep(double dt_fs) {
    if (!(dt_fs > 0.0) || !std::isfinite(dt_fs))
        throw std::invalid_argument("MrecRecorder: timestep must be finite "
                                    "and > 0 fs, got "
                                    + std::to_string(dt_fs));
    dt_fs_ = dt_fs;
}

void MrecRecorder::require_open() const {
    if (impl_->closed.load(std::memory_order_acquire))
        throw std::logic_error(
            "cuda::MrecRecorder: '" + filename_
            + "' is closed — one recorder records one run; construct a new "
              "MrecRecorder for another");
}

void MrecRecorder::on_build(const BuildContext& ctx) {
    require_open();
    Impl& s = *impl_;
    if (s.writer.closed())
        throw std::logic_error(
            "cuda::MrecRecorder: '" + filename_
            + "' is closed — one recorder records one run; construct a new "
              "MrecRecorder for another");
    const int n = ctx.frame.n_local();
    if (n < 1)
        throw std::invalid_argument("cuda::MrecRecorder: the frame has no "
                                    "atoms");
    s.compute = static_cast<cudaStream_t>(ctx.runtime.stream);
    s.n = n;
    // Each identity column travels exactly when the device frame has it — a
    // coarse-grained run writes `type_id` only, a QM geometry
    // `atomic_number` only — and the host frame drops the column it does not
    // carry, so no zero-filled species ever reaches the store.
    s.has_type = ctx.frame.atoms.type != nullptr;
    s.has_z = ctx.frame.atoms.atomic_number != nullptr;
    s.host_frame = cpu::Frame(n);
    if (!s.has_type)
        s.host_frame.block("atoms").erase(kAtom_type.key);
    if (!s.has_z)
        s.host_frame.block("atoms").erase(kAtom_atomic_number.key);
    s.order.reserve(static_cast<std::size_t>(n));
}

void MrecRecorder::consume(const CaptureView& view) {
    Impl& s = *impl_;
    s.convert(view);
    const bool has_time = dt_fs_ > 0.0;
    const double time =
        has_time ? static_cast<double>(view.step) * dt_fs_ : 0.0;
    append_frame(s.host_frame, static_cast<std::int64_t>(view.step), time,
                 has_time);
    s.written.fetch_add(1, std::memory_order_relaxed);
}

void MrecRecorder::append_frame(const cpu::Frame& frame, std::int64_t step,
                                double time, bool has_time) {
    std::lock_guard<std::mutex> lk(impl_->writer_mtx);
    impl_->writer.append(frame, step, time, has_time);
}

void MrecRecorder::flush() {
    Impl& s = *impl_;
    CaptureBuffer* buffer = capture_buffer();
    if (buffer && buffer->running()) {
        ATV_CUDA_CHECK(cudaStreamSynchronize(s.compute));
        buffer->drain(/*raise=*/true);
    }
    std::lock_guard<std::mutex> lk(s.writer_mtx);
    s.writer.flush();
}

void MrecRecorder::close() { close_(/*raise=*/true); }

void MrecRecorder::close_(bool raise) {
    Impl& s = *impl_;
    CaptureBuffer* buffer = capture_buffer();
    std::exception_ptr device;
    if (buffer && buffer->running()) {
        try {
            ATV_CUDA_CHECK(cudaStreamSynchronize(s.compute));
            buffer->drain(/*raise=*/false); // the run raises its failure
        } catch (...) {
            device = std::current_exception();
        }
    }
    s.closed.store(true, std::memory_order_release);
    std::exception_ptr storage;
    {
        std::lock_guard<std::mutex> lk(s.writer_mtx);
        if (!s.writer.closed()) {
            if (buffer) {
                // The driver's buffer: its accounting, and the run's failure
                // (this recorder's or another consumer's) if it failed.
                const CaptureStats st = buffer->stats();
                cpu::MrecBufferStatus b;
                b.budget_bytes = st.budget_bytes;
                b.frame_bytes = st.frame_bytes;
                b.peak_bytes = st.peak_bytes;
                b.blocked_episodes = st.blocked_episodes;
                b.blocked_seconds = st.blocked_seconds;
                s.writer.set_buffer_status(b);
                if (buffer->failed())
                    s.writer.set_failed(buffer->failure());
            }
            try {
                s.writer.close();
            } catch (...) {
                storage = std::current_exception();
            }
        }
    }
    if (!raise) {
        // Teardown reports, never throws.
        auto log = [&](const std::exception_ptr& e) {
            try {
                std::rethrow_exception(e);
            } catch (const std::exception& ex) {
                ATV_ERROR("cuda::MrecRecorder: closing '{}' at run teardown "
                          "failed: {}",
                          filename_, ex.what());
            }
        };
        if (device)
            log(device);
        if (storage)
            log(storage);
        return;
    }
    if (device)
        std::rethrow_exception(device);
    if (storage)
        std::rethrow_exception(storage);
}

void MrecRecorder::on_destroy(const DestroyContext& ctx) {
    (void)ctx;
    // Reached from the backend teardown and from destructors: report, never
    // throw. `close()` is the throwing door.
    close_(/*raise=*/false);
}

std::uint64_t MrecRecorder::n_frames_written() const {
    return impl_->written.load(std::memory_order_relaxed);
}

std::uint64_t MrecRecorder::n_frames_committed() const {
    std::lock_guard<std::mutex> lk(impl_->writer_mtx);
    return impl_->writer.committed();
}

} // namespace cuda
} // namespace ATV
