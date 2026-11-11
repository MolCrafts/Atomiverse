/// @file cuda/asynchronous/capture_buffer.cu
/// @brief cuda::CaptureBuffer (pinned host-mapped slots + capture kernels)
///        and cuda::AsyncConsumer (host admission, eager / graph-captured
///        launch, worker-side delivery). Ledger semantics in
///        core/capture_ledger.cpp; iron law 10 in
///        .claude/notes/cuda-only-no-dh.md.

#include "atomiverse/cuda/asynchronous/capture_buffer.cuh"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda/atomic>
#include <cuda_runtime.h>

#include "atomiverse/cuda/asynchronous/async_consumer.cuh"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/frame/precision.cuh"
#include "atomiverse/cuda/frame/types.cuh"

namespace ATV {
namespace cuda {

namespace {

using u64 = unsigned long long;

/// @brief Device-resident producer state.
struct Control {
    u64 captures = 0;    ///< Captures opened so far.
    u64 slot = 0;        ///< Slot of the newest capture.
    long long last = -1; ///< Label of the newest capture.
    int fresh = 0;       ///< 1: this launch opened the capture (packs it).
};

/// @brief Frozen per-frame scalars at the head of every slot.
struct SlotHeader {
    long long step = 0;
    double cell[9] = {};
    SystemScalars scalars{};
};

/// @brief Byte offsets of one slot (identical for every slot); 0 = absent
///        for the per-atom columns (the header is at 0).
struct Layout {
    std::size_t x = 0, v = 0, f = 0, q = 0, m = 0, tag = 0, image = 0;
    std::size_t payload = 0; ///< Start of the payload blocks.
    std::size_t bytes = 0;   ///< Slot stride (256-byte aligned).
};

constexpr std::size_t kAlign = 256;
std::size_t align_up(std::size_t v) { return (v + kAlign - 1) & ~(kAlign - 1); }

Layout make_layout(int n, CaptureColumns c, std::uint64_t payload) {
    const auto un = static_cast<std::size_t>(std::max(n, 0));
    Layout L;
    std::size_t at = align_up(sizeof(SlotHeader));
    auto take = [&](bool on, std::size_t bytes) {
        if (!on)
            return std::size_t{0};
        const std::size_t off = at;
        at = align_up(at + bytes);
        return off;
    };
    L.x = take(c.positions, 3 * un * sizeof(double));
    L.v = take(c.velocities, 3 * un * sizeof(double));
    L.f = take(c.forces, 3 * un * sizeof(double));
    L.q = take(c.charges, un * sizeof(double));
    L.m = take(c.masses, un * sizeof(double));
    L.tag = take(c.tags, un * sizeof(std::int64_t));
    L.image = take(c.images, 3 * un * sizeof(ImageInt));
    L.payload = at;
    L.bytes = align_up(at + static_cast<std::size_t>(payload));
    return L;
}

/// Completion words, one per consumer, each on its own cache line.
struct alignas(64) Word {
    u64 value = 0;
};

// ── Kernels (compute stream; captured into the step graph) ─────────────────

/// @brief Label the capture; the first launch of a label opens it.
///
/// The label is the completed-step count: `*step + offset` (in-loop, the
/// device step + 1) or `offset` (eager). In-loop labels strictly increase,
/// so a launch whose label equals the newest capture's belongs to a consumer
/// due on the same step and reuses its slot; an eager launch the host just
/// opened passes `force`. Every opened capture was admitted by the host
/// against the budget, so slot `captures % capacity` is free.
__global__ void acquire_k(Control* c, const long long* step, long long offset,
                          int force, u64 capacity) {
    const long long label = (step ? *step : 0) + offset;
    if (force || label != c->last) {
        c->slot = c->captures % capacity;
        c->captures += 1;
        c->last = label;
        c->fresh = 1;
    } else {
        c->fresh = 0;
    }
}

/// @brief Freeze this step's state into the newest capture's slot (only
///        the launch that opened it).
__global__ void pack_k(const Control* __restrict__ c,
                       unsigned char* __restrict__ buffer, Layout L,
                       Frame::Atoms a, const geom_t* __restrict__ cell,
                       const SystemScalars* __restrict__ scalars, int n) {
    if (!c->fresh)
        return;
    unsigned char* base = buffer + static_cast<std::size_t>(c->slot) * L.bytes;
    const auto un = static_cast<std::size_t>(n);
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) {
        auto* h = reinterpret_cast<SlotHeader*>(base);
        h->step = c->last;
#pragma unroll
        for (int k = 0; k < 9; ++k)
            h->cell[k] = static_cast<double>(cell[k]);
        h->scalars = *scalars;
    }
    if (i < n) {
        if (L.x) {
            auto* r = reinterpret_cast<double*>(base + L.x);
            r[i] = static_cast<double>(load_position(a.x, a.x_corr, i));
            r[un + i] = static_cast<double>(load_position(a.y, a.y_corr, i));
            r[2 * un + i] =
                static_cast<double>(load_position(a.z, a.z_corr, i));
        }
        if (L.v) {
            auto* r = reinterpret_cast<double*>(base + L.v);
            r[i] = static_cast<double>(a.vx[i]);
            r[un + i] = static_cast<double>(a.vy[i]);
            r[2 * un + i] = static_cast<double>(a.vz[i]);
        }
        if (L.f) {
            auto* r = reinterpret_cast<double*>(base + L.f);
            r[i] = decode_force<double>(a.fx[i]);
            r[un + i] = decode_force<double>(a.fy[i]);
            r[2 * un + i] = decode_force<double>(a.fz[i]);
        }
        if (L.q)
            reinterpret_cast<double*>(base + L.q)[i] =
                a.charge ? static_cast<double>(a.charge[i]) : 0.0;
        if (L.m)
            reinterpret_cast<double*>(base + L.m)[i] =
                static_cast<double>(a.mass[i]);
        if (L.tag)
            reinterpret_cast<std::int64_t*>(base + L.tag)[i] =
                a.tag ? a.tag[i] : static_cast<std::int64_t>(i + 1);
        if (L.image) {
            auto* img = reinterpret_cast<ImageInt*>(base + L.image);
#pragma unroll
            for (int d = 0; d < 3; ++d)
                img[3 * i + d] = a.image ? a.image[3 * i + d] : ImageInt{0};
        }
    }
    // The slot is host memory: this thread's writes become visible
    // system-wide before a publish_k later in stream order releases them.
    __threadfence_system();
}

/// @brief Copy a consumer's payload block into the newest capture's slot.
__global__ void payload_k(const Control* __restrict__ c,
                          unsigned char* __restrict__ buffer, u64 stride,
                          u64 offset, const unsigned char* __restrict__ src,
                          u64 bytes) {
    unsigned char* dst = buffer + c->slot * stride + offset;
    for (u64 k = blockIdx.x * static_cast<u64>(blockDim.x) + threadIdx.x;
         k < bytes; k += static_cast<u64>(gridDim.x) * blockDim.x)
        dst[k] = src[k];
    __threadfence_system();
}

/// @brief This consumer's capture is complete: publish the capture count.
__global__ void publish_k(const Control* c, u64* word) {
    __threadfence_system();
    ::cuda::atomic_ref<u64, ::cuda::thread_scope_system> w(*word);
    w.store(c->captures, ::cuda::memory_order_release);
}

u64 load_acquire(u64& word) {
    return std::atomic_ref<u64>(word).load(std::memory_order_acquire);
}

} // namespace

// ── Impl ───────────────────────────────────────────────────────────────────

struct CaptureBuffer::Impl {
    int n = 0;
    Layout layout{};
    u64 capacity = 0;
    std::vector<std::uint64_t> payload_offset; ///< by consumer id
    std::vector<std::uint64_t> payload_bytes;  ///< by consumer id
    cudaStream_t compute = nullptr;
    Control* control = nullptr;          ///< device
    unsigned char* buffer = nullptr;     ///< pinned, mapped
    unsigned char* buffer_dev = nullptr; ///< device alias
    Word* words = nullptr;               ///< pinned, mapped
    Word* words_dev = nullptr;           ///< device alias
    std::vector<std::int64_t> tag;       ///< identity at allocation
    std::vector<int> type, atomic_number;
    bool has_type = false, has_z = false;

    void free_device() noexcept {
        if (buffer)
            cudaFreeHost(buffer);
        if (words)
            cudaFreeHost(words);
        if (control)
            cudaFree(control);
        buffer = buffer_dev = nullptr;
        words = words_dev = nullptr;
        control = nullptr;
    }
};

CaptureBuffer::CaptureBuffer()
    : ledger_("cuda::CaptureBuffer"), impl_(std::make_unique<Impl>()) {}

CaptureBuffer::~CaptureBuffer() { release(); }

std::uint64_t CaptureBuffer::frame_bytes(int n_atoms, CaptureColumns columns,
                                         std::uint64_t payload_bytes) {
    if (n_atoms < 1)
        throw std::invalid_argument(
            "cuda::CaptureBuffer::frame_bytes: n_atoms must be >= 1, got "
            + std::to_string(n_atoms));
    return make_layout(n_atoms, columns, payload_bytes).bytes;
}

void CaptureBuffer::set_budget(std::uint64_t bytes) {
    if (bytes == 0)
        throw std::invalid_argument(
            "capture_buffer_bytes must be >= 1 byte, got 0");
    if (bytes != budget_)
        dirty_ = true;
    budget_ = bytes;
}

void CaptureBuffer::subscribe(AsyncConsumer& consumer) {
    if (ledger_.id_of(consumer) >= 0)
        return;
    stop();
    ledger_.subscribe(consumer);
    dirty_ = true;
}

void CaptureBuffer::unsubscribe(AsyncConsumer& consumer) {
    if (ledger_.id_of(consumer) < 0)
        return;
    stop();
    ledger_.unsubscribe(consumer);
    dirty_ = true;
}

bool CaptureBuffer::has_consumers() const {
    for (auto* s : ledger_.sinks())
        if (s)
            return true;
    return false;
}

void CaptureBuffer::stop() noexcept {
    if (!ledger_.running())
        return;
    // Every enqueued capture must be complete before the drain can trust
    // the completion words (teardown may synchronize).
    if (impl_->compute)
        cudaStreamSynchronize(impl_->compute);
    try {
        ledger_.drain(/*raise=*/false); // a failure stays pending
    } catch (...) {
    }
    ledger_.stop();
}

void CaptureBuffer::release() noexcept {
    stop();
    impl_->free_device();
    impl_->compute = nullptr;
    dirty_ = true;
}

void CaptureBuffer::ensure(const BuildContext& ctx) {
    if (!has_consumers()) {
        stop();
        return;
    }
    if (ledger_.running() && !dirty_)
        return;
    stop();
    Impl& s = *impl_;
    const int n = ctx.frame.n_local();
    if (n < 1 || ctx.frame.box == nullptr || ctx.frame.scalars == nullptr)
        throw std::invalid_argument(
            "cuda::CaptureBuffer: the frame has no atoms, cell or scalars");
    CaptureColumns columns{};
    const auto& sinks = ledger_.sinks();
    s.payload_offset.assign(sinks.size(), 0);
    s.payload_bytes.assign(sinks.size(), 0);
    std::uint64_t payload = 0;
    for (std::size_t id = 0; id < sinks.size(); ++id) {
        if (!sinks[id])
            continue;
        auto* consumer = static_cast<AsyncConsumer*>(sinks[id]);
        columns = columns | consumer->capture_columns();
        const std::uint64_t bytes = consumer->payload_bytes(ctx);
        s.payload_offset[id] = payload;
        s.payload_bytes[id] = bytes;
        payload = align_up(static_cast<std::size_t>(payload + bytes));
    }
    columns.types = false; // static identity: copied once below
    const Layout layout = make_layout(n, columns, payload);
    const std::string what = std::to_string(n) + " atoms";
    const u64 capacity = CaptureLedger::capacity_for(
        "cuda::CaptureBuffer", budget_, layout.bytes, what);

    s.free_device();
    s.compute = static_cast<cudaStream_t>(ctx.runtime.stream);
    try {
        // The whole buffer, once: pinned host memory the device writes
        // directly, so a captured frame has exactly one copy in flight.
        ATV_CUDA_CHECK(
            cudaHostAlloc(reinterpret_cast<void**>(&s.buffer),
                          static_cast<std::size_t>(capacity) * layout.bytes,
                          cudaHostAllocMapped));
        ATV_CUDA_CHECK(cudaHostGetDevicePointer(
            reinterpret_cast<void**>(&s.buffer_dev), s.buffer, 0));
        ATV_CUDA_CHECK(cudaHostAlloc(
            reinterpret_cast<void**>(&s.words),
            CaptureLedger::kMaxConsumers * sizeof(Word), cudaHostAllocMapped));
        for (int k = 0; k < CaptureLedger::kMaxConsumers; ++k)
            s.words[k] = Word{};
        ATV_CUDA_CHECK(cudaHostGetDevicePointer(
            reinterpret_cast<void**>(&s.words_dev), s.words, 0));
        ATV_CUDA_CHECK(cudaMalloc(&s.control, sizeof(Control)));
        const Control fresh{};
        ATV_CUDA_CHECK(cudaMemcpyAsync(s.control, &fresh, sizeof(Control),
                                       cudaMemcpyHostToDevice, s.compute));
        // Identity is static for the run: one D2H here (a run boundary).
        const auto un = static_cast<std::size_t>(n);
        const Frame::Atoms& a = ctx.frame.atoms;
        s.tag.resize(un);
        if (a.tag)
            ATV_CUDA_CHECK(cudaMemcpyAsync(s.tag.data(), a.tag,
                                           un * sizeof(std::int64_t),
                                           cudaMemcpyDeviceToHost, s.compute));
        else
            std::iota(s.tag.begin(), s.tag.end(), std::int64_t{1});
        s.has_type = a.type != nullptr;
        s.has_z = a.atomic_number != nullptr;
        s.type.assign(s.has_type ? un : 0, 0);
        s.atomic_number.assign(s.has_z ? un : 0, 0);
        if (s.has_type)
            ATV_CUDA_CHECK(cudaMemcpyAsync(s.type.data(), a.type,
                                           un * sizeof(int),
                                           cudaMemcpyDeviceToHost, s.compute));
        if (s.has_z)
            ATV_CUDA_CHECK(cudaMemcpyAsync(s.atomic_number.data(),
                                           a.atomic_number, un * sizeof(int),
                                           cudaMemcpyDeviceToHost, s.compute));
        ATV_CUDA_CHECK(cudaStreamSynchronize(s.compute));
    } catch (...) {
        s.free_device();
        throw;
    }
    s.n = n;
    s.layout = layout;
    s.capacity = capacity;
    columns_ = columns;
    ++generation_;
    Word* words = s.words;
    ledger_.start(
        budget_, layout.bytes,
        [words](int id, std::uint64_t index) {
            return load_acquire(words[id].value) > index;
        },
        what);
    dirty_ = false;
}

CaptureLedger::Join CaptureBuffer::join(AsyncConsumer& consumer, long long step,
                                        CaptureKind kind) {
    return ledger_.join(consumer, step, kind);
}

void CaptureBuffer::launch(AsyncConsumer& consumer, const LaunchContext& ctx,
                           const long long* step, long long offset,
                           bool force) {
    Impl& s = *impl_;
    if (!s.control)
        throw std::logic_error("cuda::CaptureBuffer: launch before ensure()");
    const int id = ledger_.id_of(consumer);
    if (id < 0)
        throw std::logic_error("cuda::CaptureBuffer: " + consumer.capture_name()
                               + " is not subscribed");
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    acquire_k<<<1, 1, 0, stream>>>(s.control, step, offset, force ? 1 : 0,
                                   s.capacity);
    pack_k<<<grid_size(s.n), kDefaultBlock, 0, stream>>>(
        s.control, s.buffer_dev, s.layout, ctx.frame.atoms, ctx.frame.box,
        ctx.frame.scalars, s.n);
    const auto uid = static_cast<std::size_t>(id);
    if (s.payload_bytes[uid] > 0) {
        const void* src = consumer.payload_source(ctx);
        if (!src)
            throw std::logic_error(consumer.capture_name()
                                   + ": no payload source to capture");
        const u64 bytes = s.payload_bytes[uid];
        const int grid = static_cast<int>(
            std::min<u64>((bytes + kDefaultBlock - 1) / kDefaultBlock, 1024));
        payload_k<<<grid, kDefaultBlock, 0, stream>>>(
            s.control, s.buffer_dev, s.layout.bytes,
            s.layout.payload + s.payload_offset[uid],
            static_cast<const unsigned char*>(src), bytes);
    }
    publish_k<<<1, 1, 0, stream>>>(s.control, &s.words_dev[id].value);
    ATV_CUDA_CHECK(cudaGetLastError());
}

CaptureView CaptureBuffer::view(const AsyncConsumer& consumer,
                                const CaptureTicket& ticket) const {
    const Impl& s = *impl_;
    const Layout& L = s.layout;
    const unsigned char* base =
        s.buffer + static_cast<std::size_t>(ticket.slot) * L.bytes;
    const auto* h = reinterpret_cast<const SlotHeader*>(base);
    const auto un = static_cast<std::size_t>(s.n);
    CaptureView v;
    v.step = ticket.step;
    v.kind = ticket.kind;
    v.n_atoms = s.n;
    v.cell = h->cell;
    v.scalars = &h->scalars;
    auto f64 = [&](std::size_t off, std::size_t k) -> const double* {
        return off ? reinterpret_cast<const double*>(base + off) + k * un
                   : nullptr;
    };
    v.x = f64(L.x, 0);
    v.y = f64(L.x, 1);
    v.z = f64(L.x, 2);
    v.vx = f64(L.v, 0);
    v.vy = f64(L.v, 1);
    v.vz = f64(L.v, 2);
    v.fx = f64(L.f, 0);
    v.fy = f64(L.f, 1);
    v.fz = f64(L.f, 2);
    v.charge = f64(L.q, 0);
    v.mass = f64(L.m, 0);
    v.tag =
        L.tag ? reinterpret_cast<const std::int64_t*>(base + L.tag) : nullptr;
    v.image =
        L.image ? reinterpret_cast<const ImageInt*>(base + L.image) : nullptr;
    v.type = s.has_type ? s.type.data() : nullptr;
    v.atomic_number = s.has_z ? s.atomic_number.data() : nullptr;
    v.tag_at_build = s.tag.data();
    const int id = ledger_.id_of(consumer);
    if (id >= 0 && s.payload_bytes[static_cast<std::size_t>(id)] > 0) {
        v.payload =
            base + L.payload + s.payload_offset[static_cast<std::size_t>(id)];
        v.payload_bytes = s.payload_bytes[static_cast<std::size_t>(id)];
    }
    return v;
}

// ── AsyncConsumer ──────────────────────────────────────────────────────────

std::uint64_t AsyncConsumer::graph_epoch() const {
    return buffer_ ? buffer_->generation() : 0;
}

void AsyncConsumer::build(const BuildContext& ctx) {
    buffer_ = ctx.runtime.capture;
    if (!buffer_)
        throw std::logic_error(
            capture_name()
            + ": an asynchronous consumer needs its driver's capture buffer; "
              "build it through a cuda driver (cuda::md::MD, …)");
    if (graph_capturable() && !ctx.runtime.device_step)
        throw std::logic_error(capture_name()
                               + ": the backend publishes no device step");
    buffer_->subscribe(*this);
    last_step_ = -1;
    admitted_ = false;
    on_build(ctx);
}

void AsyncConsumer::before_launch(const LaunchContext& ctx) {
    // In-loop: the end of the step with zero-based index ctx.step, labelled
    // with the completed-step count (LAMMPS `update->ntimestep`). In-loop
    // labels strictly increase, so every in-loop execution is admitted —
    // replayed or eager — and launch() enqueues it.
    admitted_ = false;
    if (!buffer_)
        throw std::logic_error(capture_name() + ": capture before build()");
    require_open();
    if (!graph_capturable() && !can_run(ctx))
        return;
    const BigInt step = ctx.step + 1;
    buffer_->join(*this, static_cast<long long>(step), CaptureKind::step);
    last_step_ = step;
    admitted_ = true;
}

void AsyncConsumer::launch(const LaunchContext& ctx) {
    // Called eagerly or once while a step graph is captured (then replayed
    // without calling it): device work only, labelled on the device.
    if (!admitted_)
        return;
    buffer_->launch(*this, ctx, ctx.runtime.device_step, 1, /*force=*/false);
}

void AsyncConsumer::capture_now(const LaunchContext& ctx, BigInt step,
                                CaptureKind kind) {
    if (!buffer_)
        throw std::logic_error(capture_name() + ": capture before build()");
    require_open();
    if (!can_run(ctx))
        return;
    const auto join = buffer_->join(*this, static_cast<long long>(step), kind);
    last_step_ = step;
    // Eager: the host knows whether it opened the capture.
    buffer_->launch(*this, ctx, nullptr, static_cast<long long>(step),
                    join.opened);
}

void AsyncConsumer::setup_output(const LaunchContext& ctx) {
    // LAMMPS src/output.cpp Output::setup, e8c4c278a8. Copyright (2003)
    // Sandia Corporation. GPL-2.0-only.
    const int every = frequency();
    if (every > 0 && ctx.step % every != 0)
        return;
    if (ctx.step == last_step_)
        return;
    capture_now(ctx, ctx.step, CaptureKind::setup);
}

void AsyncConsumer::final_output(const LaunchContext& ctx) {
    // LAMMPS src/min.cpp Min::run, e8c4c278a8: the final state of a
    // minimization stopped early, unless already written.
    if (ctx.step == last_step_)
        return;
    capture_now(ctx, ctx.step, CaptureKind::final_state);
}

void AsyncConsumer::destroy(const DestroyContext& ctx) {
    // Every capture this consumer was due is consumed first; on_destroy may
    // still read the buffer's accounting.
    if (buffer_)
        buffer_->unsubscribe(*this);
    struct Detach {
        CaptureBuffer*& buffer;
        ~Detach() { buffer = nullptr; }
    } detach{buffer_};
    on_destroy(ctx);
}

void AsyncConsumer::consume_capture(const CaptureTicket& ticket) {
    consume(buffer_->view(*this, ticket));
}

} // namespace cuda
} // namespace ATV
