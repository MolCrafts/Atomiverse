/// @file cpu/asynchronous/capture_buffer.cpp
/// @brief cpu::CaptureBuffer: one host arena of `budget / frame_bytes` slots,
///        allocated once per consumer set at a run boundary, filled once per
///        captured step, read by every due consumer (ledger semantics in
///        core/capture_ledger.cpp).

#include "atomiverse/cpu/asynchronous/capture_buffer.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/asynchronous/async_consumer.h"
#include "atomiverse/cpu/frame/frame.h"

namespace ATV {
namespace cpu {

namespace {

constexpr std::size_t kAlign = 64;

std::size_t align_up(std::size_t v) { return (v + kAlign - 1) & ~(kAlign - 1); }

/// @brief Copy @p n values of @p src (any arithmetic type) into @p dst,
///        widening to `Dst`.
template <class Dst, class Src>
void copy_column(Dst* dst, const Src* src, std::size_t n) {
#pragma omp parallel for schedule(static) if (n > 4096)
    for (std::size_t i = 0; i < n; ++i)
        dst[i] = static_cast<Dst>(src[i]);
}

void require_column(const Frame& frame, const char* key, const char* what) {
    if (!frame.has("atoms", key))
        throw std::invalid_argument(
            std::string("cpu::CaptureBuffer: an asynchronous consumer reads ")
            + what + " but the frame has no atoms." + key + " column");
}

} // namespace

CaptureBuffer::CaptureBuffer() : ledger_("cpu::CaptureBuffer") {}

CaptureBuffer::~CaptureBuffer() { stop(); }

CaptureBuffer::Layout CaptureBuffer::make_layout_(int n, CaptureColumns c) {
    const auto un = static_cast<std::size_t>(std::max(n, 0));
    Layout L;
    std::size_t at = 0;
    L.cell = at; // 9 doubles, then the atom count (int64)
    at = align_up(at + 9 * sizeof(double) + sizeof(std::int64_t));
    auto take = [&](bool on, std::size_t bytes) {
        if (!on)
            return std::size_t{0};
        const std::size_t off = at;
        at = align_up(at + bytes);
        return off;
    };
    L.positions = take(c.positions, 3 * un * sizeof(double));
    L.velocities = take(c.velocities, 3 * un * sizeof(double));
    L.forces = take(c.forces, 3 * un * sizeof(double));
    L.charges = take(c.charges, un * sizeof(double));
    L.masses = take(c.masses, un * sizeof(double));
    L.types = take(c.types, un * sizeof(int));
    L.tags = take(c.tags, un * sizeof(std::int64_t));
    L.images = take(c.images, 3 * un * sizeof(ImageInt));
    L.bytes = at;
    return L;
}

std::uint64_t CaptureBuffer::frame_bytes(int n_atoms, CaptureColumns columns) {
    if (n_atoms < 0)
        throw std::invalid_argument(
            "cpu::CaptureBuffer::frame_bytes: n_atoms must be >= 0");
    return make_layout_(n_atoms, columns).bytes;
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
    try {
        ledger_.drain(/*raise=*/false); // a failure stays pending
    } catch (...) {
    }
    ledger_.stop();
}

void CaptureBuffer::ensure(const Frame& frame) {
    if (!has_consumers()) {
        stop();
        return;
    }
    if (ledger_.running() && !dirty_)
        return;
    stop();
    const int n = frame.nlocal();
    CaptureColumns columns{};
    for (auto* s : ledger_.sinks())
        if (s)
            columns =
                columns | static_cast<AsyncConsumer*>(s)->capture_columns();
    if (columns.positions)
        require_column(frame, kAtom_x.key, "positions");
    if (columns.velocities)
        require_column(frame, kAtom_vx.key, "velocities");
    if (columns.forces)
        require_column(frame, kAtom_fx.key, "forces");
    if (columns.charges)
        require_column(frame, kAtom_charge.key, "charges");
    if (columns.masses)
        require_column(frame, kAtom_mass.key, "masses");
    if (columns.types)
        require_column(frame, kAtom_type.key, "force-field types");
    if (columns.tags)
        require_column(frame, kAtom_tag.key, "tags");
    if (columns.images)
        require_column(frame, kAtom_image.key, "image flags");
    const Layout layout = make_layout_(n, columns);
    const std::string what = std::to_string(n) + " atoms";
    // Validated (one frame at least) before anything is allocated; the
    // arena is then sized exactly once.
    const std::uint64_t capacity = CaptureLedger::capacity_for(
        "cpu::CaptureBuffer", budget_, layout.bytes, what);
    const std::uint64_t bytes = capacity * layout.bytes;
    if (bytes != arena_bytes_) {
        arena_.reset(); // release before acquiring the replacement
        arena_bytes_ = 0;
        // Uninitialized: pages are committed as slots are first written.
        arena_ = std::make_unique_for_overwrite<unsigned char[]>(
            static_cast<std::size_t>(bytes));
        arena_bytes_ = bytes;
    }
    columns_ = columns;
    n_atoms_ = n;
    layout_ = layout;
    published_.store(0, std::memory_order_release);
    ledger_.start(
        budget_, layout.bytes,
        [this](int, std::uint64_t index) {
            return published_.load(std::memory_order_acquire) > index;
        },
        what);
    dirty_ = false;
}

unsigned char* CaptureBuffer::slot_(std::uint64_t slot) const {
    return arena_.get() + static_cast<std::size_t>(slot) * layout_.bytes;
}

void CaptureBuffer::capture(AsyncConsumer& consumer, const Frame& frame,
                            BigInt step, CaptureKind kind) {
    const auto join =
        ledger_.join(consumer, static_cast<long long>(step), kind);
    if (!join.opened)
        return; // this step is already captured: read the same copy
    const int n = frame.nlocal();
    if (n > n_atoms_)
        throw std::runtime_error(
            "cpu::CaptureBuffer: the frame holds " + std::to_string(n)
            + " atoms but the capture buffer was laid out for "
            + std::to_string(n_atoms_)
            + " at the start of the run; asynchronous consumers need a "
              "population that does not grow within one run");
    unsigned char* base = slot_(join.ticket.slot);
    // Header: the cell and the atom count (the population may shrink).
    double cell[9];
    const Mat3& m = frame.box.cell();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            cell[3 * i + j] = static_cast<double>(m(i, j));
    std::memcpy(base + layout_.cell, cell, sizeof(cell));
    const auto count = static_cast<std::int64_t>(n);
    std::memcpy(base + layout_.cell + sizeof(cell), &count, sizeof(count));
    const auto un = static_cast<std::size_t>(n);
    const auto cap = static_cast<std::size_t>(n_atoms_);
    auto f64 = [&](std::size_t off) {
        return reinterpret_cast<double*>(base + off);
    };
    if (columns_.positions) {
        copy_column(f64(layout_.positions), frame.span(kAtom_x).data(), un);
        copy_column(f64(layout_.positions) + cap, frame.span(kAtom_y).data(),
                    un);
        copy_column(f64(layout_.positions) + 2 * cap,
                    frame.span(kAtom_z).data(), un);
    }
    if (columns_.velocities) {
        copy_column(f64(layout_.velocities), frame.span(kAtom_vx).data(), un);
        copy_column(f64(layout_.velocities) + cap, frame.span(kAtom_vy).data(),
                    un);
        copy_column(f64(layout_.velocities) + 2 * cap,
                    frame.span(kAtom_vz).data(), un);
    }
    if (columns_.forces) {
        copy_column(f64(layout_.forces), frame.span(kAtom_fx).data(), un);
        copy_column(f64(layout_.forces) + cap, frame.span(kAtom_fy).data(), un);
        copy_column(f64(layout_.forces) + 2 * cap, frame.span(kAtom_fz).data(),
                    un);
    }
    if (columns_.charges)
        copy_column(f64(layout_.charges), frame.span(kAtom_charge).data(), un);
    if (columns_.masses)
        copy_column(f64(layout_.masses), frame.span(kAtom_mass).data(), un);
    if (columns_.types)
        copy_column(reinterpret_cast<int*>(base + layout_.types),
                    frame.span(kAtom_type).data(), un);
    if (columns_.tags)
        copy_column(reinterpret_cast<std::int64_t*>(base + layout_.tags),
                    frame.span(kAtom_tag).data(), un);
    if (columns_.images)
        copy_column(reinterpret_cast<ImageInt*>(base + layout_.images),
                    frame.span(kAtom_image).data(), 3 * un);
    published_.store(join.ticket.index + 1, std::memory_order_release);
    ledger_.notify_published();
}

FrameSnapshot CaptureBuffer::view(const CaptureTicket& ticket) const {
    const unsigned char* base = slot_(ticket.slot);
    FrameSnapshot s;
    s.step = static_cast<BigInt>(ticket.step);
    s.kind = ticket.kind;
    std::int64_t count = 0;
    std::memcpy(&count, base + layout_.cell + 9 * sizeof(double),
                sizeof(count));
    const int n = static_cast<int>(count);
    s.n_owned = n;
    const auto un = static_cast<std::size_t>(n);
    const auto cap = static_cast<std::size_t>(n_atoms_);
    double cell[9];
    std::memcpy(cell, base + layout_.cell, sizeof(cell));
    bool periodic = false;
    for (double v : cell)
        periodic = periodic || v != 0.0;
    if (periodic) {
        Mat3 m;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                m(i, j) = static_cast<AccReal>(cell[3 * i + j]);
        s.box = cpu::Box(m);
    }
    auto f64 = [&](std::size_t off, std::size_t k) {
        return std::span<const double>(
            reinterpret_cast<const double*>(base + off) + k * cap, un);
    };
    if (columns_.positions) {
        s.x = f64(layout_.positions, 0);
        s.y = f64(layout_.positions, 1);
        s.z = f64(layout_.positions, 2);
    }
    if (columns_.velocities) {
        s.vx = f64(layout_.velocities, 0);
        s.vy = f64(layout_.velocities, 1);
        s.vz = f64(layout_.velocities, 2);
    }
    if (columns_.forces) {
        s.fx = f64(layout_.forces, 0);
        s.fy = f64(layout_.forces, 1);
        s.fz = f64(layout_.forces, 2);
    }
    if (columns_.charges)
        s.charge = f64(layout_.charges, 0);
    if (columns_.masses)
        s.mass = f64(layout_.masses, 0);
    if (columns_.types)
        s.type = std::span<const int>(
            reinterpret_cast<const int*>(base + layout_.types), un);
    if (columns_.tags)
        s.tag = std::span<const std::int64_t>(
            reinterpret_cast<const std::int64_t*>(base + layout_.tags), un);
    if (columns_.images)
        s.image = std::span<const ImageInt>(
            reinterpret_cast<const ImageInt*>(base + layout_.images), 3 * un);
    return s;
}

} // namespace cpu
} // namespace ATV
