/// @file cpu/potentials/force_scratch.h
/// @brief Shared scaffolding for threaded force kernels: the Runtime's
///        chunk-owned force accumulators and partial sums (`ChunkForces`,
///        `ForcePass`), a Voigt-order virial accumulator and the bonded drop
///        tally.
#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "atomiverse/core/log.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/frame/frame.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ATV {
namespace cpu {
namespace md {

/// @brief Voigt-order (xx yy zz xy xz yz) virial accumulator. Threaded
///        kernels keep one per chunk and combine them through
///        `ForcePass::sums` / `virial` (chunk order), never an OpenMP
///        floating-point reduction.
struct Voigt6 {
    double v[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    /// @brief Add a pair contribution Δ⊗f in Voigt order.
    void add_pair(double dx, double dy, double dz, double fx, double fy,
                  double fz) noexcept {
        v[0] += dx * fx;
        v[1] += dy * fy;
        v[2] += dz * fz;
        v[3] += dx * fy;
        v[4] += dx * fz;
        v[5] += dy * fz;
    }

    Voigt6& operator+=(const Voigt6& o) noexcept {
        for (int k = 0; k < 6; ++k)
            v[k] += o.v[k];
        return *this;
    }

    /// @brief Accumulate into the resolved `kStress` value.
    void flush_to(MetaReal6& stress) const noexcept {
        for (int k = 0; k < 6; ++k)
            stress[k] += v[k];
    }
};

/// @brief Raw force-row pointers one chunk of a threaded kernel scatters into.
struct ForceRows {
    double* __restrict fx;
    double* __restrict fy;
    double* __restrict fz;
};

/// @brief The Runtime's chunk-owned force accumulators and partial sums,
///        shared by every threaded force kernel of a slot.
///
/// **One mechanism, deterministic.** A threaded kernel cuts its work into a
/// fixed number of chunks (a pure function of the team size and the work,
/// never of timing). Chunk `c` scatters its forces into slot buffer `c` and
/// writes its energy / virial partials into sum slot `c`; any thread may run
/// any chunk (dynamic pickup is allowed), but a chunk always owns the same
/// slot. Slots are combined in slot order — forces once per slot walk by
/// `flush()`, sums once per launch by `total()` — so for a given thread count
/// every run is bit-identical, whichever thread ran which chunk and in which
/// order chunks finished.
///
/// **Shared across kernels** (LAMMPS `package omp`: `FixOMP` clears, the
/// last `/omp` style reduces): the `cpu::Runtime` owns one `ChunkForces`,
/// publishes it on `Context::md.chunk_forces`, and calls `flush()` once at
/// the end of every slot walk (before the reverse halo pass), so a step with
/// several threaded styles zeroes and reduces each slot buffer once instead
/// of once per style. Slot `c` of a later kernel adds onto slot `c` of an
/// earlier one; kernels run in a fixed order, so the per-row sum order is
/// fixed too.
///
/// Kernels reach it through `ForcePass` (below), once per launch:
/// @code{.cpp}
/// md::ForcePass forces(ctx);
/// forces.open(team, n_chunks);           // serial, fixes the partition
/// #pragma omp parallel num_threads(team) if (team > 1)
/// #pragma omp for schedule(dynamic, 1)
/// for (int c = 0; c < n_chunks; ++c) {
///     const md::ForceRows f = forces.rows(c);
///     double e = 0; ... f.fx[i] += ...; e += ...;
///     forces.sums(c)[0] = e;
/// }
/// const double energy = forces.total(0); // chunk order
/// // no force reduction: the Runtime flushes after the slot.
/// @endcode
///
/// A launch opened with a team of one writes straight into the frame force
/// columns (the pure-MPI layout pays no zero or reduce pass at all). A slot
/// buffer is zeroed the first time a chunk touches it after a flush, over the
/// rows the pass needs (`open`'s `n_touch`, grown as later launches need
/// more), and `flush()` sums only the slots touched since the previous flush.
///
/// A threaded launch with `forces_requested == false` (MC energy trials)
/// still receives slot buffers so kernels need no second code path; that pass
/// is discarded instead of reduced. A team of one writes the frame in either
/// case, so kernels keep gating their own writes on the request.
class ChunkForces {
  public:
    /// Partial sums per chunk: energy + Voigt virial + one style extra.
    /// Eight doubles, one cache line per chunk.
    static constexpr int kSums = 8;

    /// @brief Pre-size `n_slots` slot buffers of `rows` rows (call from a
    ///        kernel's `build()`), so steady-state launches never allocate.
    void reserve(int n_slots, int rows) {
        if (n_slots <= 1)
            return;
        grow_slots_(n_slots);
        for (int c = 0; c < n_slots; ++c)
            grow_(bufs_[static_cast<std::size_t>(c)], rows);
    }

    /// @brief Bind this launch to `frame`. Call once per launch, serially,
    ///        before `open()`.
    /// @param frame Frame whose force columns receive the flush.
    /// @param forces_requested The launch's force request; a pass opened
    ///        without it is discarded by `flush()`.
    void bind(cpu::Frame& frame, bool forces_requested) {
        if (frame_ != nullptr && frame_ != &frame)
            flush();
        // Re-resolved every launch: a structural mutation (capacity growth,
        // schema change) between launches invalidates the previous spans.
        frame_ = &frame;
        fx_frame_ = frame.span(kAtom_fx);
        fy_frame_ = frame.span(kAtom_fy);
        fz_frame_ = frame.span(kAtom_fz);
        discard_ = !forces_requested;
    }

    /// @brief Open `n_chunks` chunks over rows `[0, n_touch)` for the bound
    ///        launch and clear their sums. Serial, after `bind()`.
    void open(int n_chunks, int n_touch) {
        n_chunks = std::max(n_chunks, 1);
        grow_slots_(n_chunks);
        if (n_touch > rows_) {
            // A slot already live in this pass must cover the new rows.
            for (Buffer& b : bufs_)
                if (b.pass == pass_) {
                    grow_(b, n_touch);
                    zero_(b, rows_, n_touch);
                }
            rows_ = n_touch;
        }
        sums_.assign(static_cast<std::size_t>(n_chunks) * kSums, 0.0);
        n_chunks_ = n_chunks;
    }

    /// @brief Chunk `c`'s force rows; zeroed on its first touch in this pass.
    ///        Race-free inside the region: one chunk runs on one thread.
    ForceRows slot(int c) {
        Buffer& b = bufs_[static_cast<std::size_t>(c)];
        if (b.pass != pass_) {
            grow_(b, rows_);
            zero_(b, 0, rows_);
            b.pass = pass_;
        }
        return {b.fx.data(), b.fy.data(), b.fz.data()};
    }

    /// @brief Chunk `c`'s `kSums` partial sums of the open launch.
    double* sums(int c) { return sums_.data() + std::size_t(c) * kSums; }

    /// @brief Partial `k` summed over the open launch's chunks, in order.
    double total(int k) const {
        double s = 0.0;
        for (int c = 0; c < n_chunks_; ++c)
            s += sums_[std::size_t(c) * kSums + std::size_t(k)];
        return s;
    }

    /// @brief True when some slot buffer holds unreduced forces.
    bool pending() const noexcept {
        for (const Buffer& b : bufs_)
            if (b.pass == pass_)
                return true;
        return false;
    }

    /// @brief Reduce every slot touched since the last flush into the bound
    ///        frame's force columns in slot order (or drop it for an
    ///        energy-only pass), and close the pass. Runtime-only; a no-op
    ///        when nothing is pending.
    void flush() {
        live_.clear();
        for (std::size_t c = 0; c < bufs_.size(); ++c)
            if (bufs_[c].pass == pass_)
                live_.push_back(static_cast<int>(c));
        const int n_live = static_cast<int>(live_.size());
        if (n_live > 0 && !discard_) {
            const int* live = live_.data();
            double* __restrict fx = fx_frame_.data();
            double* __restrict fy = fy_frame_.data();
            double* __restrict fz = fz_frame_.data();
            const int rows = rows_;
            const Buffer* bufs = bufs_.data();
            // Row-parallel; each row sums its slots in slot order, so the
            // thread count of the flush itself never changes the result.
#pragma omp parallel for schedule(static) if (rows > 4096)
            for (int i = 0; i < rows; ++i) {
                double sx = 0.0, sy = 0.0, sz = 0.0;
                for (int k = 0; k < n_live; ++k) {
                    const Buffer& b = bufs[live[k]];
                    sx += b.fx[static_cast<std::size_t>(i)];
                    sy += b.fy[static_cast<std::size_t>(i)];
                    sz += b.fz[static_cast<std::size_t>(i)];
                }
                fx[i] += sx;
                fy[i] += sy;
                fz[i] += sz;
            }
        }
        ++pass_;
        rows_ = 0;
        discard_ = false;
    }

    /// @brief Drop buffers and the frame binding (Runtime teardown).
    void clear() {
        bufs_.clear();
        bufs_.shrink_to_fit();
        sums_.clear();
        frame_ = nullptr;
        fx_frame_ = Span<AccReal>{};
        fy_frame_ = Span<AccReal>{};
        fz_frame_ = Span<AccReal>{};
        ++pass_;
        rows_ = 0;
        n_chunks_ = 0;
        discard_ = false;
    }

    std::size_t memory_bytes() const noexcept {
        std::size_t bytes = sums_.capacity() * sizeof(double);
        for (const Buffer& b : bufs_)
            bytes += (b.fx.capacity() + b.fy.capacity() + b.fz.capacity())
                     * sizeof(double);
        return bytes;
    }

  private:
    struct Buffer {
        std::vector<double> fx, fy, fz;
        std::uint64_t pass = 0; ///< pass in which this buffer was zeroed.
    };

    void grow_slots_(int n) {
        if (static_cast<int>(bufs_.size()) < n) {
            bufs_.resize(static_cast<std::size_t>(n));
            live_.reserve(bufs_.size());
        }
    }

    static void grow_(Buffer& b, int rows) {
        if (static_cast<int>(b.fx.size()) < rows) {
            b.fx.resize(static_cast<std::size_t>(rows), 0.0);
            b.fy.resize(static_cast<std::size_t>(rows), 0.0);
            b.fz.resize(static_cast<std::size_t>(rows), 0.0);
        }
    }

    static void zero_(Buffer& b, int lo, int hi) {
        std::fill(b.fx.begin() + lo, b.fx.begin() + hi, 0.0);
        std::fill(b.fy.begin() + lo, b.fy.begin() + hi, 0.0);
        std::fill(b.fz.begin() + lo, b.fz.begin() + hi, 0.0);
    }

    std::vector<Buffer> bufs_;
    std::vector<int> live_;    ///< flush scratch; capacity follows bufs_.
    std::vector<double> sums_; ///< n_chunks_ x kSums partials of a launch.
    std::uint64_t pass_ = 1;   ///< buffers with `pass == pass_` are live.
    int rows_ = 0;             ///< rows zeroed in every live buffer.
    int n_chunks_ = 0;         ///< chunks of the open launch.
    bool discard_ = false;     ///< the open pass requested no forces.
    cpu::Frame* frame_ = nullptr;
    Span<AccReal> fx_frame_{}, fy_frame_{}, fz_frame_{};
};

/// @brief One launch's view of the chunk accumulators: the Runtime's
///        `ChunkForces` when published, otherwise the frame columns with a
///        team of one (a component driven without a Runtime runs serially).
///
/// Construct once per launch (serially), pick the team, `open()` the
/// partition, then inside the region take each chunk's `rows(c)` and
/// `sums(c)`; read `total(k)` after the region. A team of one is one chunk
/// that writes the frame directly.
class ForcePass {
  public:
    static constexpr int kSums = ChunkForces::kSums;

    explicit ForcePass(const cpu::LaunchContext& ctx)
        : arena_(ctx.runtime.md.chunk_forces), frame_(&ctx.frame),
          frame_rows_{ctx.frame.span(kAtom_fx).data(),
                      ctx.frame.span(kAtom_fy).data(),
                      ctx.frame.span(kAtom_fz).data()} {
        if (arena_ != nullptr) {
            arena_->bind(ctx.frame, ctx.forces_requested);
#ifdef _OPENMP
            max_team_ = omp_get_max_threads();
#endif
        }
    }

    ForcePass(const ForcePass&) = delete;
    ForcePass& operator=(const ForcePass&) = delete;

    /// @brief Threads available to the kernel's parallel region.
    int team() const noexcept { return max_team_; }

    /// Terms below which one more thread costs more (wake-up, barrier, and
    /// its buffer's zero + reduce sweep) than it saves.
    static constexpr int kMinTermsPerThread = 2048;

    /// @brief Threads worth opening for `n_terms` terms, at most `team()`. A
    ///        team of one writes the frame directly, so a small topology
    ///        does not wake the whole team every evaluation.
    int team(int n_terms) const noexcept {
        return std::clamp(n_terms / kMinTermsPerThread, 1, max_team_);
    }

    /// @brief Fix this launch's partition: `n_chunks` chunks for a team of
    ///        `team`, scattering into rows `[0, n_touch)`. Serial, before the
    ///        region. A team of one is forced to one direct chunk.
    void open(int team, int n_chunks, int n_touch) {
        direct_ = arena_ == nullptr || team <= 1;
        if (direct_) {
            n_chunks_ = 1;
            std::fill(one_, one_ + kSums, 0.0);
            return;
        }
        n_chunks_ = std::max(n_chunks, 1);
        arena_->open(n_chunks_, n_touch);
    }

    /// @brief `open(team, team, n_force)`: one chunk per thread over every
    ///        row of the frame.
    void open(int team) { open(team, team, frame_->n_total()); }

    /// @brief Chunks of the open launch.
    int chunks() const noexcept { return n_chunks_; }

    /// @brief True when the open launch writes the frame directly.
    bool direct() const noexcept { return direct_; }

    /// @brief Chunk `c`'s force rows (call by the thread running it).
    ForceRows rows(int c) const {
        return direct_ ? frame_rows_ : arena_->slot(c);
    }

    /// Sum-slot layout every kernel uses: energy, Voigt virial, one extra
    /// (e.g. a soft-core dU/dλ).
    static constexpr int kEnergy = 0;
    static constexpr int kVirial = 1;
    static constexpr int kExtra = 7;

    /// @brief Chunk `c`'s `kSums` partial sums (zeroed by `open`).
    double* sums(int c) { return direct_ ? one_ : arena_->sums(c); }

    /// @brief Store chunk `c`'s energy, virial and extra partials (call once
    ///        at the end of the chunk, from the thread that ran it).
    void store(int c, double energy, const Voigt6& vir, double extra = 0.0) {
        double* s = sums(c);
        s[kEnergy] = energy;
        for (int k = 0; k < 6; ++k)
            s[kVirial + k] = vir.v[k];
        s[kExtra] = extra;
    }

    /// @brief Partial `k` over every chunk, in chunk order.
    double total(int k) const { return direct_ ? one_[k] : arena_->total(k); }

    double energy() const { return total(kEnergy); }
    double extra() const { return total(kExtra); }
    Voigt6 virial() const {
        Voigt6 v;
        for (int k = 0; k < 6; ++k)
            v.v[k] = total(kVirial + k);
        return v;
    }

    /// @brief Run `body(c, begin, end, rows)` over `team(n_terms)` equal
    ///        static chunks of `[0, n_terms)`, one per thread, with every row
    ///        of the frame open. The body ends with `store(c, ...)`.
    template <class Body>
    void for_each_chunk(int n_terms, Body&& body) {
        const int t = team(n_terms);
        open(t);
        const int n = n_chunks_;
#pragma omp parallel for num_threads(t) schedule(static, 1) if (t > 1)
        for (int c = 0; c < n; ++c) {
            const int begin =
                static_cast<int>(static_cast<long long>(n_terms) * c / n);
            const int end =
                static_cast<int>(static_cast<long long>(n_terms) * (c + 1) / n);
            body(c, begin, end, rows(c));
        }
    }

  private:
    ChunkForces* arena_;
    cpu::Frame* frame_;
    ForceRows frame_rows_;
    int max_team_ = 1;
    int n_chunks_ = 1;
    bool direct_ = true;
    double one_[kSums] = {};
};

/// @brief Rank-local tally of bonded partners that failed to resolve.
///
/// A bonded kernel drops any tuple whose partner tag resolves to -1
/// (`ResolveView::resolve` — tag absent on this rank or outside
/// `[0, n_force)`). With a correctly sized halo this never fires; a nonzero
/// `dropped` means the cutoff/ghost layer was too thin and that tuple's
/// energy + forces were silently lost. Counting is a couple of integer adds
/// in the hot loop (cheap, allocation-free); the warning is emitted once at
/// teardown via `report()`, so the no-allocation-in-launch invariant holds.
struct DropTally {
    std::uint64_t dropped = 0;   ///< bonded partners not resolvable.
    std::uint64_t evaluated = 0; ///< bonded partners successfully resolved.

    DropTally& operator+=(const DropTally& o) noexcept {
        dropped += o.dropped;
        evaluated += o.evaluated;
        return *this;
    }

    /// @brief Emit a single warning if any partner was dropped. Call from
    /// the kernel's `destroy()` (end of run) — never inside `compute()`.
    void report(const char* kernel) const {
        if (dropped == 0)
            return;
        ATV_WARN("{}: dropped {} of {} bonded partners (rank-local) — they "
                 "resolved outside the halo, so their energy/forces were lost. "
                 "The ghost cutoff is likely too thin for the bond/angle/"
                 "dihedral span; widen the comm cutoff.",
                 kernel, dropped, dropped + evaluated);
    }

    void reset() noexcept { dropped = evaluated = 0; }
};

} // namespace md
} // namespace cpu
} // namespace ATV
