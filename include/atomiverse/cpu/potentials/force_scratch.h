/// @file cpu/potentials/force_scratch.h
/// @brief Shared scaffolding for threaded force kernels: the Runtime's
///        per-thread force accumulators (`ThreadForces`), a Voigt-order
///        virial accumulator and the bonded drop tally.
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

/// @brief Raw per-thread force pointers for one OMP thread's hot loop.
struct ThreadForce {
    double* __restrict fx;
    double* __restrict fy;
    double* __restrict fz;
};

/// @brief The Runtime's per-thread force accumulators, shared by every
///        threaded force kernel of a step (LAMMPS `package omp` style).
///
/// LAMMPS `FixOMP` clears each thread's force array once per step and the
/// last `/omp` style reduces them once (`ThrOMP::reduce_thr`). Here the
/// `cpu::Runtime` owns one `ThreadForces`, publishes it on
/// `Context::md.thread_forces`, and calls `flush()` once at the end of every
/// slot walk (before the reverse halo pass): a step with six threaded styles
/// zeroes and reduces 3·n_force doubles per thread once instead of six times.
///
/// Kernels reach it through `ForcePass` (below), once per launch:
/// @code{.cpp}
/// md::ForcePass forces(ctx);
/// #pragma omp parallel num_threads(forces.team()) if (forces.team() > 1)
/// {
///     const md::ThreadForce tf = forces.thread(tid);
///     ... tf.fx[i] += ...;
/// }
/// // no reduction: the Runtime flushes after the slot.
/// @endcode
///
/// A team of one writes straight into the frame force columns (the pure-MPI
/// layout pays no zero or reduce pass at all). A thread's buffer is zeroed
/// the first time the thread touches it after a flush (first touch also
/// places it on the thread's NUMA node), and `flush()` sums only buffers
/// touched since the previous flush. Zero/reduce cover `[0, n_force)`
/// (`Frame::n_total()` = owners + ghosts), never the allocated capacity.
///
/// A threaded launch with `forces_requested == false` (MC energy trials)
/// still receives private buffers so kernels need no second code path; that
/// pass is discarded instead of reduced. A team of one writes the frame in
/// either case, so kernels keep gating their own writes on the request.
class ThreadForces {
  public:
    /// @brief Pre-size buffers for `n_threads` threads of `rows` rows (call
    ///        from a kernel's `build()`), so steady-state launches never
    ///        allocate.
    void reserve(int n_threads, int rows) {
        if (n_threads <= 1)
            return;
        if (static_cast<int>(bufs_.size()) < n_threads) {
            bufs_.resize(static_cast<std::size_t>(n_threads));
            live_.reserve(bufs_.size());
        }
        for (int t = 0; t < n_threads; ++t) {
            Buffer& b = bufs_[static_cast<std::size_t>(t)];
            if (static_cast<int>(b.fx.size()) < rows) {
                b.fx.resize(static_cast<std::size_t>(rows), 0.0);
                b.fy.resize(static_cast<std::size_t>(rows), 0.0);
                b.fz.resize(static_cast<std::size_t>(rows), 0.0);
            }
        }
    }

    /// @brief Open (or join) this slot's accumulation on `frame`. Call once
    ///        per launch, serially, before the parallel region.
    /// @param frame Frame whose force columns receive the flush.
    /// @param forces_requested The launch's force request; a pass opened
    ///        without it is discarded by `flush()`.
    /// @return The team size a kernel should open (OpenMP max threads).
    int bind(cpu::Frame& frame, bool forces_requested) {
        if (frame_ != nullptr && frame_ != &frame)
            flush();
        // Re-resolved every launch: a structural mutation (capacity growth,
        // schema change) between launches invalidates the previous spans.
        frame_ = &frame;
        fx_frame_ = frame.span(kAtom_fx);
        fy_frame_ = frame.span(kAtom_fy);
        fz_frame_ = frame.span(kAtom_fz);
        discard_ = !forces_requested;
        const int rows = frame.n_total();
        int team = 1;
#ifdef _OPENMP
        team = omp_get_max_threads();
#endif
        if (team > 1 && static_cast<int>(bufs_.size()) < team) {
            bufs_.resize(static_cast<std::size_t>(team));
            live_.reserve(bufs_.size());
        }
        if (rows > rows_) {
            // A buffer already live in this pass must cover the new rows.
            for (Buffer& b : bufs_)
                if (b.pass == pass_) {
                    grow_(b, rows);
                    std::fill(b.fx.begin() + rows_, b.fx.begin() + rows, 0.0);
                    std::fill(b.fy.begin() + rows_, b.fy.begin() + rows, 0.0);
                    std::fill(b.fz.begin() + rows_, b.fz.begin() + rows, 0.0);
                }
            rows_ = rows;
        }
        return team;
    }

    /// @brief Thread `tid`'s force pointers inside the parallel region
    ///        opened after `bind()`. A team of one gets the frame columns.
    ThreadForce thread(int tid) {
        int team = 1;
#ifdef _OPENMP
        team = omp_get_num_threads();
#endif
        if (team == 1)
            return {fx_frame_.data(), fy_frame_.data(), fz_frame_.data()};
        Buffer& b = bufs_[static_cast<std::size_t>(tid)];
        if (b.pass != pass_) {
            // Each thread owns a distinct buffer, so this is race-free.
            grow_(b, rows_);
            std::fill(b.fx.begin(), b.fx.begin() + rows_, 0.0);
            std::fill(b.fy.begin(), b.fy.begin() + rows_, 0.0);
            std::fill(b.fz.begin(), b.fz.begin() + rows_, 0.0);
            b.pass = pass_;
        }
        return {b.fx.data(), b.fy.data(), b.fz.data()};
    }

    /// @brief True when some thread buffer holds unreduced forces.
    bool pending() const noexcept {
        for (const Buffer& b : bufs_)
            if (b.pass == pass_)
                return true;
        return false;
    }

    /// @brief Reduce every buffer touched since the last flush into the bound
    ///        frame's force columns (or drop it for an energy-only pass), and
    ///        close the pass. Runtime-only; a no-op when nothing is pending.
    void flush() {
        live_.clear();
        for (std::size_t t = 0; t < bufs_.size(); ++t)
            if (bufs_[t].pass == pass_)
                live_.push_back(static_cast<int>(t));
        const int n_live = static_cast<int>(live_.size());
        if (n_live > 0 && !discard_) {
            const int* live = live_.data();
            double* __restrict fx = fx_frame_.data();
            double* __restrict fy = fy_frame_.data();
            double* __restrict fz = fz_frame_.data();
            const int rows = rows_;
            const Buffer* bufs = bufs_.data();
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
        frame_ = nullptr;
        fx_frame_ = Span<AccReal>{};
        fy_frame_ = Span<AccReal>{};
        fz_frame_ = Span<AccReal>{};
        ++pass_;
        rows_ = 0;
        discard_ = false;
    }

    std::size_t memory_bytes() const noexcept {
        std::size_t bytes = 0;
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

    static void grow_(Buffer& b, int rows) {
        if (static_cast<int>(b.fx.size()) < rows) {
            b.fx.resize(static_cast<std::size_t>(rows), 0.0);
            b.fy.resize(static_cast<std::size_t>(rows), 0.0);
            b.fz.resize(static_cast<std::size_t>(rows), 0.0);
        }
    }

    std::vector<Buffer> bufs_;
    std::vector<int> live_;  ///< flush scratch; capacity follows bufs_.
    std::uint64_t pass_ = 1; ///< buffers with `pass == pass_` are live.
    int rows_ = 0;           ///< rows zeroed in every live buffer.
    bool discard_ = false;   ///< the open pass requested no forces.
    cpu::Frame* frame_ = nullptr;
    Span<AccReal> fx_frame_{}, fy_frame_{}, fz_frame_{};
};

/// @brief One launch's view of the force accumulators: the Runtime's
///        `ThreadForces` when published, otherwise the frame columns with a
///        team of one (a component driven without a Runtime runs serially).
class ForcePass {
  public:
    explicit ForcePass(const cpu::LaunchContext& ctx)
        : arena_(ctx.runtime.md.thread_forces) {
        if (arena_ != nullptr) {
            team_ = arena_->bind(ctx.frame, ctx.forces_requested);
        } else {
            direct_ = {ctx.frame.span(kAtom_fx).data(),
                       ctx.frame.span(kAtom_fy).data(),
                       ctx.frame.span(kAtom_fz).data()};
        }
    }

    /// @brief Threads the kernel's parallel region should open.
    int team() const noexcept { return team_; }

    /// Terms below which one more thread costs more (wake-up, barrier, and
    /// its buffer's zero + reduce sweep) than it saves.
    static constexpr int kMinTermsPerThread = 2048;

    /// @brief Threads worth opening for `n_terms` terms, at most `team()`. A
    ///        team of one writes the frame directly, so a small topology
    ///        does not wake the whole team every evaluation.
    int team(int n_terms) const noexcept {
        return std::clamp(n_terms / kMinTermsPerThread, 1, team_);
    }

    /// @brief Thread `tid`'s force pointers (call inside the region).
    ThreadForce thread(int tid) const {
        return arena_ != nullptr ? arena_->thread(tid) : direct_;
    }

  private:
    ThreadForces* arena_;
    int team_ = 1;
    ThreadForce direct_{nullptr, nullptr, nullptr};
};

/// @brief Voigt-order (xx yy zz xy xz yz) virial accumulator. Supports
///        `#pragma omp declare reduction(+ : Voigt6 : omp_out += omp_in)`
///        so kernels reduce one object instead of six scalars.
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
