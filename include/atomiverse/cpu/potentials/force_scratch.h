/// @file cpu/potentials/force_scratch.h
/// @brief Shared scaffolding for bonded/pair force kernels: per-thread force
///        accumulators and a Voigt-order virial accumulator.
///
/// Every bonded and pair kernel previously copy-pasted: (1) a triple of
/// per-thread `std::vector<double>` force buffers allocated in build(),
/// zeroed at the top of compute(), scattered into in the hot loop (so two
/// threads never write the same row), and reduced back into `frame` forces;
/// and (2) six `v_xx..v_yz` virial scalars + an unpack into `stress[0..5]`.
/// This header factors both out.
///
/// CRITICAL: the zero/reduce passes iterate `n_force = n_local + n_ghost`
/// (from `ResolveView::n_force()`), NOT `Frame::n_total()`. In multi-rank
/// `Frame::nlocal()` is the global atom count, so the old bound
/// passes did O(global N) work on every rank — the measured ~100x bonded
/// straggler. Buffers are still *allocated* to `nmax` (capacity), but only
/// the live `[0, n_force)` rows are touched per step.
///
/// Non-virtual by design: `Bond`/`Angle`/`Dihedral` forbid virtual calls in
/// the per-tuple loop, so this is a plain owned member, not a base mixin.
#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "atomiverse/core/log.h"
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

/// @brief Owns per-thread force accumulators; zero/reduce sized to n_force.
///
/// Single-thread fast path: when `n_threads == 1` (the pure-MPI case — one
/// rank per core, the production multi-node layout), there is no write race,
/// so the kernel accumulates straight into the frame's force arrays. That
/// elides BOTH the per-step zero pass (frame forces are already cleared by
/// `ZeroForces` on owned rows + `zero_ghost_forces` on ghost rows) AND the
/// reduce pass — two full O(n_force) array sweeps per kernel per step that
/// LAMMPS never pays. With three bonded kernels (bond/angle/dihedral) those
/// six sweeps were a measurable slice of the bonded-force gap vs LAMMPS.
class BondedForceScratch {
  public:
    /// @brief Allocate `n_threads` buffers of `nmax` rows each (call in
    ///        build()). With a single thread no scratch is allocated — the
    ///        kernel writes directly into the frame (see `begin_thread`).
    void build(int n_threads, int nmax) {
        n_threads_ = n_threads < 1 ? 1 : n_threads;
        direct_ = (n_threads_ == 1);
        launch_direct_ = direct_;
        if (direct_) {
            fx_.clear();
            fy_.clear();
            fz_.clear();
            return;
        }
        fx_.assign(static_cast<std::size_t>(n_threads_),
                   std::vector<double>(static_cast<std::size_t>(nmax), 0.0));
        fy_ = fx_;
        fz_ = fx_;
    }

    int n_threads() const noexcept { return n_threads_; }

    /// Terms below which one more thread costs more (wake-up, barrier, and
    /// its zero + reduce sweep over every force row) than it saves.
    static constexpr int kMinTuplesPerThread = 2048;

    /// @brief Threads worth using for `n_tuples` terms this launch, at most
    ///        `n_threads()`. Call before the parallel region and pass the
    ///        result to `num_threads(...) if (team > 1)`: a team of one makes
    ///        this launch direct (frame arrays, no zero/reduce), so a small
    ///        topology does not wake the whole team every evaluation.
    int team(int n_tuples) noexcept {
        const int by_work = n_tuples / kMinTuplesPerThread;
        const int t = std::clamp(by_work, 1, n_threads_ < 1 ? 1 : n_threads_);
        launch_direct_ = direct_ || t == 1;
        return t;
    }

    std::size_t memory_bytes() const noexcept {
        std::size_t bytes = 0;
        for (const auto* arrays : {&fx_, &fy_, &fz_})
            for (const auto& buffer : *arrays)
                bytes += buffer.capacity() * sizeof(double);
        return bytes;
    }

    /// @brief Resolve the frame's fx/fy/fz spans ONCE per launch, before the
    ///        OMP parallel region opens (never inside it — the span resolution
    ///        counter is not thread-safe, and every thread would otherwise
    ///        resolve the same key). Call at the top of `compute()`, after any
    ///        `n_force`-growth check and before `#pragma omp parallel`.
    void bind(cpu::Frame& f) {
        fx_frame_ = f.span(kAtom_fx);
        fy_frame_ = f.span(kAtom_fy);
        fz_frame_ = f.span(kAtom_fz);
    }

    /// @brief Begin thread `tid`'s accumulation over `[0, n_force)` and return
    ///        its force pointers. Single-thread direct mode returns the frame's
    ///        own force arrays (already resolved by `bind`, no zeroing — the
    ///        arrays are already cleared upstream); the multi-thread path
    ///        zeros this thread's private buffer first. Call once per thread
    ///        at the top of the parallel region, after `bind()`.
    ThreadForce begin_thread(int tid, int n_force) {
        if (tid == 0) {
#ifdef _OPENMP
            active_threads_ = omp_get_num_threads();
#else
            active_threads_ = 1;
#endif
        }
        if (launch_direct_) {
            // A caller that forgot `bind(f)` would otherwise get null force
            // pointers and silently drop every bonded force. Fail loudly.
            if (!fx_frame_)
                throw std::logic_error(
                    "BondedForceScratch: begin_thread() before bind(frame) — "
                    "resolve the frame's force columns once, outside the "
                    "parallel region");
            return {fx_frame_.data(), fy_frame_.data(), fz_frame_.data()};
        }
        auto& fx = fx_[static_cast<std::size_t>(tid)];
        auto& fy = fy_[static_cast<std::size_t>(tid)];
        auto& fz = fz_[static_cast<std::size_t>(tid)];
        // Grow this thread's buffers if n_force (n_local + n_ghost) outgrew the
        // build-time size — e.g. when a thin-subdomain halo overflow enlarged
        // the frame after build(). Each thread owns a distinct buffer, so the
        // resize is race-free inside the parallel region. (Rare — steady state
        // never reallocates, matching the no-alloc-in-launch contract.)
        if (static_cast<int>(fx.size()) < n_force) {
            fx.resize(static_cast<std::size_t>(n_force), 0.0);
            fy.resize(static_cast<std::size_t>(n_force), 0.0);
            fz.resize(static_cast<std::size_t>(n_force), 0.0);
        }
        std::fill(fx.begin(), fx.begin() + n_force, 0.0);
        std::fill(fy.begin(), fy.begin() + n_force, 0.0);
        std::fill(fz.begin(), fz.begin() + n_force, 0.0);
        return {fx.data(), fy.data(), fz.data()};
    }

    /// @brief Reduce all threads' `[0, n_force)` contributions into frame
    ///        forces. Call after the parallel region, using the same spans
    ///        `bind()` resolved at the top of this launch. The owned + ghost
    ///        rows are both reduced (ghost forces ship home via CommBrick).
    void reduce_into(int n_force) const {
        // Single-thread direct mode wrote straight into the frame — nothing to
        // gather.
        if (launch_direct_)
            return;
        double* __restrict fx = fx_frame_.data();
        double* __restrict fy = fy_frame_.data();
        double* __restrict fz = fz_frame_.data();
        // Parallelize over rows (each row independent — no cross-thread write
        // hazard), gathering every thread's contribution for row i.
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n_force; ++i) {
            double sx = 0.0, sy = 0.0, sz = 0.0;
            for (int t = 0; t < active_threads_; ++t) {
                sx += fx_[static_cast<std::size_t>(t)]
                         [static_cast<std::size_t>(i)];
                sy += fy_[static_cast<std::size_t>(t)]
                         [static_cast<std::size_t>(i)];
                sz += fz_[static_cast<std::size_t>(t)]
                         [static_cast<std::size_t>(i)];
            }
            fx[i] += sx;
            fy[i] += sy;
            fz[i] += sz;
        }
    }

    void clear() {
        fx_.clear();
        fy_.clear();
        fz_.clear();
        n_threads_ = 0;
        direct_ = false;
        launch_direct_ = false;
        fx_frame_ = Span<AccReal>{};
        fy_frame_ = Span<AccReal>{};
        fz_frame_ = Span<AccReal>{};
    }

  private:
    int n_threads_ = 0;
    int active_threads_ = 1;
    bool direct_ = false; ///< single-thread: accumulate straight into frame.
    /// This launch's mode: `direct_`, or a team of one chosen by `team()`.
    bool launch_direct_ = false;
    std::vector<std::vector<double>> fx_, fy_, fz_;
    /// Frame fx/fy/fz spans, resolved once per launch by `bind()`.
    Span<AccReal> fx_frame_{}, fy_frame_{}, fz_frame_{};
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
