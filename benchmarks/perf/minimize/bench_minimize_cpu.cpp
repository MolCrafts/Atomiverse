/// @file benchmarks/perf/minimize/bench_minimize_cpu.cpp
/// @brief CPU minimizer timing: optimizer/driver overhead and force-call counts.
///
/// Two force providers:
///   - `rosen`: chained Rosenbrock over the 3N coordinates. O(N) with a tiny
///     constant, so the wall time is the optimizer + driver overhead (vector
///     updates, history, line search, metrics, Frame access).
///   - `lj`: perturbed fcc Lennard-Jones (PairLJCut + NeighborList), the
///     realistic cheap classical case.
/// Both count force-phase launches split into initial / line-search trial /
/// committed, and declare themselves trial-invariant so trial-force reuse
/// stays active. The criterion never converges, so every run executes the
/// requested budget unless the optimizer stops itself (reported).
///
/// Prints one line per repeat set:
///   BENCH model=.. N=.. style=.. iters=.. best_s=.. us_per_iter=..
///         force_us_per_iter=.. overhead_us_per_iter=.. initial=.. trials=..
///         committed=.. energy=%.17g
/// `force_us_per_iter` is wall inside the force slot (the initial force call
/// included); overhead = total - force, so for `lj` it also holds the
/// neighbor-list decision/rebuild that runs before the slot.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/log.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/potential.h"
#include "atomiverse/cpu/driver/criteria.h"
#include "atomiverse/cpu/minimize/cg.h"
#include "atomiverse/cpu/minimize/fire.h"
#include "atomiverse/cpu/minimize/lbfgs.h"
#include "atomiverse/cpu/minimize/minimize.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

namespace {

using namespace ATV;

using Clock = std::chrono::steady_clock;

/// Force-phase launch counter and in-slot force time, shared by both models.
struct Counts {
    long initial = 0, trials = 0, committed = 0;
    double force_s =
        0; ///< Wall inside the force slot (excludes neighbor/comm).
    Clock::time_point opened{};
    void tick(const cpu::LaunchContext& ctx) {
        if (ctx.runtime.exec.initializing_forces)
            ++initial;
        else if (ctx.runtime.exec.trial_evaluation)
            ++trials;
        else
            ++committed;
    }
};

/// Chained Rosenbrock, f = sum_k [a (q_{k+1} - q_k^2)^2 + (1 - q_k)^2] over
/// q = (x0, y0, z0, x1, ...). Pure function of the Frame.
class Rosenbrock : public cpu::Potential {
  public:
    explicit Rosenbrock(Counts& c) : counts_(c) {}
    std::string name() const override { return "rosenbrock"; }
    bool trial_invariant() const override { return true; }
    void launch(const cpu::LaunchContext& ctx) override {
        counts_.tick(ctx);
        const auto t0 = Clock::now();
        auto& f = ctx.frame;
        const int n = f.nlocal();
        const StorageReal* x = f.span(kAtom_x);
        const StorageReal* y = f.span(kAtom_y);
        const StorageReal* z = f.span(kAtom_z);
        AccReal* fr[3] = {f.span(kAtom_fx), f.span(kAtom_fy), f.span(kAtom_fz)};
        const StorageReal* r[3] = {x, y, z};
        const int dim = 3 * n;
        auto q = [&](int k) { return r[k % 3][k / 3]; };
        double e = 0;
        for (int k = 0; k + 1 < dim; ++k) {
            const double qk = q(k), qn = q(k + 1);
            const double t = qn - qk * qk, u = 1.0 - qk;
            e += kA * t * t + u * u;
            fr[k % 3][k / 3] -= -4.0 * kA * t * qk - 2.0 * u;
            fr[(k + 1) % 3][(k + 1) / 3] -= 2.0 * kA * t;
        }
        f.meta.get(kEnergy) += e;
        counts_.force_s +=
            std::chrono::duration<double>(Clock::now() - t0).count();
    }

  private:
    static constexpr double kA = 10.0;
    Counts& counts_;
};

/// Brackets the LJ force slot without touching the frame: the opening probe
/// counts the launch and starts the clock, the closing one stops it.
class Probe : public cpu::Component {
  public:
    Probe(Counts& c, bool opening) : counts_(c), opening_(opening) {}
    std::string name() const override {
        return opening_ ? "probe_open" : "probe_close";
    }
    bool trial_invariant() const override { return true; }
    void launch(const cpu::LaunchContext& ctx) override {
        if (opening_) {
            counts_.tick(ctx);
            counts_.opened = Clock::now();
            return;
        }
        counts_.force_s +=
            std::chrono::duration<double>(Clock::now() - counts_.opened)
                .count();
    }

  private:
    Counts& counts_;
    bool opening_;
};

cpu::Frame rosen_frame(int n) {
    cpu::Frame f(n);
    for (int i = 0; i < n; ++i) {
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_type)[i] = 1;
        f.span(kAtom_mass)[i] = 1.0;
        f.span(kAtom_x)[i] = -0.6 + 0.3 * std::sin(1.7 * i);
        f.span(kAtom_y)[i] = 0.4 + 0.3 * std::sin(2.3 * i);
        f.span(kAtom_z)[i] = 0.9 + 0.3 * std::sin(3.1 * i);
    }
    f.box = cpu::Box::cubic(1000.0);
    return f;
}

constexpr double kLjCutoff = 2.5;

cpu::Frame lj_frame(int side) {
    const int n = 4 * side * side * side;
    const double a = std::cbrt(4.0 / 0.8); // fcc at reduced density 0.8
    const double basis[4][3] = {
        {0, 0, 0}, {0, .5, .5}, {.5, 0, .5}, {.5, .5, 0}};
    cpu::Frame f(n);
    f.box = cpu::Box::cubic(side * a);
    for (int i = 0; i < n; ++i) {
        const int cell = i / 4, b = i % 4;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_type)[i] = 1;
        f.span(kAtom_mass)[i] = 1.0;
        f.span(kAtom_x)[i] = (cell / (side * side) + basis[b][0] + 0.25
                              + 0.04 * std::sin(i * 1.7))
                             * a;
        f.span(kAtom_y)[i] = ((cell / side) % side + basis[b][1] + 0.25
                              + 0.04 * std::sin(i * 2.3))
                             * a;
        f.span(kAtom_z)[i] =
            (cell % side + basis[b][2] + 0.25 + 0.04 * std::sin(i * 3.1)) * a;
    }
    return f;
}

struct Result {
    double seconds = 0;
    long iters = 0;
    Counts counts;
    double energy = 0;
};

Result run_once(const std::string& model, int size, const std::string& style,
                int iters, int threads) {
    Result res;
    cpu::Frame frame = model == "lj" ? lj_frame(size) : rosen_frame(size);
    cpu::minimizer::Minimize opt;
    if (threads > 0)
        opt.set_threads(threads);
    if (style == "lbfgs")
        opt.set_optimizer(cpu::minimizer::LBFGS{});
    else if (style == "cg")
        opt.set_optimizer(cpu::minimizer::CG{});
    else if (style == "fire")
        opt.set_optimizer(cpu::minimizer::FIRE{});
    else
        throw std::invalid_argument("style must be lbfgs, cg or fire");
    Rosenbrock rosen(res.counts);
    Probe open(res.counts, true), close(res.counts, false);
    if (model == "lj") {
        opt.set_evaluator(cpu::PairLJCut{1.0, 1.0, kLjCutoff});
        opt.set_neighbor(cpu::md::NeighborList<>{kLjCutoff, 0.3, 1, 0, true});
        opt.on(ATV::minimizer::Phase::force).add_ref(open, -2000);
        opt.on(ATV::minimizer::Phase::force).add_ref(close, 1000);
    } else {
        opt.on(ATV::minimizer::Phase::force).add_ref(rosen, 0);
    }
    opt.add_criterion(cpu::ForceConverged{0.0, 0.0, 0.0, 0.0, 0.0});
    opt.build();
    const auto t0 = Clock::now();
    res.iters = opt.run(frame, iters);
    const auto t1 = Clock::now();
    res.seconds = std::chrono::duration<double>(t1 - t0).count();
    res.energy = frame.meta.get(kEnergy);
    return res;
}

} // namespace

int main(int argc, char** argv) {
    try {
        // The per-step [info] lines would time the console, not the optimizer.
        ATV::log_init(ATV::LogConfig{.level = spdlog::level::warn});
        if (argc != 7)
            throw std::invalid_argument(
                "usage: bench_minimize_cpu rosen|lj N_or_NSIDE lbfgs|cg|fire "
                "ITERS THREADS REPEATS");
        const std::string model = argv[1];
        const int size = std::stoi(argv[2]);
        const std::string style = argv[3];
        const int iters = std::stoi(argv[4]);
        const int threads = std::stoi(argv[5]);
        const int repeats = std::stoi(argv[6]);
        if (model != "rosen" && model != "lj")
            throw std::invalid_argument("model must be rosen or lj");
        if (size < 1 || iters < 1 || repeats < 1)
            throw std::invalid_argument("N, ITERS and REPEATS must be >= 1");
        Result best;
        best.seconds = std::numeric_limits<double>::infinity();
        for (int r = 0; r < repeats; ++r) {
            Result res = run_once(model, size, style, iters, threads);
            if (res.seconds < best.seconds)
                best = res;
        }
        const int n = model == "lj" ? 4 * size * size * size : size;
        const double per = 1e6 / std::max(1L, best.iters);
        std::printf("BENCH model=%s N=%d style=%s iters=%ld best_s=%.6f "
                    "us_per_iter=%.3f force_us_per_iter=%.3f "
                    "overhead_us_per_iter=%.3f initial=%ld trials=%ld "
                    "committed=%ld energy=%.17g\n",
                    model.c_str(), n, style.c_str(), best.iters, best.seconds,
                    per * best.seconds, per * best.counts.force_s,
                    per * (best.seconds - best.counts.force_s),
                    best.counts.initial, best.counts.trials,
                    best.counts.committed, best.energy);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "bench_minimize_cpu: %s\n", e.what());
        return 1;
    }
}
