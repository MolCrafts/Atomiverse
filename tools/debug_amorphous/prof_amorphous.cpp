// @file tools/debug_amorphous/prof_amorphous.cpp
// @brief Per-component profiling + NL internals breakdown for amorphous PE.
#include <mpi.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/dynamics/fix_langevin.h"
#include "atomiverse/cpu/dynamics/init.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/nve_limit.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/cell_list.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/neighbor/pair_list.h"
#include "atomiverse/cpu/observe/virial_stress.h"
#include "atomiverse/cpu/potentials/angle_harmonic.h"
#include "atomiverse/cpu/potentials/bond_harmonic.h"
#include "atomiverse/cpu/potentials/dihedral_multi_harmonic.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

#include "lammps_loader.h"

using namespace ATV;
using clock_t_ = std::chrono::steady_clock;

struct Bin {
    long long n = 0;
    double t = 0.0;
};
static std::unordered_map<std::string, Bin> g;
static std::vector<std::string> g_order;
static void record(const std::string& k, double t) {
    auto it = g.find(k);
    if (it == g.end())
        g_order.push_back(k);
    auto& b = g[k];
    b.n++;
    b.t += t;
}

class TimedWrapper : public cpu::Component {
  public:
    TimedWrapper(std::unique_ptr<cpu::Component> inner, std::string label,
                 int phase, int order)
        : inner_(std::move(inner)), label_(std::move(label)), phase_(phase),
          order_(order) {}
    std::string name() const override { return label_; }
    std::vector<DefaultBinding> default_bindings() const override {
        // Profiling harness is MD-only: callers pass static_cast<int>(md::Phase).
        return {ATV::make_binding(static_cast<ATV::md::Phase>(phase_), order_)};
    }
    void build(const cpu::BuildContext& ctx) override { inner_->build(ctx); }
    void destroy(const cpu::DestroyContext& ctx) override {
        inner_->destroy(ctx);
    }
    void launch(const cpu::LaunchContext& ctx) override {
        auto t0 = clock_t_::now();
        inner_->launch(ctx);
        auto t1 = clock_t_::now();
        record(label_, std::chrono::duration<double>(t1 - t0).count());
    }

  private:
    std::unique_ptr<cpu::Component> inner_;
    std::string label_;
    int phase_, order_;
};

// Instrumented NL that breaks time into bin_atoms / refresh_half. Mirrors
// LAMMPS `neigh_modify every N delay 0 check yes`: rebuild at most every
// `every` steps and only if the skin scan fires.
class InstrNeighborList : public cpu::Component {
  public:
    InstrNeighborList(double cutoff, double skin, int every)
        : strategy_(cutoff, skin), cutoff_(cutoff),
          every_(every < 1 ? 1 : every) {}
    std::string name() const override { return "neighbor_list"; }
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::force, -10)};
    }
    void build(const cpu::BuildContext& ctx) override {
        strategy_.init(ctx.frame, cutoff_);
        strategy_.build(ctx.frame, ctx.frame.nlocal());
        strategy_.refresh_half(ctx.frame, ctx.frame.nlocal(), pairs_);
        ++pairs_.version;
        ctx.runtime.md.pair_list = &pairs_;
        ago_ = 0;
    }
    void launch(const cpu::LaunchContext& ctx) override {
        const int n = ctx.frame.nlocal();
        ++ago_;
        if (ago_ < every_) {
            ctx.runtime.md.pair_list = &pairs_;
            return;
        }
        auto t0 = clock_t_::now();
        bool need = strategy_.needs_rebuild(ctx.frame, n);
        auto t1 = clock_t_::now();
        record("NL.needs_rebuild",
               std::chrono::duration<double>(t1 - t0).count());
        if (need) {
            auto ta = clock_t_::now();
            strategy_.build(ctx.frame, n); // bin_atoms + save_positions
            auto tb = clock_t_::now();
            strategy_.refresh_half(ctx.frame, n, pairs_);
            auto tc = clock_t_::now();
            ++pairs_.version;
            record("NL.bin+save",
                   std::chrono::duration<double>(tb - ta).count());
            record("NL.refresh_half",
                   std::chrono::duration<double>(tc - tb).count());
            ago_ = 0;
        }
        ctx.runtime.md.pair_list = &pairs_;
    }

  private:
    cpu::md::CellList strategy_;
    cpu::md::PairList pairs_;
    double cutoff_;
    int every_;
    int ago_ = 0;
};

// Eager NL (mimics the *old* pre-lazy path) -- always update + refresh_half
// every step. Used to reproduce the "refresh-every-step" cost for comparison.
class EagerNLInstr : public cpu::Component {
  public:
    explicit EagerNLInstr(double cutoff, double skin = 0.4)
        : strategy_(cutoff, skin), cutoff_(cutoff) {}
    std::string name() const override { return "neighbor_list"; }
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::force, -10)};
    }
    void build(const cpu::BuildContext& ctx) override {
        strategy_.init(ctx.frame, cutoff_);
        strategy_.build(ctx.frame, ctx.frame.nlocal());
        strategy_.refresh_half(ctx.frame, ctx.frame.nlocal(), pairs_);
        ++pairs_.version;
        ctx.runtime.md.pair_list = &pairs_;
    }
    void launch(const cpu::LaunchContext& ctx) override {
        const int n = ctx.frame.nlocal();
        auto t0 = clock_t_::now();
        strategy_.update(ctx.frame,
                         n); // always checks needs_rebuild → sometimes build()
        auto t1 = clock_t_::now();
        strategy_.refresh_half(ctx.frame, n,
                               pairs_); // ALWAYS re-runs full stencil
        auto t2 = clock_t_::now();
        ++pairs_.version;
        record("eager.update", std::chrono::duration<double>(t1 - t0).count());
        record("eager.refresh_half",
               std::chrono::duration<double>(t2 - t1).count());
        ctx.runtime.md.pair_list = &pairs_;
    }

  private:
    cpu::md::CellList strategy_;
    cpu::md::PairList pairs_;
    double cutoff_;
};

template <typename C>
std::unique_ptr<TimedWrapper> timed(C&& c, std::string label, int phase,
                                    int order) {
    auto up = std::make_unique<std::decay_t<C>>(std::forward<C>(c));
    return std::make_unique<TimedWrapper>(std::move(up), std::move(label),
                                          phase, order);
}

namespace params {
constexpr double kKcalPerMolToEv = 1.0 / 23.060547830619026;
constexpr double bond_K_kcal = 350.0;
constexpr double bond_r0 = 1.53;
constexpr double angle_K_kcal = 60.0;
constexpr double lj_eps_kcal = 0.112;
constexpr double lj_sig = 4.01;
constexpr double lj_cut = 10.5;
inline double deg2rad(double d) { return d * units::pi / 180.0; }
} // namespace params

void print_and_clear(double total, int n_steps, const char* title) {
    std::fprintf(stderr, "\n=== %s ===\n", title);
    std::fprintf(stderr, "%-28s %10s %14s %12s %6s\n", "bucket", "launches",
                 "total_s", "avg_us", "%");
    for (auto& k : g_order) {
        auto& b = g[k];
        double pct = 100.0 * b.t / total;
        double us = 1e6 * b.t / std::max<long long>(1, b.n);
        std::fprintf(stderr, "%-28s %10lld %14.6f %12.1f %5.1f%%\n", k.c_str(),
                     b.n, b.t, us, pct);
    }
    std::fprintf(stderr, "TOTAL wall = %.3f s, %.2f steps/s\n", total,
                 n_steps / total);
    g.clear();
    g_order.clear();
}

enum Mode { EAGER, LAZY };

void run_scenario(const std::string& data_path, int n_steps, Mode mode,
                  std::vector<double>& dihedral_ev, double bond_K_ev,
                  double angle_K_ev, double angle_theta0, double lj_eps_ev) {
    cpu::Frame frame = debug_amorphous::load_lammps_data(data_path);

    cpu::md::MD sim;
    sim.set_integrator(cpu::md::NVELimit{1.0, 0.05});
    sim.set_observer(cpu::md::InitVelocity{500.0, 1231});
    sim.set_thermostat(
        cpu::md::FixLangevin{500.0, /*T_stop=*/500.0, 10.0, 1.0, 904297});
    sim.set_observer(cpu::md::Virial{});

    const int ph_force = static_cast<int>(ATV::md::Phase::force);

    std::unique_ptr<cpu::Component> raw_nl;
    if (mode == EAGER)
        raw_nl = std::make_unique<EagerNLInstr>(params::lj_cut, 0.4);
    else
        // LAMMPS-matched: cutoff=10.5, skin=0.4, every=10 (from `neigh_modify`).
        raw_nl = std::make_unique<InstrNeighborList>(params::lj_cut, 0.4, 10);
    sim.set_observer_ref(*raw_nl);

    auto lj = timed(cpu::PairLJCut{lj_eps_ev, params::lj_sig, params::lj_cut}
                        .set_special_bonds(0.0, 0.0, 0.0),
                    "lj_pair", ph_force, 0);
    sim.set_observer_ref(*lj);
    auto hb = timed(cpu::BondHarmonic{{bond_K_ev}, {params::bond_r0}},
                    "harm_bond", ph_force, 1);
    sim.set_observer_ref(*hb);
    auto ha = timed(cpu::AngleHarmonic{{angle_K_ev}, {angle_theta0}},
                    "harm_angle", ph_force, 2);
    sim.set_observer_ref(*ha);
    auto hd = timed(cpu::DihedralMultiHarmonic{{dihedral_ev}}, "dihedral",
                    ph_force, 3);
    sim.set_observer_ref(*hd);

    sim.build();
    auto t0 = clock_t_::now();
    sim.run(frame, n_steps);
    auto t1 = clock_t_::now();
    double total = std::chrono::duration<double>(t1 - t0).count();
    print_and_clear(total, n_steps,
                    mode == EAGER
                        ? "EAGER (current code path)"
                        : "LAZY (skip refresh_half when disp < skin/2)");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    std::string data_path = "/Users/roykid/work/molcrafts/Atomiverse/tools/"
                            "debug_amorphous/PE_nc10_cl1000.dat";
    int n_steps = 200;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--steps" && i + 1 < argc)
            n_steps = std::stoi(argv[++i]);
        else if (a == "--data" && i + 1 < argc)
            data_path = argv[++i];
    }

    const double bond_K_ev = params::bond_K_kcal * params::kKcalPerMolToEv;
    const double angle_K_ev = params::angle_K_kcal * params::kKcalPerMolToEv;
    const double lj_eps_ev = params::lj_eps_kcal * params::kKcalPerMolToEv;
    const std::vector<double> dihedral_kcal = {1.73, -4.49, 0.776, 6.99, 0.0};
    std::vector<double> dihedral_ev;
    for (double v : dihedral_kcal)
        dihedral_ev.push_back(v * params::kKcalPerMolToEv);
    const double angle_theta0 = params::deg2rad(109.5);

    run_scenario(data_path, n_steps, EAGER, dihedral_ev, bond_K_ev, angle_K_ev,
                 angle_theta0, lj_eps_ev);
    run_scenario(data_path, n_steps, LAZY, dihedral_ev, bond_K_ev, angle_K_ev,
                 angle_theta0, lj_eps_ev);

    MPI_Finalize();
    return 0;
}
