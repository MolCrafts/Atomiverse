/// @file benchmarks/perf/md/bench_lj_cpu.cpp
/// @brief Public-API CPU LJ benchmark; never registered with ctest.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iomanip>
#include <random>
#include <stdexcept>
#include <string>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/observe/virial_stress.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"
#ifdef ATV_HAS_MOLPY
#include "atomiverse/cpu/potentials/eam.h"
#include "atomiverse/cpu/potentials/sw.h"
#include "atomiverse/cpu/potentials/sw_data/sw_tables.h"
#endif
#ifdef ATV_HAS_MPI
#include <mpi.h>
#endif

namespace {
// A generic workflow exercises native nested Driver/ScopedPlan execution without
// introducing a second MD implementation or changing the benchmark's force path.
class NestedWorkflow : public ATV::cpu::Driver<ATV::scf::Phase> {
  public:
    void run(ATV::cpu::Frame& frame) {
        setup_pass(frame);
        step_once(0);
        teardown_pass();
    }

  protected:
    void validate_components() const override {}
};

class Snapshot : public ATV::cpu::Component {
  public:
    explicit Snapshot(std::function<void(const ATV::cpu::Frame&)> write)
        : write_(std::move(write)) {}
    std::string name() const override { return "BenchmarkSnapshot"; }
    void launch(const ATV::cpu::LaunchContext& ctx) override {
        write_(ctx.frame);
    }

  private:
    std::function<void(const ATV::cpu::Frame&)> write_;
};
} // namespace

int main(int argc, char** argv) {
#ifdef ATV_HAS_MPI
    MPI_Init(&argc, &argv);
#endif
    int rank = 0;
#ifdef ATV_HAS_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    int result = 0;
    try {
        using namespace ATV;
        if (argc < 6 || argc > 12)
            throw std::invalid_argument(
                "usage: bench_lj_cpu NSIDE WARM STEPS THREADS OUTPUT_PREFIX "
                "[nested] [lj|eam|sw-lj] [--tilt XY XZ YZ]");
        const int side = std::stoi(argv[1]), warm = std::stoi(argv[2]);
        const int steps = std::stoi(argv[3]), threads = std::stoi(argv[4]);
        const std::string prefix = argv[5];
        int next = 6;
        const bool nested = next < argc && std::string(argv[next]) == "nested";
        if (nested)
            ++next;
        const std::string model =
            next < argc && std::string(argv[next]) != "--tilt" ? argv[next++]
                                                               : "lj";
        if (model != "lj" && model != "eam" && model != "sw-lj")
            throw std::invalid_argument("model must be lj, eam, or sw-lj");
        double tilt[3]{};
        const bool triclinic = next < argc;
        if (triclinic) {
            if (std::string(argv[next++]) != "--tilt" || argc - next != 3)
                throw std::invalid_argument(
                    "expected --tilt XY XZ YZ in Angstrom");
            for (double& value : tilt) {
                value = std::stod(argv[next++]);
                if (!std::isfinite(value))
                    throw std::invalid_argument("tilts must be finite");
            }
        }
#ifndef ATV_HAS_MOLPY
        if (model != "lj")
            throw std::invalid_argument(
                "many-body benchmarks require ATV_BUILD_MOLPY=ON");
#endif
        if (side < 1 || side > 200 || warm < 0 || steps < 1 || threads < 1)
            throw std::invalid_argument(
                "require 1 <= NSIDE <= 200, WARM >= 0, STEPS/THREADS > 0");
        const int basis_size = model == "eam" ? 4 : model == "sw-lj" ? 8 : 1;
        const int n = side * side * side * basis_size;
        const double spacing = model == "eam"     ? 3.61
                               : model == "sw-lj" ? 5.43
                                                  : std::cbrt(1.0 / 0.8);
        const double length = side * spacing;
        const double mass = model == "eam"     ? 63.546
                            : model == "sw-lj" ? 28.0855
                                               : 1.0;
        const double thermal_energy =
            model == "lj" ? 1.0 : 300.0 * 8.617333262145e-5;
        const double dt = model == "lj"
                              ? 0.005 * std::sqrt(units::UnitSystem::lj().mvv2e)
                          : model == "eam" ? 1.0
                                           : 0.5;
        const double basis[8][3] = {
            {0, 0, 0},       {0, .5, .5},     {.5, 0, .5},     {.5, .5, 0},
            {.25, .25, .25}, {.25, .75, .75}, {.75, .25, .75}, {.75, .75, .25}};
        cpu::Frame frame(n);
        Mat3 cell_matrix = Mat3::cubic(length);
        cell_matrix(0, 1) = tilt[0];
        cell_matrix(0, 2) = tilt[1];
        cell_matrix(1, 2) = tilt[2];
        frame.box.set_cell(cell_matrix);
        std::mt19937 random(87287);
        std::normal_distribution<double> velocity(
            0.0,
            std::sqrt(thermal_energy * units::UnitSystem::lj().ftm2v / mass));
        double mean[3]{};
        for (int i = 0; i < n; ++i) {
            frame.span(kAtom_tag)[i] = i + 1;
            frame.span(kAtom_type)[i] = 1;
            frame.span(kAtom_mass)[i] = mass;
            const int cell = i / basis_size, b = i % basis_size;
            const double offset = model == "lj" ? 0.5 : 0.05;
            const double distortion = model == "lj" ? 0.01 : 0.001;
            // Small deterministic distortions make the initial force gate meaningful.
            frame.span(kAtom_x)[i] = (cell / (side * side) + basis[b][0]
                                      + offset + distortion * std::sin(i * 1.7))
                                     * spacing;
            frame.span(kAtom_y)[i] = ((cell / side) % side + basis[b][1]
                                      + offset + distortion * std::sin(i * 2.3))
                                     * spacing;
            frame.span(kAtom_z)[i] = (cell % side + basis[b][2] + offset
                                      + distortion * std::sin(i * 3.1))
                                     * spacing;
            mean[0] += frame.span(kAtom_vx)[i] = velocity(random);
            mean[1] += frame.span(kAtom_vy)[i] = velocity(random);
            mean[2] += frame.span(kAtom_vz)[i] = velocity(random);
        }
        for (int i = 0; i < n; ++i) {
            frame.span(kAtom_vx)[i] -= mean[0] / n;
            frame.span(kAtom_vy)[i] -= mean[1] / n;
            frame.span(kAtom_vz)[i] -= mean[2] / n;
        }
        if (triclinic) {
            for (int i = 0; i < n; ++i) {
                const auto position =
                    frame.box.to_cartesian({frame.span(kAtom_x)[i] / length,
                                            frame.span(kAtom_y)[i] / length,
                                            frame.span(kAtom_z)[i] / length});
                frame.span(kAtom_x)[i] = position[0];
                frame.span(kAtom_y)[i] = position[1];
                frame.span(kAtom_z)[i] = position[2];
            }
        }
        if (rank == 0) {
            std::ofstream data(prefix + ".data");
            data.exceptions(std::ios::badbit | std::ios::failbit);
            data << std::setprecision(17)
                 << "Shared ATV/LAMMPS CPU state; metal units\n\n"
                 << n << " atoms\n1 atom types\n\n0 " << length
                 << " xlo xhi\n0 " << length << " ylo yhi\n0 " << length
                 << " zlo zhi\n";
            if (triclinic)
                data << tilt[0] << ' ' << tilt[1] << ' ' << tilt[2]
                     << " xy xz yz\n";
            data << "\nMasses\n\n1 " << mass << "\n\nAtoms # atomic\n\n";
            for (int i = 0; i < n; ++i)
                data << i + 1 << " 1 " << frame.span(kAtom_x)[i] << ' '
                     << frame.span(kAtom_y)[i] << ' ' << frame.span(kAtom_z)[i]
                     << '\n';
            data << "\nVelocities\n\n";
            for (int i = 0; i < n; ++i)
                data << i + 1 << ' ' << frame.span(kAtom_vx)[i] * 1000 << ' '
                     << frame.span(kAtom_vy)[i] * 1000 << ' '
                     << frame.span(kAtom_vz)[i] * 1000 << '\n';
        }
#ifdef ATV_HAS_MOLPY
        if (rank == 0 && model == "eam") {
            // Export the actual analytic potential to LAMMPS setfl. Fine grids
            // bound interpolation error; only unused r < 1 A is regularized.
            const cpu::EAMPotential potential(molrs::Element::Cu,
                                              "sutton-chen");
            constexpr double cutoff = 5.5, dr = 0.0001, drho = 0.005;
            constexpr int nr = 55001, nrho = 200001;
            std::ofstream table(prefix + ".eam");
            table.exceptions(std::ios::badbit | std::ios::failbit);
            table << std::setprecision(17)
                  << "Sutton-Chen Cu; DOI 10.1080/09500839008206493\n"
                  << "Analytic ATV parameters sampled for LAMMPS eam/alloy\n"
                  << "r<1 A regularized; benchmark never enters that interval\n"
                  << "1 Cu\n"
                  << nrho << ' ' << drho << ' ' << nr << ' ' << dr << ' '
                  << cutoff << "\n29 " << mass << " 3.61 FCC\n";
            for (int i = 0; i < nrho; ++i)
                table << potential.F(i * drho) << '\n';
            for (int i = 0; i < nr; ++i)
                table << potential.f(std::max(1.0, i * dr)) << '\n';
            for (int i = 0; i < nr; ++i)
                table << i * dr
                             * (potential.phi(std::max(1.0, i * dr))
                                - potential.phi(cutoff))
                      << '\n';
        }
        if (rank == 0 && model == "sw-lj") {
            const auto& p = cpu::sw_data::kStillingerWeber[0];
            std::ofstream table(prefix + ".sw");
            table.exceptions(std::ios::badbit | std::ios::failbit);
            table << std::setprecision(17)
                  << "# Original Si SW; DOI 10.1103/PhysRevB.31.5262\nSi Si Si "
                  << p.eps << ' ' << p.sigma << ' ' << p.a << ' ' << p.lambda
                  << ' ' << p.gamma << ' ' << p.cos_theta0 << ' ' << p.A << ' '
                  << p.B << ' ' << p.p << ' ' << p.q << " 0.0\n";
        }
#endif
        // Initial/final snapshots are outside the timed segment; each file contains
        // owners only, so the comparison script joins by tag across MPI ranks.
        const auto snapshot = [&](const cpu::Frame& state,
                                  const std::string& stage) {
            std::ofstream out(prefix + "." + stage + ".rank"
                              + std::to_string(rank) + ".csv");
            out.exceptions(std::ios::badbit | std::ios::failbit);
            out << std::setprecision(17) << "# energy "
                << state.meta.get(kEnergy) << '\n';
            out << "id,x,y,z,fx,fy,fz\n";
            const auto tags = state.span(kAtom_tag);
            const auto x = state.span(kAtom_x), y = state.span(kAtom_y);
            const auto z = state.span(kAtom_z), fx = state.span(kAtom_fx);
            const auto fy = state.span(kAtom_fy), fz = state.span(kAtom_fz);
            for (int i = 0; i < state.nlocal(); ++i)
                out << tags[i] << ',' << x[i] << ',' << y[i] << ',' << z[i]
                    << ',' << fx[i] << ',' << fy[i] << ',' << fz[i] << '\n';
        };
        cpu::md::MD sim;
        double cutoff = 2.5;
        if (model == "lj")
            sim.set_evaluator(cpu::PairLJCut{1., 1., cutoff});
#ifdef ATV_HAS_MOLPY
        else if (model == "eam") {
            cutoff = 5.5;
            sim.set_evaluator(cpu::EAM{
                cpu::EAMPotential{molrs::Element::Cu, "sutton-chen"}, cutoff});
        } else {
            const cpu::SWPotential potential(molrs::Element::Si,
                                             "stillinger-weber");
            cutoff = potential.cutoff();
            sim.set_evaluator(cpu::StillingerWeber{potential});
            sim.on(md::Phase::force).add(cpu::PairLJCut{0.01, 2.0, cutoff});
        }
#endif
        sim.set_neighbor(cpu::md::NeighborList<>{cutoff, 0.3, 1, 0, true});
        sim.set_integrator(cpu::md::VelocityVerlet{dt});
        sim.set_observer(cpu::md::Virial{});
        if (nested) {
            // Correctness-only mode: one continuous child trajectory, bracketed
            // by the same owner snapshots as the direct benchmark.
            NestedWorkflow workflow;
            workflow.set_threads(threads);
            workflow.on(scf::Phase::fock).add_ref(sim, 0, 0);
            workflow.on(scf::Phase::mix)
                .add(Snapshot{[&](const cpu::Frame& state) {
                    snapshot(state, "initial");
                }});
            workflow.on(scf::Phase::diagonalize).add_ref(sim, 0, warm + steps);
            workflow.build();
            workflow.run(frame);
            if (rank == 0)
                std::printf("CHECK engine=ATV mode=nested N=%d threads=%d "
                            "steps=%d energy=%.17g\n",
                            n, threads, warm + steps, frame.meta.get(kEnergy));
        } else {
            sim.set_threads(threads);
            sim.enable_timing(true);
            sim.build();
            sim.run(frame, 0);
            snapshot(frame, "initial");
            sim.run(frame, warm);
            sim.reset_timing();
            sim.run(frame, steps);
            const auto report = sim.timing_report();
            if (rank == 0)
                std::printf(
                    "BENCH engine=ATV model=%s N=%d ranks=%d threads=%d "
                    "warm=%d steps=%d "
                    "dt_fs=%.17g loop_s=%.9g matom_step_s=%.9g energy=%.17g\n",
                    model.c_str(), n, report.ranks, threads, warm, steps, dt,
                    report.loop_max, double(n) * steps / report.loop_max / 1e6,
                    frame.meta.get(kEnergy));
        }
        snapshot(frame, "final");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "rank %d: %s\n", rank, error.what());
        result = 1;
#ifdef ATV_HAS_MPI
        MPI_Abort(MPI_COMM_WORLD, result);
#endif
    }
#ifdef ATV_HAS_MPI
    MPI_Finalize();
#endif
    return result;
}
