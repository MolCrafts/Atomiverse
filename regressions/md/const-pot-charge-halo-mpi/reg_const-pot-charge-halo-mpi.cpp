/// @file regressions/md/const-pot-charge-halo-mpi/reg_const-pot-charge-halo-mpi.cpp
/// @brief END-TO-END: ghost-charge correctness after the const-pot Q solve
///        under MPI domain decomposition (2 ranks, planar walls).
///
/// See REFERENCE.md. Public headers + link atomiverse only. oracle=literature
/// (self-consistency invariant; LAMMPS fix electrode forward-communicates q
/// after every conp solve — Ahrens-Iwers et al., JCP 157, 084801 (2022)).
///
/// One binary, two modes:
///  - parent (default): re-launches itself through the configured mpiexec
///    (compile-time pin ATV_MPIEXEC) at np = 2, then gates
///      (a) max |q_ghost - q_owner| over ALL ghost rows, ranks, and steps
///          == 0 (<= 1e-15; the transport is memcpy — bitwise),
///      (b) electrode ghosts with |q| > 0 were actually seen (non-vacuous:
///          the stale-ghost failure mode is exercised — electrode charges
///          start at 0 and only the refreshed solve makes ghosts nonzero),
///      (c) ghost rows were checked on every step.
///  - child (`--child <out>`): MPI_Init, run a 38-atom parallel-plate
///    capacitor MD (ConstPotBO + PairCoulCut + NeighborList + VelocityVerlet)
///    for 5 steps with an out-of-library GhostChargeChecker Component bound
///    at the force phase; reduce the tallies; rank 0 writes one result line.
///
/// The charge refresh under test is posted by ConstPotBO itself after it
/// rewrites owned q (`ChargePacker` through `ctx.md.comm->forward_comm`).
/// Nothing here registers a CommBrick::forward_comm — the regression gates the default
/// MPI packer pipeline, not a hand-built one.

#include <mpi.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/precision.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/dynamics/const_pot_bo.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_coul_cut.h"

#ifndef ATV_MPIEXEC
#error "ATV_MPIEXEC must be set to the configured MPIEXEC_EXECUTABLE"
#endif

using namespace ATV;

namespace {

// ── Planar-wall capacitor (parallel plates ⊥ z, decomposed along x) ────
constexpr double kLx = 20.0, kLy = 10.0, kLz = 16.0; // [Å]
constexpr int kTypeBot = 1, kTypeTop = 2, kTypeIon = 3;
constexpr double kPsiBot = 0.0, kPsiTop = 1.0; // [V]
constexpr double kEta = 2.0;                   // [1/Å]
constexpr double kCut = 3.0;                   // Coulomb cutoff [Å]
constexpr double kSkin = 0.3;                  // NL skin [Å]
constexpr int kSteps = 5;
constexpr double kDt = 0.1; // [fs] — tiny; no migration over 5 steps

// Gates (see REFERENCE.md): transport is memcpy — expected exactly 0.
constexpr double kMaxDqTol = 1e-15; // [e]

cpu::Frame make_capacitor() {
    // 5x3 wall grid per plate + 8 electrolyte ions = 38 atoms.
    const int n_wall = 5 * 3;
    const int n = 2 * n_wall + 8;
    cpu::Frame f(n);
    int idx = 0;
    auto put = [&](double x, double y, double z, int type, double q, double m) {
        f.span(kAtom_x)[idx] = x;
        f.span(kAtom_y)[idx] = y;
        f.span(kAtom_z)[idx] = z;
        f.span(kAtom_type)[idx] = type;
        f.span(kAtom_charge)[idx] = static_cast<StorageReal>(q);
        f.span(kAtom_mass)[idx] = m;
        f.span(kAtom_tag)[idx] = idx + 1;
        f.span(kAtom_vx)[idx] = f.span(kAtom_vy)[idx] = f.span(kAtom_vz)[idx] =
            0.0;
        ++idx;
    };
    for (int iw = 0; iw < 2; ++iw) { // 0 = bottom (z=5), 1 = top (z=11)
        const double z = iw == 0 ? 5.0 : 11.0;
        const int type = iw == 0 ? kTypeBot : kTypeTop;
        for (int ix = 0; ix < 5; ++ix)
            for (int iy = 0; iy < 3; ++iy)
                put(2.0 + 4.0 * ix, (iy + 0.5) * kLy / 3.0, z, type,
                    /*q=*/0.0, /*m=*/196.966553);
    }
    // Neutral electrolyte: 8 ions, alternating ±0.25 e, between the plates.
    const double ion[8][4] = {{4.0, 2.5, 7.0, +0.25},  {9.0, 7.5, 7.5, -0.25},
                              {13.0, 2.5, 8.0, +0.25}, {17.0, 7.5, 8.5, -0.25},
                              {6.0, 5.0, 9.0, +0.25},  {11.0, 2.0, 9.5, -0.25},
                              {15.0, 5.5, 7.2, +0.25}, {18.5, 3.0, 8.8, -0.25}};
    for (const auto& r : ion)
        put(r[0], r[1], r[2], kTypeIon, r[3], 22.9898);
    f.box = cpu::Box::orthorhombic(kLx, kLy, kLz);
    f.clear_tag_index();
    f.rebuild_tag_index(0, f.nlocal());
    return f;
}

/// Out-of-library checker (the user-extension path): at the force phase —
/// i.e. AFTER the pre_force charge solve and the auto-injected ghost-charge
/// refresh — every ghost row's charge must equal its owner's current value.
/// Owner charges are looked up by tag via an allgather (small system).
class GhostChargeChecker : public cpu::Component {
  public:
    std::string name() const override { return "ghost_charge_checker"; }

    void launch(const cpu::LaunchContext& ctx) override {
        auto* dd = ctx.runtime.md.dd;
        if (!dd || ctx.runtime.mpi_size <= 1)
            return;
        auto& f = ctx.frame;
        const int nloc = dd->n_local;
        const int ngh = dd->n_ghost;
        // Resolve each span ONCE per launch (Frame enforces a per-component
        // resolution budget precisely to keep span lookups off hot loops).
        const auto tag_col = f.span(kAtom_tag);
        const auto q_col = f.span(kAtom_charge);
        const auto type_col = f.span(kAtom_type);

        int size = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        std::vector<int> counts(static_cast<std::size_t>(size), 0);
        MPI_Allgather(&nloc, 1, MPI_INT, counts.data(), 1, MPI_INT,
                      MPI_COMM_WORLD);
        std::vector<int> displs(static_cast<std::size_t>(size), 0);
        int total = 0;
        for (int r = 0; r < size; ++r) {
            displs[static_cast<std::size_t>(r)] = total;
            total += counts[static_cast<std::size_t>(r)];
        }

        std::vector<long long> my_tags(static_cast<std::size_t>(nloc));
        std::vector<double> my_q(static_cast<std::size_t>(nloc));
        for (int i = 0; i < nloc; ++i) {
            my_tags[static_cast<std::size_t>(i)] =
                static_cast<long long>(tag_col[i]);
            my_q[static_cast<std::size_t>(i)] = static_cast<double>(q_col[i]);
        }
        std::vector<long long> all_tags(static_cast<std::size_t>(total));
        std::vector<double> all_q(static_cast<std::size_t>(total));
        MPI_Allgatherv(my_tags.data(), nloc, MPI_LONG_LONG, all_tags.data(),
                       counts.data(), displs.data(), MPI_LONG_LONG,
                       MPI_COMM_WORLD);
        MPI_Allgatherv(my_q.data(), nloc, MPI_DOUBLE, all_q.data(),
                       counts.data(), displs.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);
        std::unordered_map<long long, double> owner_q;
        owner_q.reserve(static_cast<std::size_t>(total));
        for (int i = 0; i < total; ++i)
            owner_q[all_tags[static_cast<std::size_t>(i)]] =
                all_q[static_cast<std::size_t>(i)];

        for (int g = 0; g < ngh; ++g) {
            const int row = nloc + g;
            const long long tag = static_cast<long long>(tag_col[row]);
            const auto it = owner_q.find(tag);
            if (it == owner_q.end()) {
                ++unresolved_;
                continue;
            }
            const double qg = static_cast<double>(q_col[row]);
            const double dq = std::fabs(qg - it->second);
            if (dq > max_dq_)
                max_dq_ = dq;
            const int type = static_cast<int>(type_col[row]);
            if ((type == kTypeBot || type == kTypeTop) && std::fabs(qg) > 1e-12)
                ++elec_ghost_nonzero_;
            ++checked_;
        }
        ++launches_;
    }

    double max_dq_ = 0.0;              ///< max |q_ghost − q_owner| [e]
    long long checked_ = 0;            ///< ghost rows compared
    long long elec_ghost_nonzero_ = 0; ///< electrode ghosts with |q| > 0
    long long unresolved_ = 0;         ///< ghost tags with no owner (must be 0)
    int launches_ = 0;
};

int run_child(const char* out_path) {
    // Deterministic 2-rank split along x (both plates span x fully, so each
    // rank owns part of BOTH electrodes and the halo carries electrode
    // ghosts across the cut).
    ::setenv("ATV_FORCE_DIMS", "2,1,1", 1);
    MPI_Init(nullptr, nullptr);

    auto frame = make_capacitor();

    cpu::md::MD sim;
    sim.set_integrator(cpu::md::VelocityVerlet{kDt});
    sim.set_neighbor(cpu::md::NeighborList<>{kCut, kSkin});
    sim.set_evaluator(cpu::PairCoulCut{kCut});

    cpu::md::ConstPotBO conp(kEta);
    conp.add_electrode(kTypeBot, kPsiBot);
    conp.add_electrode(kTypeTop, kPsiTop);
    conp.electroneutrality = true;
    conp.xy_image_range = 0; // primary-cell min-image solve
    conp.rebuild_matrix_every_step = true;
    sim.set_observer_ref(conp); // default binding: pre_force @ -50

    GhostChargeChecker checker;
    sim.on(ATV::md::Phase::force).add_ref(checker, +500);

    sim.build();
    sim.run(frame, kSteps);

    // Reduce the tallies (slowest/strictest across ranks).
    double max_dq = 0.0;
    long long checked = 0, elec_nonzero = 0, unresolved = 0;
    MPI_Allreduce(&checker.max_dq_, &max_dq, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&checker.checked_, &checked, 1, MPI_LONG_LONG, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&checker.elec_ghost_nonzero_, &elec_nonzero, 1, MPI_LONG_LONG,
                  MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&checker.unresolved_, &unresolved, 1, MPI_LONG_LONG, MPI_SUM,
                  MPI_COMM_WORLD);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int rc = 0;
    if (rank == 0) {
        std::FILE* out = std::fopen(out_path, "w");
        if (!out) {
            std::printf("FAIL: child cannot open %s\n", out_path);
            rc = 1;
        } else {
            std::fprintf(out, "%.17e %lld %lld %lld %d\n", max_dq, checked,
                         elec_nonzero, unresolved, checker.launches_);
            std::fclose(out);
        }
    }
    MPI_Finalize();
    return rc;
}

} // namespace

int main(int argc, char** argv) {
    if (argc >= 3 && std::strcmp(argv[1], "--child") == 0)
        return run_child(argv[2]);

    char self_buf[4096];
    const ssize_t n =
        ::readlink("/proc/self/exe", self_buf, sizeof(self_buf) - 1);
    if (n <= 0) {
        std::printf("FAIL: cannot resolve /proc/self/exe\n");
        return 1;
    }
    self_buf[n] = '\0';

    ::setenv("OMP_NUM_THREADS", "1", 1);
    const std::string out = "/tmp/atv-const-pot-charge-halo-"
                            + std::to_string(static_cast<long>(::getpid()))
                            + ".txt";
    const std::string cmd = std::string(ATV_MPIEXEC) + " -n 2 \""
                            + std::string(self_buf) + "\" --child \"" + out
                            + "\"";
    std::printf("[const-pot-charge-halo-mpi] launching: %s\n", cmd.c_str());
    std::fflush(stdout);
    if (std::system(cmd.c_str()) != 0) {
        std::printf("FAIL: 2-rank child exited nonzero\n");
        return 1;
    }
    std::FILE* in = std::fopen(out.c_str(), "r");
    if (!in) {
        std::printf("FAIL: missing child report %s\n", out.c_str());
        return 1;
    }
    double max_dq = -1.0;
    long long checked = -1, elec_nonzero = -1, unresolved = -1;
    int launches = -1;
    const int got = std::fscanf(in, "%lf %lld %lld %lld %d", &max_dq, &checked,
                                &elec_nonzero, &unresolved, &launches);
    std::fclose(in);
    std::remove(out.c_str());
    if (got != 5) {
        std::printf("FAIL: malformed child report (%d fields)\n", got);
        return 1;
    }

    int failures = 0;
    // (a) The transport invariant — the pin (exactly 0; memcpy transport).
    if (!(max_dq >= 0.0) || max_dq > kMaxDqTol) {
        std::printf("FAIL: max |q_ghost - q_owner| = %.3e e > %.0e e "
                    "(stale ghost charges after the Q solve)\n",
                    max_dq, kMaxDqTol);
        ++failures;
    }
    // (b) Non-vacuous: electrode ghosts with solved (nonzero) charge seen.
    if (elec_nonzero <= 0) {
        std::printf("FAIL: no nonzero-charge electrode ghosts seen "
                    "(elec_nonzero=%lld) — the check is vacuous\n",
                    elec_nonzero);
        ++failures;
    }
    // (c) Ghost rows compared on every step (a driver setup pass may add
    //     one extra force-phase launch); every ghost tag resolved.
    if (checked <= 0 || launches < kSteps || unresolved != 0) {
        std::printf("FAIL: checked=%lld launches=%d (want >= %d) "
                    "unresolved=%lld\n",
                    checked, launches, kSteps, unresolved);
        ++failures;
    }

    std::printf("[const-pot-charge-halo-mpi] max|dq|=%.3e e checked=%lld "
                "elec_ghosts_nonzero=%lld launches=%d\n",
                max_dq, checked, elec_nonzero, launches);
    std::printf(failures == 0 ? "PASS\n" : "FAIL (%d)\n", failures);
    return failures == 0 ? 0 : 1;
}
