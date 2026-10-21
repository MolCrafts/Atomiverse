/// @file regressions/md/cuda-multi-gpu-trajectory/reg_cuda-multi-gpu-trajectory.cu
/// @brief Public-consumer NVE domain-decomposition and migration regressions.
/// Velocity Verlet: Swope et al., JCP 76, 637 (1982), doi:10.1063/1.442716.
/// Domain decomposition: Plimpton, JCP 117, 1 (1995), doi:10.1006/jcph.1995.1039.
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cuda/communication/multi_gpu.cuh"
#include "atomiverse/cuda/device/error.cuh"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace ATV;
namespace {
cpu::Frame make_grid(double L) {
    const int s = 4, n = s * s * s;
    cpu::Frame f(n);
    f.box.set_cell(Mat3::cubic(L));
    double sp = L / s;
    int idx = 0;
    for (int ix = 0; ix < s; ++ix)
        for (int iy = 0; iy < s; ++iy)
            for (int iz = 0; iz < s; ++iz) {
                f.span(kAtom_x)[idx] = (ix + 0.5) * sp;
                f.span(kAtom_y)[idx] = (iy + 0.5) * sp;
                f.span(kAtom_z)[idx] = (iz + 0.5) * sp;
                f.span(kAtom_vx)[idx] = 0.01 * idx;
                f.span(kAtom_mass)[idx] = 1.0;
                f.span(kAtom_type)[idx] = 0;
                f.span(kAtom_tag)[idx] = idx;
                ++idx;
            }
    return f;
}

cpu::Frame make_bonded_grid(double L) {
    cpu::Frame f = make_grid(L);
    const int s = 4;
    // add_bond/add_angle resolve tags through the tag→local map; build it first.
    f.rebuild_tag_index(0, f.nlocal());
    auto tag_of = [&](int ix, int iy, int iz) {
        return (ix * s + iy) * s + iz;
    };
    for (int iy = 0; iy < s; ++iy)
        for (int iz = 0; iz < s; ++iz) {
            for (int ix = 0; ix < s - 1; ++ix)
                f.add_bond(tag_of(ix, iy, iz), tag_of(ix + 1, iy, iz), 0);
            for (int ix = 1; ix < s - 1; ++ix)
                f.add_angle(tag_of(ix - 1, iy, iz), tag_of(ix, iy, iz),
                            tag_of(ix + 1, iy, iz), 0);
            for (int ix = 1; ix < s - 2; ++ix)
                f.add_dihedral(tag_of(ix - 1, iy, iz), tag_of(ix, iy, iz),
                               tag_of(ix + 1, iy, iz), tag_of(ix + 2, iy, iz),
                               0);
        }
    f.build_special_list();
    return f;
}

cpu::Frame make_grid_n(int s, double L) {
    const int n = s * s * s;
    cpu::Frame f(n);
    f.box.set_cell(Mat3::cubic(L));
    double sp = L / s;
    int idx = 0;
    for (int ix = 0; ix < s; ++ix)
        for (int iy = 0; iy < s; ++iy)
            for (int iz = 0; iz < s; ++iz) {
                f.span(kAtom_x)[idx] = (ix + 0.5) * sp;
                f.span(kAtom_y)[idx] = (iy + 0.5) * sp;
                f.span(kAtom_z)[idx] = (iz + 0.5) * sp;
                f.span(kAtom_mass)[idx] = 1.0;
                f.span(kAtom_type)[idx] = 0;
                f.span(kAtom_tag)[idx] = idx;
                ++idx;
            }
    return f;
}

void check_near(double actual, double expected, double tolerance,
                const char* name, int atom, const char* axis) {
    if (!(std::abs(actual - expected) <= tolerance)) {
        std::fprintf(
            stderr,
            "%s atom=%d axis=%s actual=%.17g reference=%.17g tolerance=%.17g\n",
            name, atom, axis, actual, expected, tolerance);
        throw std::runtime_error(
            "multi-GPU trajectory differs from single GPU");
    }
}

void NveTrajectoryParity() {
    // Run a short NVE trajectory single-GPU vs split-across-GPUs and require the
    // owned positions to agree — validating the full step loop (halo + force +
    // integrate) end to end. Small dt / few steps so atoms stay in their slab
    // (no migration) and float chaos stays bounded.
    auto g = make_grid(12.0);
    const int n = g.nlocal();
    for (int i = 0; i < n; ++i) {
        g.span(kAtom_x)[i] += 0.4 * std::sin(0.7 * i + 0.1);
        g.span(kAtom_y)[i] += 0.4 * std::sin(1.3 * i + 0.2);
        g.span(kAtom_z)[i] += 0.4 * std::sin(1.9 * i + 0.3);
        g.span(kAtom_vx)[i] = 0.02 * std::sin(0.5 * i);
        g.span(kAtom_vy)[i] = 0.02 * std::sin(0.9 * i);
        g.span(kAtom_vz)[i] = 0.02 * std::sin(1.1 * i);
        g.span(kAtom_mass)[i] = 1.0;
    }
    const double cut = 3.5, eps = 1.0, sig = 2.5, dt = 0.002;
    const int K = 10;

    cpu::Frame ref(n);
    for (int i = 0; i < n; ++i)
        ref.span(kAtom_tag)[i] = i;
    {
        ATV::cuda::md::MultiGpu mg1(1);
        mg1.partition(g);
        mg1.run_nve(K, dt, cut, eps, sig);
        mg1.gather(ref);
    }

    ATV::cuda::md::MultiGpu mg(2);
    mg.partition(g);
    mg.run_nve(K, dt, cut, eps, sig);
    cpu::Frame mf(n);
    for (int i = 0; i < n; ++i)
        mf.span(kAtom_tag)[i] = i;
    mg.gather(mf);

    for (int i = 0; i < n; ++i) {
        check_near(mf.span(kAtom_x)[i], ref.span(kAtom_x)[i], 5e-3,
                   "NveTrajectoryParity", i, "x");
        check_near(mf.span(kAtom_y)[i], ref.span(kAtom_y)[i], 5e-3,
                   "NveTrajectoryParity", i, "y");
        check_near(mf.span(kAtom_z)[i], ref.span(kAtom_z)[i], 5e-3,
                   "NveTrajectoryParity", i, "z");
    }
}

void NveTrajectoryParityNlist() {
    // End-to-end Verlet-cadence loop: single-GPU vs multi-GPU run_nve_nlist must
    // agree on owned positions after a short trajectory. Exercises reneighbor
    // (halo rebuild + neighbor-list rebuild) and forward_comm position refresh.
    auto g = make_grid_n(8, 24.0);
    const int n = g.nlocal();
    for (int i = 0; i < n; ++i) {
        g.span(kAtom_x)[i] += 0.4 * std::sin(0.7 * i + 0.1);
        g.span(kAtom_y)[i] += 0.4 * std::sin(1.3 * i + 0.2);
        g.span(kAtom_z)[i] += 0.4 * std::sin(1.9 * i + 0.3);
        g.span(kAtom_vx)[i] = 0.01 * std::sin(0.5 * i);
        g.span(kAtom_vy)[i] = 0.01 * std::sin(0.9 * i);
        g.span(kAtom_vz)[i] = 0.01 * std::sin(1.1 * i);
        g.span(kAtom_mass)[i] = 1.0;
    }
    const double cut = 5.0, eps = 1.0, sig = 2.5, dt = 0.002;
    const int K = 12, reneigh = 4;

    cpu::Frame ref(n);
    for (int i = 0; i < n; ++i)
        ref.span(kAtom_tag)[i] = i;
    {
        ATV::cuda::md::MultiGpu mg1(1);
        mg1.partition(g, 2.0);
        mg1.run_nve_nlist(K, dt, cut, eps, sig, -1.0, reneigh);
        mg1.gather(ref);
    }

    ATV::cuda::md::MultiGpu mg(2);
    mg.partition(g, 2.0);
    mg.run_nve_nlist(K, dt, cut, eps, sig, -1.0, reneigh);
    cpu::Frame mf(n);
    for (int i = 0; i < n; ++i)
        mf.span(kAtom_tag)[i] = i;
    mg.gather(mf);

    for (int i = 0; i < n; ++i) {
        check_near(mf.span(kAtom_x)[i], ref.span(kAtom_x)[i], 5e-3,
                   "NveTrajectoryParityNlist", i, "x");
        check_near(mf.span(kAtom_y)[i], ref.span(kAtom_y)[i], 5e-3,
                   "NveTrajectoryParityNlist", i, "y");
        check_near(mf.span(kAtom_z)[i], ref.span(kAtom_z)[i], 5e-3,
                   "NveTrajectoryParityNlist", i, "z");
    }
}

void NveMigrationParity() {
    // Long-run correctness: a uniform +x drift (Galilean boost — leaves all
    // relative positions and forces unchanged) carries atoms across the x=12
    // boundary during the run, so migrate() fires repeatedly inside reneighbor.
    // Multi-GPU (partition + migrate) must still match single-GPU positions.
    auto g = make_grid_n(8, 24.0);
    const int n = g.nlocal();
    const double drift = 40.0;
    for (int i = 0; i < n; ++i) {
        g.span(kAtom_x)[i] += 0.3 * std::sin(0.7 * i + 0.1);
        g.span(kAtom_y)[i] += 0.3 * std::sin(1.3 * i + 0.2);
        g.span(kAtom_z)[i] += 0.3 * std::sin(1.9 * i + 0.3);
        g.span(kAtom_vx)[i] = drift + 0.01 * std::sin(0.5 * i);
        g.span(kAtom_vy)[i] = 0.01 * std::sin(0.9 * i);
        g.span(kAtom_vz)[i] = 0.01 * std::sin(1.1 * i);
        g.span(kAtom_mass)[i] = 1.0;
    }
    const double cut = 5.0, eps = 1.0, sig = 2.5, dt = 0.002;
    const int K = 20, reneigh = 4; // 20*0.002*40 = 1.6 -> atoms cross x=12

    cpu::Frame ref(n);
    for (int i = 0; i < n; ++i)
        ref.span(kAtom_tag)[i] = i;
    {
        ATV::cuda::md::MultiGpu mg1(1);
        mg1.partition(g, 2.0);
        mg1.run_nve_nlist(K, dt, cut, eps, sig, -1.0, reneigh);
        mg1.gather(ref);
    }

    ATV::cuda::md::MultiGpu mg(2);
    mg.partition(g, 2.0);
    mg.run_nve_nlist(K, dt, cut, eps, sig, -1.0, reneigh);
    cpu::Frame mf(n);
    for (int i = 0; i < n; ++i)
        mf.span(kAtom_tag)[i] = i;
    mg.gather(mf);

    for (int i = 0; i < n; ++i) {
        check_near(mf.span(kAtom_x)[i], ref.span(kAtom_x)[i], 2e-2,
                   "NveMigrationParity", i, "x");
        check_near(mf.span(kAtom_y)[i], ref.span(kAtom_y)[i], 2e-2,
                   "NveMigrationParity", i, "y");
        check_near(mf.span(kAtom_z)[i], ref.span(kAtom_z)[i], 2e-2,
                   "NveMigrationParity", i, "z");
    }
}

void MigratePreservesBondedTopology() {
    ATV::cuda::md::MultiGpu mg;
    auto g = make_bonded_grid(12.0);
    for (int i = 0; i < g.nlocal(); ++i) {
        g.span(kAtom_vx)[i] = 2.0; // rigid translation along x
        g.span(kAtom_vy)[i] = 0.0;
        g.span(kAtom_vz)[i] = 0.0;
    }
    std::set<std::pair<long long, long long>> ref;
    for (int i = 0; i < g.nlocal(); ++i)
        for (int m = 0; m < g.span(kAtom_num_bond)[i]; ++m) {
            long long a = g.span(kAtom_tag)[i],
                      b = g.span(kAtom_bond_atom)(i, m);
            ref.insert({std::min(a, b), std::max(a, b)});
        }
    if (ref.empty())
        throw std::runtime_error("empty reference bond topology");

    mg.partition(g);
    mg.setup_force_engine(3.5, 1e-12,
                          1.0); // halo 3.5 covers the 3.0 bond; eps≈0
    // ~4 A of rigid drift over 20 steps → atoms cross subdomain boundaries;
    // migrate (carrying topology) runs every step (reneigh_every=1).
    mg.run_nve_nlist(20, 0.1, 3.5, 1e-12, 1.0, 0.5, 1);

    // Validate TRANSPORT fidelity: reconstruct the bond set straight from each
    // owned atom's migrated per-atom SoA (num_bond + bond_atom GLOBAL TAGS) — no
    // tag→local resolve, so this is independent of the halo/PBC (the resolve
    // path is covered by the static Partition test). If migrate lost or
    // corrupted any atom's topology, a bond would be missing or wrong here.
    std::set<std::pair<long long, long long>> got;
    for (int s = 0; s < mg.count(); ++s) {
        const int nl = mg.n_local(s);
        if (nl == 0)
            continue;
        const auto& F = mg.frame(s);
        const int MB = F.atoms.bond_per_atom; // runtime stride (per frame)
        std::vector<int> nbv(nl);
        std::vector<long long> ba(static_cast<std::size_t>(nl) * MB), tags(nl);
        ATV_CUDA_CHECK(cudaMemcpy(nbv.data(), F.atoms.num_bond,
                                  nl * sizeof(int), cudaMemcpyDefault));
        ATV_CUDA_CHECK(
            cudaMemcpy(ba.data(), F.atoms.bond_atom,
                       static_cast<std::size_t>(nl) * MB * sizeof(long long),
                       cudaMemcpyDefault));
        ATV_CUDA_CHECK(cudaMemcpy(tags.data(), F.atoms.tag,
                                  nl * sizeof(long long), cudaMemcpyDefault));
        for (int k = 0; k < nl; ++k)
            for (int m = 0; m < nbv[k]; ++m) {
                long long a = tags[k],
                          b = ba[static_cast<std::size_t>(k) * MB + m];
                got.insert({std::min(a, b), std::max(a, b)});
            }
    }
    std::cerr << "[diag] migrate: ref=" << ref.size() << " got=" << got.size()
              << "\n";
    if (got != ref)
        throw std::runtime_error("bond topology changed across migration");
}

} // namespace

int main(int argc, char** argv) try {
    int count = 0;
    ATV_CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count < 2)
        throw std::runtime_error(
            "cuda-multi-gpu-trajectory requires at least 2 visible GPUs");
    const std::string selected = argc > 1 ? argv[1] : "all";
    if (argc > 2)
        throw std::invalid_argument("expected at most one case name");
    bool found = false;
    if (selected == "all" || selected == "NveTrajectoryParity") {
        found = true;
        NveTrajectoryParity();
        std::printf("NveTrajectoryParity: PASS\n");
    }
    if (selected == "all" || selected == "NveTrajectoryParityNlist") {
        found = true;
        NveTrajectoryParityNlist();
        std::printf("NveTrajectoryParityNlist: PASS\n");
    }
    if (selected == "all" || selected == "NveMigrationParity") {
        found = true;
        NveMigrationParity();
        std::printf("NveMigrationParity: PASS\n");
    }
    if (selected == "all" || selected == "MigratePreservesBondedTopology") {
        found = true;
        MigratePreservesBondedTopology();
        std::printf("MigratePreservesBondedTopology: PASS\n");
    }
    if (!found)
        throw std::invalid_argument(
            "unknown case; choose all, NveTrajectoryParity, "
            "NveTrajectoryParityNlist, NveMigrationParity, or "
            "MigratePreservesBondedTopology");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "cuda-multi-gpu-trajectory: %s\n", e.what());
    return 1;
}
