/// @file test_rdf_molrs.cu
/// @brief cuda::RDF on the molrs streaming accumulator (arch-refactor-07-molrs):
/// static-trajectory g(r) parity vs a host-fed cpu::compute::RDFAccumulator,
/// nearest-neighbor lattice-peak physics, and fail-fast constructor edges.
///
/// A dt = 0 fs VelocityVerlet makes every MD frame identical, so the
/// normalized g(r) is invariant under the async ring's drop_oldest frame
/// drops — feeding ONE copy of the same frame into the host accumulator
/// yields the exact reference curve regardless of how many frames the CUDA
/// observer accepted. Units: positions / r / cell in Å, g(r) dimensionless.

#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/observer/trajectory_metrics.h"
#include "atomiverse/cuda/md/md.cuh"
#include "atomiverse/cuda/md/velocity_verlet.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "atomiverse/cuda/observer/rdf.cuh"
#include "atomiverse/cuda/potentials/pair_lj_cut.cuh"

using namespace ATV;

namespace {

// Shared geometry: simple-cubic LJ lattice, number density 0.8 atoms/Å³
// → spacing = cbrt(1/0.8) ≈ 1.0772 Å (= nearest-neighbor distance).
// n_side = 5 → 125 atoms, L = cbrt(125/0.8) ≈ 5.386 Å, L/2 ≈ 2.693 Å.
constexpr int kNSide = 5;
constexpr double kDensity = 0.8; // atoms/Å³
constexpr double kRMax = 1.7;    // Å — < L/2, ≥ 3 link cells per dimension
constexpr int kNBins = 50;       // bin width 0.034 Å
constexpr double kCutoff = 2.5;  // Å — LJ force path only; the migrated RDF
                                 // no longer reads the pair list

/// Create a simple-cubic lattice of LJ particles with ZERO velocities
/// (static trajectory under a dt = 0 integrator). Positions in Å.
cpu::Frame make_static_lj_lattice(int n_side, double density) {
    int n = n_side * n_side * n_side;
    double box_len = std::cbrt(double(n) / density);
    double spacing = box_len / double(n_side);

    cpu::Frame frame(n);
    int idx = 0;
    for (int ix = 0; ix < n_side; ++ix)
        for (int iy = 0; iy < n_side; ++iy)
            for (int iz = 0; iz < n_side; ++iz) {
                frame.atoms.x[idx] = (double(ix) + double(0.5)) * spacing;
                frame.atoms.y[idx] = (double(iy) + double(0.5)) * spacing;
                frame.atoms.z[idx] = (double(iz) + double(0.5)) * spacing;
                frame.atoms.vx[idx] = double(0);
                frame.atoms.vy[idx] = double(0);
                frame.atoms.vz[idx] = double(0);
                frame.atoms.mass[idx] = double(1.0);
                frame.atoms.type[idx] = 0;
                idx++;
            }
    frame.box = Mat3::cubic(static_cast<double>(box_len));
    return frame;
}

/// Run a handful of MD steps with a dt = 0 fs VelocityVerlet (satisfies the
/// cuda::md::MD Force-capability check while keeping every frame identical)
/// and the cuda::RDF observer; RDF::destroy at teardown writes the g(r) file.
void run_static_rdf_md(cpu::Frame& frame, const std::string& filename) {
    cuda::md::MD sim;
    sim.add(cuda::md::VelocityVerlet{double(0.0)}); // dt = 0 fs → static
    sim.on(md::Phase::force)
        .add(cuda::NeighborList{kCutoff}, -10)
        .add(cuda::PairLJCut{double(1.0), double(1.0), kCutoff});
    sim.on(md::Phase::end_of_step).add(cuda::RDF{kRMax, kNBins, filename});
    sim.build();
    sim.run(frame, 5);
}

/// Parse the g(r) file: header "# r [Angstrom]   g(r)", then n_bins lines
/// "%.6f  %.6f". Returns false when the file is missing or the header is
/// malformed.
bool parse_gr_file(const std::string& filename, std::vector<double>& r_out,
                   std::vector<double>& g_out) {
    std::ifstream file(filename);
    if (!file.is_open())
        return false;
    std::string header;
    std::getline(file, header);
    if (header.find("g(r)") == std::string::npos)
        return false;
    double r = 0.0, g = 0.0;
    while (file >> r >> g) {
        r_out.push_back(r);
        g_out.push_back(g);
    }
    return true;
}

/// Host reference: feed the SAME single (pristine) frame once into the molrs
/// streaming accumulator — positions blocked x|y|z widened to double, cell as
/// a row-major 9-element cubic matrix [Å]. One accepted frame normalizes to
/// the same g(r) as N identical frames. Empty on a rejected frame.
std::vector<double> host_reference_gr(const cpu::Frame& frame, int n,
                                      double box_len) {
    std::vector<double> positions(static_cast<std::size_t>(3 * n));
    for (int i = 0; i < n; ++i) {
        positions[static_cast<std::size_t>(i)] =
            static_cast<double>(frame.atoms.x[i]);
        positions[static_cast<std::size_t>(n + i)] =
            static_cast<double>(frame.atoms.y[i]);
        positions[static_cast<std::size_t>(2 * n + i)] =
            static_cast<double>(frame.atoms.z[i]);
    }
    const std::array<double, 9> cell = {box_len, 0.0, 0.0, 0.0,    box_len,
                                        0.0,     0.0, 0.0, box_len};
    cpu::compute::RDFAccumulator acc{kNBins, kRMax, /*r_min=*/0.0};
    if (!acc.accumulate(positions, cell))
        return {};
    return acc.finalize();
}

/// RAII cleanup for the g(r) output file (mirrors the CPU observer test).
struct ScopedFile {
    std::string path;
    explicit ScopedFile(std::string p) : path(std::move(p)) {
        std::remove(path.c_str());
    }
    ~ScopedFile() { std::remove(path.c_str()); }
};

} // anonymous namespace

// ── Integration + parity: file g(r) ≡ host molrs accumulator per bin ──────

/// The g(r) file written by the migrated cuda::RDF must match, bin for bin,
/// the curve from feeding the identical static frame into
/// cpu::compute::RDFAccumulator on the host (single source of truth = molrs).
/// Tolerance 1e-6 — the file is %.6f-limited (exact-match class otherwise).
TEST(RDFMolrs, FileMatchesHostAccumulatorPerBin) {
    const int n = kNSide * kNSide * kNSide;
    const double box_len = std::cbrt(double(n) / kDensity);

    auto frame = make_static_lj_lattice(kNSide, kDensity);
    cpu::Frame md_frame = frame; // keep the pristine copy for the reference

    ScopedFile out("test_rdf_molrs_parity.dat");
    run_static_rdf_md(md_frame, out.path);

    std::vector<double> r_file, g_file;
    ASSERT_TRUE(parse_gr_file(out.path, r_file, g_file))
        << "g(r) file missing or malformed header: " << out.path;
    ASSERT_EQ(static_cast<int>(g_file.size()), kNBins)
        << "expected one data line per bin";

    auto ref = host_reference_gr(frame, n, box_len);
    ASSERT_EQ(static_cast<int>(ref.size()), kNBins)
        << "host RDFAccumulator rejected the reference frame";

    const double bin_width = kRMax / double(kNBins);
    for (int i = 0; i < kNBins; ++i) {
        EXPECT_NEAR(r_file[static_cast<std::size_t>(i)],
                    (double(i) + 0.5) * bin_width, 1e-6)
            << "bin-center radius mismatch at bin " << i;
        EXPECT_NEAR(g_file[static_cast<std::size_t>(i)],
                    ref[static_cast<std::size_t>(i)], 1e-6)
            << "g(r) mismatch vs molrs accumulator at bin " << i;
    }
}

// ── Domain validation: nearest-neighbor lattice peak is physically there ──

/// On a static simple-cubic lattice the bin containing the nearest-neighbor
/// distance (spacing = cbrt(1/ρ) ≈ 1.0772 Å) must carry a non-zero g(r).
TEST(RDFMolrs, NearestNeighborLatticeBinNonZero) {
    auto frame = make_static_lj_lattice(kNSide, kDensity);

    ScopedFile out("test_rdf_molrs_nn.dat");
    run_static_rdf_md(frame, out.path);

    std::vector<double> r_file, g_file;
    ASSERT_TRUE(parse_gr_file(out.path, r_file, g_file))
        << "g(r) file missing or malformed header: " << out.path;
    ASSERT_EQ(static_cast<int>(g_file.size()), kNBins);

    const double spacing = std::cbrt(1.0 / kDensity); // ≈ 1.0772 Å
    const double bin_width = kRMax / double(kNBins);
    const int nn_bin = static_cast<int>(spacing / bin_width);
    ASSERT_LT(nn_bin, kNBins);
    EXPECT_GT(g_file[static_cast<std::size_t>(nn_bin)], 0.0)
        << "no g(r) weight in the nearest-neighbor bin " << nn_bin << " (r ≈ "
        << spacing << " Å)";
}

// ── Edge cases: fail-fast construction (no-fallback) ──────────────────────

/// Post-migration the constructor directly constructs the molrs accumulator:
/// n_bins < 1 must throw std::invalid_argument at construction time.
TEST(RDFMolrs, CtorThrowsOnInvalidBinCount) {
    EXPECT_THROW(cuda::RDF(kRMax, 0, "unused_rdf.dat"), std::invalid_argument);
}

/// r_max ≤ 0 Å must throw std::invalid_argument at construction time
/// (fail-fast, no-fallback — mirrors the CPU observer).
TEST(RDFMolrs, CtorThrowsOnNonPositiveRMax) {
    EXPECT_THROW(cuda::RDF(0.0, kNBins, "unused_rdf.dat"),
                 std::invalid_argument);
}
