/// @file test_rdf_molpy.cu
/// @brief cuda::RDF on the molpy streaming accumulator (arch-refactor-07-molpy):
/// static-trajectory g(r) parity vs a host-fed cpu::compute::RDFAccumulator,
/// nearest-neighbor lattice-peak physics, fail-fast constructor edges, and
/// lossless capture: behind a one-frame budget and a slow consumer the curve
/// and the sample count equal a run whose buffer never fills.
///
/// A dt = 0 fs VelocityVerlet makes every MD frame identical, so feeding ONE
/// copy of the frame into the host accumulator yields the exact reference
/// curve. Units: positions / r / cell in Å, g(r) dimensionless.

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/observe/trajectory_metrics.h"
#include "atomiverse/cuda/dynamics/md.cuh"
#include "atomiverse/cuda/dynamics/velocity_verlet.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "atomiverse/cuda/observe/rdf.cuh"
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
                frame.span(kAtom_x)[idx] = (double(ix) + double(0.5)) * spacing;
                frame.span(kAtom_y)[idx] = (double(iy) + double(0.5)) * spacing;
                frame.span(kAtom_z)[idx] = (double(iz) + double(0.5)) * spacing;
                frame.span(kAtom_vx)[idx] = double(0);
                frame.span(kAtom_vy)[idx] = double(0);
                frame.span(kAtom_vz)[idx] = double(0);
                frame.span(kAtom_mass)[idx] = double(1.0);
                frame.span(kAtom_type)[idx] = 0;
                idx++;
            }
    frame.box = cpu::Box::cubic(static_cast<double>(box_len));
    return frame;
}

/// Run a handful of MD steps with a dt = 0 fs VelocityVerlet (satisfies the
/// cuda::md::MD Force-capability check while keeping every frame identical)
/// and the cuda::RDF observer; RDF::destroy at teardown writes the g(r) file.
void run_static_rdf_md(cpu::Frame& frame, const std::string& filename) {
    cuda::md::MD sim;
    sim.set_integrator(
        cuda::md::VelocityVerlet{double(0.0)}); // dt = 0 fs → static
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

/// Host reference: feed the SAME single (pristine) frame once into the molpy
/// streaming accumulator — positions blocked x|y|z widened to double, cell as
/// a row-major 9-element cubic matrix [Å]. One accepted frame normalizes to
/// the same g(r) as N identical frames. A rejected frame throws.
std::vector<double> host_reference_gr(const cpu::Frame& frame, int n,
                                      double box_len) {
    std::vector<double> positions(static_cast<std::size_t>(3 * n));
    for (int i = 0; i < n; ++i) {
        positions[static_cast<std::size_t>(i)] =
            static_cast<double>(frame.span(kAtom_x)[i]);
        positions[static_cast<std::size_t>(n + i)] =
            static_cast<double>(frame.span(kAtom_y)[i]);
        positions[static_cast<std::size_t>(2 * n + i)] =
            static_cast<double>(frame.span(kAtom_z)[i]);
    }
    const std::array<double, 9> cell = {box_len, 0.0, 0.0, 0.0,    box_len,
                                        0.0,     0.0, 0.0, box_len};
    auto acc = cpu::compute::RDF{kNBins, kRMax, /*r_min=*/0.0}.accumulator();
    acc.push(positions, cell);
    return acc.finish().rdf;
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

// ── Integration + parity: file g(r) ≡ host molpy accumulator per bin ──────

/// The g(r) file written by the migrated cuda::RDF must match, bin for bin,
/// the curve from feeding the identical static frame into
/// cpu::compute::RDFAccumulator on the host (single source of truth = molpy).
/// Tolerance 1e-6 — the file is %.6f-limited (exact-match class otherwise).
TEST(RDFMolpy, FileMatchesHostAccumulatorPerBin) {
    const int n = kNSide * kNSide * kNSide;
    const double box_len = std::cbrt(double(n) / kDensity);

    auto frame = make_static_lj_lattice(kNSide, kDensity);
    cpu::Frame md_frame = frame; // keep the pristine copy for the reference

    ScopedFile out("test_rdf_molpy_parity.dat");
    run_static_rdf_md(md_frame, out.path);

    std::vector<double> r_file, g_file;
    ASSERT_TRUE(parse_gr_file(out.path, r_file, g_file))
        << "g(r) file missing or malformed header: " << out.path;
    ASSERT_EQ(static_cast<int>(g_file.size()), kNBins)
        << "expected one data line per bin";

    auto ref = host_reference_gr(frame, n, box_len);
    ASSERT_EQ(static_cast<int>(ref.size()), kNBins)
        << "host RDFAccumulator returned the wrong bin count";

    const double bin_width = kRMax / double(kNBins);
    for (int i = 0; i < kNBins; ++i) {
        EXPECT_NEAR(r_file[static_cast<std::size_t>(i)],
                    (double(i) + 0.5) * bin_width, 1e-6)
            << "bin-center radius mismatch at bin " << i;
        EXPECT_NEAR(g_file[static_cast<std::size_t>(i)],
                    ref[static_cast<std::size_t>(i)], 1e-6)
            << "g(r) mismatch vs molpy accumulator at bin " << i;
    }
}

// ── Domain validation: nearest-neighbor lattice peak is physically there ──

/// On a static simple-cubic lattice the bin containing the nearest-neighbor
/// distance (spacing = cbrt(1/ρ) ≈ 1.0772 Å) must carry a non-zero g(r).
TEST(RDFMolpy, NearestNeighborLatticeBinNonZero) {
    auto frame = make_static_lj_lattice(kNSide, kDensity);

    ScopedFile out("test_rdf_molpy_nn.dat");
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

/// The constructor builds the molpy accumulator, so molpy `Rdf::new` refuses
/// n_bins < 1 at construction time (std::invalid_argument).
TEST(RDFMolpy, CtorThrowsOnInvalidBinCount) {
    EXPECT_THROW(cuda::RDF(kRMax, 0, "unused_rdf.dat"), std::invalid_argument);
}

/// r_max ≤ 0 Å must throw std::invalid_argument at construction time
/// (fail-fast, no-fallback — mirrors the CPU observer).
TEST(RDFMolpy, CtorThrowsOnNonPositiveRMax) {
    EXPECT_THROW(cuda::RDF(0.0, kNBins, "unused_rdf.dat"),
                 std::invalid_argument);
}

// ── Lossless capture: no sample skipped under backpressure ────────────────

namespace {

/// A deliberately slow consumer sharing the driver's capture buffer.
class SlowConsumer : public cuda::AsyncObserver {
  public:
    std::string name() const override { return "slow"; }
    int frequency() const override { return 1; }
    CaptureColumns capture_columns() const override {
        CaptureColumns c;
        c.positions = true;
        return c;
    }
    int seen = 0;

  protected:
    void consume(const cuda::CaptureView&) override {
        std::this_thread::sleep_for(std::chrono::milliseconds(3));
        ++seen;
    }
};

/// Ten moving LJ atoms (the curve changes every step).
cpu::Frame make_moving_frame() {
    cpu::Frame frame(10);
    for (int i = 0; i < 10; ++i) {
        frame.span(kAtom_x)[i] = 8.0 + 3.5 * static_cast<double>(i % 5);
        frame.span(kAtom_y)[i] = 10.0 + 3.5 * static_cast<double>(i / 5);
        frame.span(kAtom_z)[i] = 15.0;
        frame.span(kAtom_vx)[i] = 0.002 * static_cast<double>(i % 3 - 1);
        frame.span(kAtom_vy)[i] = 0.001 * static_cast<double>(i % 2);
        frame.span(kAtom_vz)[i] = 0.0005 * static_cast<double>(i % 4 - 2);
        frame.span(kAtom_mass)[i] = 39.948;
        frame.span(kAtom_type)[i] = 0;
    }
    frame.box = cpu::Box::cubic(30.0);
    return frame;
}

struct RdfRun {
    std::vector<double> r, g;
    long long frames = 0;
    std::uint64_t blocked = 0;
};

RdfRun rdf_run(bool tight, const std::string& path) {
    constexpr int kSteps = 24;
    cuda::RDF rdf{5.5, 40, path, /*every=*/1};
    SlowConsumer slow;
    RdfRun out;
    {
        cpu::Frame f = make_moving_frame();
        cuda::md::MD sim;
        sim.set_integrator(cuda::md::VelocityVerlet{1.0});
        sim.on(md::Phase::force)
            .add(cuda::NeighborList{6.0, 0.3}, -10)
            .add(cuda::PairLJCut{0.0104, 3.4, 6.0});
        sim.on(md::Phase::end_of_step).add_ref(rdf, +20);
        if (tight) {
            sim.on(md::Phase::end_of_step).add_ref(slow, +20);
            sim.set_capture_buffer_bytes(cuda::CaptureBuffer::frame_bytes(
                10, rdf.capture_columns() | slow.capture_columns()));
        }
        sim.build();
        sim.run(f, kSteps);
        out.frames = rdf.n_frames();
        out.blocked = sim.capture_stats().blocked_episodes;
    } // teardown writes the curve
    parse_gr_file(path, out.r, out.g);
    return out;
}

} // namespace

TEST(RDFMolpy, CurveIdenticalUnderBackpressure) {
    ScopedFile a("test_rdf_lossless_free.dat"),
        b("test_rdf_lossless_tight.dat");
    const RdfRun free_run = rdf_run(false, a.path);
    const RdfRun tight = rdf_run(true, b.path);
    // Every step 0..24 is a sample, in both runs.
    EXPECT_EQ(free_run.frames, 25);
    EXPECT_EQ(tight.frames, 25);
    EXPECT_EQ(free_run.blocked, 0u);
    EXPECT_GT(tight.blocked, 0u);
    ASSERT_EQ(free_run.g.size(), 40u);
    EXPECT_EQ(tight.r, free_run.r);
    EXPECT_EQ(tight.g, free_run.g);
}
