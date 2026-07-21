/// @file tests/cpu/observer/test_rdf_observer.cpp
/// @brief Tests for `RDF` async observer (molrs-backed g(r)).
///
/// Verifies:
///   - registration via `sim.add_ref(rdf)` (explicit borrowed component)
///   - snapshots stage into the molrs trajectory (n_samples accrues)
///   - finalize() computes a nonzero g(r) via molrs and writes the output file
///   - first g(r) peak position matches FCC nearest-neighbor distance
///   - frequency() respected

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/md/md.h"
#include "atomiverse/cpu/md/velocity_verlet.h"
#include "atomiverse/cpu/observer/rdf.h"

#include "cpu/driver_test_helpers.h"

using namespace ATV;
using atv_test::NoOpForce;

namespace {

cpu::Frame make_fcc_argon(int n_cells, double a) {
    int n = n_cells * n_cells * n_cells * 4;
    cpu::Frame frame(n);
    double basis[4][3] = {
        {0.0, 0.0, 0.0},
        {0.5, 0.5, 0.0},
        {0.5, 0.0, 0.5},
        {0.0, 0.5, 0.5},
    };
    int idx = 0;
    for (int ix = 0; ix < n_cells; ++ix)
        for (int iy = 0; iy < n_cells; ++iy)
            for (int iz = 0; iz < n_cells; ++iz)
                for (int b = 0; b < 4; ++b) {
                    frame.span(ATV::kAtom_x)[idx] = (ix + basis[b][0]) * a;
                    frame.span(ATV::kAtom_y)[idx] = (iy + basis[b][1]) * a;
                    frame.span(ATV::kAtom_z)[idx] = (iz + basis[b][2]) * a;
                    frame.span(ATV::kAtom_mass)[idx] = 39.948;
                    frame.span(ATV::kAtom_type)[idx] = 18;
                    ++idx;
                }
    frame.simbox = ATV::cpu::Box::cubic(n_cells * a);
    return frame;
}

struct ScopedFile {
    std::string path;
    explicit ScopedFile(std::string p) : path(std::move(p)) {
        std::remove(path.c_str());
    }
    ~ScopedFile() { std::remove(path.c_str()); }
};

} // namespace

// ── snapshots stage into the trajectory; molrs yields a nonzero g(r) ─────

TEST(RDFObserverAsync, AccumulatesAcrossSnapshots) {
    auto frame =
        make_fcc_argon(3, 5.26); // 108 atoms, lattice 5.26 Å (Argon-ish)
    ScopedFile out("test_rdf_out_accum.dat");

    RDF rdf{6.0, 200, /*freq=*/0, out.path};
    {
        cpu::md::MD sim;
        sim.add(cpu::md::VelocityVerlet{1e-6});
        sim.add(NoOpForce{});
        sim.add_ref(rdf);
        sim.build();
        sim.run(
            frame,
            5); // steps 0..4 → publisher fires at 1,2,3,4 (freq=0 = every step >0)
    }

    EXPECT_GE(rdf.n_samples(), 1)
        << "RDF should have received at least one snapshot";
    // finalize() (run at teardown) fills g(r) via molrs compute::RDF.
    double total = 0.0;
    for (double g : rdf.gr())
        total += g;
    EXPECT_GT(total, 0.0) << "expected a nonzero g(r) from molrs";
}

// ── finalize writes the output file with valid g(r) format ──────────────

TEST(RDFObserverAsync, FinalizeWritesOutputFile) {
    auto frame = make_fcc_argon(3, 5.26);
    ScopedFile out("test_rdf_out_finalize.dat");

    RDF rdf{6.0, 100, /*freq=*/0, out.path};
    {
        cpu::md::MD sim;
        sim.add(cpu::md::VelocityVerlet{1e-6});
        sim.add(NoOpForce{});
        sim.add_ref(rdf);
        sim.build();
        sim.run(frame, 5);
    }

    std::ifstream in(out.path);
    ASSERT_TRUE(in.good()) << "output file '" << out.path
                           << "' missing after finalize";

    std::string header;
    std::getline(in, header);
    EXPECT_NE(header.find("g(r)"), std::string::npos)
        << "header missing 'g(r)' marker -- got: " << header;

    int n_lines = 0;
    double r, gr;
    while (in >> r >> gr)
        ++n_lines;
    EXPECT_EQ(n_lines, 100) << "should have one line per histogram bin";
}

// ── first peak in g(r) sits near FCC nearest-neighbor distance ──────────

TEST(RDFObserverAsync, FirstPeakNearFCCNearestNeighbor) {
    const double a_lat = 5.26;
    const double r_nn = a_lat / std::sqrt(2.0); // ≈ 3.72 Å for FCC

    auto frame = make_fcc_argon(4, a_lat); // 256 atoms for better statistics
    ScopedFile out("test_rdf_out_peak.dat");

    const int n_bins = 200;
    const double r_max = 6.0;
    const double bin_width = r_max / n_bins;

    RDF rdf{r_max, n_bins, /*freq=*/0, out.path};
    {
        cpu::md::MD sim;
        sim.add(cpu::md::VelocityVerlet{1e-6});
        sim.add(NoOpForce{});
        sim.add_ref(rdf);
        sim.build();
        sim.run(frame, 3); // 2 firing snapshots
    }

    // Find the g(r) peak away from r=0 (molrs-normalized curve).
    const auto& g = rdf.gr();
    ASSERT_EQ(static_cast<int>(g.size()), n_bins)
        << "molrs should return one g(r) value per bin";
    int peak_bin = -1;
    double peak_val = 0.0;
    // Skip the lowest few bins (no atoms within < 2 Å of each other on FCC).
    const int start_bin = static_cast<int>(2.0 / bin_width);
    for (int i = start_bin; i < n_bins; ++i) {
        if (g[i] > peak_val) {
            peak_val = g[i];
            peak_bin = i;
        }
    }
    ASSERT_GE(peak_bin, 0);

    const double r_peak = (peak_bin + 0.5) * bin_width;
    EXPECT_NEAR(r_peak, r_nn, 0.10)
        << "expected first g(r) peak near FCC NN distance " << r_nn
        << " Å, got " << r_peak << " Å";
}

// ── frequency() stride is respected ─────────────────────────────────────

TEST(RDFObserverAsync, FrequencyRespected) {
    auto frame = make_fcc_argon(3, 5.26);
    ScopedFile out("test_rdf_out_freq.dat");

    RDF rdf{6.0, 100, /*freq=*/3, out.path};
    {
        cpu::md::MD sim;
        sim.add(cpu::md::VelocityVerlet{1e-6});
        sim.add(NoOpForce{});
        sim.add_ref(rdf);
        sim.build();
        sim.run(frame, 11); // step > 0 && step % 3 == 0 → 3, 6, 9 → 3 samples
    }

    EXPECT_EQ(rdf.n_samples(), 3)
        << "frequency=3 over 11 steps should fire at steps 3/6/9";
}
