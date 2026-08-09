/// @file tests/cuda/test_materialize.cu
/// @brief Round-trip + alignment tests for cuda::from_molrs / to_molrs.
///
/// Covers the rust-frame-lowering-04 acceptance criteria:
///   - ac-002: a cuda::Frame produced by cuda::from_molrs carries the
///             aligned field set -- per-atom charge / tag / image plus a
///             device-resident SystemScalars -- matching cpu::Frame.
///   - ac-003: molrs.Frame -> cuda::Frame -> molrs.Frame round-trips
///             positions / velocities / charge element-wise (explicit
///             H2D on from_molrs, explicit D2H on to_molrs).

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "atomiverse/cuda/frame/molrs_exchange.cuh"
#include "atomiverse/cuda/frame/transfer.cuh"
#include "atomiverse/molrs.hpp"

using namespace ATV;

namespace {

// Three atoms, one bond, a cubic box -- all per-atom result columns set.
::rust::Box<molrs::FrameRef> make_full_frame() {
    auto fr = molrs::frame_new();

    const std::vector<double> x = {0.0, 1.5, 3.0};
    const std::vector<double> y = {0.1, 1.6, 3.1};
    const std::vector<double> z = {0.2, 1.7, 3.2};
    const std::vector<double> vx = {0.01, 0.02, 0.03};
    const std::vector<double> vy = {-0.01, -0.02, -0.03};
    const std::vector<double> vz = {0.04, 0.05, 0.06};
    const std::vector<double> mass = {12.011, 1.008, 15.999};
    const std::vector<double> charge = {-0.5, 0.25, 0.25};
    const std::vector<std::int32_t> type = {1, 2, 3};
    const std::vector<std::int32_t> tag = {10, 11, 12};

    molrs::frame_set_column_f64(
        *fr, "atoms", "x", ::rust::Slice<const double>(x.data(), x.size()));
    molrs::frame_set_column_f64(
        *fr, "atoms", "y", ::rust::Slice<const double>(y.data(), y.size()));
    molrs::frame_set_column_f64(
        *fr, "atoms", "z", ::rust::Slice<const double>(z.data(), z.size()));
    molrs::frame_set_column_f64(
        *fr, "atoms", "vx", ::rust::Slice<const double>(vx.data(), vx.size()));
    molrs::frame_set_column_f64(
        *fr, "atoms", "vy", ::rust::Slice<const double>(vy.data(), vy.size()));
    molrs::frame_set_column_f64(
        *fr, "atoms", "vz", ::rust::Slice<const double>(vz.data(), vz.size()));
    molrs::frame_set_column_f64(
        *fr, "atoms", "mass",
        ::rust::Slice<const double>(mass.data(), mass.size()));
    molrs::frame_set_column_f64(
        *fr, "atoms", "charge",
        ::rust::Slice<const double>(charge.data(), charge.size()));
    molrs::frame_set_column_u32(
        *fr, "atoms", "atomic_number",
        ::rust::Slice<const std::int32_t>(type.data(), type.size()));
    molrs::frame_set_column_i32(
        *fr, "atoms", "tag",
        ::rust::Slice<const std::int32_t>(tag.data(), tag.size()));

    // bonds reference atoms by GLOBAL TAG (not row index): cpu::from_molrs
    // calls atoms.add_bond(ti, tj, ...) which resolves ti/tj through the
    // tag->local map, mirroring the CPU sibling test. Atom tags are {10,11,12},
    // so this bond joins tags 10 and 11. Using row indices {0,1} here would be
    // silently dropped by add_bond (tag 0 not in the map) -> n_bonds()==0.
    const std::vector<std::int32_t> bi = {10}, bj = {11}, bt = {1};
    molrs::frame_set_column_u32(
        *fr, "bonds", "atomi",
        ::rust::Slice<const std::int32_t>(bi.data(), bi.size()));
    molrs::frame_set_column_u32(
        *fr, "bonds", "atomj",
        ::rust::Slice<const std::int32_t>(bj.data(), bj.size()));
    molrs::frame_set_column_u32(
        *fr, "bonds", "type_id",
        ::rust::Slice<const std::int32_t>(bt.data(), bt.size()));

    const std::vector<double> h = {20.0, 0.0, 0.0, 0.0, 20.0,
                                   0.0,  0.0, 0.0, 20.0};
    molrs::frame_set_box(*fr, ::rust::Slice<const double>(h.data(), h.size()));
    return fr;
}

} // namespace

// ── ac-002: the materialized cuda::Frame carries the aligned field set ──

TEST(CudaMaterialize, AlignedFieldSet) {
    auto fr = make_full_frame();
    cuda::Frame d = cuda::from_molrs(*fr);

    EXPECT_EQ(d.n_atoms(), 3);
    EXPECT_EQ(d.n_bonds(), 1);

    // Per-atom arrays aligned with cpu::Frame: charge / tag / image present.
    EXPECT_NE(d.atoms.x, nullptr);
    EXPECT_NE(d.atoms.charge, nullptr);
    EXPECT_NE(d.atoms.tag, nullptr);
    EXPECT_NE(d.atoms.image, nullptr);
    // System scalars are a single device-resident SystemScalars struct.
    EXPECT_NE(d.scalars, nullptr);
    EXPECT_NE(d.box, nullptr);

    cuda::destroy(d);
}

// ── ac-003: molrs.Frame -> cuda::Frame -> to_molrs round-trips ────────

TEST(CudaMaterialize, RoundTrip) {
    auto fr = make_full_frame();

    const std::vector<double> x = {0.0, 1.5, 3.0};
    const std::vector<double> vy = {-0.01, -0.02, -0.03};
    const std::vector<double> charge = {-0.5, 0.25, 0.25};

    cuda::Frame d = cuda::from_molrs(*fr);
    ASSERT_EQ(d.n_atoms(), 3);

    // Explicit D2H write-back into the same molrs.Frame.
    cuda::to_molrs(d, *fr);

    auto rx = molrs::frame_column_f64(*fr, "atoms", "x");
    auto rvy = molrs::frame_column_f64(*fr, "atoms", "vy");
    auto rq = molrs::frame_column_f64(*fr, "atoms", "charge");
    ASSERT_EQ(rx.size(), 3u);
    ASSERT_EQ(rvy.size(), 3u);
    ASSERT_EQ(rq.size(), 3u);
    // cuda::Frame stores per-atom arrays as real_t (float by default), so the
    // H2D/D2H round trip is element-wise equal only to float precision.
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(rx[i], x[i], 1e-5);
        EXPECT_NEAR(rvy[i], vy[i], 1e-5);
        EXPECT_NEAR(rq[i], charge[i], 1e-5);
    }

    // System scalars cross the boundary as exact typed meta, never a synthetic
    // "scalars" block (schema v2) — mirrors tests/cpu/test_materialize.cpp.
    EXPECT_FALSE(molrs::frame_has_block(*fr, "scalars"));

    cuda::destroy(d);
}

// ── ac-003: required fields fail loud (delegated to cpu::from_molrs) ───

TEST(CudaMaterialize, MissingPositionsThrows) {
    auto fr = molrs::frame_new();
    const std::vector<std::int32_t> type = {1, 2};
    molrs::frame_set_column_u32(
        *fr, "atoms", "atomic_number",
        ::rust::Slice<const std::int32_t>(type.data(), type.size()));

    try {
        cuda::Frame d = cuda::from_molrs(*fr);
        cuda::destroy(d);
        FAIL() << "cuda::from_molrs must throw when atoms.x is absent";
    } catch (const std::exception& e) {
        EXPECT_NE(std::string(e.what()).find("x"), std::string::npos)
            << "exception message must name the missing field: " << e.what();
    }
}
