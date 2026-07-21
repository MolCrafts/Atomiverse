/// @file tests/test_torch_materialize.cu
/// @brief Round-trip test for torch::materialize / torch::write_back.
///
/// Mirrors tests/cpu/test_materialize.cpp: a molrs.Frame is materialized
/// into a torch::Frame (H2D) and written back (D2H), and the per-atom
/// results plus system scalars are compared element-wise.

#include <gtest/gtest.h>

#ifdef ATV_HAS_TORCH

#include <cstdint>
#include <exception>
#include <vector>

#include "atomiverse/molrs.hpp"
#include "atomiverse/torch/frame/frame.cuh"
#include "atomiverse/torch/frame/materialize.cuh"

namespace atorch = ATV::torch;

namespace {

/// @brief Build a fully-populated molrs.Frame with three atoms, one bond,
/// one angle, one dihedral, and a cubic box.
::rust::Box<molrs::FrameRef> make_full_frame() {
    auto fr = molrs::frame_new();

    const std::vector<double> x = {0.0, 1.5, 3.0};
    const std::vector<double> y = {0.1, 1.6, 3.1};
    const std::vector<double> z = {0.2, 1.7, 3.2};
    const std::vector<double> vx = {0.01, 0.02, 0.03};
    const std::vector<double> vy = {-0.01, -0.02, -0.03};
    const std::vector<double> vz = {0.04, 0.05, 0.06};
    const std::vector<double> fx = {1.0, 2.0, 3.0};
    const std::vector<double> fy = {-1.0, -2.0, -3.0};
    const std::vector<double> fz = {0.5, 0.6, 0.7};
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
        *fr, "atoms", "fx", ::rust::Slice<const double>(fx.data(), fx.size()));
    molrs::frame_set_column_f64(
        *fr, "atoms", "fy", ::rust::Slice<const double>(fy.data(), fy.size()));
    molrs::frame_set_column_f64(
        *fr, "atoms", "fz", ::rust::Slice<const double>(fz.data(), fz.size()));
    molrs::frame_set_column_f64(
        *fr, "atoms", "mass",
        ::rust::Slice<const double>(mass.data(), mass.size()));
    molrs::frame_set_column_f64(
        *fr, "atoms", "charge",
        ::rust::Slice<const double>(charge.data(), charge.size()));
    molrs::frame_set_column_i32(
        *fr, "atoms", "type",
        ::rust::Slice<const std::int32_t>(type.data(), type.size()));
    molrs::frame_set_column_i32(
        *fr, "atoms", "tag",
        ::rust::Slice<const std::int32_t>(tag.data(), tag.size()));

    // Topology: one bond, one angle, one dihedral.
    const std::vector<std::int32_t> bi = {0}, bj = {1}, bt = {1};
    molrs::frame_set_column_i32(
        *fr, "bonds", "atomi",
        ::rust::Slice<const std::int32_t>(bi.data(), bi.size()));
    molrs::frame_set_column_i32(
        *fr, "bonds", "atomj",
        ::rust::Slice<const std::int32_t>(bj.data(), bj.size()));
    molrs::frame_set_column_i32(
        *fr, "bonds", "type",
        ::rust::Slice<const std::int32_t>(bt.data(), bt.size()));

    const std::vector<std::int32_t> ai = {0}, aj = {1}, ak = {2}, at = {1};
    molrs::frame_set_column_i32(
        *fr, "angles", "atomi",
        ::rust::Slice<const std::int32_t>(ai.data(), ai.size()));
    molrs::frame_set_column_i32(
        *fr, "angles", "atomj",
        ::rust::Slice<const std::int32_t>(aj.data(), aj.size()));
    molrs::frame_set_column_i32(
        *fr, "angles", "atomk",
        ::rust::Slice<const std::int32_t>(ak.data(), ak.size()));
    molrs::frame_set_column_i32(
        *fr, "angles", "type",
        ::rust::Slice<const std::int32_t>(at.data(), at.size()));

    const std::vector<std::int32_t> di = {0}, dj = {1}, dk = {2}, dl = {2},
                                    dt = {1};
    molrs::frame_set_column_i32(
        *fr, "dihedrals", "atomi",
        ::rust::Slice<const std::int32_t>(di.data(), di.size()));
    molrs::frame_set_column_i32(
        *fr, "dihedrals", "atomj",
        ::rust::Slice<const std::int32_t>(dj.data(), dj.size()));
    molrs::frame_set_column_i32(
        *fr, "dihedrals", "atomk",
        ::rust::Slice<const std::int32_t>(dk.data(), dk.size()));
    molrs::frame_set_column_i32(
        *fr, "dihedrals", "atoml",
        ::rust::Slice<const std::int32_t>(dl.data(), dl.size()));
    molrs::frame_set_column_i32(
        *fr, "dihedrals", "type",
        ::rust::Slice<const std::int32_t>(dt.data(), dt.size()));

    // Cubic box, row-major 3x3 H.
    const std::vector<double> h = {20.0, 0.0, 0.0, 0.0, 20.0,
                                   0.0,  0.0, 0.0, 20.0};
    molrs::frame_set_simbox(*fr,
                            ::rust::Slice<const double>(h.data(), h.size()));

    return fr;
}

/// @brief Read a 1-element CUDA Tensor as a host double.
double scalar_of(const atorch::Tensor& t) {
    return t.to(::at::ScalarType::Double).to(::torch::kCPU).item<double>();
}

} // anonymous namespace

// -- molrs.Frame -> torch::Frame -> write_back round trip ----------------

TEST(TorchMaterialize, RoundTrip) {
    auto fr = make_full_frame();

    atorch::Frame tf = atorch::materialize(*fr);

    ASSERT_EQ(tf.n_atoms(), 3);
    ASSERT_EQ(tf.n_bonds(), 1);
    ASSERT_EQ(tf.n_angles(), 1);
    ASSERT_EQ(tf.n_dihedrals(), 1);

    // Materialized positions land on the CUDA device.
    ASSERT_TRUE(tf.atoms.x.defined());
    EXPECT_TRUE(tf.atoms.x.is_cuda());
    EXPECT_EQ(tf.atoms.x.numel(), 3);

    auto cpu_x = tf.atoms.x.to(::torch::kCPU);
    auto cpu_q = tf.atoms.charge.to(::torch::kCPU);
    EXPECT_NEAR(cpu_x[1].item<double>(), 1.5, 1e-10);
    EXPECT_NEAR(cpu_q[0].item<double>(), -0.5, 1e-10);

    // Mutate the torch::Frame so write-back has something distinct to carry.
    auto real_opts = ::torch::TensorOptions()
                         .dtype(atorch::kRealDtype)
                         .device(::torch::kCUDA);
    auto scalar_opts = ::torch::TensorOptions()
                           .dtype(::at::ScalarType::Double)
                           .device(::torch::kCUDA);
    tf.atoms.x = ::torch::tensor({0.0, 7.25, 3.0}, real_opts);
    tf.atoms.vy = ::torch::tensor({-0.01, -0.02, -9.5}, real_opts);
    tf.atoms.fz = ::torch::tensor({42.0, 0.6, 0.7}, real_opts);
    tf.atoms.charge = ::torch::tensor({-0.875, 0.25, 0.25}, real_opts);
    tf.energy = ::torch::full({1}, -123.5, scalar_opts);
    tf.kinetic = ::torch::full({1}, 64.25, scalar_opts);
    tf.conserved = ::torch::full({1}, -59.25, scalar_opts);
    tf.stress = ::torch::tensor({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, scalar_opts);
    tf.stress_kinetic =
        ::torch::tensor({10.0, 11.0, 12.0, 13.0, 14.0, 15.0}, scalar_opts);

    atorch::write_back(tf, *fr);

    // Per-atom result columns round-trip element-wise.
    auto rx = molrs::frame_column_f64(*fr, "atoms", "x");
    auto rvy = molrs::frame_column_f64(*fr, "atoms", "vy");
    auto rfz = molrs::frame_column_f64(*fr, "atoms", "fz");
    auto rq = molrs::frame_column_f64(*fr, "atoms", "charge");
    ASSERT_EQ(rx.size(), 3u);
    EXPECT_DOUBLE_EQ(rx[1], 7.25);
    EXPECT_DOUBLE_EQ(rvy[2], -9.5);
    EXPECT_DOUBLE_EQ(rfz[0], 42.0);
    EXPECT_DOUBLE_EQ(rq[0], -0.875);

    // System scalars round-trip through the dedicated blocks.
    auto se = molrs::frame_column_f64(*fr, "scalars", "energy");
    auto sk = molrs::frame_column_f64(*fr, "scalars", "kinetic");
    auto sc = molrs::frame_column_f64(*fr, "scalars", "conserved");
    ASSERT_EQ(se.size(), 1u);
    EXPECT_DOUBLE_EQ(se[0], -123.5);
    EXPECT_DOUBLE_EQ(sk[0], 64.25);
    EXPECT_DOUBLE_EQ(sc[0], -59.25);

    auto ss = molrs::frame_column_f64(*fr, "stress", "value");
    auto ssk = molrs::frame_column_f64(*fr, "stress_kinetic", "value");
    ASSERT_EQ(ss.size(), 6u);
    ASSERT_EQ(ssk.size(), 6u);
    for (int k = 0; k < 6; ++k) {
        EXPECT_DOUBLE_EQ(ss[k], 1.0 + k);
        EXPECT_DOUBLE_EQ(ssk[k], 10.0 + k);
    }
}

// -- required-field validation surfaces through the CPU staging frame ----

TEST(TorchMaterialize, MissingPositionsThrows) {
    auto fr = molrs::frame_new();
    const std::vector<std::int32_t> type = {1, 2};
    molrs::frame_set_column_i32(
        *fr, "atoms", "type",
        ::rust::Slice<const std::int32_t>(type.data(), type.size()));

    EXPECT_THROW({ atorch::materialize(*fr); }, std::exception);
}

// -- optional fields default to zero without throwing -------------------

TEST(TorchMaterialize, OptionalFieldsDefaultToZero) {
    auto fr = molrs::frame_new();
    const std::vector<double> x = {0.0, 1.0, 2.0};
    const std::vector<std::int32_t> type = {1, 1, 1};
    molrs::frame_set_column_f64(
        *fr, "atoms", "x", ::rust::Slice<const double>(x.data(), x.size()));
    molrs::frame_set_column_f64(
        *fr, "atoms", "y", ::rust::Slice<const double>(x.data(), x.size()));
    molrs::frame_set_column_f64(
        *fr, "atoms", "z", ::rust::Slice<const double>(x.data(), x.size()));
    molrs::frame_set_column_i32(
        *fr, "atoms", "type",
        ::rust::Slice<const std::int32_t>(type.data(), type.size()));

    atorch::Frame tf = atorch::materialize(*fr); // must NOT throw

    ASSERT_EQ(tf.n_atoms(), 3);
    auto vx = tf.atoms.vx.to(::torch::kCPU);
    auto q = tf.atoms.charge.to(::torch::kCPU);
    for (int i = 0; i < 3; ++i) {
        EXPECT_DOUBLE_EQ(vx[i].item<double>(), 0.0);
        EXPECT_DOUBLE_EQ(q[i].item<double>(), 0.0);
    }
    EXPECT_EQ(tf.n_bonds(), 0);
    EXPECT_DOUBLE_EQ(scalar_of(tf.energy), 0.0);
    EXPECT_DOUBLE_EQ(scalar_of(tf.conserved), 0.0);
}

#else

TEST(TorchMaterialize, Disabled) {
    GTEST_SKIP() << "ATV_HAS_TORCH not defined";
}

#endif // ATV_HAS_TORCH
