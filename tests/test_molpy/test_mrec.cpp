/// @file tests/test_molpy/test_mrec.cpp
/// @brief Drift fence for molpy MrecWriterRef / read_mrec_trajectory_frame and
///        write_mrec_frame / read_mrec_frame, and the cpu::write_mrec_frame
///        checkpoint door over them.

#include <array>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/cpu/io/frame_writer.h"
#include "atomiverse/molpy.hpp"
#include "rust/cxx.h"

namespace {

std::string tmp_mrec(const char* name) {
    auto p = std::filesystem::temp_directory_path() / name;
    std::error_code ec;
    std::filesystem::remove_all(p, ec);
    return p.string();
}

::rust::Box<molrs::FrameRef> one_atom() {
    auto fr = molrs::frame_new();
    const std::vector<double> x = {0.0};
    const std::vector<double> y = {0.0};
    const std::vector<double> z = {0.0};
    const std::vector<std::uint64_t> znum = {18};
    const std::vector<double> mass = {39.948};
    molrs::frame_set_column_f64(
        *fr, "atoms", "x", ::rust::Slice<const double>(x.data(), x.size()));
    molrs::frame_set_column_f64(
        *fr, "atoms", "y", ::rust::Slice<const double>(y.data(), y.size()));
    molrs::frame_set_column_f64(
        *fr, "atoms", "z", ::rust::Slice<const double>(z.data(), z.size()));
    molrs::frame_set_column_u64(
        *fr, "atoms", "atomic_number",
        ::rust::Slice<const std::uint64_t>(znum.data(), znum.size()));
    molrs::frame_set_column_f64(
        *fr, "atoms", "mass",
        ::rust::Slice<const double>(mass.data(), mass.size()));
    const std::vector<double> h = {20, 0, 0, 0, 20, 0, 0, 0, 20};
    molrs::frame_set_box(*fr, ::rust::Slice<const double>(h.data(), h.size()),
                         ::rust::Slice<const double>(
                             std::array<double, 3>{0.0, 0.0, 0.0}.data(), 3),
                         ::rust::Slice<const bool>(
                             std::array<bool, 3>{true, true, true}.data(), 3));
    return fr;
}

TEST(MolpyMrec, MrecWriterCreateAppendFlushClose) {
    const std::string path = tmp_mrec("atv_molpy_traj.mrec");
    auto schema = one_atom();
    auto writer = molrs::mrec_writer_create({path.data(), path.size()}, *schema,
                                            /*flush_every=*/1,
                                            /*durable=*/false);
    auto frame = one_atom();
    molrs::mrec_writer_append(*writer, *frame, /*step=*/0, /*time=*/0.0,
                              /*has_time=*/true);
    molrs::mrec_writer_flush(*writer);
    EXPECT_GE(molrs::mrec_writer_committed(*writer), 1u);
    molrs::mrec_writer_close(std::move(writer));

    auto back = molrs::read_mrec_trajectory_frame({path.data(), path.size()},
                                                  /*index=*/0);
    auto z = molrs::frame_column_u64(*back, "atoms", "atomic_number");
    ASSERT_EQ(z.size(), 1u);
    EXPECT_EQ(z[0], 18u);
    EXPECT_THROW(molrs::read_mrec_trajectory_frame({path.data(), path.size()},
                                                   /*index=*/1),
                 rust::Error);
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
}

TEST(MolpyMrec, WriteMrecFrameThenReadMrecFrame) {
    const std::string path = tmp_mrec("atv_molpy_single.mrec");
    auto frame = one_atom();
    molrs::write_mrec_frame({path.data(), path.size()}, *frame);
    auto fr = molrs::read_mrec_frame({path.data(), path.size()});
    auto xr = molrs::frame_column_f64(*fr, "atoms", "x");
    ASSERT_EQ(xr.size(), 1u);
    EXPECT_DOUBLE_EQ(xr[0], 0.0);
    auto mass = molrs::frame_column_f64(*fr, "atoms", "mass");
    ASSERT_EQ(mass.size(), 1u);
    EXPECT_DOUBLE_EQ(mass[0], 39.948);
    // mrec stores the species as atomic_number, not element symbols.
    auto zr = molrs::frame_column_u64(*fr, "atoms", "atomic_number");
    ASSERT_EQ(zr.size(), 1u);
    EXPECT_EQ(zr[0], 18u);
    const auto h = molrs::frame_box(*fr).h;
    ASSERT_EQ(h.size(), 9u);
    EXPECT_DOUBLE_EQ(h[0], 20.0);
    EXPECT_DOUBLE_EQ(h[4], 20.0);
    EXPECT_DOUBLE_EQ(h[8], 20.0);
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
}

// The engine's checkpoint door: one frame + named per-atom fields, replaced
// whole on every call.
TEST(MolpyMrec, CpuWriteMrecFrameRoundTripsItsFields) {
    const std::string path = tmp_mrec("atv_cpu_write_mrec_frame.mrec");
    const std::array<double, 9> box = {5, 0, 0, 0, 5, 0, 0, 0, 5};
    ATV::cpu::write_mrec_frame(path, {8, 1}, {0.0, 1.0}, {0.5, 0.25},
                               {0.0, -1.0}, box, {"q", "w"},
                               {-0.8, 0.4, 1.0, 2.0});
    ATV::cpu::write_mrec_frame(path, {8, 1}, {0.0, 1.0}, {0.5, 0.25},
                               {0.0, -1.0}, box, {"q", "w"},
                               {-0.6, 0.3, 3.0, 4.0});
    auto fr = molrs::read_mrec_frame({path.data(), path.size()});
    const auto zr = molrs::frame_column_u64(*fr, "atoms", "atomic_number");
    EXPECT_EQ(std::vector<std::uint64_t>(zr.begin(), zr.end()),
              (std::vector<std::uint64_t>{8, 1}));
    const auto y = molrs::frame_column_f64(*fr, "atoms", "y");
    EXPECT_EQ(std::vector<double>(y.begin(), y.end()),
              (std::vector<double>{0.5, 0.25}));
    const auto q = molrs::frame_column_f64(*fr, "atoms", "q");
    EXPECT_EQ(std::vector<double>(q.begin(), q.end()),
              (std::vector<double>{-0.6, 0.3}));
    const auto w = molrs::frame_column_f64(*fr, "atoms", "w");
    EXPECT_EQ(std::vector<double>(w.begin(), w.end()),
              (std::vector<double>{3.0, 4.0}));
    const auto h = molrs::frame_box(*fr).h;
    EXPECT_EQ(std::vector<double>(h.begin(), h.end()),
              std::vector<double>(box.begin(), box.end()));
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
}

TEST(MolpyMrec, CpuWriteMrecFrameRefusesMalformedInput) {
    const std::string path = tmp_mrec("atv_cpu_write_mrec_frame_bad.mrec");
    const std::array<double, 9> box = {5, 0, 0, 0, 5, 0, 0, 0, 5};
    // A field_data length other than n_fields x n_atoms.
    EXPECT_THROW(ATV::cpu::write_mrec_frame(path, {8, 1}, {0.0, 1.0},
                                            {0.0, 0.0}, {0.0, 0.0}, box, {"q"},
                                            {1.0}),
                 std::invalid_argument);
    // A field that would overwrite a coordinate, or one named twice.
    EXPECT_THROW(ATV::cpu::write_mrec_frame(path, {8}, {0.0}, {0.0}, {0.0}, box,
                                            {"x"}, {1.0}),
                 std::invalid_argument);
    EXPECT_THROW(ATV::cpu::write_mrec_frame(path, {8}, {0.0}, {0.0}, {0.0}, box,
                                            {"q", "q"}, {1.0, 2.0}),
                 std::invalid_argument);
    // kAtom_atomic_number carries Z: negative is refused before it crosses, Z > 118
    // by molpy.
    EXPECT_THROW(ATV::cpu::write_mrec_frame(path, {-1}, {0.0}, {0.0}, {0.0},
                                            box, {}, {}),
                 std::invalid_argument);
    EXPECT_THROW(ATV::cpu::write_mrec_frame(path, {119}, {0.0}, {0.0}, {0.0},
                                            box, {}, {}),
                 std::runtime_error);
    // A coordinate count other than the atom count, and a singular cell.
    EXPECT_THROW(ATV::cpu::write_mrec_frame(path, {8}, {0.0, 1.0}, {0.0, 1.0},
                                            {0.0, 1.0}, box, {}, {}),
                 std::runtime_error);
    const std::array<double, 9> singular = {1, 0, 0, 0, 0, 0, 0, 0, 1};
    EXPECT_THROW(ATV::cpu::write_mrec_frame(path, {8}, {0.0}, {0.0}, {0.0},
                                            singular, {}, {}),
                 std::runtime_error);
    EXPECT_FALSE(std::filesystem::exists(path));
}

} // namespace
