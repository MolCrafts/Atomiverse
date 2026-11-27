/// @file tests/mpi/test_pencil_fft.cpp
/// @brief PencilFft against a serial FFTW r2c / c2r of the same mesh.
///
/// The mesh is a fixed pseudo-random real field on a non-cubic 6 x 10 x 8
/// grid; bricks are x slabs by rank (uneven when the rank count does not
/// divide 6). Every rank rebuilds the whole field and its serial spectrum,
/// so the check needs no gather.

#include <gtest/gtest.h>

#include <mpi.h>

#include <array>
#include <complex>
#include <cstddef>
#include <vector>

#include <fftw3.h>

#include "atomiverse/cpu/longrange/pencil_fft.h"

using namespace ATV;

namespace {

constexpr int kK0 = 6, kK1 = 10, kK2 = 8;
constexpr int kN2c = kK2 / 2 + 1;

double field(int x, int y, int z) {
    // Deterministic, aperiodic in every index.
    unsigned h = static_cast<unsigned>(x * 73856093)
                 ^ static_cast<unsigned>(y * 19349663)
                 ^ static_cast<unsigned>(z * 83492791);
    h = h * 1664525u + 1013904223u;
    return static_cast<double>(h % 20011u) / 20011.0 - 0.5;
}

std::vector<std::complex<double>> serial_spectrum() {
    std::vector<double> real(static_cast<std::size_t>(kK0 * kK1 * kK2));
    for (int x = 0; x < kK0; ++x)
        for (int y = 0; y < kK1; ++y)
            for (int z = 0; z < kK2; ++z)
                real[static_cast<std::size_t>((x * kK1 + y) * kK2 + z)] =
                    field(x, y, z);
    std::vector<std::complex<double>> spec(
        static_cast<std::size_t>(kK0 * kK1 * kN2c));
    fftw_plan p = fftw_plan_dft_r2c_3d(
        kK0, kK1, kK2, real.data(),
        reinterpret_cast<fftw_complex*>(spec.data()), FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);
    return spec;
}

void check(int active) {
    int me = 0, nprocs = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    const int xlo = me * kK0 / nprocs;
    const int xhi = (me + 1) * kK0 / nprocs - 1;
    const std::array<int, 6> brick{xlo, xhi, 0, kK1 - 1, 0, kK2 - 1};
    longrange::PencilFft fft(MPI_COMM_WORLD, {kK0, kK1, kK2}, brick, active);

    std::vector<double> mine;
    for (int x = xlo; x <= xhi; ++x)
        for (int y = 0; y < kK1; ++y)
            for (int z = 0; z < kK2; ++z)
                mine.push_back(field(x, y, z));
    fft.forward(mine.data());

    const auto ref = serial_spectrum();
    const auto box = fft.spectrum_box();
    const int n2 = box[3] - box[2] + 1;
    std::size_t owned = 0;
    for (int m1 = box[0]; m1 <= box[1]; ++m1)
        for (int m2 = box[2]; m2 <= box[3]; ++m2)
            for (int m0 = 0; m0 < kK0; ++m0) {
                const auto got = fft.spectrum()[static_cast<std::size_t>(
                    ((m1 - box[0]) * n2 + (m2 - box[2])) * kK0 + m0)];
                const auto want =
                    ref[static_cast<std::size_t>((m0 * kK1 + m1) * kN2c + m2)];
                EXPECT_NEAR(got.real(), want.real(), 1e-11)
                    << m0 << ' ' << m1 << ' ' << m2;
                EXPECT_NEAR(got.imag(), want.imag(), 1e-11)
                    << m0 << ' ' << m1 << ' ' << m2;
                ++owned;
            }
    EXPECT_EQ(owned, fft.spectrum_size());
    // The spectrum boxes tile the half spectrum.
    long long mine_n = static_cast<long long>(owned), all_n = 0;
    MPI_Allreduce(&mine_n, &all_n, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    EXPECT_EQ(all_n, static_cast<long long>(kK0) * kK1 * kN2c);
    if (me >= active)
        EXPECT_EQ(owned, 0u);

    // Unnormalized round trip.
    std::vector<double> back(mine.size(), 0.0);
    fft.backward(back.data());
    const double scale = static_cast<double>(kK0) * kK1 * kK2;
    for (std::size_t i = 0; i < mine.size(); ++i)
        EXPECT_NEAR(back[i], scale * mine[i], 1e-10) << i;
}

} // namespace

TEST(PencilFft, ForwardMatchesSerialR2cAndRoundTrips) {
    int nprocs = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    check(nprocs);
}

TEST(PencilFft, InactiveRanksOnlyFeedAndReceiveTheirBrick) {
    int nprocs = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    check(nprocs > 1 ? nprocs - 1 : 1);
}

TEST(PencilFft, BifactorIsTheMostEvenSplit) {
    EXPECT_EQ(longrange::pencil_bifactor(1), (std::array<int, 2>{1, 1}));
    EXPECT_EQ(longrange::pencil_bifactor(7), (std::array<int, 2>{1, 7}));
    EXPECT_EQ(longrange::pencil_bifactor(12), (std::array<int, 2>{3, 4}));
    EXPECT_EQ(longrange::pencil_bifactor(192), (std::array<int, 2>{12, 16}));
    EXPECT_THROW(longrange::pencil_bifactor(0), std::invalid_argument);
}
