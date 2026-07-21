/// @file tests/cuda/md/test_atom_cell_sort.cu
/// @brief Unit test for the device-side owned-atom cell-sort primitive
///        (spec multi-gpu-atom-cell-sort-01-permute).
///
/// Criteria:
///   (a) every per-atom array (LJ + full bonded SoA) is reordered by the
///       SAME permutation — verified by tagging each atom's payload with a
///       function of its global tag and checking the relation holds post-sort;
///   (b) permute∘inverse == identity, byte-for-byte, on every array;
///   (c) sorted atoms are non-decreasing in cell index.
///
/// Fixture mirrors test_neighborlist_skin_gating.cu (cpu::Frame builder) but
/// materializes the device frame directly via cpu::to_cuda +
/// cuda::materialize_per_atom_bonded (the multi-GPU bonded path), since the
/// primitive permutes the per-atom bonded SoA which the single-GPU
/// materialize leaves null.

#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/frame/transfer.h"
#include "atomiverse/cuda/frame/transfer.cuh"

#include "atom_cell_sort.cuh"

#include <cuda_runtime.h>

using namespace ATV;
using ATV::cuda::real_t;
using ATV::cuda::md::nlist::apply_permutation_owned;
using ATV::cuda::md::nlist::AtomCellSortBuffers;
using ATV::cuda::md::nlist::cell_sort_owned;

namespace {

// Fixture topology strides — set on the frame via reserve_topology
// (per-atom topology column widths are runtime values now).
constexpr int MB = 6;
constexpr int MA = 6;
constexpr int MD = 8;
constexpr int MS = 16;

/// A cubic lattice of n_side^3 atoms in a box split into dim^3 cells, with a
/// fully populated per-atom bonded SoA whose entries are a deterministic
/// function of the atom's global tag — so we can detect any cross-array
/// permutation mismatch after the sort.
cpu::Frame make_bonded_lattice(int n_side, double spacing) {
    const int n = n_side * n_side * n_side;
    const double box_len = double(n_side) * spacing;

    cpu::Frame frame(n);
    // Populate the wide bonded SoA directly (not via add_bond), so size the
    // molecular storage explicitly before writing.
    frame.reserve_topology(MB, MA, MD, MS);
    auto x = frame.span(kAtom_x);
    auto y = frame.span(kAtom_y);
    auto z = frame.span(kAtom_z);
    auto vx = frame.span(kAtom_vx);
    auto vy = frame.span(kAtom_vy);
    auto vz = frame.span(kAtom_vz);
    auto fx = frame.span(kAtom_fx);
    auto fy = frame.span(kAtom_fy);
    auto fz = frame.span(kAtom_fz);
    auto mass = frame.span(kAtom_mass);
    auto charge = frame.span(kAtom_charge);
    auto type = frame.span(kAtom_type);
    auto tag = frame.span(kAtom_tag);
    auto image = frame.span(kAtom_image);
    auto num_bond = frame.span(kAtom_num_bond);
    auto bond_atom = frame.span(kAtom_bond_atom);
    auto bond_type = frame.span(kAtom_bond_type);
    auto num_angle = frame.span(kAtom_num_angle);
    auto angle_atom = frame.span(kAtom_angle_atom);
    auto angle_type = frame.span(kAtom_angle_type);
    auto num_dihedral = frame.span(kAtom_num_dihedral);
    auto dihedral_atom = frame.span(kAtom_dihedral_atom);
    auto dihedral_type = frame.span(kAtom_dihedral_type);
    auto nspecial = frame.span(kAtom_nspecial);
    auto special = frame.span(kAtom_special);

    int idx = 0;
    for (int ix = 0; ix < n_side; ++ix)
        for (int iy = 0; iy < n_side; ++iy)
            for (int iz = 0; iz < n_side; ++iz) {
                x[idx] = (double(ix) + 0.5) * spacing;
                y[idx] = (double(iy) + 0.5) * spacing;
                z[idx] = (double(iz) + 0.5) * spacing;
                vx[idx] = 0.1 * idx;
                vy[idx] = 0.2 * idx;
                vz[idx] = 0.3 * idx;
                fx[idx] = -0.1 * idx;
                fy[idx] = -0.2 * idx;
                fz[idx] = -0.3 * idx;
                mass[idx] = 1.0 + 0.01 * idx;
                charge[idx] = (idx % 2 == 0) ? 0.5 : -0.5;
                type[idx] = idx % 3;
                tag[idx] = idx; // identity tags
                image(idx, 0) = static_cast<std::int8_t>(idx % 3 - 1);
                image(idx, 1) = static_cast<std::int8_t>(idx % 5 - 2);
                image(idx, 2) = static_cast<std::int8_t>(idx % 7 - 3);

                // Bonded SoA — partner tags are deterministic in idx so any
                // mispermutation across arrays is observable. (Values need
                // not form a physically valid topology for this primitive.)
                const TagInt t = static_cast<TagInt>(idx);
                num_bond[idx] = 2;
                for (int m = 0; m < MB; ++m) {
                    bond_atom(idx, m) = t * 100 + m;
                    bond_type(idx, m) = idx % 4 + m;
                }
                num_angle[idx] = 1;
                for (int m = 0; m < MA; ++m) {
                    angle_atom(idx, 2 * m + 0) = t * 200 + 2 * m;
                    angle_atom(idx, 2 * m + 1) = t * 200 + 2 * m + 1;
                    angle_type(idx, m) = idx % 5 + m;
                }
                num_dihedral[idx] = 1;
                for (int m = 0; m < MD; ++m) {
                    dihedral_atom(idx, 3 * m + 0) = t * 300 + 3 * m;
                    dihedral_atom(idx, 3 * m + 1) = t * 300 + 3 * m + 1;
                    dihedral_atom(idx, 3 * m + 2) = t * 300 + 3 * m + 2;
                    dihedral_type(idx, m) = idx % 6 + m;
                }
                nspecial(idx, 0) = 1;
                nspecial(idx, 1) = 2;
                nspecial(idx, 2) = 3;
                for (int m = 0; m < MS; ++m)
                    special(idx, m) = t * 400 + m;
                ++idx;
            }
    frame.simbox = cpu::Box::cubic(box_len);
    return frame;
}

template <typename T>
std::vector<T> read_back(const T* d_ptr, std::size_t count) {
    std::vector<T> h(count);
    if (count > 0)
        EXPECT_EQ(cudaMemcpy(h.data(), d_ptr, count * sizeof(T),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
    return h;
}

// Compute the host-side cell index using the SAME convention as the primitive
// (and nlist_direct.cu::cell_of): cell = (cx*dy + cy)*dz + cz.
int host_cell_of(double x, double y, double z, double L, int dim) {
    auto comp = [&](double q) {
        double f = q / L;
        f -= std::floor(f);
        int c = static_cast<int>(f * dim);
        if (c < 0)
            c = 0;
        if (c >= dim)
            c = dim - 1;
        return c;
    };
    int cx = comp(x), cy = comp(y), cz = comp(z);
    return (cx * dim + cy) * dim + cz;
}

} // namespace

class AtomCellSort : public ::testing::Test {
  protected:
    void SetUp() override {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
            GTEST_SKIP() << "no CUDA device";
    }
};

// ── (a) consistent permutation + (c) non-decreasing cell index ──────────────
TEST_F(AtomCellSort, ConsistentPermutationAndSortedByCell) {
    const int n_side = 8; // 512 atoms across multiple cells
    const double spacing = 1.5;
    const int dim = 4; // 4^3 = 64 cells
    const double L = double(n_side) * spacing;

    cpu::Frame frame = make_bonded_lattice(n_side, spacing);
    const int n = frame.atoms.nlocal;

    cuda::Frame d = cpu::to_cuda(frame, /*capacity=*/n);
    cuda::materialize_per_atom_bonded(d, frame);

    AtomCellSortBuffers buf;
    cell_sort_owned(d, real_t(L), real_t(L), real_t(L), dim, dim, dim,
                    /*periodic=*/true, buf);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Read back the sorted per-atom arrays.
    auto h_x = read_back(d.atoms.x, n);
    auto h_y = read_back(d.atoms.y, n);
    auto h_z = read_back(d.atoms.z, n);
    auto h_tag = read_back(d.atoms.tag, n);
    auto h_type = read_back(d.atoms.type, n);
    auto h_charge = read_back(d.atoms.charge, n);
    auto h_image = read_back(d.atoms.image, std::size_t(n) * 3);
    auto h_num_bond = read_back(d.atoms.num_bond, n);
    auto h_bond_atom = read_back(d.atoms.bond_atom, std::size_t(n) * MB);
    auto h_bond_type = read_back(d.atoms.bond_type, std::size_t(n) * MB);
    auto h_angle_atom = read_back(d.atoms.angle_atom, std::size_t(n) * 2 * MA);
    auto h_dihedral_atom =
        read_back(d.atoms.dihedral_atom, std::size_t(n) * 3 * MD);
    auto h_special = read_back(d.atoms.special, std::size_t(n) * MS);

    // (c) cell index non-decreasing across sorted rows.
    int prev_cell = -1;
    for (int k = 0; k < n; ++k) {
        int cell = host_cell_of(h_x[k], h_y[k], h_z[k], L, dim);
        EXPECT_GE(cell, prev_cell) << "row " << k << " out of cell order";
        prev_cell = cell;
    }

    // (a) every array consistent with the SAME permutation. We tagged each
    // payload as a function of the original tag; since tags were identity, the
    // original index of sorted row k is exactly h_tag[k]. Verify EVERY array
    // matches the value the original index would have produced.
    for (int k = 0; k < n; ++k) {
        const std::int64_t orig = h_tag[k];
        ASSERT_GE(orig, 0);
        ASSERT_LT(orig, n);

        EXPECT_EQ(h_type[k], int(orig % 3));
        EXPECT_DOUBLE_EQ(double(h_charge[k]), (orig % 2 == 0) ? 0.5 : -0.5);

        EXPECT_EQ(int(h_image[std::size_t(k) * 3 + 0]),
                  int(std::int8_t(orig % 3 - 1)));
        EXPECT_EQ(int(h_image[std::size_t(k) * 3 + 1]),
                  int(std::int8_t(orig % 5 - 2)));
        EXPECT_EQ(int(h_image[std::size_t(k) * 3 + 2]),
                  int(std::int8_t(orig % 7 - 3)));

        EXPECT_EQ(h_num_bond[k], 2);
        for (int m = 0; m < MB; ++m) {
            EXPECT_EQ(h_bond_atom[std::size_t(k) * MB + m], orig * 100 + m);
            EXPECT_EQ(h_bond_type[std::size_t(k) * MB + m], int(orig % 4 + m));
        }
        for (int m = 0; m < 2 * MA; ++m)
            EXPECT_EQ(h_angle_atom[std::size_t(k) * 2 * MA + m],
                      orig * 200 + m);
        for (int m = 0; m < 3 * MD; ++m)
            EXPECT_EQ(h_dihedral_atom[std::size_t(k) * 3 * MD + m],
                      orig * 300 + m);
        for (int m = 0; m < MS; ++m)
            EXPECT_EQ(h_special[std::size_t(k) * MS + m], orig * 400 + m);
    }

    cuda::destroy(d);
}

// ── (b) permute∘inverse == identity, byte-for-byte ──────────────────────────
TEST_F(AtomCellSort, PermuteThenInverseIsIdentity) {
    const int n_side = 8;
    const double spacing = 1.5;
    const int dim = 4;
    const double L = double(n_side) * spacing;

    cpu::Frame frame = make_bonded_lattice(n_side, spacing);
    const int n = frame.atoms.nlocal;

    // Two device frames from the same source: `ref` untouched, `d` sorted then
    // inverse-applied. They must end byte-for-byte equal.
    cuda::Frame ref = cpu::to_cuda(frame, n);
    cuda::materialize_per_atom_bonded(ref, frame);

    cuda::Frame d = cpu::to_cuda(frame, n);
    cuda::materialize_per_atom_bonded(d, frame);

    AtomCellSortBuffers buf;
    cell_sort_owned(d, real_t(L), real_t(L), real_t(L), dim, dim, dim, true,
                    buf);
    // Undo with the inverse produced by the sort.
    apply_permutation_owned(d, buf.inverse.data(), n, buf);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    auto eq_r = [&](const real_t* da, const real_t* db, std::size_t c) {
        auto ha = read_back(da, c);
        auto hb = read_back(db, c);
        EXPECT_EQ(ha, hb);
    };
    auto eq_i = [&](const int* da, const int* db, std::size_t c) {
        auto ha = read_back(da, c);
        auto hb = read_back(db, c);
        EXPECT_EQ(ha, hb);
    };
    auto eq_i64 = [&](const std::int64_t* da, const std::int64_t* db,
                      std::size_t c) {
        auto ha = read_back(da, c);
        auto hb = read_back(db, c);
        EXPECT_EQ(ha, hb);
    };
    auto eq_i8 = [&](const std::int8_t* da, const std::int8_t* db,
                     std::size_t c) {
        auto ha = read_back(da, c);
        auto hb = read_back(db, c);
        EXPECT_EQ(ha, hb);
    };

    const std::size_t un = std::size_t(n);
    eq_r(d.atoms.x, ref.atoms.x, un);
    eq_r(d.atoms.y, ref.atoms.y, un);
    eq_r(d.atoms.z, ref.atoms.z, un);
    eq_r(d.atoms.vx, ref.atoms.vx, un);
    eq_r(d.atoms.vy, ref.atoms.vy, un);
    eq_r(d.atoms.vz, ref.atoms.vz, un);
    eq_r(d.atoms.fx, ref.atoms.fx, un);
    eq_r(d.atoms.fy, ref.atoms.fy, un);
    eq_r(d.atoms.fz, ref.atoms.fz, un);
    eq_r(d.atoms.mass, ref.atoms.mass, un);
    eq_r(d.atoms.charge, ref.atoms.charge, un);
    eq_i(d.atoms.type, ref.atoms.type, un);
    eq_i64(d.atoms.tag, ref.atoms.tag, un);
    eq_i8(d.atoms.image, ref.atoms.image, un * 3);

    eq_i(d.atoms.num_bond, ref.atoms.num_bond, un);
    eq_i64(d.atoms.bond_atom, ref.atoms.bond_atom, un * MB);
    eq_i(d.atoms.bond_type, ref.atoms.bond_type, un * MB);
    eq_i(d.atoms.num_angle, ref.atoms.num_angle, un);
    eq_i64(d.atoms.angle_atom, ref.atoms.angle_atom, un * 2 * MA);
    eq_i(d.atoms.angle_type, ref.atoms.angle_type, un * MA);
    eq_i(d.atoms.num_dihedral, ref.atoms.num_dihedral, un);
    eq_i64(d.atoms.dihedral_atom, ref.atoms.dihedral_atom, un * 3 * MD);
    eq_i(d.atoms.dihedral_type, ref.atoms.dihedral_type, un * MD);
    eq_i(d.atoms.nspecial, ref.atoms.nspecial, un * 3);
    eq_i64(d.atoms.special, ref.atoms.special, un * MS);

    cuda::destroy(d);
    cuda::destroy(ref);
}
