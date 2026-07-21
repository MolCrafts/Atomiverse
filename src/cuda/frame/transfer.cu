/// @file cuda/frame/transfer.cu
/// @brief cpu::to_cuda(), cuda::to_cpu(), cuda::destroy() implementations.
///
/// Implements double<->real_t conversion with temporary buffers when the
/// device storage type differs from the host Real type. The system scalars
/// travel as a single device-resident `SystemScalars` struct.

// NOTE (topology representation): CUDA kernels have not been ported to
// per-atom topology — the single-GPU path consumes the flat
// BondList/AngleList/DihedralList, so cpu::to_cuda re-flattens from the
// CPU per-atom layout on every transfer. The multi-GPU path additionally
// carries the per-atom SoA for migration. See cuda/frame/frame.cuh.

#include "atomiverse/cpu/frame/transfer.h"
#include "atomiverse/cuda/frame/transfer.cuh"
#include "atomiverse/cuda/frame/types.cuh"

#include <cstdint>

#include <cuda_runtime.h>
#include <vector>

namespace ATV {

using cuda::prec_t;
using cuda::real_t;

// -- Narrowing helpers ------------------------------------------------
//
// cpu::Frame stores per-atom arrays as Real (double by default).
// cuda::Frame stores per-atom arrays as real_t (float by default).
// When real_t != Real, we must convert through a temporary buffer.

namespace {

/// @brief Copy host double array to device real_t array with narrowing.
void copy_h2d_narrow(real_t* d_dst, const Real* h_src, int n) {
    if constexpr (std::is_same_v<real_t, Real>) {
        cudaMemcpy(d_dst, h_src, n * sizeof(real_t), cudaMemcpyHostToDevice);
    } else {
        std::vector<real_t> tmp(n);
        for (int i = 0; i < n; ++i)
            tmp[i] = static_cast<real_t>(h_src[i]);
        cudaMemcpy(d_dst, tmp.data(), n * sizeof(real_t),
                   cudaMemcpyHostToDevice);
    }
}

/// @brief Copy device real_t array to host double array with widening.
void copy_d2h_widen(Real* h_dst, const real_t* d_src, int n) {
    if constexpr (std::is_same_v<real_t, Real>) {
        cudaMemcpy(h_dst, d_src, n * sizeof(real_t), cudaMemcpyDeviceToHost);
    } else {
        std::vector<real_t> tmp(n);
        cudaMemcpy(tmp.data(), d_src, n * sizeof(real_t),
                   cudaMemcpyDeviceToHost);
        for (int i = 0; i < n; ++i)
            h_dst[i] = static_cast<Real>(tmp[i]);
    }
}

} // anonymous namespace

// -- cpu::to_cuda ----------------------------------------------------

namespace cpu {

cuda::Frame to_cuda(const Frame& frame, int capacity) {
    int n = frame.atoms.size();
    // Per-atom arrays are allocated to `cap` (>= n) so multi-GPU halo ghosts
    // can be appended in [n_local, cap) without reallocation; only n values are
    // copied. cap == n for the default single-GPU path.
    int cap = (capacity > n) ? capacity : n;
    cuda::Frame d{};

    // Flatten per-atom topology (LAMMPS num_bond[i] / bond_atom[i][m]
    // layout on `cpu::Atom`) into flat (atomi[k] / atomj[k]) host buffers
    // for the CUDA-side flat `BondList`. CUDA kernels haven't been
    // ported to per-atom topology yet (out-of-scope for this refactor);
    // the bridge does the conversion every transfer. Each per-atom bond
    // appears as one flat row: (atom-i, partner_tag). Same for angles
    // (i, j, k) and dihedrals (i, j, k, l). Tags here are global atom
    // IDs (= local-row index in single-rank, since materialize sets
    // tag[row] = row).
    std::vector<int> flat_bond_i, flat_bond_j, flat_bond_t;
    std::vector<int> flat_ang_i, flat_ang_j, flat_ang_k, flat_ang_t;
    std::vector<int> flat_dih_i, flat_dih_j, flat_dih_k, flat_dih_l, flat_dih_t;
    for (int i = 0; i < n; ++i) {
        const int nb = frame.atoms.num_bond[i];
        for (int m = 0; m < nb; ++m) {
            flat_bond_i.push_back(i);
            flat_bond_j.push_back(
                static_cast<int>(frame.atoms.bond_atom(i, m)));
            flat_bond_t.push_back(frame.atoms.bond_type(i, m));
        }
        const int na = frame.atoms.num_angle[i];
        for (int m = 0; m < na; ++m) {
            flat_ang_i.push_back(
                static_cast<int>(frame.atoms.angle_atom(i, 2 * m + 0)));
            flat_ang_j.push_back(i);
            flat_ang_k.push_back(
                static_cast<int>(frame.atoms.angle_atom(i, 2 * m + 1)));
            flat_ang_t.push_back(frame.atoms.angle_type(i, m));
        }
        const int nd = frame.atoms.num_dihedral[i];
        for (int m = 0; m < nd; ++m) {
            flat_dih_i.push_back(
                static_cast<int>(frame.atoms.dihedral_atom(i, 3 * m + 0)));
            flat_dih_j.push_back(i);
            flat_dih_k.push_back(
                static_cast<int>(frame.atoms.dihedral_atom(i, 3 * m + 1)));
            flat_dih_l.push_back(
                static_cast<int>(frame.atoms.dihedral_atom(i, 3 * m + 2)));
            flat_dih_t.push_back(frame.atoms.dihedral_type(i, m));
        }
    }

    // Host-cached metadata. Single-GPU defaults for the domain-decomposition
    // counts: all atoms owned, no ghosts, capacity == atom count. A multi-GPU
    // driver overrides n_local_/n_ghost_/capacity_ when it materializes a
    // subdomain frame with halo headroom.
    d.n_atoms_ = n;
    d.n_local_ = n;
    d.n_ghost_ = 0;
    d.capacity_ = cap;
    d.n_bonds_ = static_cast<int>(flat_bond_i.size());
    d.n_angles_ = static_cast<int>(flat_ang_i.size());
    d.n_dihedrals_ = static_cast<int>(flat_dih_i.size());

    // Scalar metadata on device
    cudaMalloc(&d.atoms.d_size, sizeof(int));
    cudaMemcpy(d.atoms.d_size, &n, sizeof(int), cudaMemcpyHostToDevice);

    // Per-atom arrays (real_t storage) — allocated to `cap` for ghost headroom.
    cudaMalloc(&d.atoms.x, cap * sizeof(real_t));
    cudaMalloc(&d.atoms.y, cap * sizeof(real_t));
    cudaMalloc(&d.atoms.z, cap * sizeof(real_t));
    cudaMalloc(&d.atoms.vx, cap * sizeof(real_t));
    cudaMalloc(&d.atoms.vy, cap * sizeof(real_t));
    cudaMalloc(&d.atoms.vz, cap * sizeof(real_t));
    cudaMalloc(&d.atoms.fx, cap * sizeof(real_t));
    cudaMalloc(&d.atoms.fy, cap * sizeof(real_t));
    cudaMalloc(&d.atoms.fz, cap * sizeof(real_t));
    cudaMalloc(&d.atoms.mass, cap * sizeof(real_t));
    cudaMalloc(&d.atoms.charge, cap * sizeof(real_t));
    cudaMalloc(&d.atoms.type, cap * sizeof(int));
    cudaMalloc(&d.atoms.tag, cap * sizeof(std::int64_t));
    cudaMalloc(&d.atoms.image, 3 * cap * sizeof(std::int8_t));

    // System scalars: one device-resident SystemScalars struct
    cudaMalloc(&d.scalars, sizeof(SystemScalars));
    cudaMemcpy(d.scalars, &frame.scalars, sizeof(SystemScalars),
               cudaMemcpyHostToDevice);

    // H2D with narrowing conversion for per-atom arrays
    copy_h2d_narrow(d.atoms.x, frame.atoms.x.data(), n);
    copy_h2d_narrow(d.atoms.y, frame.atoms.y.data(), n);
    copy_h2d_narrow(d.atoms.z, frame.atoms.z.data(), n);
    copy_h2d_narrow(d.atoms.vx, frame.atoms.vx.data(), n);
    copy_h2d_narrow(d.atoms.vy, frame.atoms.vy.data(), n);
    copy_h2d_narrow(d.atoms.vz, frame.atoms.vz.data(), n);
    copy_h2d_narrow(d.atoms.mass, frame.atoms.mass.data(), n);
    copy_h2d_narrow(d.atoms.charge, frame.atoms.charge.data(), n);
    cudaMemcpy(d.atoms.type, frame.atoms.type.data(), n * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d.atoms.tag, frame.atoms.tag.data(), n * sizeof(std::int64_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d.atoms.image, frame.atoms.image.data(),
               3 * n * sizeof(std::int8_t), cudaMemcpyHostToDevice);

    cudaMemset(d.atoms.fx, 0, cap * sizeof(real_t));
    cudaMemset(d.atoms.fy, 0, cap * sizeof(real_t));
    cudaMemset(d.atoms.fz, 0, cap * sizeof(real_t));

    // Box: 3x3 row-major on device (prec_t)
    cudaMalloc(&d.box, 9 * sizeof(prec_t));
    prec_t box_data[9];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            box_data[i * 3 + j] = static_cast<prec_t>(frame.box(i, j));
    cudaMemcpy(d.box, box_data, 9 * sizeof(prec_t), cudaMemcpyHostToDevice);

    // Allocation headroom for the flat bonded arrays. The multi-GPU path
    // rebuilds these per reneighbor via rematerialize_*_k, whose resolved tuple
    // count can exceed this partition-time count by a few near subdomain
    // boundaries (cross-GPU partners resolving against shifting ghosts).
    // Over-allocating is always safe (n_*_ stays the real count); it prevents
    // the rematerialize write from overrunning the array. Single-GPU is
    // unaffected (it reads n_*_, never rewrites these arrays).
    auto bond_cap = [](int c) { return c + c / 8 + 64; };

    // Bond topology — copy from the flattened host buffers built above.
    int nb = d.n_bonds_;
    if (nb > 0) {
        cudaMalloc(&d.bonds.n_bonds, sizeof(int));
        cudaMemcpy(d.bonds.n_bonds, &nb, sizeof(int), cudaMemcpyHostToDevice);
        cudaMalloc(&d.bonds.atomi, bond_cap(nb) * sizeof(int));
        cudaMalloc(&d.bonds.atomj, bond_cap(nb) * sizeof(int));
        cudaMalloc(&d.bonds.type, bond_cap(nb) * sizeof(int));
        cudaMemcpy(d.bonds.atomi, flat_bond_i.data(), nb * sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d.bonds.atomj, flat_bond_j.data(), nb * sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d.bonds.type, flat_bond_t.data(), nb * sizeof(int),
                   cudaMemcpyHostToDevice);
    }

    // Angle topology
    int na = d.n_angles_;
    if (na > 0) {
        cudaMalloc(&d.angles.n_angles, sizeof(int));
        cudaMemcpy(d.angles.n_angles, &na, sizeof(int), cudaMemcpyHostToDevice);
        cudaMalloc(&d.angles.atomi, bond_cap(na) * sizeof(int));
        cudaMalloc(&d.angles.atomj, bond_cap(na) * sizeof(int));
        cudaMalloc(&d.angles.atomk, bond_cap(na) * sizeof(int));
        cudaMalloc(&d.angles.type, bond_cap(na) * sizeof(int));
        cudaMemcpy(d.angles.atomi, flat_ang_i.data(), na * sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d.angles.atomj, flat_ang_j.data(), na * sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d.angles.atomk, flat_ang_k.data(), na * sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d.angles.type, flat_ang_t.data(), na * sizeof(int),
                   cudaMemcpyHostToDevice);
    }

    // Dihedral topology
    int nd = d.n_dihedrals_;
    if (nd > 0) {
        cudaMalloc(&d.dihedrals.n_dihedrals, sizeof(int));
        cudaMemcpy(d.dihedrals.n_dihedrals, &nd, sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaMalloc(&d.dihedrals.atomi, bond_cap(nd) * sizeof(int));
        cudaMalloc(&d.dihedrals.atomj, bond_cap(nd) * sizeof(int));
        cudaMalloc(&d.dihedrals.atomk, bond_cap(nd) * sizeof(int));
        cudaMalloc(&d.dihedrals.atoml, bond_cap(nd) * sizeof(int));
        cudaMalloc(&d.dihedrals.type, bond_cap(nd) * sizeof(int));
        cudaMemcpy(d.dihedrals.atomi, flat_dih_i.data(), nd * sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d.dihedrals.atomj, flat_dih_j.data(), nd * sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d.dihedrals.atomk, flat_dih_k.data(), nd * sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d.dihedrals.atoml, flat_dih_l.data(), nd * sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d.dihedrals.type, flat_dih_t.data(), nd * sizeof(int),
                   cudaMemcpyHostToDevice);
    }

    return d;
}

} // namespace cpu

// -- cuda::to_cpu / cuda::destroy ------------------------------------

namespace cuda {

void to_cpu(const Frame& d, cpu::Frame& frame) {
    int n = frame.atoms.size();

    // D2H with widening conversion for per-atom arrays
    copy_d2h_widen(frame.atoms.x.data(), d.atoms.x, n);
    copy_d2h_widen(frame.atoms.y.data(), d.atoms.y, n);
    copy_d2h_widen(frame.atoms.z.data(), d.atoms.z, n);
    copy_d2h_widen(frame.atoms.vx.data(), d.atoms.vx, n);
    copy_d2h_widen(frame.atoms.vy.data(), d.atoms.vy, n);
    copy_d2h_widen(frame.atoms.vz.data(), d.atoms.vz, n);
    copy_d2h_widen(frame.atoms.fx.data(), d.atoms.fx, n);
    copy_d2h_widen(frame.atoms.fy.data(), d.atoms.fy, n);
    copy_d2h_widen(frame.atoms.fz.data(), d.atoms.fz, n);

    // Global atom tags — round-trip so they survive a device round trip (used
    // by multi-GPU gather/migration to identify atoms after they move between
    // subdomains; previously only copied H2D, leaving stale host tags on D2H).
    if (d.atoms.tag)
        cudaMemcpy(frame.atoms.tag.data(), d.atoms.tag,
                   n * sizeof(std::int64_t), cudaMemcpyDeviceToHost);

    // Per-atom partial charge (mutable: charge-equilibration methods)
    if (d.atoms.charge)
        copy_d2h_widen(frame.atoms.charge.data(), d.atoms.charge, n);

    // Per-atom PBC image counts (mutable: updated as atoms cross box faces)
    if (d.atoms.image)
        cudaMemcpy(frame.atoms.image.data(), d.atoms.image,
                   3 * n * sizeof(std::int8_t), cudaMemcpyDeviceToHost);

    // System scalars: one device-resident SystemScalars struct
    if (d.scalars)
        cudaMemcpy(&frame.scalars, d.scalars, sizeof(SystemScalars),
                   cudaMemcpyDeviceToHost);

    // Box (may have changed via Deform or Barostat)
    if (d.box) {
        prec_t box_data[9];
        cudaMemcpy(box_data, d.box, 9 * sizeof(prec_t), cudaMemcpyDeviceToHost);
        Mat3 m;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                m(i, j) = static_cast<Real>(box_data[i * 3 + j]);
        frame.box.set_cell(m);
    }
}

void materialize_per_atom_bonded(Frame& d, const cpu::Frame& src) {
    const int n = src.atoms.nlocal;
    const int cap = d.capacity_;
    // Adopt the source cpu::Atom's runtime topology strides; downstream
    // kernel launches and migration copies read them back off d.atoms.
    const int MB = src.atoms.bond_per_atom;
    const int MA = src.atoms.angle_per_atom;
    const int MD = src.atoms.dihedral_per_atom;
    const int MS = src.atoms.maxspecial;
    d.atoms.bond_per_atom = MB;
    d.atoms.angle_per_atom = MA;
    d.atoms.dihedral_per_atom = MD;
    d.atoms.maxspecial = MS;
    auto alloc_i = [](int*& p, std::size_t rows) {
        if (rows == 0) {
            p = nullptr; // stride 0 (bonded-free): no wide array to allocate
            return;
        }
        cudaMalloc(&p, rows * sizeof(int));
        cudaMemset(p, 0, rows * sizeof(int));
    };
    auto alloc_t = [](std::int64_t*& p, std::size_t rows) {
        if (rows == 0) {
            p = nullptr;
            return;
        }
        cudaMalloc(&p, rows * sizeof(std::int64_t));
        cudaMemset(p, 0, rows * sizeof(std::int64_t));
    };
    // Allocate to capacity (+ zero) so ghost/headroom rows are inert (num_*=0)
    // until a halo/migrate fills them. Per-atom storage is row-major [cap, cols].
    const std::size_t c = static_cast<std::size_t>(cap);
    alloc_i(d.atoms.num_bond, c);
    alloc_t(d.atoms.bond_atom, c * MB);
    alloc_i(d.atoms.bond_type, c * MB);
    alloc_i(d.atoms.num_angle, c);
    alloc_t(d.atoms.angle_atom, c * 2 * MA);
    alloc_i(d.atoms.angle_type, c * MA);
    alloc_i(d.atoms.num_dihedral, c);
    alloc_t(d.atoms.dihedral_atom, c * 3 * MD);
    alloc_i(d.atoms.dihedral_type, c * MD);
    alloc_i(d.atoms.nspecial, c * 3);
    alloc_t(d.atoms.special, c * MS);
    if (n <= 0)
        return;
    // The source CPU per-atom arrays are row-major [nlocal, cols]; copy the
    // owned rows into the device array's leading [0, n) rows (same column
    // count, so each is one contiguous block).
    const std::size_t un = static_cast<std::size_t>(n);
    auto h2d_i = [](int* dst, const int* s, std::size_t cnt) {
        cudaMemcpy(dst, s, cnt * sizeof(int), cudaMemcpyHostToDevice);
    };
    auto h2d_t = [](std::int64_t* dst, const std::int64_t* s, std::size_t cnt) {
        cudaMemcpy(dst, s, cnt * sizeof(std::int64_t), cudaMemcpyHostToDevice);
    };
    // Count arrays are always host-allocated (zeroed when bonded-free).
    h2d_i(d.atoms.num_bond, src.atoms.num_bond.data(), un);
    h2d_i(d.atoms.num_angle, src.atoms.num_angle.data(), un);
    h2d_i(d.atoms.num_dihedral, src.atoms.num_dihedral.data(), un);
    h2d_i(d.atoms.nspecial, src.atoms.nspecial.data(), un * 3);
    // The wide partner/type/special arrays are host-allocated ONLY for
    // atom_style-molecular systems (cpu::Atom lazy bonded storage). For a
    // bonded-free system they are empty host-side; the device arrays stay
    // allocated-and-zeroed and the counts above are 0, so the GPU never reads
    // them. Copy only when the host actually has the data.
    if (src.atoms.has_bonded()) {
        h2d_t(d.atoms.bond_atom, src.atoms.bond_atom.data(), un * MB);
        h2d_i(d.atoms.bond_type, src.atoms.bond_type.data(), un * MB);
        h2d_t(d.atoms.angle_atom, src.atoms.angle_atom.data(), un * 2 * MA);
        h2d_i(d.atoms.angle_type, src.atoms.angle_type.data(), un * MA);
        h2d_t(d.atoms.dihedral_atom, src.atoms.dihedral_atom.data(),
              un * 3 * MD);
        h2d_i(d.atoms.dihedral_type, src.atoms.dihedral_type.data(), un * MD);
        h2d_t(d.atoms.special, src.atoms.special.data(), un * MS);
    }
}

void destroy(Frame& d) {
    cudaFree(d.atoms.x);
    cudaFree(d.atoms.y);
    cudaFree(d.atoms.z);
    cudaFree(d.atoms.vx);
    cudaFree(d.atoms.vy);
    cudaFree(d.atoms.vz);
    cudaFree(d.atoms.fx);
    cudaFree(d.atoms.fy);
    cudaFree(d.atoms.fz);
    cudaFree(d.atoms.mass);
    cudaFree(d.atoms.charge);
    cudaFree(d.atoms.type);
    cudaFree(d.atoms.tag);
    cudaFree(d.atoms.image);
    cudaFree(d.atoms.d_size);
    cudaFree(d.scalars);
    cudaFree(d.box);

    cudaFree(d.bonds.atomi);
    cudaFree(d.bonds.atomj);
    cudaFree(d.bonds.type);
    cudaFree(d.bonds.n_bonds);

    cudaFree(d.angles.atomi);
    cudaFree(d.angles.atomj);
    cudaFree(d.angles.atomk);
    cudaFree(d.angles.type);
    cudaFree(d.angles.n_angles);

    cudaFree(d.dihedrals.atomi);
    cudaFree(d.dihedrals.atomj);
    cudaFree(d.dihedrals.atomk);
    cudaFree(d.dihedrals.atoml);
    cudaFree(d.dihedrals.type);
    cudaFree(d.dihedrals.n_dihedrals);

    // Multi-GPU per-atom bonded SoA (null on single-GPU frames; cudaFree(0) ok).
    cudaFree(d.atoms.num_bond);
    cudaFree(d.atoms.bond_atom);
    cudaFree(d.atoms.bond_type);
    cudaFree(d.atoms.num_angle);
    cudaFree(d.atoms.angle_atom);
    cudaFree(d.atoms.angle_type);
    cudaFree(d.atoms.num_dihedral);
    cudaFree(d.atoms.dihedral_atom);
    cudaFree(d.atoms.dihedral_type);
    cudaFree(d.atoms.nspecial);
    cudaFree(d.atoms.special);

    d = {};
}

} // namespace cuda
} // namespace ATV
