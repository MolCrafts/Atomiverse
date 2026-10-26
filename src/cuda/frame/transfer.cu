/// @file cuda/frame/transfer.cu
/// @brief cpu::to_cuda(), cuda::to_cpu(), cuda::destroy() implementations.
///
/// Implements host<->device conversion through the CUDA precision contract
/// (cuda/frame/types.cuh): coordinates are split into primary + correction,
/// velocities and masses stay in integration precision, charges in force
/// precision, and forces are encoded as 64-bit fixed point. The host cpu::Frame is
/// blocks+meta+box: per-atom columns are read via span(kAtom_*), the
/// per-frame scalars are gathered from / scattered to typed meta through the
/// device-resident SystemScalars bundle, and the cell is frame.box.

// NOTE (topology representation): CUDA kernels have not been ported to
// per-atom topology — the single-GPU path consumes the flat
// BondList/AngleList/DihedralList, so cpu::to_cuda re-flattens from the
// CPU per-atom layout on every transfer. The multi-GPU path additionally
// carries the per-atom SoA for migration. See cuda/frame/frame.cuh.

#include "atomiverse/cpu/frame/transfer.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/frame/precision.cuh"
#include "atomiverse/cuda/frame/transfer.cuh"
#include "atomiverse/cuda/frame/types.cuh"

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/system_scalars.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <cuda_runtime.h>
#include <vector>

namespace ATV {

using cuda::fixed_t;
using cuda::geom_t;
using cuda::pos_t;

// -- Conversion helpers -----------------------------------------------
//
// cpu::Frame stores per-atom positions/velocities as StorageReal and forces as
// AccReal (double). Each device column has its contract type; a column whose
// type differs from the host tier converts through a temporary buffer.

namespace {

/// @brief Copy a host column (tier H) to a device column of type D.
template <typename D, typename H>
void copy_h2d(D* d_dst, const H* h_src, int n) {
    if constexpr (std::is_same_v<D, H>) {
        ATV_CUDA_CHECK(
            cudaMemcpy(d_dst, h_src, n * sizeof(D), cudaMemcpyHostToDevice));
    } else {
        std::vector<D> tmp(n);
        for (int i = 0; i < n; ++i)
            tmp[i] = static_cast<D>(h_src[i]);
        ATV_CUDA_CHECK(cudaMemcpy(d_dst, tmp.data(), n * sizeof(D),
                                  cudaMemcpyHostToDevice));
    }
}

/// @brief Copy a device column of type D to a host column (tier H).
template <typename H, typename D>
void copy_d2h(H* h_dst, const D* d_src, int n) {
    if constexpr (std::is_same_v<D, H>) {
        ATV_CUDA_CHECK(
            cudaMemcpy(h_dst, d_src, n * sizeof(D), cudaMemcpyDeviceToHost));
    } else {
        std::vector<D> tmp(n);
        ATV_CUDA_CHECK(cudaMemcpy(tmp.data(), d_src, n * sizeof(D),
                                  cudaMemcpyDeviceToHost));
        for (int i = 0; i < n; ++i)
            h_dst[i] = static_cast<H>(tmp[i]);
    }
}

/// @brief Split a host coordinate column into device primary + correction.
template <typename H>
void upload_position(pos_t* d_primary, pos_t* d_correction, const H* h_src,
                     int n) {
    std::vector<pos_t> hi(n), lo(n, pos_t(0));
    for (int i = 0; i < n; ++i)
        cuda::split_position(static_cast<cuda::mixed_t>(h_src[i]), hi[i],
                             lo[i]);
    ATV_CUDA_CHECK(cudaMemcpy(d_primary, hi.data(), n * sizeof(pos_t),
                              cudaMemcpyHostToDevice));
    if constexpr (cuda::kPositionCorrection)
        ATV_CUDA_CHECK(cudaMemcpy(d_correction, lo.data(), n * sizeof(pos_t),
                                  cudaMemcpyHostToDevice));
}

/// @brief Reconstruct a device coordinate column on the host.
template <typename H>
void download_position(H* h_dst, const pos_t* d_primary,
                       const pos_t* d_correction, int n) {
    std::vector<pos_t> hi(n), lo(n, pos_t(0));
    ATV_CUDA_CHECK(cudaMemcpy(hi.data(), d_primary, n * sizeof(pos_t),
                              cudaMemcpyDeviceToHost));
    if constexpr (cuda::kPositionCorrection)
        ATV_CUDA_CHECK(cudaMemcpy(lo.data(), d_correction, n * sizeof(pos_t),
                                  cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i)
        h_dst[i] = static_cast<H>(cuda::load_position(hi.data(), lo.data(), i));
}

/// @brief Decode a device fixed-point force column on the host.
template <typename H>
void download_force(H* h_dst, const fixed_t* d_src, int n) {
    std::vector<fixed_t> tmp(n);
    ATV_CUDA_CHECK(cudaMemcpy(tmp.data(), d_src, n * sizeof(fixed_t),
                              cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i)
        h_dst[i] = static_cast<H>(cuda::decode_force<double>(tmp[i]));
}

/// @brief Read a serializable double meta value, or 0 when absent.
double meta_or_zero(const cpu::Frame& f, const MetaKey<double>& k) {
    return f.meta.contains(k.key) ? f.meta.get(k) : 0.0;
}

/// @brief Write a double meta value, inserting the key if it does not exist.
void meta_put(cpu::Frame& f, const MetaKey<double>& k, double v) {
    if (f.meta.contains(k.key))
        f.meta.set(k, v);
    else
        f.meta.insert(k, v);
}

/// @brief Read an int topology stride from meta, or 0 when absent.
int stride_or_zero(const cpu::Frame& f, const MetaKey<std::int64_t>& k) {
    return f.meta.contains(k.key) ? static_cast<int>(f.meta.get(k)) : 0;
}

/// @brief Gather the host frame's per-frame scalars into a SystemScalars.
SystemScalars gather_scalars(const cpu::Frame& f) {
    SystemScalars sc{};
    sc.energy = meta_or_zero(f, kEnergy);
    sc.energy_pair = meta_or_zero(f, kEnergyPair);
    sc.energy_bond = meta_or_zero(f, kEnergyBond);
    sc.energy_angle = meta_or_zero(f, kEnergyAngle);
    sc.energy_dihedral = meta_or_zero(f, kEnergyDihedral);
    sc.energy_improper = meta_or_zero(f, kEnergyImproper);
    sc.kinetic = meta_or_zero(f, kKinetic);
    sc.conserved = meta_or_zero(f, kConserved);
    if (f.meta.contains(kStress.key)) {
        const auto& s = f.meta.get(kStress);
        for (int i = 0; i < 6; ++i)
            sc.stress[i] = s[i];
    }
    if (f.meta.contains(kStressKinetic.key)) {
        const auto& s = f.meta.get(kStressKinetic);
        for (int i = 0; i < 6; ++i)
            sc.stress_kinetic[i] = s[i];
    }
    return sc;
}

/// @brief Scatter a SystemScalars back into the host frame's typed meta.
void scatter_scalars(cpu::Frame& f, const SystemScalars& sc) {
    meta_put(f, kEnergy, sc.energy);
    meta_put(f, kEnergyPair, sc.energy_pair);
    meta_put(f, kEnergyBond, sc.energy_bond);
    meta_put(f, kEnergyAngle, sc.energy_angle);
    meta_put(f, kEnergyDihedral, sc.energy_dihedral);
    meta_put(f, kEnergyImproper, sc.energy_improper);
    meta_put(f, kKinetic, sc.kinetic);
    meta_put(f, kConserved, sc.conserved);
    MetaF64x6 stress{};
    MetaF64x6 stress_kin{};
    for (int i = 0; i < 6; ++i) {
        stress[i] = sc.stress[i];
        stress_kin[i] = sc.stress_kinetic[i];
    }
    if (f.meta.contains(kStress.key))
        f.meta.set(kStress, stress);
    else
        f.meta.insert(kStress, stress);
    if (f.meta.contains(kStressKinetic.key))
        f.meta.set(kStressKinetic, stress_kin);
    else
        f.meta.insert(kStressKinetic, stress_kin);
}

} // anonymous namespace

// -- cpu::to_cuda ----------------------------------------------------

namespace cpu {

cuda::Frame to_cuda(const Frame& frame, int capacity) {
    int n = frame.nlocal();
    // Per-atom arrays are allocated to `cap` (>= n) so multi-GPU halo ghosts
    // can be appended in [n_local, cap) without reallocation; only n values are
    // copied. cap == n for the default single-GPU path.
    int cap = (capacity > n) ? capacity : n;
    cuda::Frame d{};
    struct AllocationGuard {
        cuda::Frame& frame;
        bool complete = false;
        ~AllocationGuard() {
            if (!complete)
                cuda::destroy(frame);
        }
    } allocation{d};

    // Flatten per-atom topology (LAMMPS num_bond[i] / bond_atom[i][m] layout in
    // the atoms Block) into flat (atomi[k] / atomj[k]) host buffers for the
    // CUDA-side flat `BondList`. CUDA kernels haven't been ported to per-atom
    // topology yet (out-of-scope for this refactor); the bridge does the
    // conversion every transfer. Each per-atom bond appears as one flat row:
    // (atom-i, partner_tag). Same for angles (i, j, k) and dihedrals
    // (i, j, k, l). Tags here are global atom IDs (= local-row index in
    // single-rank, since materialize sets tag[row] = row). A bonded-free frame
    // has no topology columns; the flat buffers stay empty.
    std::vector<int> flat_bond_i, flat_bond_j, flat_bond_t;
    std::vector<int> flat_ang_i, flat_ang_j, flat_ang_k, flat_ang_t;
    std::vector<int> flat_dih_i, flat_dih_j, flat_dih_k, flat_dih_l, flat_dih_t;
    if (frame.has_bonded()) {
        const int MB = stride_or_zero(frame, kBondPerAtom);
        const int MA = stride_or_zero(frame, kAnglePerAtom);
        const int MD = stride_or_zero(frame, kDihedralPerAtom);
        const auto num_bond = frame.span(kAtom_num_bond);
        const auto bond_atom = frame.span(kAtom_bond_atom);
        const auto bond_type = frame.span(kAtom_bond_type);
        const auto num_angle = frame.span(kAtom_num_angle);
        const auto angle_atom = frame.span(kAtom_angle_atom);
        const auto angle_type = frame.span(kAtom_angle_type);
        const auto num_dihedral = frame.span(kAtom_num_dihedral);
        const auto dihedral_atom = frame.span(kAtom_dihedral_atom);
        const auto dihedral_type = frame.span(kAtom_dihedral_type);
        for (int i = 0; i < n; ++i) {
            const int nb = num_bond[i];
            for (int m = 0; m < nb; ++m) {
                flat_bond_i.push_back(i);
                flat_bond_j.push_back(static_cast<int>(bond_atom[i * MB + m]));
                flat_bond_t.push_back(bond_type[i * MB + m]);
            }
            const int na = num_angle[i];
            for (int m = 0; m < na; ++m) {
                flat_ang_i.push_back(
                    static_cast<int>(angle_atom[i * 2 * MA + 2 * m + 0]));
                flat_ang_j.push_back(i);
                flat_ang_k.push_back(
                    static_cast<int>(angle_atom[i * 2 * MA + 2 * m + 1]));
                flat_ang_t.push_back(angle_type[i * MA + m]);
            }
            const int nd = num_dihedral[i];
            for (int m = 0; m < nd; ++m) {
                flat_dih_i.push_back(
                    static_cast<int>(dihedral_atom[i * 3 * MD + 3 * m + 0]));
                flat_dih_j.push_back(i);
                flat_dih_k.push_back(
                    static_cast<int>(dihedral_atom[i * 3 * MD + 3 * m + 1]));
                flat_dih_l.push_back(
                    static_cast<int>(dihedral_atom[i * 3 * MD + 3 * m + 2]));
                flat_dih_t.push_back(dihedral_type[i * MD + m]);
            }
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
    ATV_CUDA_CHECK(cudaMalloc(&d.atoms.d_size, sizeof(int)));
    ATV_CUDA_CHECK(
        cudaMemcpy(d.atoms.d_size, &n, sizeof(int), cudaMemcpyHostToDevice));

    // Per-atom arrays (contract types) — allocated to `cap` for ghost headroom.
    ATV_CUDA_CHECK(cudaMalloc(&d.atoms.x, cap * sizeof(pos_t)));
    ATV_CUDA_CHECK(cudaMalloc(&d.atoms.y, cap * sizeof(pos_t)));
    ATV_CUDA_CHECK(cudaMalloc(&d.atoms.z, cap * sizeof(pos_t)));
    if constexpr (cuda::kPositionCorrection) {
        ATV_CUDA_CHECK(cudaMalloc(&d.atoms.x_corr, cap * sizeof(pos_t)));
        ATV_CUDA_CHECK(cudaMalloc(&d.atoms.y_corr, cap * sizeof(pos_t)));
        ATV_CUDA_CHECK(cudaMalloc(&d.atoms.z_corr, cap * sizeof(pos_t)));
        // Ghost headroom rows carry no correction until an owner moves in.
        ATV_CUDA_CHECK(cudaMemset(d.atoms.x_corr, 0, cap * sizeof(pos_t)));
        ATV_CUDA_CHECK(cudaMemset(d.atoms.y_corr, 0, cap * sizeof(pos_t)));
        ATV_CUDA_CHECK(cudaMemset(d.atoms.z_corr, 0, cap * sizeof(pos_t)));
    }
    ATV_CUDA_CHECK(cudaMalloc(&d.atoms.vx, cap * sizeof(cuda::mixed_t)));
    ATV_CUDA_CHECK(cudaMalloc(&d.atoms.vy, cap * sizeof(cuda::mixed_t)));
    ATV_CUDA_CHECK(cudaMalloc(&d.atoms.vz, cap * sizeof(cuda::mixed_t)));
    ATV_CUDA_CHECK(cudaMalloc(&d.atoms.fx, cap * sizeof(fixed_t)));
    ATV_CUDA_CHECK(cudaMalloc(&d.atoms.fy, cap * sizeof(fixed_t)));
    ATV_CUDA_CHECK(cudaMalloc(&d.atoms.fz, cap * sizeof(fixed_t)));
    ATV_CUDA_CHECK(cudaMalloc(&d.atoms.mass, cap * sizeof(cuda::mixed_t)));
    ATV_CUDA_CHECK(cudaMalloc(&d.atoms.charge, cap * sizeof(cuda::force_t)));
    ATV_CUDA_CHECK(cudaMalloc(&d.atoms.type, cap * sizeof(int)));
    ATV_CUDA_CHECK(cudaMalloc(&d.atoms.tag, cap * sizeof(std::int64_t)));
    ATV_CUDA_CHECK(cudaMalloc(&d.atoms.image, 3 * cap * sizeof(std::int8_t)));

    // System scalars: gather the host frame's typed meta into one
    // device-resident SystemScalars struct.
    SystemScalars host_scalars = gather_scalars(frame);
    ATV_CUDA_CHECK(cudaMalloc(&d.scalars, sizeof(SystemScalars)));
    ATV_CUDA_CHECK(cudaMemcpy(d.scalars, &host_scalars, sizeof(SystemScalars),
                              cudaMemcpyHostToDevice));
    const cuda::ForceFault clean_fault{};
    ATV_CUDA_CHECK(cudaMalloc(&d.force_fault, sizeof(cuda::ForceFault)));
    ATV_CUDA_CHECK(cudaMemcpy(d.force_fault, &clean_fault,
                              sizeof(cuda::ForceFault),
                              cudaMemcpyHostToDevice));

    upload_position(d.atoms.x, d.atoms.x_corr, frame.span(kAtom_x).data(), n);
    upload_position(d.atoms.y, d.atoms.y_corr, frame.span(kAtom_y).data(), n);
    upload_position(d.atoms.z, d.atoms.z_corr, frame.span(kAtom_z).data(), n);
    copy_h2d(d.atoms.vx, frame.span(kAtom_vx).data(), n);
    copy_h2d(d.atoms.vy, frame.span(kAtom_vy).data(), n);
    copy_h2d(d.atoms.vz, frame.span(kAtom_vz).data(), n);
    copy_h2d(d.atoms.mass, frame.span(kAtom_mass).data(), n);
    copy_h2d(d.atoms.charge, frame.span(kAtom_charge).data(), n);
    ATV_CUDA_CHECK(cudaMemcpy(d.atoms.type, frame.span(kAtom_type).data(),
                              n * sizeof(int), cudaMemcpyHostToDevice));
    ATV_CUDA_CHECK(cudaMemcpy(d.atoms.tag, frame.span(kAtom_tag).data(),
                              n * sizeof(std::int64_t),
                              cudaMemcpyHostToDevice));
    ATV_CUDA_CHECK(cudaMemcpy(d.atoms.image, frame.span(kAtom_image).data(),
                              3 * n * sizeof(std::int8_t),
                              cudaMemcpyHostToDevice));

    ATV_CUDA_CHECK(cudaMemset(d.atoms.fx, 0, cap * sizeof(fixed_t)));
    ATV_CUDA_CHECK(cudaMemset(d.atoms.fy, 0, cap * sizeof(fixed_t)));
    ATV_CUDA_CHECK(cudaMemset(d.atoms.fz, 0, cap * sizeof(fixed_t)));

    // Box: 3x3 row-major on device (geometry precision)
    ATV_CUDA_CHECK(cudaMalloc(&d.box, 9 * sizeof(geom_t)));
    geom_t box_data[9];
    const Mat3& cell = frame.box.cell();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            box_data[i * 3 + j] = static_cast<geom_t>(cell(i, j));
    ATV_CUDA_CHECK(cudaMemcpy(d.box, box_data, 9 * sizeof(geom_t),
                              cudaMemcpyHostToDevice));

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
        ATV_CUDA_CHECK(cudaMalloc(&d.bonds.n_bonds, sizeof(int)));
        ATV_CUDA_CHECK(cudaMemcpy(d.bonds.n_bonds, &nb, sizeof(int),
                                  cudaMemcpyHostToDevice));
        ATV_CUDA_CHECK(cudaMalloc(&d.bonds.atomi, bond_cap(nb) * sizeof(int)));
        ATV_CUDA_CHECK(cudaMalloc(&d.bonds.atomj, bond_cap(nb) * sizeof(int)));
        ATV_CUDA_CHECK(cudaMalloc(&d.bonds.type, bond_cap(nb) * sizeof(int)));
        ATV_CUDA_CHECK(cudaMemcpy(d.bonds.atomi, flat_bond_i.data(),
                                  nb * sizeof(int), cudaMemcpyHostToDevice));
        ATV_CUDA_CHECK(cudaMemcpy(d.bonds.atomj, flat_bond_j.data(),
                                  nb * sizeof(int), cudaMemcpyHostToDevice));
        ATV_CUDA_CHECK(cudaMemcpy(d.bonds.type, flat_bond_t.data(),
                                  nb * sizeof(int), cudaMemcpyHostToDevice));
    }

    // Angle topology
    int na = d.n_angles_;
    if (na > 0) {
        ATV_CUDA_CHECK(cudaMalloc(&d.angles.n_angles, sizeof(int)));
        ATV_CUDA_CHECK(cudaMemcpy(d.angles.n_angles, &na, sizeof(int),
                                  cudaMemcpyHostToDevice));
        ATV_CUDA_CHECK(cudaMalloc(&d.angles.atomi, bond_cap(na) * sizeof(int)));
        ATV_CUDA_CHECK(cudaMalloc(&d.angles.atomj, bond_cap(na) * sizeof(int)));
        ATV_CUDA_CHECK(cudaMalloc(&d.angles.atomk, bond_cap(na) * sizeof(int)));
        ATV_CUDA_CHECK(cudaMalloc(&d.angles.type, bond_cap(na) * sizeof(int)));
        ATV_CUDA_CHECK(cudaMemcpy(d.angles.atomi, flat_ang_i.data(),
                                  na * sizeof(int), cudaMemcpyHostToDevice));
        ATV_CUDA_CHECK(cudaMemcpy(d.angles.atomj, flat_ang_j.data(),
                                  na * sizeof(int), cudaMemcpyHostToDevice));
        ATV_CUDA_CHECK(cudaMemcpy(d.angles.atomk, flat_ang_k.data(),
                                  na * sizeof(int), cudaMemcpyHostToDevice));
        ATV_CUDA_CHECK(cudaMemcpy(d.angles.type, flat_ang_t.data(),
                                  na * sizeof(int), cudaMemcpyHostToDevice));
    }

    // Dihedral topology
    int nd = d.n_dihedrals_;
    if (nd > 0) {
        ATV_CUDA_CHECK(cudaMalloc(&d.dihedrals.n_dihedrals, sizeof(int)));
        ATV_CUDA_CHECK(cudaMemcpy(d.dihedrals.n_dihedrals, &nd, sizeof(int),
                                  cudaMemcpyHostToDevice));
        ATV_CUDA_CHECK(
            cudaMalloc(&d.dihedrals.atomi, bond_cap(nd) * sizeof(int)));
        ATV_CUDA_CHECK(
            cudaMalloc(&d.dihedrals.atomj, bond_cap(nd) * sizeof(int)));
        ATV_CUDA_CHECK(
            cudaMalloc(&d.dihedrals.atomk, bond_cap(nd) * sizeof(int)));
        ATV_CUDA_CHECK(
            cudaMalloc(&d.dihedrals.atoml, bond_cap(nd) * sizeof(int)));
        ATV_CUDA_CHECK(
            cudaMalloc(&d.dihedrals.type, bond_cap(nd) * sizeof(int)));
        ATV_CUDA_CHECK(cudaMemcpy(d.dihedrals.atomi, flat_dih_i.data(),
                                  nd * sizeof(int), cudaMemcpyHostToDevice));
        ATV_CUDA_CHECK(cudaMemcpy(d.dihedrals.atomj, flat_dih_j.data(),
                                  nd * sizeof(int), cudaMemcpyHostToDevice));
        ATV_CUDA_CHECK(cudaMemcpy(d.dihedrals.atomk, flat_dih_k.data(),
                                  nd * sizeof(int), cudaMemcpyHostToDevice));
        ATV_CUDA_CHECK(cudaMemcpy(d.dihedrals.atoml, flat_dih_l.data(),
                                  nd * sizeof(int), cudaMemcpyHostToDevice));
        ATV_CUDA_CHECK(cudaMemcpy(d.dihedrals.type, flat_dih_t.data(),
                                  nd * sizeof(int), cudaMemcpyHostToDevice));
    }

    allocation.complete = true;
    return d;
}

} // namespace cpu

// -- cuda::to_cpu / cuda::destroy ------------------------------------

namespace cuda {

namespace {
__global__ void store_forces_k(fixed_t* fx, fixed_t* fy, fixed_t* fz,
                               ForceFault* fault, const double* gx,
                               const double* gy, const double* gz, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    store_force(fx, i, gx[i], fault);
    store_force(fy, i, gy[i], fault);
    store_force(fz, i, gz[i], fault);
}
} // namespace

void store_forces(Frame& frame, const double* fx, const double* fy,
                  const double* fz, int n, void* stream) {
    if (n <= 0)
        return;
    constexpr int block = 256;
    store_forces_k<<<(n + block - 1) / block, block, 0,
                     static_cast<cudaStream_t>(stream)>>>(
        frame.atoms.fx, frame.atoms.fy, frame.atoms.fz, frame.force_fault, fx,
        fy, fz, n);
    ATV_CUDA_CHECK(cudaGetLastError());
}

void throw_if_force_fault(const Frame& d) {
    if (d.force_fault == nullptr)
        return;
    ForceFault fault{};
    ATV_CUDA_CHECK(cudaMemcpy(&fault, d.force_fault, sizeof(ForceFault),
                              cudaMemcpyDeviceToHost));
    if (fault.kinds == 0)
        return;
    long long tag = fault.tag;
    if (tag < 0 && fault.row >= 0 && d.atoms.tag != nullptr) {
        std::int64_t row_tag = -1;
        ATV_CUDA_CHECK(cudaMemcpy(&row_tag, d.atoms.tag + fault.row,
                                  sizeof(std::int64_t),
                                  cudaMemcpyDeviceToHost));
        tag = row_tag;
    }
    std::string what;
    if (fault.kinds & ForceFault::kConversion)
        what = "a force is NaN/Inf or exceeds the fixed-point range";
    if (fault.kinds & ForceFault::kAccumulation)
        what += std::string(what.empty() ? "" : "; ")
                + "a fixed-point force sum wrapped";
    throw std::runtime_error(
        "CUDA force fault: " + what + " (|F| must stay below "
        + std::to_string(kFixedLimit * kForceFixedResolution)
        + " eV/Angstrom); atom tag " + std::to_string(tag)
        + (fault.step >= 0
               ? ", integration refused at step " + std::to_string(fault.step)
               : std::string(", detected after force evaluation"))
        + ". The device state is invalid; check for overlapping atoms or an "
          "unstable time step.");
}

void update_inputs(const cpu::Frame& src, Frame& dst, InputChange changes) {
    if (has_input_change(changes, InputChange::Identity)
        || src.nlocal() != dst.n_atoms())
        throw std::invalid_argument(
            "CUDA input update requires unchanged identity");
    const int n = dst.n_atoms();
    if (has_input_change(changes, InputChange::Geometry)) {
        upload_position(dst.atoms.x, dst.atoms.x_corr, src.span(kAtom_x).data(),
                        n);
        upload_position(dst.atoms.y, dst.atoms.y_corr, src.span(kAtom_y).data(),
                        n);
        upload_position(dst.atoms.z, dst.atoms.z_corr, src.span(kAtom_z).data(),
                        n);
        ATV_CUDA_CHECK(cudaMemcpy(dst.atoms.image, src.span(kAtom_image).data(),
                                  3 * n * sizeof(std::int8_t),
                                  cudaMemcpyHostToDevice));
    }
    if (has_input_change(changes, InputChange::Velocity)) {
        copy_h2d(dst.atoms.vx, src.span(kAtom_vx).data(), n);
        copy_h2d(dst.atoms.vy, src.span(kAtom_vy).data(), n);
        copy_h2d(dst.atoms.vz, src.span(kAtom_vz).data(), n);
    }
    if (has_input_change(changes, InputChange::Charge))
        copy_h2d(dst.atoms.charge, src.span(kAtom_charge).data(), n);
    if (has_input_change(changes, InputChange::Cell)) {
        geom_t cell[9];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                cell[3 * i + j] = src.box.cell()(i, j);
        ATV_CUDA_CHECK(
            cudaMemcpy(dst.box, cell, sizeof(cell), cudaMemcpyHostToDevice));
    }
}

void to_cpu(const Frame& d, cpu::Frame& frame, bool copy_forces) {
    throw_if_force_fault(d);
    int n = frame.nlocal();

    download_position(frame.span(kAtom_x).data(), d.atoms.x, d.atoms.x_corr, n);
    download_position(frame.span(kAtom_y).data(), d.atoms.y, d.atoms.y_corr, n);
    download_position(frame.span(kAtom_z).data(), d.atoms.z, d.atoms.z_corr, n);
    copy_d2h(frame.span(kAtom_vx).data(), d.atoms.vx, n);
    copy_d2h(frame.span(kAtom_vy).data(), d.atoms.vy, n);
    copy_d2h(frame.span(kAtom_vz).data(), d.atoms.vz, n);
    if (copy_forces) {
        download_force(frame.span(kAtom_fx).data(), d.atoms.fx, n);
        download_force(frame.span(kAtom_fy).data(), d.atoms.fy, n);
        download_force(frame.span(kAtom_fz).data(), d.atoms.fz, n);
    }

    // Global atom tags — round-trip so they survive a device round trip (used
    // by multi-GPU gather/migration to identify atoms after they move between
    // subdomains; previously only copied H2D, leaving stale host tags on D2H).
    if (d.atoms.tag)
        ATV_CUDA_CHECK(cudaMemcpy(frame.span(kAtom_tag).data(), d.atoms.tag,
                                  n * sizeof(std::int64_t),
                                  cudaMemcpyDeviceToHost));

    // Per-atom partial charge (mutable: charge-equilibration methods)
    if (d.atoms.charge)
        copy_d2h(frame.span(kAtom_charge).data(), d.atoms.charge, n);

    // Per-atom PBC image counts (mutable: updated as atoms cross box faces)
    if (d.atoms.image)
        ATV_CUDA_CHECK(cudaMemcpy(frame.span(kAtom_image).data(), d.atoms.image,
                                  3 * n * sizeof(std::int8_t),
                                  cudaMemcpyDeviceToHost));

    // System scalars: read the device SystemScalars back and scatter into the
    // host frame's typed meta.
    if (d.scalars) {
        SystemScalars host_scalars{};
        ATV_CUDA_CHECK(cudaMemcpy(&host_scalars, d.scalars,
                                  sizeof(SystemScalars),
                                  cudaMemcpyDeviceToHost));
        scatter_scalars(frame, host_scalars);
    }

    // Box (may have changed via Deform or Barostat)
    if (d.box) {
        geom_t box_data[9];
        ATV_CUDA_CHECK(cudaMemcpy(box_data, d.box, 9 * sizeof(geom_t),
                                  cudaMemcpyDeviceToHost));
        Mat3 m;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                m(i, j) = static_cast<AccReal>(box_data[i * 3 + j]);
        frame.box.set_cell(m);
    }
}

void materialize_per_atom_bonded(Frame& d, const cpu::Frame& src) {
    const int n = src.nlocal();
    const int cap = d.capacity_;
    // Adopt the source frame's runtime topology strides (from typed meta);
    // downstream kernel launches and migration copies read them back off
    // d.atoms. Absent strides (bonded-free frame) default to 0.
    const int MB = stride_or_zero(src, kBondPerAtom);
    const int MA = stride_or_zero(src, kAnglePerAtom);
    const int MD = stride_or_zero(src, kDihedralPerAtom);
    const int MS = stride_or_zero(src, kMaxSpecial);
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
    if (n <= 0 || !src.has_bonded())
        return;
    // The source per-atom columns are row-major [nlocal, cols]; copy the owned
    // rows into the device array's leading [0, n) rows (same column count, so
    // each is one contiguous block). For a bonded-free frame the topology
    // columns do not exist and the device arrays stay allocated-and-zeroed
    // (num_*=0), so the GPU never reads them — handled by the guard above.
    const std::size_t un = static_cast<std::size_t>(n);
    auto h2d_i = [](int* dst, const int* s, std::size_t cnt) {
        cudaMemcpy(dst, s, cnt * sizeof(int), cudaMemcpyHostToDevice);
    };
    auto h2d_t = [](std::int64_t* dst, const std::int64_t* s, std::size_t cnt) {
        cudaMemcpy(dst, s, cnt * sizeof(std::int64_t), cudaMemcpyHostToDevice);
    };
    h2d_i(d.atoms.num_bond, src.span(kAtom_num_bond).data(), un);
    h2d_i(d.atoms.num_angle, src.span(kAtom_num_angle).data(), un);
    h2d_i(d.atoms.num_dihedral, src.span(kAtom_num_dihedral).data(), un);
    h2d_i(d.atoms.nspecial, src.span(kAtom_nspecial).data(), un * 3);
    h2d_t(d.atoms.bond_atom, src.span(kAtom_bond_atom).data(), un * MB);
    h2d_i(d.atoms.bond_type, src.span(kAtom_bond_type).data(), un * MB);
    h2d_t(d.atoms.angle_atom, src.span(kAtom_angle_atom).data(), un * 2 * MA);
    h2d_i(d.atoms.angle_type, src.span(kAtom_angle_type).data(), un * MA);
    h2d_t(d.atoms.dihedral_atom, src.span(kAtom_dihedral_atom).data(),
          un * 3 * MD);
    h2d_i(d.atoms.dihedral_type, src.span(kAtom_dihedral_type).data(), un * MD);
    h2d_t(d.atoms.special, src.span(kAtom_special).data(), un * MS);
}

void destroy(Frame& d) {
    cudaFree(d.atoms.x);
    cudaFree(d.atoms.y);
    cudaFree(d.atoms.z);
    cudaFree(d.atoms.x_corr);
    cudaFree(d.atoms.y_corr);
    cudaFree(d.atoms.z_corr);
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
    cudaFree(d.force_fault);
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
