/// @file cuda/communication/spatial_grid.cu
/// @brief SpatialGrid implementation (pure host topology).

#include "atomiverse/cuda/communication/spatial_grid.cuh"

#include <cmath>
#include <stdexcept>

namespace ATV {
namespace cuda {

namespace {

// Balanced 3D factorization of n into dims[0]*dims[1]*dims[2], minimizing the
// surface-to-volume proxy (sum of dims) so subdomains are as cubic as possible
// — the single-process analog of MPI_Dims_create. Optionally biases the split
// toward longer box axes by weighting with edge lengths.
void factor3d(int n, const double L[3], int dims[3]) {
    dims[0] = dims[1] = dims[2] = 1;
    double best = 1e300;
    // Iterate factors DESCENDING so that, among equal-surface splits (e.g. a
    // cubic box where the axis is arbitrary), the first found — and thus chosen
    // by the strict `<` — is the x-heavy, non-increasing one (dims[0] >=
    // dims[1] >= dims[2]), matching MPI_Dims_create / LAMMPS convention.
    for (int a = n; a >= 1; --a) {
        if (n % a)
            continue;
        int na = n / a;
        for (int b = na; b >= 1; --b) {
            if (na % b)
                continue;
            int c = na / b;
            // Subdomain edge lengths under this split.
            double ex = L[0] / a, ey = L[1] / b, ez = L[2] / c;
            // Surface area of one subdomain (∝ communication volume); pick the
            // split that minimizes it → most cubic cells along the real box.
            double surf = ex * ey + ey * ez + ez * ex;
            if (surf < best - 1e-12) {
                best = surf;
                dims[0] = a;
                dims[1] = b;
                dims[2] = c;
            }
        }
    }
}

} // namespace

SpatialGrid::SpatialGrid(int n_gpus, double Lx, double Ly, double Lz) {
    if (n_gpus < 1)
        throw std::invalid_argument("SpatialGrid: n_gpus < 1");
    if (Lx <= 0 || Ly <= 0 || Lz <= 0)
        throw std::invalid_argument("SpatialGrid: non-positive box");
    n_ = n_gpus;
    L_[0] = Lx;
    L_[1] = Ly;
    L_[2] = Lz;
    factor3d(n_, L_, dims_);
}

int SpatialGrid::slot_of(int cx, int cy, int cz) const {
    return (cz * dims_[1] + cy) * dims_[0] + cx; // x-fastest
}

std::array<int, 3> SpatialGrid::coords(int slot) const {
    int cx = slot % dims_[0];
    int cy = (slot / dims_[0]) % dims_[1];
    int cz = slot / (dims_[0] * dims_[1]);
    return {cx, cy, cz};
}

std::array<double, 3> SpatialGrid::sublo(int slot) const {
    auto c = coords(slot);
    return {L_[0] * c[0] / dims_[0], L_[1] * c[1] / dims_[1],
            L_[2] * c[2] / dims_[2]};
}

std::array<double, 3> SpatialGrid::subhi(int slot) const {
    auto c = coords(slot);
    return {L_[0] * (c[0] + 1) / dims_[0], L_[1] * (c[1] + 1) / dims_[1],
            L_[2] * (c[2] + 1) / dims_[2]};
}

int SpatialGrid::face_neighbor(int slot, int face) const {
    auto c = coords(slot);
    int axis = face / 2;            // 0:x 1:y 2:z
    int dir = (face & 1) ? +1 : -1; // even=-, odd=+
    int nc[3] = {c[0], c[1], c[2]};
    nc[axis] = (nc[axis] + dir + dims_[axis]) % dims_[axis]; // periodic wrap
    return slot_of(nc[0], nc[1], nc[2]);
}

int SpatialGrid::owner_of(double x, double y, double z) const {
    double q[3] = {x, y, z};
    int c[3];
    for (int d = 0; d < 3; ++d) {
        // Wrap into [0, L).
        double w = q[d] - L_[d] * std::floor(q[d] / L_[d]);
        if (w < 0)
            w += L_[d]; // guard fp edge
        int ci = static_cast<int>(w * dims_[d] / L_[d]);
        if (ci >= dims_[d])
            ci = dims_[d] - 1; // clamp the L⁻ boundary
        if (ci < 0)
            ci = 0;
        c[d] = ci;
    }
    return slot_of(c[0], c[1], c[2]);
}

} // namespace cuda
} // namespace ATV
