/// @file cpu/planewave/internal/g_vector.h
/// @brief G-vector enumeration for plane-wave basis.
#pragma once

#include "atomiverse/core/mat3.h"
#include "atomiverse/core/precision.h"
#include "atomiverse/core/vec3.h"
#include "atomiverse/cpu/frame/box.h"
#include <vector>

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {

struct GVectorList {
    std::vector<Vec3> g;       // G-vectors in Cartesian coords [1/Bohr]
    std::vector<AccReal> ekin; // kinetic energy 0.5*|k+G|^2 [Hartree]
    std::vector<int> h, k, l;  // integer indices: G = h*b0 + k*b1 + l*b2
    int nx, ny, nz;            // FFT grid dimensions
    Vec3 kvec{};               // k-point of this sphere [1/Bohr]; nonlocal
                               // projectors and wfc live on q = k+G

    int count() const { return static_cast<int>(g.size()); }
};

GVectorList enumerate_g_vectors(const cpu::Box& cell_bohr, AccReal ecut,
                                const Vec3& k);

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV
