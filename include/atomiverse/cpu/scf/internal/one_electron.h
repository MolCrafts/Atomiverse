/// @file cpu/scf/internal/one_electron.h
/// @brief Nuclear repulsion energy for Hartree-Fock / DFT.
///
/// The one-electron integral kernels (overlap / kinetic / nuclear attraction)
/// are private static members of `internal::IntegralWorkspace`
/// (`internal::IntegralWorkspace::overlap/kinetic/nuclear_attraction`, defined in
/// one_electron.cpp) — construct an `internal::IntegralWorkspace{frame, basis}` to obtain
/// S / T / V / Hcore:
///
/// @code
///   auto atoms  = AtomList{frame, charge};
///   double vnn = nuclear_repulsion_energy(atoms);
/// @endcode
///
/// These kernels use the native integral primitives in native_integrals.h.
/// The libcint reference path stays under test utilities and is not used
/// in production code.
#pragma once

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/scf/internal/basis_workspace.h"

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Classical nucleus-nucleus repulsion energy
///        V_nuc = Σ_{A<B} Z_A Z_B / |R_A - R_B|, in Hartree.
/// @param atoms AtomList with positions in Bohr.
/// @return Nuclear repulsion energy [Hartree].
[[nodiscard]] double nuclear_repulsion_energy(const AtomList& atoms);

/// @brief Multipole one-electron integrals about a fixed origin C.
///
/// Self-contained primitive-pair kernel (Gaussian product + even-moment
/// closed forms): ⟨μ|(r−C)_i|ν⟩ (3 matrices) and ⟨μ|(r−C)_i(r−C)_j|ν⟩
/// (6 matrices, xx,xy,xz,yy,yz,zz order). Engine Cartesian convention;
/// spherical-convention consumers sandwich with the pack's `c2s`.
/// Shared kernel for the dipole/quadrupole observers, the uniform-field
/// core-Hamiltonian term, and (later) picture-change property operators.
/// Obara & Saika, J. Chem. Phys. 84, 3963 (1986), DOI 10.1063/1.450106.
class MultipoleIntegrals {
  public:
    /// @brief Compute all first- and second-moment matrices at construction.
    /// @param shells Per-Cartesian-AO shell descriptors (engine convention).
    /// @param n_ao   Cartesian AO count (== shells.size()).
    /// @param origin_bohr Expansion origin C [Bohr], length 3.
    MultipoleIntegrals(const native::Shell* shells, int n_ao,
                       const double* origin_bohr);

    /// @brief First moment ⟨μ|(r−C)_a|ν⟩.
    /// @param a Cartesian axis 0..2. @return (n×n) matrix [Bohr].
    const xt::xtensor<double, 2>& dipole(int a) const { return m1_[a]; }

    /// @brief Second moment ⟨μ|(r−C)_a(r−C)_b|ν⟩ in packed order
    ///        xx,xy,xz,yy,yz,zz.
    /// @param k Packed index 0..5. @return (n×n) matrix [Bohr²].
    const xt::xtensor<double, 2>& second(int k) const { return m2_[k]; }

  private:
    xt::xtensor<double, 2> m1_[3];
    xt::xtensor<double, 2> m2_[6];
};

} // namespace scf
} // namespace cpu
} // namespace ATV
