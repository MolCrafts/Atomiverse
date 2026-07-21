/// @file torch/potentials/harmonic_bond.cuh
/// @brief Differentiable harmonic bond operator for the torch backend.
///
/// Forward reuses cuda::kernels::harmonic_bond_k. Backward provides analytical
/// gradients w.r.t. positions, K, and r0 via autograd::Function.
///
/// @see cuda::kernels::harmonic_bond_k for the shared kernel
/// @see torch::Frame for the Tensor-based system state
#pragma once

#ifdef ATV_HAS_TORCH

#include "atomiverse/torch/frame/frame.cuh"

namespace ATV {
namespace torch {

/// @brief Compute differentiable harmonic bond energy and forces.
///
/// Accumulates energy into frame.energy and forces into frame.atoms.fx/fy/fz.
/// Supports autograd w.r.t. positions (frame.atoms.x/y/z), K, and r0.
///
/// @param frame  torch::Frame with positions, box, and bond topology.
/// @param K      Spring constants per bond type [n_types]. May require grad.
/// @param r0     Equilibrium distances per bond type [n_types]. May require grad.
/// @return Energy Tensor [1] (for .backward() calls).
Tensor harmonic_bond(Frame& frame, const Tensor& K, const Tensor& r0);

} // namespace torch
} // namespace ATV

#endif // ATV_HAS_TORCH
