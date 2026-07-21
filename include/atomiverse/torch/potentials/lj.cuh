/// @file torch/potentials/lj.cuh
/// @brief Differentiable LJ 12-6 operator for the torch backend.
///
/// Forward reuses cuda::kernels::lj_force. Backward provides analytical
/// gradients w.r.t. positions, epsilon, and sigma via autograd::Function.
///
/// The neighbor pair list is no longer carried by `torch::Frame`: it is
/// owned by `torch::NeighborList`. `lj()` therefore takes the pair list
/// from a `torch::NeighborList&`, consistent with how the CUDA `PairLJCut`
/// consumes the pair list published by `cuda::NeighborList`.
///
/// @see cuda::kernels::lj_force for the shared kernel
/// @see torch::Frame for the Tensor-based system state
/// @see torch::NeighborList for the owner of the neighbor pair list
#pragma once

#ifdef ATV_HAS_TORCH

#include "atomiverse/torch/frame/frame.cuh"
#include "atomiverse/torch/neighbor/neighbor_list.cuh"

namespace ATV {
namespace torch {

/// @brief Compute differentiable LJ energy and forces.
///
/// Accumulates energy into frame.energy and forces into
/// frame.atoms.fx/fy/fz. Supports autograd w.r.t. positions
/// (frame.atoms.x/y/z), epsilon, and sigma. The pair list is read from the
/// supplied NeighborList, which must have been built/updated for `frame`.
///
/// @param frame     torch::Frame with positions, forces, and energy tensors.
/// @param neighbors torch::NeighborList holding the up-to-date pair list.
/// @param epsilon   LJ well depth [eV]. Scalar Tensor, may require grad.
/// @param sigma     LJ collision diameter [Angstrom]. Scalar Tensor, may require grad.
/// @param cutoff    Interaction cutoff [Angstrom]. Not differentiable.
/// @return Energy Tensor [1] (for .backward() calls).
Tensor lj(Frame& frame, const NeighborList& neighbors, const Tensor& epsilon,
          const Tensor& sigma, double cutoff);

} // namespace torch
} // namespace ATV

#endif // ATV_HAS_TORCH
