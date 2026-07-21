# LibTorch Differentiable Potentials

This page belongs in the user guide because it explains how to use one part of
Atomiverse, not because it teaches a new piece of computational chemistry.
If your goal is differentiable simulation, parameter fitting, or PyTorch
interop, this is the right place to start. If your goal is to learn SCF, RHF,
UHF, or UKS, stay in the tutorial track instead.

`ATV::torch` is **not a third execution backend** — it has no `Driver`, no
`Component`, no `backend_traits`, and no device tag. It is a
*differentiable-potential library*: a tensor-backed `Frame`, a neighbor-list
builder, and a small set of potential kernels exposed as free functions on
`at::Tensor`. Use it for workflows where a simulation operator is not the end of
the computation but one differentiable step inside a larger optimization
problem. You may want the derivative of a Lennard-Jones energy with respect to
$\varepsilon$ or $\sigma$. You may want to fit force-field parameters against
reference data. You may want a PyTorch model to exchange GPU tensors with
Atomiverse directly. The library makes those workflows possible by exposing
Atomiverse kernels through `at::Tensor` objects that can participate in autograd.

## How It Relates To The CUDA Backend

The CUDA backend stores system data in raw device pointers. The torch layer
stores the same physical quantities in tensors. That does not create a separate
simulation engine, and it does not add a driver: the forward computations still
reuse the existing CUDA kernels wherever possible. The practical change is that
positions or parameters can now carry `requires_grad=true`, so a forward call
followed by `.backward()` yields derivatives in the usual PyTorch way.

## Building With LibTorch

Enable it at CMake configure time:

```bash
cmake -B build \
  -DATV_BUILD_TORCH=ON \
  -DATV_BUILD_TESTS=ON \
  -DCMAKE_PREFIX_PATH=/path/to/libtorch
cmake --build build
```

`CMAKE_PREFIX_PATH` must point to the LibTorch installation that provides
`TorchConfig.cmake`, and the LibTorch CUDA version must match the CUDA toolkit
used to build Atomiverse.

The repository does not currently ship a standalone `examples/torch_*.cpp`
target, so the best reading path is through the library headers and tests:
[`tests/torch/test_lj.cu`](https://github.com/MolCrafts/Atomiverse/blob/dev/tests/torch/test_lj.cu),
[`tests/torch/test_transfer.cu`](https://github.com/MolCrafts/Atomiverse/blob/dev/tests/torch/test_transfer.cu), and
[`tests/torch/test_neighbor_list.cu`](https://github.com/MolCrafts/Atomiverse/blob/dev/tests/torch/test_neighbor_list.cu).

## The Core Type: `torch::Frame`

The defining type is
[`include/atomiverse/torch/frame/frame.cuh`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/torch/frame/frame.cuh).
Conceptually it mirrors `cpu::Frame` and `cuda::Frame`: positions, forces, pair
lists, energy, box, and host-cached counts. The difference is only the storage
container. In the torch backend those fields are tensors.

That one substitution changes how the data can be used. A position tensor or a
parameter tensor can be made differentiable. A call such as `ATV::torch::lj`
then becomes part of the autograd graph, so a later `.backward()` produces
gradients with respect to the inputs that requested them.

## A Minimal Example

The following sketch matches the public API that exists today:

```cpp
#include "atomiverse/torch/frame/frame.cuh"
#include "atomiverse/torch/neighbor/neighbor_list.cuh"
#include "atomiverse/torch/potentials/lj.cuh"

namespace atorch = ATV::torch;

int main() {
    auto real_opts = ::torch::TensorOptions()
        .dtype(atorch::kRealDtype)
        .device(::torch::kCUDA);
    auto int_opts = ::torch::TensorOptions()
        .dtype(atorch::kIntDtype)
        .device(::torch::kCUDA);

    atorch::Frame frame;
    frame.n_atoms_ = 2;
    // Per-atom tensors live in the `atoms` sub-struct (mirroring cpu::Frame
    // and the molrs "atoms" block); the system scalars sit at the top level.
    frame.atoms.x = ::torch::tensor({0.0, 1.5}, real_opts).requires_grad_(true);
    frame.atoms.y = ::torch::zeros({2}, real_opts);
    frame.atoms.z = ::torch::zeros({2}, real_opts);
    frame.atoms.fx = ::torch::zeros({2}, real_opts);
    frame.atoms.fy = ::torch::zeros({2}, real_opts);
    frame.atoms.fz = ::torch::zeros({2}, real_opts);
    frame.atoms.type = ::torch::zeros({2}, int_opts);
    frame.energy = ::torch::zeros({1}, real_opts);
    frame.box = ::torch::tensor(
        {20.0, 0.0, 0.0,
          0.0, 20.0, 0.0,
          0.0, 0.0, 20.0},
        real_opts);

    auto neighbors = atorch::neighbor_list(frame, 3.0);

    auto epsilon = ::torch::tensor({1.0}, real_opts).requires_grad_(true);
    auto sigma   = ::torch::tensor({1.0}, real_opts).requires_grad_(true);

    auto energy = atorch::lj(frame, neighbors, epsilon, sigma, 3.0);
    energy.backward();
    return 0;
}
```

The sequence is simple. Allocate a tensor-backed frame on the GPU, build the
pair list, call a differentiable operator, then invoke `.backward()`. The
important point is that the numerical work in the forward and backward passes
belongs to Atomiverse while the differentiation interface belongs to PyTorch.

## Transfers Stay Explicit

The library follows the same explicit data-residency model as the rest of the
project. The key functions are declared in
[`include/atomiverse/torch/frame/transfer.cuh`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/torch/frame/transfer.cuh).

`to_torch(const cuda::Frame&)` wraps an existing CUDA frame as tensors without a
copy. `to_cuda(const torch::Frame&, cuda::Frame&)` extracts raw pointers from
the tensors and lets CUDA-side code borrow them. `from_cpu(const cpu::Frame&)`
performs a host-to-device copy into new tensors, and `to_cpu` copies results
back to the CPU representation. Once you understand those four names, the torch
library stops feeling like a special case. It is simply another explicit data
representation with one extra capability: autograd.

## What Is Differentiable

The right distinction is between smooth physical operators and discrete
infrastructure. Lennard-Jones and harmonic-bond operators are differentiable
because they map continuous inputs such as positions or parameters to continuous
outputs such as energy. The neighbor-list builder is not differentiable because
pair membership changes discretely. That is why `neighbor_list` prepares the
`pair_*` tensors but is kept outside the gradient story.

## Where To Read Next

If you want to understand the implementation, start with
[`include/atomiverse/torch/frame/frame.cuh`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/torch/frame/frame.cuh),
then read
[`include/atomiverse/torch/frame/transfer.cuh`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/torch/frame/transfer.cuh),
then
[`include/atomiverse/torch/potentials/lj.cuh`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/torch/potentials/lj.cuh)
and
[`src/torch/potentials/lj.cu`](https://github.com/MolCrafts/Atomiverse/blob/dev/src/torch/potentials/lj.cu).
Finish with
[`tests/torch/test_lj.cu`](https://github.com/MolCrafts/Atomiverse/blob/dev/tests/torch/test_lj.cu) and
[`tests/torch/test_neighbor_list.cu`](https://github.com/MolCrafts/Atomiverse/blob/dev/tests/torch/test_neighbor_list.cu),
which show real calls and gradient checks against the current API.
