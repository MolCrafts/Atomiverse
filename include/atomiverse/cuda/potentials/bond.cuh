/// @file cuda/potentials/bond.cuh
/// @brief LAMMPS-style abstract base for bonded 2-body styles (CUDA).
///
/// Mirror of `cpu::Bond` shape. No multi-rank DD on the CUDA backend
/// today; `select_owned_tuples` is a shim returning `0..n_bonds-1`
/// until DD lands.
#pragma once
#include "atomiverse/facade/component_role.h"

#include <cstdint>
#include <functional>
#include <vector>

#include "atomiverse/cuda/component/component.cuh"

namespace ATV {
namespace cuda {

class Bond : public Component {
  public:
    Role role() const override { return 0; }
    ~Bond() override = default;

    /* role via Potential/kind base */

    void launch(const LaunchContext& ctx) final { compute(ctx); }

  protected:
    Bond() = default;

    virtual void compute(const LaunchContext& ctx) = 0;

    /// @brief Indices into `frame.bonds` this rank should evaluate.
    /// CUDA has no MPI DD today — returns `0..n_tuples-1`. Signature
    /// matches `cpu::Bond::select_owned_tuples` so derived classes look
    /// the same on both backends.
    const std::vector<int>&
    select_owned_tuples(const LaunchContext& ctx, int n_tuples,
                        const std::function<std::int64_t(int)>& owner_tag_of);

  private:
    std::vector<int> owned_;
};

} // namespace cuda
} // namespace ATV
