/// @file cuda/potentials/dihedral.cuh
/// @brief LAMMPS-style abstract base for bonded 4-body styles (CUDA).
#pragma once
#include "atomiverse/facade/component_role.h"

#include <cstdint>
#include <functional>
#include <vector>

#include "atomiverse/cuda/component/component.cuh"

namespace ATV {
namespace cuda {

class Dihedral : public Component {
  public:
    Role role() const override { return 0; }
    ~Dihedral() override = default;

    /* role via Potential/kind base */
    /// @brief Capabilities — Energy + Force + Stress. The kernel writes
    ///        per-tuple virial contributions into `frame.scalars->stress[]`
    ///        (see `cuda/potentials/dihedral_multi_harmonic.cu`).

    void launch(const LaunchContext& ctx) final { compute(ctx); }

  protected:
    Dihedral() = default;

    virtual void compute(const LaunchContext& ctx) = 0;

    const std::vector<int>&
    select_owned_tuples(const LaunchContext& ctx, int n_tuples,
                        const std::function<std::int64_t(int)>& owner_tag_of);

  private:
    std::vector<int> owned_;
};

} // namespace cuda
} // namespace ATV
