/// @file cpu/planewave/norm_conserving_potential.h
/// @brief Built-in providers of the public scalar NC radial contract.
#pragma once

#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/electronic/norm_conserving_potential.h"
#include <memory>
#include <string>
#include <string_view>

namespace ATV::cpu::pw {

/// @brief Published or explicitly supplied analytic GTH data, admitted as NC.
class GthPseudopotential final
    : public ::ATV::electronic::NormConservingPotential {
  public:
    /// @brief Select a published table for one explicit element.
    GthPseudopotential(molrs::Element element, std::string_view name);
    /// @brief Use an explicitly supplied analytic GTH parameterization.
    explicit GthPseudopotential(gpw::GthPotential potential);
    ~GthPseudopotential() override;
    int atomic_number() const noexcept override;
    int valence() const noexcept override;
    ::ATV::electronic::NcProjectorSetView projectors() const override;

  protected:
    void evaluate_local(std::span<const AccReal>, std::span<AccReal>,
                        std::span<AccReal>) const override;

  private:
    struct Data;
    std::shared_ptr<const Data> data_;
};

/// @brief Read a scalar NC UPF once; reject unsupported physical channels.
class UpfPseudopotential final
    : public ::ATV::electronic::NormConservingPotential {
  public:
    /// @brief Load a declared NC UPF, fixing element and Ha units at admission.
    explicit UpfPseudopotential(const std::string& path);
    ~UpfPseudopotential() override;
    int atomic_number() const noexcept override;
    int valence() const noexcept override;
    ::ATV::electronic::NcProjectorSetView projectors() const override;

  protected:
    void evaluate_local(std::span<const AccReal>, std::span<AccReal>,
                        std::span<AccReal>) const override;

  private:
    struct Data;
    std::shared_ptr<const Data> data_;
};

} // namespace ATV::cpu::pw
