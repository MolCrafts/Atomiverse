/// @file cpu/planewave/internal/response_mixer.h
/// @brief Joint real/complex density packing for the canonical PW Pulay mix.
#pragma once

#include "atomiverse/cpu/planewave/internal/pw_setup.h"

#include <complex>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace ATV::cpu::pw::internal {

/// Per-perturbation mixing history, independent of the response stopping rule.
/// All spin channels enter one residual vector. A uniform real-space volume
/// weight multiplies every Gram entry equally and leaves its minimum unchanged.
class ResponseMixer {
  public:
    ResponseMixer(std::size_t grid_points, std::size_t channels,
                  AccReal fraction, int history_depth)
        : grid_points_(grid_points), channels_(channels), fraction_(fraction),
          history_depth_(history_depth) {
        if (grid_points == 0 || channels < 1 || channels > 2)
            throw std::invalid_argument("PW response mix: invalid shape");
        input_.resize({grid_points * channels});
        output_.resize(input_.shape());
    }

    /// Clear on changing the perturbation right-hand side; retain scratch.
    void reset() {
        residual_history_.clear();
        density_history_.clear();
    }

    template <typename Scalar>
    struct Channel {
        std::span<Scalar> density;
        std::span<const Scalar> unmixed;
    };

    /// Replace the current density with a Pulay update of the unmixed output.
    /// The caller checks its raw residual and accepts a converged output first.
    template <typename Scalar>
    void mix(std::initializer_list<Channel<Scalar>> channels) {
        static_assert(std::is_same_v<Scalar, AccReal>
                      || std::is_same_v<Scalar, std::complex<AccReal>>);
        if (channels.size() != channels_)
            throw std::invalid_argument("PW response mix: channel mismatch");
        for (const auto& channel : channels)
            if (channel.density.size() != grid_points_
                || channel.unmixed.size() != grid_points_)
                throw std::invalid_argument("PW response mix: grid mismatch");

        std::size_t offset = 0;
        for (const auto& channel : channels) {
            for (std::size_t i = 0; i < grid_points_; ++i) {
                input_[offset + i] = channel.density[i];
                output_[offset + i] = channel.unmixed[i];
            }
            offset += grid_points_;
        }
        pulay_mix(output_, input_, residual_history_, density_history_,
                  fraction_, history_depth_);
        offset = 0;
        for (const auto& channel : channels) {
            for (std::size_t i = 0; i < grid_points_; ++i) {
                if constexpr (std::is_same_v<Scalar, AccReal>)
                    channel.density[i] = output_[offset + i].real();
                else
                    channel.density[i] = output_[offset + i];
            }
            offset += grid_points_;
        }
    }

  private:
    std::size_t grid_points_, channels_;
    AccReal fraction_;
    int history_depth_;
    xt::xarray<std::complex<AccReal>> input_, output_;
    std::vector<xt::xarray<std::complex<AccReal>>> residual_history_;
    std::vector<xt::xarray<std::complex<AccReal>>> density_history_;
};

} // namespace ATV::cpu::pw::internal
