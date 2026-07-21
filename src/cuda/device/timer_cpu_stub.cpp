/// @file cuda/device/timer_cpu_stub.cpp
/// @brief Inert cuda::Timer linkage used only on CPU-only builds
///        (`-DATV_BUILD_CUDA=OFF`).
///
/// The CPU build still needs out-of-line definitions for cuda::Timer
/// special members because `facade/driver.h` declares a
/// `std::unique_ptr<cuda::Timer>` member (the CUDA-free PIMPL facade)
/// and the destructor of every CPU `Driver` instantiation must link
/// against that symbol. On CUDA builds those definitions come from
/// `src/cuda/device/timer.cu`; on non-CUDA builds the .cu source is omitted,
/// so we provide a parallel inert TU here that defines the same
/// symbols against a trivial CPU-only `Impl`.

#include "atomiverse/cuda/device/timer.h"

namespace ATV {
namespace cuda {

class Timer::Impl {
  public:
    void init() {}
    void step_done() {}
    std::string summary(const std::vector<std::string>&) const { return {}; }
    void shutdown() {}
};

Timer::Timer() : impl_(std::make_unique<Impl>()) {}

Timer::~Timer() = default;

Timer::Timer(Timer&&) noexcept = default;

Timer& Timer::operator=(Timer&&) noexcept = default;

void Timer::init() { impl_->init(); }

void Timer::step_done() { impl_->step_done(); }

std::string Timer::summary(const std::vector<std::string>& phase_names) const {
    return impl_->summary(phase_names);
}

void Timer::shutdown() { impl_->shutdown(); }

} // namespace cuda
} // namespace ATV
