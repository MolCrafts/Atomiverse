/// @file cuda/device/timer.cu
/// @brief cuda::Timer PIMPL facade — out-of-line special members / methods.
///
/// These definitions live in a `.cu` translation unit (where `Timer::Impl`
/// is complete) so that non-CUDA translation units holding a
/// `std::unique_ptr<cuda::Timer>` member — notably `src/cpu/driver/driver.cpp`'s
/// `Driver<PhaseEnum, cpu::device_tag>` instantiations — can compile and
/// link the `Driver` destructor against only the CUDA-free `cuda/timer.h`
/// declaration.

#include "atomiverse/cuda/device/timer.cuh"

namespace ATV {
namespace cuda {

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
