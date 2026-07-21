/// @file cpu/driver/driver.cpp
/// @brief CPU Driver explicit instantiations.

#include "atomiverse/cpu/driver/backend_traits.h"
#include "atomiverse/facade/driver_impl.h"
#include "atomiverse/facade/stage.h"

namespace ATV {

template class Driver<md::Phase, cpu::device_tag>;
template class Driver<minimizer::Phase, cpu::device_tag>;
template class Driver<evaluate::Phase, cpu::device_tag>;
template class Driver<vib::Phase, cpu::device_tag>;
template class Driver<sampling::Phase, cpu::device_tag>;
template class Driver<path::Phase, cpu::device_tag>;
template class Driver<scf::Phase, cpu::device_tag>;

} // namespace ATV
