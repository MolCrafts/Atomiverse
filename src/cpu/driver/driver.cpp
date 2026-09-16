/// @file cpu/driver/driver.cpp
/// @brief CPU Driver explicit instantiations.

#include "atomiverse/core/driver_template_impl.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/driver/backend_traits.h"

namespace ATV {

template class Driver<md::Phase, cpu::device_tag>;
template class Driver<minimizer::Phase, cpu::device_tag>;
template class Driver<evaluate::Phase, cpu::device_tag>;
template class Driver<vib::Phase, cpu::device_tag>;
template class Driver<sampling::Phase, cpu::device_tag>;
template class Driver<path::Phase, cpu::device_tag>;
template class Driver<scf::Phase, cpu::device_tag>;
template class Driver<gpw::Phase, cpu::device_tag>;
template class Driver<pw::Phase, cpu::device_tag>;

} // namespace ATV
