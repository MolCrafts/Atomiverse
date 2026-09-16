/// @file cpu/observe/elastance_observer.cpp
#include "atomiverse/cpu/observe/elastance_observer.h"

#include <cmath>

namespace ATV {
namespace cpu {

void ElastanceObserver::launch(const LaunchContext&) {
    last_cond_ = conp_->matrix_condition();
    last_C_ = conp_->capacitance();
}

} // namespace cpu
} // namespace ATV
