#pragma once

#include <proxy.h>

#include "tensor.h"

PRO_DEF_MEM_DISPATCH(MemComputeCpuGolden, ComputeCpuGolden);

struct PrecisionProxy
  : pro::facade_builder                                                                  //
    ::add_convention<MemCreateInputs,
                     Tensors() const noexcept>                                           //
    ::add_convention<MemCompare, bool(const Tensor& golden, const Tensor& actual) const> //
    ::support_copy<pro::constraint_level::nontrivial>                                    //
    ::build { };

template <class Facade>
auto MakePrecisionProxy(Facade& facade) -> pro::proxy<PrecisionProxy> {
  return pro::make_proxy<PrecisionProxy, Facade>(facade);
}
