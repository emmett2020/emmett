#pragma once

#include <proxy.h>

#include "tensor.h"

PRO_DEF_MEM_DISPATCH(MemComputeCpuGolden, ComputeCpuGolden);
PRO_DEF_MEM_DISPATCH(MemComputeActual, ComputeActual);
PRO_DEF_MEM_DISPATCH(MemCreateInputs, CreateInputs);
PRO_DEF_MEM_DISPATCH(MemCompare, Compare);

struct PrecisionProxy
  : pro::facade_builder                                                                  //
    ::add_convention<MemCreateInputs,
                     Tensors() const noexcept>                                           //
    ::add_convention<MemComputeCpuGolden, Tensors(const Tensors&)>                       //
    ::add_convention<MemComputeActual, Tensors(const Tensors&)>                          //
    ::add_convention<MemCompare, bool(const Tensor& golden, const Tensor& actual) const> //
    ::support_copy<pro::constraint_level::nontrivial>                                    //
    ::build { };

template <class Facade>
auto MakePrecisionProxy(Facade& facade) -> pro::proxy<PrecisionProxy> {
  return pro::make_proxy<PrecisionProxy, Facade>(facade);
}

void TestPrecision(pro::proxy<PrecisionProxy> op);
