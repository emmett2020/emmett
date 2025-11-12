#include "instance_norm/instance_norm.h"

namespace cuda_op {
  __inline__ __device__ float warp_reduce_sum(float val) {
    for (int i = 16; i > 0; i /= 2) {
      val += __shfl_down_sync(0xFFFFFFFF, val, i);
    }
    return val;
  }

} // namespace cuda_op
