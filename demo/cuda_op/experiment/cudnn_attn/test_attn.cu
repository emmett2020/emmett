/*
  https://github.com/NVIDIA/cudnn-frontend/blob/main/samples/cpp/sdpa/fp16_fwd.cpp
*/

#include "cudnn_graph.h"
#include "cudnn_ops.h"
#include <cudnn.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

namespace {
  inline void cudnn_check(cudnnStatus_t err) {
    if (err != cudnnStatus_t::CUDNN_STATUS_SUCCESS) {
      throw std::runtime_error{cudnnGetErrorString(err)};
    }
  }

  // data type, shape
  // TODO: stride
  cudnnBackendDescriptor_t
  createTensor(int64_t uid, cudnnDataType_t dtype, const std::vector<int> &dimensions) {
    cudnnBackendDescriptor_t desc = nullptr;

    cudnn_check(cudnnBackendCreateDescriptor(
      cudnnBackendDescriptorType_t::CUDNN_BACKEND_TENSOR_DESCRIPTOR,
      &desc));
    cudnn_check(cudnnBackendInitialize(desc));
    cudnn_check(cudnnBackendSetAttribute(
      desc,
      cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_DATA_TYPE,
      cudnnBackendAttributeType_t::CUDNN_TYPE_DATA_TYPE,
      1,
      &dtype));

    cudnn_check(cudnnBackendSetAttribute(
      desc,
      cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_DIMENSIONS,
      cudnnBackendAttributeType_t::CUDNN_TYPE_INT32,
      static_cast<int>(dimensions.size()),
      dimensions.data()));

    cudnn_check(cudnnBackendSetAttribute(
      desc,
      cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_UNIQUE_ID,
      cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
      1,
      &uid));

    cudnn_check(cudnnBackendFinalize(desc));
    return desc;
  }

  cudnnBackendDescriptor_t createAttention() {
    cudnnBackendDescriptor_t attn_op = nullptr;

    return attn_op;
  }

} // namespace

int main() noexcept(false) {
  cudnnHandle_t cudnn = nullptr;
  cudnn_check(cudnnCreate(&cudnn));

  const int bs        = 1;
  const int nh        = 16;
  const int seq_len   = 16;
  const int head_embd = 16;

  cudnnBackendDescriptor_t Q =
    createTensor(0, cudnnDataType_t::CUDNN_DATA_HALF, {bs, nh, seq_len, head_embd});
  cudnnBackendDescriptor_t K =
    createTensor(1, cudnnDataType_t::CUDNN_DATA_HALF, {bs, nh, seq_len, head_embd});
  cudnnBackendDescriptor_t V =
    createTensor(2, cudnnDataType_t::CUDNN_DATA_HALF, {bs, nh, seq_len, head_embd});
  cudnnBackendDescriptor_t O =
    createTensor(3, cudnnDataType_t::CUDNN_DATA_HALF, {bs, nh, seq_len, head_embd});

  /// Create op
  cudnnBackendDescriptor_t attn_op = nullptr;
  cudnn_check(cudnnBackendCreateDescriptor(
    cudnnBackendDescriptorType_t::
      CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR, // TODO: where is attention?
    &attn_op));

  cudnn_check(cudnnBackendSetAttribute(
    attn_op,
    cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINE_OPERATION_GRAPH,
    cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
    1,
    nullptr));

  /// Create op_graph
  cudnnBackendDescriptor_t op_graph = nullptr;
  cudnn_check(cudnnBackendCreateDescriptor(
    cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR,
    &op_graph));

  cudnn_check(cudnnBackendSetAttribute(
    op_graph,
    cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MATMUL_DESC,
    cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
    1,
    attn_op));

  cudnnBackendDescriptor_t engine_heuristic = nullptr;
  cudnn_check(cudnnBackendCreateDescriptor(
    cudnnBackendDescriptorType_t::CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR,
    &engine_heuristic));
  auto mode = cudnnBackendHeurMode_t::CUDNN_HEUR_MODE_A;
  cudnn_check(cudnnBackendSetAttribute(
    engine_heuristic,
    cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINEHEUR_MODE,
    cudnnBackendAttributeType_t::CUDNN_TYPE_HEUR_MODE,
    1,
    &mode));

  cudnnBackendDescriptor_t var_pack = nullptr;
  cudnn_check(cudnnBackendCreateDescriptor(
    cudnnBackendDescriptorType_t::CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR,
    &var_pack));
  // cudnnBackendSetAttribute(var_pack,
  //                          cudnnBackendAttributeName_t::CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, );


  cudnnBackendDescriptor_t plan = nullptr;
  cudnn_check(cudnnBackendCreateDescriptor(
    cudnnBackendDescriptorType_t::CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR,
    &plan));

  cudnn_check(cudnnBackendSetAttribute(
    plan,
    cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINE_OPERATION_GRAPH,
    cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
    1,
    engine_heuristic));

  cudnn_check(cudnnDestroy(cudnn));
  return 0;
}

