/*
  https://github.com/NVIDIA/cudnn-frontend/blob/main/samples/cpp/sdpa/fp16_fwd.cpp
*/

#include "cudnn_graph.h"
#include "cudnn_ops.h"
#include <cudnn.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>
#include <array>

namespace {
  inline void cudnn_check(cudnnStatus_t err) {
    if (err != cudnnStatus_t::CUDNN_STATUS_SUCCESS) {
      throw std::runtime_error{cudnnGetErrorString(err)};
    }
  }

  inline void throw_if(bool cond, const std::string& message) {
    if (cond) {
      throw std::runtime_error{message};
    }
  }

  inline void throw_unless(bool cond, const std::string& message) {
    if (!cond) {
      throw std::runtime_error{message};
    }
  }

  int64_t global_uid = 0;

  /// Create cudnn tensor descriptor.
  cudnnBackendDescriptor_t createTensor(
    int64_t uid,
    cudnnDataType_t dtype,
    const std::vector<int>& dimensions,
    const std::vector<int>& strides) {
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
      cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_STRIDES,
      cudnnBackendAttributeType_t::CUDNN_TYPE_INT32,
      static_cast<int>(strides.size()),
      strides.data()));

    cudnn_check(cudnnBackendSetAttribute(
      desc,
      cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_UNIQUE_ID,
      cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
      1,
      &uid));

    cudnn_check(cudnnBackendFinalize(desc));
    return desc;
  }

  cudnnBackendDescriptor_t createMatmul(cudnnDataType_t compute_dtype) {
    cudnnBackendDescriptor_t matmul = nullptr;
    cudnn_check(cudnnBackendCreateDescriptor(
      cudnnBackendDescriptorType_t::CUDNN_BACKEND_MATMUL_DESCRIPTOR,
      &matmul));

    cudnn_check(cudnnBackendInitialize(matmul));
    cudnn_check(cudnnBackendSetAttribute(
      matmul,
      cudnnBackendAttributeName_t::CUDNN_ATTR_MATMUL_COMP_TYPE,
      CUDNN_TYPE_DATA_TYPE,
      1,
      &compute_dtype));

    cudnn_check(cudnnBackendFinalize(matmul));
    return matmul;
  }

  cudnnBackendDescriptor_t createMatmulOp(
    cudnnBackendDescriptor_t matmul,
    cudnnBackendDescriptor_t a_desc,
    cudnnBackendDescriptor_t b_desc,
    cudnnBackendDescriptor_t c_desc) {
    cudnnBackendDescriptor_t matmul_op = nullptr;
    cudnn_check(cudnnBackendCreateDescriptor(
      cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR,
      &matmul_op));

    cudnn_check(cudnnBackendInitialize(matmul_op));
    cudnn_check(cudnnBackendSetAttribute(
      matmul_op,
      cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MATMUL_ADESC,
      CUDNN_TYPE_BACKEND_DESCRIPTOR,
      1,
      a_desc));

    cudnn_check(cudnnBackendSetAttribute(
      matmul_op,
      cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MATMUL_BDESC,
      CUDNN_TYPE_BACKEND_DESCRIPTOR,
      1,
      b_desc));

    cudnn_check(cudnnBackendSetAttribute(
      matmul_op,
      cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MATMUL_CDESC,
      CUDNN_TYPE_BACKEND_DESCRIPTOR,
      1,
      c_desc));

    cudnn_check(cudnnBackendSetAttribute(
      matmul_op,
      cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MATMUL_DESC,
      CUDNN_TYPE_BACKEND_DESCRIPTOR,
      1,
      matmul));

    cudnn_check(cudnnBackendFinalize(matmul_op));
    return matmul_op;
  }

  cudnnBackendDescriptor_t createPointwise(cudnnPointwiseMode_t mode) {
    cudnnBackendDescriptor_t pointwise = nullptr;
    cudnn_check(cudnnBackendCreateDescriptor(
      cudnnBackendDescriptorType_t::CUDNN_BACKEND_POINTWISE_DESCRIPTOR,
      &pointwise));

    cudnn_check(cudnnBackendInitialize(pointwise));

    cudnn_check(cudnnBackendSetAttribute(
      pointwise,
      cudnnBackendAttributeName_t::CUDNN_ATTR_POINTWISE_MODE,
      cudnnBackendAttributeType_t::CUDNN_TYPE_POINTWISE_MODE,
      1,
      &mode));

    cudnn_check(cudnnBackendFinalize(pointwise));
    return pointwise;
  }

  inline bool pointwiseNeedsTwoOperand(cudnnPointwiseMode_t mode) {
    return mode
        == cudnnPointwiseMode_t::CUDNN_POINTWISE_DIV
        || mode
        == cudnnPointwiseMode_t::CUDNN_POINTWISE_SUB
        || mode
        == cudnnPointwiseMode_t::CUDNN_POINTWISE_ADD
        || mode
        == cudnnPointwiseMode_t::CUDNN_POINTWISE_MUL;
  }

  // Decided by pointwise mode, we'll set operands follow these rule:
  // X -> Y
  // X + B -> Y
  cudnnBackendDescriptor_t createPointwiseOp(
    cudnnPointwiseMode_t mode,
    cudnnBackendDescriptor_t pointwise,
    cudnnBackendDescriptor_t x_desc,
    cudnnBackendDescriptor_t y_desc,
    cudnnBackendDescriptor_t b_desc = nullptr) {
    cudnnBackendDescriptor_t pointwise_op = nullptr;
    cudnn_check(cudnnBackendCreateDescriptor(
      cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
      &pointwise_op));

    cudnn_check(cudnnBackendInitialize(pointwise_op));

    cudnn_check(cudnnBackendSetAttribute(
      pointwise_op,
      cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_POINTWISE_XDESC,
      CUDNN_TYPE_BACKEND_DESCRIPTOR,
      1,
      x_desc));

    cudnn_check(cudnnBackendSetAttribute(
      pointwise_op,
      cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_POINTWISE_YDESC,
      CUDNN_TYPE_BACKEND_DESCRIPTOR,
      1,
      y_desc));

    if (pointwiseNeedsTwoOperand(mode)) {
      throw_if(b_desc == nullptr, "b_desc mustn't be a null pointer");

      cudnn_check(cudnnBackendSetAttribute(
        pointwise_op,
        cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_POINTWISE_BDESC,
        CUDNN_TYPE_BACKEND_DESCRIPTOR,
        1,
        b_desc));
    }

    cudnn_check(cudnnBackendSetAttribute(
      pointwise_op,
      cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR,
      CUDNN_TYPE_BACKEND_DESCRIPTOR,
      1,
      pointwise));

    cudnn_check(cudnnBackendFinalize(pointwise_op));
    return pointwise_op;
  }

  cudnnBackendDescriptor_t
  createReduction(cudnnDataType_t compute_dtype, cudnnReduceTensorOp_t reduction_mode) {
    cudnnBackendDescriptor_t reduction = nullptr;
    cudnn_check(cudnnBackendCreateDescriptor(
      cudnnBackendDescriptorType_t::CUDNN_BACKEND_REDUCTION_DESCRIPTOR,
      &reduction));
    cudnn_check(cudnnBackendInitialize(reduction));
    cudnn_check(cudnnBackendSetAttribute(
      reduction,
      cudnnBackendAttributeName_t::CUDNN_ATTR_REDUCTION_COMP_TYPE,
      cudnnBackendAttributeType_t::CUDNN_TYPE_DATA_TYPE,
      1,
      &compute_dtype));

    cudnn_check(cudnnBackendSetAttribute(
      reduction,
      cudnnBackendAttributeName_t::CUDNN_ATTR_REDUCTION_OPERATOR,
      cudnnBackendAttributeType_t::CUDNN_TYPE_REDUCTION_OPERATOR_TYPE,
      1,
      &reduction_mode));

    cudnn_check(cudnnBackendFinalize(reduction));
    return reduction;
  }

  cudnnBackendDescriptor_t createReductionOp(
    cudnnBackendDescriptor_t reduction,
    cudnnBackendDescriptor_t x_desc,
    cudnnBackendDescriptor_t y_desc) {
    cudnnBackendDescriptor_t reduction_op = nullptr;
    cudnn_check(cudnnBackendCreateDescriptor(
      cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR,
      &reduction_op));

    cudnn_check(cudnnBackendInitialize(reduction_op));

    cudnn_check(cudnnBackendSetAttribute(
      reduction_op,
      cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_POINTWISE_XDESC,
      CUDNN_TYPE_BACKEND_DESCRIPTOR,
      1,
      x_desc));

    cudnn_check(cudnnBackendSetAttribute(
      reduction_op,
      cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_POINTWISE_YDESC,
      CUDNN_TYPE_BACKEND_DESCRIPTOR,
      1,
      y_desc));

    cudnn_check(cudnnBackendSetAttribute(
      reduction_op,
      cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_REDUCTION_DESC,
      cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
      1,
      reduction));

    cudnn_check(cudnnBackendFinalize(reduction_op));
    return reduction_op;
  }

  std::array<std::vector<cudnnBackendDescriptor_t>, 2>
  createSoftmax(cudnnBackendDescriptor_t input,
                cudnnBackendDescriptor_t output,
                const std::vector<int>& dimensions) {
    throw_unless(dimensions.size() == 4U, "dimensions must be equal to 4");
    const int b      = dimensions[0];
    const int nh     = dimensions[1];
    const int seqlen = dimensions[2];
    throw_unless(dimensions[3] == seqlen, "dimension 3 must be equal to dimension 4");

    std::vector<cudnnBackendDescriptor_t> op_descriptors;
    std::vector<cudnnBackendDescriptor_t> tensor_descriptors;

    cudnnBackendDescriptor_t max_out = createTensor(
      ++global_uid,
      cudnnDataType_t::CUDNN_DATA_FLOAT,
      {b, nh, seqlen, 1},
      {nh * seqlen, seqlen, 1, 1});
    cudnnBackendDescriptor_t reduction_max =
      createReduction(cudnnDataType_t::CUDNN_DATA_FLOAT, CUDNN_REDUCE_TENSOR_MAX);
    cudnnBackendDescriptor_t reduction_max_op = createReductionOp(reduction_max, input, max_out);
    op_descriptors.push_back(reduction_max);
    op_descriptors.push_back(reduction_max_op);
    tensor_descriptors.push_back(max_out);

    cudnnBackendDescriptor_t sub_out = createTensor(
      ++global_uid,
      cudnnDataType_t::CUDNN_DATA_FLOAT,
      {b, nh, seqlen, 1},
      {nh * seqlen, seqlen, 1, 1});
    cudnnBackendDescriptor_t pw_sub = createPointwise(cudnnPointwiseMode_t::CUDNN_POINTWISE_SUB);
    cudnnBackendDescriptor_t pw_sub_op =
      createPointwiseOp(CUDNN_POINTWISE_SUB, pw_sub, input, sub_out, max_out);
    op_descriptors.push_back(pw_sub);
    op_descriptors.push_back(pw_sub_op);
    tensor_descriptors.push_back(sub_out);

    cudnnBackendDescriptor_t exp_out = createTensor(
      ++global_uid,
      cudnnDataType_t::CUDNN_DATA_FLOAT,
      dimensions,
      {nh * seqlen, seqlen, 1, 1}); // TODO: stride maybe error.
    cudnnBackendDescriptor_t pw_exp = createPointwise(cudnnPointwiseMode_t::CUDNN_POINTWISE_EXP);
    cudnnBackendDescriptor_t pw_exp_op =
      createPointwiseOp(CUDNN_POINTWISE_EXP, pw_exp, sub_out, exp_out);
    op_descriptors.push_back(pw_exp);
    op_descriptors.push_back(pw_exp_op);
    tensor_descriptors.push_back(exp_out);

    cudnnBackendDescriptor_t sum_exp_out = createTensor(
      ++global_uid,
      cudnnDataType_t::CUDNN_DATA_FLOAT,
      dimensions,
      {nh * seqlen, seqlen, 1, 1});
    cudnnBackendDescriptor_t reduction_sum =
      createReduction(cudnnDataType_t::CUDNN_DATA_FLOAT, CUDNN_REDUCE_TENSOR_ADD);
    cudnnBackendDescriptor_t reduction_sum_op =
      createReductionOp(reduction_max, exp_out, sum_exp_out);
    op_descriptors.push_back(reduction_sum);
    op_descriptors.push_back(reduction_sum_op);
    tensor_descriptors.push_back(sum_exp_out);

    cudnnBackendDescriptor_t pw_div = createPointwise(cudnnPointwiseMode_t::CUDNN_POINTWISE_DIV);
    cudnnBackendDescriptor_t pw_div_op =
      createPointwiseOp(CUDNN_POINTWISE_DIV, pw_div, exp_out, output, sum_exp_out);
    op_descriptors.push_back(pw_div);
    op_descriptors.push_back(pw_div_op);

    return {op_descriptors, tensor_descriptors};
  }

  cudnnBackendDescriptor_t createOpGraph(const std::vector<cudnnBackendDescriptor_t>& ops) {
    cudnnBackendDescriptor_t op_graph = nullptr;
    cudnn_check(cudnnBackendCreateDescriptor(
      cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR,
      &op_graph));
    cudnn_check(cudnnBackendInitialize(op_graph));

    cudnn_check(cudnnBackendSetAttribute(
      op_graph,
      cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATIONGRAPH_OPS,
      cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
      static_cast<int>(ops.size()),
      reinterpret_cast<const void*>(ops.data())));

    cudnn_check(cudnnBackendFinalize(op_graph));

    cudnn_check(cudnnBackendFinalize(op_graph));
    return op_graph;
  }

  cudnnBackendDescriptor_t
  createEngineHeuristic(const cudnnBackendDescriptor_t& op_graph, cudnnBackendHeurMode_t mode) {
    cudnnBackendDescriptor_t engine_heuristic = nullptr;
    cudnn_check(cudnnBackendInitialize(engine_heuristic));
    cudnn_check(cudnnBackendCreateDescriptor(
      cudnnBackendDescriptorType_t::CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR,
      &engine_heuristic));
    cudnn_check(cudnnBackendSetAttribute(
      engine_heuristic,
      cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINEHEUR_MODE,
      cudnnBackendAttributeType_t::CUDNN_TYPE_HEUR_MODE,
      1,
      &mode));
    cudnn_check(cudnnBackendSetAttribute(
      engine_heuristic,
      cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
      cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
      1,
      reinterpret_cast<const void*>(&op_graph)));

    cudnn_check(cudnnBackendFinalize(engine_heuristic));
    return engine_heuristic;
  }

  cudnnBackendDescriptor_t
  createVarPack(const std::vector<void*>& pointers,
                const std::vector<int>& uids,
                void* workspace = nullptr) {
    cudnnBackendDescriptor_t var_pack = nullptr;
    cudnn_check(cudnnBackendCreateDescriptor(
      cudnnBackendDescriptorType_t::CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR,
      &var_pack));
    cudnn_check(cudnnBackendInitialize(var_pack));

    cudnn_check(cudnnBackendSetAttribute(
      var_pack,
      cudnnBackendAttributeName_t::CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
      CUDNN_TYPE_INT32,
      static_cast<int>(uids.size()),
      uids.data()));

    cudnn_check(cudnnBackendSetAttribute(
      var_pack,
      cudnnBackendAttributeName_t::CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
      CUDNN_TYPE_VOID_PTR,
      static_cast<int>(pointers.size()),
      reinterpret_cast<const void*>(pointers.data())));

    if (workspace != nullptr) {
      cudnn_check(cudnnBackendSetAttribute(
        var_pack,
        cudnnBackendAttributeName_t::CUDNN_ATTR_VARIANT_PACK_WORKSPACE,
        cudnnBackendAttributeType_t::CUDNN_TYPE_VOID_PTR,
        1,
        workspace));
    }

    cudnn_check(cudnnBackendFinalize(var_pack));
    return var_pack;
  }

  cudnnBackendDescriptor_t createEngine(cudnnBackendDescriptor_t op_graph) {
    cudnnBackendDescriptor_t engine = nullptr;
    cudnn_check(cudnnBackendCreateDescriptor(
      cudnnBackendDescriptorType_t::CUDNN_BACKEND_ENGINE_DESCRIPTOR,
      &engine));
    cudnn_check(cudnnBackendInitialize(engine));

    cudnn_check(cudnnBackendSetAttribute(
      engine,
      cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINE_OPERATION_GRAPH,
      cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
      1,
      reinterpret_cast<const void*>(&op_graph)));

    cudnn_check(cudnnBackendFinalize(engine));
    return engine;
  }

  cudnnBackendDescriptor_t createEngineCfg(cudnnBackendDescriptor_t engine) {
    cudnnBackendDescriptor_t engine_cfg = nullptr;
    cudnn_check(cudnnBackendCreateDescriptor(
      cudnnBackendDescriptorType_t::CUDNN_BACKEND_ENGINECFG_DESCRIPTOR,
      &engine_cfg));
    cudnn_check(cudnnBackendInitialize(engine_cfg));

    cudnn_check(cudnnBackendSetAttribute(
      engine_cfg,
      cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINECFG_ENGINE,
      cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
      1,
      engine));

    cudnn_check(cudnnBackendFinalize(engine_cfg));
    return engine_cfg;
  }

  cudnnBackendDescriptor_t createPlan(cudnnBackendDescriptor_t engine_cfg) {
    cudnnBackendDescriptor_t plan = nullptr;
    cudnn_check(cudnnBackendCreateDescriptor(
      cudnnBackendDescriptorType_t::CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR,
      &plan));
    cudnn_check(cudnnBackendInitialize(plan));

    cudnn_check(cudnnBackendSetAttribute(
      plan,
      cudnnBackendAttributeName_t::CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
      cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
      1,
      engine_cfg));

    cudnn_check(cudnnBackendFinalize(plan));
    return plan;
  }

  template <class Type = void>
  Type* allocate(size_t byte_size) {
    void* ptr = nullptr;
    auto ret  = cudaMalloc(&ptr, byte_size);
    throw_unless(ret == cudaSuccess, "malloc failed");
    return reinterpret_cast<Type*>(ptr);
  }

} // namespace

int main() noexcept(false) {
  const int bs        = 1;
  const int nh        = 16;
  const int seq_len   = 16;
  const int head_embd = 16;

  cudnnBackendDescriptor_t Q =
    createTensor(++global_uid, cudnnDataType_t::CUDNN_DATA_HALF, {bs, nh, seq_len, head_embd}, {});
  cudnnBackendDescriptor_t K =
    createTensor(++global_uid, cudnnDataType_t::CUDNN_DATA_HALF, {bs, nh, seq_len, head_embd}, {});
  cudnnBackendDescriptor_t V =
    createTensor(++global_uid, cudnnDataType_t::CUDNN_DATA_HALF, {bs, nh, seq_len, head_embd}, {});
  cudnnBackendDescriptor_t O =
    createTensor(++global_uid, cudnnDataType_t::CUDNN_DATA_HALF, {bs, nh, seq_len, head_embd}, {});

  cudnnBackendDescriptor_t P =
    createTensor(++global_uid, cudnnDataType_t::CUDNN_DATA_HALF, {bs, nh, seq_len, seq_len}, {});
  cudnnBackendDescriptor_t S =
    createTensor(++global_uid, cudnnDataType_t::CUDNN_DATA_HALF, {bs, nh, seq_len, seq_len}, {});

  // Attention
  cudnnBackendDescriptor_t matmul_qk         = createMatmul(cudnnDataType_t::CUDNN_DATA_FLOAT);
  cudnnBackendDescriptor_t matmul_qk_op      = createMatmulOp(matmul_qk, Q, K, P);
  auto [softmax_sub_op, softmax_sub_tensors] = createSoftmax(P, S, {bs, nh, seq_len, head_embd});

  cudnnBackendDescriptor_t matmul_sv    = createMatmul(cudnnDataType_t::CUDNN_DATA_FLOAT);
  cudnnBackendDescriptor_t matmul_sv_op = createMatmulOp(matmul_sv, S, V, O);

  std::vector<cudnnBackendDescriptor_t> total_ops;
  total_ops.push_back(matmul_qk);
  total_ops.insert(total_ops.end(), softmax_sub_op.begin(), softmax_sub_op.end());
  total_ops.push_back(matmul_sv);

  cudnnBackendDescriptor_t op_graph   = createOpGraph(total_ops);
  cudnnBackendDescriptor_t engine     = createEngine(op_graph);
  cudnnBackendDescriptor_t engine_cfg = createEngineCfg(op_graph);
  cudnnBackendDescriptor_t plan       = createPlan(op_graph);

  // TODO: mode meaning?
  cudnnBackendDescriptor_t engine_heuristic =
    createEngineHeuristic(op_graph, cudnnBackendHeurMode_t::CUDNN_HEUR_MODE_A);

  // Execute
  half* q_ptr = allocate<half>(1);
  half* k_ptr = allocate<half>(1);
  half* v_ptr = allocate<half>(1);
  half* o_ptr = allocate<half>(1);

  int64_t workspace_size = -1;
  int64_t ele_cnt        = -1;
  cudnn_check(cudnnBackendGetAttribute(
    plan,
    cudnnBackendAttributeName_t::CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
    CUDNN_TYPE_INT64,
    1,
    &ele_cnt,
    &workspace_size));
  void* workspace_ptr = allocate(workspace_size);

  cudnnBackendDescriptor_t var_pack =
    createVarPack({q_ptr, k_ptr, v_ptr, o_ptr}, {0, 1, 2, 3}, workspace_ptr);

  cudnnHandle_t handle = nullptr;
  cudnn_check(cudnnCreate(&handle));
  cudnn_check(cudnnBackendExecute(handle, plan, var_pack));

  cudnn_check(cudnnDestroy(handle));
  // TODO: Recycle other resources.
  return 0;
}

