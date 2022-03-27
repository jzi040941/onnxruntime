// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <limits>

#include "gru_base.h"

#include "core/framework/op_kernel.h"
#include "core/providers/cpu/rnn/rnn_helpers.h"

namespace onnxruntime {

/// The class represents GRU operator using DeepCPU implementation for
/// fast inference computation on CPU machines.
class DeepCpuGruOp final : public OpKernel, public GRUBase {
 public:
  DeepCpuGruOp(const OpKernelInfo& info) : OpKernel(info), GRUBase(info) {} 

  Status Compute(OpKernelContext* context) const override;

  template <typename T>
  Status ComputeImpl(OpKernelContext& context) const;
};

}  // namespace onnxruntime
