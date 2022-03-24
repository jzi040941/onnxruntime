// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "mobile_ops.h"
#include "core/mlas/inc/mlas.h"
#include "core/common/safeint.h"

namespace onnxruntime {
using ConvPadVector = ConvAttributes::ConvPadVector;
namespace contrib {

Status MobileConv::Compute(OpKernelContext* context) const {
  size_t num_inputs = OpKernel::Node().InputDefs().size();
  ORT_ENFORCE(num_inputs == 4, "Expected 4 tensor inputs. But Got " + std::to_string(num_inputs));
  const Tensor* X = context->Input<Tensor>(0);
  const Tensor* W = context->Input<Tensor>(1);
  const Tensor* B = context->Input<Tensor>(2);
  const Tensor* Sum = context->Input<Tensor>(3);
  return ComputeBase(context, X, W, B, Sum);
}

#define ONNX_CPU_OPERATOR_TYPED_Mobile_KERNEL(name, ver, type, builder, ...) \
  ONNX_OPERATOR_TYPED_KERNEL_EX(name, kMSMobileDomain, ver, type, kCpuExecutionProvider, builder, __VA_ARGS__)
ONNX_CPU_OPERATOR_TYPED_Mobile_KERNEL(
    FusedConv,
    1,
    float,
    KernelDefBuilder()
        .MayInplace(3, 0)
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    MobileConv);

}  // namespace contrib
}  // namespace onnxruntime
