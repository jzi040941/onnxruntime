// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cpu/nn/conv.h"
#include "contrib_ops/cpu/fused_activation.h"

namespace onnxruntime {
namespace contrib {

class MobileConv final : public Conv<float> {
 public:
  MobileConv(const OpKernelInfo& info) : Conv<float>(info) {
    ORT_ENFORCE(GetFusedActivationAttr(info, activation_).IsOK());
  }

  Status Compute(OpKernelContext* context) const override;

};

}  // namespace contrib
}  // namespace onnxruntime
