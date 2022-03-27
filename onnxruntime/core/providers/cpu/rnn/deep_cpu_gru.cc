// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// there's no way to use a raw pointer as the copy destination with std::copy_n
// (which gsl::copy uses with span::data() which returns a raw pointer) with the 14.11 toolset
// without generating a 4996 warning. going through an iterator is way too much overhead so turn off the warning.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

#include "core/providers/cpu/rnn/deep_cpu_gru.h"

#ifdef _MSC_VER
#pragma warning(pop)
#endif
//TODO: fix the warnings
#if defined(_MSC_VER) && !defined(__clang__)
// Chance of arithmetic overflow could be reduced
#pragma warning(disable : 26451)
#endif
/*
ONNX_OPERATOR_SCHEMA(GRU)
    .SetDoc(R"DOC(
Computes an one-layer GRU. This operator is usually supported via some custom
implementation such as CuDNN.

Notations:

`X` - input tensor
`z` - update gate
`r` - reset gate
`h` - hidden gate
`t` - time step (t-1 means previous time step)
`W[zrh]` - W parameter weight matrix for update, reset, and hidden gates
`R[zrh]` - R recurrence weight matrix for update, reset, and hidden gates
`Wb[zrh]` - W bias vectors for update, reset, and hidden gates
`Rb[zrh]` - R bias vectors for update, reset, and hidden gates
`WB[zrh]` - W parameter weight matrix for backward update, reset, and hidden gates
`RB[zrh]` - R recurrence weight matrix for backward update, reset, and hidden gates
`WBb[zrh]` - W bias vectors for backward update, reset, and hidden gates
`RBb[zrh]` - R bias vectors for backward update, reset, and hidden gates
`H` - Hidden state
`num_directions` - 2 if direction == bidirectional else 1

Activation functions:

  Relu(x)                - max(0, x)
  Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})
  Sigmoid(x)             - 1/(1 + e^{-x})

  (NOTE: Below are optional)
  Affine(x)              - alpha*x + beta
  LeakyRelu(x)           - x if x >= 0 else alpha * x
  ThresholdedRelu(x)     - x if x >= alpha else 0
  ScaledTanh(x)          - alpha*Tanh(beta*x)
  HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)
  Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)
  Softsign(x)            - x/(1 + |x|)
  Softplus(x)            - log(1 + e^x)

Equations (Default: f=Sigmoid, g=Tanh):
  - zt = f(Xt*(Wz^T) + Ht-1*Rz + Wbz + Rbz)
  - rt = f(Xt*(Wr^T) + Ht-1*Rr + Wbr + Rbr)
  - ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*Rh + Rbh + Wbh) # default, when linear_before_reset = 0
  - ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*Rh + Rbh) + Wbh) # when linear_before_reset != 0
  - Ht = (1 - zt) (.) ht + zt (.) Ht-1
)DOC")
    .SinceVersion(3)
    .Attr("direction", "Specify if the RNN is forward, reverse, or bidirectional. "
                "Must be one of forward (default), reverse, or bidirectional.",
                AttributeProto::STRING,
                std::string("forward"))
    .Attr("hidden_size", "Number of neurons in the hidden layer", AttributeProto::INT, OPTIONAL)
    .Attr("activations", "A list of 2 (or 4 if bidirectional) activation functions "
                "for update, reset, and hidden gates. The activation functions must be one "
                "of the activation functions specified above. Optional: See the equations "
                "for default if not specified.",
                AttributeProto::STRINGS,
                OPTIONAL)
    .Attr("activation_alpha",
                "Optional scaling values used by some activation functions. The values "
                "are consumed in the order of activation functions, for example (f, g, h) "
                "in LSTM.",
                AttributeProto::FLOATS,
                OPTIONAL)
    .Attr("activation_beta",
                "Optional scaling values used by some activation functions. The values "
                "are consumed in the order of activation functions, for example (f, g, h) "
                "in LSTM.",
                AttributeProto::FLOATS,
                OPTIONAL)
    .Attr("output_sequence",
                "The sequence output for the hidden is optional if 0. Default 0.",
                AttributeProto::INT,
                static_cast<int64_t>(0));
    .Attr("clip", "Cell clip threshold. Clipping bounds the elements of a tensor "
                "in the range of [-threshold, +threshold] and is applied to the input "
                "of activations. No clip if not specified.", AttributeProto::FLOAT, OPTIONAL)
    .Attr("linear_before_reset", "When computing the output of the hidden gate, "
                "apply the linear transformation before multiplying by the output of the "
                "reset gate.",
                AttributeProto::INT,
                static_cast<int64_t>(0))
    .Input(0, "X",
                "The input sequences packed (and potentially padded) into one 3-D "
                "tensor with the shape of `[seq_length, batch_size, input_size]`.", "T")
    .Input(1, "W",
                "The weight tensor for the gates. Concatenation of `W[zrh]` and `WB[zrh]` "
                "(if bidirectional) along dimension 0. This tensor has shape "
                "`[num_directions, 3*hidden_size, input_size]`.", "T")
    .Input(2, "R",
                "The recurrence weight tensor. Concatenation of `R[zrh]` and `RB[zrh]` "
                "(if bidirectional) along dimension 0. This tensor has shape "
                "`[num_directions, 3*hidden_size, hidden_size]`.", "T")
    .Input(3, "B",
                "The bias tensor for the gates. Concatenation of `[Wb[zrh], Rb[zrh]]` and "
                "`[WBb[zrh], RBb[zrh]]` (if bidirectional) along dimension 0. This tensor "
                "has shape `[num_directions, 6*hidden_size]`. Optional: If not specified "
                "- assumed to be 0", "T",
                OpSchema::Optional)
    .Input(4, "sequence_lens",
                "Optional tensor specifying lengths of the sequences in a batch. "
                "If not specified - assumed all sequences in the batch to have "
                "length `seq_length`. It has shape `[batch_size]`.", "T1",
                OpSchema::Optional)
    .Input(5, "initial_h",
                "Optional initial value of the hidden. If not specified - assumed "
                "to be 0. It has shape `[num_directions, batch_size, hidden_size]`.",
                "  T", OpSchema::Optional)
    .Output(0, "Y",
                "A tensor that concats all the intermediate output values of the hidden. "
                "It has shape `[seq_length, num_directions, batch_size, hidden_size]`. ",
                "T", OpSchema::Optional);
    .Output(1, "Y_h",
                "The last output value of the hidden. It has shape "
                "`[num_directions, batch_size, hidden_size]`.", "T", OpSchema::Optional);
    .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
                "Constrain input and output types to float tensors.")
    .TypeConstraint("T1", { "tensor(int32)" }, "Constrain seq_lens to integer tensor.");
*/

namespace onnxruntime {

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    GRU,
    7,
    13,
    KernelDefBuilder().TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                                            DataTypeImpl::GetTensorType<double>()})
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int32_t>()),
    DeepCpuGruOp);

ONNX_CPU_OPERATOR_KERNEL(
    GRU,
    14,
    KernelDefBuilder().TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                                            DataTypeImpl::GetTensorType<double>()})
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int32_t>()),
    DeepCpuGruOp);

using namespace rnn::detail;


// #define DUMP_MATRIXES to provide lots of diagnostic output
#if defined(DUMP_MATRIXES)
#define DumpMatrix(...) onnxruntime::rnn::detail::DumpMatrixImpl(__VA_ARGS__)
#else
#define DumpMatrix(...) ((void)0)
#endif

Status DeepCpuGruOp::TryPackWeights(const Tensor& weights, PackedWeights& packed_weights, bool& is_packed, AllocatorPtr& alloc) {
  const auto& shape = weights.Shape();
  if (shape.NumDimensions() != 3) {
    return Status::OK();
  }

  // weights: [num_directions, 3*hidden_size, input_size]
  // recurrence weights: [num_directions, 3*hidden_size, hidden_size]
  const size_t N = static_cast<size_t>(shape[1]);
  const size_t K = static_cast<size_t>(shape[2]);

  if ((shape[0] != num_directions_) || (N != static_cast<size_t>(hidden_size_) * 3)) {
    return Status::OK();
  }

  const size_t packed_weights_size = MlasGemmPackBSize(N, K);
  if (packed_weights_size == 0) {
    return Status::OK();
  }

  size_t packed_weights_data_size = SafeInt<size_t>(packed_weights_size) * num_directions_;
  auto* packed_weights_data = alloc->Alloc(packed_weights_data_size);

  // Initialize memory to 0 as there could be some padding associated with pre-packed
  // buffer memory and we don not want it uninitialized and generate different hashes
  // if and when we try to cache this pre-packed buffer for sharing between sessions.
  memset(packed_weights_data, 0, packed_weights_data_size);

  packed_weights.buffer_ = BufferUniquePtr(packed_weights_data, BufferDeleter(alloc));
  packed_weights.buffer_size_ = packed_weights_data_size;
  packed_weights.weights_size_ = packed_weights_size;
  packed_weights.shape_ = shape;

  const auto* weights_data = weights.Data<float>();
  for (int i = 0; i < num_directions_; i++) {
    MlasGemmPackB(CblasTrans, N, K, weights_data, K, packed_weights_data);
    packed_weights_data = static_cast<uint8_t*>(packed_weights_data) + packed_weights_size;
    weights_data += N * K;
  }

  is_packed = true;
  return Status::OK();
}

static void UseSharedPrePackedBuffersImpl(std::vector<BufferUniquePtr>& prepacked_buffers,
                                          rnn::detail::PackedWeights& packed_tensor) {
  packed_tensor.buffer_ = std::move(prepacked_buffers[0]);
}

Status DeepCpuGruOp::PrePack(const Tensor& tensor, int input_idx,
                              AllocatorPtr alloc, /*out*/ bool& is_packed,
                              /*out*/ PrePackedWeights* prepacked_weights) {
  is_packed = false;

  if (tensor.IsDataType<float>()) {
    if (input_idx == 1) {
      ORT_RETURN_IF_ERROR(TryPackWeights(tensor, packed_W_, is_packed, alloc));

      bool share_prepacked_weights = (prepacked_weights != nullptr);
      if (is_packed && share_prepacked_weights) {
        prepacked_weights->buffers_.push_back(std::move(packed_W_.buffer_));
        prepacked_weights->buffer_sizes_.push_back(packed_W_.buffer_size_);
      }
    } else if (input_idx == 2) {
      ORT_RETURN_IF_ERROR(TryPackWeights(tensor, packed_R_, is_packed, alloc));

      bool share_prepacked_weights = (prepacked_weights != nullptr);
      if (is_packed && share_prepacked_weights) {
        prepacked_weights->buffers_.push_back(std::move(packed_R_.buffer_));
        prepacked_weights->buffer_sizes_.push_back(packed_R_.buffer_size_);
      }
    }
  }

  return Status::OK();
}

Status DeepCpuGruOp::UseSharedPrePackedBuffers(std::vector<BufferUniquePtr>& prepacked_buffers,
                                                int input_idx,
                                                /*out*/ bool& used_shared_buffers) {
  used_shared_buffers = false;

  if (input_idx == 1) {
    used_shared_buffers = true;
    UseSharedPrePackedBuffersImpl(prepacked_buffers, packed_W_);
  } else if (input_idx == 2) {
    used_shared_buffers = true;
    UseSharedPrePackedBuffersImpl(prepacked_buffers, packed_R_);
  }

  return Status::OK();
}

Status DeepCpuGruOp::Compute(OpKernelContext* context) const {
  const Tensor& X = *context->Input<Tensor>(0);  // inputs. [seq_length, batch_size, input_size]

  Status status;

  if (X.IsDataType<float>()) {
    status = ComputeImpl<float>(*context);
    const Tensor* W = context->Input<Tensor>(1);
    // weights. [num_directions, 3*hidden_size, input_size]
    const Tensor* R = context->Input<Tensor>(2);
    // recurrence weights. [num_directions, 3*hidden_size, hidden_size]

    const auto& W_shape = W->Shape()
    const auto& R_shape = R->Shape()

    const auto* input_weights = (W != nullptr) ? W->Data<float>() : nullptr;
    const auto* recurrent_weights = (R != nullptr) ? R->Data<float>() : nullptr;

    // spans for first direction
    const size_t input_weights_size_per_direction = W_shape[1] * W_shape[2];
    const size_t hidden_weights_size_per_direction = R_shape[1] * R_shape[2];

    GemmWeights<float> W_1(0, input_weights, input_weights_size_per_direction, packed_W_);
    GemmWeights<float> R_1(0, recurrent_weights, hidden_weights_size_per_direction, packed_R_);

    GemmWeights<float> W_2;
    GemmWeights<float> R_2;
    if (direction_ == Direction::kBidirectional) {
      W_2.Init(1, input_weights, input_weights_size_per_direction, packed_W_, nullptr);
      R_2.Init(1, recurrent_weights, hidden_weights_size_per_direction, packed_R_, nullptr);
    }

    return GRUBase::ComputeImpl<float, float>(*context, W_1, W_2, R_1, R_2);
  }
  else if (X.IsDataType<double>()) {
    /* Need to update all the helpers to support double...
    status = ComputeImpl<double>(*context); */
    ORT_NOT_IMPLEMENTED("GRU operator does not support double yet");
  } else {
    ORT_THROW("Invalid data type for GRU operator of ", X.DataType());
  }
}

}  // namespace onnxruntime