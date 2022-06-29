// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gru_base.h"
#include "uni_directional_gru.h"

namespace onnxruntime {

using namespace rnn::detail;

// #define DUMP_MATRIXES to provide lots of diagnostic output
#if defined(DUMP_MATRIXES)
#define DumpMatrix(...) ::onnxruntime::rnn::detail::DumpMatrixImpl(__VA_ARGS__)
#else
#define DumpMatrix(...) ((void)0)
#endif

template <typename T, typename WeightT>
Status GRUBase::ComputeImpl(OpKernelContext& context,
                            const rnn::detail::GemmWeights<WeightT>& W_1,
                            const rnn::detail::GemmWeights<WeightT>& W_2,
                            const rnn::detail::GemmWeights<WeightT>& R_1,
                            const rnn::detail::GemmWeights<WeightT>& R_2) const {
  concurrency::ThreadPool* thread_pool = context.GetOperatorThreadPool();

  const Tensor& X = *context.Input<Tensor>(0);  // inputs. [seq_length, batch_size, input_size]
  const Tensor& W = *context.Input<Tensor>(1);  // weights. [num_directions, 3*hidden_size, input_size]
  const Tensor& R = *context.Input<Tensor>(2);  // recurrence weights. [num_directions, 3*hidden_size, hidden_size]

  // optional
  const auto* B = context.Input<Tensor>(3);              // bias. [num_directions, 6*hidden_size]
  const auto* sequence_lens = context.Input<Tensor>(4);  // [batch_size]
  const auto* initial_h = context.Input<Tensor>(5);      // initial hidden. [num_directions, batch_size, hidden_size]

  auto& X_shape = X.Shape();

  int seq_length = gsl::narrow<int>(X_shape[0]);
  int batch_size = gsl::narrow<int>(X_shape[1]);
  int input_size = gsl::narrow<int>(X_shape[2]);

  auto status = ValidateCommonRnnInputs(X, W.Shape(), R.Shape(), B, 3, sequence_lens, initial_h, num_directions_, hidden_size_);
  ORT_RETURN_IF_ERROR(status);

  // GRU outputs are optional but must be in the same order
  TensorShape Y_dims{seq_length, num_directions_, batch_size, hidden_size_};
  Tensor* Y = context.Output(/*index*/ 0, Y_dims);

  TensorShape Y_h_dims{num_directions_, batch_size, hidden_size_};
  Tensor* Y_h = context.Output(/*index*/ 1, Y_h_dims);

  // Reset output and return if max sequence length is 0
  if (sequence_lens != nullptr) {
    int32_t max_sequence_length = *std::max_element(sequence_lens->Data<int32_t>(), sequence_lens->Data<int32_t>() + sequence_lens->Shape().Size());
    if (max_sequence_length == 0) {
      if (Y != nullptr) std::fill_n(Y->MutableData<T>(), Y_dims.Size(), T{});
      if (Y_h != nullptr) std::fill_n(Y_h->MutableData<T>(), Y_h_dims.Size(), T{});
      return Status::OK();
    }
  }

  AllocatorPtr alloc;
  status = context.GetTempSpaceAllocator(&alloc);
  ORT_RETURN_IF_ERROR(status);

  gsl::span<const T> bias = B != nullptr ? B->DataAsSpan<T>() : gsl::span<const T>();

  // spans for first direction
  const size_t bias_size_per_direction = 6 * hidden_size_;

  gsl::span<const T> bias_1 = bias.empty() ? bias : bias.subspan(0, bias_size_per_direction);

  gsl::span<const T> input = X.DataAsSpan<T>();
  gsl::span<const int> sequence_lens_span = sequence_lens != nullptr ? sequence_lens->DataAsSpan<int>()
                                                                     : gsl::span<const int>();

  const size_t initial_hidden_size_per_direction = batch_size * hidden_size_;
  gsl::span<const T> initial_hidden = initial_h != nullptr ? initial_h->DataAsSpan<T>() : gsl::span<const T>();
  gsl::span<const T> initial_hidden_1 = initial_hidden.empty()
                                            ? initial_hidden
                                            : initial_hidden.subspan(0, initial_hidden_size_per_direction);

  // output shape is [seq_length, num_directions, batch_size, hidden_size]
  // so it's not a case of all the output for one direction being first.
  // due to that we can only easily check that the end of the output for each direction is valid.
  const size_t output_size = Y != nullptr ? Y->Shape().Size() : 0;
  const size_t per_direction_offset = batch_size * hidden_size_;
  gsl::span<T> output = Y != nullptr ? Y->MutableDataAsSpan<T>() : gsl::span<T>();
  gsl::span<T> output_1 = output.empty()
                              ? output
                              : output.subspan(0, output_size - (num_directions_ - 1) * per_direction_offset);

  // UniDirectionalGru needs somewhere to write output, so even if we aren't returning Y_h
  // we provide an appropriately sized buffer for that purpose.
  const size_t hidden_output_size_per_direction = batch_size * hidden_size_;
  IAllocatorUniquePtr<T> local_hidden_output;
  gsl::span<T> hidden_output =
      Y_h ? Y_h->MutableDataAsSpan<T>()
          : Allocate<T>(alloc, hidden_output_size_per_direction * num_directions_, local_hidden_output);

  gsl::span<T> hidden_output_1 = hidden_output.subspan(0, hidden_output_size_per_direction);

  if (direction_ == Direction::kBidirectional) {
    // spans for second direction
    gsl::span<const T> bias_2 = bias.empty() ? bias : bias.subspan(bias_size_per_direction, bias_size_per_direction);

    gsl::span<const T> initial_hidden_2 = initial_hidden.empty()
                                              ? initial_hidden
                                              : initial_hidden.subspan(initial_hidden_size_per_direction,
                                                                       initial_hidden_size_per_direction);
    gsl::span<T> output_2 = output.empty()
                                ? output
                                : output.subspan(per_direction_offset, output_size - per_direction_offset);

    gsl::span<T> hidden_output_2 = hidden_output.subspan(hidden_output_size_per_direction,
                                                         hidden_output_size_per_direction);

    gru::UniDirectionalGru<T> fw(alloc, seq_length, batch_size, input_size, hidden_size_,
                                    linear_before_reset_ != 0, Direction::kForward, bias_1, initial_hidden_1,
                                    activation_funcs_.Entries()[0],
                                    activation_funcs_.Entries()[1],
                                    clip_, thread_pool);
    fw.Compute(input, sequence_lens_span, num_directions_, W_1, R_1,
               output_1, hidden_output_1);

    gru::UniDirectionalGru<T> bw(alloc, seq_length, batch_size, input_size, hidden_size_,
                                    linear_before_reset_ != 0, Direction::kReverse, bias_2, initial_hidden_2,
                                    activation_funcs_.Entries()[2],
                                    activation_funcs_.Entries()[3],
                                    clip_, thread_pool);
    bw.Compute(input, sequence_lens_span, num_directions_, W_2, R_2,
               output_2, hidden_output_2);
  } else {
    gru::UniDirectionalGru<T> gru_p(alloc, seq_length, batch_size, input_size, hidden_size_,
                                       linear_before_reset_ != 0, direction_, bias_1, initial_hidden_1,
                                       activation_funcs_.Entries()[0],
                                       activation_funcs_.Entries()[1],
                                       clip_, thread_pool);
    gru_p.Compute(input, sequence_lens_span, num_directions_, W_1, R_1,
                  output_1, hidden_output_1);
  }

  if (!output.empty())
    DumpMatrix("Y", output.data(), seq_length * num_directions_ * batch_size, hidden_size_);

  DumpMatrix("Y_h", hidden_output.data(), num_directions_ * batch_size, hidden_size_);

  return Status::OK();
}

template Status GRUBase::ComputeImpl<float, float>(OpKernelContext& context,
                                                    const rnn::detail::GemmWeights<float>& W_1,
                                                    const rnn::detail::GemmWeights<float>& W_2,
                                                    const rnn::detail::GemmWeights<float>& R_1,
                                                    const rnn::detail::GemmWeights<float>& R_2) const;

template Status GRUBase::ComputeImpl<float, uint8_t>(OpKernelContext& context,
                                                      const rnn::detail::GemmWeights<uint8_t>& W_1,
                                                      const rnn::detail::GemmWeights<uint8_t>& W_2,
                                                      const rnn::detail::GemmWeights<uint8_t>& R_1,
                                                      const rnn::detail::GemmWeights<uint8_t>& R_2) const;

}  // namespace onnxruntime