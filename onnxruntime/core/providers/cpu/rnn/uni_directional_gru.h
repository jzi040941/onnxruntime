// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/providers/cpu/rnn/rnn_helpers.h"

namespace onnxruntime {
namespace gru {

using namespace rnn::detail;

template <typename T>
class UniDirectionalGru {
 public:
  UniDirectionalGru(AllocatorPtr allocator, int seq_length, int batch_size, int input_size, int hidden_size,
                    bool linear_before_reset, Direction direction, const gsl::span<const T>& bias,
                    const gsl::span<const T>& initial_hidden_state, const ActivationFuncs::Entry& activation_func_f,
                    const ActivationFuncs::Entry& activation_func_g, float clip,
                    onnxruntime::concurrency::ThreadPool* ttp);
                    
  template <typename WeightT>
  void Compute(const gsl::span<const T>& inputs, const gsl::span<const int>& sequence_lengths, int num_directions,
               const GemmWeights<WeightT>& input_weights, const GemmWeights<WeightT>& recurrent_weights,
               gsl::span<T>& outputs, gsl::span<T>& final_hidden_state);

  ~UniDirectionalGru() = default;

 private:
  AllocatorPtr allocator_;

  int seq_length_;
  int batch_size_;
  int input_size_;
  int hidden_size_;
  bool linear_before_reset_;

  const float clip_;

  Direction direction_;
  bool use_bias_;

  IAllocatorUniquePtr<T> outputZRH_ptr_;
  gsl::span<T> outputZRH_;

  IAllocatorUniquePtr<T> cur_h_ptr_;
  IAllocatorUniquePtr<T> batched_hidden0_ptr_;
  IAllocatorUniquePtr<int> sequence_lengths_ptr_;
  gsl::span<T> cur_h_;
  gsl::span<T> batched_hidden0_;
  gsl::span<int> sequence_lengths_;

  // Wb[zr] and Rb[zr] can always be added together upfront, and repeated to match the batch size for
  // faster GEMM calculations, so these two members are all the
  // Wb[z] + Rb[z] values added together, repeated batch_size_ times
  IAllocatorUniquePtr<T> batched_bias_WRz_ptr_, batched_bias_WRr_ptr_;
  gsl::span<T> batched_bias_WRz_, batched_bias_WRr_;

  // Wbh and Rbh can only be combined upfront if linear_before_reset_ is false
  IAllocatorUniquePtr<T> batched_bias_WRh_ptr_;
  gsl::span<T> batched_bias_WRh_;

  // if linear_before_reset_ is true, we need to setup Wbh and Rbh separately
  IAllocatorUniquePtr<T> batched_bias_Wh_ptr_, batched_bias_Rh_ptr_;
  gsl::span<T> batched_bias_Wh_, batched_bias_Rh_;

  IAllocatorUniquePtr<T> linear_output_ptr_;
  gsl::span<T> linear_output_;

  IAllocatorUniquePtr<T> inputs_reverse_ptr_;
  IAllocatorUniquePtr<T> outputs_reverse_ptr_;
  gsl::span<T> inputs_reverse_;
  gsl::span<T> outputs_reverse_;

  deepcpu::ClipWithBiasFuncPtr clip_with_bias_ptr_{};

  float zr_alpha_{};
  float zr_beta_{};
  float h_alpha_{};
  float h_beta_{};

  deepcpu::GruResetGateFuncPtr reset_gate_{};
  deepcpu::ActivationFuncPtr update_gate_{};
  deepcpu::GruOutputGateFuncPtr output_gate_{};

  void AllocateBuffers();

  onnxruntime::concurrency::ThreadPool* ttp_;

  // Quantized operation related allocation members
  template <typename WeightT>
  void AllocateQuantizeBuffers(int max_sequence_length);

  // Buffer shared for quantized input whole, and quantized a each sequence step
  IAllocatorUniquePtr<uint8_t> quantized_input_or_a_ptr_;
  gsl::span<uint8_t> quantized_input_or_a_;

  IAllocatorUniquePtr<int32_t> quantized_C_buffer_ptr_;
  gsl::span<int32_t> quantized_C_buffer_;
};

}  // namespace gru
}  // namespace onnxruntime
