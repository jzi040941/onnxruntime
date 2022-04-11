// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "uni_directional_gru.h"

#include "core/platform/threadpool.h"

namespace onnxruntime {
namespace gru {

// #define DUMP_MATRIXES to provide lots of diagnostic output
#if defined(DUMP_MATRIXES)
#define DumpMatrix(...) onnxruntime::rnn::detail::DumpMatrixImpl(__VA_ARGS__)
#else
#define DumpMatrix(...) ((void)0)
#endif

using namespace rnn::detail;


template <typename T>
UniDirectionalGru<T>::UniDirectionalGru(AllocatorPtr allocator,
                                        const int seq_length,
                                        const int batch_size,
                                        const int input_size,
                                        const int hidden_size,
                                        const bool linear_before_reset,
                                        Direction direction,
                                        const gsl::span<const T>& bias,
                                        const gsl::span<const T>& initial_hidden_state,
                                        const ActivationFuncs::Entry& activation_func_f,
                                        const ActivationFuncs::Entry& activation_func_g,
                                        const float clip, onnxruntime::concurrency::ThreadPool* ttp)
    : allocator_(allocator),
      seq_length_(seq_length),
      batch_size_(batch_size),
      input_size_(input_size),
      hidden_size_(hidden_size),
      linear_before_reset_(linear_before_reset),
      clip_(clip),
      direction_(direction),
      use_bias_(!bias.empty()),
      ttp_(ttp) {
  clip_with_bias_ptr_ = use_bias_ ? deepcpu::clip_add_bias : deepcpu::clip_ignore_bias;

  // setup activation function pointers and alpha/beta values to use with them
  reset_gate_ = deepcpu::GruResetGateFuncByName(activation_func_f.name);
  update_gate_ = deepcpu::ActivationFuncByName(activation_func_f.name);
  output_gate_ = deepcpu::GruOutputGateFuncByName(activation_func_g.name);

  zr_alpha_ = activation_func_f.alpha;
  zr_beta_ = activation_func_f.beta;
  h_alpha_ = activation_func_g.alpha;
  h_beta_ = activation_func_g.beta;

  AllocateBuffers();

  if (use_bias_) {
    auto bias_Wz = bias.subspan(0 * hidden_size_, hidden_size_);
    auto bias_Wr = bias.subspan(1 * hidden_size_, hidden_size_);
    auto bias_Wo = bias.subspan(2 * hidden_size_, hidden_size_);
    auto bias_Rz = bias.subspan(3 * hidden_size_, hidden_size_);
    auto bias_Rr = bias.subspan(4 * hidden_size_, hidden_size_);
    auto bias_Ro = bias.subspan(5 * hidden_size_, hidden_size_);

    // add Wb[zr] and Rb[zr] and replicate so we have batch_size_ copies of the result
    auto combine_and_replicate = [&](gsl::span<const T>& bias_w,
                                     gsl::span<const T>& bias_r,
                                     gsl::span<T>& output) {
      // add once
      for (int i = 0; i < hidden_size_; ++i) {
        output[i] = bias_w[i] + bias_r[i];
      }

      // replicate what we just wrote to the start of the output span so we have batch_size_ copies
      auto values = output.cbegin();
      ORT_IGNORE_RETURN_VALUE(RepeatVectorToConstructArray(values, values + hidden_size_,
                                                           output.begin() + hidden_size_,  // skip the first batch
                                                           batch_size_ - 1));              // and replicate batch size - 1 times
    };

    // we can always combine the z and r weights
    combine_and_replicate(bias_Wz, bias_Rz, batched_bias_WRz_);
    combine_and_replicate(bias_Wr, bias_Rr, batched_bias_WRr_);

    // how we treat the h weight depends on whether linear_before_reset_ is set
    if (linear_before_reset_) {
      // need to replicate Wb[o] and Rb[o] separately
      ORT_IGNORE_RETURN_VALUE(RepeatVectorToConstructArray(bias_Wo.cbegin(), bias_Wo.cend(), batched_bias_Wh_.begin(), batch_size_));
      ORT_IGNORE_RETURN_VALUE(RepeatVectorToConstructArray(bias_Ro.cbegin(), bias_Ro.cend(), batched_bias_Rh_.begin(), batch_size_));
    } else {
      combine_and_replicate(bias_Wo, bias_Ro, batched_bias_WRh_);
    }
  }

  if (!initial_hidden_state.empty()) {
    gsl::copy(initial_hidden_state, batched_hidden0_);
  }
}

template <typename T>
template <typename WeightT>
void UniDirectionalLstm<T>::AllocateQuantizeBuffers(int max_sequence_length) {
  // Can not specialize on WeightT without specify T explicitly, so use sizeof
  if constexpr(sizeof(WeightT) == 1) {
    const int hidden_size_x4 = 4 * hidden_size_;
    const int total_rows = max_sequence_length * batch_size_;

    int input_or_a_size = std::max(total_rows * input_size_, batch_size_ * hidden_size_);
    quantized_input_or_a_ = Allocate(allocator_, input_or_a_size, quantized_input_or_a_ptr_, false);
    quantized_C_buffer_ = Allocate(allocator_, batch_size_ * hidden_size_x4, quantized_C_buffer_ptr_, false);
  }
}

template <typename T>
template <typename WeightT>
void UniDirectionalGru<T>::Compute(const gsl::span<const T>& inputs_arg,
                                   const gsl::span<const int>& sequence_lengths_arg,
                                   const int num_directions,
                                   const GemmWeights<WeightT>& input_weights,
                                   const GemmWeights<WeightT>& recurrent_weights,
                                   gsl::span<T>& outputs,
                                   gsl::span<T>& final_hidden_state) {
  using span_T_const_iter = typename gsl::span<T>::const_iterator;
  using span_T_iter = typename gsl::span<T>::iterator;

  // copy inputs_arg as we may change it to point to inputs_reverse_
  gsl::span<const T> inputs = inputs_arg;
  gsl::span<const int> sequence_lengths = sequence_lengths_arg;

  // if sequence lengths weren't provided, use internal array and init all to seq_length
  if (sequence_lengths.empty()) {
    sequence_lengths_ = Allocate(allocator_, batch_size_, sequence_lengths_ptr_, true, seq_length_);
    sequence_lengths = sequence_lengths_;
  }

  DumpMatrix("Inputs", inputs.data(), seq_length_ * batch_size_, input_size_);
  //DumpMatrix("input_weights", input_weights.data(), 3 * hidden_size_, input_size_);
  //DumpMatrix("recurrent_weights", recurrent_weights.data(), 3 * hidden_size_, hidden_size_);

  gsl::span<const WeightT> recurrent_weightsZR = recurrent_weights.subspan(0, 2 * hidden_size_ * hidden_size_);
  gsl::span<const WeightT> recurrent_weightsH = recurrent_weights.subspan(2 * hidden_size_ * hidden_size_, hidden_size_ * hidden_size_);

  gsl::span<T> original_outputs = outputs;
  const bool output_sequence = !outputs.empty();

  if (direction_ == kReverse) {
    ReverseSequence(inputs, inputs_reverse_, sequence_lengths, seq_length_, batch_size_, input_size_, 1, ttp_);
    // DumpMatrix("Reversed inputs", inputs_reverse_.data(), seq_length_ * batch_size_, input_size_);

    inputs = inputs_reverse_;

    if (output_sequence) {
      outputs = outputs_reverse_;
    }
  }

  // Calculate the max and min length
  int32_t max_sequence_length = *std::max_element(sequence_lengths.cbegin(), sequence_lengths.cend());
  int32_t min_sequence_length = std::min(seq_length_, *std::min_element(sequence_lengths.cbegin(),
                                                                        sequence_lengths.cend()));

  const int hidden_size_x2 = 2 * hidden_size_;
  const int hidden_size_x3 = 3 * hidden_size_;
  const int total_rows = max_sequence_length * batch_size_;

  float alpha = 1.0f;

  // apply weights to all the inputs
  ComputeGemm(total_rows, hidden_size_x3, input_size_, alpha,
              inputs.cbegin(), inputs.cend(),
              input_weights, 0.f,
              outputZRH_.begin(), outputZRH_.end(),
              hidden_size_x3, 
              quantized_input_or_a_.begin(), 
              nullptr,
              ttp_);

  DumpMatrix("inputs with weights applied", outputZRH_.data(), seq_length_ * batch_size_ * 3, hidden_size_);

  // output shape is [seq_length, num_directions, batch_size, hidden_size]
  // if we are doing 2 directions and this is the forward pass we're writing to the real output so
  // need to include num_directions in the step length.
  // we do not need to do that if there are two directions and we're doing the backwards pass as we
  // are writing to a temporary buffer (as outputs == outputs_reverse_) which is later copied
  // to the real output by ReverseSequence. this later copy includes num_directions in the step length.
  int output_step_length = batch_size_ * hidden_size_;
  if (direction_ == kForward && num_directions == 2)
    output_step_length = 2 * batch_size_ * hidden_size_;

  // convenience end iterators we use in the loops below to detect any bounds issues
  span_T_const_iter batched_bias_WRz_local_end = batched_bias_WRz_.cend();
  span_T_const_iter batched_bias_WRr_local_end = batched_bias_WRr_.cend();
  span_T_const_iter batched_bias_Wh_local_end = batched_bias_Wh_.cend();
  span_T_const_iter batched_bias_Rh_local_end = batched_bias_Rh_.cend();
  span_T_const_iter batched_bias_WRh_local_end = batched_bias_WRh_.cend();

  size_t out_added_offset;

  span_T_const_iter prev_Ht = batched_hidden0_.cbegin();  // Ht-1
  span_T_const_iter prev_Ht_end = batched_hidden0_.cend();
  span_T_iter cur_h_local = cur_h_.begin();
  span_T_iter cur_h_local_end = cur_h_.end();

  span_T_const_iter batched_bias_WRz_local{};
  span_T_const_iter batched_bias_WRr_local{};
  span_T_const_iter batched_bias_WRh_local{};
  span_T_const_iter batched_bias_Wh_local{};
  span_T_const_iter batched_bias_Rh_local{};

  if (use_bias_) {
    batched_bias_WRz_local = batched_bias_WRz_.cbegin();
    batched_bias_WRr_local = batched_bias_WRr_.cbegin();

    if (linear_before_reset_) {
      batched_bias_Wh_local = batched_bias_Wh_.cbegin();
      batched_bias_Rh_local = batched_bias_Rh_.cbegin();
    } else {
      batched_bias_WRh_local = batched_bias_WRh_.cbegin();
    }
  }

  {
    // Enter a parallel section encompassing the kernels invoked
    // below.  This lets the runtime system amortize loop entry/exit
    // costs over a series of short kernels, and promotes cache
    // affinity between iterations of successive loops.
    onnxruntime::concurrency::ThreadPool::ParallelSection ps(ttp_);

    // for each item in sequence run all calculations
    for (int step = 0; step < max_sequence_length; step++) {
#if defined(DUMP_MATRIXES)
      const std::string seqno_str = " [seqno=" + std::to_string(step) + "]";
#endif
      DumpMatrix("Ht-1" + seqno_str, &*prev_Ht, batch_size_, hidden_size_);

      out_added_offset = (step * batch_size_) * hidden_size_x3;

      // calculate Ht-1*R[zr], and add to the weighted inputs that are in outputZRH_
      // Ht-1 * R[zr] + Xt*(W[zr]^T)
      ComputeGemm(batch_size_, hidden_size_x2, hidden_size_, alpha,
                  prev_Ht, prev_Ht_end,
                  hidden_size_,
                  recurrent_weightsZR.cbegin(), recurrent_weightsZR.cend(),
                  hidden_size_, 1.f,  // beta == 1 so we add existing values in outputZRH_
                  outputZRH_.begin() + out_added_offset, outputZRH_.end(),
                  hidden_size_x3, ttp_);

      DumpMatrix("Ht-1 * R[zr] + Xt*(W[zr]^T)" + seqno_str,
                 outputZRH_.data() + out_added_offset, batch_size_, hidden_size_x2, 0, hidden_size_x3);

      if (linear_before_reset_) {
        // copy Rbh to linear output
        if (use_bias_) {
          gsl::copy(batched_bias_Rh_.subspan(batched_bias_Rh_local - batched_bias_Rh_.begin(),
                                             batched_bias_Rh_local_end - batched_bias_Rh_local),
                    linear_output_);
        }

        // compute Ht-1 * (Rh^T) + Rbh
        ComputeGemm(batch_size_, hidden_size_, hidden_size_, alpha,
                    prev_Ht, prev_Ht_end,  // Ht-1
                    hidden_size_,
                    recurrent_weightsH.cbegin(), recurrent_weightsH.cend(),  // Rh^T
                    hidden_size_,
                    use_bias_ ? 1.f : 0.f,  // don't add values in linear_output_ if no bias input
                    linear_output_.begin(),
                    linear_output_.end(),  // pre: Rbh if use_bias_, post:output
                    hidden_size_, ttp_);

        DumpMatrix("Ht-1 * (Rh^T) + Rbh " + seqno_str, linear_output_.data(), batch_size_, hidden_size_);
      }

      // 1st Set Of Activations
      for (int r = 0; r < batch_size_; r++) {
        const T* p_bias_r = use_bias_ ? SafeRawConstPointer<T>(batched_bias_WRr_local + r * hidden_size_,
                                                               batched_bias_WRr_local_end, hidden_size_)
                                      : nullptr;

        // initialize p_rt with input to calculate rt. outputZRH_ has Xt*(Wr^T) + Ht-1*(Rr^T).
        T* p_rt = SafeRawPointer(outputZRH_, out_added_offset + r * hidden_size_x3 + hidden_size_, hidden_size_);

        // add the bias and clip. post: p_rt == Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr
        clip_with_bias_ptr_(clip_, p_bias_r, p_rt, hidden_size_);

        if (linear_before_reset_) {
          // p_linear_output = Ht-1 * (Rh^T) + Rbh
          T* p_linear_output = SafeRawPointer<T>(linear_output_, r * hidden_size_, hidden_size_);
          T* p_cur_h = SafeRawPointer<T>(cur_h_local + r * hidden_size_, cur_h_local_end, hidden_size_);

          // calculate rt in-place [p_rt = f(p_rt)]
          // calculate rt (.) (Ht-1 * (Rh^T) + Rbh) using p_linear_output. write to p_cur_h
          reset_gate_(p_linear_output, p_rt, p_cur_h, hidden_size_, zr_alpha_, zr_beta_);

        } else {
          const T* p_prev_Ht = SafeRawConstPointer<T>(prev_Ht + r * hidden_size_, prev_Ht_end, hidden_size_);
          T* p_cur_h = SafeRawPointer<T>(cur_h_local + r * hidden_size_, cur_h_local_end, hidden_size_);

          // calculate rt in-place [p_rt = f(p_rt)]
          // calculate rt (.) Ht-1 using p_prev_Ht, and write to p_cur_h
          reset_gate_(p_prev_Ht, p_rt, p_cur_h, hidden_size_, zr_alpha_, zr_beta_);
        }
      }

#if defined(DUMP_MATRIXES)
      std::string label = linear_before_reset_ ? "rt (.) (Ht-1 * (Rh^T) + Rbh)" : "rt (.) Ht-1";
#endif
      DumpMatrix(label + seqno_str, &*cur_h_local, batch_size_, hidden_size_);

      if (linear_before_reset_) {
        // input contains rt (.) (Ht-1*(Rh^T) + Rbh)
        auto input = cur_h_local;
        // out_H currently contains Xt*(W[zrh]^T).
        auto out_H = outputZRH_.begin() + out_added_offset;

        for (int r = 0; r < batch_size_; r++) {
          // skip over the inputs with Z and R weights
          out_H += hidden_size_x2;
          for (int h = 0; h < hidden_size_; ++h) {
            *out_H += *input;
            ++out_H;
            ++input;
          }
        }
      } else {
#if defined(DUMP_MATRIXES)
        label += " * Rh^T";
#endif

        // out_H currently contains Xt*(Wh^T).
        auto out_H = outputZRH_.begin() + out_added_offset + hidden_size_x2;

        // Calculate Xt*(Wh^T) + rt (.) Ht-1 * Rh
        ComputeGemm(batch_size_, hidden_size_, hidden_size_, alpha,
                    cur_h_local, cur_h_local_end,  // rt (.) Ht-1
                    hidden_size_,
                    recurrent_weightsH.cbegin(), recurrent_weightsH.cend(),  // Rh^T
                    hidden_size_, 1.f,                                       // beta == 1 to add Xt*(Wh^T) from out_H
                    out_H, outputZRH_.end(),
                    hidden_size_x3, ttp_);
      }

      DumpMatrix("Xt*(Wh^T) + (" + label + ")" + seqno_str, outputZRH_.data() + out_added_offset,
                 batch_size_, hidden_size_, hidden_size_x2, hidden_size_x3);

      //2nd Set of Activations
      span_T_iter output;
      span_T_iter output_end;
      if (output_sequence) {
        output = outputs.begin() + step * output_step_length;
        output_end = outputs.end();

      } else {
        output = final_hidden_state.begin();
        output_end = final_hidden_state.end();
      }

      for (int r = 0; r < batch_size_; r++) {
        if (step >= min_sequence_length && step >= sequence_lengths[r]) {
          // if we need output for every step,
          // or we need to set prev_Ht for an empty sequence to avoid warnings about using uninitialized values
          if (output_sequence || (step == 0 && sequence_lengths[r] == 0)) {
            auto fill_output = output + r * hidden_size_;
            std::fill_n(&*fill_output, hidden_size_, T{});
          }

          continue;
        }

        const T* p_bias_z = use_bias_ ? SafeRawConstPointer<T>(batched_bias_WRz_local,
                                                               batched_bias_WRz_local_end, hidden_size_)
                                      : nullptr;

        // initialize p_zt with Xt*(Wz^T) + Ht-1*(Rz^T), which is most of the input to calculate zt:
        T* p_zt = SafeRawPointer<T>(outputZRH_, out_added_offset + r * hidden_size_x3, hidden_size_);

        // using p_zt, add bias and clip in-place
        clip_with_bias_ptr_(clip_, p_bias_z, p_zt, hidden_size_);

        // calculate zt in-place. p_zt = f(p_zt)
        update_gate_(p_zt, hidden_size_, zr_alpha_, zr_beta_);

        DumpMatrix("zt[" + std::to_string(r) + "]" + seqno_str, p_zt, 1, hidden_size_);

        const T* p_bias_h = nullptr;
        if (use_bias_) {
          if (linear_before_reset_) {
            // Wbh
            p_bias_h = SafeRawConstPointer<T>(batched_bias_Wh_local + r * hidden_size_,
                                              batched_bias_Wh_local_end, hidden_size_);

          } else {
            // Wbh + Wrh
            p_bias_h = SafeRawConstPointer<T>(batched_bias_WRh_local + r * hidden_size_,
                                              batched_bias_WRh_local_end, hidden_size_);
          }
        }

        // setup p_ht with input to calculate ht
        // p_ht = Xt*(Wh^T) + (rt (.) Ht-1 * Rh^T)          #  linear_before_reset_ == false
        //      = Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh))  #  linear_before_reset_ == true
        T* p_ht = SafeRawPointer<T>(outputZRH_, out_added_offset + r * hidden_size_x3 + hidden_size_x2, hidden_size_);

        // add Wbh [and Wrh] and clip
        clip_with_bias_ptr_(clip_, p_bias_h, p_ht, hidden_size_);  // post: p_ht == input to g() for calculating ht

        DumpMatrix("ht input [" + std::to_string(r) + "]" + seqno_str, p_ht, 1, hidden_size_);

        const T* p_prev_Ht = SafeRawConstPointer<T>(prev_Ht + r * hidden_size_, prev_Ht_end, hidden_size_);
        T* p_Ht = SafeRawPointer<T>(output + r * hidden_size_, output_end, hidden_size_);

        // calculate ht = g(p_ht) and write in-place to p_ht
        // calculate Ht = (1 - zt) (.) ht + zt (.) Ht-1 and write to p_Ht
        output_gate_(p_ht, p_zt, p_prev_Ht, p_Ht, hidden_size_, h_alpha_, h_beta_);  // calculate ht and Ht
      }

      DumpMatrix("output" + seqno_str, &*output, batch_size_, hidden_size_);

      prev_Ht = output;
      prev_Ht_end = output_end;
    }
  }  // End parallel section

  // copy last output to final_hidden_state
  for (int i = 0; i < batch_size_; i++) {
    const int seq_len = sequence_lengths[i];
    if (output_sequence) {
      if (seq_len == 0) {
        auto final_hidden_state_dst = final_hidden_state.begin() + i * hidden_size_;
        std::fill_n(&*final_hidden_state_dst, hidden_size_, T{});
      } else {
        auto src = outputs.subspan((seq_len - 1) * output_step_length + i * hidden_size_, hidden_size_);
        auto dest = final_hidden_state.subspan(i * hidden_size_, hidden_size_);
        gsl::copy(src, dest);
      }
    }
  }

  // zero any values beyond the evaluated steps if the maximum explicit sequence length we saw (max_sequence_length)
  // was shorter than the maximum possible sequence length (seq_length_)
  if (output_sequence && max_sequence_length < seq_length_) {
    if (output_step_length == batch_size_ * hidden_size_) {  // contiguous
      const auto span_to_zero = outputs.subspan(
          max_sequence_length * output_step_length, (seq_length_ - max_sequence_length) * output_step_length);
      std::fill_n(&*span_to_zero.begin(), span_to_zero.size(), T{});
    } else {
      for (int i = max_sequence_length; i < seq_length_; ++i) {  // non-contiguous
        const auto span_to_zero = outputs.subspan(i * output_step_length, batch_size_ * hidden_size_);
        std::fill_n(&*span_to_zero.begin(), span_to_zero.size(), T{});
      }
    }
  }

  if (output_sequence && direction_ == kReverse) {
    ReverseSequence<T>(outputs, original_outputs,
                       sequence_lengths, seq_length_,
                       batch_size_, hidden_size_, num_directions, ttp_);
  }
}

template <typename T>
void UniDirectionalGru<T>::AllocateBuffers() {
  cur_h_ = Allocate(allocator_, hidden_size_ * batch_size_, cur_h_ptr_);
  batched_hidden0_ = Allocate(allocator_, batch_size_ * hidden_size_, batched_hidden0_ptr_, true);

  if (use_bias_) {
    batched_bias_WRz_ = Allocate(allocator_, batch_size_ * hidden_size_, batched_bias_WRz_ptr_);
    batched_bias_WRr_ = Allocate(allocator_, batch_size_ * hidden_size_, batched_bias_WRr_ptr_);

    if (linear_before_reset_) {
      batched_bias_Wh_ = Allocate(allocator_, batch_size_ * hidden_size_, batched_bias_Wh_ptr_);
      batched_bias_Rh_ = Allocate(allocator_, batch_size_ * hidden_size_, batched_bias_Rh_ptr_);
    } else {
      batched_bias_WRh_ = Allocate(allocator_, batch_size_ * hidden_size_, batched_bias_WRh_ptr_);
    }
  }

  if (linear_before_reset_) {
    linear_output_ = Allocate(allocator_, batch_size_ * hidden_size_, linear_output_ptr_);
  }

  auto batch_times_seq_length = batch_size_ * seq_length_;

  outputZRH_ = Allocate(allocator_, hidden_size_ * 3 * batch_times_seq_length, outputZRH_ptr_, true);

  if (direction_ == kReverse) {
    inputs_reverse_ = Allocate(allocator_, batch_times_seq_length * input_size_, inputs_reverse_ptr_);
    outputs_reverse_ = Allocate(allocator_, batch_times_seq_length * hidden_size_, outputs_reverse_ptr_);
  }
}

template class UniDirectionalGru<float>;
template void UniDirectionalGru<float>::Compute<float>(const gsl::span<const float>& inputs_arg,
                                   const gsl::span<const int>& sequence_lengths_arg,
                                   const int num_directions,
                                   const GemmWeights<float>& input_weights,
                                   const GemmWeights<float>& recurrent_weights,
                                   gsl::span<float>& outputs,
                                   gsl::span<float>& final_hidden_state);

template void UniDirectionalGru<float>::Compute<uint8_t>(const gsl::span<const float>& inputs_arg,
                                   const gsl::span<const int>& sequence_lengths_arg,
                                   const int num_directions,
                                   const GemmWeights<uint8_t>& input_weights,
                                   const GemmWeights<uint8_t>& recurrent_weights,
                                   gsl::span<float>& outputs,
                                   gsl::span<float>& final_hidden_state);


}  // namespace gru 
}  // namespace onnxruntime