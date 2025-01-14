/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "src/fastertransformer/layers/attention_layers_fp8/TensorParallelDecoderSelfAttentionFP8Layer.h"

namespace fastertransformer {

template<typename T1, typename T2>
TensorParallelDecoderSelfAttentionFP8Layer<T1, T2>::TensorParallelDecoderSelfAttentionFP8Layer(
    size_t           head_num,
    size_t           size_per_head,
    size_t           rotary_embedding_dim,
    bool             neox_rotary_style,
    size_t           d_model,
    float            q_scaling,
    NcclParam        tensor_para,
    cudaStream_t     stream,
    cublasMMWrapper* cublas_wrapper,
    IAllocator*      allocator,
    bool             is_free_buffer_after_forward,
    bool             is_sparse):
    DecoderSelfAttentionFP8Layer<T1, T2>(head_num,
                                         size_per_head,
                                         head_num / tensor_para.world_size_,
                                         rotary_embedding_dim,
                                         neox_rotary_style,
                                         d_model,
                                         q_scaling,
                                         stream,
                                         cublas_wrapper,
                                         allocator,
                                         is_free_buffer_after_forward,
                                         is_sparse),
    tensor_para_(tensor_para)
{
    FT_CHECK(head_num % tensor_para_.world_size_ == 0);
}

template<typename T1, typename T2>
TensorParallelDecoderSelfAttentionFP8Layer<T1, T2>::TensorParallelDecoderSelfAttentionFP8Layer(
    TensorParallelDecoderSelfAttentionFP8Layer<T1, T2> const& attention_layer):
    DecoderSelfAttentionFP8Layer<T1, T2>(attention_layer), tensor_para_(attention_layer.tensor_para_)
{
}

template<typename T1, typename T2>
void TensorParallelDecoderSelfAttentionFP8Layer<T1, T2>::forward(TensorMap*                 output_tensors,
                                                                 TensorMap*                 input_tensors,
                                                                 const AttentionWeight<T1>* attention_weights,
                                                                 const int max_seq_len)
{
    // input tensors:
    //      attention_input [batch_size, hidden_dimension],
    //      finished [batch_size],
    //      sequence_lengths [batch_size]
    //      input_lengths [batch_size]
    //      max_input_length [1] on cpu
    //      step [1] on cpu

    // output tensors:
    //      attention_output [batch_size, hidden_dimension],
    //      key_cache [batch, head_num, size_per_head // x, max_seq_len, x]
    //      value_cache [batch, head_num, max_seq_len, size_per_head]

    DecoderSelfAttentionFP8Layer<T1, T2>::forward(output_tensors, input_tensors, attention_weights);

    const size_t batch_size   = output_tensors->at("attention_output").shape[0];
    const size_t hidden_units = output_tensors->at("attention_output").shape[1];

    T2* attention_out = (T2*)(output_tensors->at("attention_output").data);
    if (tensor_para_.world_size_ > 1) {
        ftNcclAllReduceSum(attention_out,
                           attention_out,
                           batch_size * hidden_units,
                           tensor_para_,
                           DecoderSelfAttentionFP8Layer<T1, T2>::stream_);
        sync_check_cuda_error();
    }
}

template class TensorParallelDecoderSelfAttentionFP8Layer<__nv_fp8_e4m3, __nv_bfloat16>;

}  // namespace fastertransformer
