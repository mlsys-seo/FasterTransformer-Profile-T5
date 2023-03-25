# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
This example is used to verify the correctess on summarization task. So, we don't
put benchmark testing in this example.
'''

from __future__ import print_function
import argparse
import numpy as np
import os
import sys
import torch
import torch.distributed as dist
# dir_path = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(dir_path + "/../../../3rdparty/transformers/src/")

from transformers import T5ForConditionalGeneration, T5Config
import math

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../..")
from examples.pytorch.t5.utils.ft_decoding import FTT5DecodingWeight, FTT5Decoding, FTT5
from examples.pytorch.t5.utils.ft_encoder import FTT5EncoderWeight, FTT5Encoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_max_seq_len', type=int, default=200, metavar='NUMBER',
                        help='max sequence length (default: 200)')
    parser.add_argument('--decoder_max_seq_len', type=int, default=200, metavar='NUMBER',
                        help='max sequence length (default: 200)')
    parser.add_argument('-model', '--model', type=str, default="t5-small", metavar='STRING',
                        help='T5 model size. Only used when --model_path=None')
    parser.add_argument('-tensor_para_size', '--tensor_para_size', type=int, default=1, metavar='NUMBER',
                        help='size of tensor parallelism (default: 1)')
    parser.add_argument('-pipeline_para_size', '--pipeline_para_size', type=int, default=1, metavar='NUMBER',
                        help='size of pipeline parallelism (default: 1)')
    parser.add_argument('--start_batch_size', type=int, default=512, metavar='NUMBER',
                        help='Candidate (k) value of top k sampling in decoding. Default is 1.')
    parser.add_argument('--end_batch_size', type=int, default=2048, metavar='NUMBER',
                        help='Candidate (k) value of top k sampling in decoding. Default is 1.')
    parser.add_argument('--batch_size_hop', type=int, default=1, metavar='NUMBER',
                        help='Candidate (k) value of top k sampling in decoding. Default is 1.')
    parser.add_argument('--profile_iters', type=int, default=50, metavar='NUMBER',
                        help='Candidate (k) value of top k sampling in decoding. Default is 1.')
    parser.add_argument('-lib_path', '--lib_path', type=str, default="lib/libth_transformer.so", metavar='STRING',
                        help='the path of FasterTransformer pytorch t5 op library.')
    parser.add_argument('-d', '--data_type', type=str, default="fp32", metavar='STRING',
                        help='data type for inference (default: fp32)', choices=['fp32', 'fp16', 'bf16'])
    parser.add_argument('--config_path', type=str, default="/workspace/code/multi/model_config", metavar='STRING',
                        help='the path of FasterTransformer pytorch t5 op library.')
    
    
    args = parser.parse_args()
    
    print(f"Profile Start: {args.start_batch_size} End: {args.end_batch_size} Hop: {args.batch_size_hop}")
    
    
    np.random.seed(0) # rouge score use sampling to compute the score

    if dist.is_mpi_available():
        try:
            dist.init_process_group(backend='mpi')
            rank = dist.get_rank()
        except:
            rank = dist.get_rank()
    else:
        rank = 0
        
    weight_data_type = {"fp16": np.float16, "fp32": np.float32}[args.data_type]

    config = T5Config.from_pretrained(os.path.join(args.config_path, f"{args.model}.json"))
    t5_model = T5ForConditionalGeneration._from_config(config)
    
    encoder_config = t5_model.encoder.config
    decoder_config = t5_model.decoder.config
    encoder_config.num_layers = 1
    decoder_config.num_layers = 1
    
    ft_encoder_weight = FTT5EncoderWeight(
        encoder_config,
        args.tensor_para_size,
        args.pipeline_para_size,
        t5_with_bias=False,
        use_gated_activation=False,
        t5_with_moe=False,
        position_embedding_type=0,
        weight_data_type=weight_data_type,
    )
    ft_decoding_weight = FTT5DecodingWeight(
        decoder_config,
        args.tensor_para_size,
        args.pipeline_para_size,
        t5_with_bias=False,
        use_gated_activation=False,
        t5_with_moe=False,
        position_embedding_type=0,
        weight_data_type=weight_data_type,
    )

    ft_encoder_weight.load_from_model(t5_model)
    ft_decoding_weight.load_from_model(t5_model)

    if args.data_type == "fp32":
        ft_encoder_weight.to_float()
        ft_decoding_weight.to_float()
    elif args.data_type == "fp16":
        ft_encoder_weight.to_half()
        ft_decoding_weight.to_half()
    elif args.data_type == "bf16":
        ft_encoder_weight.to_bfloat16()
        ft_decoding_weight.to_bfloat16()

    ft_encoder_weight.to_cuda()
    ft_decoding_weight.to_cuda()

    q_scaling = 1.0 / (math.sqrt(encoder_config.d_kv))
    remove_padding = True
    ft_encoder = FTT5Encoder(ft_encoder_weight.w, args.lib_path, encoder_config.num_heads,
                                encoder_config.d_kv, encoder_config.d_ff,
                                encoder_config.d_model, remove_padding, encoder_config.num_layers,
                                encoder_config.relative_attention_num_buckets, 0, [],
                                128, False, q_scaling, args.tensor_para_size, args.pipeline_para_size, False,
                                0, moe_k=0,
                                activation_type="gelu",)
    ft_decoding = FTT5Decoding(ft_decoding_weight.w, args.lib_path,
                                decoder_config.num_heads, decoder_config.d_kv,
                                decoder_config.d_ff, encoder_config.d_model,
                                decoder_config.d_model, decoder_config.num_layers,
                                decoder_config.decoder_start_token_id, decoder_config.eos_token_id,
                                decoder_config.vocab_size,
                                q_scaling,
                                decoder_config.relative_attention_num_buckets, 0, [], max_distance=128,
                                tensor_para_size=args.tensor_para_size, pipeline_para_size=args.pipeline_para_size,
                                t5_with_bias=False,
                                position_embedding_type=0, moe_k=0,
                                activation_type="gelu", tie_word_embeddings=decoder_config.tie_word_embeddings,)

    ft_t5 = FTT5(ft_encoder, ft_decoding)

    
    for _BATCH in range(args.end_batch_size, args.start_batch_size - 1, -args.batch_size_hop):
        input_token = torch.randint(5, (_BATCH, args.encoder_max_seq_len),
                                dtype=torch.int32).to(rank)
        mem_seq_len = torch.tensor([args.encoder_max_seq_len for _ in range(_BATCH)], dtype=torch.int32).to(rank)
        
        with torch.no_grad():
            ft_t5(input_token,
                mem_seq_len,
                args.decoder_max_seq_len,
                args.profile_iters)

if __name__ == '__main__':
    main()
