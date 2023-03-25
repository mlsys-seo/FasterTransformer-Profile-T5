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
import json
import numpy as np
import os
import sys
import torch
import torch.distributed as dist
from datasets import load_dataset, load_metric
# dir_path = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(dir_path + "/../../../3rdparty/transformers/src/")

from transformers import T5ForConditionalGeneration, AutoTokenizer, T5Config
from tqdm import tqdm
import configparser
import math
import datetime

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../..")
from examples.pytorch.t5.utils.ft_decoding import FTT5DecodingWeight, FTT5Decoding, FTT5
from examples.pytorch.t5.utils.ft_encoder import FTT5EncoderWeight, FTT5Encoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ft_model_location', type=str,
                        default='/models/T5/HF/t5-base/c-models/')
    parser.add_argument('--hf_model_location', type=str,
                        default='/models/T5/HF/t5-base/')
    parser.add_argument('--disable_summarize', action='store_true')
    parser.add_argument('--test_hf', action='store_true')
    parser.add_argument('--test_ft', action='store_true')
    parser.add_argument('--data_type', type=str, choices=['fp32', 'fp16', 'bf16'], default='fp32')
    parser.add_argument("--cache_path", type=str, default="/workdir/datasets/ccdv/")
    parser.add_argument("--max_ite", type=int, default=20)
    parser.add_argument("--max_seq_len", type=int, default=200)
    parser.add_argument("--ft_use_hf_config", action="store_true",
                        help="use the hyper-parameters from the hf model")
    parser.add_argument('--lib_path', type=str, default='./lib/libth_transformer.so',
                        help='path to the pyt_fastertransformer dynamic lib file.')
    parser.add_argument('--tensor_para_size', type=int, default=1,
                        help='tensor parallel size')
    parser.add_argument('--pipeline_para_size', type=int, default=1,
                        help='pipeline parallel size')
    parser.add_argument('--rougeLsum_threshold', type=float,
                        help='Threshold of FT rougeLsum score')
    parser.add_argument("--top_k", type=int, default=1, help="top k for sampling")
    parser.add_argument("--top_p", type=float, default=0.0, help="top p for sampling")
    parser.add_argument("--beam_width", type=int, default=1, help="beam width for beam search")

    args = parser.parse_args()
    np.random.seed(0) # rouge score use sampling to compute the score

    if dist.is_mpi_available():
        try:
            dist.init_process_group(backend='mpi')
            rank = dist.get_rank()
        except:
            rank = dist.get_rank()
    else:
        rank = 0

    disable_summarize = args.disable_summarize
    test_hf = args.test_hf
    test_ft = args.test_ft

    tensor_para_size = args.tensor_para_size
    pipeline_para_size = args.pipeline_para_size
    ft_model_location = args.ft_model_location + f"/{tensor_para_size}-gpu/"
    hf_model_location = args.hf_model_location

    # tokenizer = AutoTokenizer.from_pretrained('t5-11b')
    # tokenizer.pad_token = tokenizer.eos_token

    if rank == 0 and test_hf:
        start_time = datetime.datetime.now()
        if args.data_type == "fp32":
            model = T5ForConditionalGeneration.from_pretrained(hf_model_location, torch_dtype=torch.float32).cuda()
        elif args.data_type == "fp16":
            model = T5ForConditionalGeneration.from_pretrained(hf_model_location, torch_dtype=torch.float16).cuda()
        elif args.data_type == "bf16":
            model = T5ForConditionalGeneration.from_pretrained(hf_model_location, torch_dtype=torch.bfloat16).cuda()
        stop_time = datetime.datetime.now()
        print(f"[INFO] load HF model spend {(stop_time - start_time).total_seconds()} sec")

    if test_ft:
        ckpt_config = configparser.ConfigParser()

        ckpt_config_path = os.path.join('/workspace/data/ckpt/ft/t5/1-gpu/config.ini')
        if os.path.isfile(ckpt_config_path):
            ckpt_config.read(ckpt_config_path)
        else:
            assert False, "[ERROR] This example only support loading model with FT format directly."

        weight_data_type = np.float32
        weight_data_type = {"fp16": np.float16, "fp32": np.float32}[ckpt_config.get("encoder", "weight_data_type")]
        relative_attention_max_distance = 128
        encoder_config = T5Config(vocab_size=ckpt_config.getint("encoder", "vocab_size"),
                                  d_model=ckpt_config.getint("encoder", "d_model"),
                                  d_kv=ckpt_config.getint("encoder", "d_kv"),
                                  d_ff=ckpt_config.getint("encoder", "d_ff"),
                                  num_layers=ckpt_config.getint("encoder", "num_layers"),
                                  num_decoder_layers=ckpt_config.getint("encoder", "num_decoder_layers"),
                                  num_heads=ckpt_config.getint("encoder", "num_heads"),
                                  relative_attention_num_buckets=ckpt_config.getint(
                                      "encoder", "relative_attention_num_buckets_or_max_pos_seq_len"),
                                  feed_forward_proj=ckpt_config.get("encoder", "feed_forward_proj"),
                                  pad_token_id=ckpt_config.getint("encoder", "pad_token_id"),
                                  eos_token_id=ckpt_config.getint("encoder", "eos_token_id"),
                                  is_gated_act=ckpt_config.getboolean("encoder", "is_gated_act", fallback=0),
                                  )
        decoder_config = T5Config(vocab_size=ckpt_config.getint("decoder", "vocab_size"),
                                  d_model=ckpt_config.getint("decoder", "d_model"),
                                  d_kv=ckpt_config.getint("decoder", "d_kv"),
                                  d_ff=ckpt_config.getint("decoder", "d_ff"),
                                  num_layers=ckpt_config.getint("decoder", "num_layers"),
                                  num_decoder_layers=ckpt_config.getint("decoder", "num_decoder_layers"),
                                  num_heads=ckpt_config.getint("decoder", "num_heads"),
                                  relative_attention_num_buckets=ckpt_config.getint(
                                      "decoder", "relative_attention_num_buckets_or_max_pos_seq_len"),
                                  feed_forward_proj=ckpt_config.get("decoder", "feed_forward_proj"),
                                  pad_token_id=ckpt_config.getint("decoder", "pad_token_id"),
                                  eos_token_id=ckpt_config.getint("decoder", "eos_token_id"),
                                  decoder_start_token_id=ckpt_config.getint("decoder", "decoder_start_token_id"),
                                  is_gated_act=ckpt_config.getboolean("decoder", "is_gated_act", fallback=0),
                                  )
        assert decoder_config.feed_forward_proj == encoder_config.feed_forward_proj
        assert decoder_config.feed_forward_proj == encoder_config.feed_forward_proj
        
        encoder_config.num_layers = 1
        decoder_config.num_layers = 1
        
        print(encoder_config)
        
        t5_with_bias = ckpt_config.getboolean("structure", "t5_with_bias")
        use_gated_activation = encoder_config.is_gated_act
        position_embedding_type = 0 if ckpt_config.get('structure', 'position_embedding_type') == 'relative' else 1
        activation_type = encoder_config.feed_forward_proj

        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py#L1660
        # if tie_word_embeddings == True, scale the decoder output by sequence_output = sequence_output * (self.model_dim**-0.5)
        tie_word_embeddings = ckpt_config.getboolean("decoder", "tie_word_embeddings")
        ft_encoder_weight = FTT5EncoderWeight(
            encoder_config,
            tensor_para_size,
            pipeline_para_size,
            t5_with_bias=t5_with_bias,
            use_gated_activation=use_gated_activation,
            position_embedding_type=position_embedding_type,
            weight_data_type=weight_data_type
        )
        ft_decoding_weight = FTT5DecodingWeight(
            decoder_config,
            tensor_para_size,
            pipeline_para_size,
            t5_with_bias=t5_with_bias,
            use_gated_activation=use_gated_activation,
            position_embedding_type=position_embedding_type,
            weight_data_type=weight_data_type,
        )
        
        config = T5Config.from_pretrained("/workspace/code/multi/model_config/t5-11b.json")
        t5_model = T5ForConditionalGeneration._from_config(config)

        start_time = datetime.datetime.now()
        ft_encoder_weight.load_from_model(t5_model)
        stop_time = datetime.datetime.now()
        print(f"[INFO] load FT encoder model spend {(stop_time - start_time).total_seconds()} sec")
        start_time = datetime.datetime.now()
        ft_decoding_weight.load_from_model(t5_model)
        stop_time = datetime.datetime.now()
        print(f"[INFO] load FT decoding model spend {(stop_time - start_time).total_seconds()} sec")
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
        remove_padding = False
        ft_encoder = FTT5Encoder(ft_encoder_weight.w, args.lib_path, encoder_config.num_heads,
                                 encoder_config.d_kv, encoder_config.d_ff,
                                 encoder_config.d_model, remove_padding, encoder_config.num_layers,
                                 encoder_config.relative_attention_num_buckets,
                                 0, # num_experts
                                 [], # moe_layer_index
                                 relative_attention_max_distance, False, q_scaling, tensor_para_size,
                                 pipeline_para_size, t5_with_bias,
                                 position_embedding_type, moe_k=0, activation_type=activation_type)

        ft_decoding = FTT5Decoding(ft_decoding_weight.w, args.lib_path,
                                   decoder_config.num_heads, decoder_config.d_kv,
                                   decoder_config.d_ff, encoder_config.d_model,
                                   decoder_config.d_model, decoder_config.num_layers,
                                   decoder_config.decoder_start_token_id, decoder_config.eos_token_id,
                                   decoder_config.vocab_size, q_scaling,
                                   decoder_config.relative_attention_num_buckets,
                                   0, # num_experts
                                   [], # moe_layer_index
                                   max_distance=relative_attention_max_distance,
                                   tensor_para_size=tensor_para_size, pipeline_para_size=pipeline_para_size,
                                   t5_with_bias=t5_with_bias, position_embedding_type=position_embedding_type,
                                   moe_k=0, activation_type=activation_type, tie_word_embeddings=tie_word_embeddings)

        ft_t5 = FTT5(ft_encoder, ft_decoding)

    beam_width = args.beam_width
    top_k = args.top_k
    top_p = args.top_p

    def summarize_ft(line):
        # line = line.strip()
        # line = line.replace(" n't", "n't")

        # line_tokens = tokenizer(line, return_tensors='pt')
        _MAX_SEQ = 128
        _BATCH = 512
        
        
        line_tokens = torch.randint(1, 10000, (_BATCH, _MAX_SEQ), dtype=torch.int32, requires_grad=False).to('cuda')
        mem_seq_len = torch.tensor([_MAX_SEQ for _ in range(_BATCH)], dtype=torch.int32, requires_grad=False).to('cuda')

        with torch.no_grad():
            ft_t5(line_tokens,
                mem_seq_len,
                beam_width,
                args.max_seq_len,
                top_k,
                top_p,
                beam_search_diversity_rate=0.0,
                is_return_output_log_probs=False,
                len_penalty=1.0,
                is_return_cum_log_probs=False)


    datapoint = "hello world"
    for data_point_idx in tqdm(range(1, 11490, int(11490 / args.max_ite))):
        try:
            summarize_ft(datapoint)
        except Exception as e:
            print(f'Error with datapoint: {data_point_idx} with error {e}')


if __name__ == '__main__':
    # tokenizer = AutoTokenizer.from_pretrained("t5-11b")
    # tokenizer.pad_token = tokenizer.eos_token
    # line_tokens = tokenizer(["Hello World", "Hello abcd"], return_tensors='pt', padding=True)
    # print(line_tokens.input_ids)
    # print(line_tokens.input_ids.shape)
    # print("=============================")
    # mem_seq_len = torch.sum(line_tokens.attention_mask, dim=1).type(torch.int32)
    # print(mem_seq_len)
    # print(mem_seq_len.shape)
    # a = 1
    main()