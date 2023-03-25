PROFILE_NSYS="nsys profile --cuda-graph-trace graph -t cuda,nvtx -o /workspace/profile/test_ft_tp -w true -f true"

	
# FT_NVTX=ON mpirun -n 2 --allow-run-as-root \
# CUDA_LAUNCH_BLOCKING=1 FT_DEBUG_LEVEL=DEBUG python /workspace/code/multi/FasterTransformer/examples/pytorch/t5/mnli_task_example.py \
#         --tensor_para_size 1 \
#         --pipeline_para_size 1 \
#         --start_batch_size 510 \
#         --end_batch_size 512 \
#         --batch_size_hop 8 \
#         --beam_width 1 \
#         --encoder_max_seq_len 128 \
#         --decoder_max_seq_len 32 \
#         --data_type fp16 \
#         --test_time 1 \
#         --sampling_topk 1 \
#         --model t5-11b \
#         --profile_iters 10 \
#         --config_path /workspace/code/multi/model_config

python /workspace/code/multi/FasterTransformer/examples/pytorch/t5/mnli_task_example.py \
        --start_batch_size 510 \
        --end_batch_size 512 \
        --batch_size_hop 8 \
        --encoder_max_seq_len 128 \
        --decoder_max_seq_len 32 \
        --data_type fp16 \
        --model t5-11b \
        --config_path /workspace/code/multi/model_config \
        --profile_iters 20