# reference: https://huggingface.co/docs/trl/main/en/vllm_integration
set -xe


CUDA_DEVICE_ORDER=PCI_BUS_ID 
CUDA_VISIBLE_DEVICES=0,1,2,3 trl vllm-serve \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --data-parallel-size 2 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 11674


# # TP=1, DP=4 crashes the vllm backend..
# CUDA_DEVICE_ORDER=PCI_BUS_ID 
# CUDA_VISIBLE_DEVICES=0,1,2,3 trl vllm-serve \
#   --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
#   --port 8000 \
#   --tensor-parallel-size 1 \
#   --data-parallel-size 4 \
#   --gpu-memory-utilization 0.95 \
#   --max-model-len 11674
#   # --max-model-len 20480



  # --max-model-len 4096 ## this works with promptlength 1024 + 512, completionlength 2048