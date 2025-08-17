set -xe

# . .venv/bin/activate

CUDA_DEVICE_ORDER=PCI_BUS_ID 
NCCL_ASYNC_ERROR_HANDLING=1 
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch \
  --num_processes=4 \
  --mixed_precision=bf16 \
  train.py