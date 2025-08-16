set -xe
## TODO: accelerate config
## TODO: MXFP4 quantization requires triton >= 3.4.0 and kernels installed, we will default to dequantizing the model to bf16  (fixed!)

# uv pip install torch --index-url https://download.pytorch.org/whl/cu128
# uv pip install "trl>=0.20.0" "peft>=0.17.0" "transformers>=4.55.0"

# uv pip install trl accelerate peft vllm
# uv pip install git+https://github.com/huggingface/transformers triton==3.4 kernels

uv pip install bitsandbytes accelerate vllm trl peft


# CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
accelerate launch \
  train.py \
  --model_name_or_path "Qwen/QwQ-32B" \
  --output_dir grpo-qwq-32b \
  --learning_rate 1e-5 \
  --gradient_checkpointing \
  --max_prompt_length 2048 \
  --max_completion_length 1024 \
  --use_vllm \
  --vllm_mode colocate \
  --use_peft \
  --load_in_8bit \
  --lora_dropout 0 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_target_modules "q_proj", "k_proj", "v_proj" \
  --vllm_gpu_memory_utilization 0.5 \
  --ddp_find_unused_parameters False