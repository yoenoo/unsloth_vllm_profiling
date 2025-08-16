set -xe
## TODO: accelerate config
## TODO: MXFP4 quantization requires triton >= 3.4.0 and kernels installed, we will default to dequantizing the model to bf16  (fixed!)

uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install "trl>=0.20.0" "peft>=0.17.0" "transformers>=4.55.0"

# uv pip install trl accelerate peft vllm
# uv pip install git+https://github.com/huggingface/transformers triton==3.4 kernels

uv pip install --pre vllm==0.10.1+gptoss \
  --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
  --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
  --index-strategy unsafe-best-match

uv pip install vllm

# UPDATE: without vLLM works (just very slow)

CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
  train.py \
  --model_name_or_path "openai/gpt-oss-20b" \
  --output_dir grpo-gpt-oss-20b \
  --learning_rate 1e-5 \
  --gradient_checkpointing \
  --max_prompt_length 2048 \
  --max_completion_length 1024 \
  --use_vllm \
  --vllm_mode colocate \
  --use_peft \
  --lora_target_modules "q_proj", "k_proj", "v_proj" \
  --ddp_find_unused_parameters False