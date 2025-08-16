set -xe

uv pip install --pre vllm==0.10.1+gptoss \
  --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
  --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
  --index-strategy unsafe-best-match

uv pip install trl accelerate fastapi uvicorn vllm

CUDA_VISIBLE_DEVICES=0,1 trl vllm-serve \
  --model unsloth/gpt-oss-20b \ 
  --tensor-parallel-size 2 
  # --tensor-parallel-size 1 \
  # --data-parallel-size 1

  