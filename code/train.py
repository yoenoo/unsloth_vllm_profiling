from datasets import load_dataset
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig

dataset = load_dataset("trl-lib/tldr", split="train")

def reward_num_unique_chars(completions, **kwargs):
  out = [len(set(c)) for c in completions]
  print([len(c) for c in completions]) # n. characters (not tokens)
  return out

peft_cfg = LoraConfig(
  r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
  task_type="CAUSAL_LM",
  target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)

training_args = GRPOConfig(
  output_dir="my_test",
  bf16=True,
  optim="paged_adamw_8bit",
  per_device_train_batch_size=1,
  gradient_accumulation_steps=8,

  use_vllm=True,
  # vllm_mode="colocate", # this leads to OOM in my setting (8 x H200)
  vllm_mode="server",
  vllm_server_base_url="http://127.0.0.1:8000",
  vllm_gpu_memory_utilization=0.8,
  
  generation_batch_size = 64, ## TODO: test this
  num_generations=8,
  max_prompt_length=1024 + 512,
  max_completion_length=8192,
  logging_steps = 1,
  model_init_kwargs={
    "torch_dtype": "bfloat16",
    "use_cache": False,  # saves activation memory in training
  },

  # multi-gpu stuff
  ddp_find_unused_parameters=True,
  ddp_broadcast_buffers=False,        # avoids extra sync on buffers
  remove_unused_columns=False,        # HF Trainer won’t drop inputs you use
  gradient_checkpointing=True,
  gradient_checkpointing_kwargs={"use_reentrant": False},  # safer with PyTorch 2.x
)

trainer = GRPOTrainer(
  model="Qwen/Qwen3-30B-A3B-Instruct-2507",
  args=training_args,
  reward_funcs=reward_num_unique_chars,
  train_dataset=dataset,
  peft_config=peft_cfg,
)

trainer.train()