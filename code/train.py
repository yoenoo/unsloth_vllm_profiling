# # from datasets import load_dataset
# # from trl import GRPOTrainer, GRPOConfig

# # dataset = load_dataset("trl-lib/tldr", split="train")

# # # Dummy reward function: count the number of unique characters in the completions
# # def reward_num_unique_chars(completions, **kwargs):
# #     return [len(set(c)) for c in completions]

# # training_args = GRPOConfig(
# #     output_dir="my_test",
# #     use_vllm=True,
# #     bf16=True,
# #     gradient_checkpointing=True,
# #     vllm_gpu_memory_utilization=0.5,
# # )


# # trainer = GRPOTrainer(
# #     # model="Qwen/Qwen3-30B-A3B-Instruct-2507",
# #     model="Qwen/Qwen2.5-7B",
# #     args=training_args,
# #     reward_funcs=reward_num_unique_chars,
# #     train_dataset=dataset,
# # )

# # trainer.train()


# from datasets import load_dataset
# from trl import GRPOTrainer, GRPOConfig

# dataset = load_dataset("trl-lib/tldr", split="train")

# def reward_num_unique_chars(completions, **kwargs):
#     return [len(set(c)) for c in completions]

# training_args = GRPOConfig(
#     output_dir="my_test",
#     use_vllm=True,
#     vllm_mode="server",                      # <- tell TRL it's an external server
#     vllm_server_base_url="http://127.0.0.1:8000",  # <- match your vLLM serve port
#     bf16=True,
#     gradient_checkpointing=True,

#     # Keep generation light so vLLM’s KV cache doesn’t explode:
#     per_device_train_batch_size=1,
#     num_generations=2,            # aka group size (lower = less KV cache)
#     generation_batch_size=2,      # don’t let this mirror your train batch
#     max_prompt_length=256,
#     max_completion_length=128,
# )

# trainer = GRPOTrainer(
#     model="Qwen/Qwen2.5-7B",
#     args=training_args,
#     reward_funcs=reward_num_unique_chars,
#     train_dataset=dataset,
# )

# trainer.train()


from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig

dataset = load_dataset("trl-lib/tldr", split="train")

def reward_num_unique_chars(completions, **kwargs):
  out = [len(set(c)) for c in completions]
  # print(completions[0])
  print([len(c) for c in completions]) # n. characters (not tokens)
  return out

peft_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)

training_args = GRPOConfig(
    output_dir="my_test",
    # trainer-side memory savers
    bf16=True,
    # gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,

    # vLLM server mode (already running on GPU 1)
    use_vllm=True,
    vllm_mode="server",
    vllm_server_base_url="http://127.0.0.1:8000",

    # keep rollout memory small
    num_generations=8,
    # generation_batch_size=2,
    max_prompt_length=1024 + 512,
    max_completion_length=16384,
    logging_steps = 1,
    # log_completions=True,
    # pass through to model.from_pretrained
    model_init_kwargs={
        "torch_dtype": "bfloat16",
        # "attn_implementation": "flash_attention_2",
        "use_cache": False,  # saves activation memory in training
    },
    # vllm_gpu_memory_utilization=0.95, # default is 0.3

    # ddp_find_unused_parameters=False,
    ddp_find_unused_parameters=True,
    ddp_broadcast_buffers=False,        # avoids extra sync on buffers
    remove_unused_columns=False,        # HF Trainer won’t drop inputs you use
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},  # safer with PyTorch 2.x
)

trainer = GRPOTrainer(
    # model="Qwen/Qwen2.5-7B",
    model="Qwen/Qwen3-30B-A3B-Instruct-2507",
    args=training_args,
    reward_funcs=reward_num_unique_chars,
    train_dataset=dataset,
    peft_config=peft_cfg,
)

trainer.train()