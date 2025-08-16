import torch
from trl import (
  GRPOTrainer, 
  GRPOConfig, 
  ModelConfig, 
  ScriptArguments, 
  TrlParser, 
  get_peft_config, 
  get_quantization_config, 
  get_kbit_device_map
)
from transformers import AutoModelForCausalLM, Mxfp4Config
from datasets import load_dataset


dataset = load_dataset("trl-lib/tldr", split="train")

# Dummy reward function: count the number of unique characters in the completions
def reward_num_unique_chars(completions, **kwargs):
    return [len(set(c)) for c in completions]

parser = TrlParser((ScriptArguments, GRPOConfig, ModelConfig))
script_args, training_args, model_args = parser.parse_args_and_config()
print(model_args)
print(training_args)
print(script_args)

torch_dtype = (
  model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
)
quantization_config = get_quantization_config(model_args)
# quantization_config = Mxfp4Config(dequantize=True) # gpt-oss

training_args.model_init_kwargs = dict(
  revision=model_args.model_revision,
  attn_implementation=model_args.attn_implementation,
  # attn_implementation="eager", # gpt-oss
  torch_dtype=torch_dtype,
  device_map=get_kbit_device_map() if quantization_config is not None else None,
  quantization_config=quantization_config,
  # use_cache=False, # gpt-oss
)


trainer = GRPOTrainer(
  model=model_args.model_name_or_path,
  args=training_args,
  reward_funcs=reward_num_unique_chars,
  train_dataset=dataset,
  peft_config=get_peft_config(model_args),
)

trainer.train()