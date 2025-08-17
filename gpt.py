import torch
from transformers import AutoModelForCausalLM, Mxfp4Config, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, GRPOConfig
from trl import SFTTrainer
from trl import GRPOTrainer


# Dummy reward function: count the number of unique characters in the completions
def reward_num_unique_chars(completions, **kwargs):
  out = [len(set(c)) for c in completions]
  print(out)
  return out

def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
    # dataset = load_dataset("HuggingFaceH4/Multilingual-Thinking", split="train")
    dataset = load_dataset("trl-lib/tldr", split="train")


    quantization_config = Mxfp4Config(dequantize=True)
    model_kwargs = dict(
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        use_cache=False,
    )

    model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b", **model_kwargs)


    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules="all-linear",
        target_parameters=[
            "7.mlp.experts.gate_up_proj",
            "7.mlp.experts.down_proj",
            "15.mlp.experts.gate_up_proj",
            "15.mlp.experts.down_proj",
            "23.mlp.experts.gate_up_proj",
            "23.mlp.experts.down_proj",
        ],
    )
    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()

    # training_args = SFTConfig(
    training_args = GRPOConfig(
        learning_rate=2e-4,
        # gradient_checkpointing=True,
        # gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        num_train_epochs=1,
        logging_steps=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        # max_length=2048,
        max_prompt_length=1024,
        max_completion_length=128,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": 0.1},
        output_dir="gpt-oss-20b-multilingual-reasoner",
        num_generations=2,
        use_vllm=True,
        vllm_mode="colocate",
        # vllm_mode="server",
        # vllm_server_host="http://127.0.0.1:8000"
    )

    trainer = GRPOTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_num_unique_chars,
    )
    trainer.train()


if __name__ == "__main__":
    main()