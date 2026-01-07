"""
Multi-Node Multi-GPU LLaMA 13B Training Script
===============================================
이종 GPU 환경:
- goguma3: 4x RTX 3060 Ti (8GB)  
- goguma4: 2x RTX 2080 Ti (11GB)

총 6개 GPU, 4-bit 양자화 + LoRA로 메모리 최적화
"""

import os
import torch
from dataclasses import dataclass, field
from datasets import load_dataset
import torch.distributed as dist
import time
import json
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    TrainerCallback
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

class MetricsCallback(TrainerCallback):
    """통신, 타이밍, 동기화 메트릭 수집"""
    
    def __init__(self, log_dir="./metrics"):
        os.makedirs(log_dir, exist_ok=True)
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.node = os.environ.get("HOSTNAME", "unknown")
        self.log_file = f"{log_dir}/training_rank{self.rank}.jsonl"
        self.step_start = None
        
    def on_step_begin(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        self.step_start = time.time()
        
    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        step_time = time.time() - self.step_start
        
        log_entry = {
            "step": state.global_step,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "node": self.node,
            "step_time": step_time,
            "timestamp": time.time(),
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "loss": state.log_history[-1].get("loss") if state.log_history else None,
        }
        
        # 통신 메트릭 (distributed 초기화 됐을 때만)
        if dist.is_initialized():
            # Barrier 대기 시간 (straggler 감지)
            sync_start = time.time()
            dist.barrier()
            log_entry["barrier_wait"] = time.time() - sync_start
            
            # AllReduce latency (1KB)
            dummy = torch.zeros(256).cuda()
            torch.cuda.synchronize()
            ar_start = time.time()
            dist.all_reduce(dummy)
            torch.cuda.synchronize()
            log_entry["allreduce_latency"] = time.time() - ar_start
            
            # Bandwidth 측정 (1MB)
            tensor = torch.zeros(256 * 1024).cuda()  # 1MB
            torch.cuda.synchronize()
            bw_start = time.time()
            dist.all_reduce(tensor)
            torch.cuda.synchronize()
            bw_time = time.time() - bw_start
            effective_mb = 2 * (self.world_size - 1) / self.world_size
            log_entry["bandwidth_MBps"] = effective_mb / bw_time if bw_time > 0 else 0
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")


@dataclass
class ModelConfig:
    model_name: str = "NousResearch/Llama-2-7b-hf"
    max_seq_length: int = 512  # 8GB GPU 기준으로 보수적으로
    use_4bit: bool = True
    use_gradient_checkpointing: bool = True


@dataclass 
class LoRAConfig:
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])


def setup_model_and_tokenizer(model_config: ModelConfig, lora_config: LoRAConfig):
    """4-bit 양자화 + LoRA로 메모리 효율적 로딩"""
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name,
        trust_remote_code=True,
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # 4-bit 양자화
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
   
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="sdpa",
        use_cache=False,
        device_map={"":local_rank},
    )
    
    model = prepare_model_for_kbit_training(model)
    
    if model_config.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # LoRA
    peft_config = LoraConfig(
        r=lora_config.lora_r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        target_modules=lora_config.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model, tokenizer


def prepare_dataset(tokenizer, max_seq_length: int):
    """Alpaca 데이터셋 준비"""
    
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    
    # 테스트용 작은 서브셋 (전체 학습시 이 줄 주석처리)
    #dataset = dataset.select(range(500))
    
    def format_prompt(example):
        if example.get("input", ""):
            text = f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}</s>"""
        else:
            text = f"""### Instruction:
{example['instruction']}

### Response:
{example['output']}</s>"""
        return {"text": text}
    
    dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)
    
    def tokenize(example):
        result = tokenizer(
            example["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
        result["labels"] = result["input_ids"].copy()
        return result
    
    tokenized_dataset = dataset.map(
        tokenize,
        remove_columns=["text"],
        num_proc=4,
    )
    
    return tokenized_dataset


def get_training_arguments(output_dir: str = "./llama_output"):
    """이종 GPU 환경용 학습 설정"""
    
    return TrainingArguments(
        output_dir=output_dir,
        
        # 학습
        num_train_epochs=100,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        
        # Optimizer
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        
        # 메모리
        fp16=True,
        gradient_checkpointing=True,
        
        # Logging
        logging_steps=1,
        logging_dir="./logs",
        report_to="tensorboard",
        logging_first_step=True,
        
        # 저장
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        
        # DeepSpeed
        deepspeed="ds_config.json",
        
        # 분산학습
        ddp_find_unused_parameters=False,
        dataloader_num_workers=2,
        
        seed=42,
        remove_unused_columns=False,
    )


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    print("="*60)
    print(f"GPU count (this node): {torch.cuda.device_count()}")
    print(f"Local rank: {local_rank}")
    print(f"World size: {world_size}")
    print("="*60)
    
    model_config = ModelConfig()
    lora_config = LoRAConfig()
    
    model, tokenizer = setup_model_and_tokenizer(model_config, lora_config)
    dataset = prepare_dataset(tokenizer, model_config.max_seq_length)
    training_args = get_training_arguments()
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        callbacks=[MetricsCallback(log_dir="./metrics")], 
    )
    
    trainer.train()
    
    if local_rank == 0:
        trainer.save_model("./llama_final")
        tokenizer.save_pretrained("./llama_final")
        print("Done! Model saved to ./llama_final")


if __name__ == "__main__":
    main()
