# Red-teaming LLMs via Adaptive Environments

All training runs on [Tinker](https://tinker-docs.thinkingmachines.ai/) — no local GPU required.

## Installation
```bash
pip install -r requirements.txt
export TINKER_API_KEY=<your key>  # from https://tinker-console.thinkingmachines.ai
```

## Warm-up SFT
Fine-tune the attacker LLM on pre-collected attack examples:
```bash
python main.py \
    --mode sft \
    --model_name Qwen/Qwen3-30B-A3B \
    --lr 3e-5 \
    --train_steps 100 \
    --batch_size 32 \
    --exp_name attacker-sft
```

## RL Red-teaming (GRPO)
Train the attacker to generate adversarial prompts against a victim model:
```bash
python main.py \
    --mode redteam \
    --model_name Qwen/Qwen3-30B-A3B \
    --victim_model Qwen/Qwen3-8B-Instruct \
    --judge_model Qwen/Qwen3-30B-A3B-Instruct \
    --sft_ckpt tinker://<run-id>/weights/final \
    --lr 1e-4 \
    --train_steps 5000 \
    --batch_size 16 \
    --group_size 4 \
    --seed 0 \
    --exp_name attacker-rl
```

## Active Attacks (adversarial arms race)
Alternates RL attacks with safety fine-tuning of the victim:
```bash
python main.py \
    --mode redteam \
    --model_name Qwen/Qwen3-30B-A3B \
    --victim_model Qwen/Qwen3-8B-Instruct \
    --judge_model Qwen/Qwen3-30B-A3B-Instruct \
    --sft_ckpt tinker://<run-id>/weights/final \
    --lr 1e-4 \
    --train_steps 5000 \
    --batch_size 16 \
    --group_size 4 \
    --active_attacks \
    --interval 1000 \
    --safety_steps 200 \
    --exp_name active-attacks
```

## MLE Smoothing
Re-train the attacker on collected successful attacks:
```bash
python main.py \
    --mode mle \
    --model_name Qwen/Qwen3-30B-A3B \
    --attack_ckpt tinker://<run-id>/weights/final \
    --lr 3e-5 \
    --train_steps 200 \
    --batch_size 32 \
    --exp_name attacker-mle
```

## Safety Fine-tuning
Fine-tune the victim on attack prompts + safe responses:
```bash
python main.py \
    --mode safety \
    --victim_model Qwen/Qwen3-8B-Instruct \
    --safety_data_file ./safety_data.json \
    --lr 3e-5 \
    --train_steps 200 \
    --batch_size 32 \
    --exp_name victim-safety
```

## Evaluation
```bash
python eval_baselines.py \
    --input_file attacks/attack_dataset.json \
    --victim_model Qwen/Qwen3-8B-Instruct \
    --judge_model Qwen/Qwen3-30B-A3B-Instruct \
    --output_dir results/
```
