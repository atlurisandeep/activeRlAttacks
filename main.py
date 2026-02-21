import argparse
import logging
import random

from trainers.sft_trainer import SFTTrainer
from trainers.rl_trainer import RLTrainer
from trainers.safety_trainer import SafetyTrainer
from trainers.mle_trainer import MLETrainer
from utils import seed, JUDGE_MODEL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SAFE_REFUSALS = [
    "I'm not able to help with that request. It could lead to harm. Is there something constructive I can help you with instead?",
    "I can't assist with this as it involves potentially harmful activities. I'd be happy to help with something else.",
    "This request asks for information that could be used to cause harm, so I'll have to decline. Let me know if you have other questions.",
    "I'm designed to be helpful, but I can't provide assistance with requests that could lead to harm or illegal activities.",
    "I understand you have a question, but I'm not able to provide guidance on potentially harmful topics. Can I help with something else?",
]


def generate_safe_refusal(instruction):
    """Return a safe refusal response for a harmful instruction."""
    return random.choice(SAFE_REFUSALS)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--mode", type=str, default="sft",
                      choices=["sft", "redteam", "mle", "safety"])

    # Model names (Tinker model identifiers)
    args.add_argument("--model_name", type=str,
                      default="Qwen/Qwen3-30B-A3B-Instruct-2507",
                      help="Attacker model (Tinker model name)")
    args.add_argument("--victim_model", type=str,
                      default="Qwen/Qwen3-30B-A3B-Instruct-2507",
                      help="Victim model (Tinker model name)")
    args.add_argument("--judge_model", type=str,
                      default=JUDGE_MODEL,
                      help="LLM judge model for toxicity scoring")

    # Checkpoint paths (Tinker paths like tinker://run-id/weights/name)
    args.add_argument("--sft_ckpt", type=str, default="",
                      help="Tinker path to SFT checkpoint for attacker")
    args.add_argument("--attack_ckpt", type=str, default="",
                      help="Tinker path to RL attack checkpoint")
    args.add_argument("--victim_ckpt", type=str, default="",
                      help="Tinker path to victim checkpoint")

    # Training hyperparameters
    args.add_argument("--lr", type=float, default=3e-5)
    args.add_argument("--lr_min", type=float, default=3e-6,
                      help="Minimum LR after cosine decay")
    args.add_argument("--warmup_ratio", type=float, default=0.1,
                      help="Fraction of steps for LR warmup")
    args.add_argument("--weight_decay", type=float, default=0.0)
    args.add_argument("--train_steps", type=int, default=200)
    args.add_argument("--batch_size", type=int, default=16)
    args.add_argument("--group_size", type=int, default=4,
                      help="Number of rollouts per prompt for GRPO")

    # LoRA config
    args.add_argument("--lora_r", type=int, default=32)

    # Generation parameters
    args.add_argument("--max_len", type=int, default=20,
                      help="Max tokens for attacker generation")
    args.add_argument("--victim_max_len", type=int, default=128)
    args.add_argument("--victim_top_p", type=float, default=0.95)
    args.add_argument("--victim_temp", type=float, default=0.7)

    args.add_argument("--seed", type=int, default=42)

    # Data files
    args.add_argument("--prompt_file", type=str,
                      default="./prompts/attack_prompt.jsonl")
    args.add_argument("--few_shot_file", type=str,
                      default="./prompts/sft_dataset.json")
    args.add_argument("--safety_data_file", type=str, default="",
                      help="JSON file with safety training data")

    ######################### Category Diversity #########################
    args.add_argument("--seed_data_file", type=str, default="",
                      help="Annotated seed data JSON with S1-S8 categories")
    args.add_argument("--stratify_categories", action="store_true",
                      help="Round-robin sample across harm categories per batch")
    args.add_argument("--condition_on_category", action="store_true",
                      help="Inject category target into attacker prompt")
    args.add_argument("--adaptive_category_weights", action="store_true",
                      help="Boost rewards for underperforming categories")
    args.add_argument("--category_weight_beta", type=float, default=2.0,
                      help="Category boost strength: w = exp(-beta * ASR)")
    args.add_argument("--category_ema_alpha", type=float, default=0.1,
                      help="EMA decay for per-category ASR tracking")
    ##################################################################

    # Logging / saving
    args.add_argument("--exp_name", type=str, default="active-attacks")
    args.add_argument("--log_dir", type=str, default="logs")

    ######################### Active Attacks #########################
    args.add_argument("--active_attacks", action="store_true")
    args.add_argument("--interval", type=int, default=1000,
                      help="RL steps per round in active attacks")
    args.add_argument("--safety_steps", type=int, default=200,
                      help="Safety fine-tuning steps per round")
    ##################################################################

    ######################### Reward Shaping #########################
    args.add_argument("--min_attack_tokens", type=int, default=5,
                      help="Attacks shorter than this get a length penalty")
    args.add_argument("--length_penalty", type=float, default=2.0,
                      help="Penalty subtracted for too-short or max-length attacks")
    args.add_argument("--max_repetition_rate", type=float, default=0.5,
                      help="3-gram repetition rate above this triggers penalty")
    args.add_argument("--repetition_penalty", type=float, default=3.0,
                      help="Penalty multiplied by repetition rate when above threshold")
    args.add_argument("--kl_coeff", type=float, default=0.03,
                      help="KL/entropy regularization coefficient; bonus for high-entropy outputs")
    args.add_argument("--diversity_threshold", type=float, default=0.8,
                      help="Jaccard overlap above this marks attacks as duplicates")
    args.add_argument("--diversity_penalty", type=float, default=2.0,
                      help="Penalty for near-duplicate attacks in a batch")
    args.add_argument("--reward_clip", type=float, default=5.0,
                      help="Clip shaped rewards to [-clip, clip]")
    args.add_argument("--adv_clip", type=float, default=2.0,
                      help="Clip per-group advantages to [-clip, clip]")
    ######################### Diversity Controls #########################
    args.add_argument("--reward_temp", type=float, default=0.5,
                      help="Reward power transform: sign(r)*|r|^α. Lower=flatter=more diverse (0.3-0.7)")
    args.add_argument("--temp_max", type=float, default=1.3,
                      help="Initial attacker sampling temperature (higher=more exploration)")
    args.add_argument("--temp_min", type=float, default=0.8,
                      help="Final attacker sampling temperature after annealing")
    args.add_argument("--min_diversity", type=float, default=0.0,
                      help="Early stop if batch unique_ratio falls below this (0=disabled)")
    ##################################################################
    args.add_argument("--es_patience", type=int, default=0,
                      help="Early stop after N steps without improvement (0=disabled)")
    args.add_argument("--es_min_delta", type=float, default=0.01,
                      help="Min success-rate improvement to reset patience")
    ##################################################################

    args = args.parse_args()

    seed(args.seed)

    if args.mode == "sft":
        logger.info("Running optional SFT warm-up for attacker")
        trainer = SFTTrainer(args)
        trainer.train()

    elif args.mode == "redteam":
        if not args.sft_ckpt:
            logger.info("No --sft_ckpt provided; starting RL from base model")

        if args.active_attacks:
            num_rounds = args.train_steps // args.interval
            victim_ckpt = args.victim_ckpt
            original_sft_ckpt = args.sft_ckpt
            all_collected_attacks = []

            for round_idx in range(num_rounds):
                args.sft_ckpt = original_sft_ckpt  # reinitialize attacker each round
                logger.info(f"=== Active Attacks Round {round_idx + 1}/{num_rounds} ===")

                # Attack phase: train attacker against current victim
                args.victim_ckpt = victim_ckpt
                rl_trainer = RLTrainer(args)
                attacker_ckpt = rl_trainer.train()
                all_collected_attacks.extend(rl_trainer.collected_attacks)
                logger.info(f"Round {round_idx + 1}: {len(rl_trainer.collected_attacks)} new attacks, "
                            f"{len(all_collected_attacks)} total accumulated")

                # Safety phase: harden victim on collected attacks
                if round_idx < num_rounds - 1 and all_collected_attacks:
                    # Replace harmful responses with safe refusals
                    import json, os
                    safety_data = [
                        {"instruction": atk["instruction"],
                         "response": generate_safe_refusal(atk["instruction"])}
                        for atk in all_collected_attacks
                    ]
                    safety_file = os.path.join(
                        args.log_dir,
                        f"{args.exp_name}_safety_round{round_idx + 1}.json"
                    )
                    with open(safety_file, "w") as f:
                        json.dump(safety_data, f, indent=2)
                    args.safety_data_file = safety_file

                    safety_args = argparse.Namespace(**vars(args))
                    safety_args.train_steps = args.safety_steps
                    safety_trainer = SafetyTrainer(safety_args)
                    victim_ckpt = safety_trainer.train()

                    logger.info(f"Round {round_idx + 1}: victim hardened, ckpt={victim_ckpt}")

            # Save final global attack dataset
            if all_collected_attacks:
                import json, os
                final_attacks_file = os.path.join(args.log_dir, f"{args.exp_name}_all_attacks.json")
                with open(final_attacks_file, "w") as f:
                    json.dump(all_collected_attacks, f, indent=2)
                logger.info(f"Saved {len(all_collected_attacks)} total attacks to {final_attacks_file}")
                logger.info("Run MLE mode on this dataset to produce the final smoothed attacker")
        else:
            trainer = RLTrainer(args)
            trainer.train()

    elif args.mode == "mle":
        trainer = MLETrainer(args)
        trainer.train()

    elif args.mode == "safety":
        trainer = SafetyTrainer(args)
        trainer.train()

    else:
        raise ValueError(f"Mode {args.mode} not supported")
