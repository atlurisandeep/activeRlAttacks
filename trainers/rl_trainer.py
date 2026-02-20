import json
import logging
import math
import os
import random
import time
from typing import List

import numpy as np
import tinker
from tinker import types as tinker_types
from tinker.types.tensor_data import TensorData
import torch
from tqdm import tqdm

from dataset import RedTeamDataset
from utils import (LLMJudgeClassifier, compute_ngram_repetition_rate,
                   compute_batch_diversity_penalties, compute_batch_diversity_metrics,
                   apply_chat_template, is_coherent)

logger = logging.getLogger(__name__)


class RLTrainer:
    """GRPO-style RL trainer for red-teaming using Tinker.

    Trains the attacker LLM to generate adversarial prompts that elicit
    harmful responses from a victim model. Uses group-based rollouts with
    advantage centering and importance-sampling policy gradient.

    Flow per step:
    1. Sample attacker prompt templates from dataset
    2. Generate attack completions from attacker (via Tinker SamplingClient)
    3. Feed attacks to victim model (via Tinker SamplingClient)
    4. Score victim responses with LLM-as-judge (via Tinker SamplingClient)
    5. Compute advantages (GRPO: per-group reward centering)
    6. Train attacker with importance_sampling loss (via Tinker forward_backward)
    """

    def __init__(self, args) -> None:
        self.args = args
        self.service_client = tinker.ServiceClient()

        # Attacker model — the model we're training
        if hasattr(args, 'sft_ckpt') and args.sft_ckpt:
            self.training_client = (
                self.service_client.create_training_client_from_state(
                    path=args.sft_ckpt
                )
            )
        else:
            self.training_client = self.service_client.create_lora_training_client(
                base_model=args.model_name,
                rank=args.lora_r,
            )

        # Frozen reference for KL penalty
        self.reference_client = self.training_client.save_weights_and_get_sampling_client()

        self.attacker_tokenizer = self.training_client.get_tokenizer()

        self.adam_params = tinker_types.AdamParams(
            learning_rate=args.lr,
            beta1=0.9, beta2=0.95, eps=1e-8,
        )

        # Victim model — we sample from it but don't train it
        if hasattr(args, 'victim_ckpt') and args.victim_ckpt:
            self.victim_client = self.service_client.create_sampling_client(
                model_path=args.victim_ckpt
            )
        else:
            self.victim_client = self.service_client.create_sampling_client(
                base_model=args.victim_model
            )

        self.victim_tokenizer = self.victim_client.get_tokenizer()

        self.victim_sampling_params = tinker_types.SamplingParams(
            max_tokens=args.victim_max_len,
            temperature=args.victim_temp,
            top_p=args.victim_top_p,
        )

        # Attacker sampling params
        self.attacker_sampling_params = tinker_types.SamplingParams(
            max_tokens=args.max_len,
            temperature=1.0,
        )

        # Toxicity judge
        self.toxicity_fn = LLMJudgeClassifier(
            self.service_client, judge_model=args.judge_model
        )

        # Dataset
        self.dataset = RedTeamDataset(args.prompt_file)

        # Logging
        self.log_file = os.path.join(
            args.log_dir, f"{args.exp_name}.jsonl"
        )
        os.makedirs(args.log_dir, exist_ok=True)

        # Collected attacks for downstream MLE/safety training
        self.collected_attacks = []

        # Early stopping state
        self.best_success_rate = -1.0
        self.steps_without_improvement = 0

        # Diversity tracking for collapse detection
        self.diversity_history = []

    def _generate_attacks(self, sampling_client, prompt_templates: List[str]
                          ) -> List[List[dict]]:
        """Generate attack completions from the attacker model.

        Returns a list of lists (one inner list per prompt template), where
        each inner list contains dicts with keys:
            prompt_template, model_input, sampled_tokens, logprobs, attack_text
        Incoherent outputs are filtered, so inner lists may be shorter than
        group_size or empty.
        """
        futures = []
        model_inputs = []
        for tmpl in prompt_templates:
            convo = [{"role": "user", "content": tmpl}]
            tokens = apply_chat_template(
                self.attacker_tokenizer, convo, add_generation_prompt=True
            )
            model_input = tinker_types.ModelInput(
                chunks=[tinker_types.EncodedTextChunk(tokens=tokens)]
            )
            future = sampling_client.sample(
                prompt=model_input,
                num_samples=self.args.group_size,
                sampling_params=self.attacker_sampling_params,
            )
            futures.append(future)
            model_inputs.append(model_input)

        grouped_results = []
        n_filtered = 0
        for future, model_input, tmpl in zip(futures, model_inputs, prompt_templates):
            group = []
            sample_result = future.result()
            for seq in sample_result.sequences:
                attack_text = self.attacker_tokenizer.decode(
                    seq.tokens, skip_special_tokens=True
                ).strip()
                # Filter incoherent outputs before wasting victim/judge calls
                if not is_coherent(attack_text):
                    n_filtered += 1
                    continue
                group.append({
                    "prompt_template": tmpl,
                    "model_input": model_input,
                    "sampled_tokens": seq.tokens,
                    "logprobs": seq.logprobs,
                    "attack_text": attack_text,
                })
            grouped_results.append(group)
        if n_filtered:
            logger.debug(f"Filtered {n_filtered} incoherent attacks")
        return grouped_results

    def _get_victim_responses(self, attack_texts: List[str]) -> List[str]:
        """Get victim model responses to attack prompts."""
        futures = []
        for attack in attack_texts:
            convo = [{"role": "user", "content": attack}]
            tokens = apply_chat_template(
                self.victim_tokenizer, convo, add_generation_prompt=True
            )
            model_input = tinker_types.ModelInput(
                chunks=[tinker_types.EncodedTextChunk(tokens=tokens)]
            )
            future = self.victim_client.sample(
                prompt=model_input,
                num_samples=1,
                sampling_params=self.victim_sampling_params,
            )
            futures.append(future)

        responses = []
        for future in futures:
            result = future.result()
            text = self.victim_tokenizer.decode(
                result.sequences[0].tokens, skip_special_tokens=True
            ).strip()
            responses.append(text)
        return responses

    def _score_responses(self, attacks: List[str], responses: List[str]
                         ) -> List[float]:
        """Score victim responses using the LLM judge. Returns linear rewards."""
        return self.toxicity_fn.compute(attacks, responses)

    def _shape_rewards(self, attacks: List[dict], raw_log_rewards: List[float]
                       ) -> List[float]:
        """Apply reward shaping to prevent reward hacking.

        Applies:
        1. Length penalty — too-short or max-length attacks get penalised
        2. Repetition penalty — high n-gram repetition rate is penalised
        3. Entropy bonus — encourage diverse token distributions
        4. Duplicate penalty — near-identical attacks in the batch are penalised
        5. Reward clipping — cap extreme values to stabilise gradients
        """
        attack_texts = [a["attack_text"] for a in attacks]

        # Reward temperature: compress reward range to reduce winner-take-all
        # sign(r) * |r|^α — flattens the landscape so GRPO explores more modes
        # (Haarnoja et al. 2018 SAC; Eysenbach & Levine 2022 MaxEnt RL)
        reward_temp = getattr(self.args, 'reward_temp', 1.0)
        if reward_temp != 1.0:
            raw_log_rewards = [
                np.sign(r) * (np.abs(r) ** reward_temp) for r in raw_log_rewards
            ]

        # Batch-level: duplicate penalty
        dup_penalties = compute_batch_diversity_penalties(
            attack_texts, threshold=self.args.diversity_threshold
        )

        shaped = []
        for i, (attack, reward) in enumerate(zip(attacks, raw_log_rewards)):
            penalty = 0.0
            text = attack["attack_text"]
            tokens = attack["sampled_tokens"]
            logprobs = attack.get("logprobs", [])

            # --- Length penalty ---
            if len(tokens) < self.args.min_attack_tokens:
                penalty -= self.args.length_penalty
            if len(tokens) >= self.args.max_len:
                # Hit max-length cap → likely degenerate / run-on
                penalty -= self.args.length_penalty

            # --- Repetition penalty ---
            rep_rate = compute_ngram_repetition_rate(text, n=3)
            if rep_rate > self.args.max_repetition_rate:
                penalty -= self.args.repetition_penalty * rep_rate

            # --- KL/entropy regularization: bonus for high-entropy (diverse) outputs ---
            if logprobs and self.args.kl_coeff > 0:
                mean_logprob = sum(logprobs) / len(logprobs)
                # mean_logprob is negative; more negative = higher entropy
                penalty += self.args.kl_coeff * (-mean_logprob)

            # --- Duplicate penalty ---
            penalty += dup_penalties[i] * self.args.diversity_penalty

            # --- Combine & clip ---
            shaped_reward = reward + penalty
            shaped_reward = max(-self.args.reward_clip,
                                min(self.args.reward_clip, shaped_reward))
            shaped.append(shaped_reward)

        return shaped

    def _build_training_datums(self, attacks: List[dict],
                               rewards_per_group: List[List[float]]
                               ) -> List[tinker_types.Datum]:
        """Build Tinker Datum objects with GRPO-style advantages.

        Groups attacks by prompt template and centers rewards within each group.
        """
        datums = []
        idx = 0
        for group_rewards in rewards_per_group:
            mean_reward = sum(group_rewards) / len(group_rewards)
            std_reward = (
                sum((r - mean_reward) ** 2 for r in group_rewards)
                / len(group_rewards)
            ) ** 0.5

            # Skip groups with zero variance (all same reward → no signal)
            if std_reward < 1e-8:
                idx += len(group_rewards)
                continue

            # Normalize advantages (zero mean, unit variance)
            advantages = [(r - mean_reward) / std_reward for r in group_rewards]
            # Clip to prevent extreme updates
            adv_clip = self.args.adv_clip
            advantages = [max(-adv_clip, min(adv_clip, a)) for a in advantages]

            for advantage in advantages:
                attack = attacks[idx]
                prompt_input = attack["model_input"]
                sampled_tokens = attack["sampled_tokens"]
                logprobs = attack["logprobs"]

                ob_len = prompt_input.length - 1
                model_input = prompt_input.append(
                    tinker_types.EncodedTextChunk(tokens=sampled_tokens[:-1])
                )
                target_tokens = [0] * ob_len + sampled_tokens
                padded_logprobs = [0.0] * ob_len + logprobs
                padded_advantages = (
                    [0.0] * ob_len
                    + [advantage] * (model_input.length - ob_len)
                )

                datum = tinker_types.Datum(
                    model_input=model_input,
                    loss_fn_inputs={
                        "target_tokens": TensorData.from_torch(
                            torch.tensor(target_tokens)
                        ),
                        "logprobs": TensorData.from_torch(
                            torch.tensor(padded_logprobs)
                        ),
                        "advantages": TensorData.from_torch(
                            torch.tensor(padded_advantages)
                        ),
                    },
                )
                datums.append(datum)
                idx += 1

        return datums

    def _get_lr(self, step: int, total_steps: int) -> float:
        """Compute LR with linear warmup and cosine decay."""
        warmup_steps = max(1, int(total_steps * self.args.warmup_ratio))
        if step <= warmup_steps:
            return self.args.lr * (step / warmup_steps)
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return self.args.lr_min + 0.5 * (self.args.lr - self.args.lr_min) * (1 + math.cos(math.pi * progress))

    def _check_early_stop(self, step: int) -> bool:
        """Check if training should stop early.

        Returns True if success rate has not improved for `es_patience` steps.
        """
        if self.args.es_patience <= 0:
            return False
        if self.steps_without_improvement >= self.args.es_patience:
            logger.info(
                f"Early stopping at step {step}: no improvement for "
                f"{self.args.es_patience} steps (best success={self.best_success_rate:.2%})"
            )
            return True
        return False

    def _log_attacks(self, step: int, attacks: List[dict],
                     responses: List[str], rewards: List[float]):
        """Log attacks to JSONL file."""
        with open(self.log_file, "a") as f:
            for attack, response, reward in zip(attacks, responses, rewards):
                entry = {
                    "step": step,
                    "attack": attack["attack_text"],
                    "response": response,
                    "reward": reward,
                }
                f.write(json.dumps(entry) + "\n")

    def train(self):
        train_steps = self.args.train_steps
        if self.args.active_attacks:
            train_steps = self.args.interval

        logger.info("=" * 60)
        logger.info("RL Red-Team Training started")
        logger.info("  Attacker:         %s (ckpt=%s)",
                     self.args.model_name, self.args.sft_ckpt or "base")
        logger.info("  Victim:           %s (ckpt=%s)",
                     self.args.victim_model,
                     getattr(self.args, 'victim_ckpt', '') or "base")
        logger.info("  Judge:            %s", self.args.judge_model)
        logger.info("  LoRA rank:        %d", self.args.lora_r)
        logger.info("  LR:               %s", self.args.lr)
        logger.info("  LR schedule:      warmup_ratio=%.2f, lr_min=%s",
                     self.args.warmup_ratio, self.args.lr_min)
        logger.info("  Batch size:       %d", self.args.batch_size)
        logger.info("  Group size:       %d", self.args.group_size)
        logger.info("  Train steps:      %d", train_steps)
        logger.info("  Active attacks:   %s", self.args.active_attacks)
        logger.info("  --- Reward Shaping ---")
        logger.info("  Length penalty:    %.1f (min_tokens=%d)",
                     self.args.length_penalty, self.args.min_attack_tokens)
        logger.info("  Repetition pen:   %.1f (max_rate=%.2f)",
                     self.args.repetition_penalty, self.args.max_repetition_rate)
        logger.info("  KL coeff:         %.4f", self.args.kl_coeff)
        logger.info("  Diversity pen:    %.1f (threshold=%.2f)",
                     self.args.diversity_penalty, self.args.diversity_threshold)
        logger.info("  Reward clip:      %.1f", self.args.reward_clip)
        logger.info("  Advantage clip:   %.1f", self.args.adv_clip)
        logger.info("  --- Diversity Controls ---")
        logger.info("  Reward temp (α):  %.2f", getattr(self.args, 'reward_temp', 1.0))
        logger.info("  Sampling T:       %.2f → %.2f (anneal)",
                     getattr(self.args, 'temp_max', 1.0),
                     getattr(self.args, 'temp_min', 1.0))
        logger.info("  Min diversity:    %.2f", getattr(self.args, 'min_diversity', 0.0))
        logger.info("  Early stopping:   patience=%d, min_delta=%.3f",
                     self.args.es_patience, self.args.es_min_delta)
        logger.info("=" * 60)

        t = tqdm(range(1, train_steps + 1),
                 desc="RL training", dynamic_ncols=True)

        for step in t:
            t_start = time.time()

            # 1. Get attacker prompt templates
            prompt_templates = self.dataset.get_batch(self.args.batch_size)

            # 2. Get current attacker weights for sampling
            attacker_sampler = (
                self.training_client.save_weights_and_get_sampling_client()
            )

            # 3. Generate attacks with temperature annealing for diversity
            temp_max = getattr(self.args, 'temp_max', 1.0)
            temp_min = getattr(self.args, 'temp_min', 1.0)
            if temp_max != temp_min:
                # Exponential decay: T(t) = T_max * (T_min / T_max)^(t / T_total)
                current_temp = temp_max * (temp_min / temp_max) ** (step / train_steps)
            else:
                current_temp = temp_max
            self.attacker_sampling_params = tinker_types.SamplingParams(
                max_tokens=self.args.max_len,
                temperature=current_temp,
            )

            attack_groups = self._generate_attacks(attacker_sampler, prompt_templates)
            # Drop empty groups (all samples filtered for that prompt)
            attack_groups = [g for g in attack_groups if g]
            # Flatten for victim/judge calls
            attacks = [a for group in attack_groups for a in group]
            if not attacks:
                logger.warning(f"Step {step}: all attacks filtered as incoherent, skipping")
                self.steps_without_improvement += 1
                if self._check_early_stop(step):
                    break
                continue
            attack_texts = [a["attack_text"] for a in attacks]

            # 4. Get victim responses
            victim_responses = self._get_victim_responses(attack_texts)

            # 5. Score with LLM judge
            raw_log_rewards = self._score_responses(attack_texts, victim_responses)

            # 5b. Shape rewards (length, repetition, entropy, diversity penalties)
            log_rewards = self._shape_rewards(attacks, raw_log_rewards)

            # 6. Group rewards by prompt template using actual group structure
            rewards_per_group = []
            offset = 0
            for group in attack_groups:
                rewards_per_group.append(log_rewards[offset:offset + len(group)])
                offset += len(group)

            # 7. Build training datums with GRPO advantages
            datums = self._build_training_datums(attacks, rewards_per_group)

            if not datums:
                logger.warning(f"Step {step}: no valid datums, skipping")
                continue

            # 8. Train
            fwd_bwd_future = self.training_client.forward_backward(
                datums, loss_fn="importance_sampling"
            )
            current_lr = self._get_lr(step, train_steps)
            adam_params = tinker_types.AdamParams(
                learning_rate=current_lr,
                beta1=0.9, beta2=0.95, eps=1e-8,
            )
            optim_future = self.training_client.optim_step(adam_params)
            fwd_bwd_future.result()
            optim_future.result()

            # Metrics (report both raw and shaped for diagnostics)
            mean_reward = np.mean(log_rewards)
            mean_raw = np.mean(raw_log_rewards)
            success_rate = np.mean(np.array(raw_log_rewards) > 0.0)

            # Diversity metrics (Technique 5)
            div_metrics = compute_batch_diversity_metrics(attack_texts)
            self.diversity_history.append(div_metrics["unique_ratio"])

            t.set_description(
                f"Step {step}: raw={mean_raw:.3f} shaped={mean_reward:.3f} "
                f"success={success_rate:.2%} uniq={div_metrics['unique_ratio']:.0%} "
                f"jacc={div_metrics['avg_jaccard']:.2f} T={current_temp:.2f} "
                f"lr={current_lr:.2e}"
            )

            # Early stopping check
            if success_rate > self.best_success_rate + self.args.es_min_delta:
                self.best_success_rate = success_rate
                self.steps_without_improvement = 0
            else:
                self.steps_without_improvement += 1

            if self._check_early_stop(step):
                break

            # Diversity collapse detection
            min_diversity = getattr(self.args, 'min_diversity', 0.0)
            if min_diversity > 0 and len(self.diversity_history) >= 10:
                recent_div = np.mean(self.diversity_history[-10:])
                if recent_div < min_diversity:
                    logger.warning(
                        f"Diversity collapsed to {recent_div:.1%} "
                        f"(threshold {min_diversity:.1%}). Early stopping."
                    )
                    break

            # Log attacks (include both raw and shaped rewards)
            self._log_attacks(step, attacks, victim_responses, log_rewards)

            # Collect successful attacks using RAW rewards (not shaped)
            # so that quality penalties don't block legitimate successes
            for attack, response, reward in zip(
                attacks, victim_responses, raw_log_rewards
            ):
                if reward > 0.0:
                    self.collected_attacks.append({
                        "instruction": attack["attack_text"],
                        "response": response,
                    })

        # Save checkpoint
        save_result = self.training_client.save_state(name="final").result()
        logger.info(f"RL training complete. Checkpoint: {save_result.path}")

        # Save collected attacks
        if self.collected_attacks:
            attacks_file = os.path.join(
                self.args.log_dir, f"{self.args.exp_name}_attacks.json"
            )
            with open(attacks_file, "w") as f:
                json.dump(self.collected_attacks, f, indent=2)
            logger.info(
                f"Saved {len(self.collected_attacks)} collected attacks to {attacks_file}"
            )

        return save_result.path
