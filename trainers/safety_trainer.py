import logging
import os

import tinker
from tinker import types as tinker_types
from tinker.types.tensor_data import TensorData
import torch
from tqdm import tqdm

from dataset import SafetyDataset
from utils import apply_chat_template

logger = logging.getLogger(__name__)


class SafetyTrainer:
    """Safety fine-tuning trainer using Tinker.

    Fine-tunes the victim LLM on harmful prompts paired with safe responses,
    using cross-entropy loss. This hardens the victim against attacks.
    """

    def __init__(self, args) -> None:
        self.args = args

        self.service_client = tinker.ServiceClient()

        if hasattr(args, 'victim_ckpt') and args.victim_ckpt:
            self.training_client = (
                self.service_client.create_training_client_from_state_with_optimizer(
                    path=args.victim_ckpt
                )
            )
        else:
            self.training_client = self.service_client.create_lora_training_client(
                base_model=args.victim_model,
                rank=args.lora_r,
            )

        self.tokenizer = self.training_client.get_tokenizer()

        self.adam_params = tinker_types.AdamParams(
            learning_rate=args.lr,
            beta1=0.9, beta2=0.95, eps=1e-8,
            weight_decay=args.weight_decay,
        )

        self.dataset = SafetyDataset(args.safety_data_file)

    def _build_datum(self, item: dict) -> tinker_types.Datum:
        """Convert a safety item (instruction + safe response) into a Tinker Datum."""
        prompt_tokens = apply_chat_template(
            self.tokenizer,
            [{"role": "user", "content": item["instruction"]}],
            add_generation_prompt=True,
        )
        full_tokens = apply_chat_template(
            self.tokenizer,
            [
                {"role": "user", "content": item["instruction"]},
                {"role": "assistant", "content": item["response"]},
            ],
        )
        prompt_len = len(prompt_tokens)
        weights = [0.0] * prompt_len + [1.0] * (len(full_tokens) - prompt_len)

        model_input = tinker_types.ModelInput(
            chunks=[tinker_types.EncodedTextChunk(tokens=full_tokens)]
        )
        return tinker_types.Datum(
            model_input=model_input,
            loss_fn_inputs={
                "target_tokens": TensorData.from_torch(torch.tensor(full_tokens)),
                "weights": TensorData.from_torch(torch.tensor(weights)),
            },
        )

    def save(self, name: str) -> str:
        result = self.training_client.save_state(name=name).result()
        logger.info(f"Saved victim checkpoint: {result.path}")
        return result.path

    def train(self):
        logger.info("=" * 60)
        logger.info("Safety Fine-Tuning started")
        logger.info("  Victim model: %s (ckpt=%s)",
                     self.args.victim_model,
                     getattr(self.args, 'victim_ckpt', '') or "base")
        logger.info("  LoRA rank:    %d", self.args.lora_r)
        logger.info("  LR:           %s", self.args.lr)
        logger.info("  Batch size:   %d", self.args.batch_size)
        logger.info("  Train steps:  %d", self.args.train_steps)
        logger.info("  Dataset:      %d examples", len(self.dataset))
        logger.info("=" * 60)

        t = tqdm(range(1, self.args.train_steps + 1),
                 desc="Safety training", dynamic_ncols=True)
        for step in t:
            batch = self.dataset.get_batch(self.args.batch_size)
            datums = [self._build_datum(item) for item in batch]

            fwd_bwd_future = self.training_client.forward_backward(
                datums, loss_fn="cross_entropy"
            )
            optim_future = self.training_client.optim_step(self.adam_params)

            fwd_bwd_result = fwd_bwd_future.result()
            optim_future.result()

            loss_val = 0.0
            if hasattr(fwd_bwd_result, 'loss_fn_outputs'):
                for output in fwd_bwd_result.loss_fn_outputs:
                    if 'loss:sum' in output:
                        loss_val += output['loss:sum']

            t.set_description(f"Step {step}: loss={loss_val:.4f}")

        victim_ckpt_path = self.save(name="final")
        logger.info("Safety training complete")
        return victim_ckpt_path