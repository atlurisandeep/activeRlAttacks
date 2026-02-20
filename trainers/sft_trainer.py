import logging
import os

import tinker
from tinker import types as tinker_types
from tinker.types.tensor_data import TensorData
import torch
from tqdm import tqdm

from dataset import SFTDataset
from utils import apply_chat_template

logger = logging.getLogger(__name__)


class SFTTrainer:
    """Supervised fine-tuning trainer using Tinker.

    Trains the attacker LLM to generate attack prompts by fine-tuning on
    pre-collected attack examples using cross-entropy loss.
    """

    def __init__(self, args) -> None:
        self.args = args

        self.service_client = tinker.ServiceClient()
        self.training_client = self.service_client.create_lora_training_client(
            base_model=args.model_name,
            rank=args.lora_r,
        )
        self.tokenizer = self.training_client.get_tokenizer()

        self.adam_params = tinker_types.AdamParams(
            learning_rate=args.lr,
            beta1=0.9, beta2=0.95, eps=1e-8,
            weight_decay=args.weight_decay,
        )

        self.tr_dataset = SFTDataset(
            args.prompt_file, args.few_shot_file, split="train"
        )
        self.val_dataset = SFTDataset(
            args.prompt_file, args.few_shot_file, split="val"
        )

    def _build_datum(self, item: dict) -> tinker_types.Datum:
        """Convert an SFT item (prompt + completion) into a Tinker Datum.

        Uses the tokenizer's chat template to build prompt tokens, then
        appends the completion tokens. Loss weights are 0 for the prompt
        and 1 for the completion.
        """
        # Tokenize the prompt portion (with generation prompt suffix)
        prompt_tokens = apply_chat_template(
            self.tokenizer,
            [{"role": "user", "content": item["prompt"]}],
            add_generation_prompt=True,
        )
        # Tokenize the full conversation (prompt + completion)
        full_tokens = apply_chat_template(
            self.tokenizer,
            [
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": item["completion"]},
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

    def validation(self) -> float:
        """Compute validation loss using forward pass."""
        datums = [self._build_datum(self.val_dataset[i])
                  for i in range(len(self.val_dataset))]
        # Process in chunks
        total_loss = 0.0
        chunk_size = self.args.batch_size
        n_chunks = 0
        for i in range(0, len(datums), chunk_size):
            chunk = datums[i:i + chunk_size]
            fwd_result = self.training_client.forward(
                chunk, loss_fn="cross_entropy"
            ).result()
            if hasattr(fwd_result, 'loss_fn_outputs'):
                for output in fwd_result.loss_fn_outputs:
                    if 'loss:sum' in output:
                        total_loss += output['loss:sum']
                        n_chunks += 1
        return total_loss / max(n_chunks, 1)

    def save(self, name: str):
        """Save model weights for sampling and full state for resuming."""
        self.training_client.save_state(name=name).result()
        logger.info(f"Saved checkpoint: {name}")

    def train(self):
        logger.info("=" * 60)
        logger.info("SFT Training started")
        logger.info("  Model:       %s", self.args.model_name)
        logger.info("  LoRA rank:   %d", self.args.lora_r)
        logger.info("  LR:          %s", self.args.lr)
        logger.info("  Batch size:  %d", self.args.batch_size)
        logger.info("  Train steps: %d", self.args.train_steps)
        logger.info("  Train data:  %d examples", len(self.tr_dataset))
        logger.info("  Val data:    %d examples", len(self.val_dataset))
        logger.info("=" * 60)

        t = tqdm(range(1, self.args.train_steps + 1),
                 desc="SFT training", dynamic_ncols=True)
        for step in t:
            batch = self.tr_dataset.get_batch(self.args.batch_size)
            datums = [self._build_datum(item) for item in batch]

            # forward_backward + optim_step pipelined
            fwd_bwd_future = self.training_client.forward_backward(
                datums, loss_fn="cross_entropy"
            )
            optim_future = self.training_client.optim_step(self.adam_params)

            fwd_bwd_result = fwd_bwd_future.result()
            optim_result = optim_future.result()

            # Extract loss for logging
            loss_val = 0.0
            if hasattr(fwd_bwd_result, 'loss_fn_outputs'):
                for output in fwd_bwd_result.loss_fn_outputs:
                    if 'loss:sum' in output:
                        loss_val += output['loss:sum']

            t.set_description(f"Step {step}: loss={loss_val:.4f}")

            if step % 10 == 0:
                val_loss = self.validation()
                logger.info(f"Step {step} | val_loss={val_loss:.4f}")

        self.save(name="final")
        logger.info("SFT training complete")